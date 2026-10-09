#!/usr/bin/env python3
"""Download and locally validate v3 CDSE candidate queues.

Credentials are configured in 00_config.py as CDSE_ACCOUNTS. The downloader is
resumable, writes state after each MGRS record, and can process several records
in parallel while protecting individual .part files with per-path locks.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from itertools import cycle
from pathlib import Path

import requests

import s23_common as C

cfg = C.cfg

_token_lock = threading.Lock()
_token_cache = {}
_path_locks_lock = threading.Lock()
_path_locks = {}
_state_lock = threading.Lock()
_progress_lock = threading.Lock()
_progress_bytes = {}
_progress_done = set()


def geojson_bounds(geom):
    return C.bounds_from_geometry_like(geom)


def configured_accounts():
    paths = []
    import os
    if os.environ.get("S23_CDSE_ACCOUNTS_JSON"):
        paths.append(Path(os.environ["S23_CDSE_ACCOUNTS_JSON"]))
    paths.append(Path(getattr(cfg, "ROOT", Path.cwd())).parent / "accounts.local.json")
    paths.append(Path(getattr(cfg, "ROOT", Path.cwd())) / "accounts.local.json")

    accounts = []
    for path in paths:
        if not path.exists():
            continue
        data = C.read_json(path, [])
        if isinstance(data, dict):
            data = data.get("accounts", [])
        accounts.extend(item for item in data if isinstance(item, dict))
        if accounts:
            break

    if not accounts:
        accounts = list(getattr(cfg, "CDSE_ACCOUNTS", []) or [])
    accounts = [item for item in accounts if item.get("username") and item.get("password")]
    if not accounts:
        checked = [str(path) for path in paths]
        raise SystemExit("Missing CDSE accounts. Create s23_pipeline_v4/accounts.local.json or set S23_CDSE_ACCOUNTS_JSON. Checked: " + json.dumps(checked, indent=2))
    return accounts
def token(username, password, force_refresh=False):
    with _token_lock:
        if not force_refresh and username in _token_cache:
            return _token_cache[username]
    response = requests.post(
        cfg.CDSE_TOKEN_URL,
        data={"client_id": "cdse-public", "username": username, "password": password, "grant_type": "password"},
        timeout=60,
    )
    response.raise_for_status()
    access_token = response.json()["access_token"]
    with _token_lock:
        _token_cache[username] = access_token
    return access_token


def product_filename(product):
    name = product.get("name") or product.get("Name") or product.get("id") or product.get("Id")
    name = str(name)
    return name if name.lower().endswith(".zip") else name + ".zip"


def local_path(product):
    sensor = C.sensor_from_name(product.get("name", ""))
    sub = "s2" if sensor == "s2" else "s3" if sensor == "s3" else "other"
    return cfg.DOWNLOAD_ROOT / sub / product_filename(product)


def path_lock(path):
    key = str(path.resolve())
    with _path_locks_lock:
        if key not in _path_locks:
            _path_locks[key] = threading.Lock()
        return _path_locks[key]


def product_sensor(product):
    return C.sensor_from_name(product.get("name", ""))


def validate_product_zip(path, product):
    sensor = product_sensor(product)
    ok, reason = C.validate_zip(path, sensor)
    if not ok:
        raise RuntimeError(f"invalid_{sensor}_zip:{reason}:{path}")
    return reason


def download_product(product, account):
    path = local_path(product)
    lock = path_lock(path)
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > 0:
            validate_product_zip(path, product)
            with _progress_lock:
                _progress_done.add(str(path))
            return path

        tmp = path.with_suffix(path.suffix + ".part")
        url = f"{cfg.CDSE_DOWNLOAD}/Products({product['id']})/$value"
        last_error = None
        for attempt in range(1, int(cfg.DOWNLOAD_MAX_RETRIES) + 1):
            try:
                access_token = token(account["username"], account["password"], force_refresh=attempt > 1)
                resume_at = tmp.stat().st_size if tmp.exists() else 0
                headers = {"Authorization": f"Bearer {access_token}"}
                if resume_at:
                    headers["Range"] = f"bytes={resume_at}-"
                mode = "ab" if resume_at else "wb"
                C.progress(
                    f"02 download {product_sensor(product)} {product.get('name')} "
                    f"attempt={attempt}/{cfg.DOWNLOAD_MAX_RETRIES} resume={resume_at / 1024**2:.1f} MiB "
                    f"account={account['username']}"
                )
                with requests.get(
                    url,
                    headers=headers,
                    stream=True,
                    timeout=(30, cfg.DOWNLOAD_TIMEOUT_SECONDS),
                ) as response:
                    if resume_at and response.status_code == 200:
                        C.progress(f"02 server ignored Range; restart {product.get('name')}")
                        tmp.unlink(missing_ok=True)
                        resume_at = 0
                        mode = "wb"
                    elif resume_at and response.status_code == 416:
                        tmp.replace(path)
                        validate_product_zip(path, product)
                        with _progress_lock:
                            _progress_done.add(str(path))
                        return path
                    response.raise_for_status()
                    with tmp.open(mode) as fd:
                        for chunk in response.iter_content(chunk_size=cfg.DOWNLOAD_CHUNK_SIZE):
                            if not chunk:
                                continue
                            fd.write(chunk)
                            with _progress_lock:
                                _progress_bytes[str(path)] = _progress_bytes.get(str(path), 0) + len(chunk)
                tmp.replace(path)
                validate_product_zip(path, product)
                with _progress_lock:
                    _progress_done.add(str(path))
                return path
            except Exception as exc:
                last_error = exc
                C.progress(f"02 retryable download error {product.get('name')}: {exc}")
                if attempt >= int(cfg.DOWNLOAD_MAX_RETRIES):
                    raise
                time.sleep(min(120, 5 * attempt))
        raise RuntimeError(last_error)


def local_s3_covers_s2_geofootprint(s3_product, s2_product):
    s2_bounds = geojson_bounds(s2_product.get("geofootprint"))
    if not s2_bounds:
        return False, "missing_s2_geofootprint_bounds"
    s3_path = local_path(s3_product)
    try:
        s3_bounds = C.s3_bounds_wgs84(s3_path)
    except Exception as exc:
        return False, f"cannot_read_s3_bounds:{exc}"
    if not s3_bounds:
        return False, "missing_s3_bounds"
    if not C.bbox_contains(s3_bounds, s2_bounds):
        ratio = C.bbox_overlap_ratio(s2_bounds, s3_bounds)
        return False, f"s3_local_bounds_do_not_cover_s2_geofootprint:overlap={ratio:.4f}"
    return True, "ok"


def load_state():
    return C.read_json(cfg.CDSE_ACCEPTED_DOWNLOADS_JSON, {"accepted": {}, "rejected": {}})


def save_state(state):
    with _state_lock:
        C.write_json(cfg.CDSE_ACCEPTED_DOWNLOADS_JSON, state)


def candidate_products(pair):
    t1_s2 = pair["t1"]["s2"]
    t2_s2 = pair["t2"]["s2"]
    t1_s3 = pair["t1"]["candidate_s3"][0]
    t2_s3 = pair["t2"]["candidate_s3"][0]
    return t1_s2, t2_s2, t1_s3, t2_s3


def try_candidate(pair, account):
    t1_s2, t2_s2, t1_s3, t2_s3 = candidate_products(pair)
    download_product(t1_s3, account)
    ok, reason = local_s3_covers_s2_geofootprint(t1_s3, t1_s2)
    if not ok:
        return None, "t1_" + reason
    download_product(t2_s3, account)
    ok, reason = local_s3_covers_s2_geofootprint(t2_s3, t2_s2)
    if not ok:
        return None, "t2_" + reason
    t1_s2_path = download_product(t1_s2, account)
    t2_s2_path = download_product(t2_s2, account)
    p1 = C.read_s2_data_percentage(t1_s2_path)
    p2 = C.read_s2_data_percentage(t2_s2_path)
    try:
        if p1 != "" and float(p1) < cfg.MIN_S2_DATA_PERCENTAGE:
            return None, f"t1_s2_data_percentage_lt_{cfg.MIN_S2_DATA_PERCENTAGE}:{p1}"
        if p2 != "" and float(p2) < cfg.MIN_S2_DATA_PERCENTAGE:
            return None, f"t2_s2_data_percentage_lt_{cfg.MIN_S2_DATA_PERCENTAGE}:{p2}"
    except Exception:
        pass
    accepted = dict(pair)
    accepted["status"] = "accepted_downloaded"
    accepted["downloaded_paths"] = {
        "t1_s2": str(local_path(t1_s2)),
        "t2_s2": str(local_path(t2_s2)),
        "t1_s3": str(local_path(t1_s3)),
        "t2_s3": str(local_path(t2_s3)),
    }
    accepted["local_validation"] = {"t1_s2_data_percentage": p1, "t2_s2_data_percentage": p2}
    accepted["download_account"] = account["username"]
    return accepted, "ok"



def final_pair_limit():
    return max(1, int(getattr(cfg, "PAIRS_PER_MGRS", 1)))


def accepted_list(value):
    if not value:
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [value]
    return []


def process_record(record, account, existing_accepted, existing_rejections):
    tile = record.get("mgrs_tile")
    target = final_pair_limit()
    accepted_items = accepted_list(existing_accepted)
    reject_history = list(existing_rejections or [])
    accepted_ranks = {item.get("rank") for item in accepted_items}
    rejected_ranks = {item.get("rank") for item in reject_history}
    for pair in record.get("candidate_pairs", []):
        if len(accepted_items) >= target:
            break
        if pair.get("rank") in accepted_ranks or pair.get("rank") in rejected_ranks:
            continue
        C.progress(
            f"02 try {tile} rank={pair.get('rank')} interval={pair.get('interval_days')} "
            f"round={pair.get('download_selection_round', '')} account={account['username']}"
        )
        try:
            result, reason = try_candidate(pair, account)
        except Exception as exc:
            result, reason = None, f"exception:{exc}"
        if result is not None:
            accepted_items.append(result)
            accepted_ranks.add(pair.get("rank"))
            C.progress(f"02 accepted {tile} rank={pair.get('rank')} ({len(accepted_items)}/{target})")
            continue
        reject_history.append({"rank": pair.get("rank"), "reason": reason, "time": datetime.now().isoformat(timespec="seconds")})
        rejected_ranks.add(pair.get("rank"))
        C.progress(f"02 rejected {tile} rank={pair.get('rank')}: {reason}")
    return tile, accepted_items, reject_history


def progress_reporter(stop_event):
    last_total = 0
    last_time = time.time()
    while not stop_event.wait(10):
        with _progress_lock:
            total = sum(_progress_bytes.values())
            done = len(_progress_done)
        now = time.time()
        speed = (total - last_total) / max(now - last_time, 0.001) / (1024 * 1024)
        C.progress(f"02 download speed={speed:.2f} MiB/s completed_files={done}")
        last_total = total
        last_time = now



def main():
    accounts = configured_accounts()
    queue = C.read_json(cfg.CDSE_CANDIDATE_QUEUE_JSON, {})
    records = queue.get("mgrs_records", [])
    if not records:
        raise SystemExit(f"No candidate records found: {cfg.CDSE_CANDIDATE_QUEUE_JSON}")
    state = load_state()
    accepted = state.setdefault("accepted", {})
    rejected = state.setdefault("rejected", {})
    target = final_pair_limit()
    jobs = []
    for record in records:
        tile = record.get("mgrs_tile")
        if not tile or not record.get("candidate_pairs"):
            continue
        if len(accepted_list(accepted.get(tile))) < target:
            jobs.append(record)
    accepted_tiles = sum(1 for value in accepted.values() if accepted_list(value))
    accepted_pairs = sum(len(accepted_list(value)) for value in accepted.values())
    C.progress(
        f"02 v3 download candidates: records={len(records)}, pending={len(jobs)}, "
        f"accepted_tiles_existing={accepted_tiles}, accepted_pairs_existing={accepted_pairs}, "
        f"target_pairs_per_mgrs={target}, accounts={len(accounts)}, workers={cfg.DOWNLOAD_WORKERS}"
    )

    stop_event = threading.Event()
    reporter = threading.Thread(target=progress_reporter, args=(stop_event,), daemon=True)
    reporter.start()
    account_cycle = cycle(accounts)
    try:
        with ThreadPoolExecutor(max_workers=max(1, int(cfg.DOWNLOAD_WORKERS))) as executor:
            futures = {}
            for record in jobs:
                tile = record.get("mgrs_tile")
                account = next(account_cycle)
                futures[executor.submit(process_record, record, account, accepted.get(tile), rejected.get(tile, []))] = tile
            for idx, future in enumerate(as_completed(futures), 1):
                tile = futures[future]
                try:
                    tile, accepted_items, reject_history = future.result()
                    if accepted_items:
                        accepted[tile] = accepted_items
                        C.progress(f"02 accepted tile {tile}: pairs={len(accepted_items)}/{target} ({idx}/{len(futures)})")
                    rejected[tile] = reject_history
                except Exception as exc:
                    rejected.setdefault(tile, []).append({"rank": "record", "reason": f"exception:{exc}", "time": datetime.now().isoformat(timespec="seconds")})
                    C.progress(f"02 failed record {tile}: {exc}")
                save_state(state)
    finally:
        stop_event.set()
        reporter.join(timeout=2)

    C.write_json(cfg.CDSE_ACCEPTED_DOWNLOADS_JSON, state)
    accepted_tiles = sum(1 for value in accepted.values() if accepted_list(value))
    accepted_pairs = sum(len(accepted_list(value)) for value in accepted.values())
    C.progress(f"02 finish: accepted_tiles={accepted_tiles}, accepted_pairs={accepted_pairs}, rejected_tiles={len(rejected)}")
    print({"accepted_tiles": accepted_tiles, "accepted_pairs": accepted_pairs, "rejected_tiles": len(rejected)})


if __name__ == "__main__":
    main()

