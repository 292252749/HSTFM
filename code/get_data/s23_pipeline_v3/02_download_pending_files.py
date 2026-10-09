"""Download pending files from download_plan.json with resume and retries."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

import s23_common as C

cfg = C.cfg


def validate_download(path, sensor):
    ok, reason = C.validate_zip(path, sensor)
    return ok, reason


def download_one(item):
    out_path = Path(item["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok, reason = validate_download(out_path, item["sensor"])
    if ok:
        return {**item, "status": "exists_valid", "reason": reason}

    part_path = out_path.with_suffix(out_path.suffix + ".part")
    last_error = reason
    for attempt in range(1, cfg.DOWNLOAD_MAX_RETRIES + 1):
        try:
            resume_from = part_path.stat().st_size if part_path.exists() else 0
            headers = {"Range": f"bytes={resume_from}-"} if resume_from > 0 else {}
            C.progress(f"02 download: {item['name']} attempt {attempt}/{cfg.DOWNLOAD_MAX_RETRIES}, resume_from={resume_from}")
            with requests.get(item["url"], headers=headers, stream=True, timeout=cfg.DOWNLOAD_TIMEOUT_SECONDS) as response:
                if resume_from > 0 and response.status_code == 200:
                    part_path.unlink(missing_ok=True)
                    resume_from = 0
                response.raise_for_status()
                mode = "ab" if resume_from > 0 and response.status_code == 206 else "wb"
                with part_path.open(mode) as fd:
                    for chunk in response.iter_content(chunk_size=cfg.DOWNLOAD_CHUNK_SIZE):
                        if chunk:
                            fd.write(chunk)
            part_path.replace(out_path)
            ok, reason = validate_download(out_path, item["sensor"])
            if ok:
                return {**item, "status": "downloaded_valid", "reason": reason, "attempts": attempt}
            last_error = reason
        except Exception as exc:
            last_error = str(exc)
            C.progress(f"02 download: {item['name']} failed attempt {attempt}: {last_error}")
            C.sleep_retry(attempt)
    return {**item, "status": "failed", "reason": last_error}


def main():
    C.progress(f"02 start: dataset_group={cfg.DATASET_GROUP}")
    plan = C.read_json(cfg.DOWNLOAD_PLAN_JSON, {"items": []})
    status = C.read_json(cfg.DOWNLOAD_STATUS_JSON, {"items": []})
    done = {item["name"]: item for item in status.get("items", []) if item.get("status") in {"exists_valid", "downloaded_valid"}}
    results = list(done.values())
    pending_by_name = {}
    seen_existing = set(done)
    for item in plan.get("items", []):
        if item["name"] in done or item["name"] in seen_existing:
            continue
        if item.get("status") == "exists_valid":
            results.append(item)
            seen_existing.add(item["name"])
        else:
            pending_by_name.setdefault(item["name"], item)
    pending = list(pending_by_name.values())
    total = len(pending)
    C.progress(f"02 plan: {len(done)} already done, {total} pending downloads")

    workers = max(1, int(getattr(cfg, "DOWNLOAD_WORKERS", 1)))
    step = C.progress_every(0, total)
    if workers == 1:
        for idx, item in enumerate(pending, 1):
            result = download_one(item)
            results.append(result)
            C.write_json(cfg.DOWNLOAD_STATUS_JSON, {"schema": "s23_download_status_v1", "items": results})
            print(f"{result['status']}: {result['name']}", flush=True)
            if idx == total or idx % step == 0:
                C.progress(f"02 download: completed {idx}/{total} pending downloads")
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(download_one, item) for item in pending]
            for idx, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)
                C.write_json(cfg.DOWNLOAD_STATUS_JSON, {"schema": "s23_download_status_v1", "items": results})
                print(f"{result['status']}: {result['name']}", flush=True)
                if idx == total or idx % step == 0:
                    C.progress(f"02 download: completed {idx}/{total} pending downloads")
    C.write_json(cfg.DOWNLOAD_STATUS_JSON, {"schema": "s23_download_status_v1", "items": results})
    C.progress(f"02 finish: wrote download status with {len(results)} records")


if __name__ == "__main__":
    main()
