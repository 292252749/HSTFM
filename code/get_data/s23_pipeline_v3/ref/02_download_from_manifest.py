"""
Download all products listed in a CDSE batch manifest.

Edit the CONFIG section below to change accounts, concurrency, paths, retries,
or validation behavior. This script intentionally does not use command-line
arguments.
"""

import json
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import cycle
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CDSE_DOWNLOAD = "https://download.dataspace.copernicus.eu/odata/v1"
CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)


def get_access_token(username, password):
    response = requests.post(
        CDSE_TOKEN_URL,
        data={
            "client_id": "cdse-public",
            "username": username,
            "password": password,
            "grant_type": "password",
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["access_token"]

# =========================
# CONFIG: edit values here.
# =========================
MANIFEST_PATH = Path("/data1/code/download/manifest.json")
DOWNLOAD_DIR = Path("/data2/Sentinel23_globe")
STATUS_PATH = Path("/data2/Sentinel23_globe/download_status.json")
MAX_WORKERS = 8
MAX_RETRIES = 8
RETRY_SLEEP_SECONDS = 8
CHUNK_SIZE = 1024 * 1024
PROGRESS_PRINT_SECONDS = 5
CONNECT_TIMEOUT_SECONDS = 30
READ_TIMEOUT_SECONDS = 60
RESUME_PARTIAL_DOWNLOADS = True

# Put one or more CDSE accounts here. Downloads are assigned round-robin.
ACCOUNTS = [
    {"username": "2216031697@qq.com", "password": "Zhangjie780~!"},
    {"username": "76949069@qq.com", "password": "Zhangjie780~!"},
    {"username": "xuchen@aircas.ac.cn", "password": "Aa_123456789"},
    {"username": "jieryuer@163.com", "password": "Zhangjie780~!"},

]


progress_lock = threading.Lock()
progress_bytes = {}
progress_done = {}
status_lock = threading.Lock()


def iter_products(manifest):
    seen = set()
    for group in manifest["groups"]:
        for time_key in ("t1", "t2"):
            for sensor in ("s2", "s3"):
                product = group[time_key][sensor]
                key = product["id"]
                if key in seen:
                    continue
                seen.add(key)
                yield {
                    "id": product["id"],
                    "name": product["name"],
                    "sensor": sensor,
                    "group_id": group["group_id"],
                    "time_key": time_key,
                    "expected_size": product.get("size"),
                }


def zip_name(product):
    name = product["name"]
    if not name.endswith(".zip"):
        name += ".zip"
    return name


def validate_zip(path, sensor):
    if not path.exists() or path.stat().st_size <= 0:
        return False, "missing_or_empty"
    try:
        with zipfile.ZipFile(path, "r") as zip_ref:
            bad = zip_ref.testzip()
            if bad:
                return False, f"bad_member:{bad}"
            names = zip_ref.namelist()
    except zipfile.BadZipFile:
        return False, "bad_zip"

    if sensor == "s2":
        required = ("B02", "B03", "B04", "B8A")
        for band in required:
            if not any(band in name and "20m" in name and name.endswith(".jp2") for name in names):
                return False, f"missing_s2_band:{band}"
    elif sensor == "s3":
        if not any("geo_coordinates.nc" in name for name in names):
            return False, "missing_s3_geo"
        for band in ("Oa04", "Oa06", "Oa08", "Oa17"):
            if not any(Path(name).name.startswith(band + "_") and name.endswith("radiance.nc") for name in names):
                return False, f"missing_s3_band:{band}"
    return True, "ok"


def write_status(results):
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with status_lock:
        STATUS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")


def progress_reporter(stop_event):
    last_total = 0
    last_time = time.time()
    while not stop_event.wait(PROGRESS_PRINT_SECONDS):
        with progress_lock:
            total = sum(progress_bytes.values())
            done = len(progress_done)
        now = time.time()
        speed = (total - last_total) / max(now - last_time, 0.001) / (1024 * 1024)
        print(f"Download speed: {speed:.2f} MB/s, completed files: {done}")
        last_total = total
        last_time = now


def expected_size_text(product):
    size = product.get("expected_size")
    if not isinstance(size, (int, float)):
        return "unknown"
    return f"{size / 1024**3:.2f} GiB"


def download_one(product, account):
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DOWNLOAD_DIR / zip_name(product)
    ok, reason = validate_zip(out_path, product["sensor"])
    if ok:
        with progress_lock:
            progress_done[str(out_path)] = True
        return {"name": product["name"], "path": str(out_path), "status": "exists_valid", "reason": reason}

    url = f"{CDSE_DOWNLOAD}/Products({product['id']})/$value"
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    last_error = reason

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            token = get_access_token(account["username"], account["password"])
            resume_from = tmp_path.stat().st_size if RESUME_PARTIAL_DOWNLOADS and tmp_path.exists() else 0
            expected_size = product.get("expected_size")
            if isinstance(expected_size, int) and resume_from > expected_size:
                tmp_path.unlink()
                resume_from = 0

            headers = {"Authorization": f"Bearer {token}"}
            if resume_from > 0:
                headers["Range"] = f"bytes={resume_from}-"
            print(
                f"download {product['sensor']} {product['name']} "
                f"attempt={attempt}/{MAX_RETRIES} resume={resume_from / 1024**2:.1f} MiB "
                f"expected={expected_size_text(product)}"
            )

            with requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
            ) as response:
                if resume_from > 0 and response.status_code == 200:
                    print(f"  server ignored Range for {product['name']}; restarting this file")
                    if tmp_path.exists():
                        tmp_path.unlink()
                    resume_from = 0
                elif response.status_code == 416:
                    tmp_path.replace(out_path)
                    ok, reason = validate_zip(out_path, product["sensor"])
                    if ok:
                        with progress_lock:
                            progress_done[str(out_path)] = True
                        return {
                            "name": product["name"],
                            "path": str(out_path),
                            "status": "downloaded_valid",
                            "account": account["username"],
                            "attempts": attempt,
                            "resumed": True,
                        }
                response.raise_for_status()
                mode = "ab" if resume_from > 0 and response.status_code == 206 else "wb"
                with tmp_path.open(mode) as fd:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        fd.write(chunk)
                        with progress_lock:
                            progress_bytes[str(out_path)] = progress_bytes.get(str(out_path), 0) + len(chunk)
            tmp_path.replace(out_path)
            ok, reason = validate_zip(out_path, product["sensor"])
            if ok:
                with progress_lock:
                    progress_done[str(out_path)] = True
                return {
                    "name": product["name"],
                    "path": str(out_path),
                    "status": "downloaded_valid",
                    "account": account["username"],
                    "attempts": attempt,
                    "resumed": resume_from > 0,
                }
            last_error = reason
        except Exception as exc:
            last_error = str(exc)
            print(f"  retryable download error for {product['name']}: {last_error}")
        time.sleep(RETRY_SLEEP_SECONDS * attempt)

    return {
        "name": product["name"],
        "path": str(out_path),
        "status": "failed",
        "account": account["username"],
        "error": last_error,
    }


def main():
    if not ACCOUNTS or ACCOUNTS[0]["username"].startswith("your_account"):
        raise SystemExit("Edit ACCOUNTS in this file before downloading.")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    products = list(iter_products(manifest))
    print(f"Products to download: {len(products)}")

    account_cycle = cycle(ACCOUNTS)
    assignments = [(product, next(account_cycle)) for product in products]

    stop_event = threading.Event()
    reporter = threading.Thread(target=progress_reporter, args=(stop_event,), daemon=True)
    reporter.start()

    results = []
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(download_one, product, account)
                for product, account in assignments
            ]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"{result['status']}: {result['name']}")
                write_status(results)
    finally:
        stop_event.set()
        reporter.join(timeout=2)

    write_status(results)
    failed = [item for item in results if item["status"] == "failed"]
    print(f"Status written: {STATUS_PATH}")
    if failed:
        raise SystemExit(f"Failed downloads: {len(failed)}")
    print("All downloads are complete and zip integrity checks passed.")


if __name__ == "__main__":
    main()
