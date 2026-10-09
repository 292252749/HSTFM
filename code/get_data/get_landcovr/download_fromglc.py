"""
Download FROMGLC30 V002 tiles from iEarth URL lists with resume and retries.

Edit CONFIG below, then run:
    python get_data/get_landcovr/download_fromglc.py

The input text file may contain comments and separator lines; every http(s)
line is treated as a download URL. Existing complete files are skipped. Partial
files are stored as .part and resumed with HTTP Range when the server supports
it.
"""

import json
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests


# =========================
# CONFIG: edit values here.
# =========================
LINK_FILE = Path("get_data/get_landcovr/iearth_download_links_2026-07-02T04-14-41-816Z.txt")
OUT_DIR = Path("get_data/get_landcovr/fromglc2024")
STATUS_PATH = OUT_DIR / "download_status.json"
TIMEOUT_SECONDS = 120
CHUNK_SIZE = 1024 * 1024
MAX_RETRIES = 5
RETRY_SLEEP_SECONDS = 10


def iter_urls(path):
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("="):
            continue
        if line.startswith("http://") or line.startswith("https://"):
            yield line


def filename_from_url(url):
    parsed = urlparse(url)
    return unquote(Path(parsed.path).name)


def load_status(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"downloads": {}}


def save_status(path, status):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")


def remote_size(url):
    try:
        response = requests.head(url, timeout=TIMEOUT_SECONDS, allow_redirects=True)
        if response.ok and response.headers.get("Content-Length"):
            return int(response.headers["Content-Length"])
    except Exception:
        return None
    return None


def is_complete(path, expected_size):
    if not path.exists() or path.stat().st_size == 0:
        return False
    return expected_size is None or path.stat().st_size == expected_size


def download_one(url, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = filename_from_url(url)
    out_path = out_dir / filename
    part_path = out_path.with_suffix(out_path.suffix + ".part")
    expected_size = remote_size(url)

    if is_complete(out_path, expected_size):
        return {"status": "exists", "path": str(out_path), "bytes": out_path.stat().st_size}

    for attempt in range(1, MAX_RETRIES + 1):
        resume_at = part_path.stat().st_size if part_path.exists() else 0
        headers = {}
        mode = "wb"
        if resume_at > 0:
            headers["Range"] = f"bytes={resume_at}-"
            mode = "ab"

        try:
            with requests.get(url, stream=True, timeout=TIMEOUT_SECONDS, headers=headers) as response:
                if response.status_code == 416:
                    part_path.replace(out_path)
                    return {"status": "downloaded", "path": str(out_path), "bytes": out_path.stat().st_size}
                response.raise_for_status()
                if resume_at > 0 and response.status_code == 200:
                    # Server ignored Range; restart this file.
                    mode = "wb"
                    resume_at = 0
                with part_path.open(mode) as fd:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            fd.write(chunk)

            if expected_size is not None and part_path.stat().st_size != expected_size:
                raise RuntimeError(
                    f"size mismatch for {filename}: got {part_path.stat().st_size}, expected {expected_size}"
                )
            part_path.replace(out_path)
            return {"status": "downloaded", "path": str(out_path), "bytes": out_path.stat().st_size}
        except Exception as exc:
            if attempt >= MAX_RETRIES:
                return {"status": "failed", "path": str(out_path), "error": str(exc)}
            print(f"retry {attempt}/{MAX_RETRIES} for {filename}: {exc}")
            time.sleep(RETRY_SLEEP_SECONDS)

    return {"status": "failed", "path": str(out_path), "error": "exhausted retries"}


def main():
    urls = list(iter_urls(LINK_FILE))
    if not urls:
        raise SystemExit(f"No URLs found in {LINK_FILE}")
    status = load_status(STATUS_PATH)
    print(f"FROMGLC URLs: {len(urls)}")
    print(f"Output: {OUT_DIR}")

    for index, url in enumerate(urls, start=1):
        filename = filename_from_url(url)
        print(f"[{index}/{len(urls)}] {filename}")
        result = download_one(url, OUT_DIR)
        status["downloads"][filename] = result
        save_status(STATUS_PATH, status)
        print(f"  {result['status']}: {result.get('bytes', '')} {result.get('error', '')}")

    counts = {}
    for item in status["downloads"].values():
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    status["summary"] = counts
    save_status(STATUS_PATH, status)
    print(f"Summary: {counts}")


if __name__ == "__main__":
    main()


