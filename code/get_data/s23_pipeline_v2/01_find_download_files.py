"""Find local S2/S3 zip files and build a pending download plan from URL lists."""

from pathlib import Path

import s23_common as C

cfg = C.cfg


def read_urls():
    urls = []
    for list_path in cfg.DOWNLOAD_LINK_FILES:
        path = Path(list_path)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            urls.append(text)
    return urls


def main():
    C.progress(f"01 start: dataset_group={cfg.DATASET_GROUP}")
    local = C.scan_local_zips(cfg.SEARCH_ROOTS + [cfg.DOWNLOAD_ROOT], label="01 local inventory")
    by_name = {Path(item["name"]).name: item for item in local if item["valid"]}
    pending = []
    known_urls = []
    urls = read_urls()
    C.progress(f"01 url plan: read {len(urls)} URLs")
    step = C.progress_every(0, len(urls))
    for idx, url in enumerate(urls, 1):
        filename = C.filename_from_url(url)
        name = C.product_name_from_path(filename)
        sensor = C.sensor_from_name(name)
        if not sensor:
            continue
        status = "exists_valid" if name in by_name else "pending"
        item = {
            "url": url,
            "filename": filename,
            "name": name,
            "sensor": sensor,
            "date": C.product_date(name),
            "mgrs_tile": C.extract_mgrs_from_s2_name(name) if sensor == "s2" else "",
            "out_path": str(cfg.DOWNLOAD_ROOT / filename),
            "status": status,
        }
        known_urls.append(item)
        if status == "pending":
            pending.append(item)
        if idx == len(urls) or idx % step == 0:
            C.progress(f"01 url plan: processed {idx}/{len(urls)} URLs")

    inventory = {
        "schema": "s23_file_inventory_v1",
        "dataset_group": cfg.DATASET_GROUP,
        "dataset_name": cfg.DATASET_NAME,
        "search_roots": [str(path) for path in cfg.SEARCH_ROOTS],
        "local_files": local,
        "counts": {
            "local_total": len(local),
            "local_valid": sum(1 for item in local if item["valid"]),
            "local_invalid": sum(1 for item in local if not item["valid"]),
        },
    }
    plan = {
        "schema": "s23_download_plan_from_urls_v1",
        "dataset_group": cfg.DATASET_GROUP,
        "dataset_name": cfg.DATASET_NAME,
        "download_root": str(cfg.DOWNLOAD_ROOT),
        "items": known_urls,
        "counts": {"total_urls": len(known_urls), "pending": len(pending), "exists_valid": len(known_urls) - len(pending)},
    }
    C.write_json(cfg.FILE_INVENTORY_JSON, inventory)
    C.write_json(cfg.DOWNLOAD_PLAN_JSON, plan)
    C.progress("01 finish: wrote inventory and download plan")
    print(f"dataset group: {cfg.DATASET_GROUP}")
    print(f"file inventory written: {cfg.FILE_INVENTORY_JSON}")
    print(f"download plan written: {cfg.DOWNLOAD_PLAN_JSON}")
    print(inventory["counts"])
    print(plan["counts"])


if __name__ == "__main__":
    main()

