"""
Prepare S2/S3 inputs for the v2 512 LMDB builders.

This file is the new first-step entry point. It does not modify the original
cdse_batch_pipeline scripts. Edit CONFIG below and run the phases you need:

1. CDSE global data: optionally rerun the old manifest/download scripts, keeping
   their original outputs under get_data/cdse_batch_pipeline_out.
2. Existing downloads without manifest: build a lightweight manifest from zip
   names. This is only a fallback when the downloads were already curated as
   matched S2/S3 pairs.
3. Domestic/server data without manifest: scan server-style processed folders
   containing t1c/t1f/t2c/t2f and write an index for
   build_lmdb_512_from_server_tiles.py.
"""

import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]  # get_data
SCRIPT_DIR = Path(__file__).resolve().parent
OLD_PIPELINE_DIR = ROOT / "cdse_batch_pipeline"

# =========================
# CONFIG: edit values here.
# =========================
RUN_OLD_CDSE_MANIFEST = False
RUN_OLD_CDSE_DOWNLOAD = False
BUILD_LOCAL_MANIFEST_FROM_DOWNLOADS = False
BUILD_SERVER_TILE_INDEX = True

PYTHON_EXE = Path(sys.executable)
OLD_MANIFEST_SCRIPT = OLD_PIPELINE_DIR / "01_build_manifest.py"
OLD_DOWNLOAD_SCRIPT = OLD_PIPELINE_DIR / "02_download_from_manifest.py"

CDSE_OUT_DIR = Path("cdse_batch_pipeline_out")
DOWNLOAD_ROOT = CDSE_OUT_DIR / "downloads"
LOCAL_MANIFEST_PATH = CDSE_OUT_DIR / "local_manifest_from_downloads.json"
LOCAL_MANIFEST_SUMMARY_CSV = CDSE_OUT_DIR / "local_manifest_from_downloads_summary.csv"

# Point this to the domestic processed tile root produced by server_S23/utils.py.
# The scanner accepts any nested folder that contains t1c, t1f, t2c, and t2f.
SERVER_TILE_ROOT = Path("server_S23_out/tiles")
SERVER_TILE_INDEX_JSON = Path("s23_lmdb_512_out/server_tile_index.json")
SERVER_TILE_INDEX_CSV = Path("s23_lmdb_512_out/server_tile_index.csv")

LOCAL_PAIR_STRATEGY = "adjacent_dates"


def run_old_script(script_path):
    if not script_path.exists():
        raise SystemExit(f"Missing old pipeline script: {script_path}")
    print(f"Running old pipeline script without changing its logic: {script_path}")
    subprocess.run([str(PYTHON_EXE), str(script_path)], cwd=str(ROOT), check=True)


def product_name_from_zip(path):
    name = path.name
    return name[:-4] if name.lower().endswith(".zip") else name


def extract_mgrs_from_s2_name(name):
    match = re.search(r"_T([0-9]{2}[A-Z]{3})_", name)
    return match.group(1) if match else None


def extract_s2_date_from_name(name):
    match = re.search(r"_MSIL2A_(\d{8})T", name)
    return match.group(1) if match else None


def extract_s3_date_from_name(name):
    match = re.search(r"OL_1_EFR____(\d{8})T", name)
    return match.group(1) if match else None


def interval_months(date1, date2):
    d1 = datetime.strptime(date1, "%Y%m%d")
    d2 = datetime.strptime(date2, "%Y%m%d")
    return (d2.year - d1.year) * 12 + d2.month - d1.month


def product_record(name):
    return {"id": None, "name": name}


def build_local_manifest_from_downloads(download_root):
    files = sorted(Path(download_root).rglob("*.zip"))
    s2_by_tile_date = defaultdict(list)
    s3_by_date = defaultdict(list)
    for path in files:
        name = product_name_from_zip(path)
        if "MSIL2A_" in name:
            date = extract_s2_date_from_name(name)
            tile = extract_mgrs_from_s2_name(name)
            if date and tile:
                s2_by_tile_date[(tile, date)].append(name)
        elif "OL_1_EFR" in name:
            date = extract_s3_date_from_name(name)
            if date:
                s3_by_date[date].append(name)

    observations_by_tile = defaultdict(list)
    warnings = []
    for (tile, date), s2_names in sorted(s2_by_tile_date.items()):
        s3_names = sorted(s3_by_date.get(date, []))
        if not s3_names:
            continue
        if len(s2_names) > 1:
            warnings.append({"type": "multiple_s2_same_tile_date", "tile": tile, "date": date, "count": len(s2_names)})
        if len(s3_names) > 1:
            warnings.append({"type": "multiple_s3_same_date_selected_first", "date": date, "count": len(s3_names)})
        observations_by_tile[tile].append(
            {"date": date, "s2": product_record(sorted(s2_names)[0]), "s3": product_record(s3_names[0])}
        )

    groups = []
    for tile, observations in sorted(observations_by_tile.items()):
        observations = sorted(observations, key=lambda item: item["date"])
        if LOCAL_PAIR_STRATEGY != "adjacent_dates":
            raise ValueError(f"Unsupported LOCAL_PAIR_STRATEGY: {LOCAL_PAIR_STRATEGY}")
        for idx in range(len(observations) - 1):
            t1 = observations[idx]
            t2 = observations[idx + 1]
            if t1["date"] == t2["date"]:
                continue
            groups.append(
                {
                    "group_id": f"local_{tile}_{idx + 1:04d}",
                    "location": {"id": tile, "tile_id": tile, "mgrs_tile_id": tile},
                    "interval_months": interval_months(t1["date"], t2["date"]),
                    "target_interval_months": "",
                    "target_start_year": "",
                    "t1": t1,
                    "t2": t2,
                    "source": "local_download_inventory",
                }
            )

    manifest = {
        "schema": "local_s2_s3_group_manifest_from_downloads_v1",
        "notes": [
            "Fallback manifest built from local zip names only.",
            "Use only when downloads were already curated as matched pairs; CDSE manifest is preferred for S3 coverage provenance.",
        ],
        "download_root": str(download_root),
        "groups": groups,
        "warnings": warnings,
    }
    return manifest


def write_local_manifest(manifest, path, summary_csv):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows = []
    for group in manifest["groups"]:
        rows.append(
            {
                "group_id": group["group_id"],
                "tile_id": group["location"].get("tile_id", ""),
                "t1_date": group["t1"]["date"],
                "t2_date": group["t2"]["date"],
                "interval_months": group.get("interval_months", ""),
                "t1_s2": group["t1"]["s2"]["name"],
                "t1_s3": group["t1"]["s3"]["name"],
                "t2_s2": group["t2"]["s2"]["name"],
                "t2_s3": group["t2"]["s3"]["name"],
            }
        )
    with summary_csv.open("w", newline="", encoding="utf-8") as fd:
        writer = csv.DictWriter(fd, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"Local manifest written: {path} groups={len(manifest['groups'])} warnings={len(manifest['warnings'])}")
    print(f"Local manifest summary written: {summary_csv}")


def is_server_tile_group(folder):
    required = [folder / name for name in ("t1c", "t1f", "t2c", "t2f")]
    if not all(path.is_dir() for path in required):
        return False
    names = None
    for sub in required:
        current = {path.name for path in sub.glob("*.tif")}
        names = current if names is None else names & current
    return bool(names)


def scan_server_tile_groups(root):
    root = Path(root)
    groups = []
    if not root.exists():
        print(f"Server tile root does not exist yet: {root}")
        return groups
    for folder in sorted([root] + [path for path in root.rglob("*") if path.is_dir()]):
        if not is_server_tile_group(folder):
            continue
        common_names = None
        for sub_name in ("t1c", "t1f", "t2c", "t2f"):
            current = {path.name for path in (folder / sub_name).glob("*.tif")}
            common_names = current if common_names is None else common_names & current
        groups.append(
            {
                "group_id": folder.name,
                "base_path": str(folder),
                "tile_names": sorted(common_names),
                "tile_count": len(common_names),
                "source": "server_style_processed_tiles",
            }
        )
    return groups


def write_server_tile_index(groups, json_path, csv_path):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({"schema": "server_tile_index_v1", "groups": groups}, indent=2), encoding="utf-8")
    rows = [{"group_id": g["group_id"], "base_path": g["base_path"], "tile_count": g["tile_count"]} for g in groups]
    with csv_path.open("w", newline="", encoding="utf-8") as fd:
        writer = csv.DictWriter(fd, fieldnames=list(rows[0]) if rows else ["group_id", "base_path", "tile_count"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Server tile index written: {json_path} groups={len(groups)}")
    print(f"Server tile index CSV written: {csv_path}")


def main():
    if RUN_OLD_CDSE_MANIFEST:
        run_old_script(OLD_MANIFEST_SCRIPT)
    if RUN_OLD_CDSE_DOWNLOAD:
        run_old_script(OLD_DOWNLOAD_SCRIPT)
    if BUILD_LOCAL_MANIFEST_FROM_DOWNLOADS:
        manifest = build_local_manifest_from_downloads(DOWNLOAD_ROOT)
        write_local_manifest(manifest, LOCAL_MANIFEST_PATH, LOCAL_MANIFEST_SUMMARY_CSV)
    if BUILD_SERVER_TILE_INDEX:
        groups = scan_server_tile_groups(SERVER_TILE_ROOT)
        write_server_tile_index(groups, SERVER_TILE_INDEX_JSON, SERVER_TILE_INDEX_CSV)


if __name__ == "__main__":
    main()
