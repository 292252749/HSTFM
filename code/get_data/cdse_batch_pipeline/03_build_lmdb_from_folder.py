"""
Traverse downloaded files, pair S2/S3 products from a manifest, and build LMDB.

Edit the CONFIG section below to change input/output paths and processing
behavior. This script intentionally does not use command-line arguments.
"""

import json
import pickle
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cdse_one_location import (
    S2_BAD_VALUE,
    S3_BAD_VALUE,
    SAMPLE_HR_SIZE,
    SAMPLE_LR_SIZE,
    has_bad_value,
    pack_arrays,
    write_s2_tiles,
    write_s3_tile,
)


# =========================
# CONFIG: edit values here.
# =========================
MANIFEST_PATH = Path("cdse_batch_pipeline_out/manifest.json")
DOWNLOAD_ROOT = Path("cdse_batch_pipeline_out/downloads")
PROCESSED_ROOT = Path("cdse_batch_pipeline_out/processed")
LMDB_PATH = Path("cdse_batch_pipeline_out/lmdb")
TEST_MODE = True
TEST_LMDB_PATH = Path("cdse_batch_pipeline_out/lmdb_test_one_group")
TEST_PROCESSED_ROOT = Path("cdse_batch_pipeline_out/processed_test_one_group")
MAX_GROUPS_TO_PROCESS = 1
PROCESS_FULL_MGRS_TILE = True
FILTER_INVALID_GROUPS_BEFORE_PROCESSING = True
ESTIMATED_BLOCKS_PER_GROUP = 25
LMDB_MAP_SIZE = None
LMDB_BYTES_PER_SAMPLE_ESTIMATE = 6 * 1024**2
LMDB_MAP_SIZE_MARGIN = 2.0
LMDB_MIN_MAP_SIZE = 8 * 1024**3
LMDB_BATCH_SIZE = 500
STOP_ON_PROCESS_ERROR = False
SUBSAMPLE_GRID = 2


def product_zip_name(product):
    name = product["name"]
    if not name.endswith(".zip"):
        name += ".zip"
    return name


def index_downloads(root):
    files = {}
    for path in root.rglob("*.zip"):
        files[path.name] = path
    return files


def find_zip(product, files):
    name = product_zip_name(product)
    path = files.get(name)
    if path is None:
        raise FileNotFoundError(f"Missing downloaded file: {name}")
    return path


def read_arrays(paths, group_id, sub_i, sub_j):
    arrays = []
    windows = [
        ((sub_i * SAMPLE_LR_SIZE, (sub_i + 1) * SAMPLE_LR_SIZE), (sub_j * SAMPLE_LR_SIZE, (sub_j + 1) * SAMPLE_LR_SIZE)),
        ((sub_i * SAMPLE_HR_SIZE, (sub_i + 1) * SAMPLE_HR_SIZE), (sub_j * SAMPLE_HR_SIZE, (sub_j + 1) * SAMPLE_HR_SIZE)),
        ((sub_i * SAMPLE_LR_SIZE, (sub_i + 1) * SAMPLE_LR_SIZE), (sub_j * SAMPLE_LR_SIZE, (sub_j + 1) * SAMPLE_LR_SIZE)),
        ((sub_i * SAMPLE_HR_SIZE, (sub_i + 1) * SAMPLE_HR_SIZE), (sub_j * SAMPLE_HR_SIZE, (sub_j + 1) * SAMPLE_HR_SIZE)),
    ]
    bad_values = [S3_BAD_VALUE, S2_BAD_VALUE, S3_BAD_VALUE, S2_BAD_VALUE]
    for path, window, bad_value in zip(paths, windows, bad_values):
        with rasterio.open(path) as src:
            arr = src.read(window=window).astype(np.uint16)
        if has_bad_value(arr, bad_value):
            return None, f"bad_value:{path}"
        arrays.append(arr)
    return arrays, "ok"


def tile_key(path):
    return path.stem


def existing_tiles(folder):
    return sorted(folder.glob("tile_*.tif"))


def ensure_s2_tile_set(zip_path, folder, label):
    existing = existing_tiles(folder)
    if existing:
        print(f"  reuse S2 tile set {label}: {len(existing)} files in {folder}")
        return existing, "reused"
    print(f"  generate S2 tile set {label}: {folder}")
    written = write_s2_tiles(zip_path, folder)
    print(f"  generated S2 tile set {label}: {len(written)} files")
    return written, "generated"


def ensure_s3_tile(s3_zip_path, example_s2_tif, out_path, label):
    if out_path.exists():
        return "reused"
    print(f"    generate S3 tile {label}: {out_path}")
    write_s3_tile(s3_zip_path, example_s2_tif, out_path)
    return "generated"


def validate_zip(path):
    with zipfile.ZipFile(path, "r") as zip_ref:
        bad = zip_ref.testzip()
        if bad:
            raise RuntimeError(f"Bad zip member in {path}: {bad}")


def active_lmdb_path():
    return TEST_LMDB_PATH if TEST_MODE else LMDB_PATH


def active_processed_root():
    return TEST_PROCESSED_ROOT if TEST_MODE else PROCESSED_ROOT


def selected_manifest_groups(manifest):
    groups = manifest["groups"]
    selected = []
    for group in groups:
        if FILTER_INVALID_GROUPS_BEFORE_PROCESSING:
            try:
                validate_group_pairing(group)
            except Exception as exc:
                print(f"skip manifest group before processing {group.get('group_id')}: {exc}")
                continue
        selected.append(group)
        if MAX_GROUPS_TO_PROCESS is not None and len(selected) >= MAX_GROUPS_TO_PROCESS:
            break
    return selected


def product_label(product):
    return product.get("name") or product.get("id") or "<unknown>"


def extract_mgrs_from_s2_name(name):
    match = re.search(r"_T([0-9]{2}[A-Z]{3})_", name)
    return match.group(1) if match else None


def extract_s2_date_from_name(name):
    match = re.search(r"_MSIL2A_(\d{8})T", name)
    return match.group(1) if match else None


def extract_s3_date_from_name(name):
    match = re.search(r"OL_1_EFR____(\d{8})T", name)
    return match.group(1) if match else None


def validate_same_mgrs_tile(group):
    t1_tile = extract_mgrs_from_s2_name(group["t1"]["s2"]["name"])
    t2_tile = extract_mgrs_from_s2_name(group["t2"]["s2"]["name"])
    if t1_tile is None or t2_tile is None:
        raise RuntimeError(f"Cannot parse MGRS tile from S2 product names for {group['group_id']}")
    if t1_tile != t2_tile:
        raise RuntimeError(f"S2 MGRS mismatch for {group['group_id']}: t1={t1_tile}, t2={t2_tile}")
    return t1_tile


def validate_group_pairing(group):
    mgrs_tile = validate_same_mgrs_tile(group)
    t1_s2_date = extract_s2_date_from_name(group["t1"]["s2"]["name"])
    t1_s3_date = extract_s3_date_from_name(group["t1"]["s3"]["name"])
    t2_s2_date = extract_s2_date_from_name(group["t2"]["s2"]["name"])
    t2_s3_date = extract_s3_date_from_name(group["t2"]["s3"]["name"])
    dates = {
        "t1_s2": t1_s2_date,
        "t1_s3": t1_s3_date,
        "t2_s2": t2_s2_date,
        "t2_s3": t2_s3_date,
    }
    missing = [key for key, value in dates.items() if value is None]
    if missing:
        raise RuntimeError(f"Cannot parse product dates for {group['group_id']}: {missing}")
    if t1_s2_date != t1_s3_date:
        raise RuntimeError(f"t1 S2/S3 date mismatch for {group['group_id']}: s2={t1_s2_date}, s3={t1_s3_date}")
    if t2_s2_date != t2_s3_date:
        raise RuntimeError(f"t2 S2/S3 date mismatch for {group['group_id']}: s2={t2_s2_date}, s3={t2_s3_date}")
    if t1_s2_date == t2_s2_date:
        raise RuntimeError(f"t1/t2 dates must differ for {group['group_id']}: date={t1_s2_date}")
    manifest_dates = {"t1": group["t1"].get("date"), "t2": group["t2"].get("date")}
    if manifest_dates["t1"] and manifest_dates["t1"] != t1_s2_date:
        raise RuntimeError(f"manifest t1 date mismatch for {group['group_id']}: manifest={manifest_dates['t1']}, product={t1_s2_date}")
    if manifest_dates["t2"] and manifest_dates["t2"] != t2_s2_date:
        raise RuntimeError(f"manifest t2 date mismatch for {group['group_id']}: manifest={manifest_dates['t2']}, product={t2_s2_date}")
    return mgrs_tile, dates


def print_group_inputs(group, group_index, total_groups):
    print(f"\n=== Group {group_index}/{total_groups}: {group['group_id']} ===")
    print(f"location: lon={group['location']['lon']}, lat={group['location']['lat']}")
    print(f"dates: t1={group['t1']['date']} t2={group['t2']['date']} interval={group['interval_months']} months")
    print("inputs:")
    print(f"  t1_s2: {product_label(group['t1']['s2'])}")
    print(f"  t1_s3: {product_label(group['t1']['s3'])}")
    print(f"  t2_s2: {product_label(group['t2']['s2'])}")
    print(f"  t2_s3: {product_label(group['t2']['s3'])}")


def ensure_tile(path, label, writer):
    if path.exists():
        print(f"  reuse tile {label}: {path}")
        return "reused"
    print(f"  generate tile {label}: {path}")
    writer()
    return "generated"


def estimate_lmdb_map_size(group_count):
    if LMDB_MAP_SIZE is not None:
        return LMDB_MAP_SIZE
    estimated_samples = group_count * ESTIMATED_BLOCKS_PER_GROUP * SUBSAMPLE_GRID * SUBSAMPLE_GRID
    estimated_bytes = int(estimated_samples * LMDB_BYTES_PER_SAMPLE_ESTIMATE * LMDB_MAP_SIZE_MARGIN)
    return max(LMDB_MIN_MAP_SIZE, estimated_bytes)


def source_product_metadata(group):
    return {
        "t1_s2": group["t1"]["s2"],
        "t1_s3": group["t1"]["s3"],
        "t2_s2": group["t2"]["s2"],
        "t2_s3": group["t2"]["s3"],
    }


def write_lmdb_metadata(env, total_samples, original_total_samples, skipped, map_size):
    with env.begin(write=True) as txn:
        txn.put(
            b"__metadata__",
            pickle.dumps(
                {
                    "total_samples": total_samples,
                    "original_total_samples": original_total_samples,
                    "skipped_groups": skipped,
                    "subsample_grid": [SUBSAMPLE_GRID, SUBSAMPLE_GRID],
                    "hr_shape": [4, SAMPLE_HR_SIZE, SAMPLE_HR_SIZE],
                    "lr_shape": [4, SAMPLE_LR_SIZE, SAMPLE_LR_SIZE],
                    "map_size": map_size,
                    "test_mode": TEST_MODE,
                    "max_groups_to_process": MAX_GROUPS_TO_PROCESS,
                    "process_full_mgrs_tile": PROCESS_FULL_MGRS_TILE,
                    "filter_invalid_groups_before_processing": FILTER_INVALID_GROUPS_BEFORE_PROCESSING,
                    "estimated_blocks_per_group": ESTIMATED_BLOCKS_PER_GROUP,
                    "sample_structure": "Each sample has 4 numpy arrays: [LR_t1, HR_t1, LR_t2, HR_t2]",
                }
            ),
        )


def main():
    import lmdb

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    files = index_downloads(DOWNLOAD_ROOT)
    if not files:
        raise SystemExit(f"No zip files found under {DOWNLOAD_ROOT}")

    groups = selected_manifest_groups(manifest)
    processed_root = active_processed_root()
    lmdb_path = active_lmdb_path()
    processed_root.mkdir(parents=True, exist_ok=True)
    lmdb_path.mkdir(parents=True, exist_ok=True)
    map_size = estimate_lmdb_map_size(len(groups))
    print(
        f"LMDB map_size={map_size / 1024**3:.1f} GiB for {len(groups)}/{len(manifest['groups'])} groups "
        f"test_mode={TEST_MODE}"
    )
    env = lmdb.open(str(lmdb_path), map_size=map_size)
    txn = env.begin(write=True)
    sample_count = 0
    original_total = 0
    skipped = []
    group_results = []

    for group_index, group in enumerate(groups, start=1):
        group_id = group["group_id"]
        lon = group["location"]["lon"]
        lat = group["location"]["lat"]
        group_dir = processed_root / group_id
        print_group_inputs(group, group_index, len(groups))
        try:
            mgrs_tile, pair_dates = validate_group_pairing(group)
            print(f"mgrs_tile: {mgrs_tile}")
            print(
                "pairing check: "
                f"t1_s2=t1_s3={pair_dates['t1_s2']}, "
                f"t2_s2=t2_s3={pair_dates['t2_s2']}, "
                "t1!=t2"
            )
            s2_t1 = find_zip(group["t1"]["s2"], files)
            s3_t1 = find_zip(group["t1"]["s3"], files)
            s2_t2 = find_zip(group["t2"]["s2"], files)
            s3_t2 = find_zip(group["t2"]["s3"], files)
            for path in (s2_t1, s3_t1, s2_t2, s3_t2):
                validate_zip(path)

            t1f_dir = group_dir / "t1f"
            t1c_dir = group_dir / "t1c"
            t2f_dir = group_dir / "t2f"
            t2c_dir = group_dir / "t2c"
            t1f_tiles, t1f_status = ensure_s2_tile_set(s2_t1, t1f_dir, "t1f")
            t2f_tiles, t2f_status = ensure_s2_tile_set(s2_t2, t2f_dir, "t2f")
            t1f_by_key = {tile_key(path): path for path in t1f_tiles}
            t2f_by_key = {tile_key(path): path for path in t2f_tiles}
            common_tile_keys = sorted(set(t1f_by_key) & set(t2f_by_key))
            if not common_tile_keys:
                raise RuntimeError(f"No common S2 tile blocks for {group_id}")
            print(
                f"S2 tile blocks: t1={len(t1f_by_key)}, t2={len(t2f_by_key)}, "
                f"common={len(common_tile_keys)}"
            )

            group_original = 0
            group_written = 0
            group_skipped = []
            generated_s3_tiles = 0
            reused_s3_tiles = 0
            total_subtiles = len(common_tile_keys) * SUBSAMPLE_GRID * SUBSAMPLE_GRID
            sub_tile_index = 0
            for block_index, block_key in enumerate(common_tile_keys, start=1):
                t1f = t1f_by_key[block_key]
                t2f = t2f_by_key[block_key]
                t1c = t1c_dir / f"{block_key}.tif"
                t2c = t2c_dir / f"{block_key}.tif"
                print(f"  block {block_index}/{len(common_tile_keys)}: {block_key}")
                for status in (
                    ensure_s3_tile(s3_t1, t1f, t1c, f"t1c {block_key}"),
                    ensure_s3_tile(s3_t2, t2f, t2c, f"t2c {block_key}"),
                ):
                    if status == "generated":
                        generated_s3_tiles += 1
                    else:
                        reused_s3_tiles += 1
                paths = [t1c, t1f, t2c, t2f]
                for sub_i in range(SUBSAMPLE_GRID):
                    for sub_j in range(SUBSAMPLE_GRID):
                        sub_tile_index += 1
                        original_total += 1
                        group_original += 1
                        print(
                            f"    subtile {sub_tile_index}/{total_subtiles} "
                            f"block={block_key} sub_index=[{sub_i},{sub_j}]"
                        )
                        arrays, reason = read_arrays(paths, group_id, sub_i, sub_j)
                        if arrays is None:
                            skipped_item = {
                                "group_id": group_id,
                                "block": block_key,
                                "sub_index": [sub_i, sub_j],
                                "reason": reason,
                            }
                            skipped.append(skipped_item)
                            group_skipped.append(skipped_item)
                            print(f"skip {group_id}/{block_key}[{sub_i},{sub_j}]: {reason}")
                            continue
                        sample_data = {
                            "arrays": pack_arrays(arrays),
                            "metadata": {
                                "tile": f"{group_id}_{block_key}",
                                "group_id": group_id,
                                "mgrs_tile": mgrs_tile,
                                "pair_dates": pair_dates,
                                "block": block_key,
                                "sub_index": [sub_i, sub_j],
                                "location": group["location"],
                                "interval_months": group["interval_months"],
                                "target_interval_months": group.get("target_interval_months"),
                                "target_start_year": group.get("target_start_year"),
                                "t1_date": group["t1"]["date"],
                                "t2_date": group["t2"]["date"],
                                "source_products": source_product_metadata(group),
                            },
                        }
                        txn.put(f"{sample_count:08d}".encode(), pickle.dumps(sample_data))
                        sample_count += 1
                        group_written += 1
                        print(f"    wrote subtile {sub_tile_index}/{total_subtiles} -> sample {sample_count}")
                        if sample_count % LMDB_BATCH_SIZE == 0:
                            txn.commit()
                            txn = env.begin(write=True)
            group_result = {
                "group_id": group_id,
                "status": "processed",
                "mgrs_tile": mgrs_tile,
                "pair_dates": pair_dates,
                "s2_tile_blocks": len(common_tile_keys),
                "s2_tile_status": {"t1f": t1f_status, "t2f": t2f_status},
                "generated_s3_tile_files": generated_s3_tiles,
                "reused_s3_tile_files": reused_s3_tiles,
                "subtiles_total": group_original,
                "samples_written": group_written,
                "subtiles_skipped": len(group_skipped),
                "skipped": group_skipped,
            }
            group_results.append(group_result)
            print(
                f"result {group_id}: samples_written={group_written}/{group_original}, "
                f"skipped={len(group_skipped)}, cumulative_samples={sample_count}"
            )
        except Exception as exc:
            skipped_item = {"group_id": group_id, "reason": str(exc)}
            skipped.append(skipped_item)
            group_results.append({"group_id": group_id, "status": "error", "reason": str(exc)})
            print(f"error {group_id}: {exc}")
            if STOP_ON_PROCESS_ERROR:
                raise

    txn.commit()
    write_lmdb_metadata(env, sample_count, original_total, skipped, map_size)
    env.sync()
    env.close()

    report_path = lmdb_path / "build_report.json"
    report_path.write_text(
        json.dumps(
            {
                "samples": sample_count,
                "original_groups": original_total,
                "manifest_groups": len(manifest["groups"]),
                "processed_groups": len(groups),
                "subsample_grid": [SUBSAMPLE_GRID, SUBSAMPLE_GRID],
                "map_size": map_size,
                "test_mode": TEST_MODE,
                "max_groups_to_process": MAX_GROUPS_TO_PROCESS,
                "process_full_mgrs_tile": PROCESS_FULL_MGRS_TILE,
                "filter_invalid_groups_before_processing": FILTER_INVALID_GROUPS_BEFORE_PROCESSING,
                "estimated_blocks_per_group": ESTIMATED_BLOCKS_PER_GROUP,
                "group_results": group_results,
                "skipped": skipped,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"LMDB written: {lmdb_path}")
    print(f"Build report written: {report_path}")


if __name__ == "__main__":
    main()
