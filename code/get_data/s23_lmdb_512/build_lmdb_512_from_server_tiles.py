"""
Build a 512-sample LMDB directly from server_S23-style processed tile folders.

Use this for domestic data that already has processed folders like:
    <group>/t1c/*.tif
    <group>/t1f/*.tif
    <group>/t2c/*.tif
    <group>/t2f/*.tif

This path does not require a CDSE manifest. It follows the successful server
LMDB logic: each 1024/128 tile is split into 2x2 samples of 512/64, with the
same array order [LR_t1, HR_t1, LR_t2, HR_t2]. It adds FROMGLC and local ocean
shapefile filtering/metadata before writing samples.
"""

import csv
import io
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds, bounds as window_bounds
from rasterio.warp import transform_bounds

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import build_lmdb_512 as common


# =========================
# CONFIG: edit values here.
# =========================
DATASET_NAME = "china_2024_server_tiles"
SOURCE_DATASET = "china_server_tiles"
SERVER_TILE_ROOT = Path("server_S23_out/tiles")
SERVER_TILE_INDEX_JSON = Path("s23_lmdb_512_out/server_tile_index.json")
USE_SERVER_TILE_INDEX = True

OUT_ROOT = Path("s23_lmdb_512_out/china_server_tiles_v2_512")
LMDB_PATH = OUT_ROOT / "lmdb_512"
SAMPLE_INDEX_CSV = OUT_ROOT / "sample_index.csv"
BUILD_REPORT_JSON = OUT_ROOT / "build_report.json"

PATCH_HR_SIZE = 512
PATCH_LR_SIZE = 64
LR_FACTOR = 8
LMDB_BATCH_SIZE = 500
LMDB_MAP_SIZE = None
LMDB_BYTES_PER_SAMPLE_ESTIMATE = 6 * 1024**2
LMDB_MAP_SIZE_MARGIN = 2.0
LMDB_MIN_MAP_SIZE = 8 * 1024**3
MAX_GROUPS_TO_PROCESS = None
STOP_ON_PROCESS_ERROR = False

S2_BAD_VALUE = common.S2_BAD_VALUE
S3_BAD_VALUE = common.S3_BAD_VALUE


def pack_arrays(arrays):
    return common.pack_arrays(arrays)


def has_bad_value(arr, bad_value):
    return common.has_bad_value(arr, bad_value)


def scan_server_tile_groups(root):
    root = Path(root)
    groups = []
    if not root.exists():
        raise SystemExit(f"SERVER_TILE_ROOT does not exist: {root}")
    candidates = [root] + [path for path in root.rglob("*") if path.is_dir()]
    for folder in sorted(candidates):
        subdirs = {name: folder / name for name in ("t1c", "t1f", "t2c", "t2f")}
        if not all(path.is_dir() for path in subdirs.values()):
            continue
        common_names = None
        for subdir in subdirs.values():
            current = {path.name for path in subdir.glob("*.tif")}
            common_names = current if common_names is None else common_names & current
        if common_names:
            groups.append(
                {
                    "group_id": folder.name,
                    "base_path": str(folder),
                    "tile_names": sorted(common_names),
                    "tile_count": len(common_names),
                }
            )
    return groups


def load_groups():
    if USE_SERVER_TILE_INDEX and SERVER_TILE_INDEX_JSON.exists():
        data = json.loads(SERVER_TILE_INDEX_JSON.read_text(encoding="utf-8"))
        groups = data.get("groups", [])
    else:
        groups = scan_server_tile_groups(SERVER_TILE_ROOT)
    if MAX_GROUPS_TO_PROCESS is not None:
        groups = groups[:MAX_GROUPS_TO_PROCESS]
    return groups


def estimate_lmdb_map_size(candidate_samples):
    if LMDB_MAP_SIZE is not None:
        return LMDB_MAP_SIZE
    estimated_bytes = int(max(1, candidate_samples) * LMDB_BYTES_PER_SAMPLE_ESTIMATE * LMDB_MAP_SIZE_MARGIN)
    return max(LMDB_MIN_MAP_SIZE, estimated_bytes)


def read_window(path, window, bad_value):
    with rasterio.open(path) as src:
        arr = src.read(window=window).astype(np.uint16)
    if has_bad_value(arr, bad_value):
        return None
    return arr


def subpatch_windows(t1f_path, t1c_path):
    with rasterio.open(t1f_path) as hr, rasterio.open(t1c_path) as lr:
        rows = min(hr.height // PATCH_HR_SIZE, lr.height // PATCH_LR_SIZE)
        cols = min(hr.width // PATCH_HR_SIZE, lr.width // PATCH_LR_SIZE)
    for row in range(rows):
        for col in range(cols):
            yield row, col, Window(col * PATCH_HR_SIZE, row * PATCH_HR_SIZE, PATCH_HR_SIZE, PATCH_HR_SIZE), Window(
                col * PATCH_LR_SIZE, row * PATCH_LR_SIZE, PATCH_LR_SIZE, PATCH_LR_SIZE
            )


def bounds_stats_for_window(path, window, ocean_lookup):
    with rasterio.open(path) as src:
        src_bounds = window_bounds(window, src.transform)
        src_crs = src.crs
        bounds_wgs84 = transform_bounds(src_crs, "EPSG:4326", *src_bounds, densify_pts=21)
    return {
        "ocean_intersects": bool(ocean_lookup(bounds_wgs84)),
        "ocean_filter_source": str(common.OCEAN_SHP_PATH),
        "bounds_wgs84_json": json.dumps([float(value) for value in bounds_wgs84]),
    }, src_crs, src_bounds


def landcover_stats_for_bounds(landcover, src_crs, src_bounds):
    arrays = []
    used_tiles = []
    for tile in landcover["tiles"]:
        bounds = transform_bounds(src_crs, tile["crs"], *src_bounds, densify_pts=21)
        if not common._bounds_overlap(bounds, tile["bounds"]):
            continue
        try:
            with rasterio.open(tile["path"]) as ds:
                window = from_bounds(*bounds, transform=ds.transform)
                arr = ds.read(
                    1,
                    window=window,
                    out_shape=common.FROMGLC_RESAMPLE_SHAPE,
                    boundless=True,
                    fill_value=0,
                    resampling=Resampling.nearest,
                )
                arrays.append(arr)
                used_tiles.append(str(tile["path"]))
        except Exception as exc:
            print(f"Warning: FROMGLC read failed for {tile['path']}: {exc}")
    if not arrays:
        return common._empty_landcover_stats("landcover_unavailable")
    merged = np.concatenate([arr.reshape(-1) for arr in arrays]).reshape(1, -1)
    stats = common._landcover_stats_from_array(
        merged,
        common.FROMGLC_CLASSES,
        common.FROMGLC_WATER_CLASSES,
        common.FROMGLC_NODATA_VALUES,
        str(common.FROMGLC_DIR),
    )
    stats["landcover_tiles_json"] = json.dumps(used_tiles, sort_keys=True)
    return stats


def sample_metrics(arrays):
    return common.sample_metrics(arrays)


def csv_safe(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def write_index_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as fd:
        writer = csv.DictWriter(fd, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_safe(row.get(key, "")) for key in fieldnames})


def write_metadata(env, total_samples, original_candidates, skipped, map_size):
    with env.begin(write=True) as txn:
        txn.put(
            b"__metadata__",
            pickle.dumps(
                {
                    "schema": "s23_lmdb_512_server_tiles_v1",
                    "dataset_name": DATASET_NAME,
                    "source_dataset": SOURCE_DATASET,
                    "total_samples": total_samples,
                    "original_candidate_samples": original_candidates,
                    "skipped_samples": len(skipped),
                    "hr_shape": [4, PATCH_HR_SIZE, PATCH_HR_SIZE],
                    "lr_shape": [4, PATCH_LR_SIZE, PATCH_LR_SIZE],
                    "sample_structure": "Each sample has 4 numpy arrays: [LR_t1, HR_t1, LR_t2, HR_t2]",
                    "bad_values": {"s2": S2_BAD_VALUE, "s3": S3_BAD_VALUE},
                    "lr_factor": LR_FACTOR,
                    "map_size": map_size,
                    "fromglc_dir": str(common.FROMGLC_DIR),
                    "ocean_shapefile": str(common.OCEAN_SHP_PATH),
                }
            ),
        )


def main():
    import lmdb

    groups = load_groups()
    if not groups:
        raise SystemExit("No server-style tile groups found. Check SERVER_TILE_ROOT or SERVER_TILE_INDEX_JSON.")

    candidate_samples = sum(group.get("tile_count", len(group.get("tile_names", []))) * 4 for group in groups)
    map_size = estimate_lmdb_map_size(candidate_samples)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LMDB_PATH.mkdir(parents=True, exist_ok=True)
    landcover = common.open_landcover()
    ocean_lookup = common.load_ocean_lookup()

    print(f"Building server-tile 512 LMDB at {LMDB_PATH}")
    print(f"Groups={len(groups)} candidate_samples~={candidate_samples} map_size={map_size / 1024**3:.1f} GiB")

    env = lmdb.open(str(LMDB_PATH), map_size=map_size)
    txn = env.begin(write=True)
    sample_count = 0
    original_candidates = 0
    skipped = []
    index_rows = []
    group_results = []

    for group_idx, group in enumerate(groups, start=1):
        group_id = group["group_id"]
        base = Path(group["base_path"])
        tile_names = group.get("tile_names") or sorted({path.name for path in (base / "t1c").glob("*.tif")})
        group_written = 0
        group_skipped = Counter()
        print(f"\n=== Group {group_idx}/{len(groups)}: {group_id} tiles={len(tile_names)} ===")
        try:
            for tile_name in tile_names:
                paths = {
                    "t1c": base / "t1c" / tile_name,
                    "t1f": base / "t1f" / tile_name,
                    "t2c": base / "t2c" / tile_name,
                    "t2f": base / "t2f" / tile_name,
                }
                if not all(path.exists() for path in paths.values()):
                    skipped.append({"group_id": group_id, "tile": tile_name, "reason": "missing_tile_member"})
                    group_skipped["missing_tile_member"] += 1
                    continue
                for sub_row, sub_col, hr_window, lr_window in subpatch_windows(paths["t1f"], paths["t1c"]):
                    original_candidates += 1
                    patch_id = f"{Path(tile_name).stem}_r{sub_row}_c{sub_col}"
                    sea_stats, src_crs, src_bounds = bounds_stats_for_window(paths["t1f"], hr_window, ocean_lookup)
                    if common.SKIP_IF_OCEAN_INTERSECTS and sea_stats["ocean_intersects"]:
                        skipped.append({"group_id": group_id, "patch_id": patch_id, "reason": "ocean_intersects"})
                        group_skipped["ocean_intersects"] += 1
                        continue

                    lc_stats = landcover_stats_for_bounds(landcover, src_crs, src_bounds)
                    ok_land, land_reason = common.passes_land_filter(lc_stats)
                    if not ok_land:
                        skipped.append({"group_id": group_id, "patch_id": patch_id, "reason": land_reason})
                        group_skipped[land_reason] += 1
                        continue

                    arrays = [
                        read_window(paths["t1c"], lr_window, S3_BAD_VALUE),
                        read_window(paths["t1f"], hr_window, S2_BAD_VALUE),
                        read_window(paths["t2c"], lr_window, S3_BAD_VALUE),
                        read_window(paths["t2f"], hr_window, S2_BAD_VALUE),
                    ]
                    if any(arr is None for arr in arrays):
                        skipped.append({"group_id": group_id, "patch_id": patch_id, "reason": "bad_value"})
                        group_skipped["bad_value"] += 1
                        continue

                    lmdb_key = f"{sample_count:08d}"
                    sample_id = f"{DATASET_NAME}_{lmdb_key}"
                    metadata = {
                        "schema": "s23_sample_512_server_tiles_v1",
                        "sample_id": sample_id,
                        "lmdb_key": lmdb_key,
                        "dataset_name": DATASET_NAME,
                        "source_dataset": SOURCE_DATASET,
                        "group_id": group_id,
                        "base_path": str(base),
                        "tile": tile_name,
                        "patch_id": patch_id,
                        "sub_row": sub_row,
                        "sub_col": sub_col,
                    }
                    metadata.update(sea_stats)
                    metadata.update(lc_stats)
                    metadata.update(sample_metrics(arrays))
                    txn.put(lmdb_key.encode(), pickle.dumps({"arrays": pack_arrays(arrays), "metadata": metadata}))
                    index_rows.append(dict(metadata))
                    sample_count += 1
                    group_written += 1
                    if sample_count % LMDB_BATCH_SIZE == 0:
                        txn.commit()
                        txn = env.begin(write=True)
                        write_index_csv(index_rows, SAMPLE_INDEX_CSV)
                        print(f"  wrote sample {sample_count}")
            group_results.append({"group_id": group_id, "status": "processed", "samples_written": group_written, "skipped": dict(group_skipped)})
            print(f"result {group_id}: written={group_written}, skipped={dict(group_skipped)}")
        except Exception as exc:
            skipped.append({"group_id": group_id, "reason": str(exc)})
            group_results.append({"group_id": group_id, "status": "error", "reason": str(exc)})
            print(f"error {group_id}: {exc}")
            if STOP_ON_PROCESS_ERROR:
                raise

    txn.commit()
    write_metadata(env, sample_count, original_candidates, skipped, map_size)
    env.sync()
    env.close()
    write_index_csv(index_rows, SAMPLE_INDEX_CSV)

    report = {
        "schema": "s23_lmdb_512_server_tiles_build_report_v1",
        "dataset_name": DATASET_NAME,
        "source_dataset": SOURCE_DATASET,
        "server_tile_root": str(SERVER_TILE_ROOT),
        "server_tile_index_json": str(SERVER_TILE_INDEX_JSON),
        "lmdb_path": str(LMDB_PATH),
        "sample_index_csv": str(SAMPLE_INDEX_CSV),
        "samples": sample_count,
        "original_candidate_samples": original_candidates,
        "skipped_samples": len(skipped),
        "group_results": group_results,
        "skipped": skipped,
    }
    BUILD_REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"LMDB written: {LMDB_PATH}")
    print(f"Sample index written: {SAMPLE_INDEX_CSV}")
    print(f"Build report written: {BUILD_REPORT_JSON}")


if __name__ == "__main__":
    main()
