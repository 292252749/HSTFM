"""Build 512-tile LMDB directly from local S2/S3 zip pairs."""

import csv
import hashlib
import json
import pickle
from collections import Counter

import lmdb

import s23_common as C

cfg = C.cfg

SAMPLE_INDEX_FIELDS = [
    "sample_id", "lmdb_key", "source_sample_uid", "pair_id", "mgrs_tile",
    "t1_date", "t2_date", "interval_days", "patch_id", "patch_row", "patch_col",
    "t1_s2", "t2_s2", "t1_s3", "t2_s3", "t1_s3_score", "t2_s3_score",
    "t1_s3_overlap_ratio", "t2_s3_overlap_ratio", "bounds_wgs84_json",
    "land_fraction", "water_fraction", "nodata_fraction", "landcover_major",
    "landcover_counts_json", "landcover_tiles_json",
    "s2_change_mean", "s3_change_mean", "s2_ndvi_t1_mean", "s2_ndvi_t2_mean",
    "s3_ndvi_t1_mean", "s3_ndvi_t2_mean",
]

def load_state():
    return C.read_json(cfg.BUILD_STATE_JSON, {"sample_count": 0, "completed_pairs": [], "active_pair_id": "", "active_completed_samples": []})


def save_state(state):
    C.write_json(cfg.BUILD_STATE_JSON, state)


def write_lmdb_metadata(env, summary):
    with env.begin(write=True) as txn:
        txn.put(
            b"__metadata__",
            pickle.dumps(
                {
                    "schema": "s23_lmdb_512_direct_zip_v1",
                    "dataset_group": cfg.DATASET_GROUP,
                    "dataset_name": cfg.DATASET_NAME,
                    "source_dataset": cfg.SOURCE_DATASET,
                    "total_samples": summary["samples"],
                    "sample_structure": "Each sample has 4 numpy arrays: [LR_t1, HR_t1, LR_t2, HR_t2]",
                    "hr_shape": [4, cfg.PATCH_HR_SIZE, cfg.PATCH_HR_SIZE],
                    "lr_shape": [4, cfg.PATCH_LR_SIZE, cfg.PATCH_LR_SIZE],
                    "dtype": "uint16",
                    "bad_values": {"s2": cfg.S2_BAD_VALUE, "s3": cfg.S3_BAD_VALUE},
                    "precision_note": "Arrays are stored as uint16 with np.save inside blosc lz4; LMDB writing does not quantize values.",
                    "s3_resample_mode": "aligned_valid_grid",
                    "summary": summary,
                }
            ),
        )


def put_sample(txn, key, arrays, metadata):
    txn.put(key.encode("utf-8"), pickle.dumps({"arrays": C.pack_arrays(arrays), "metadata": metadata}))


def build_one_sample(pair, patch_key, row, col, window, landcover_tiles, ocean_lookup, swath_cache):
    t1_s2 = pair["t1"]["s2"]["path"]
    t2_s2 = pair["t2"]["s2"]["path"]
    s2_crs, s2_bounds, wgs84 = C.s2_patch_ref(t1_s2, window)
    if ocean_lookup(wgs84):
        return None, "ocean"
    lc = C.landcover_stats(landcover_tiles, s2_crs, s2_bounds)
    ok_land, reason = C.passes_land_filter(lc)
    if not ok_land:
        return None, reason

    hr_t1 = C.read_s2_patch(t1_s2, window)
    if hr_t1 is None:
        return None, "bad_s2_t1"
    hr_t2 = C.read_s2_patch(t2_s2, window)
    if hr_t2 is None:
        return None, "bad_s2_t2"

    lr_t1 = C.choose_s3_patch(pair["t1"]["candidate_s3"], s2_crs, s2_bounds, swath_cache)
    if lr_t1 is None:
        return None, "no_valid_s3_t1"
    lr_t2 = C.choose_s3_patch(pair["t2"]["candidate_s3"], s2_crs, s2_bounds, swath_cache)
    if lr_t2 is None:
        return None, "no_valid_s3_t2"

    arrays = [lr_t1["array"], hr_t1, lr_t2["array"], hr_t2]
    metadata = {
        "schema": "s23_sample_512_direct_zip_v1",
        "pair_id": pair["pair_id"],
        "mgrs_tile": pair["mgrs_tile"],
        "t1_date": pair["t1_date"],
        "t2_date": pair["t2_date"],
        "interval_days": pair["interval_days"],
        "patch_id": patch_key,
        "patch_row": row,
        "patch_col": col,
        "t1_s2": pair["t1"]["s2"]["name"],
        "t2_s2": pair["t2"]["s2"]["name"],
        "t1_s3": lr_t1["name"],
        "t2_s3": lr_t2["name"],
        "t1_s3_score": lr_t1["score"],
        "t2_s3_score": lr_t2["score"],
        "t1_s3_overlap_ratio": lr_t1.get("overlap_ratio", ""),
        "t2_s3_overlap_ratio": lr_t2.get("overlap_ratio", ""),
        "bounds_wgs84_json": C.json.dumps([float(v) for v in wgs84]),
    }
    metadata.update(lc)
    metadata.update(C.sample_metrics(arrays))
    return (arrays, metadata), "ok"


def patch_bounds_from_grid(grid_ref, row, col):
    left, bottom, right, top = [float(v) for v in grid_ref["bounds"]]
    px = (right - left) / float(grid_ref["width"])
    py = (top - bottom) / float(grid_ref["height"])
    patch_left = left + col * cfg.PATCH_HR_SIZE * px
    patch_right = patch_left + cfg.PATCH_HR_SIZE * px
    patch_top = top - row * cfg.PATCH_HR_SIZE * py
    patch_bottom = patch_top - cfg.PATCH_HR_SIZE * py
    return (patch_left, patch_bottom, patch_right, patch_top)


def slice_hr_patch(arr, row, col):
    y0 = row * cfg.PATCH_HR_SIZE
    x0 = col * cfg.PATCH_HR_SIZE
    return arr[:, y0:y0 + cfg.PATCH_HR_SIZE, x0:x0 + cfg.PATCH_HR_SIZE]


def slice_lr_patch(arr, row, col):
    y0 = row * cfg.PATCH_LR_SIZE
    x0 = col * cfg.PATCH_LR_SIZE
    return arr[:, y0:y0 + cfg.PATCH_LR_SIZE, x0:x0 + cfg.PATCH_LR_SIZE]


def build_pair_context(pair, landcover_tiles, swath_cache):
    t1_s2 = pair["t1"]["s2"]["path"]
    t2_s2 = pair["t2"]["s2"]["path"]
    grid_ref = C.s2_valid_grid_ref(t1_s2)
    if grid_ref["rows"] <= 0 or grid_ref["cols"] <= 0:
        raise RuntimeError("no_complete_s2_patch_grid")
    hr_t1_full = C.read_s2_valid_grid(t1_s2, grid_ref)
    hr_t2_full = C.read_s2_valid_grid(t2_s2, grid_ref)
    if hr_t1_full.shape != hr_t2_full.shape:
        raise RuntimeError(f"s2_shape_mismatch:{hr_t1_full.shape}!={hr_t2_full.shape}")
    lr_width = grid_ref["cols"] * cfg.PATCH_LR_SIZE
    lr_height = grid_ref["rows"] * cfg.PATCH_LR_SIZE
    t1_s3_grids = C.choose_s3_grid(pair["t1"]["candidate_s3"], grid_ref["crs"], grid_ref["bounds"], lr_width, lr_height, swath_cache)
    t2_s3_grids = C.choose_s3_grid(pair["t2"]["candidate_s3"], grid_ref["crs"], grid_ref["bounds"], lr_width, lr_height, swath_cache)
    if not t1_s3_grids:
        raise RuntimeError("no_valid_s3_grid_t1")
    if not t2_s3_grids:
        raise RuntimeError("no_valid_s3_grid_t2")
    return {
        "grid_ref": grid_ref,
        "hr_t1_full": hr_t1_full,
        "hr_t2_full": hr_t2_full,
        "t1_s3_grids": t1_s3_grids,
        "t2_s3_grids": t2_s3_grids,
        "landcover_tiles": landcover_tiles,
    }


def choose_s3_patch_from_grids(grids, row, col):
    best = None
    for item in grids:
        arr = slice_lr_patch(item["array"], row, col)
        if C.has_bad_value(arr, cfg.S3_BAD_VALUE):
            continue
        rank_score = (1.0, float(item.get("overlap_ratio", 0.0)))
        if best is None or rank_score > best["rank_score"]:
            best = {"array": arr, "name": item["name"], "path": item["path"], "score": 1.0, "overlap_ratio": item.get("overlap_ratio", ""), "rank_score": rank_score}
    return best


def build_one_sample_from_context(pair, context, patch_key, row, col, ocean_lookup):
    grid_ref = context["grid_ref"]
    s2_bounds = patch_bounds_from_grid(grid_ref, row, col)
    wgs84 = C.transform_bounds(grid_ref["crs"], "EPSG:4326", *s2_bounds, densify_pts=21)
    if ocean_lookup(wgs84):
        return None, "ocean"
    lc = C.landcover_stats(context["landcover_tiles"], grid_ref["crs"], s2_bounds)
    ok_land, reason = C.passes_land_filter(lc)
    if not ok_land:
        return None, reason

    hr_t1 = slice_hr_patch(context["hr_t1_full"], row, col)
    if C.has_bad_value(hr_t1, cfg.S2_BAD_VALUE):
        return None, "bad_s2_t1"
    hr_t2 = slice_hr_patch(context["hr_t2_full"], row, col)
    if C.has_bad_value(hr_t2, cfg.S2_BAD_VALUE):
        return None, "bad_s2_t2"

    lr_t1 = choose_s3_patch_from_grids(context["t1_s3_grids"], row, col)
    if lr_t1 is None:
        return None, "no_valid_s3_t1"
    lr_t2 = choose_s3_patch_from_grids(context["t2_s3_grids"], row, col)
    if lr_t2 is None:
        return None, "no_valid_s3_t2"

    arrays = [lr_t1["array"], hr_t1, lr_t2["array"], hr_t2]
    metadata = {
        "schema": "s23_sample_512_direct_zip_v1",
        "pair_id": pair["pair_id"],
        "mgrs_tile": pair["mgrs_tile"],
        "t1_date": pair["t1_date"],
        "t2_date": pair["t2_date"],
        "interval_days": pair["interval_days"],
        "patch_id": patch_key,
        "patch_row": row,
        "patch_col": col,
        "t1_s2": pair["t1"]["s2"]["name"],
        "t2_s2": pair["t2"]["s2"]["name"],
        "t1_s3": lr_t1["name"],
        "t2_s3": lr_t2["name"],
        "t1_s3_score": lr_t1["score"],
        "t2_s3_score": lr_t2["score"],
        "t1_s3_overlap_ratio": lr_t1.get("overlap_ratio", ""),
        "t2_s3_overlap_ratio": lr_t2.get("overlap_ratio", ""),
        "bounds_wgs84_json": C.json.dumps([float(v) for v in wgs84]),
        "s3_resample_mode": "aligned_valid_grid",
    }
    metadata.update(lc)
    metadata.update(C.sample_metrics(arrays))
    return (arrays, metadata), "ok"



def append_index_rows(rows):
    if not rows:
        return
    path = cfg.SAMPLE_INDEX_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    extra_fields = sorted({key for row in rows for key in row.keys()} - set(SAMPLE_INDEX_FIELDS))
    fieldnames = SAMPLE_INDEX_FIELDS + extra_fields
    with path.open("a", newline="", encoding="utf-8") as fd:
        writer = csv.DictWriter(fd, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key, ""), sort_keys=True) if isinstance(row.get(key, ""), (dict, list, tuple)) else row.get(key, "") for key in fieldnames})
    rows.clear()

def write_sample_result(sample_uid, result, env, txn, batch, pair_skipped, report, completed_samples, rows_buffer, state):
    built, reason = result
    completed_samples.add(sample_uid)
    if built is None:
        pair_skipped[reason] += 1
        report["skipped"].append({"sample_uid": sample_uid, "reason": reason})
        return txn, batch, 0
    arrays, metadata = built
    key = f"{state['sample_count']:08d}"
    metadata.update({"sample_id": f"{cfg.DATASET_NAME}_{key}", "lmdb_key": key, "source_sample_uid": sample_uid})
    put_sample(txn, key, arrays, metadata)
    rows_buffer.append(metadata)
    state["sample_count"] += 1
    batch += 1
    if batch >= cfg.LMDB_BATCH_SIZE:
        txn.commit()
        txn = env.begin(write=True)
        batch = 0
        state["active_completed_samples"] = sorted(completed_samples)
        save_state(state)
        append_index_rows(rows_buffer)
    return txn, batch, 1



def pair_index_hash(pairs):
    payload = [
        {
            "pair_id": pair.get("pair_id"),
            "t1_s2": pair.get("t1", {}).get("s2", {}).get("path"),
            "t2_s2": pair.get("t2", {}).get("s2", {}).get("path"),
            "t1_s3": [item.get("path") for item in pair.get("t1", {}).get("candidate_s3", [])],
            "t2_s3": [item.get("path") for item in pair.get("t2", {}).get("candidate_s3", [])],
        }
        for pair in pairs
    ]
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def validate_state_for_pairs(state, pairs):
    digest = pair_index_hash(pairs)
    old_digest = state.get("pair_index_hash")
    if old_digest and old_digest != digest:
        raise SystemExit(
            f"Existing build_state.json belongs to a different pair_index.json. "
            f"Delete {cfg.BUILD_STATE_JSON} and the LMDB output before rebuilding."
        )
    state["pair_index_hash"] = digest
    return state


def lmdb_sample_key_status(env):
    keys = []
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, _ in cursor:
            text = key.decode("utf-8", errors="ignore")
            if len(text) == 8 and text.isdigit():
                keys.append(int(text))
    if not keys:
        return {"count": 0, "next_sample_count": 0, "min_key": "", "max_key": ""}
    keys.sort()
    key_set = set(keys)
    missing = [idx for idx in range(keys[0], keys[-1] + 1) if idx not in key_set]
    return {
        "count": len(keys),
        "next_sample_count": keys[-1] + 1,
        "min_key": f"{keys[0]:08d}",
        "max_key": f"{keys[-1]:08d}",
        "missing_key_count": len(missing),
        "first_missing_keys": [f"{idx:08d}" for idx in missing[:10]],
    }


def lmdb_source_uids_for_pair(env, pair_id):
    if not pair_id:
        return []
    uids = []
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            text = key.decode("utf-8", errors="ignore")
            if len(text) != 8 or not text.isdigit():
                continue
            try:
                payload = pickle.loads(value)
            except Exception:
                continue
            metadata = payload.get("metadata", {})
            if metadata.get("pair_id") == pair_id and metadata.get("source_sample_uid"):
                uids.append(metadata["source_sample_uid"])
    return sorted(set(uids))

def reconcile_state_with_lmdb(state, env):
    status = lmdb_sample_key_status(env)
    state_count = int(state.get("sample_count", 0))
    next_count = int(status["next_sample_count"])
    if status["count"] > 0 and (status.get("min_key") != "00000000" or status.get("missing_key_count", 0) > 0):
        raise SystemExit(
            "LMDB sample keys are not contiguous from 00000000. "
            f"key range={status.get('min_key')}..{status.get('max_key')}, "
            f"missing_key_count={status.get('missing_key_count', 0)}, "
            f"first_missing_keys={status.get('first_missing_keys', [])}. "
            "Use a clean LMDB/state/index set before continuing."
        )
    state_has_progress = bool(state.get("completed_pairs") or state.get("active_pair_id") or state.get("active_completed_samples") or state_count)
    if status["count"] == 0 and state_has_progress:
        C.progress(
            "04 resume repair: LMDB has no committed sample keys but build_state has "
            f"sample_count={state_count}; resetting build_state progress to 0"
        )
        state["sample_count"] = 0
        state["completed_pairs"] = []
        state["active_pair_id"] = ""
        state["active_completed_samples"] = []
        save_state(state)
        return state, status
    if state_count == next_count:
        return state, status
    if state_count < next_count:
        active_pair_id = state.get("active_pair_id", "")
        C.progress(
            "04 resume repair: build_state is behind committed LMDB samples; "
            f"sample_count {state_count} -> {next_count}"
        )
        state["sample_count"] = next_count
        if active_pair_id:
            state["active_completed_samples"] = lmdb_source_uids_for_pair(env, active_pair_id)
        save_state(state)
        return state, status
    raise SystemExit(
        "LMDB/build_state resume mismatch. "
        f"build_state sample_count={state_count}, LMDB next_sample_count={next_count}, "
        f"LMDB key range={status.get('min_key')}..{status.get('max_key')}, "
        f"missing_key_count={status.get('missing_key_count', 0)}. "
        "This usually means the previous run stopped after build_state was saved but before LMDB commit. "
        "Start a clean rebuild, or repair manually after inspecting LMDB keys and sample_index.csv."
    )

def main():
    C.progress(f"04 start: dataset_group={cfg.DATASET_GROUP}")
    pair_index = C.read_json(cfg.PAIR_INDEX_JSON, {})
    pairs = pair_index.get("pairs", [])
    if cfg.MAX_PAIRS_TO_PROCESS is not None:
        pairs = pairs[: cfg.MAX_PAIRS_TO_PROCESS]
    if not pairs:
        raise SystemExit(f"No pairs found in {cfg.PAIR_INDEX_JSON}")

    candidate_samples = sum(int(pair.get("candidate_samples", 0)) for pair in pairs)
    map_size = C.estimate_map_size(candidate_samples)
    C.progress(f"04 plan: {len(pairs)} pairs, candidate_samples={candidate_samples}, map_size={map_size}")
    cfg.LMDB_OUT.mkdir(parents=True, exist_ok=True)
    C.progress("04 filters: loading FROMGLC tiles")
    landcover_tiles = C.scan_landcover_tiles()
    landcover_count = len(landcover_tiles.get("tiles", landcover_tiles)) if isinstance(landcover_tiles, dict) else len(landcover_tiles)
    C.progress(f"04 filters: loaded {landcover_count} FROMGLC tiles")
    C.progress("04 filters: loading ocean lookup")
    ocean_lookup = C.load_ocean_lookup()
    C.progress("04 filters: ocean lookup ready")
    state = validate_state_for_pairs(load_state(), pairs)
    rows_buffer = []
    report = C.read_json(cfg.BUILD_REPORT_JSON, {"pairs": [], "skipped": []})
    swath_cache = {}

    env = lmdb.open(str(cfg.LMDB_OUT), map_size=map_size)
    state, lmdb_status = reconcile_state_with_lmdb(state, env)
    C.progress(f"04 resume check: lmdb_samples={lmdb_status['count']}, next_sample_count={state['sample_count']}")
    completed_pairs = set(state.get("completed_pairs", []))
    txn = env.begin(write=True)
    batch = 0

    total_pairs = len(pairs)
    pair_step = C.progress_every(0, total_pairs)
    for pair_idx, pair in enumerate(pairs, 1):
        pair_id = pair["pair_id"]
        if pair_id in completed_pairs:
            C.progress(f"04 pair {pair_idx}/{total_pairs}: skip completed {pair_id}")
            continue
        C.progress(f"04 pair {pair_idx}/{total_pairs}: start {pair_id}")
        if state.get("active_pair_id") == pair_id:
            completed_samples = set(state.get("active_completed_samples", []))
        else:
            completed_samples = set()
            state["active_pair_id"] = pair_id
            state["active_completed_samples"] = []
            save_state(state)
        pair_written = 0
        pair_skipped = Counter()
        try:
            context = build_pair_context(pair, landcover_tiles, swath_cache)
            tasks = []
            for row in range(context["grid_ref"]["rows"]):
                for col in range(context["grid_ref"]["cols"]):
                    patch_key = f"tile_r{row:04d}_c{col:04d}"
                    sample_uid = f"{pair['pair_id']}:{patch_key}"
                    if sample_uid not in completed_samples:
                        tasks.append((sample_uid, patch_key, row, col))

            total_tasks = len(tasks)
            C.progress(
                f"04 pair {pair_idx}/{total_pairs}: {total_tasks} pending patches, "
                f"grid={context['grid_ref']['rows']}x{context['grid_ref']['cols']}, "
                f"s3_resample_mode=aligned_valid_grid"
            )
            sample_step = C.progress_every(0, total_tasks)
            for sample_idx, (sample_uid, patch_key, row, col) in enumerate(tasks, 1):
                result = build_one_sample_from_context(pair, context, patch_key, row, col, ocean_lookup)
                txn, batch, wrote = write_sample_result(sample_uid, result, env, txn, batch, pair_skipped, report, completed_samples, rows_buffer, state)
                pair_written += wrote
                if sample_idx == total_tasks or sample_idx % sample_step == 0:
                    C.progress(f"04 pair {pair_idx}/{total_pairs}: processed {sample_idx}/{total_tasks} patches, written={pair_written}, total_written={state['sample_count']}")
            if batch > 0:
                txn.commit()
                txn = env.begin(write=True)
                batch = 0
                state["active_completed_samples"] = sorted(completed_samples)
                save_state(state)
                append_index_rows(rows_buffer)
            completed_pairs.add(pair_id)
            state["active_pair_id"] = ""
            state["active_completed_samples"] = []
            report["pairs"].append({"pair_id": pair_id, "status": "complete", "samples_written": pair_written, "skipped": dict(pair_skipped)})
            C.progress(f"04 pair {pair_idx}/{total_pairs}: complete {pair_id}, written={pair_written}, skipped={dict(pair_skipped)}")
        except Exception as exc:
            try:
                txn.abort()
            except Exception:
                pass
            txn = env.begin(write=True)
            batch = 0
            rows_buffer.clear()
            report["pairs"].append({"pair_id": pair_id, "status": "error", "reason": str(exc)})
            C.progress(f"04 pair {pair_idx}/{total_pairs}: error {pair_id}: {exc}")
            if cfg.STOP_ON_ERROR:
                raise
        state["completed_pairs"] = sorted(completed_pairs)
        state["active_completed_samples"] = sorted(completed_samples) if state.get("active_pair_id") else []
        save_state(state)
        C.write_json(cfg.BUILD_REPORT_JSON, report)
        if pair_idx == total_pairs or pair_idx % pair_step == 0:
            C.progress(f"04 overall: processed {pair_idx}/{total_pairs} pairs, completed_pairs={len(completed_pairs)}, total_written={state['sample_count']}")

    txn.commit()
    summary = {
        "samples": state["sample_count"],
        "candidate_samples": candidate_samples,
        "completed_pairs": len(completed_pairs),
        "map_size_bytes": map_size,
        "skipped_samples": len(report.get("skipped", [])),
    }
    C.progress("04 finalizing: writing LMDB metadata")
    write_lmdb_metadata(env, summary)
    env.sync()
    env.close()
    append_index_rows(rows_buffer)
    C.write_json(cfg.METADATA_JSON, {"schema": "s23_sample_metadata_direct_zip_v1", "dataset_group": cfg.DATASET_GROUP, "dataset_name": cfg.DATASET_NAME, "summary": summary, "sample_index_csv": str(cfg.SAMPLE_INDEX_CSV)})
    report["summary"] = summary
    C.write_json(cfg.BUILD_REPORT_JSON, report)
    C.progress("04 finish: LMDB build complete")
    print(f"LMDB written: {cfg.LMDB_OUT}")
    print(summary)


if __name__ == "__main__":
    main()





















