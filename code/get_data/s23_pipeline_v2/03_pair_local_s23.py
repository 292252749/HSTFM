"""Fast server-style local S2/S3 pairing without using a manifest."""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pickle

import s23_common as C

cfg = C.cfg


def cloud_value(item):
    value = item.get("cloud", "")
    if value == "" or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def passes_cloud_filter(item):
    cloud = cloud_value(item)
    return cloud is not None and cloud <= cfg.MAX_S2_CLOUD_PERCENT


def date_parts(date_text):
    if not date_text:
        return {"year": "", "month": ""}
    return {"year": date_text[:4], "month": date_text[4:6]}


def merged_bbox(bounds_list):
    bounds = [b for b in bounds_list if b]
    if not bounds:
        return []
    return [min(b[0] for b in bounds), min(b[1] for b in bounds), max(b[2] for b in bounds), max(b[3] for b in bounds)]


def spatial_split_config():
    bbox = getattr(cfg, "PAIR_SPATIAL_SPLIT_BBOX_WGS84", None)
    mode = str(getattr(cfg, "PAIR_SPATIAL_SPLIT_MODE", "none") or "none").lower()
    if not bbox or mode not in {"inside", "outside"}:
        return [], "none"
    return [float(v) for v in bbox], mode


def passes_dataset_bbox_filter(item):
    bbox, mode = spatial_split_config()
    if not bbox or mode == "none":
        return True
    bounds = item.get("bounds_wgs84")
    intersects = C.bbox_intersects(bounds, bbox)
    return intersects if mode == "inside" else not intersects


def filter_by_dataset_bbox(items, label):
    bbox, mode = spatial_split_config()
    valid = [item for item in items if item.get("bounds_wgs84")]
    if not bbox or mode == "none":
        return valid, {"label": label, "mode": mode, "bbox_wgs84": bbox, "before": len(items), "after_valid_bounds": len(valid), "after_spatial_filter": len(valid), "removed_by_spatial_filter": 0}
    kept = [item for item in valid if passes_dataset_bbox_filter(item)]
    removed = len(valid) - len(kept)
    C.progress(f"03 spatial split {label}: mode={mode}, bbox={bbox}, valid={len(valid)}, kept={len(kept)}, removed={removed}")
    return kept, {"label": label, "mode": mode, "bbox_wgs84": bbox, "before": len(items), "after_valid_bounds": len(valid), "after_spatial_filter": len(kept), "removed_by_spatial_filter": removed}

def read_cloud_one(item):
    item = dict(item)
    item["cloud"] = C.read_s2_cloud(item["path"])
    return item


def read_s2_cloud_parallel(items):
    total = len(items)
    C.progress(f"03 S2 cloud: reading cloud metadata for {total} S2 products")
    if not items:
        return []
    workers = max(1, int(getattr(cfg, "PAIR_METADATA_WORKERS", 1)))
    step = C.progress_every(0, total)
    if workers == 1:
        out = []
        for idx, item in enumerate(items, 1):
            out.append(read_cloud_one(item))
            if idx == total or idx % step == 0:
                C.progress(f"03 S2 cloud: processed {idx}/{total}")
        return out
    out = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(read_cloud_one, item) for item in items]
        for idx, future in enumerate(as_completed(futures), 1):
            out.append(future.result())
            if idx == total or idx % step == 0:
                C.progress(f"03 S2 cloud: processed {idx}/{total}")
    return sorted(out, key=lambda item: item["path"])


def with_s2_bounds(item):
    item = dict(item)
    try:
        item["bounds_wgs84"] = C.s2_bounds_wgs84(item["path"])
        item["candidate_samples"] = int(getattr(cfg, "S2_CANDIDATE_SAMPLES_ESTIMATE", 100))
    except Exception as exc:
        item["bounds_wgs84"] = []
        item["candidate_samples"] = 0
        item["bounds_error"] = str(exc)
    return item


def _append_path_candidates(out, value):
    if value in (None, ""):
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _append_path_candidates(out, item)
        return
    out.append(Path(value))


def configured_s3_geo_pickle_paths():
    paths = []
    _append_path_candidates(paths, getattr(cfg, "S3_GEO_PICKLE_PATHS", []))
    _append_path_candidates(paths, getattr(cfg, "S3_GEO_PICKLE_PATH", None))
    seen = set()
    out = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def load_s3_geo_cache():
    for path in configured_s3_geo_pickle_paths():
        if not path.exists():
            continue
        C.progress(f"03 S3 geo cache: loading {path}")
        with path.open("rb") as fd:
            records = pickle.load(fd)
        cache = {}
        failed = 0
        for record in records:
            try:
                s3_path, geom = record[0], record[1]
                bounds = C.bounds_from_geometry_like(geom)
                if bounds:
                    cache[Path(str(s3_path)).name] = tuple(float(v) for v in bounds)
                else:
                    failed += 1
            except Exception:
                failed += 1
        C.progress(f"03 S3 geo cache: loaded {len(cache)} footprints from {path}, failed={failed}")
        return cache, str(path)
    C.progress("03 S3 geo cache: no configured pickle found, fallback to zip geo_coordinates.nc")
    return {}, ""


def with_s3_bounds(item, s3_geo_cache=None):
    item = dict(item)
    cache = s3_geo_cache or {}
    try:
        cached = cache.get(Path(item["path"]).name) or cache.get(item.get("name", ""))
        if cached:
            item["bounds_wgs84"] = cached
            item["bounds_source"] = "server_s3_geo_pickle"
        else:
            item["bounds_wgs84"] = C.s3_bounds_wgs84(item["path"])
            item["bounds_source"] = "zip_geo_coordinates"
    except Exception as exc:
        item["bounds_wgs84"] = []
        item["bounds_error"] = str(exc)
    return item


def map_parallel(items, fn, label):
    total = len(items)
    C.progress(f"{label}: start {total}")
    if not items:
        return []
    workers = max(1, int(getattr(cfg, "PAIR_METADATA_WORKERS", 1)))
    step = C.progress_every(0, total)
    if workers == 1:
        out = []
        for idx, item in enumerate(items, 1):
            out.append(fn(item))
            if idx == total or idx % step == 0:
                C.progress(f"{label}: processed {idx}/{total}")
        return out
    out = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fn, item) for item in items]
        for idx, future in enumerate(as_completed(futures), 1):
            out.append(future.result())
            if idx == total or idx % step == 0:
                C.progress(f"{label}: processed {idx}/{total}")
    return sorted(out, key=lambda item: item["path"])


def best_s3_candidates_for_s2(s2, s3_by_date):
    candidates = []
    for s3 in sorted(s3_by_date.get(s2["date"], []), key=lambda x: x["name"]):
        s3_item = dict(s3)
        s3_item["s2_overlap_ratio"] = C.bbox_overlap_ratio(s2.get("bounds_wgs84"), s3.get("bounds_wgs84"))
        s3_item["intersects_s2_bbox"] = s3_item["s2_overlap_ratio"] > 0.0
        s3_item["contains_s2_bbox"] = C.bbox_contains(s3.get("bounds_wgs84"), s2.get("bounds_wgs84"))
        if getattr(cfg, "REQUIRE_S3_BBOX_CONTAINS_S2", False) and not s3_item["contains_s2_bbox"]:
            continue
        candidates.append(s3_item)
    return sorted(candidates, key=lambda x: (-float(x["s2_overlap_ratio"]), x["name"]))[: cfg.MAX_S3_CANDIDATES_PER_DATE]


def make_pair(tile, t1, t2, skipped):
    days = C.interval_days(t1["date"], t2["date"])
    if days < cfg.MIN_INTERVAL_DAYS or days > cfg.MAX_INTERVAL_DAYS:
        skipped.append({"mgrs_tile": tile, "t1_date": t1["date"], "t2_date": t2["date"], "reason": "interval_out_of_range"})
        return None
    samples = min(int(t1.get("candidate_samples", 0)), int(t2.get("candidate_samples", 0)))
    if samples <= 0:
        skipped.append({"mgrs_tile": tile, "t1_date": t1["date"], "t2_date": t2["date"], "reason": "no_s2_bounds_or_grid"})
        return None
    t1_intersects = sum(1 for item in t1["candidate_s3"] if item.get("intersects_s2_bbox"))
    t2_intersects = sum(1 for item in t2["candidate_s3"] if item.get("intersects_s2_bbox"))
    t1_best_overlap = float(t1["candidate_s3"][0].get("s2_overlap_ratio", 0.0)) if t1["candidate_s3"] else 0.0
    t2_best_overlap = float(t2["candidate_s3"][0].get("s2_overlap_ratio", 0.0)) if t2["candidate_s3"] else 0.0
    return {
        "pair_id": f"{tile}_{t1['date']}_{t2['date']}",
        "source_type": "raw_zip",
        "status": "pending",
        "mgrs_tile": tile,
        "t1_date": t1["date"],
        "t2_date": t2["date"],
        "interval_days": days,
        "candidate_samples": samples,
        "t1": t1,
        "t2": t2,
        "quality": {
            "match_level": "server_style_fast_lowest_cloud_two_dates",
            "pair_strategy": cfg.PAIR_STRATEGY,
            "max_s2_cloud_percent": cfg.MAX_S2_CLOUD_PERCENT,
            "allowed_years": sorted(C.configured_allowed_years()),
            "require_s3_bbox_contains_s2": getattr(cfg, "REQUIRE_S3_BBOX_CONTAINS_S2", False),
            "t1_s3_candidates": len(t1["candidate_s3"]),
            "t2_s3_candidates": len(t2["candidate_s3"]),
            "t1_s3_intersect_count": t1_intersects,
            "t2_s3_intersect_count": t2_intersects,
            "t1_best_s3_overlap_ratio": t1_best_overlap,
            "t2_best_s3_overlap_ratio": t2_best_overlap,
            "t1_cloud": t1["s2"].get("cloud", ""),
            "t2_cloud": t2["s2"].get("cloud", ""),
        },
    }


def build_analysis(all_products, s2_cloud_items, selected_s2, bounded_s3, pairs, skipped, s3_geo_cache_path="", s3_bounds_sources=None, spatial_filter_report=None):
    s2_items = [item for item in all_products if item["sensor"] == "s2"]
    s3_items = [item for item in all_products if item["sensor"] == "s3"]
    clouds = [pair["t1"]["s2"].get("cloud", "") for pair in pairs] + [pair["t2"]["s2"].get("cloud", "") for pair in pairs]
    intervals = [pair["interval_days"] for pair in pairs]
    inter_counts, overlap_ratios = [], []
    for pair in pairs:
        inter_counts.append(pair["quality"].get("t1_s3_intersect_count", 0))
        inter_counts.append(pair["quality"].get("t2_s3_intersect_count", 0))
        overlap_ratios.append(pair["quality"].get("t1_best_s3_overlap_ratio", 0.0))
        overlap_ratios.append(pair["quality"].get("t2_best_s3_overlap_ratio", 0.0))
    s3_bounds_sources = s3_bounds_sources or Counter()
    spatial_filter_report = spatial_filter_report or []
    return {
        "schema": "s23_pair_analysis_v2_fast",
        "file_counts": {
            "name_prefilter_total": len(all_products),
            "name_prefilter_s2": len(s2_items),
            "name_prefilter_s3": len(s3_items),
            "s2_cloud_checked": len(s2_cloud_items),
            "s2_after_cloud_filter": sum(1 for item in s2_cloud_items if passes_cloud_filter(item)),
            "s2_selected_for_bounds": len(selected_s2),
            "s3_selected_for_bounds": len(bounded_s3),
            "unique_mgrs_tiles": len({item["mgrs_tile"] for item in s2_items if item.get("mgrs_tile")}),
            "unique_s2_dates": len({item["date"] for item in s2_items if item.get("date")}),
            "unique_s3_dates": len({item["date"] for item in s3_items if item.get("date")}),
        },
        "space": {
            "mgrs_pair_counts": dict(sorted(Counter(pair["mgrs_tile"] for pair in pairs).items())),
            "s2_selected_bbox_wgs84": merged_bbox([item.get("bounds_wgs84") for item in selected_s2]),
            "pair_mgrs_tile_count": len({pair["mgrs_tile"] for pair in pairs}),
            "dataset_spatial_split": spatial_filter_report,
        },
        "time": {
            "t1_year_counts": C.histogram(date_parts(pair["t1_date"])["year"] for pair in pairs),
            "t2_year_counts": C.histogram(date_parts(pair["t2_date"])["year"] for pair in pairs),
            "t1_month_counts": C.histogram(date_parts(pair["t1_date"])["month"] for pair in pairs),
            "t2_month_counts": C.histogram(date_parts(pair["t2_date"])["month"] for pair in pairs),
            "interval_days": C.number_summary(intervals),
        },
        "cloud": {
            "max_s2_cloud_percent": cfg.MAX_S2_CLOUD_PERCENT,
            "allowed_years": sorted(C.configured_allowed_years()),
            "s2_cloud_percent": C.number_summary(clouds),
            "missing_cloud_records": sum(1 for item in s2_cloud_items if cloud_value(item) is None),
            "rejected_by_cloud": sum(1 for item in s2_cloud_items if not passes_cloud_filter(item)),
        },
        "s2_s3_spatial_intersection": {
            "require_s3_bbox_contains_s2": getattr(cfg, "REQUIRE_S3_BBOX_CONTAINS_S2", False),
            "s3_geo_cache_path": s3_geo_cache_path,
            "s3_bounds_source_counts": dict(sorted(s3_bounds_sources.items())),
            "observation_times": len(inter_counts),
            "observation_times_with_intersection": sum(1 for count in inter_counts if count > 0),
            "candidate_intersection_counts": C.number_summary(inter_counts),
            "best_candidate_overlap_ratio": C.number_summary(overlap_ratios),
        },
        "pair_quality": {
            "pair_strategy": cfg.PAIR_STRATEGY,
            "pairs": len(pairs),
            "candidate_samples": sum(pair["candidate_samples"] for pair in pairs),
            "skipped_observations": len(skipped),
        },
    }


def main():
    C.progress(f"03 start: dataset_group={cfg.DATASET_GROUP}, strategy={cfg.PAIR_STRATEGY}, max_cloud={cfg.MAX_S2_CLOUD_PERCENT}")
    all_products = C.scan_product_names(cfg.SEARCH_ROOTS + [cfg.DOWNLOAD_ROOT], label="03 filename inventory")
    raw_s2_name_items = [item for item in all_products if item["sensor"] == "s2" and item.get("mgrs_tile")]
    s3_name_items = [item for item in all_products if item["sensor"] == "s3"]
    s3_dates = {item["date"] for item in s3_name_items}
    raw_s2_name_items = [item for item in raw_s2_name_items if item["date"] in s3_dates]

    s2_by_mgrs_date = defaultdict(list)
    for item in raw_s2_name_items:
        s2_by_mgrs_date[(item["mgrs_tile"], item["date"])].append(item)
    s2_name_items = [sorted(items, key=lambda x: (x["name"], x["path"]))[0] for items in s2_by_mgrs_date.values()]
    C.progress(
        f"03 name filter: S2 same-day raw={len(raw_s2_name_items)}, "
        f"unique MGRS-date={len(s2_name_items)}, S3 candidates by name={len(s3_name_items)}"
    )

    s2_cloud_items = read_s2_cloud_parallel(s2_name_items)
    low_cloud_s2 = [item for item in s2_cloud_items if passes_cloud_filter(item)]
    C.progress(f"03 S2 cloud: kept {len(low_cloud_s2)}/{len(s2_cloud_items)} with cloud <= {cfg.MAX_S2_CLOUD_PERCENT}")

    by_mgrs_date = defaultdict(list)
    for item in low_cloud_s2:
        by_mgrs_date[(item["mgrs_tile"], item["date"])].append(item)

    best_by_mgrs = defaultdict(list)
    for (tile, date), items in by_mgrs_date.items():
        best = sorted(items, key=lambda x: (cloud_value(x), x["name"]))[0]
        best_by_mgrs[tile].append(best)

    topk = max(2, int(getattr(cfg, "S2_TOPK_PER_MGRS", 4)))
    selected_s2_light = []
    for tile, items in sorted(best_by_mgrs.items()):
        selected_s2_light.extend(sorted(items, key=lambda x: (cloud_value(x), x["date"], x["name"]))[:topk])
    C.progress(f"03 S2 select: selected {len(selected_s2_light)} low-cloud S2 observations from {len(best_by_mgrs)} MGRS tiles, topk={topk}")

    needed_dates = {item["date"] for item in selected_s2_light}
    s3_needed = [item for item in s3_name_items if item["date"] in needed_dates]
    C.progress(f"03 S3 select: selected {len(s3_needed)} S3 products for {len(needed_dates)} needed dates")

    spatial_filter_report = []
    selected_s2 = map_parallel(selected_s2_light, with_s2_bounds, "03 S2 bounds")
    selected_s2, s2_spatial_report = filter_by_dataset_bbox(selected_s2, "S2")
    spatial_filter_report.append(s2_spatial_report)
    s3_geo_cache, s3_geo_cache_path = load_s3_geo_cache()
    bounded_s3 = map_parallel(s3_needed, lambda item: with_s3_bounds(item, s3_geo_cache), "03 S3 bounds")
    s3_bounds_sources = Counter(item.get("bounds_source", "failed") for item in bounded_s3)
    C.progress(f"03 S3 bounds: source counts {dict(sorted(s3_bounds_sources.items()))}, cache={s3_geo_cache_path}")
    bounded_s3 = [item for item in bounded_s3 if item.get("bounds_wgs84")]
    s3_by_date = defaultdict(list)
    for item in bounded_s3:
        s3_by_date[item["date"]].append(item)

    observations_by_tile = defaultdict(list)
    skipped = []
    for item in selected_s2:
        s3_candidates = best_s3_candidates_for_s2(item, s3_by_date)
        if not s3_candidates:
            skipped.append({"mgrs_tile": item["mgrs_tile"], "date": item["date"], "reason": "no_same_day_s3_contains_s2_bbox"})
            continue
        observations_by_tile[item["mgrs_tile"]].append({
            "date": item["date"],
            "s2": item,
            "candidate_s3": s3_candidates,
            "candidate_samples": item.get("candidate_samples", 0),
        })

    C.progress(f"03 observations: {sum(len(v) for v in observations_by_tile.values())} S2/S3 observations across {len(observations_by_tile)} MGRS tiles")
    pairs = []
    tiles = sorted(observations_by_tile.items())
    step = C.progress_every(0, len(tiles))
    for tile_idx, (tile, observations) in enumerate(tiles, 1):
        if len(observations) < 2:
            skipped.append({"mgrs_tile": tile, "reason": "fewer_than_two_matched_low_cloud_observations", "observations": len(observations)})
            continue
        selected = sorted(observations, key=lambda x: (cloud_value(x["s2"]), x["date"], x["s2"]["name"]))[:2]
        t1, t2 = sorted(selected, key=lambda x: x["date"])
        pair = make_pair(tile, t1, t2, skipped)
        if pair is not None:
            pairs.append(pair)
        if tile_idx == len(tiles) or tile_idx % step == 0:
            C.progress(f"03 pairing: processed {tile_idx}/{len(tiles)} MGRS tiles, pairs={len(pairs)}")
        if cfg.MAX_PAIRS_TO_PROCESS is not None and len(pairs) >= cfg.MAX_PAIRS_TO_PROCESS:
            C.progress(f"03 pairing: reached MAX_PAIRS_TO_PROCESS={cfg.MAX_PAIRS_TO_PROCESS}, stopping early")
            break

    total_pairs_before_limit = len(pairs)
    if cfg.MAX_PAIRS_TO_PROCESS is not None:
        pairs = pairs[: cfg.MAX_PAIRS_TO_PROCESS]

    analysis = build_analysis(all_products, s2_cloud_items, selected_s2, bounded_s3, pairs, skipped, s3_geo_cache_path, s3_bounds_sources, spatial_filter_report)
    analysis["pair_quality"]["total_pairs_before_limit"] = total_pairs_before_limit
    analysis["pair_quality"]["max_pairs_to_process"] = cfg.MAX_PAIRS_TO_PROCESS if cfg.MAX_PAIRS_TO_PROCESS is not None else ""
    index = {
        "schema": "s23_pair_index_raw_zip_v5_fast_server_style",
        "dataset_group": cfg.DATASET_GROUP,
        "dataset_name": cfg.DATASET_NAME,
        "pair_strategy": cfg.PAIR_STRATEGY,
        "max_s2_cloud_percent": cfg.MAX_S2_CLOUD_PERCENT,
        "allowed_years": sorted(C.configured_allowed_years()),
        "require_s3_bbox_contains_s2": getattr(cfg, "REQUIRE_S3_BBOX_CONTAINS_S2", False),
        "s3_geo_cache_path": s3_geo_cache_path,
        "s3_bounds_source_counts": dict(sorted(s3_bounds_sources.items())),
        "dataset_spatial_split": spatial_filter_report,
        "search_roots": [str(path) for path in cfg.SEARCH_ROOTS],
        "pairs": pairs,
        "skipped_observations": skipped,
        "analysis_json": str(cfg.PAIR_ANALYSIS_JSON),
        "counts": analysis["pair_quality"],
    }
    C.write_json(cfg.PAIR_INDEX_JSON, index)
    C.write_json(cfg.PAIR_ANALYSIS_JSON, analysis)
    C.write_csv(
        cfg.PAIR_SUMMARY_CSV,
        [
            {
                "pair_id": p["pair_id"],
                "mgrs_tile": p["mgrs_tile"],
                "t1_date": p["t1_date"],
                "t2_date": p["t2_date"],
                "interval_days": p["interval_days"],
                "candidate_samples": p["candidate_samples"],
                "max_s2_cloud_percent": p["quality"].get("max_s2_cloud_percent", ""),
                "t1_cloud": p["quality"].get("t1_cloud", ""),
                "t2_cloud": p["quality"].get("t2_cloud", ""),
                "t1_s3_candidates": p["quality"]["t1_s3_candidates"],
                "t2_s3_candidates": p["quality"]["t2_s3_candidates"],
                "t1_s3_intersect_count": p["quality"]["t1_s3_intersect_count"],
                "t2_s3_intersect_count": p["quality"]["t2_s3_intersect_count"],
                "t1_best_s3_overlap_ratio": p["quality"]["t1_best_s3_overlap_ratio"],
                "t2_best_s3_overlap_ratio": p["quality"]["t2_best_s3_overlap_ratio"],
            }
            for p in pairs
        ],
    )
    C.progress(f"03 finish: wrote {len(pairs)} pairs")
    print(f"pair index written: {cfg.PAIR_INDEX_JSON}")
    print(f"pair analysis written: {cfg.PAIR_ANALYSIS_JSON}")
    print(index["counts"])


if __name__ == "__main__":
    main()








