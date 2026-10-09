"""v3 local S2/S3 pairing from downloaded/raw zip files.

Differences from v2:
- filters S2 by cloud and data percentage before pairing;
- builds valid same-day S2/S3 observations first;
- selects first-next-valid pairs instead of the two lowest-cloud dates;
- records S2 percentage, S3 coverage source, and rejection reasons.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pickle

import s23_common as C

cfg = C.cfg


def cloud_value(item):
    value = item.get("cloud", "")
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def data_percentage_value(item):
    value = item.get("s2_data_percentage", "")
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def passes_cloud_filter(item):
    value = cloud_value(item)
    return value is not None and value <= cfg.MAX_S2_CLOUD_PERCENT


def passes_data_percentage_filter(item):
    value = data_percentage_value(item)
    return value is not None and value >= cfg.MIN_S2_DATA_PERCENTAGE


def date_parts(date_text):
    if not date_text:
        return {"year": "", "month": ""}
    return {"year": date_text[:4], "month": date_text[4:6]}


def merged_bbox(bounds_list):
    bounds = [b for b in bounds_list if b]
    if not bounds:
        return []
    return [min(b[0] for b in bounds), min(b[1] for b in bounds), max(b[2] for b in bounds), max(b[3] for b in bounds)]


def read_s2_quality_one(item):
    item = dict(item)
    item["cloud"] = C.read_s2_cloud(item["path"])
    item["s2_data_percentage"] = C.read_s2_data_percentage(item["path"])
    try:
        item["bounds_wgs84"] = C.s2_bounds_wgs84(item["path"])
        item["candidate_samples"] = int(getattr(cfg, "S2_CANDIDATE_SAMPLES_ESTIMATE", 100))
    except Exception as exc:
        item["bounds_wgs84"] = []
        item["candidate_samples"] = 0
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


def best_s3_candidates_for_s2(s2, s3_by_date):
    candidates = []
    for s3 in sorted(s3_by_date.get(s2["date"], []), key=lambda x: x["name"]):
        item = dict(s3)
        item["s2_overlap_ratio"] = C.bbox_overlap_ratio(s2.get("bounds_wgs84"), s3.get("bounds_wgs84"))
        item["intersects_s2_bbox"] = item["s2_overlap_ratio"] > 0.0
        item["contains_s2_bbox"] = C.bbox_contains(s3.get("bounds_wgs84"), s2.get("bounds_wgs84"))
        if getattr(cfg, "REQUIRE_S3_BBOX_CONTAINS_S2", False) and not item["contains_s2_bbox"]:
            continue
        candidates.append(item)
    return sorted(candidates, key=lambda x: (-float(x["s2_overlap_ratio"]), x["name"]))[: cfg.MAX_S3_CANDIDATES_PER_DATE]


def make_pair(tile, t1, t2, skipped):
    days = C.interval_days(t1["date"], t2["date"])
    if days < cfg.MIN_INTERVAL_DAYS or days > cfg.FALLBACK_MAX_INTERVAL_DAYS:
        skipped.append({"mgrs_tile": tile, "t1_date": t1["date"], "t2_date": t2["date"], "reason": "interval_out_of_range"})
        return None
    samples = min(int(t1.get("candidate_samples", 0)), int(t2.get("candidate_samples", 0)))
    if samples <= 0:
        skipped.append({"mgrs_tile": tile, "t1_date": t1["date"], "t2_date": t2["date"], "reason": "no_s2_bounds_or_grid"})
        return None
    t1_best = float(t1["candidate_s3"][0].get("s2_overlap_ratio", 0.0)) if t1["candidate_s3"] else 0.0
    t2_best = float(t2["candidate_s3"][0].get("s2_overlap_ratio", 0.0)) if t2["candidate_s3"] else 0.0
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
            "match_level": "v3_first_next_valid_local_s2s3_observation",
            "pair_strategy": cfg.PAIR_STRATEGY,
            "interval_pool": "primary" if days <= cfg.MAX_INTERVAL_DAYS else "fallback",
            "max_s2_cloud_percent": cfg.MAX_S2_CLOUD_PERCENT,
            "min_s2_data_percentage": cfg.MIN_S2_DATA_PERCENTAGE,
            "require_s3_bbox_contains_s2": getattr(cfg, "REQUIRE_S3_BBOX_CONTAINS_S2", False),
            "t1_s3_candidates": len(t1["candidate_s3"]),
            "t2_s3_candidates": len(t2["candidate_s3"]),
            "t1_s3_intersect_count": sum(1 for item in t1["candidate_s3"] if item.get("intersects_s2_bbox")),
            "t2_s3_intersect_count": sum(1 for item in t2["candidate_s3"] if item.get("intersects_s2_bbox")),
            "t1_best_s3_overlap_ratio": t1_best,
            "t2_best_s3_overlap_ratio": t2_best,
            "t1_cloud": t1["s2"].get("cloud", ""),
            "t2_cloud": t2["s2"].get("cloud", ""),
            "t1_s2_data_percentage": t1["s2"].get("s2_data_percentage", ""),
            "t2_s2_data_percentage": t2["s2"].get("s2_data_percentage", ""),
        },
    }


def select_first_next_pair(tile, observations, skipped):
    observations = sorted(observations, key=lambda x: (x["date"], cloud_value(x["s2"]), x["s2"]["name"]))
    fallback = None
    for i, t1 in enumerate(observations):
        for t2 in observations[i + 1 :]:
            days = C.interval_days(t1["date"], t2["date"])
            if days < cfg.MIN_INTERVAL_DAYS:
                continue
            if days > cfg.FALLBACK_MAX_INTERVAL_DAYS:
                break
            pair = make_pair(tile, t1, t2, skipped)
            if pair is None:
                continue
            if days <= cfg.MAX_INTERVAL_DAYS:
                return pair
            if fallback is None:
                fallback = pair
            break
    return fallback


def observation_score(obs):
    return (cloud_value(obs["s2"]) if cloud_value(obs["s2"]) is not None else 999.0, obs["date"], obs["s2"].get("name", ""))


def select_s2_topk_balanced(items, topk):
    if len(items) <= topk:
        return sorted(items, key=lambda x: (x["date"], cloud_value(x), x["name"]))
    if getattr(cfg, "S2_TOPK_SELECTION", "month_round_robin") != "month_round_robin":
        return sorted(items, key=lambda x: (x["date"], cloud_value(x), x["name"]))[:topk]
    by_month = defaultdict(list)
    for item in items:
        by_month[item["date"][:6]].append(item)
    for month in by_month:
        by_month[month].sort(key=lambda x: (cloud_value(x) if cloud_value(x) is not None else 999.0, x["date"], x["name"]))
    selected = []
    months = sorted(by_month)
    while len(selected) < topk and any(by_month.values()):
        for month in months:
            if by_month[month]:
                selected.append(by_month[month].pop(0))
                if len(selected) >= topk:
                    break
    return sorted(selected, key=lambda x: (x["date"], cloud_value(x), x["name"]))


def candidate_pairs_for_tile(tile, observations, skipped):
    observations = sorted(observations, key=lambda x: (x["date"], cloud_value(x["s2"]), x["s2"].get("name", "")))
    candidates = []
    for i, t1 in enumerate(observations):
        best_primary = None
        best_fallback = None
        for t2 in observations[i + 1 :]:
            days = C.interval_days(t1["date"], t2["date"])
            if days < cfg.MIN_INTERVAL_DAYS:
                continue
            if days > cfg.FALLBACK_MAX_INTERVAL_DAYS:
                break
            pair = make_pair(tile, t1, t2, skipped)
            if pair is None:
                continue
            target = float(getattr(cfg, "PAIR_TARGET_INTERVAL_DAYS", 45))
            pair["selection_score"] = (
                abs(days - target),
                cloud_value(t1["s2"]) if cloud_value(t1["s2"]) is not None else 999.0,
                cloud_value(t2["s2"]) if cloud_value(t2["s2"]) is not None else 999.0,
                t1["date"],
                t2["date"],
            )
            if days <= cfg.MAX_INTERVAL_DAYS:
                if best_primary is None or pair["selection_score"] < best_primary["selection_score"]:
                    best_primary = pair
            elif best_fallback is None or pair["selection_score"] < best_fallback["selection_score"]:
                best_fallback = pair
        if best_primary is not None:
            candidates.append(best_primary)
        elif best_fallback is not None:
            candidates.append(best_fallback)
    return candidates


def select_balanced_pairs(tile, observations, skipped):
    limit = max(1, int(getattr(cfg, "PAIRS_PER_MGRS", 1)))
    candidates = candidate_pairs_for_tile(tile, observations, skipped)
    if not candidates:
        return []
    selected = []
    used_keys = set()
    used_months = set()
    pools = [
        [p for p in candidates if p.get("quality", {}).get("interval_pool") == "primary"],
        [p for p in candidates if p.get("quality", {}).get("interval_pool") != "primary"],
    ]
    for pool in pools:
        for prefer_new_month in (True, False):
            ordered = sorted(pool, key=lambda p: ((p["t1_date"][:6] in used_months) if prefer_new_month else False, p.get("selection_score", ()), p["t1_date"], p["t2_date"]))
            for pair in ordered:
                if len(selected) >= limit:
                    return selected
                key = (pair["t1_date"], pair["t2_date"])
                if key in used_keys:
                    continue
                if prefer_new_month and pair["t1_date"][:6] in used_months:
                    continue
                rank = len(selected) + 1
                pair["selected_pair_rank"] = rank
                pair["quality"]["selection_score"] = list(pair.pop("selection_score", ()))
                selected.append(pair)
                used_keys.add(key)
                used_months.add(pair["t1_date"][:6])
    return selected


def item_from_downloaded_path(path, sensor, mgrs_tile=""):
    name = C.product_name_from_path(path)
    return {
        "name": name,
        "path": str(path),
        "sensor": sensor,
        "date": C.product_date(name),
        "mgrs_tile": mgrs_tile if sensor == "s2" else "",
        "valid": True,
        "reason": "cdse_accepted_downloaded",
        "size": Path(path).stat().st_size if Path(path).exists() else 0,
    }


def observation_from_accepted(pair, side, s3_geo_cache):
    s2_product = pair[side]["s2"]
    s3_product = pair[side]["candidate_s3"][0]
    paths = pair.get("downloaded_paths", {})
    s2_path = paths[f"{side}_s2"]
    s3_path = paths[f"{side}_s3"]
    s2 = item_from_downloaded_path(s2_path, "s2", pair["mgrs_tile"])
    s2["cloud"] = s2_product.get("cloud", "")
    s2["s2_data_percentage"] = C.read_s2_data_percentage(s2_path)
    s2["bounds_wgs84"] = C.s2_bounds_wgs84(s2_path)
    s2["candidate_samples"] = int(getattr(cfg, "S2_CANDIDATE_SAMPLES_ESTIMATE", 100))
    s3 = item_from_downloaded_path(s3_path, "s3")
    s3 = with_s3_bounds(s3, s3_geo_cache)
    s3["s2_overlap_ratio"] = C.bbox_overlap_ratio(s2.get("bounds_wgs84"), s3.get("bounds_wgs84"))
    s3["intersects_s2_bbox"] = s3["s2_overlap_ratio"] > 0.0
    s3["contains_s2_bbox"] = C.bbox_contains(s3.get("bounds_wgs84"), s2.get("bounds_wgs84"))
    return {
        "date": s2["date"],
        "s2": s2,
        "candidate_s3": [s3],
        "candidate_samples": s2.get("candidate_samples", 0),
    }, s2, s3


def accepted_pair_list(value):
    if not value:
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [value]
    return []


def build_pairs_from_accepted_json():
    state = C.read_json(cfg.CDSE_ACCEPTED_DOWNLOADS_JSON, {})
    accepted = state.get("accepted", {}) if isinstance(state, dict) else {}
    if not accepted:
        return None
    s3_geo_cache, s3_geo_cache_path = load_s3_geo_cache()
    pairs, skipped, all_products, s2_items, s3_items = [], [], [], [], []
    for tile, accepted_value in sorted(accepted.items()):
        for accepted_pair in accepted_pair_list(accepted_value):
            try:
                t1, s2_1, s3_1 = observation_from_accepted(accepted_pair, "t1", s3_geo_cache)
                t2, s2_2, s3_2 = observation_from_accepted(accepted_pair, "t2", s3_geo_cache)
                all_products.extend([s2_1, s2_2, s3_1, s3_2])
                s2_items.extend([s2_1, s2_2])
                s3_items.extend([s3_1, s3_2])
                if not (passes_cloud_filter(s2_1) and passes_cloud_filter(s2_2)):
                    skipped.append({"mgrs_tile": tile, "rank": accepted_pair.get("rank", ""), "reason": "accepted_json_s2_cloud_filter_failed"})
                    continue
                if not (passes_data_percentage_filter(s2_1) and passes_data_percentage_filter(s2_2)):
                    skipped.append({"mgrs_tile": tile, "rank": accepted_pair.get("rank", ""), "reason": "accepted_json_s2_data_percentage_filter_failed"})
                    continue
                if getattr(cfg, "REQUIRE_S3_BBOX_CONTAINS_S2", False) and (not s3_1.get("contains_s2_bbox") or not s3_2.get("contains_s2_bbox")):
                    skipped.append({"mgrs_tile": tile, "rank": accepted_pair.get("rank", ""), "reason": "accepted_json_local_s3_bounds_filter_failed"})
                    continue
                pair = make_pair(tile, t1, t2, skipped)
                if pair is not None:
                    pair["selected_candidate_rank"] = accepted_pair.get("rank", "")
                    pair["download_selection_round"] = accepted_pair.get("download_selection_round", "")
                    pair["candidate_source"] = "cdse_accepted_downloads_json"
                    pairs.append(pair)
            except Exception as exc:
                skipped.append({"mgrs_tile": tile, "rank": accepted_pair.get("rank", ""), "reason": "accepted_json_exception", "error": str(exc)})
    s3_bounds_sources = Counter(item.get("bounds_source", "failed") for item in s3_items)
    analysis = build_analysis(all_products, s2_items, s2_items, s3_items, pairs, skipped, s3_geo_cache_path, s3_bounds_sources)
    analysis["pair_quality"]["total_pairs_before_limit"] = len(pairs)
    analysis["pair_quality"]["max_pairs_to_process"] = ""
    return pairs, skipped, analysis, s3_geo_cache_path, s3_bounds_sources

def build_analysis(all_products, s2_quality_items, selected_s2, bounded_s3, pairs, skipped, s3_geo_cache_path="", s3_bounds_sources=None):
    s2_items = [item for item in all_products if item["sensor"] == "s2"]
    s3_items = [item for item in all_products if item["sensor"] == "s3"]
    clouds = [pair["quality"].get("t1_cloud", "") for pair in pairs] + [pair["quality"].get("t2_cloud", "") for pair in pairs]
    percentages = [pair["quality"].get("t1_s2_data_percentage", "") for pair in pairs] + [pair["quality"].get("t2_s2_data_percentage", "") for pair in pairs]
    intervals = [pair["interval_days"] for pair in pairs]
    overlaps = []
    inter_counts = []
    for pair in pairs:
        overlaps.extend([pair["quality"].get("t1_best_s3_overlap_ratio", 0.0), pair["quality"].get("t2_best_s3_overlap_ratio", 0.0)])
        inter_counts.extend([pair["quality"].get("t1_s3_intersect_count", 0), pair["quality"].get("t2_s3_intersect_count", 0)])
    s3_bounds_sources = s3_bounds_sources or Counter()
    return {
        "schema": "s23_pair_analysis_v3_balanced_month_interval",
        "file_counts": {
            "name_prefilter_total": len(all_products),
            "name_prefilter_s2": len(s2_items),
            "name_prefilter_s3": len(s3_items),
            "s2_quality_checked": len(s2_quality_items),
            "s2_after_cloud_filter": sum(1 for item in s2_quality_items if passes_cloud_filter(item)),
            "s2_after_data_percentage_filter": sum(1 for item in s2_quality_items if passes_cloud_filter(item) and passes_data_percentage_filter(item)),
            "s2_selected_for_pairing": len(selected_s2),
            "s3_selected_for_bounds": len(bounded_s3),
            "unique_mgrs_tiles": len({item["mgrs_tile"] for item in selected_s2 if item.get("mgrs_tile")}),
            "unique_s2_dates": len({item["date"] for item in s2_items if item.get("date")}),
            "unique_s3_dates": len({item["date"] for item in s3_items if item.get("date")}),
        },
        "space": {
            "s2_selected_bbox_wgs84": merged_bbox([item.get("bounds_wgs84") for item in selected_s2]),
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
            "missing_cloud_records": sum(1 for item in s2_quality_items if cloud_value(item) is None),
            "rejected_by_cloud": sum(1 for item in s2_quality_items if not passes_cloud_filter(item)),
        },
        "s2_data_percentage": {
            "min_s2_data_percentage": cfg.MIN_S2_DATA_PERCENTAGE,
            "accepted_pair_values": C.number_summary(percentages),
            "missing_records": sum(1 for item in s2_quality_items if data_percentage_value(item) is None),
            "rejected_by_data_percentage": sum(1 for item in s2_quality_items if passes_cloud_filter(item) and not passes_data_percentage_filter(item)),
        },
        "s2_s3_spatial_intersection": {
            "require_s3_bbox_contains_s2": getattr(cfg, "REQUIRE_S3_BBOX_CONTAINS_S2", False),
            "s3_geo_cache_path": s3_geo_cache_path,
            "s3_bounds_source_counts": dict(sorted(s3_bounds_sources.items())),
            "observation_times": len(inter_counts),
            "observation_times_with_intersection": sum(1 for count in inter_counts if count > 0),
            "candidate_intersection_counts": C.number_summary(inter_counts),
            "best_candidate_overlap_ratio": C.number_summary(overlaps),
        },
        "pair_quality": {
            "pair_strategy": cfg.PAIR_STRATEGY,
            "pairs": len(pairs),
            "candidate_samples": sum(pair["candidate_samples"] for pair in pairs),
            "skipped_observations": len(skipped),
        },
        "skip_reasons": dict(Counter(item.get("reason", "unknown") for item in skipped).most_common()),
    }


def main():
    C.progress(f"03 v3 start: role={cfg.DATASET_ROLE}, strategy={cfg.PAIR_STRATEGY}, cloud<={cfg.MAX_S2_CLOUD_PERCENT}, s2_pct>={cfg.MIN_S2_DATA_PERCENTAGE}")
    input_mode = getattr(cfg, "PAIR_INPUT_MODE", "auto")
    if input_mode not in {"auto", "json", "directory"}:
        raise ValueError("PAIR_INPUT_MODE must be auto, json, or directory")
    if input_mode in {"auto", "json"} and cfg.CDSE_ACCEPTED_DOWNLOADS_JSON.exists():
        built = build_pairs_from_accepted_json()
        if built is not None:
            pairs, skipped, analysis, s3_geo_cache_path, s3_bounds_sources = built
            index = {
                "schema": "s23_pair_index_raw_zip_v3_cdse_accepted_json",
                "dataset_role": cfg.DATASET_ROLE,
                "dataset_group": cfg.DATASET_GROUP,
                "dataset_name": cfg.DATASET_NAME,
                "pair_strategy": cfg.PAIR_STRATEGY,
                "pair_input_mode": "json",
                "accepted_downloads_json": str(cfg.CDSE_ACCEPTED_DOWNLOADS_JSON),
                "max_s2_cloud_percent": cfg.MAX_S2_CLOUD_PERCENT,
                "min_s2_data_percentage": cfg.MIN_S2_DATA_PERCENTAGE,
                "allowed_years": sorted(C.configured_allowed_years()),
                "require_s3_bbox_contains_s2": getattr(cfg, "REQUIRE_S3_BBOX_CONTAINS_S2", False),
                "s3_geo_cache_path": s3_geo_cache_path,
                "s3_bounds_source_counts": dict(sorted(s3_bounds_sources.items())),
                "pairs": pairs,
                "skipped_observations": skipped,
                "analysis_json": str(cfg.PAIR_ANALYSIS_JSON),
                "counts": analysis["pair_quality"],
            }
            C.write_json(cfg.PAIR_INDEX_JSON, index)
            C.write_json(cfg.PAIR_ANALYSIS_JSON, analysis)
            C.write_csv(cfg.PAIR_SUMMARY_CSV, [{
                "pair_id": p["pair_id"], "mgrs_tile": p["mgrs_tile"], "t1_date": p["t1_date"], "t2_date": p["t2_date"],
                "interval_days": p["interval_days"], "interval_pool": p["quality"].get("interval_pool", ""),
                "candidate_samples": p["candidate_samples"], "selected_candidate_rank": p.get("selected_candidate_rank", ""),
                "max_s2_cloud_percent": p["quality"].get("max_s2_cloud_percent", ""),
                "min_s2_data_percentage": p["quality"].get("min_s2_data_percentage", ""),
                "t1_cloud": p["quality"].get("t1_cloud", ""), "t2_cloud": p["quality"].get("t2_cloud", ""),
                "t1_s2_data_percentage": p["quality"].get("t1_s2_data_percentage", ""),
                "t2_s2_data_percentage": p["quality"].get("t2_s2_data_percentage", ""),
                "t1_s3_candidates": p["quality"].get("t1_s3_candidates", ""),
                "t2_s3_candidates": p["quality"].get("t2_s3_candidates", ""),
                "t1_best_s3_overlap_ratio": p["quality"].get("t1_best_s3_overlap_ratio", ""),
                "t2_best_s3_overlap_ratio": p["quality"].get("t2_best_s3_overlap_ratio", ""),
            } for p in pairs])
            C.progress(f"03 finish json mode: wrote {len(pairs)} pairs")
            print(f"pair index written: {cfg.PAIR_INDEX_JSON}")
            print(f"pair analysis written: {cfg.PAIR_ANALYSIS_JSON}")
            print(index["counts"])
            return
        if input_mode == "json":
            raise SystemExit(f"No accepted downloads found in {cfg.CDSE_ACCEPTED_DOWNLOADS_JSON}")
    C.progress("03 input mode=directory scan")
    all_products = C.scan_product_names(cfg.SEARCH_ROOTS + [cfg.DOWNLOAD_ROOT], label="03 filename inventory")
    s2_name_items = [item for item in all_products if item["sensor"] == "s2" and item.get("mgrs_tile")]
    s3_name_items = [item for item in all_products if item["sensor"] == "s3"]
    s3_dates = {item["date"] for item in s3_name_items}
    s2_name_items = [item for item in s2_name_items if item["date"] in s3_dates]

    by_mgrs_date = defaultdict(list)
    for item in s2_name_items:
        by_mgrs_date[(item["mgrs_tile"], item["date"])].append(item)
    s2_unique = [sorted(items, key=lambda x: (x["name"], x["path"]))[0] for items in by_mgrs_date.values()]

    C.progress(f"03 name filter: unique S2 MGRS-date={len(s2_unique)}, S3 candidates by name={len(s3_name_items)}")

    s2_quality_items = map_parallel(s2_unique, read_s2_quality_one, "03 S2 quality")
    
    valid_s2 = [item for item in s2_quality_items if passes_cloud_filter(item) and passes_data_percentage_filter(item) and item.get("bounds_wgs84")]
    C.progress(f"03 S2 filters: kept {len(valid_s2)}/{len(s2_quality_items)} cloud<= {cfg.MAX_S2_CLOUD_PERCENT}, data_pct>= {cfg.MIN_S2_DATA_PERCENTAGE}")

    topk = max(2, int(getattr(cfg, "S2_TOPK_PER_MGRS", 64)))
    grouped = defaultdict(list)
    for item in valid_s2:
        grouped[item["mgrs_tile"]].append(item)
    selected_s2 = []
    for tile, items in sorted(grouped.items()):
        selected_s2.extend(select_s2_topk_balanced(items, topk))
    needed_dates = {item["date"] for item in selected_s2}
    s3_needed = [item for item in s3_name_items if item["date"] in needed_dates]
    C.progress(f"03 S3 select: {len(s3_needed)} S3 products for {len(needed_dates)} needed dates")

    s3_geo_cache, s3_geo_cache_path = load_s3_geo_cache()
    bounded_s3 = map_parallel(s3_needed, lambda item: with_s3_bounds(item, s3_geo_cache), "03 S3 bounds")
    s3_bounds_sources = Counter(item.get("bounds_source", "failed") for item in bounded_s3)
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

    C.progress(f"03 observations: {sum(len(v) for v in observations_by_tile.values())} valid S2/S3 observations across {len(observations_by_tile)} MGRS")
    pairs = []
    tiles = sorted(observations_by_tile.items())
    step = C.progress_every(0, len(tiles))
    for idx, (tile, observations) in enumerate(tiles, 1):
        if len(observations) < 2:
            skipped.append({"mgrs_tile": tile, "reason": "fewer_than_two_valid_observations", "observations": len(observations)})
            continue
        selected_pairs = select_balanced_pairs(tile, observations, skipped)
        if selected_pairs:
            pairs.extend(selected_pairs)
        else:
            skipped.append({"mgrs_tile": tile, "reason": "no_balanced_pair_in_interval", "observations": len(observations)})
        if idx == len(tiles) or idx % step == 0:
            C.progress(f"03 pairing: processed {idx}/{len(tiles)} MGRS, pairs={len(pairs)}")
        if cfg.MAX_PAIRS_TO_PROCESS is not None and len(pairs) >= cfg.MAX_PAIRS_TO_PROCESS:
            break

    total_pairs_before_limit = len(pairs)
    if cfg.MAX_PAIRS_TO_PROCESS is not None:
        pairs = pairs[: cfg.MAX_PAIRS_TO_PROCESS]

    analysis = build_analysis(all_products, s2_quality_items, selected_s2, bounded_s3, pairs, skipped, s3_geo_cache_path, s3_bounds_sources)
    analysis["pair_quality"]["total_pairs_before_limit"] = total_pairs_before_limit
    analysis["pair_quality"]["max_pairs_to_process"] = cfg.MAX_PAIRS_TO_PROCESS if cfg.MAX_PAIRS_TO_PROCESS is not None else ""
    index = {
        "schema": "s23_pair_index_raw_zip_v3_balanced_month_interval",
        "dataset_role": cfg.DATASET_ROLE,
        "dataset_group": cfg.DATASET_GROUP,
        "dataset_name": cfg.DATASET_NAME,
        "pair_strategy": cfg.PAIR_STRATEGY,
        "max_s2_cloud_percent": cfg.MAX_S2_CLOUD_PERCENT,
        "min_s2_data_percentage": cfg.MIN_S2_DATA_PERCENTAGE,
        "allowed_years": sorted(C.configured_allowed_years()),
        "require_s3_bbox_contains_s2": getattr(cfg, "REQUIRE_S3_BBOX_CONTAINS_S2", False),
        "s3_geo_cache_path": s3_geo_cache_path,
        "s3_bounds_source_counts": dict(sorted(s3_bounds_sources.items())),
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
                "interval_pool": p["quality"].get("interval_pool", ""),
                "candidate_samples": p["candidate_samples"],
                "max_s2_cloud_percent": p["quality"].get("max_s2_cloud_percent", ""),
                "min_s2_data_percentage": p["quality"].get("min_s2_data_percentage", ""),
                "t1_cloud": p["quality"].get("t1_cloud", ""),
                "t2_cloud": p["quality"].get("t2_cloud", ""),
                "t1_s2_data_percentage": p["quality"].get("t1_s2_data_percentage", ""),
                "t2_s2_data_percentage": p["quality"].get("t2_s2_data_percentage", ""),
                "t1_s3_candidates": p["quality"].get("t1_s3_candidates", ""),
                "t2_s3_candidates": p["quality"].get("t2_s3_candidates", ""),
                "t1_best_s3_overlap_ratio": p["quality"].get("t1_best_s3_overlap_ratio", ""),
                "t2_best_s3_overlap_ratio": p["quality"].get("t2_best_s3_overlap_ratio", ""),
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


