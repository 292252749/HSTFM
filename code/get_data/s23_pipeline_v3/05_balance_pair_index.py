#!/usr/bin/env python3
"""Balance a v3 pair_index before LMDB construction.

This script is intended for the internal/local workflow after 03_pair_local_s23.py
and before 04_build_lmdb_from_pairs.py. It keeps pair selection explicit and
cheap: no image pixels are read here. The balancing uses only metadata already
available in pair_index.json: MGRS, S2 footprint center, t1 month, interval bin,
cloud, S2 data percentage, and candidate sample count.

By default it backs up the original pair_index.json to pair_index_unbalanced.json
and writes the balanced result back to pair_index.json so 04 can run unchanged.
"""

from __future__ import annotations

import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import s23_common as C

cfg = C.cfg

TARGET_PAIRS = int(os.environ.get("S23_BALANCE_TARGET_PAIRS", "2000"))
OVERWRITE = os.environ.get("S23_BALANCE_OVERWRITE", "1").strip().lower() not in {"0", "false", "no"}
IN_JSON = Path(os.environ.get("S23_BALANCE_INPUT", str(cfg.PAIR_INDEX_JSON)))
OUT_JSON = Path(os.environ.get("S23_BALANCE_OUTPUT", str(cfg.PAIR_INDEX_JSON if OVERWRITE else cfg.PIPELINE_OUT / f"pair_index_balanced_{TARGET_PAIRS}.json")))
BACKUP_JSON = Path(os.environ.get("S23_BALANCE_BACKUP", str(cfg.PIPELINE_OUT / "pair_index_unbalanced.json")))
REPORT_JSON = Path(os.environ.get("S23_BALANCE_REPORT", str(cfg.PIPELINE_OUT / f"pair_balance_report_{TARGET_PAIRS}.json")))
SUMMARY_CSV = Path(os.environ.get("S23_BALANCE_SUMMARY", str(cfg.PIPELINE_OUT / f"pair_summary_balanced_{TARGET_PAIRS}.csv")))

INTERVAL_BINS = [(5, 30), (31, 60), (61, 120), (121, 180), (181, 365)]


def to_float(value, default=None):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def interval_bin(days):
    try:
        days = int(days)
    except Exception:
        return "unknown"
    for lo, hi in INTERVAL_BINS:
        if lo <= days <= hi:
            return f"{lo:03d}_{hi:03d}d"
    return "other"


def pair_month(pair):
    date = str(pair.get("t1_date", ""))
    return date[4:6] if len(date) >= 6 else "unknown"


def pair_year_month(pair):
    date = str(pair.get("t1_date", ""))
    return date[:6] if len(date) >= 6 else "unknown"


def pair_quality(pair):
    return pair.get("quality", {}) or {}


def cloud_score(pair):
    q = pair_quality(pair)
    c1 = to_float(q.get("t1_cloud"), 0.0) or 0.0
    c2 = to_float(q.get("t2_cloud"), 0.0) or 0.0
    return (c1 + c2) / 2.0


def data_pct_score(pair):
    q = pair_quality(pair)
    p1 = to_float(q.get("t1_s2_data_percentage"), 1.0)
    p2 = to_float(q.get("t2_s2_data_percentage"), 1.0)
    vals = [v for v in (p1, p2) if v is not None]
    return sum(vals) / len(vals) if vals else 1.0


def candidate_samples_score(pair):
    return max(1.0, to_float(pair.get("candidate_samples"), 1.0) or 1.0)


def bounds_center(bounds):
    if not bounds or len(bounds) != 4:
        return None
    try:
        minx, miny, maxx, maxy = [float(v) for v in bounds]
    except Exception:
        return None
    return ((minx + maxx) / 2.0, (miny + maxy) / 2.0)


def pair_center(pair):
    for side in ("t1", "t2"):
        bounds = (((pair.get(side) or {}).get("s2") or {}).get("bounds_wgs84"))
        center = bounds_center(bounds)
        if center:
            return center
    return None


def spatial_bin(pair):
    center = pair_center(pair)
    if not center:
        return "unknown"
    lon, lat = center
    lon_bin = int(lon // 5) * 5
    lat_bin = int(lat // 5) * 5
    return f"lon{lon_bin:+04d}_lat{lat_bin:+04d}"


def latitude_band(pair):
    center = pair_center(pair)
    if not center:
        return "unknown"
    lat = center[1]
    if lat < -30:
        return "south_mid_high"
    if lat < 0:
        return "south_tropical"
    if lat < 30:
        return "north_tropical"
    if lat < 55:
        return "north_mid"
    return "north_high"


def pair_sort_quality(pair):
    # Lower is better. Keep quality as a tie-breaker; balance dominates.
    cloud = cloud_score(pair)
    pct_penalty = max(0.0, 1.0 - data_pct_score(pair))
    sample_bonus = min(candidate_samples_score(pair), 256.0) / 256.0
    interval_penalty = abs(int(pair.get("interval_days", 0) or 0) - float(getattr(cfg, "PAIR_TARGET_INTERVAL_DAYS", 60))) / 1000.0
    return cloud + pct_penalty * 5.0 + interval_penalty - sample_bonus * 0.05


def balance_score(pair, counts, selected_by_mgrs):
    mgrs = pair.get("mgrs_tile", "unknown")
    month = pair_month(pair)
    year_month = pair_year_month(pair)
    ibin = interval_bin(pair.get("interval_days"))
    sbin = spatial_bin(pair)
    lat_band = latitude_band(pair)
    score = pair_sort_quality(pair)
    score += counts["mgrs"][mgrs] * 2.50
    score += counts["month"][month] * 1.00
    score += counts["interval_bin"][ibin] * 1.15
    score += counts["spatial_bin"][sbin] * 0.55
    score += counts["lat_band"][lat_band] * 0.35
    score += counts["year_month"][year_month] * 0.20
    score += selected_by_mgrs[mgrs] * 0.75
    return score


def summarize_pairs(pairs):
    return {
        "pairs": len(pairs),
        "mgrs": len({p.get("mgrs_tile", "") for p in pairs}),
        "candidate_samples_total": sum(int(to_float(p.get("candidate_samples"), 0) or 0) for p in pairs),
        "month": dict(Counter(pair_month(p) for p in pairs)),
        "year_month": dict(Counter(pair_year_month(p) for p in pairs).most_common()),
        "interval_bin": dict(Counter(interval_bin(p.get("interval_days")) for p in pairs)),
        "latitude_band": dict(Counter(latitude_band(p) for p in pairs)),
        "spatial_bins": len(set(spatial_bin(p) for p in pairs)),
        "top_mgrs": dict(Counter(p.get("mgrs_tile", "unknown") for p in pairs).most_common(20)),
        "cloud_mean": sum(cloud_score(p) for p in pairs) / max(len(pairs), 1),
        "s2_data_percentage_mean": sum(data_pct_score(p) for p in pairs) / max(len(pairs), 1),
    }


def select_balanced_pairs(pairs, target):
    remaining = list(pairs)
    selected = []
    selected_ids = set()
    selected_by_mgrs = Counter()
    counts = {key: Counter() for key in ["mgrs", "month", "year_month", "interval_bin", "spatial_bin", "lat_band"]}

    # First pass gives every MGRS at least one chance when possible.
    by_mgrs = defaultdict(list)
    for pair in remaining:
        by_mgrs[pair.get("mgrs_tile", "unknown")].append(pair)
    mgrs_order = sorted(by_mgrs, key=lambda tile: len(by_mgrs[tile]))
    for tile in mgrs_order:
        if len(selected) >= target:
            break
        candidates = [p for p in by_mgrs[tile] if p.get("pair_id") not in selected_ids]
        if not candidates:
            continue
        candidates.sort(key=lambda p: balance_score(p, counts, selected_by_mgrs))
        add_pair(candidates[0], selected, selected_ids, selected_by_mgrs, counts)

    while len(selected) < target:
        candidates = [p for p in remaining if p.get("pair_id") not in selected_ids]
        if not candidates:
            break
        candidates.sort(key=lambda p: balance_score(p, counts, selected_by_mgrs))
        add_pair(candidates[0], selected, selected_ids, selected_by_mgrs, counts)
    return selected


def add_pair(pair, selected, selected_ids, selected_by_mgrs, counts):
    selected.append(pair)
    selected_ids.add(pair.get("pair_id"))
    mgrs = pair.get("mgrs_tile", "unknown")
    selected_by_mgrs[mgrs] += 1
    counts["mgrs"][mgrs] += 1
    counts["month"][pair_month(pair)] += 1
    counts["year_month"][pair_year_month(pair)] += 1
    counts["interval_bin"][interval_bin(pair.get("interval_days"))] += 1
    counts["spatial_bin"][spatial_bin(pair)] += 1
    counts["lat_band"][latitude_band(pair)] += 1


def write_summary_csv(path, pairs):
    rows = []
    for idx, pair in enumerate(pairs, 1):
        q = pair_quality(pair)
        rows.append({
            "balance_rank": idx,
            "pair_id": pair.get("pair_id", ""),
            "mgrs_tile": pair.get("mgrs_tile", ""),
            "t1_date": pair.get("t1_date", ""),
            "t2_date": pair.get("t2_date", ""),
            "t1_month": pair_month(pair),
            "interval_days": pair.get("interval_days", ""),
            "interval_bin": interval_bin(pair.get("interval_days")),
            "candidate_samples": pair.get("candidate_samples", ""),
            "spatial_bin": spatial_bin(pair),
            "latitude_band": latitude_band(pair),
            "t1_cloud": q.get("t1_cloud", ""),
            "t2_cloud": q.get("t2_cloud", ""),
            "t1_s2_data_percentage": q.get("t1_s2_data_percentage", ""),
            "t2_s2_data_percentage": q.get("t2_s2_data_percentage", ""),
        })
    C.write_csv(path, rows)


def main():
    if not IN_JSON.exists():
        raise SystemExit(f"Missing pair index: {IN_JSON}")
    index = C.read_json(IN_JSON, {})
    pairs = list(index.get("pairs", []) or [])
    if not pairs:
        raise SystemExit(f"No pairs found in {IN_JSON}")
    target = min(TARGET_PAIRS, len(pairs))
    C.progress(f"05 balance pair_index: input_pairs={len(pairs)}, target={target}, role={getattr(cfg, 'DATASET_ROLE', '')}")
    selected = select_balanced_pairs(pairs, target)
    for idx, pair in enumerate(selected, 1):
        pair["balance_rank"] = idx
        pair["balance_source_pair_index"] = str(IN_JSON)

    out_index = dict(index)
    out_index["schema"] = str(index.get("schema", "s23_pair_index")) + "_balanced"
    out_index["pairs"] = selected
    out_index["balance"] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_pair_index": str(IN_JSON),
        "target_pairs": TARGET_PAIRS,
        "selected_pairs": len(selected),
        "method": "greedy_balance_mgrs_month_interval_spatial_quality",
        "note": "Pair-level landcover is not used here because exact FROMGLC statistics are computed per patch in 04.",
    }
    out_index["counts"] = dict(out_index.get("counts", {}) or {})
    out_index["counts"].update({
        "pairs": len(selected),
        "total_pairs_before_balance": len(pairs),
        "candidate_samples": sum(int(to_float(p.get("candidate_samples"), 0) or 0) for p in selected),
    })

    if OVERWRITE and IN_JSON.resolve() == OUT_JSON.resolve() and not BACKUP_JSON.exists():
        BACKUP_JSON.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(IN_JSON, BACKUP_JSON)
        C.progress(f"05 backup original pair_index: {BACKUP_JSON}")

    C.write_json(OUT_JSON, out_index)
    report = {
        "input": summarize_pairs(pairs),
        "selected": summarize_pairs(selected),
        "output_pair_index": str(OUT_JSON),
        "backup_pair_index": str(BACKUP_JSON) if BACKUP_JSON.exists() else "",
        "summary_csv": str(SUMMARY_CSV),
    }
    C.write_json(REPORT_JSON, report)
    write_summary_csv(SUMMARY_CSV, selected)
    C.progress(f"05 finish: wrote {OUT_JSON}")
    print({"input_pairs": len(pairs), "selected_pairs": len(selected), "output": str(OUT_JSON), "report": str(REPORT_JSON)})


if __name__ == "__main__":
    main()
