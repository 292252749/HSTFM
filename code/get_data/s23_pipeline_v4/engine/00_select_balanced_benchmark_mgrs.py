#!/usr/bin/env python3
"""Select balanced benchmark MGRS tiles from global candidates."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import random

import s23_common as C

cfg = C.cfg

IN_JSON = cfg.PIPELINE_OUT / "global_mgrs_candidates.json"
OUT_JSON = cfg.BENCHMARK_MGRS_JSON
TARGET = int(getattr(cfg, "BENCHMARK_TARGET_MGRS", 100))
MAX_ATTEMPTS = int(getattr(cfg, "BENCHMARK_MAX_MGRS_ATTEMPTS", 200))
MAIN_LANDCOVERS = ["cropland", "forest", "grassland", "shrubland", "bareland"]


def distance_km(a, b):
    try:
        lon_a = float(a["center_lon"])
        lat_a = float(a["center_lat"])
        lon_b = float(b["center_lon"])
        lat_b = float(b["center_lat"])
    except (KeyError, TypeError, ValueError):
        return 0.0
    r = 6371.0088
    lat1 = math.radians(lat_a)
    lat2 = math.radians(lat_b)
    dlat = lat2 - lat1
    dlon = math.radians(lon_b - lon_a)
    x = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    x = max(0.0, min(1.0, x))
    return r * 2 * math.atan2(math.sqrt(x), math.sqrt(max(0.0, 1.0 - x)))


def min_dist(candidate, selected):
    if not selected:
        return float("inf")
    return min(distance_km(candidate, item) for item in selected)


def score_candidate(item, selected, lc_counts, continent_counts, lat_counts, target_per_lc):
    lc = item.get("dominant_landcover", "unknown")
    continent = item.get("continent", "Other")
    lat_band = item.get("lat_band", "unknown")
    lc_penalty = lc_counts[lc] / max(target_per_lc.get(lc, 1), 1)
    continent_penalty = continent_counts[continent] * 0.03
    lat_penalty = lat_counts[lat_band] * 0.05
    distance_bonus = min(min_dist(item, selected), 2000.0) / 2000.0
    land_bonus = float(item.get("land_fraction", 0.0)) * 0.2
    return distance_bonus + land_bonus - lc_penalty - continent_penalty - lat_penalty


def main():
    data = C.read_json(IN_JSON, {})
    candidates = data.get("candidates", [])
    if not candidates:
        raise SystemExit(f"No candidates found: {IN_JSON}")
    candidates = [c for c in candidates if c.get("dominant_landcover") in MAIN_LANDCOVERS]
    by_lc = defaultdict(list)
    for item in candidates:
        by_lc[item["dominant_landcover"]].append(item)
    base = TARGET // len(MAIN_LANDCOVERS)
    target_per_lc = {lc: base for lc in MAIN_LANDCOVERS}
    for lc in MAIN_LANDCOVERS[: TARGET - base * len(MAIN_LANDCOVERS)]:
        target_per_lc[lc] += 1

    selected = []
    used = set()
    lc_counts = Counter()
    continent_counts = Counter()
    lat_counts = Counter()
    rng = random.Random(20260706)
    for lc in MAIN_LANDCOVERS:
        pool = list(by_lc.get(lc, []))
        rng.shuffle(pool)
        while pool and lc_counts[lc] < target_per_lc[lc] and len(selected) < MAX_ATTEMPTS:
            pool.sort(key=lambda item: score_candidate(item, selected, lc_counts, continent_counts, lat_counts, target_per_lc), reverse=True)
            item = pool.pop(0)
            if item["mgrs_tile"] in used:
                continue
            used.add(item["mgrs_tile"])
            selected.append(item)
            lc_counts[lc] += 1
            continent_counts[item.get("continent", "Other")] += 1
            lat_counts[item.get("lat_band", "unknown")] += 1

    if len(selected) < TARGET:
        remaining = [c for c in candidates if c["mgrs_tile"] not in used]
        while remaining and len(selected) < TARGET:
            remaining.sort(key=lambda item: score_candidate(item, selected, lc_counts, continent_counts, lat_counts, target_per_lc), reverse=True)
            item = remaining.pop(0)
            used.add(item["mgrs_tile"])
            selected.append(item)
            lc_counts[item.get("dominant_landcover", "unknown")] += 1
            continent_counts[item.get("continent", "Other")] += 1
            lat_counts[item.get("lat_band", "unknown")] += 1

    selected = selected[:TARGET]
    out = [{**item, "selection_rank": idx + 1} for idx, item in enumerate(selected)]
    C.write_json(OUT_JSON, out)
    summary = {"selected": len(out), "landcover": dict(lc_counts), "continent": dict(continent_counts), "lat_band": dict(lat_counts), "path": str(OUT_JSON)}
    C.progress(f"00 selected benchmark MGRS: {summary}")
    print(summary)


if __name__ == "__main__":
    main()
