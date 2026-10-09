#!/usr/bin/env python3
"""Build global land MGRS candidates for v3 benchmark selection.

The script samples a global lon/lat grid, converts points to Sentinel-2 MGRS
100 km tile IDs, removes ocean-dominated candidates using FROMGLC, and writes
metadata used by 00_select_balanced_benchmark_mgrs.py.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling

import s23_common as C

cfg = C.cfg

MGRS_LAT_BANDS = "CDEFGHJKLMNPQRSTUVWX"
MGRS_COLUMN_SETS = ("ABCDEFGH", "JKLMNPQR", "STUVWXYZ")
MGRS_ROW_ODD = "ABCDEFGHJKLMNPQRSTUV"
MGRS_ROW_EVEN = "FGHJKLMNPQRSTUVABCDE"

OUT_JSON = cfg.PIPELINE_OUT / "global_mgrs_candidates.json"
GRID_STEP_DEG = float(getattr(cfg, "GLOBAL_GRID_STEP_DEG", 2.0))
MIN_LAND_FRACTION = float(getattr(cfg, "GLOBAL_CANDIDATE_MIN_LAND_FRACTION", 0.70))
MAX_WATER_FRACTION = float(getattr(cfg, "GLOBAL_CANDIDATE_MAX_WATER_FRACTION", 0.20))
LC_WINDOW_DEG = float(getattr(cfg, "GLOBAL_CANDIDATE_LC_WINDOW_DEG", 0.6))


def utm_zone_for_lonlat(lon, lat):
    zone = int((lon + 180) // 6) + 1
    if 56 <= lat < 64 and 3 <= lon < 12:
        zone = 32
    if 72 <= lat < 84:
        if 0 <= lon < 9:
            zone = 31
        elif 9 <= lon < 21:
            zone = 33
        elif 21 <= lon < 33:
            zone = 35
        elif 33 <= lon < 42:
            zone = 37
    return max(1, min(zone, 60))


def mgrs_lat_band(lat):
    if lat < -80 or lat > 84:
        raise ValueError(f"MGRS latitude out of range: {lat}")
    index = int((lat + 80) // 8)
    return MGRS_LAT_BANDS[min(index, len(MGRS_LAT_BANDS) - 1)]


def lonlat_to_utm(lon, lat, zone):
    a = 6378137.0
    f = 1 / 298.257223563
    k0 = 0.9996
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    lon0 = math.radians((zone - 1) * 6 - 180 + 3)
    n = a / math.sqrt(1 - e2 * math.sin(lat_rad) ** 2)
    t = math.tan(lat_rad) ** 2
    c = ep2 * math.cos(lat_rad) ** 2
    aa = math.cos(lat_rad) * (lon_rad - lon0)
    m = a * ((1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * lat_rad - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * math.sin(2 * lat_rad) + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * math.sin(4 * lat_rad) - (35 * e2**3 / 3072) * math.sin(6 * lat_rad))
    easting = k0 * n * (aa + (1 - t + c) * aa**3 / 6 + (5 - 18 * t + t**2 + 72 * c - 58 * ep2) * aa**5 / 120) + 500000
    northing = k0 * (m + n * math.tan(lat_rad) * (aa**2 / 2 + (5 - t + 9 * c + 4 * c**2) * aa**4 / 24 + (61 - 58 * t + t**2 + 600 * c - 330 * ep2) * aa**6 / 720))
    if lat < 0:
        northing += 10000000
    return easting, northing


def lonlat_to_mgrs_tile(lon, lat):
    zone = utm_zone_for_lonlat(lon, lat)
    band = mgrs_lat_band(lat)
    easting, northing = lonlat_to_utm(lon, lat, zone)
    column_set = MGRS_COLUMN_SETS[(zone - 1) % 3]
    row_set = MGRS_ROW_ODD if zone % 2 else MGRS_ROW_EVEN
    column_index = int(easting // 100000) - 1
    row_index = int(northing // 100000) % 20
    if column_index < 0 or column_index >= len(column_set):
        raise ValueError("MGRS column out of range")
    return f"{zone:02d}{band}{column_set[column_index]}{row_set[row_index]}"


def continent_from_lonlat(lon, lat):
    if lat < -55:
        return "Antarctica"
    if -170 <= lon <= -30 and lat >= 5:
        return "North America"
    if -90 <= lon <= -30 and lat < 15:
        return "South America"
    if -25 <= lon <= 55 and -35 <= lat <= 38:
        return "Africa"
    if -25 <= lon <= 45 and lat > 35:
        return "Europe"
    if 45 < lon <= 180 and lat > -10:
        return "Asia"
    if 110 <= lon <= 180 and lat <= -10:
        return "Oceania"
    return "Other"


def lat_band_name(lat):
    alat = abs(lat)
    if alat < 23.5:
        return "tropical"
    if alat < 50:
        return "mid_latitude"
    return "high_latitude"


def load_lc_tiles():
    tiles = []
    for path in sorted(Path(cfg.FROMGLC_DIR).glob("*.tif")):
        try:
            ds = rasterio.open(path)
            tiles.append({"path": str(path), "bounds": tuple(ds.bounds), "crs": ds.crs})
            ds.close()
        except Exception:
            continue
    if not tiles:
        raise SystemExit(f"No FROMGLC tif files found under {cfg.FROMGLC_DIR}")
    return tiles


def lc_stats_at(lon, lat, tiles):
    bounds = (lon - LC_WINDOW_DEG / 2, lat - LC_WINDOW_DEG / 2, lon + LC_WINDOW_DEG / 2, lat + LC_WINDOW_DEG / 2)
    counts = Counter()
    for tile in tiles:
        minx, miny, maxx, maxy = tile["bounds"]
        if maxx <= bounds[0] or minx >= bounds[2] or maxy <= bounds[1] or miny >= bounds[3]:
            continue
        with rasterio.open(tile["path"]) as ds:
            arr = ds.read(1, window=from_bounds(*bounds, transform=ds.transform), out_shape=(128, 128), boundless=True, fill_value=0, resampling=Resampling.nearest)
        for value, n in zip(*np.unique(arr, return_counts=True)):
            counts[int(value)] += int(n)
    total = sum(counts.values())
    if not total:
        return None
    water = sum(counts.get(v, 0) for v in C.FROMGLC_WATER_CLASSES)
    nodata = sum(counts.get(v, 0) for v in C.FROMGLC_NODATA_VALUES)
    valid = max(total - nodata, 1)
    named = Counter({C.FROMGLC_CLASSES.get(code, f"class_{code}"): count for code, count in counts.items() if code in C.FROMGLC_CLASSES})
    major = named.most_common(1)[0][0] if named else "unknown"
    return {"land_fraction": float((valid - water) / valid), "water_fraction": float(water / valid), "nodata_fraction": float(nodata / total), "dominant_landcover": major, "landcover_counts": dict(named)}


def main():
    cfg.PIPELINE_OUT.mkdir(parents=True, exist_ok=True)
    tiles = load_lc_tiles()
    candidates = {}
    lats = np.arange(-58, 76 + 1e-6, GRID_STEP_DEG)
    lons = np.arange(-180, 180, GRID_STEP_DEG)
    total = len(lats) * len(lons)
    seen = 0
    for lat in lats:
        for lon in lons:
            seen += 1
            try:
                mgrs = lonlat_to_mgrs_tile(float(lon), float(lat))
            except Exception:
                continue
            if mgrs in candidates:
                continue
            stats = lc_stats_at(float(lon), float(lat), tiles)
            if not stats:
                continue
            if stats["land_fraction"] < MIN_LAND_FRACTION or stats["water_fraction"] > MAX_WATER_FRACTION:
                continue
            candidates[mgrs] = {"mgrs_tile": mgrs, "center_lon": float(lon), "center_lat": float(lat), "continent": continent_from_lonlat(float(lon), float(lat)), "lat_band": lat_band_name(float(lat)), **stats}
            if len(candidates) % 100 == 0:
                C.progress(f"00 global candidates: {len(candidates)} candidates after {seen}/{total} grid points")
    out = {"schema": "s23_global_mgrs_candidates_v3", "grid_step_deg": GRID_STEP_DEG, "count": len(candidates), "candidates": list(candidates.values())}
    C.write_json(OUT_JSON, out)
    C.progress(f"00 wrote {len(candidates)} candidates: {OUT_JSON}")
    print({"candidates": len(candidates), "path": str(OUT_JSON)})


if __name__ == "__main__":
    main()
