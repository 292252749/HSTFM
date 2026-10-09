#!/usr/bin/env python3
"""Build resumable CDSE candidate S2/S3 pair queues for v3.

For each configured MGRS tile, this script searches Sentinel-2 L2A products,
then searches same-day Sentinel-3 OLCI products using each S2 GeoFootprint. A
valid observation requires at least one same-day S3 GeoFootprint covering the S2
GeoFootprint. Up to CANDIDATE_PAIRS_PER_MGRS candidate t1/t2 pairs are stored.

This is a download-planning stage only. Final pair acceptance is still performed
from local downloaded files by 03_pair_local_s23.py and 04_build_lmdb_from_pairs.py.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import requests

import s23_common as C

cfg = C.cfg


def date_days(a, b):
    return C.interval_days(a, b)


def cdse_sleep():
    delay = float(getattr(cfg, "CDSE_REQUEST_SLEEP_SECONDS", 0.0) or 0.0)
    if delay > 0:
        time.sleep(delay)


def product_attr(product, *names):
    wanted = {name.lower() for name in names}
    for attr in product.get("Attributes", []) or []:
        name = str(attr.get("Name", "")).lower()
        if name in wanted:
            for key in ("Value", "DoubleAttribute", "IntegerAttribute", "StringAttribute"):
                if attr.get(key) is not None:
                    return attr[key]
    return None


def product_cloud(product):
    value = product_attr(product, "cloudCover", "cloudCoverPercentage", "cloudyPixelPercentage", "CLOUDY_PIXEL_PERCENTAGE")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def product_date(product, sensor):
    name = product.get("Name", "")
    try:
        return C.extract_s2_date(name) if sensor == "s2" else C.extract_s3_date(name)
    except Exception:
        return ""


def geojson_vertices(geometry_json):
    if not geometry_json:
        return []
    if isinstance(geometry_json, str):
        try:
            geometry_json = json.loads(geometry_json)
        except json.JSONDecodeError:
            return []
    geometry_type = geometry_json.get("type")
    coordinates = geometry_json.get("coordinates", [])
    vertices = []
    if geometry_type == "Polygon":
        for ring in coordinates[:1]:
            vertices.extend(point[:2] for point in ring)
    elif geometry_type == "MultiPolygon":
        for polygon in coordinates:
            for ring in polygon[:1]:
                vertices.extend(point[:2] for point in ring)
    return vertices


def point_in_ring(lon, lat, ring):
    inside = False
    if not ring:
        return False
    x1, y1 = ring[-1]
    for x2, y2 in ring:
        if ((y1 > lat) != (y2 > lat)) and (lon < (x2 - x1) * (lat - y1) / ((y2 - y1) or 1e-12) + x1):
            inside = not inside
        x1, y1 = x2, y2
    return inside


def geojson_contains_point(geometry_json, lon, lat):
    if not geometry_json:
        return False
    if isinstance(geometry_json, str):
        try:
            geometry_json = json.loads(geometry_json)
        except json.JSONDecodeError:
            return False
    geometry_type = geometry_json.get("type")
    coordinates = geometry_json.get("coordinates", [])
    if geometry_type == "Polygon":
        return any(point_in_ring(lon, lat, ring) for ring in coordinates[:1])
    if geometry_type == "MultiPolygon":
        return any(any(point_in_ring(lon, lat, ring) for ring in polygon[:1]) for polygon in coordinates)
    return False


def geojson_covers_geojson(cover_geometry, target_geometry):
    vertices = geojson_vertices(target_geometry)
    if not cover_geometry or not vertices:
        return False
    return all(geojson_contains_point(cover_geometry, lon, lat) for lon, lat in vertices)


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


def ring_area_xy(points):
    if len(points) < 3:
        return 0.0
    total = 0.0
    for idx in range(len(points)):
        x1, y1 = points[idx]
        x2, y2 = points[(idx + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def polygon_area_m2(rings, zone):
    if not rings:
        return 0.0
    areas = []
    for ring in rings:
        pts = [lonlat_to_utm(float(p[0]), float(p[1]), zone) for p in ring]
        areas.append(abs(ring_area_xy(pts)))
    if not areas:
        return 0.0
    return max(0.0, areas[0] - sum(areas[1:]))


def geojson_area_m2(geometry_json, mgrs_tile=""):
    if not geometry_json:
        return 0.0
    if isinstance(geometry_json, str):
        geometry_json = json.loads(geometry_json)
    bounds = C.bounds_from_geometry_like(geometry_json)
    if not bounds:
        return 0.0
    center_lon = (float(bounds[0]) + float(bounds[2])) / 2.0
    center_lat = (float(bounds[1]) + float(bounds[3])) / 2.0
    try:
        zone = int(str(mgrs_tile)[:2]) if mgrs_tile else utm_zone_for_lonlat(center_lon, center_lat)
    except Exception:
        zone = utm_zone_for_lonlat(center_lon, center_lat)
    geometry_type = geometry_json.get("type")
    coordinates = geometry_json.get("coordinates", [])
    if geometry_type == "Polygon":
        return polygon_area_m2(coordinates, zone)
    if geometry_type == "MultiPolygon":
        return sum(polygon_area_m2(polygon, zone) for polygon in coordinates)
    return 0.0


def s2_data_coverage_estimate(product, mgrs_tile):
    geom = product.get("GeoFootprint")
    if not geom:
        return ""
    try:
        area = geojson_area_m2(geom, mgrs_tile)
        full = float(getattr(cfg, "S2_FULL_TILE_SIZE_M", 109800.0)) ** 2
        if area <= 0 or full <= 0:
            return ""
        return max(0.0, min(1.0, area / full))
    except Exception:
        return ""

def geojson_to_wkt(geometry_json):
    if isinstance(geometry_json, str):
        geometry_json = json.loads(geometry_json)
    geometry_type = geometry_json.get("type")
    coordinates = geometry_json.get("coordinates", [])
    if geometry_type == "Polygon":
        rings = []
        for ring in coordinates:
            rings.append("(" + ",".join(f"{p[0]} {p[1]}" for p in ring) + ")")
        return "POLYGON (" + ",".join(rings) + ")"
    if geometry_type == "MultiPolygon":
        polygons = []
        for polygon in coordinates:
            rings = []
            for ring in polygon:
                rings.append("(" + ",".join(f"{p[0]} {p[1]}" for p in ring) + ")")
            polygons.append("(" + ",".join(rings) + ")")
        return "MULTIPOLYGON (" + ",".join(polygons) + ")"
    raise ValueError(f"Unsupported GeoJSON type for WKT: {geometry_type}")


def bbox_geojson(bounds):
    minx, miny, maxx, maxy = [float(v) for v in bounds]
    return {
        "type": "Polygon",
        "coordinates": [[
            [minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]
        ]],
    }


def union_bounds(geometries):
    bounds = [C.bounds_from_geometry_like(g) for g in geometries if g]
    bounds = [b for b in bounds if b]
    if not bounds:
        return None
    return [
        min(b[0] for b in bounds),
        min(b[1] for b in bounds),
        max(b[2] for b in bounds),
        max(b[3] for b in bounds),
    ]


def build_products_url(collection, name_contains, start, end, top, skip, extra_filters=None, expand_attributes=True):
    start_str = f"{start}T00:00:00.000Z"
    end_str = f"{end}T23:59:59.999Z"
    parts = name_contains if isinstance(name_contains, (list, tuple)) else [name_contains]
    filters = [
        f"Collection/Name eq '{collection}'",
        f"ContentDate/Start ge {start_str}",
        f"ContentDate/Start le {end_str}",
    ]
    filters.extend(f"contains(Name,'{part}')" for part in parts if part)
    if extra_filters:
        filters.extend(extra_filters)
    url = (
        f"{cfg.CDSE_ODATA}/Products?$filter="
        + quote(" and ".join(filters), safe=":/(),';")
        + f"&$top={top}&$skip={skip}&$orderby=ContentDate/Start asc"
    )
    if expand_attributes:
        url += "&$expand=Attributes"
    else:
        url += "&$select=Id,Name,Online,GeoFootprint,ContentDate"
    return url


def query_cache_path(url):
    cache_dir = getattr(cfg, "CDSE_QUERY_CACHE_DIR", None)
    if not cache_dir or not bool(getattr(cfg, "CDSE_QUERY_CACHE_ENABLED", False)):
        return None
    return Path(cache_dir) / (hashlib.sha1(url.encode("utf-8")).hexdigest() + ".json")


def request_json(url):
    cache_path = query_cache_path(url)
    if cache_path and cache_path.exists():
        try:
            return C.read_json(cache_path, {})
        except Exception:
            pass
    last = None
    retry_status = {429, 500, 502, 503, 504}
    for attempt in range(1, int(cfg.CDSE_MAX_RETRIES) + 1):
        try:
            cdse_sleep()
            response = requests.get(url, timeout=120)
            if response.status_code in retry_status:
                raise requests.HTTPError(f"HTTP {response.status_code}", response=response)
            response.raise_for_status()
            data = response.json()
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                C.write_json(cache_path, data)
            return data
        except Exception as exc:
            last = exc
            if attempt < int(cfg.CDSE_MAX_RETRIES):
                delay = float(cfg.CDSE_RETRY_SLEEP_SECONDS) * attempt
                C.progress(f"CDSE retry {attempt}/{cfg.CDSE_MAX_RETRIES} after {delay:.1f}s: {exc}")
                time.sleep(delay)
    raise RuntimeError(f"CDSE request failed: {last}\nURL: {url}")


def odata_search_paged(collection, name_contains, start, end, page_size, max_products=None, extra_filters=None):
    out = []
    skip = 0
    while True:
        top = int(page_size)
        if max_products is not None:
            top = min(top, max(0, int(max_products) - len(out)))
            if top <= 0:
                break
        url = build_products_url(collection, name_contains, start, end, top, skip, extra_filters=extra_filters, expand_attributes=(collection == cfg.CDSE_S2_COLLECTION))
        values = request_json(url).get("value", [])
        out.extend(values)
        if len(values) < top:
            break
        skip += top
    return out


def search_s2_for_mgrs(mgrs):
    extra = [f"contains(Name,'_T{mgrs}_')"]
    products = odata_search_paged(
        cfg.CDSE_S2_COLLECTION,
        cfg.CDSE_S2_NAME_CONTAINS,
        cfg.CDSE_SEARCH_START,
        cfg.CDSE_SEARCH_END,
        cfg.CDSE_S2_PAGE_SIZE,
        cfg.CDSE_MAX_S2_PRODUCTS_PER_MGRS,
        extra_filters=extra,
    )
    kept = []
    years = C.configured_allowed_years()
    for product in products:
        date = product_date(product, "s2")
        cloud = product_cloud(product)
        if not date or (years and date[:4] not in years):
            continue
        if cloud is None or cloud > cfg.MAX_S2_CLOUD_PERCENT:
            continue
        if not product.get("GeoFootprint"):
            continue
        coverage = s2_data_coverage_estimate(product, mgrs)
        product["S2DataCoverageEstimate"] = coverage
        try:
            if coverage == "" or float(coverage) < cfg.MIN_S2_DATA_PERCENTAGE:
                continue
        except Exception:
            continue
        kept.append(product)
    return kept


def search_same_day_s3_for_geometries(date, geometries):
    if not date or not geometries:
        return []
    bounds = union_bounds(geometries)
    if not bounds:
        return []
    wkt = geojson_to_wkt(bbox_geojson(bounds))
    extra = [f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')"]
    day = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    products = odata_search_paged(
        cfg.CDSE_S3_COLLECTION,
        cfg.CDSE_S3_NAME_CONTAINS,
        day,
        day,
        cfg.CDSE_S3_PAGE_SIZE,
        cfg.CDSE_MAX_S3_PRODUCTS_PER_DATE,
        extra_filters=extra,
    )
    return [s3 for s3 in products if product_date(s3, "s3") == date]


def search_same_day_s3_covering_s2(s2):
    date = product_date(s2, "s2")
    geom = s2.get("GeoFootprint")
    candidates = search_same_day_s3_for_geometries(date, [geom])
    out = []
    for s3 in candidates:
        if cfg.CDSE_REQUIRE_S3_FOOTPRINT_COVERS_S2 and not geojson_covers_geojson(s3.get("GeoFootprint"), geom):
            continue
        out.append(s3)
    return out


def product_record(product, sensor, extra=None):
    extra = extra or {}
    rec = {
        "id": product.get("Id"),
        "name": product.get("Name"),
        "date": product_date(product, sensor),
        "online": product.get("Online"),
        "cloud": product_cloud(product) if sensor == "s2" else None,
        "s2_data_coverage_estimate": product.get("S2DataCoverageEstimate", "") if sensor == "s2" else "",
        "geofootprint": product.get("GeoFootprint"),
        "content_start": (product.get("ContentDate") or {}).get("Start"),
        "content_end": (product.get("ContentDate") or {}).get("End"),
    }
    rec.update(extra)
    return rec



def search_pair_limit():
    return max(1, int(getattr(cfg, "CANDIDATE_PAIRS_PER_MGRS", 8)))


def final_pair_limit():
    return max(1, int(getattr(cfg, "PAIRS_PER_MGRS", 2)))


def min_candidate_pairs_per_mgrs():
    return max(1, int(getattr(cfg, "MIN_CANDIDATE_PAIRS_PER_MGRS", final_pair_limit())))


def interval_bin_name(days):
    bins = getattr(cfg, "PAIR_INTERVAL_BINS", [(5, 30), (31, 60), (61, 120), (121, 180), (181, 365)])
    try:
        days = int(days)
    except Exception:
        return "unknown"
    for lo, hi in bins:
        if int(lo) <= days <= int(hi):
            return f"{int(lo):03d}_{int(hi):03d}d"
    return "other"


def choose_candidate_pairs(observations):
    observations = sorted(observations, key=lambda x: (x["date"], x["cloud"], x["s2"]["name"]))
    primary = []
    fallback = []
    for i, t1 in enumerate(observations):
        for t2 in observations[i + 1 :]:
            days = date_days(t1["date"], t2["date"])
            if days < cfg.MIN_INTERVAL_DAYS:
                continue
            if days > cfg.FALLBACK_MAX_INTERVAL_DAYS:
                break
            row = {
                "t1": t1,
                "t2": t2,
                "interval_days": days,
                "start_month": t1["date"][:6],
                "score": float(t1["cloud"] or 0.0) + float(t2["cloud"] or 0.0) + abs(days - float(getattr(cfg, "PAIR_TARGET_INTERVAL_DAYS", 45))) / 10000.0,
            }
            if days <= cfg.MAX_INTERVAL_DAYS:
                primary.append(row)
            else:
                fallback.append(row)
            break
    selected = []
    used_months = set()
    for pool_name, pool in (("primary", primary), ("fallback", fallback)):
        for row in sorted(pool, key=lambda x: (x["start_month"] in used_months, x["score"], x["interval_days"])):
            if len(selected) >= search_pair_limit():
                break
            key = (row["t1"]["date"], row["t2"]["date"])
            if any((p["t1"]["date"], p["t2"]["date"]) == key for p in selected):
                continue
            item = dict(row)
            item["interval_pool"] = pool_name
            selected.append(item)
            used_months.add(row["start_month"])
        if len(selected) >= search_pair_limit():
            break
    return selected


def read_mgrs_records():
    path = Path(cfg.BENCHMARK_MGRS_JSON)
    if not path.exists():
        raise SystemExit(f"Missing benchmark MGRS file: {path}")
    data = C.read_json(path, [])
    out = []
    seen = set()
    for item in data:
        if isinstance(item, str):
            tile = item
            record = {"mgrs_tile": item}
        elif isinstance(item, dict):
            tile = item.get("mgrs_tile") or item.get("mgrs") or item.get("tile") or ""
            record = dict(item)
            record["mgrs_tile"] = tile
        else:
            continue
        tile = str(tile).strip().upper()
        if not tile or tile in seen:
            continue
        seen.add(tile)
        record["mgrs_tile"] = tile
        out.append(record)
    if not out:
        raise SystemExit(f"No MGRS tiles found in {path}")
    return out


def load_state():
    return C.read_json(cfg.CDSE_CANDIDATE_STATE_JSON, {"completed_mgrs": {}, "failed_mgrs": {}})


def save_state(state):
    C.write_json(cfg.CDSE_CANDIDATE_STATE_JSON, state)


def build_one_mgrs(mgrs_record):
    mgrs = mgrs_record["mgrs_tile"] if isinstance(mgrs_record, dict) else str(mgrs_record)
    s2_products = search_s2_for_mgrs(mgrs)
    observations = []
    rejected = Counter()
    s2_by_date = defaultdict(list)
    for s2 in s2_products:
        s2_by_date[product_date(s2, "s2")].append(s2)

    s3_by_date = {}
    for date, items in sorted(s2_by_date.items()):
        geoms = [item.get("GeoFootprint") for item in items if item.get("GeoFootprint")]
        s3_by_date[date] = search_same_day_s3_for_geometries(date, geoms)

    for s2 in s2_products:
        date = product_date(s2, "s2")
        geom = s2.get("GeoFootprint")
        s3_products = []
        for s3 in s3_by_date.get(date, []):
            if cfg.CDSE_REQUIRE_S3_FOOTPRINT_COVERS_S2 and not geojson_covers_geojson(s3.get("GeoFootprint"), geom):
                continue
            s3_products.append(s3)
        if not s3_products:
            rejected["no_same_day_s3_covering_s2_geofootprint"] += 1
            continue
        observations.append(
            {
                "date": product_date(s2, "s2"),
                "cloud": product_cloud(s2),
                "s2": product_record(s2, "s2", {"mgrs_tile": mgrs}),
                "candidate_s3": [product_record(s3, "s3") for s3 in s3_products[: int(getattr(cfg, "MAX_S3_CANDIDATES_PER_DATE", 1) or 1)]],
            }
        )
    pairs = choose_candidate_pairs(observations)
    out_pairs = []
    for idx, pair in enumerate(pairs, 1):
        out_pairs.append(
            {
                "rank": idx,
                "status": "pending",
                "mgrs_tile": mgrs,
                "t1_date": pair["t1"]["date"],
                "t2_date": pair["t2"]["date"],
                "interval_days": pair["interval_days"],
                "interval_bin": interval_bin_name(pair["interval_days"]),
                "interval_pool": pair["interval_pool"],
                "score": pair["score"],
                "t1": pair["t1"],
                "t2": pair["t2"],
            }
        )
    meta = dict(mgrs_record) if isinstance(mgrs_record, dict) else {"mgrs_tile": mgrs}
    record = {
        "mgrs_tile": mgrs,
        "status": "ok" if out_pairs else "no_candidate_pairs",
        "selected_for_download": False,
        "s2_low_cloud_count": len(s2_products),
        "valid_observation_count": len(observations),
        "raw_candidate_pair_count": len(out_pairs),
        "candidate_pair_count": 0,
        "s3_query_count": len(s3_by_date),
        "rejected_observations": dict(rejected),
        "raw_candidate_pairs": out_pairs,
        "candidate_pairs": [],
    }
    for key in ("selection_rank", "dominant_landcover", "continent", "lat_band", "center_lon", "center_lat", "land_fraction", "water_fraction", "nodata_fraction"):
        if key in meta:
            record[key] = meta[key]
    return record


def raw_pairs(record):
    pairs = record.get("raw_candidate_pairs")
    if pairs is None:
        pairs = record.get("candidate_pairs", [])
    return list(pairs or [])


def record_lonlat(record):
    if record.get("center_lon") not in (None, "") and record.get("center_lat") not in (None, ""):
        try:
            return float(record["center_lon"]), float(record["center_lat"])
        except Exception:
            pass
    center = record_center(record)
    if center:
        return center[0], center[1]
    return None


def record_distance_km(a, b):
    pa = record_lonlat(a)
    pb = record_lonlat(b)
    if not pa or not pb:
        return 0.0
    lon1, lat1 = pa
    lon2, lat2 = pb
    r = 6371.0088
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    dlat = lat2 - lat1
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(x), math.sqrt(max(0.0, 1 - x)))


def min_record_distance_km(record, selected):
    if not selected:
        return 2000.0
    return min(record_distance_km(record, item) for item in selected)


def record_balance_score(record, selected, lc_counts, continent_counts, lat_counts):
    lc = record.get("dominant_landcover", "unknown")
    continent = record.get("continent", "Other")
    lat_band = record.get("lat_band", "unknown")
    dist_bonus = min(min_record_distance_km(record, selected), 2500.0) / 2500.0
    raw_bonus = min(len(raw_pairs(record)), search_pair_limit()) / max(search_pair_limit(), 1) * 0.35
    land_bonus = float(record.get("land_fraction", 0.0) or 0.0) * 0.15
    penalty = lc_counts[lc] * 0.35 + continent_counts[continent] * 0.08 + lat_counts[lat_band] * 0.10
    return dist_bonus + raw_bonus + land_bonus - penalty


def select_balanced_records(records):
    target = int(getattr(cfg, "BENCHMARK_FINAL_MGRS", getattr(cfg, "BENCHMARK_TARGET_MGRS", len(records))))
    min_pairs = min_candidate_pairs_per_mgrs()
    eligible = [r for r in records if len(raw_pairs(r)) >= min_pairs]
    fallback = [r for r in records if 0 < len(raw_pairs(r)) < min_pairs]
    selected = []
    used = set()
    lc_counts = Counter()
    continent_counts = Counter()
    lat_counts = Counter()
    pool = list(eligible)
    while pool and len(selected) < target:
        pool.sort(key=lambda r: record_balance_score(r, selected, lc_counts, continent_counts, lat_counts), reverse=True)
        item = pool.pop(0)
        tile = item.get("mgrs_tile")
        if tile in used:
            continue
        used.add(tile)
        selected.append(item)
        lc_counts[item.get("dominant_landcover", "unknown")] += 1
        continent_counts[item.get("continent", "Other")] += 1
        lat_counts[item.get("lat_band", "unknown")] += 1
    if len(selected) < target:
        pool = [r for r in fallback if r.get("mgrs_tile") not in used]
        while pool and len(selected) < target:
            pool.sort(key=lambda r: record_balance_score(r, selected, lc_counts, continent_counts, lat_counts), reverse=True)
            item = pool.pop(0)
            used.add(item.get("mgrs_tile"))
            selected.append(item)
            lc_counts[item.get("dominant_landcover", "unknown")] += 1
            continent_counts[item.get("continent", "Other")] += 1
            lat_counts[item.get("lat_band", "unknown")] += 1
    return {r.get("mgrs_tile") for r in selected}


def pair_balance_score(pair, record, counts, used_dates):
    t1 = pair.get("t1_date", "")
    t2 = pair.get("t2_date", "")
    start_month = t1[4:6] if len(t1) >= 6 else "unknown"
    start_year_month = t1[:6] if len(t1) >= 6 else "unknown"
    interval_bin = pair.get("interval_bin") or interval_bin_name(pair.get("interval_days"))
    lc = record.get("dominant_landcover", "unknown")
    continent = record.get("continent", "Other")
    lat_band = record.get("lat_band", "unknown")
    score = float(pair.get("score", 0.0) or 0.0)
    score += counts["month"][start_month] * 0.65
    score += counts["year_month"][start_year_month] * 0.25
    score += counts["interval_bin"][interval_bin] * 0.85
    score += counts["landcover"][lc] * 0.04
    score += counts["continent"][continent] * 0.03
    score += counts["lat_band"][lat_band] * 0.04
    if t1 in used_dates or t2 in used_dates:
        score += 5.0
    if pair.get("interval_pool") == "fallback":
        score += 0.2
    return score


def select_balanced_pairs(records):
    selected_tiles = select_balanced_records(records)
    limit = final_pair_limit()
    counts = {
        "month": Counter(),
        "year_month": Counter(),
        "interval_bin": Counter(),
        "landcover": Counter(),
        "continent": Counter(),
        "lat_band": Counter(),
    }
    by_tile_selected_dates = defaultdict(set)
    for record in records:
        record["raw_candidate_pairs"] = raw_pairs(record)
        record["raw_candidate_pair_count"] = len(record["raw_candidate_pairs"])
        record["candidate_pairs"] = []
        record["candidate_pair_count"] = 0
        record["selected_for_download"] = False

    selected_records = [r for r in records if r.get("mgrs_tile") in selected_tiles]
    for round_idx in range(limit):
        for record in selected_records:
            chosen = record["candidate_pairs"]
            if len(chosen) > round_idx:
                continue
            used_ranks = {p.get("rank") for p in chosen}
            candidates = [p for p in record["raw_candidate_pairs"] if p.get("rank") not in used_ranks]
            if not candidates:
                continue
            tile = record.get("mgrs_tile")
            candidates.sort(key=lambda p: pair_balance_score(p, record, counts, by_tile_selected_dates[tile]))
            pair = dict(candidates[0])
            pair["download_selection_round"] = round_idx + 1
            pair["download_selection_reason"] = "global_month_interval_landcover_spatial_balance"
            record["candidate_pairs"].append(pair)
            by_tile_selected_dates[tile].update([pair.get("t1_date", ""), pair.get("t2_date", "")])
            t1 = pair.get("t1_date", "")
            start_month = t1[4:6] if len(t1) >= 6 else "unknown"
            start_year_month = t1[:6] if len(t1) >= 6 else "unknown"
            interval_bin = pair.get("interval_bin") or interval_bin_name(pair.get("interval_days"))
            counts["month"][start_month] += 1
            counts["year_month"][start_year_month] += 1
            counts["interval_bin"][interval_bin] += 1
            counts["landcover"][record.get("dominant_landcover", "unknown")] += 1
            counts["continent"][record.get("continent", "Other")] += 1
            counts["lat_band"][record.get("lat_band", "unknown")] += 1

    for record in records:
        record["candidate_pair_count"] = len(record.get("candidate_pairs", []))
        record["selected_for_download"] = record["candidate_pair_count"] > 0
        record["status"] = "ok" if record["candidate_pair_count"] else ("has_raw_candidates_not_selected" if record.get("raw_candidate_pair_count", 0) else "no_candidate_pairs")
    return counts


def pair_balance_summary(records):
    selected_pairs = []
    raw_pair_count = 0
    for record in records:
        raw_pair_count += len(raw_pairs(record))
        for pair in record.get("candidate_pairs", []) or []:
            selected_pairs.append((record, pair))
    return {
        "selected_pairs": len(selected_pairs),
        "raw_candidate_pairs": raw_pair_count,
        "selected_mgrs": sum(1 for r in records if r.get("selected_for_download")),
        "raw_mgrs_ge_1_pair": sum(1 for r in records if len(raw_pairs(r)) >= 1),
        "raw_mgrs_ge_min_pairs": sum(1 for r in records if len(raw_pairs(r)) >= min_candidate_pairs_per_mgrs()),
        "selected_start_month": dict(Counter((p.get("t1_date", "")[4:6] or "unknown") for _, p in selected_pairs)),
        "selected_start_year_month": dict(Counter((p.get("t1_date", "")[:6] or "unknown") for _, p in selected_pairs)),
        "selected_interval_bin": dict(Counter((p.get("interval_bin") or interval_bin_name(p.get("interval_days"))) for _, p in selected_pairs)),
        "selected_landcover": dict(Counter(r.get("dominant_landcover", "unknown") for r, _ in selected_pairs)),
        "selected_continent": dict(Counter(r.get("continent", "Other") for r, _ in selected_pairs)),
        "selected_lat_band": dict(Counter(r.get("lat_band", "unknown") for r, _ in selected_pairs)),
    }


def record_center(record):
    pairs = record.get("candidate_pairs", [])
    geoms = []
    for pair in pairs:
        for side in ("t1", "t2"):
            geom = pair.get(side, {}).get("s2", {}).get("geofootprint")
            if geom:
                geoms.append(geom)
    if not geoms:
        return None
    bounds_list = [C.bounds_from_geometry_like(g) for g in geoms]
    bounds_list = [b for b in bounds_list if b]
    if not bounds_list:
        return None
    minx = min(b[0] for b in bounds_list)
    miny = min(b[1] for b in bounds_list)
    maxx = max(b[2] for b in bounds_list)
    maxy = max(b[3] for b in bounds_list)
    return (float(minx + maxx) / 2.0, float(miny + maxy) / 2.0, [minx, miny, maxx, maxy])


def write_spatial_outputs(records):
    rows = []
    for record in records:
        center = record_center(record)
        if not center:
            continue
        lon, lat, bounds = center
        first_pair = (record.get("candidate_pairs") or [{}])[0]
        rows.append({
            "mgrs_tile": record.get("mgrs_tile", ""),
            "lon": lon,
            "lat": lat,
            "candidate_pair_count": record.get("candidate_pair_count", 0),
            "valid_observation_count": record.get("valid_observation_count", 0),
            "first_t1_date": first_pair.get("t1_date", ""),
            "first_t2_date": first_pair.get("t2_date", ""),
            "first_interval_days": first_pair.get("interval_days", ""),
            "bbox_wgs84": json.dumps(bounds),
        })
    csv_path = cfg.PIPELINE_OUT / "cdse_candidate_spatial.csv"
    geojson_path = cfg.PIPELINE_OUT / "cdse_candidate_spatial.geojson"
    svg_path = cfg.PIPELINE_OUT / "cdse_candidate_spatial.svg"
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as fd:
            writer = csv.DictWriter(fd, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    features = []
    for row in rows:
        props = dict(row)
        lon = props.pop("lon")
        lat = props.pop("lat")
        features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]}, "properties": props})
    C.write_json(geojson_path, {"type": "FeatureCollection", "features": features})
    write_svg_map(svg_path, rows)
    return {"csv": str(csv_path), "geojson": str(geojson_path), "svg": str(svg_path), "points": len(rows)}


def write_svg_map(path, rows):
    width, height = 1200, 620
    def x(lon):
        return (float(lon) + 180.0) / 360.0 * width
    def y(lat):
        return (90.0 - float(lat)) / 180.0 * height
    circles = []
    for row in rows:
        count = int(row.get("candidate_pair_count", 0) or 0)
        color = "#1f77b4" if count > 0 else "#d62728"
        radius = 3.5 + min(count, 4) * 1.2
        circles.append(f'<circle cx="{x(row["lon"]):.2f}" cy="{y(row["lat"]):.2f}" r="{radius:.1f}" fill="{color}" fill-opacity="0.72"><title>{row["mgrs_tile"]} pairs={count}</title></circle>')
    graticule = []
    for lon in range(-180, 181, 30):
        graticule.append(f'<line x1="{x(lon):.2f}" y1="0" x2="{x(lon):.2f}" y2="{height}" stroke="#ddd" stroke-width="1"/>')
    for lat in range(-60, 91, 30):
        graticule.append(f'<line x1="0" y1="{y(lat):.2f}" x2="{width}" y2="{y(lat):.2f}" stroke="#ddd" stroke-width="1"/>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#f8fafc"/>
<rect x="0" y="0" width="{width}" height="{height}" fill="none" stroke="#94a3b8" stroke-width="1"/>
{''.join(graticule)}
<text x="20" y="32" font-family="Arial" font-size="20" fill="#0f172a">S23 v3 CDSE candidate MGRS global distribution</text>
<text x="20" y="56" font-family="Arial" font-size="13" fill="#475569">blue: MGRS with candidate pairs; radius scales with number of candidate pairs</text>
{''.join(circles)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def main():
    cfg.PIPELINE_OUT.mkdir(parents=True, exist_ok=True)
    mgrs_records = read_mgrs_records()
    mgrs_tiles = [item["mgrs_tile"] for item in mgrs_records]
    mgrs_record_by_tile = {item["mgrs_tile"]: item for item in mgrs_records}
    state = load_state()
    done = set(state.get("completed_mgrs", {}))
    failed = set(state.get("failed_mgrs", {}))
    # Failed entries are treated as transient. Re-running 01 retries them while
    # still preserving the previous error message for diagnostics until success.
    pending = [record for record in mgrs_records if record["mgrs_tile"] not in done]
    retry_failed = [record["mgrs_tile"] for record in pending if record["mgrs_tile"] in failed]
    C.progress(
        f"01 v3 CDSE candidate queue: total={len(mgrs_tiles)}, pending={len(pending)}, "
        f"done={len(done)}, retry_failed={len(retry_failed)}, failed_recorded={len(failed)}"
    )

    workers = max(1, int(getattr(cfg, "CDSE_SEARCH_WORKERS", 1)))
    processed = 0
    if workers == 1:
        for record in pending:
            tile = record["mgrs_tile"]
            try:
                result = build_one_mgrs(record)
                state.setdefault("completed_mgrs", {})[tile] = result
                state.setdefault("failed_mgrs", {}).pop(tile, None)
                processed += 1
                save_state(state)
                C.progress(f"01 saved {tile}: observations={result['valid_observation_count']}, raw_pairs={result['raw_candidate_pair_count']} ({processed}/{len(pending)})")
            except Exception as exc:
                state.setdefault("failed_mgrs", {})[tile] = {"error": str(exc), "time": datetime.now().isoformat(timespec="seconds")}
                save_state(state)
                C.progress(f"01 failed {tile}: {exc}")
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(build_one_mgrs, record): record["mgrs_tile"] for record in pending}
            for future in as_completed(futures):
                tile = futures[future]
                try:
                    result = future.result()
                    state.setdefault("completed_mgrs", {})[tile] = result
                    state.setdefault("failed_mgrs", {}).pop(tile, None)
                    processed += 1
                    save_state(state)
                    C.progress(f"01 saved {tile}: observations={result['valid_observation_count']}, raw_pairs={result['raw_candidate_pair_count']} ({processed}/{len(pending)})")
                except Exception as exc:
                    state.setdefault("failed_mgrs", {})[tile] = {"error": str(exc), "time": datetime.now().isoformat(timespec="seconds")}
                    save_state(state)
                    C.progress(f"01 failed {tile}: {exc}")

    records = []
    for tile in mgrs_tiles:
        record = state.get("completed_mgrs", {}).get(tile)
        if not record:
            continue
        meta = mgrs_record_by_tile.get(tile, {})
        for key in ("selection_rank", "dominant_landcover", "continent", "lat_band", "center_lon", "center_lat", "land_fraction", "water_fraction", "nodata_fraction"):
            if key in meta and key not in record:
                record[key] = meta[key]
        records.append(record)
    select_balanced_pairs(records)
    spatial_outputs = write_spatial_outputs(records)
    balance = pair_balance_summary(records)
    output = {
        "schema": "s23_cdse_candidate_queue_v3",
        "dataset_role": cfg.DATASET_ROLE,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "settings": {
            "max_s2_cloud_percent": cfg.MAX_S2_CLOUD_PERCENT,
            "min_s2_data_percentage": cfg.MIN_S2_DATA_PERCENTAGE,
            "s2_data_coverage_method": "CDSE S2 GeoFootprint projected area divided by full 20m tile area",
            "pair_strategy": cfg.PAIR_STRATEGY,
            "search_candidate_pairs_per_mgrs": search_pair_limit(),
            "download_pairs_per_mgrs": final_pair_limit(),
            "min_candidate_pairs_per_mgrs": min_candidate_pairs_per_mgrs(),
            "benchmark_final_mgrs": int(getattr(cfg, "BENCHMARK_FINAL_MGRS", 0) or 0),
            "pair_target_interval_days": float(getattr(cfg, "PAIR_TARGET_INTERVAL_DAYS", 45)),
            "pair_interval_bins": list(getattr(cfg, "PAIR_INTERVAL_BINS", [])),
            "min_interval_days": cfg.MIN_INTERVAL_DAYS,
            "max_interval_days": cfg.MAX_INTERVAL_DAYS,
            "fallback_max_interval_days": cfg.FALLBACK_MAX_INTERVAL_DAYS,
            "require_s3_footprint_covers_s2": cfg.CDSE_REQUIRE_S3_FOOTPRINT_COVERS_S2,
        },
        "mgrs_records": records,
        "spatial_outputs": spatial_outputs,
        "balance_summary": balance,
        "summary": {
            "mgrs_total": len(mgrs_tiles),
            "mgrs_completed": len(records),
            "mgrs_with_raw_candidate_pairs": balance["raw_mgrs_ge_1_pair"],
            "mgrs_with_min_raw_candidate_pairs": balance["raw_mgrs_ge_min_pairs"],
            "mgrs_selected_for_download": balance["selected_mgrs"],
            "raw_candidate_pairs": balance["raw_candidate_pairs"],
            "candidate_pairs": balance["selected_pairs"],
        },
    }
    C.write_json(cfg.CDSE_CANDIDATE_QUEUE_JSON, output)
    C.progress(f"01 finish: wrote {cfg.CDSE_CANDIDATE_QUEUE_JSON}")
    print(output["summary"])
    print({"selected_start_month": balance["selected_start_month"], "selected_interval_bin": balance["selected_interval_bin"]})


if __name__ == "__main__":
    main()
