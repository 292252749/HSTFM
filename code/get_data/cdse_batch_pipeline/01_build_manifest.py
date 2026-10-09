"""
Build a CDSE Sentinel-2/Sentinel-3 download manifest.

Edit the CONFIG section below to change behavior. This script intentionally
does not use command-line arguments.
"""

import csv
import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cdse_one_location import (
    CDSE_ODATA,
    build_point_wkt,
    extract_s2_date,
    extract_s3_date,
    group_candidates,
    product_cloud,
)


# =========================
# CONFIG: edit values here.
# =========================
OUT_DIR = Path("cdse_batch_pipeline_out")
MANIFEST_PATH = OUT_DIR / "manifest.json"
SUMMARY_CSV_PATH = OUT_DIR / "manifest_summary.csv"
STATE_PATH = OUT_DIR / "manifest_state.json"
TARGET_GROUPS = 200
TARGET_GROUPS_PER_LOCATION = 1
SEARCH_START = "2023-01-01"
SEARCH_END = "2026-12-31"
CLOUD_LIMIT = 0.5
INTERVAL_MONTH_MIN = 0
INTERVAL_MONTH_MAX = 24
REQUEST_SLEEP_SECONDS = 0.25
SEARCH_MAX_RETRIES = 4
SEARCH_RETRY_SLEEP_SECONDS = 10
MAX_SEARCH_WORKERS = 4
S2_PAGE_SIZE = 100
S3_PAGE_SIZE = 100
S2_MAX_PRODUCTS = None
S3_MAX_PRODUCTS_PER_DATE = 20
MAX_S3_DATES_TO_VERIFY_PER_LOCATION = 32
STOP_S3_AFTER_GROUP_CANDIDATES = True
MIN_GROUP_CANDIDATES_BEFORE_STOP = 1
EARLY_STOP_MAX_INTERVAL_ERROR_MONTHS = 1
REQUIRE_S3_FOOTPRINT_COVERS_S2 = True
SEARCH_STRATEGY_VERSION = "target_interval_cloud_s3_covers_s2_v3"
TARGET_INTERVALS = list(range(INTERVAL_MONTH_MIN, INTERVAL_MONTH_MAX + 1))
TARGET_START_YEARS = ["2023", "2024", "2025", "2026"]
TILE_CANDIDATES_PER_TARGET = 40
MIN_TILE_DISTANCE_KM = 80
# Independent land anchor strata. Counts sum to TARGET_GROUPS, so every
# generated candidate point is a distinct anchor rather than an offset copy.
ANCHOR_STRATA = [
    {"prefix": "na_alaska_tundra", "region": "North America", "landform": "tundra", "count": 5, "lon_range": (-164, -142), "lat_range": (61, 68)},
    {"prefix": "na_canada_boreal", "region": "North America", "landform": "boreal_forest", "count": 6, "lon_range": (-123, -86), "lat_range": (50, 59)},
    {"prefix": "na_rockies", "region": "North America", "landform": "mountain_forest", "count": 5, "lon_range": (-119, -107), "lat_range": (39, 52)},
    {"prefix": "na_great_plains", "region": "North America", "landform": "grassland", "count": 5, "lon_range": (-105, -94), "lat_range": (34, 48)},
    {"prefix": "na_west_cropland", "region": "North America", "landform": "cropland_valley", "count": 4, "lon_range": (-122, -116), "lat_range": (35, 42)},
    {"prefix": "na_mexico_arid", "region": "North America", "landform": "semiarid_plateau", "count": 5, "lon_range": (-108, -99), "lat_range": (20, 29)},
    {"prefix": "sa_amazon", "region": "South America", "landform": "tropical_forest", "count": 6, "lon_range": (-72, -51), "lat_range": (-10, 3)},
    {"prefix": "sa_cerrado", "region": "South America", "landform": "savanna", "count": 5, "lon_range": (-55, -42), "lat_range": (-18, -8)},
    {"prefix": "sa_andes", "region": "South America", "landform": "high_mountain", "count": 5, "lon_range": (-74, -68), "lat_range": (-35, -8)},
    {"prefix": "sa_pampas", "region": "South America", "landform": "temperate_grassland", "count": 5, "lon_range": (-64, -55), "lat_range": (-38, -30)},
    {"prefix": "sa_patagonia", "region": "South America", "landform": "cold_steppe", "count": 5, "lon_range": (-72, -66), "lat_range": (-49, -40)},
    {"prefix": "sa_altiplano", "region": "South America", "landform": "high_plateau", "count": 4, "lon_range": (-70, -64), "lat_range": (-23, -16)},
    {"prefix": "af_sahara", "region": "Africa", "landform": "hot_desert", "count": 6, "lon_range": (-8, 29), "lat_range": (20, 29)},
    {"prefix": "af_sahel", "region": "Africa", "landform": "dry_savanna", "count": 5, "lon_range": (-10, 32), "lat_range": (11, 16)},
    {"prefix": "af_congo", "region": "Africa", "landform": "tropical_forest", "count": 5, "lon_range": (13, 29), "lat_range": (-5, 4)},
    {"prefix": "af_east_rift", "region": "Africa", "landform": "rift_highland", "count": 5, "lon_range": (33, 42), "lat_range": (-5, 7)},
    {"prefix": "af_south_savanna", "region": "Africa", "landform": "savanna", "count": 5, "lon_range": (18, 32), "lat_range": (-24, -14)},
    {"prefix": "af_madagascar", "region": "Africa", "landform": "dry_forest", "count": 4, "lon_range": (43, 48), "lat_range": (-23, -14)},
    {"prefix": "eu_iberia", "region": "Europe", "landform": "dry_cropland", "count": 5, "lon_range": (-8, 1), "lat_range": (37, 42)},
    {"prefix": "eu_alps", "region": "Europe", "landform": "alpine", "count": 4, "lon_range": (6, 15), "lat_range": (45, 48)},
    {"prefix": "eu_baltic", "region": "Europe", "landform": "boreal_forest", "count": 5, "lon_range": (20, 31), "lat_range": (56, 61)},
    {"prefix": "eu_balkan", "region": "Europe", "landform": "mixed_cropland_forest", "count": 4, "lon_range": (17, 27), "lat_range": (41, 46)},
    {"prefix": "eu_scandinavia", "region": "Europe", "landform": "taiga", "count": 2, "lon_range": (15, 27), "lat_range": (62, 67)},
    {"prefix": "me_arabia", "region": "Middle East", "landform": "hot_desert", "count": 5, "lon_range": (39, 51), "lat_range": (19, 28)},
    {"prefix": "me_anatolia", "region": "Middle East", "landform": "plateau_cropland", "count": 3, "lon_range": (30, 41), "lat_range": (37, 41)},
    {"prefix": "me_iran", "region": "Middle East", "landform": "arid_plateau", "count": 2, "lon_range": (49, 58), "lat_range": (29, 35)},
    {"prefix": "ca_kazakh_steppe", "region": "Central Asia", "landform": "steppe", "count": 5, "lon_range": (55, 75), "lat_range": (45, 53)},
    {"prefix": "ca_tien_shan", "region": "Central Asia", "landform": "mountain", "count": 3, "lon_range": (69, 79), "lat_range": (40, 44)},
    {"prefix": "ca_uzbek_desert", "region": "Central Asia", "landform": "cold_desert", "count": 2, "lon_range": (58, 67), "lat_range": (39, 44)},
    {"prefix": "sas_deccan", "region": "South Asia", "landform": "tropical_cropland", "count": 4, "lon_range": (74, 80), "lat_range": (14, 20)},
    {"prefix": "sas_thar", "region": "South Asia", "landform": "arid_shrubland", "count": 3, "lon_range": (69, 74), "lat_range": (24, 29)},
    {"prefix": "sas_sri_lanka", "region": "South Asia", "landform": "tropical_mosaic", "count": 3, "lon_range": (80, 81.7), "lat_range": (6.5, 8.8)},
    {"prefix": "sea_thailand", "region": "Southeast Asia", "landform": "rice_cropland", "count": 4, "lon_range": (99, 104), "lat_range": (13, 18)},
    {"prefix": "sea_borneo", "region": "Southeast Asia", "landform": "tropical_forest", "count": 3, "lon_range": (111, 117), "lat_range": (-2, 4)},
    {"prefix": "sea_cambodia", "region": "Southeast Asia", "landform": "tropical_savanna", "count": 3, "lon_range": (103, 106), "lat_range": (11, 14)},
    {"prefix": "nas_siberia", "region": "North Asia", "landform": "taiga", "count": 5, "lon_range": (82, 116), "lat_range": (56, 64)},
    {"prefix": "nas_far_east", "region": "North Asia", "landform": "temperate_forest", "count": 5, "lon_range": (131, 142), "lat_range": (45, 53)},
    {"prefix": "ea_mongolia", "region": "East Asia", "landform": "cold_steppe", "count": 4, "lon_range": (96, 114), "lat_range": (43, 50)},
    {"prefix": "ea_japan", "region": "East Asia", "landform": "cool_cropland_forest", "count": 3, "lon_range": (138, 145), "lat_range": (37, 44)},
    {"prefix": "ea_korea", "region": "East Asia", "landform": "temperate_mixed", "count": 3, "lon_range": (126, 129), "lat_range": (35, 38)},
    {"prefix": "aus_wheatbelt", "region": "Australia", "landform": "dry_cropland", "count": 5, "lon_range": (116, 122), "lat_range": (-34, -29)},
    {"prefix": "aus_outback", "region": "Australia", "landform": "arid_desert", "count": 5, "lon_range": (124, 139), "lat_range": (-27, -20)},
    {"prefix": "aus_savanna", "region": "Australia", "landform": "tropical_savanna", "count": 5, "lon_range": (129, 143), "lat_range": (-18, -12)},
    {"prefix": "oc_new_zealand", "region": "Oceania", "landform": "alpine", "count": 5, "lon_range": (168, 173), "lat_range": (-45, -41)},
    {"prefix": "arc_greenland", "region": "Arctic", "landform": "ice_edge_tundra", "count": 5, "lon_range": (-53, -42), "lat_range": (64, 70)},
    {"prefix": "arc_iceland", "region": "Arctic", "landform": "subarctic_mosaic", "count": 5, "lon_range": (-22, -15), "lat_range": (64, 66)},
]


def is_in_china_bbox(lon, lat):
    boxes = [
        (73.0, 27.0, 96.0, 49.5),
        (96.0, 30.0, 124.5, 43.5),
        (104.0, 18.0, 123.5, 30.5),
        (118.0, 40.0, 135.2, 54.0),
    ]
    return any(lon_min <= lon <= lon_max and lat_min <= lat <= lat_max for lon_min, lat_min, lon_max, lat_max in boxes)


MGRS_LAT_BANDS = "CDEFGHJKLMNPQRSTUVWX"
MGRS_COLUMN_SETS = ("ABCDEFGH", "JKLMNPQR", "STUVWXYZ")
MGRS_ROW_ODD = "ABCDEFGHJKLMNPQRSTUV"
MGRS_ROW_EVEN = "FGHJKLMNPQRSTUVABCDE"


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
    m = a * (
        (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * lat_rad
        - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * math.sin(2 * lat_rad)
        + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * math.sin(4 * lat_rad)
        - (35 * e2**3 / 3072) * math.sin(6 * lat_rad)
    )
    easting = k0 * n * (
        aa
        + (1 - t + c) * aa**3 / 6
        + (5 - 18 * t + t**2 + 72 * c - 58 * ep2) * aa**5 / 120
    ) + 500000
    northing = k0 * (
        m
        + n
        * math.tan(lat_rad)
        * (
            aa**2 / 2
            + (5 - t + 9 * c + 4 * c**2) * aa**4 / 24
            + (61 - 58 * t + t**2 + 600 * c - 330 * ep2) * aa**6 / 720
        )
    )
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
        raise ValueError(f"Cannot derive MGRS column for lon={lon}, lat={lat}, easting={easting}")
    return f"{zone:02d}{band}{column_set[column_index]}{row_set[row_index]}"


def haversine_km(lon1, lat1, lon2, lat2):
    radius_km = 6371.0088
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = lat2_rad - lat1_rad
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    return radius_km * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def min_distance_to_selected(candidate, selected):
    if not selected:
        return float("inf")
    return min(haversine_km(candidate["lon"], candidate["lat"], item["lon"], item["lat"]) for item in selected)


def target_time_for_index(point_index):
    return {
        "target_interval_months": TARGET_INTERVALS[point_index % len(TARGET_INTERVALS)],
        "target_start_year": TARGET_START_YEARS[
            ((point_index // len(TARGET_INTERVALS)) + point_index) % len(TARGET_START_YEARS)
        ],
    }


def build_tile_candidate_pool(stratum):
    lon_min, lon_max = stratum["lon_range"]
    lat_min, lat_max = stratum["lat_range"]
    target_pool_size = max(stratum["count"] * TILE_CANDIDATES_PER_TARGET, stratum["count"])
    max_attempts = target_pool_size * 12
    candidates = []
    used_tiles = set()
    for attempt in range(1, max_attempts + 1):
        lon_fraction = (attempt * 0.61803398875 + 0.173) % 1.0
        lat_fraction = (attempt * 0.41421356237 + 0.317) % 1.0
        lon = round(lon_min + (lon_max - lon_min) * lon_fraction, 4)
        lat = round(lat_min + (lat_max - lat_min) * lat_fraction, 4)
        if is_in_china_bbox(lon, lat):
            continue
        try:
            tile_id = lonlat_to_mgrs_tile(lon, lat)
        except ValueError:
            continue
        if tile_id in used_tiles:
            continue
        used_tiles.add(tile_id)
        candidates.append(
            {
                "id": tile_id,
                "tile_id": tile_id,
                "mgrs_tile_id": tile_id,
                "region": stratum["region"],
                "landform": stratum["landform"],
                "stratum": stratum["prefix"],
                "lon": lon,
                "lat": lat,
                "candidate_pool_size": 0,
                "selection_note": "stratified_farthest_candidate",
            }
        )
        if len(candidates) >= target_pool_size:
            break
    for candidate in candidates:
        candidate["candidate_pool_size"] = len(candidates)
    return candidates


def build_locations():
    locations = []
    used_tiles = set()
    for stratum in ANCHOR_STRATA:
        candidate_pool = [candidate for candidate in build_tile_candidate_pool(stratum) if candidate["tile_id"] not in used_tiles]
        if len(candidate_pool) < stratum["count"]:
            raise RuntimeError(
                f"Stratum {stratum['prefix']} only has {len(candidate_pool)} unique MGRS tile candidates; "
                f"need {stratum['count']}. Increase TILE_CANDIDATES_PER_TARGET or revise the bbox."
            )
        selected_for_stratum = []
        for selected_index in range(stratum["count"]):
            ranked_candidates = []
            remaining_needed = stratum["count"] - selected_index
            for candidate in candidate_pool:
                if candidate["tile_id"] in used_tiles:
                    continue
                distance_km = min(
                    min_distance_to_selected(candidate, selected_for_stratum),
                    min_distance_to_selected(candidate, locations),
                )
                ranked_candidates.append((distance_km, candidate))
            if not ranked_candidates:
                raise RuntimeError(f"Stratum {stratum['prefix']} ran out of unused MGRS tiles.")
            ranked_candidates.sort(key=lambda item: item[0], reverse=True)
            viable = [
                item
                for item in ranked_candidates
                if item[0] >= MIN_TILE_DISTANCE_KM or len(ranked_candidates) <= remaining_needed
            ]
            distance_km, chosen = viable[0] if viable else ranked_candidates[0]
            used_tiles.add(chosen["tile_id"])
            point_index = len(locations)
            location = dict(chosen)
            location.update(
                {
                    "sample_id": f"{stratum['prefix']}_{selected_index + 1:03d}",
                    "selection_min_distance_km": None if math.isinf(distance_km) else round(distance_km, 1),
                }
            )
            location.update(target_time_for_index(point_index))
            selected_for_stratum.append(location)
            locations.append(location)
    if len(locations) < TARGET_GROUPS:
        raise RuntimeError(f"Only generated {len(locations)} unique MGRS tiles; need {TARGET_GROUPS}.")
    return locations[:TARGET_GROUPS]


LOCATIONS = build_locations()


def parse_date(value):
    return datetime.strptime(value, "%Y%m%d")


def yyyymmdd_to_iso(value):
    return datetime.strptime(value, "%Y%m%d").strftime("%Y-%m-%d")


def month_delta(start_yyyymmdd, end_yyyymmdd):
    start = parse_date(start_yyyymmdd)
    end = parse_date(end_yyyymmdd)
    return (end.year - start.year) * 12 + (end.month - start.month)


def product_record(product):
    return {
        "id": product["Id"],
        "name": product["Name"],
        "size": product.get("ContentLength"),
        "covered_s2_ids": product.get("CoveredS2Ids"),
    }


def compact_product(product):
    return {
        "Id": product.get("Id"),
        "Name": product.get("Name"),
        "Online": product.get("Online"),
        "ContentLength": product.get("ContentLength"),
        "Attributes": product.get("Attributes", []),
        "GeoFootprint": product.get("GeoFootprint"),
        "CoveredS2Ids": product.get("CoveredS2Ids"),
    }


def point_in_ring(lon, lat, ring):
    inside = False
    if not ring:
        return False
    x1, y1 = ring[0][:2]
    for index in range(1, len(ring) + 1):
        x2, y2 = ring[index % len(ring)][:2]
        if (y1 > lat) != (y2 > lat):
            cross_lon = (x2 - x1) * (lat - y1) / (y2 - y1) + x1
            if lon < cross_lon:
                inside = not inside
        x1, y1 = x2, y2
    return inside


def point_in_polygon(lon, lat, polygon_coordinates):
    if not polygon_coordinates or not point_in_ring(lon, lat, polygon_coordinates[0]):
        return False
    return not any(point_in_ring(lon, lat, hole) for hole in polygon_coordinates[1:])


def geojson_contains_point(geometry_json, lon, lat):
    if not geometry_json:
        return False
    geometry_type = geometry_json.get("type")
    coordinates = geometry_json.get("coordinates", [])
    if geometry_type == "Polygon":
        return point_in_polygon(lon, lat, coordinates)
    if geometry_type == "MultiPolygon":
        return any(point_in_polygon(lon, lat, polygon) for polygon in coordinates)
    return False


def geojson_vertices(geometry_json):
    if not geometry_json:
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


def geojson_covers_geojson(cover_geometry, target_geometry):
    target_vertices = geojson_vertices(target_geometry)
    if not cover_geometry or not target_vertices:
        return False
    return all(geojson_contains_point(cover_geometry, lon, lat) for lon, lat in target_vertices)


def filter_s3_products_covering_s2(date_key, s2_products, s3_products):
    if not REQUIRE_S3_FOOTPRINT_COVERS_S2:
        return s3_products
    s2_same_date = []
    for s2 in s2_products:
        if safe_extract_date(s2, extract_s2_date) != date_key:
            continue
        if product_cloud(s2) is not None and product_cloud(s2) > CLOUD_LIMIT:
            continue
        if s2.get("GeoFootprint"):
            s2_same_date.append(s2)
    filtered = []
    for s3 in s3_products:
        covered_s2_ids = [
            s2.get("Id")
            for s2 in s2_same_date
            if s2.get("Id") and geojson_covers_geojson(s3.get("GeoFootprint"), s2.get("GeoFootprint"))
        ]
        if covered_s2_ids:
            item = dict(s3)
            item["CoveredS2Ids"] = covered_s2_ids
            filtered.append(item)
    return filtered


def cloud_ok_s2_dates(s2_products):
    dates = set()
    for product in s2_products:
        date_key = safe_extract_date(product, extract_s2_date)
        cloud = product_cloud(product)
        if date_key is not None and cloud is not None and cloud <= CLOUD_LIMIT:
            dates.add(date_key)
    return sorted(dates)


def flatten_s3_by_date(s3_by_date):
    products = []
    for date_key in sorted(s3_by_date):
        products.extend(s3_by_date[date_key])
    return products


def min_cloud_by_s2_date(s2_products):
    clouds_by_date = defaultdict(list)
    for product in s2_products:
        date_key = safe_extract_date(product, extract_s2_date)
        cloud = product_cloud(product)
        if date_key is not None and cloud is not None:
            clouds_by_date[date_key].append(cloud)
    return {date_key: min(values) for date_key, values in clouds_by_date.items()}


def ordered_s3_dates_for_location(location, target_dates, s2_products):
    target_dates = sorted(target_dates)
    target_interval = location["target_interval_months"]
    target_start_year = location["target_start_year"]
    date_clouds = min_cloud_by_s2_date(s2_products)
    scored_pairs = []
    for first_index, first in enumerate(target_dates):
        for second in target_dates[first_index + 1 :]:
            interval = month_delta(first, second)
            if INTERVAL_MONTH_MIN <= interval <= INTERVAL_MONTH_MAX:
                first_cloud = date_clouds.get(first, 100.0)
                second_cloud = date_clouds.get(second, 100.0)
                scored_pairs.append(
                    (
                        abs(interval - target_interval),
                        0 if first[:4] == target_start_year else 1,
                        abs(int(first[:4]) - int(target_start_year)),
                        (first_cloud + second_cloud) / 2,
                        max(first_cloud, second_cloud),
                        first,
                        second,
                    )
                )
    ordered_dates = []
    seen_dates = set()
    for _, _, _, _, _, first, second in sorted(scored_pairs):
        for date_key in (first, second):
            if date_key not in seen_dates:
                seen_dates.add(date_key)
                ordered_dates.append(date_key)
    for date_key in sorted(target_dates, key=lambda item: (date_clouds.get(item, 100.0), item)):
        if date_key not in seen_dates:
            ordered_dates.append(date_key)
    if MAX_S3_DATES_TO_VERIFY_PER_LOCATION is None:
        return ordered_dates
    return ordered_dates[:MAX_S3_DATES_TO_VERIFY_PER_LOCATION]


def eligible_group_status(location, s2_products, s3_by_date):
    s3_products = flatten_s3_by_date(s3_by_date)
    candidates = group_candidates(s2_products, s3_products, CLOUD_LIMIT)
    candidates = [item for item in candidates if item["cloud"] is not None and item["cloud"] <= CLOUD_LIMIT]
    groups = choose_groups(
        location,
        candidates,
        TARGET_GROUPS_PER_LOCATION,
        Counter(),
        Counter(),
    )
    best_interval_error = None
    if groups:
        best_interval_error = min(
            abs(group["interval_months"] - group.get("target_interval_months", group["interval_months"]))
            for group in groups
        )
    interval_ok = best_interval_error is not None and (
        EARLY_STOP_MAX_INTERVAL_ERROR_MONTHS is None
        or best_interval_error <= EARLY_STOP_MAX_INTERVAL_ERROR_MONTHS
    )
    return {
        "group_count": len(groups),
        "candidate_count": len(candidates),
        "best_interval_error": best_interval_error,
        "interval_ok": interval_ok,
    }


def choose_groups(location, candidates, group_quota, used_interval_counts, used_year_counts):
    groups = []
    possible_pairs = []
    target_interval = location["target_interval_months"]
    target_start_year = location["target_start_year"]
    for first_index, first in enumerate(candidates):
        if first["s2"].get("Online") is False or first["s3"].get("Online") is False:
            continue
        if len(groups) >= group_quota:
            break
        for second in candidates[first_index + 1 :]:
            if second["s2"].get("Online") is False or second["s3"].get("Online") is False:
                continue
            interval = month_delta(first["date"], second["date"])
            if INTERVAL_MONTH_MIN <= interval <= INTERVAL_MONTH_MAX:
                possible_pairs.append((first, second, interval))

    used_dates = set()
    while possible_pairs and len(groups) < group_quota:
        possible_pairs.sort(
            key=lambda row: (
                abs(row[2] - target_interval),
                0 if row[0]["date"][:4] == target_start_year else 1,
                used_interval_counts[row[2]],
                used_year_counts[row[0]["date"][:4]] + used_year_counts[row[1]["date"][:4]],
                row[0]["date"],
                row[1]["date"],
            )
        )
        first, second, interval = possible_pairs.pop(0)
        if first["date"] in used_dates or second["date"] in used_dates:
            continue
        used_dates.add(first["date"])
        used_dates.add(second["date"])
        used_interval_counts[interval] += 1
        used_year_counts[first["date"][:4]] += 1
        used_year_counts[second["date"][:4]] += 1
        groups.append(
            {
                "group_id": f"{location['id']}_{len(groups) + 1:03d}",
                "location": location,
                "interval_months": interval,
                "target_interval_months": target_interval,
                "target_start_year": target_start_year,
                "t1": {
                    "date": first["date"],
                    "cloud": first["cloud"],
                    "s2": product_record(first["s2"]),
                    "s3": product_record(first["s3"]),
                },
                "t2": {
                    "date": second["date"],
                    "cloud": second["cloud"],
                    "s2": product_record(second["s2"]),
                    "s3": product_record(second["s3"]),
                },
            }
        )
    return groups


def safe_extract_date(product, extractor):
    try:
        return extractor(product["Name"])
    except Exception:
        return None


def count_interval_pairs(dates):
    ordered = sorted(dates)
    count = 0
    examples = []
    for first_index, first in enumerate(ordered):
        for second in ordered[first_index + 1 :]:
            interval = month_delta(first, second)
            if INTERVAL_MONTH_MIN <= interval <= INTERVAL_MONTH_MAX:
                count += 1
                if len(examples) < 5:
                    examples.append({"t1": first, "t2": second, "interval_months": interval})
    return count, examples


def diagnose_no_match(s2_products, s3_products, candidates):
    s2_by_date = defaultdict(list)
    s3_by_date = defaultdict(list)
    s2_bad_date = 0
    s3_bad_date = 0
    for product in s2_products:
        date_key = safe_extract_date(product, extract_s2_date)
        if date_key is None:
            s2_bad_date += 1
            continue
        s2_by_date[date_key].append(product)
    for product in s3_products:
        date_key = safe_extract_date(product, extract_s3_date)
        if date_key is None:
            s3_bad_date += 1
            continue
        s3_by_date[date_key].append(product)

    common_dates = sorted(set(s2_by_date) & set(s3_by_date))
    cloud_ok_dates = []
    cloud_none_dates = []
    online_ok_dates = []
    cloud_values = []
    for date_key in common_dates:
        s2_clouds = [product_cloud(product) for product in s2_by_date[date_key]]
        valid_clouds = [value for value in s2_clouds if value is not None]
        cloud_values.extend(valid_clouds)
        if not valid_clouds:
            cloud_none_dates.append(date_key)
            continue
        if min(valid_clouds) <= CLOUD_LIMIT:
            cloud_ok_dates.append(date_key)
            if any(product.get("Online") is not False for product in s2_by_date[date_key]) and any(
                product.get("Online") is not False for product in s3_by_date[date_key]
            ):
                online_ok_dates.append(date_key)

    candidate_dates = sorted({item["date"] for item in candidates})
    interval_pair_count, interval_examples = count_interval_pairs(candidate_dates)
    online_interval_pair_count, online_interval_examples = count_interval_pairs(online_ok_dates)

    if not s2_products:
        reason = "no_s2_products_in_search_window"
    elif not s3_products:
        reason = "no_s3_products_in_search_window"
    elif not common_dates:
        reason = "no_same_day_s2_s3_dates"
    elif not cloud_ok_dates:
        reason = f"same_day_dates_exist_but_no_s2_cloud_le_{CLOUD_LIMIT}"
    elif not candidate_dates:
        reason = "same_day_cloud_ok_dates_exist_but_no_candidates_after_filter"
    elif interval_pair_count == 0:
        reason = f"cloud_ok_same_day_dates_exist_but_no_two_dates_with_{INTERVAL_MONTH_MIN}_{INTERVAL_MONTH_MAX}_month_interval"
    elif online_interval_pair_count == 0:
        reason = "interval_pairs_exist_but_products_marked_offline"
    else:
        reason = "eligible_pairs_exist_but_selection_returned_empty"

    return {
        "reason": reason,
        "s2_count": len(s2_products),
        "s3_count": len(s3_products),
        "s2_bad_date_count": s2_bad_date,
        "s3_bad_date_count": s3_bad_date,
        "s2_date_count": len(s2_by_date),
        "s3_date_count": len(s3_by_date),
        "same_day_date_count": len(common_dates),
        "same_day_date_examples": common_dates[:10],
        "cloud_none_same_day_date_count": len(cloud_none_dates),
        "cloud_ok_same_day_date_count": len(cloud_ok_dates),
        "cloud_ok_same_day_date_examples": cloud_ok_dates[:10],
        "online_cloud_ok_same_day_date_count": len(online_ok_dates),
        "candidate_count": len(candidates),
        "candidate_date_count": len(candidate_dates),
        "candidate_date_examples": candidate_dates[:10],
        "interval_pair_count": interval_pair_count,
        "interval_pair_examples": interval_examples,
        "online_interval_pair_count": online_interval_pair_count,
        "online_interval_pair_examples": online_interval_examples,
        "min_same_day_s2_cloud": min(cloud_values) if cloud_values else None,
        "max_same_day_s2_cloud": max(cloud_values) if cloud_values else None,
    }


def load_state():
    if not STATE_PATH.exists():
        return {
            "schema": "cdse_s2_s3_manifest_state_v1",
            "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "locations": {},
        }
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_state(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["updated_utc"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    tmp_path = STATE_PATH.with_suffix(STATE_PATH.suffix + ".part")
    tmp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp_path.replace(STATE_PATH)


def build_odata_products_url(collection, name_contains, start, end, wkt, top, skip):
    start_str = f"{start}T00:00:00.000Z"
    end_str = f"{end}T23:59:59.999Z"
    name_parts = name_contains if isinstance(name_contains, (list, tuple)) else [name_contains]
    filters = [
        f"Collection/Name eq '{collection}'",
        f"ContentDate/Start ge {start_str}",
        f"ContentDate/Start le {end_str}",
        f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')",
    ]
    filters.extend(f"contains(Name,'{part}')" for part in name_parts)
    return (
        f"{CDSE_ODATA}/Products?$filter="
        + quote(" and ".join(filters), safe=":/(),';")
        + f"&$top={top}&$skip={skip}&$expand=Attributes&$orderby=ContentDate/Start asc"
    )


def odata_search_paged(collection, name_contains, start, end, wkt, page_size, max_products):
    products = []
    skip = 0
    while max_products is None or len(products) < max_products:
        top = page_size if max_products is None else min(page_size, max_products - len(products))
        url = build_odata_products_url(collection, name_contains, start, end, wkt, top, skip)
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        page = response.json().get("value", [])
        products.extend(page)
        if len(page) < top:
            break
        skip += len(page)
        time.sleep(REQUEST_SLEEP_SECONDS)
    return products


def search_with_retry(collection, name_contains, start, end, wkt, page_size, max_products, location_id):
    last_error = None
    for attempt in range(1, SEARCH_MAX_RETRIES + 1):
        try:
            return odata_search_paged(collection, name_contains, start, end, wkt, page_size, max_products)
        except Exception as exc:
            last_error = str(exc)
            if attempt >= SEARCH_MAX_RETRIES:
                break
            sleep_seconds = SEARCH_RETRY_SLEEP_SECONDS * attempt
            print(
                f"  search retry {attempt}/{SEARCH_MAX_RETRIES} for {location_id} "
                f"{collection}: {last_error}; sleep {sleep_seconds}s"
            )
            time.sleep(sleep_seconds)
    raise RuntimeError(f"{collection} search failed after {SEARCH_MAX_RETRIES} attempts: {last_error}")


def write_summary(groups):
    rows = []
    for group in groups:
        rows.append(
            {
                "group_id": group["group_id"],
                "location_id": group["location"]["id"],
                "tile_id": group["location"].get("tile_id"),
                "region": group["location"]["region"],
                "landform": group["location"]["landform"],
                "stratum": group["location"].get("stratum"),
                "lon": group["location"]["lon"],
                "lat": group["location"]["lat"],
                "sample_id": group["location"].get("sample_id"),
                "candidate_pool_size": group["location"].get("candidate_pool_size"),
                "selection_min_distance_km": group["location"].get("selection_min_distance_km"),
                "t1_date": group["t1"]["date"],
                "t2_date": group["t2"]["date"],
                "t1_year": group["t1"]["date"][:4],
                "t2_year": group["t2"]["date"][:4],
                "target_start_year": group.get("target_start_year"),
                "target_interval_months": group.get("target_interval_months"),
                "interval_months": group["interval_months"],
                "interval_error_months": abs(group["interval_months"] - group.get("target_interval_months", group["interval_months"])),
            }
        )
    SUMMARY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV_PATH.open("w", newline="", encoding="utf-8") as fd:
        writer = csv.DictWriter(fd, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def print_distribution(groups):
    region_counts = Counter(group["location"]["region"] for group in groups)
    landform_counts = Counter(group["location"]["landform"] for group in groups)
    interval_counts = Counter(group["interval_months"] for group in groups)
    year_counts = Counter()
    for group in groups:
        year_counts[group["t1"]["date"][:4]] += 1
        year_counts[group["t2"]["date"][:4]] += 1
    print(f"Selected groups: {len(groups)}")
    print(f"Candidate points: {len(LOCATIONS)}")
    print(f"Regions: {dict(sorted(region_counts.items()))}")
    print(f"Landforms: {dict(sorted(landform_counts.items()))}")
    print(f"Intervals: {dict(sorted(interval_counts.items()))}")
    print(f"Observation years: {dict(sorted(year_counts.items()))}")


def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def print_phase_progress(phase, completed, total, status_counts, started_at):
    elapsed = time.time() - started_at
    rate = completed / elapsed if elapsed > 0 else 0
    remaining = total - completed
    eta = remaining / rate if rate > 0 else None
    eta_text = format_duration(eta) if eta is not None else "unknown"
    print(
        f"{phase} progress: {completed}/{total} "
        f"({completed / total * 100:.1f}%) "
        f"searched={status_counts.get('searched', 0)} "
        f"skipped={status_counts.get('skipped', 0)} "
        f"failed={status_counts.get('failed', 0)} "
        f"elapsed={format_duration(elapsed)} eta={eta_text} "
        f"rate={rate:.3f} tile/s"
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    state.setdefault("locations", {})
    state_lock = threading.Lock()
    searched_count = sum(
        1
        for record in state["locations"].values()
        if record.get("search_status") == "searched"
        and record.get("search_strategy_version") == SEARCH_STRATEGY_VERSION
        and "s2_products" in record
        and "s3_by_date" in record
    )
    print(f"Loaded checkpoint: searched={searched_count}/{len(LOCATIONS)} state={STATE_PATH}")

    def save_location_record(location_id, record):
        with state_lock:
            state["locations"][location_id] = record
            save_state(state)

    def search_location(location):
        location_id = location["id"]
        with state_lock:
            record = dict(state["locations"].get(location_id, {}))
        previous_strategy_version = record.get("search_strategy_version")
        if (
            record.get("search_status") == "searched"
            and record.get("search_strategy_version") == SEARCH_STRATEGY_VERSION
            and "s2_products" in record
            and "s3_by_date" in record
        ):
            return location_id, "skipped"
        if previous_strategy_version != SEARCH_STRATEGY_VERSION:
            for key in (
                "s2_products",
                "s2_count",
                "s3_by_date",
                "s3_count",
                "s3_verified_date_count",
                "s3_raw_count_by_date",
                "s3_coverage_filtered_count_by_date",
            ):
                record.pop(key, None)

        record.update(
            {
                "location": location,
                "search_status": "running",
                "search_started_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "search_attempts": record.get("search_attempts", 0) + 1,
                "search_strategy": "s2_full_range_then_target_interval_cloud_ranked_s3_dates",
                "search_strategy_version": SEARCH_STRATEGY_VERSION,
                "search_range": {"start": SEARCH_START, "end": SEARCH_END},
            }
        )
        save_location_record(location_id, record)

        try:
            wkt = build_point_wkt(location["lon"], location["lat"])
            if "s2_products" not in record:
                print(f"Searching S2 {location_id} ({location['lon']}, {location['lat']})")
                s2_products = search_with_retry(
                    "SENTINEL-2",
                    ["MSIL2A", f"T{location['tile_id']}"],
                    SEARCH_START,
                    SEARCH_END,
                    wkt,
                    S2_PAGE_SIZE,
                    S2_MAX_PRODUCTS,
                    location_id,
                )
                record["s2_products"] = [compact_product(product) for product in s2_products]
                record["s2_count"] = len(s2_products)
                save_location_record(location_id, record)
            else:
                s2_products = record["s2_products"]

            target_dates = cloud_ok_s2_dates(s2_products)
            s3_by_date = record.get("s3_by_date", {})
            planned_s3_dates = ordered_s3_dates_for_location(location, target_dates, s2_products)
            record["cloud_ok_s2_date_count"] = len(target_dates)
            record["cloud_ok_s2_dates"] = target_dates
            record["s3_planned_date_count"] = len(planned_s3_dates)
            record["s3_search_date_limit"] = MAX_S3_DATES_TO_VERIFY_PER_LOCATION
            record["early_stop_max_interval_error_months"] = EARLY_STOP_MAX_INTERVAL_ERROR_MONTHS
            save_location_record(location_id, record)

            stop_reason = None
            if STOP_S3_AFTER_GROUP_CANDIDATES:
                group_status = eligible_group_status(location, s2_products, s3_by_date)
                record["early_stop_group_count"] = group_status["group_count"]
                record["early_stop_candidate_count"] = group_status["candidate_count"]
                record["early_stop_best_interval_error_months"] = group_status["best_interval_error"]
                if group_status["group_count"] >= MIN_GROUP_CANDIDATES_BEFORE_STOP and group_status["interval_ok"]:
                    stop_reason = "target_interval_group_found_from_cached_s3"
                    print(
                        f"Stop S3 {location_id}: eligible_groups={group_status['group_count']} "
                        f"candidates={group_status['candidate_count']} "
                        f"best_interval_error={group_status['best_interval_error']} from cached dates"
                    )
                    save_location_record(location_id, record)

            for date_index, date_key in enumerate(planned_s3_dates, start=1):
                if stop_reason:
                    break
                if date_key in s3_by_date:
                    continue
                iso_date = yyyymmdd_to_iso(date_key)
                print(
                    f"Searching S3 {location_id} date={date_key} "
                    f"({date_index}/{len(planned_s3_dates)}, cloud_ok_total={len(target_dates)})"
                )
                s3_products = search_with_retry(
                    "SENTINEL-3",
                    "OL_1_EFR",
                    iso_date,
                    iso_date,
                    wkt,
                    S3_PAGE_SIZE,
                    S3_MAX_PRODUCTS_PER_DATE,
                    location_id,
                )
                raw_s3_count = len(s3_products)
                s3_products = filter_s3_products_covering_s2(date_key, s2_products, s3_products)
                if REQUIRE_S3_FOOTPRINT_COVERS_S2:
                    print(f"  S3 coverage filter {location_id} date={date_key}: {raw_s3_count} -> {len(s3_products)}")
                s3_by_date[date_key] = [compact_product(product) for product in s3_products]
                record["s3_by_date"] = s3_by_date
                record.setdefault("s3_raw_count_by_date", {})[date_key] = raw_s3_count
                record.setdefault("s3_coverage_filtered_count_by_date", {})[date_key] = len(s3_products)
                record["s3_verified_date_count"] = len(s3_by_date)
                record["s3_count"] = sum(len(items) for items in s3_by_date.values())
                save_location_record(location_id, record)
                time.sleep(REQUEST_SLEEP_SECONDS)
                if STOP_S3_AFTER_GROUP_CANDIDATES:
                    group_status = eligible_group_status(location, s2_products, s3_by_date)
                    record["early_stop_group_count"] = group_status["group_count"]
                    record["early_stop_candidate_count"] = group_status["candidate_count"]
                    record["early_stop_best_interval_error_months"] = group_status["best_interval_error"]
                    save_location_record(location_id, record)
                    if group_status["group_count"] >= MIN_GROUP_CANDIDATES_BEFORE_STOP and group_status["interval_ok"]:
                        stop_reason = "target_interval_group_found"
                        print(
                            f"Stop S3 {location_id}: eligible_groups={group_status['group_count']} "
                            f"candidates={group_status['candidate_count']} "
                            f"best_interval_error={group_status['best_interval_error']} "
                            f"after {len(s3_by_date)}/{len(planned_s3_dates)} dates"
                        )
                        break

            if stop_reason is None:
                if MAX_S3_DATES_TO_VERIFY_PER_LOCATION is not None and len(planned_s3_dates) < len(target_dates):
                    stop_reason = "max_s3_dates_verified_without_eligible_group"
                else:
                    stop_reason = "all_cloud_ok_dates_verified"

            record.update(
                {
                    "search_status": "searched",
                    "search_finished_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "s3_by_date": s3_by_date,
                    "s3_verified_date_count": len(s3_by_date),
                    "s3_count": sum(len(items) for items in s3_by_date.values()),
                    "s3_search_stop_reason": stop_reason,
                    "search_error": None,
                }
            )
            save_location_record(location_id, record)
            return location_id, "searched"
        except Exception as exc:
            record.update(
                {
                    "search_status": "failed",
                    "search_finished_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "search_error": str(exc),
                }
            )
            save_location_record(location_id, record)
            return location_id, f"failed: {exc}"

    print("Phase 1/2: caching S2 full-range results and S3 only for cloud-ok same-day candidates.")
    phase1_started_at = time.time()
    phase1_completed = 0
    phase1_status_counts = Counter()
    with ThreadPoolExecutor(max_workers=MAX_SEARCH_WORKERS) as executor:
        futures = [executor.submit(search_location, location) for location in LOCATIONS]
        for future in as_completed(futures):
            location_id, status = future.result()
            print(f"  {location_id}: {status}")
            phase1_completed += 1
            if status == "skipped":
                phase1_status_counts["skipped"] += 1
            elif status == "searched":
                phase1_status_counts["searched"] += 1
            elif status.startswith("failed:"):
                phase1_status_counts["failed"] += 1
            else:
                phase1_status_counts[status] += 1
            print_phase_progress("Phase 1/2", phase1_completed, len(LOCATIONS), phase1_status_counts, phase1_started_at)

    print("Phase 2/2: selecting groups from cached search results.")
    all_groups = []
    used_interval_counts = Counter()
    used_year_counts = Counter()
    selected_location_ids = set()
    phase2_started_at = time.time()
    phase2_status_counts = Counter()
    for selection_index, location in enumerate(LOCATIONS, start=1):
        if len(all_groups) >= TARGET_GROUPS:
            break
        record = state["locations"].get(location["id"], {})
        if record.get("search_status") != "searched":
            print(f"  no_match {location['id']}: search_not_available")
            record.update({"selection_status": "no_search", "selection_error": "search_not_available"})
            state["locations"][location["id"]] = record
            phase2_status_counts["no_search"] += 1
            print(
                f"Phase 2/2 progress: {selection_index}/{len(LOCATIONS)} "
                f"selected_groups={len(all_groups)}/{TARGET_GROUPS} "
                f"selected_tiles={len(selected_location_ids)} "
                f"no_match={phase2_status_counts.get('no_match', 0)} "
                f"no_search={phase2_status_counts.get('no_search', 0)} "
                f"elapsed={format_duration(time.time() - phase2_started_at)}"
            )
            continue

        s2_products = record.get("s2_products", [])
        s3_products = flatten_s3_by_date(record.get("s3_by_date", {}))
        candidates = group_candidates(s2_products, s3_products, CLOUD_LIMIT)
        candidates = [
            item for item in candidates if item["cloud"] is not None and item["cloud"] <= CLOUD_LIMIT
        ]
        quota = min(TARGET_GROUPS_PER_LOCATION, TARGET_GROUPS - len(all_groups))
        groups = choose_groups(location, candidates, quota, used_interval_counts, used_year_counts)
        diagnosis = diagnose_no_match(s2_products, s3_products, candidates)
        if groups:
            all_groups.extend(groups)
            selected_location_ids.add(location["id"])
            phase2_status_counts["selected"] += 1
            record.update(
                {
                    "selection_status": "selected",
                    "selection_finished_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "candidate_count": len(candidates),
                    "diagnosis": diagnosis,
                    "groups": groups,
                    "selection_error": None,
                }
            )
            print(f"  selected {location['id']}: candidates={len(candidates)} total={len(all_groups)}")
        else:
            phase2_status_counts["no_match"] += 1
            record.update(
                {
                    "selection_status": "no_match",
                    "selection_finished_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "candidate_count": len(candidates),
                    "diagnosis": diagnosis,
                    "groups": [],
                    "selection_error": diagnosis["reason"],
                }
            )
            print(f"  no_match {location['id']}: {diagnosis['reason']}")
        state["locations"][location["id"]] = record
        print(
            f"Phase 2/2 progress: {selection_index}/{len(LOCATIONS)} "
            f"selected_groups={len(all_groups)}/{TARGET_GROUPS} "
            f"selected_tiles={len(selected_location_ids)} "
            f"no_match={phase2_status_counts.get('no_match', 0)} "
            f"no_search={phase2_status_counts.get('no_search', 0)} "
            f"elapsed={format_duration(time.time() - phase2_started_at)}"
        )

    for location in LOCATIONS:
        if location["id"] in selected_location_ids:
            continue
        record = state["locations"].get(location["id"], {})
        if record.get("selection_status") == "selected":
            record["selection_status"] = "not_selected_this_run"
            record["groups"] = []
            state["locations"][location["id"]] = record
    save_state(state)

    manifest = {
        "schema": "cdse_s2_s3_group_manifest_v1",
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "requirements": {
            "exclude_china": True,
            "land_only_by_configured_points": True,
            "cloud_limit": CLOUD_LIMIT,
            "date_range": [SEARCH_START, SEARCH_END],
            "interval_months": [INTERVAL_MONTH_MIN, INTERVAL_MONTH_MAX],
            "sampling_unit": "sentinel2_mgrs_tile",
            "tile_selection_strategy": "configured_landform_strata_candidate_pool_farthest_sampling",
            "tile_candidates_per_target": TILE_CANDIDATES_PER_TARGET,
            "min_tile_distance_km": MIN_TILE_DISTANCE_KM,
            "search_strategy": "s2_full_range_then_target_interval_cloud_ranked_s3_dates_with_strict_early_stop_then_select",
            "search_strategy_version": SEARCH_STRATEGY_VERSION,
            "search_page_size": {"s2": S2_PAGE_SIZE, "s3": S3_PAGE_SIZE},
            "search_max_products": {"s2": S2_MAX_PRODUCTS, "s3_per_date": S3_MAX_PRODUCTS_PER_DATE},
            "require_s3_footprint_covers_s2": REQUIRE_S3_FOOTPRINT_COVERS_S2,
            "max_s3_dates_to_verify_per_location": MAX_S3_DATES_TO_VERIFY_PER_LOCATION,
            "stop_s3_after_group_candidates": STOP_S3_AFTER_GROUP_CANDIDATES,
            "min_group_candidates_before_stop": MIN_GROUP_CANDIDATES_BEFORE_STOP,
            "early_stop_max_interval_error_months": EARLY_STOP_MAX_INTERVAL_ERROR_MONTHS,
            "max_search_workers": MAX_SEARCH_WORKERS,
            "target_groups": TARGET_GROUPS,
            "target_groups_per_location": TARGET_GROUPS_PER_LOCATION,
            "candidate_locations": len(LOCATIONS),
            "time_uniformity_strategy": (
                "Each concrete point is assigned a target start year round-robin over "
                f"{TARGET_START_YEARS} and a target interval round-robin over "
                f"{TARGET_INTERVALS}; selection minimizes interval/year deviation, then balances global counts."
            ),
        },
        "checkpoint": {
            "state_path": str(STATE_PATH),
            "search_status_counts": dict(
                sorted(Counter(record.get("search_status", "unknown") for record in state["locations"].values()).items())
            ),
            "selection_status_counts": dict(
                sorted(Counter(record.get("selection_status", "unknown") for record in state["locations"].values()).items())
            ),
        },
        "groups": all_groups,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_summary(all_groups)
    print_distribution(all_groups)
    print(f"Manifest written: {MANIFEST_PATH}")
    print(f"Summary CSV written: {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    main()
