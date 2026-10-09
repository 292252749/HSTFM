"""
Build a 512-sample Sentinel-2/Sentinel-3 LMDB from an existing manifest.

This is the v2 data builder. It intentionally leaves the original
cdse_batch_pipeline scripts unchanged.

Edit the CONFIG section before running. One LMDB record is one 512x512 S2
sample with its paired 64x64 S3 arrays:
    [LR_t1, HR_t1, LR_t2, HR_t2]

FROMGLC30 V002 2024 tiles are required for land/water filtering and land-cover
statistics. Local ne_10m_ocean polygons are used as a hard ocean rejection
filter before samples are written.
"""

import csv
import io
import json
import math
import pickle
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from pathlib import PurePosixPath
import struct

import numpy as np
import rasterio
import xarray as xr
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_bounds
from shapely.geometry import Point, Polygon, box
from shapely.strtree import STRtree


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = ROOT / "cdse_batch_pipeline"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))


S2_BANDS = ["B02", "B03", "B04", "B8A"]
S3_BANDS = ["Oa04", "Oa06", "Oa08", "Oa17"]

HR_SIZE = 1024
LR_FACTOR = 8
LR_SIZE = HR_SIZE // LR_FACTOR
SAMPLE_HR_SIZE = 512
SAMPLE_LR_SIZE = SAMPLE_HR_SIZE // LR_FACTOR

S2_BAD_VALUE = 0
S3_BAD_VALUE = 65535

def load_s3_swath_bands(s3_zip_path):
    with zipfile.ZipFile(s3_zip_path, "r") as zip_ref:
        names = zip_ref.namelist()
        geo_name = [name for name in names if "geo_coordinates.nc" in name][0]
        with zip_ref.open(geo_name) as geo_file:
            geo_data = xr.open_dataset(io.BytesIO(geo_file.read()))
            latitude = geo_data["latitude"].values
            longitude = geo_data["longitude"].values

        bands = {}
        for name in names:
            base = PurePosixPath(name).name
            band_name = base.split("_")[0]
            if band_name in S3_BANDS and name.endswith("radiance.nc"):
                with zip_ref.open(name) as rad_file:
                    rad_data = xr.open_dataset(io.BytesIO(rad_file.read()), decode_cf=False)
                    bands[band_name] = np.squeeze(rad_data.to_dataarray().values)

    missing = [band for band in S3_BANDS if band not in bands]
    if missing:
        raise RuntimeError(f"Missing S3 radiance bands in {s3_zip_path}: {missing}")
    return latitude, longitude, bands


def has_bad_value(arr, bad_value):
    return np.any(arr == bad_value) or np.any(np.isnan(arr))


def find_s2_band_paths(zip_path):
    band_paths = {}
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        names = zip_ref.namelist()
    for band in S2_BANDS:
        matches = [name for name in names if band in name and "20m" in name and name.endswith(".jp2")]
        if not matches:
            raise RuntimeError(f"Cannot find {band} 20m JP2 in {zip_path}")
        band_paths[band] = "/vsizip/" + str(zip_path) + "/" + matches[0]
    return band_paths


def extract_tif_info(tif_path, resolution_factor=LR_FACTOR):
    with rasterio.open(tif_path) as ds:
        proj_wkt = ds.crs.to_wkt()
        transform = ds.transform
        pixel_size_x = transform.a
        pixel_size_y = transform.e
        new_pixel_size_x = pixel_size_x * resolution_factor
        new_pixel_size_y = pixel_size_y * resolution_factor
        width = ds.width
        height = ds.height
        new_width = int(width * (pixel_size_x / new_pixel_size_x))
        new_height = int(height * (pixel_size_y / new_pixel_size_y))
        min_x, min_y, max_x, max_y = ds.bounds
        epsg_code = ds.crs.to_epsg()

    if epsg_code is None or epsg_code < 32601 or epsg_code > 32760:
        raise RuntimeError(f"Expected a Sentinel-2 UTM CRS, got EPSG:{epsg_code}")
    zone = epsg_code % 100
    area_id = "sentinel2_grid"
    description = f"Sentinel-2 Grid (Reprojected to UTM Zone {zone})"
    proj_id = f"utm_{epsg_code}"
    projection = {"proj": "utm", "zone": zone, "datum": "WGS84", "units": "m"}
    if epsg_code >= 32700:
        projection["south"] = True
    return (
        area_id,
        description,
        proj_id,
        projection,
        new_width,
        new_height,
        (min_x, min_y, max_x, max_y),
        proj_wkt,
        new_pixel_size_x,
        new_pixel_size_y,
    )


def pack_arrays(arr_list, cname="lz4", clevel=3):
    import blosc

    buf = io.BytesIO()
    for arr in arr_list:
        np.save(buf, arr, allow_pickle=False)
    return blosc.compress(buf.getvalue(), cname=cname, clevel=clevel)


# =========================
# CONFIG: edit values here.
# =========================
DATASET_NAME = "global_multiyear"
SOURCE_DATASET = "global"
MANIFEST_PATH = Path("cdse_batch_pipeline_out/manifest.json")
DOWNLOAD_ROOT = Path("cdse_batch_pipeline_out/downloads")
OUT_ROOT = Path("cdse_batch_pipeline_out/v2_512")
PROCESSED_ROOT = OUT_ROOT / "processed_512"
LMDB_PATH = OUT_ROOT / "lmdb_512"
SAMPLE_INDEX_CSV = OUT_ROOT / "sample_index.csv"
BUILD_REPORT_JSON = OUT_ROOT / "build_report.json"

# FROMGLC30 V002 2024 tiles for filtering/statistics.
FROMGLC_DIR = Path("get_data/get_landcovr/fromglc2024")
FROMGLC_RESAMPLE_SHAPE = (128, 128)
MIN_LAND_FRACTION = 0.80
MAX_WATER_FRACTION = 0.20
MAX_NODATA_FRACTION = 0.10

# Local ocean shapefile. This is not downloaded here.
OCEAN_SHP_PATH = Path("get_data/ne_10m_ocean/ne_10m_ocean.shp")
SKIP_IF_OCEAN_INTERSECTS = True

PATCH_HR_SIZE = 512
PATCH_LR_SIZE = PATCH_HR_SIZE // LR_FACTOR
LMDB_BATCH_SIZE = 500
LMDB_MAP_SIZE = None
LMDB_BYTES_PER_SAMPLE_ESTIMATE = 6 * 1024**2
LMDB_MAP_SIZE_MARGIN = 2.0
LMDB_MIN_MAP_SIZE = 8 * 1024**3
MAX_GROUPS_TO_PROCESS = None
FILTER_INVALID_GROUPS_BEFORE_PROCESSING = True
STOP_ON_PROCESS_ERROR = False
REUSE_PROCESSED_PATCHES = True
KEEP_PROCESSED_PATCHES = True
ENABLE_CHANGE_METRICS = True
ENABLE_S3_SHIFT_METRICS = True
ALLOW_LOCAL_MANIFEST_FROM_DOWNLOADS = True
LOCAL_PAIR_STRATEGY = "adjacent_dates"


FROMGLC_CLASSES = {
    10: "cropland",
    20: "forest",
    30: "grassland",
    40: "shrubland",
    50: "wetland",
    60: "water",
    70: "tundra",
    80: "impervious_surface",
    90: "bareland",
    100: "snow_ice",
}

FROMGLC_WATER_CLASSES = {60}
FROMGLC_NODATA_VALUES = {0, 255}


def product_zip_name(product):
    name = product["name"]
    return name if name.endswith(".zip") else name + ".zip"

def product_name_from_zip(path):
    name = path.name
    return name[:-4] if name.endswith(".zip") else name


def index_downloads(root):
    return {path.name: path for path in root.rglob("*.zip")}


def find_zip(product, files):
    path = files.get(product_zip_name(product))
    if path is None:
        raise FileNotFoundError(f"Missing downloaded file: {product_zip_name(product)}")
    return path


def validate_zip(path):
    with zipfile.ZipFile(path, "r") as zip_ref:
        bad = zip_ref.testzip()
        if bad:
            raise RuntimeError(f"Bad zip member in {path}: {bad}")


def extract_mgrs_from_s2_name(name):
    match = re.search(r"_T([0-9]{2}[A-Z]{3})_", name)
    return match.group(1) if match else None


def extract_s2_date_from_name(name):
    match = re.search(r"_MSIL2A_(\d{8})T", name)
    return match.group(1) if match else None


def extract_s3_date_from_name(name):
    match = re.search(r"OL_1_EFR____(\d{8})T", name)
    return match.group(1) if match else None


def build_manifest_from_downloads(files):
    pairs_by_tile_date = defaultdict(dict)
    for path in files.values():
        name = product_name_from_zip(path)
        if "MSIL2A_" not in name:
            continue
        date = extract_s2_date_from_name(name)
        tile = extract_mgrs_from_s2_name(name)
        if date is None or tile is None:
            continue
        pairs_by_tile_date[(tile, date)]["s2"] = {"id": None, "name": name}

    s3_by_date = {}
    for path in files.values():
        name = product_name_from_zip(path)
        if "OL_1_EFR" not in name:
            continue
        date = extract_s3_date_from_name(name)
        if date is not None and date not in s3_by_date:
            s3_by_date[date] = {"id": None, "name": name}

    by_tile = defaultdict(list)
    for (tile, date), item in pairs_by_tile_date.items():
        s3 = s3_by_date.get(date)
        if "s2" not in item or s3 is None:
            continue
        by_tile[tile].append({"date": date, "s2": item["s2"], "s3": s3})

    groups = []
    for tile, items in sorted(by_tile.items()):
        items = sorted(items, key=lambda item: item["date"])
        if LOCAL_PAIR_STRATEGY != "adjacent_dates":
            raise ValueError(f"Unsupported LOCAL_PAIR_STRATEGY: {LOCAL_PAIR_STRATEGY}")
        for idx in range(len(items) - 1):
            t1 = items[idx]
            t2 = items[idx + 1]
            if t1["date"] == t2["date"]:
                continue
            groups.append(
                {
                    "group_id": f"{tile}_{idx + 1:04d}",
                    "location": {"id": tile, "tile_id": tile, "mgrs_tile_id": tile},
                    "interval_months": "",
                    "target_interval_months": "",
                    "target_start_year": "",
                    "t1": {"date": t1["date"], "cloud": None, "s2": t1["s2"], "s3": t1["s3"]},
                    "t2": {"date": t2["date"], "cloud": None, "s2": t2["s2"], "s3": t2["s3"]},
                }
            )
    return {
        "schema": "local_download_manifest_v1",
        "created_from": str(DOWNLOAD_ROOT),
        "pair_strategy": LOCAL_PAIR_STRATEGY,
        "groups": groups,
    }


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
    dates = {
        "t1_s2": extract_s2_date_from_name(group["t1"]["s2"]["name"]),
        "t1_s3": extract_s3_date_from_name(group["t1"]["s3"]["name"]),
        "t2_s2": extract_s2_date_from_name(group["t2"]["s2"]["name"]),
        "t2_s3": extract_s3_date_from_name(group["t2"]["s3"]["name"]),
    }
    missing = [key for key, value in dates.items() if value is None]
    if missing:
        raise RuntimeError(f"Cannot parse product dates for {group['group_id']}: {missing}")
    if dates["t1_s2"] != dates["t1_s3"]:
        raise RuntimeError(
            f"t1 S2/S3 date mismatch for {group['group_id']}: "
            f"s2={dates['t1_s2']}, s3={dates['t1_s3']}"
        )
    if dates["t2_s2"] != dates["t2_s3"]:
        raise RuntimeError(
            f"t2 S2/S3 date mismatch for {group['group_id']}: "
            f"s2={dates['t2_s2']}, s3={dates['t2_s3']}"
        )
    if dates["t1_s2"] == dates["t2_s2"]:
        raise RuntimeError(f"t1/t2 dates must differ for {group['group_id']}: date={dates['t1_s2']}")
    return mgrs_tile, dates


def date_parts(date_text):
    dt = datetime.strptime(date_text, "%Y%m%d")
    if dt.month in (12, 1, 2):
        season = "winter"
    elif dt.month in (3, 4, 5):
        season = "spring"
    elif dt.month in (6, 7, 8):
        season = "summer"
    else:
        season = "autumn"
    return {
        "year": dt.year,
        "month": dt.month,
        "season": season,
        "date_iso": dt.strftime("%Y-%m-%d"),
    }


def interval_days(t1_date, t2_date):
    t1 = datetime.strptime(t1_date, "%Y%m%d")
    t2 = datetime.strptime(t2_date, "%Y%m%d")
    return (t2 - t1).days


def _ring_area(points):
    if len(points) < 4:
        return 0.0
    total = 0.0
    for idx in range(len(points)):
        x1, y1 = points[idx]
        x2, y2 = points[(idx + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def _read_shp_polygon_parts(path):
    data = Path(path).read_bytes()
    shape_type = struct.unpack("<i", data[32:36])[0]
    if shape_type != 5:
        raise RuntimeError(f"Expected polygon shapefile type 5 at {path}, got {shape_type}")

    offset = 100
    while offset < len(data):
        if offset + 8 > len(data):
            break
        _, content_words = struct.unpack(">2i", data[offset:offset + 8])
        offset += 8
        content_size = content_words * 2
        content = data[offset:offset + content_size]
        offset += content_size
        if len(content) < 44:
            continue
        record_type = struct.unpack("<i", content[:4])[0]
        if record_type == 0:
            continue
        if record_type != 5:
            raise RuntimeError(f"Unsupported shape record type {record_type} in {path}")
        num_parts, num_points = struct.unpack("<2i", content[36:44])
        parts_offset = 44
        points_offset = parts_offset + 4 * num_parts
        part_starts = list(struct.unpack("<" + "i" * num_parts, content[parts_offset:points_offset]))
        raw_points = struct.unpack("<" + "d" * (2 * num_points), content[points_offset:points_offset + 16 * num_points])
        part_starts.append(num_points)
        parts = []
        for idx in range(num_parts):
            start = part_starts[idx]
            end = part_starts[idx + 1]
            points = [(raw_points[i * 2], raw_points[i * 2 + 1]) for i in range(start, end)]
            if len(points) >= 4:
                parts.append(points)
        yield parts


def _polygons_from_shp_parts(parts):
    rings = []
    for points in parts:
        area = _ring_area(points)
        if abs(area) < 1e-10:
            continue
        if points[0] != points[-1]:
            points = points + [points[0]]
        rings.append({"points": points, "area": area, "abs_area": abs(area)})
    if not rings:
        return []

    exteriors = [ring for ring in rings if ring["area"] < 0]
    holes = [ring for ring in rings if ring["area"] > 0]
    if not exteriors:
        largest = max(rings, key=lambda item: item["abs_area"])
        exteriors = [largest]
        holes = [ring for ring in rings if ring is not largest]

    polygons = []
    for exterior in exteriors:
        exterior_poly = Polygon(exterior["points"])
        if not exterior_poly.is_valid:
            exterior_poly = exterior_poly.buffer(0)
        assigned_holes = []
        for hole in holes:
            try:
                if exterior_poly.contains(Polygon(hole["points"]).representative_point()):
                    assigned_holes.append(hole["points"])
            except Exception:
                continue
        polygon = Polygon(exterior["points"], assigned_holes)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if not polygon.is_empty:
            polygons.append(polygon)
    return polygons


def load_ocean_lookup():
    if not OCEAN_SHP_PATH.exists():
        raise SystemExit(f"Ocean shapefile not found: {OCEAN_SHP_PATH}")

    exteriors = []
    holes = []
    for parts in _read_shp_polygon_parts(OCEAN_SHP_PATH):
        for points in parts:
            area = _ring_area(points)
            if abs(area) < 1e-10:
                continue
            if points[0] != points[-1]:
                points = points + [points[0]]
            polygon = Polygon(points)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if polygon.is_empty:
                continue
            if area < 0:
                exteriors.append(polygon)
            else:
                holes.append(polygon)

    if not exteriors:
        raise SystemExit(f"No ocean exterior rings could be read from {OCEAN_SHP_PATH}")
    hole_tree = STRtree(holes) if holes else None
    print(f"Ocean shapefile loaded from {OCEAN_SHP_PATH}: exteriors={len(exteriors)} land_holes={len(holes)}")

    def _tree_candidates(point):
        if hole_tree is None:
            return []
        candidates = hole_tree.query(point)
        out = []
        for item in candidates:
            if isinstance(item, (int, np.integer)):
                out.append(holes[int(item)])
            else:
                out.append(item)
        return out

    def point_is_ocean(lon, lat):
        point = Point(float(lon), float(lat))
        if not any(exterior.contains(point) or exterior.touches(point) for exterior in exteriors):
            return False
        for land_hole in _tree_candidates(point):
            if land_hole.contains(point) or land_hole.touches(point):
                return False
        return True

    def lookup(bounds_wgs84):
        minx, miny, maxx, maxy = [float(value) for value in bounds_wgs84]
        probes = [
            ((minx + maxx) / 2.0, (miny + maxy) / 2.0),
            (minx, miny),
            (minx, maxy),
            (maxx, miny),
            (maxx, maxy),
        ]
        return all(point_is_ocean(lon, lat) for lon, lat in probes)

    return lookup

def ocean_stats(ocean_lookup, s2_patch_path):
    with rasterio.open(s2_patch_path) as src:
        bounds_wgs84 = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
    intersects = bool(ocean_lookup(bounds_wgs84)) if ocean_lookup else False
    return {
        "ocean_intersects": intersects,
        "ocean_filter_source": str(OCEAN_SHP_PATH),
        "bounds_wgs84_json": json.dumps([float(value) for value in bounds_wgs84]),
    }

# FROMGLC land-cover reader used by main().
def _bounds_overlap(a, b):
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def _scan_raster_tiles(folder):
    tiles = []
    for path in sorted(Path(folder).glob("*.tif")):
        try:
            with rasterio.open(path) as ds:
                tiles.append({"path": path, "bounds": tuple(ds.bounds), "crs": ds.crs})
        except Exception as exc:
            print(f"Warning: skip unreadable landcover tile {path}: {exc}")
    return tiles


def open_landcover():
    tiles = _scan_raster_tiles(FROMGLC_DIR)
    if not tiles:
        raise SystemExit(f"No FROMGLC GeoTIFF tiles found under {FROMGLC_DIR}")
    print(f"FROMGLC landcover tiles loaded: {len(tiles)} from {FROMGLC_DIR}")
    return {"source": "fromglc", "tiles": tiles}


def _landcover_stats_from_array(arr, classes, water_classes, nodata_values, source_label):
    total = int(arr.size)
    counts = Counter(int(value) for value in arr.reshape(-1))
    nodata_count = sum(counts.get(value, 0) for value in nodata_values)
    water_count = sum(counts.get(value, 0) for value in water_classes)
    land_counts = {
        value: count
        for value, count in counts.items()
        if value not in nodata_values and value not in water_classes
    }
    land_count = sum(land_counts.values())
    if land_counts:
        major_value, major_count = max(land_counts.items(), key=lambda item: item[1])
        major_name = classes.get(major_value, f"class_{major_value}")
        dominant_ratio = major_count / total if total else 0.0
    elif water_count:
        major_name = "water"
        dominant_ratio = water_count / total if total else 0.0
    else:
        major_name = "nodata"
        dominant_ratio = nodata_count / total if total else 0.0
    named_counts = {classes.get(value, f"class_{value}"): count for value, count in sorted(counts.items())}
    return {
        "land_fraction": land_count / total if total else 0.0,
        "water_fraction": water_count / total if total else 0.0,
        "nodata_fraction": nodata_count / total if total else 0.0,
        "landcover_major": major_name,
        "dominant_landcover_ratio": dominant_ratio,
        "landcover_counts_json": json.dumps(named_counts, sort_keys=True),
        "landcover_source": source_label,
    }


def _empty_landcover_stats(reason):
    return {
        "land_fraction": "",
        "water_fraction": "",
        "nodata_fraction": "",
        "landcover_major": reason,
        "dominant_landcover_ratio": "",
        "landcover_counts_json": "{}",
        "landcover_source": str(FROMGLC_DIR),
    }


def landcover_stats(landcover, s2_patch_path):
    if not landcover:
        return _empty_landcover_stats("landcover_unavailable")
    with rasterio.open(s2_patch_path) as src:
        src_bounds = src.bounds
        src_crs = src.crs

    arrays = []
    used_tiles = []
    for tile in landcover["tiles"]:
        bounds = transform_bounds(src_crs, tile["crs"], *src_bounds, densify_pts=21)
        if not _bounds_overlap(bounds, tile["bounds"]):
            continue
        try:
            with rasterio.open(tile["path"]) as ds:
                window = from_bounds(*bounds, transform=ds.transform)
                arr = ds.read(
                    1,
                    window=window,
                    out_shape=FROMGLC_RESAMPLE_SHAPE,
                    boundless=True,
                    fill_value=0,
                    resampling=Resampling.nearest,
                )
                arrays.append(arr)
                used_tiles.append(str(tile["path"]))
        except Exception as exc:
            print(f"Warning: FROMGLC read failed for {tile['path']}: {exc}")
    if not arrays:
        return _empty_landcover_stats("landcover_unavailable")
    merged = np.concatenate([arr.reshape(-1) for arr in arrays]).reshape(1, -1)
    stats = _landcover_stats_from_array(
        merged,
        FROMGLC_CLASSES,
        FROMGLC_WATER_CLASSES,
        FROMGLC_NODATA_VALUES,
        str(FROMGLC_DIR),
    )
    stats["landcover_tiles_json"] = json.dumps(used_tiles, sort_keys=True)
    return stats


def passes_land_filter(stats):
    if stats["land_fraction"] == "":
        return False, "landcover_unavailable"
    if float(stats["land_fraction"]) < MIN_LAND_FRACTION:
        return False, "low_land_fraction"
    if float(stats["water_fraction"]) > MAX_WATER_FRACTION:
        return False, "high_water_fraction"
    if float(stats["nodata_fraction"]) > MAX_NODATA_FRACTION:
        return False, "high_nodata_fraction"
    return True, "ok"

def patch_starts(width, height):
    y_starts = list(range(0, height - PATCH_HR_SIZE + 1, PATCH_HR_SIZE))
    x_starts = list(range(0, width - PATCH_HR_SIZE + 1, PATCH_HR_SIZE))
    return y_starts, x_starts


def patch_key_from_offsets(yoff, xoff):
    return f"patch_r{yoff // PATCH_HR_SIZE:04d}_c{xoff // PATCH_HR_SIZE:04d}"


def write_s2_patches(zip_path, output_folder):
    band_paths = find_s2_band_paths(zip_path)
    output_folder.mkdir(parents=True, exist_ok=True)
    written = {}
    with rasterio.open(band_paths["B02"]) as example:
        y_starts, x_starts = patch_starts(example.width, example.height)
        base_profile = example.profile.copy()
        for yoff in y_starts:
            for xoff in x_starts:
                key = patch_key_from_offsets(yoff, xoff)
                out_path = output_folder / f"{key}.tif"
                if REUSE_PROCESSED_PATCHES and out_path.exists():
                    written[key] = out_path
                    continue
                window = Window(xoff, yoff, PATCH_HR_SIZE, PATCH_HR_SIZE)
                transform = rasterio.windows.transform(window, example.transform)
                profile = base_profile.copy()
                profile.update(
                    driver="GTiff",
                    width=PATCH_HR_SIZE,
                    height=PATCH_HR_SIZE,
                    count=len(S2_BANDS),
                    dtype="uint16",
                    transform=transform,
                )
                with rasterio.open(out_path, "w", **profile) as dst:
                    for band_index, band in enumerate(S2_BANDS, 1):
                        with rasterio.open(band_paths[band]) as src:
                            arr = src.read(1, window=window).astype(np.uint16)
                        dst.write(arr, band_index)
                written[key] = out_path
    return written


def write_s3_patch(s3_zip_path, example_s2_tif, out_path, swath_cache):
    if REUSE_PROCESSED_PATCHES and out_path.exists():
        return
    if s3_zip_path not in swath_cache:
        swath_cache[s3_zip_path] = load_s3_swath_bands(s3_zip_path)
    latitude, longitude, bands = swath_cache[s3_zip_path]

    from pyresample import geometry
    from pyresample.kd_tree import resample_nearest

    (
        area_id,
        description,
        proj_id,
        projection,
        new_width,
        new_height,
        area_extent,
        proj_wkt,
        new_pixel_size_x,
        new_pixel_size_y,
    ) = extract_tif_info(example_s2_tif)

    swath_def = geometry.SwathDefinition(lons=longitude, lats=latitude)
    area_def = geometry.AreaDefinition(
        area_id, description, proj_id, projection, new_width, new_height, area_extent
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=new_height,
        width=new_width,
        count=len(S3_BANDS),
        dtype="uint16",
        crs=proj_wkt,
        transform=from_origin(area_extent[0], area_extent[-1], new_pixel_size_x, new_pixel_size_y),
    ) as dst:
        for index, band in enumerate(S3_BANDS, 1):
            resampled = resample_nearest(
                swath_def,
                np.array(bands[band]),
                area_def,
                fill_value=S3_BAD_VALUE,
                radius_of_influence=500,
                epsilon=0.5,
            )
            dst.write(resampled.astype(np.uint16), index)


def read_full_array(path, bad_value):
    with rasterio.open(path) as src:
        arr = src.read().astype(np.uint16)
    if has_bad_value(arr, bad_value):
        return None
    return arr


def read_sample_arrays(t1c, t1f, t2c, t2f):
    arrays = []
    for path, bad_value in ((t1c, S3_BAD_VALUE), (t1f, S2_BAD_VALUE), (t2c, S3_BAD_VALUE), (t2f, S2_BAD_VALUE)):
        arr = read_full_array(path, bad_value)
        if arr is None:
            return None, f"bad_value:{path}"
        arrays.append(arr)
    return arrays, "ok"


def normalized_abs_change(a, b, scale):
    x = a.astype(np.float32) / scale
    y = b.astype(np.float32) / scale
    return float(np.mean(np.abs(y - x)))


def ndvi(arr, red_index, nir_index):
    red = arr[red_index].astype(np.float32)
    nir = arr[nir_index].astype(np.float32)
    denom = nir + red
    return np.divide(nir - red, denom, out=np.zeros_like(nir, dtype=np.float32), where=np.abs(denom) > 1e-6)


def phase_shift(a, b):
    x = np.mean(a.astype(np.float32), axis=0)
    y = np.mean(b.astype(np.float32), axis=0)
    x = (x - np.mean(x)) / (np.std(x) + 1e-6)
    y = (y - np.mean(y)) / (np.std(y) + 1e-6)
    f1 = np.fft.fft2(x)
    f2 = np.fft.fft2(y)
    cross_power = f1 * np.conj(f2)
    cross_power /= np.abs(cross_power) + 1e-9
    corr = np.fft.ifft2(cross_power)
    corr_abs = np.abs(corr)
    peak = np.unravel_index(np.argmax(corr_abs), corr_abs.shape)
    shifts = np.array(peak, dtype=np.float32)
    for dim, size in enumerate(corr_abs.shape):
        if shifts[dim] > size // 2:
            shifts[dim] -= size
    return {
        "s3_t1_t2_shift_y": float(shifts[0]),
        "s3_t1_t2_shift_x": float(shifts[1]),
        "s3_t1_t2_shift_magnitude": float(math.sqrt(shifts[0] ** 2 + shifts[1] ** 2)),
        "s3_t1_t2_registration_peak": float(corr_abs[peak] / (np.mean(corr_abs) + 1e-9)),
    }


def sample_metrics(arrays):
    metrics = {}
    if ENABLE_CHANGE_METRICS:
        lr_t1, hr_t1, lr_t2, hr_t2 = arrays
        metrics["s3_change_mean"] = normalized_abs_change(lr_t1, lr_t2, 1.0)
        metrics["s2_change_mean"] = normalized_abs_change(hr_t1, hr_t2, 10000.0)
        s2_ndvi_t1 = ndvi(hr_t1, 2, 3)
        s2_ndvi_t2 = ndvi(hr_t2, 2, 3)
        s3_ndvi_t1 = ndvi(lr_t1, 2, 3)
        s3_ndvi_t2 = ndvi(lr_t2, 2, 3)
        metrics["s2_ndvi_t1_mean"] = float(np.mean(s2_ndvi_t1))
        metrics["s2_ndvi_t2_mean"] = float(np.mean(s2_ndvi_t2))
        metrics["s2_ndvi_change_mean"] = float(np.mean(np.abs(s2_ndvi_t2 - s2_ndvi_t1)))
        metrics["s3_ndvi_t1_mean"] = float(np.mean(s3_ndvi_t1))
        metrics["s3_ndvi_t2_mean"] = float(np.mean(s3_ndvi_t2))
        metrics["s3_ndvi_change_mean"] = float(np.mean(np.abs(s3_ndvi_t2 - s3_ndvi_t1)))
    if ENABLE_S3_SHIFT_METRICS:
        metrics.update(phase_shift(arrays[0], arrays[2]))
    return metrics


def estimate_lmdb_map_size(group_count):
    if LMDB_MAP_SIZE is not None:
        return LMDB_MAP_SIZE
    # S2 tiles are 10980x10980 at 20 m. 512-step patches give about 441 candidates.
    estimated_samples = max(1, group_count) * 450
    estimated_bytes = int(estimated_samples * LMDB_BYTES_PER_SAMPLE_ESTIMATE * LMDB_MAP_SIZE_MARGIN)
    return max(LMDB_MIN_MAP_SIZE, estimated_bytes)


def source_product_metadata(group):
    return {
        "t1_s2": group["t1"]["s2"],
        "t1_s3": group["t1"]["s3"],
        "t2_s2": group["t2"]["s2"],
        "t2_s3": group["t2"]["s3"],
    }


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


def write_lmdb_metadata(env, total_samples, original_candidates, skipped, map_size, aux_status):
    with env.begin(write=True) as txn:
        txn.put(
            b"__metadata__",
            pickle.dumps(
                {
                    "schema": "s23_lmdb_512_v1",
                    "dataset_name": DATASET_NAME,
                    "source_dataset": SOURCE_DATASET,
                    "total_samples": total_samples,
                    "original_candidate_samples": original_candidates,
                    "skipped_samples": len(skipped),
                    "hr_shape": [4, PATCH_HR_SIZE, PATCH_HR_SIZE],
                    "lr_shape": [4, PATCH_LR_SIZE, PATCH_LR_SIZE],
                    "map_size": map_size,
                    "sample_structure": "Each sample has 4 numpy arrays: [LR_t1, HR_t1, LR_t2, HR_t2]",
                    "bad_values": {"s2": S2_BAD_VALUE, "s3": S3_BAD_VALUE},
                    "lr_factor": LR_FACTOR,
                    "fromglc_dir": str(FROMGLC_DIR),
                    "land_filter": {
                        "min_land_fraction": MIN_LAND_FRACTION,
                        "max_water_fraction": MAX_WATER_FRACTION,
                        "max_nodata_fraction": MAX_NODATA_FRACTION,
                    },
                    "ocean_filter": {
                        "shapefile": str(OCEAN_SHP_PATH),
                        "skip_if_intersects": SKIP_IF_OCEAN_INTERSECTS,
                    },
                    "aux_status": aux_status,
                }
            ),
        )


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


def group_location_metadata(group):
    location = group.get("location", {})
    lon = location.get("lon")
    lat = location.get("lat")

    return {
        "lon": lon,
        "lat": lat,
        "region": location.get("region", ""),
        "landform": location.get("landform", ""),
        "stratum": location.get("stratum", ""),
        "location_id": location.get("id", ""),
    }


def main():
    import lmdb

    files = index_downloads(DOWNLOAD_ROOT)
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    elif ALLOW_LOCAL_MANIFEST_FROM_DOWNLOADS:
        print(f"Manifest not found at {MANIFEST_PATH}; building local manifest from downloads.")
        manifest = build_manifest_from_downloads(files)
    else:
        raise SystemExit(f"Manifest not found: {MANIFEST_PATH}")
    if not files:
        raise SystemExit(f"No zip files found under {DOWNLOAD_ROOT}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    LMDB_PATH.mkdir(parents=True, exist_ok=True)

    landcover = open_landcover()
    ocean_lookup = load_ocean_lookup()
    aux_status = {
        "fromglc": bool(landcover),
        "ocean_shapefile": str(OCEAN_SHP_PATH),
    }

    groups = selected_manifest_groups(manifest)
    map_size = estimate_lmdb_map_size(len(groups))
    print(f"Building 512 LMDB at {LMDB_PATH} from {len(groups)} manifest groups")
    print(f"LMDB map_size={map_size / 1024**3:.1f} GiB")

    env = lmdb.open(str(LMDB_PATH), map_size=map_size)
    txn = env.begin(write=True)
    sample_count = 0
    original_candidates = 0
    skipped = []
    group_results = []
    index_rows = []
    swath_cache = {}

    for group_index, group in enumerate(groups, start=1):
        group_id = group["group_id"]
        group_dir = PROCESSED_ROOT / group_id
        print(f"\n=== Group {group_index}/{len(groups)}: {group_id} ===")
        try:
            mgrs_tile, pair_dates = validate_group_pairing(group)
            s2_t1 = find_zip(group["t1"]["s2"], files)
            s3_t1 = find_zip(group["t1"]["s3"], files)
            s2_t2 = find_zip(group["t2"]["s2"], files)
            s3_t2 = find_zip(group["t2"]["s3"], files)
            for path in (s2_t1, s3_t1, s2_t2, s3_t2):
                validate_zip(path)

            t1f_dir = group_dir / "t1f_512"
            t2f_dir = group_dir / "t2f_512"
            t1c_dir = group_dir / "t1c_64"
            t2c_dir = group_dir / "t2c_64"
            t1f_patches = write_s2_patches(s2_t1, t1f_dir)
            t2f_patches = write_s2_patches(s2_t2, t2f_dir)
            common_keys = sorted(set(t1f_patches) & set(t2f_patches))
            print(f"S2 512 patches: t1={len(t1f_patches)}, t2={len(t2f_patches)}, common={len(common_keys)}")

            group_written = 0
            group_skipped = Counter()
            loc_meta = group_location_metadata(group)
            t1_parts = date_parts(group["t1"]["date"])
            t2_parts = date_parts(group["t2"]["date"])
            for patch_index, patch_key in enumerate(common_keys, start=1):
                original_candidates += 1
                t1f = t1f_patches[patch_key]
                t2f = t2f_patches[patch_key]
                sea_stats = ocean_stats(ocean_lookup, t1f)
                if SKIP_IF_OCEAN_INTERSECTS and sea_stats["ocean_intersects"]:
                    skipped.append({"group_id": group_id, "patch_id": patch_key, "reason": "ocean_intersects"})
                    group_skipped["ocean_intersects"] += 1
                    continue

                lc_stats = landcover_stats(landcover, t1f)
                ok_land, land_reason = passes_land_filter(lc_stats)
                if not ok_land:
                    skipped.append({"group_id": group_id, "patch_id": patch_key, "reason": land_reason})
                    group_skipped[land_reason] += 1
                    continue

                t1c = t1c_dir / f"{patch_key}.tif"
                t2c = t2c_dir / f"{patch_key}.tif"
                write_s3_patch(s3_t1, t1f, t1c, swath_cache)
                write_s3_patch(s3_t2, t2f, t2c, swath_cache)
                arrays, reason = read_sample_arrays(t1c, t1f, t2c, t2f)
                if arrays is None:
                    skipped.append({"group_id": group_id, "patch_id": patch_key, "reason": reason})
                    group_skipped[reason] += 1
                    continue

                lmdb_key = f"{sample_count:08d}"
                sample_id = f"{DATASET_NAME}_{lmdb_key}"
                metrics = sample_metrics(arrays)
                metadata = {
                    "schema": "s23_sample_512_v1",
                    "sample_id": sample_id,
                    "lmdb_key": lmdb_key,
                    "dataset_name": DATASET_NAME,
                    "source_dataset": SOURCE_DATASET,
                    "group_id": group_id,
                    "patch_id": patch_key,
                    "mgrs_tile": mgrs_tile,
                    "pair_dates": pair_dates,
                    "t1_date": group["t1"]["date"],
                    "t2_date": group["t2"]["date"],
                    "t1_year": t1_parts["year"],
                    "t2_year": t2_parts["year"],
                    "t1_month": t1_parts["month"],
                    "t2_month": t2_parts["month"],
                    "t1_season": t1_parts["season"],
                    "t2_season": t2_parts["season"],
                    "interval_days": interval_days(group["t1"]["date"], group["t2"]["date"]),
                    "interval_months": group.get("interval_months", ""),
                    "target_interval_months": group.get("target_interval_months", ""),
                    "target_start_year": group.get("target_start_year", ""),
                    "source_products": source_product_metadata(group),
                }
                metadata.update(loc_meta)
                metadata.update(sea_stats)
                metadata.update(lc_stats)
                metadata.update(metrics)

                sample_data = {"arrays": pack_arrays(arrays), "metadata": metadata}
                txn.put(lmdb_key.encode(), pickle.dumps(sample_data))
                index_rows.append(dict(metadata))
                sample_count += 1
                group_written += 1

                if sample_count % LMDB_BATCH_SIZE == 0:
                    txn.commit()
                    txn = env.begin(write=True)
                    write_index_csv(index_rows, SAMPLE_INDEX_CSV)
                    print(f"  wrote sample {sample_count} at patch {patch_index}/{len(common_keys)}")

            group_results.append(
                {
                    "group_id": group_id,
                    "status": "processed",
                    "mgrs_tile": mgrs_tile,
                    "candidate_patches": len(common_keys),
                    "samples_written": group_written,
                    "skipped": dict(group_skipped),
                }
            )
            print(f"result {group_id}: written={group_written}/{len(common_keys)}, skipped={dict(group_skipped)}")
        except Exception as exc:
            skipped.append({"group_id": group_id, "reason": str(exc)})
            group_results.append({"group_id": group_id, "status": "error", "reason": str(exc)})
            print(f"error {group_id}: {exc}")
            if STOP_ON_PROCESS_ERROR:
                raise

    txn.commit()
    write_lmdb_metadata(env, sample_count, original_candidates, skipped, map_size, aux_status)
    env.sync()
    env.close()

    write_index_csv(index_rows, SAMPLE_INDEX_CSV)
    report = {
        "schema": "s23_lmdb_512_build_report_v1",
        "dataset_name": DATASET_NAME,
        "source_dataset": SOURCE_DATASET,
        "lmdb_path": str(LMDB_PATH),
        "sample_index_csv": str(SAMPLE_INDEX_CSV),
        "samples": sample_count,
        "original_candidate_samples": original_candidates,
        "skipped_samples": len(skipped),
        "manifest_groups": len(manifest["groups"]),
        "processed_groups": len(groups),
        "patch_hr_size": PATCH_HR_SIZE,
        "patch_lr_size": PATCH_LR_SIZE,
        "lr_factor": LR_FACTOR,
        "aux_status": aux_status,
        "land_filter": {
            "min_land_fraction": MIN_LAND_FRACTION,
            "max_water_fraction": MAX_WATER_FRACTION,
            "max_nodata_fraction": MAX_NODATA_FRACTION,
        },
        "group_results": group_results,
        "skipped": skipped,
    }
    BUILD_REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"LMDB written: {LMDB_PATH}")
    print(f"Sample index written: {SAMPLE_INDEX_CSV}")
    print(f"Build report written: {BUILD_REPORT_JSON}")
    if not KEEP_PROCESSED_PATCHES:
        print("KEEP_PROCESSED_PATCHES=False is set, but cleanup is intentionally manual in this version.")


if __name__ == "__main__":
    main()
















