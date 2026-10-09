"""Shared helpers for the manifest-free S2/S3 512-tile pipeline."""

import csv
import io
import json
import math
import re
import struct
import time
import threading
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import importlib.util
import numpy as np
import rasterio
import xarray as xr
from rasterio.enums import Resampling
from rasterio.windows import Window, bounds as window_bounds, from_bounds
from rasterio.warp import transform_bounds
from shapely.geometry import Point, Polygon, box
from shapely.strtree import STRtree


CONFIG_PATH = Path(__file__).with_name("00_config.py")
spec = importlib.util.spec_from_file_location("s23_config", CONFIG_PATH)
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)

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
_SWATH_CACHE_LOCK = threading.Lock()



def progress(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def progress_every(done, total):
    if total <= 0:
        return 1
    return max(1, min(100, total // 100 if total >= 100 else 1))

def read_json(path, default):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fd:
        writer = csv.DictWriter(fd, fieldnames=fields)
        if fields:
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


def product_name_from_path(path):
    name = Path(path).name
    return name[:-4] if name.lower().endswith(".zip") else name


def filename_from_url(url):
    name = unquote(Path(urlparse(url).path).name)
    return name.split("?")[0]


def extract_mgrs_from_s2_name(name):
    match = re.search(r"_T([0-9]{2}[A-Z]{3})_", str(name))
    return match.group(1) if match else ""


def extract_s2_date(name):
    match = re.search(r"_MSIL2A_(\d{8})T", str(name))
    return match.group(1) if match else ""


def extract_s3_date(name):
    match = re.search(r"OL_1_EFR____(\d{8})T", str(name))
    return match.group(1) if match else ""


def sensor_from_name(name):
    if "MSIL2A_" in str(name):
        return "s2"
    if "OL_1_EFR" in str(name):
        return "s3"
    return ""


def product_date(name):
    sensor = sensor_from_name(name)
    if sensor == "s2":
        return extract_s2_date(name)
    if sensor == "s3":
        return extract_s3_date(name)
    return ""


def parse_date(text):
    return datetime.strptime(text, "%Y%m%d")


def interval_days(t1, t2):
    return (parse_date(t2) - parse_date(t1)).days



def configured_allowed_years():
    years = getattr(cfg, "ALLOWED_YEARS", None)
    if years in (None, "", [], (), set()):
        return set()
    if isinstance(years, str):
        return {item.strip() for item in years.split(",") if item.strip()}
    return {str(item) for item in years}


def product_path_allowed_by_name(path):
    name = product_name_from_path(path)
    sensor = sensor_from_name(name)
    if not sensor:
        return False
    date = product_date(name)
    if not date:
        return False
    years = configured_allowed_years()
    if years and date[:4] not in years:
        return False
    return True

def validate_zip(path, sensor):
    path = Path(path)
    if not path.exists() or path.stat().st_size <= 0:
        return False, "missing_or_empty"
    try:
        with zipfile.ZipFile(path, "r") as zip_ref:
            bad = zip_ref.testzip()
            if bad:
                return False, f"bad_member:{bad}"
            names = zip_ref.namelist()
    except zipfile.BadZipFile:
        return False, "bad_zip"
    if sensor == "s2":
        for band in cfg.S2_BANDS:
            if not any(band in name and "20m" in name and name.endswith(".jp2") for name in names):
                return False, f"missing_s2_band:{band}"
    elif sensor == "s3":
        if not any("geo_coordinates.nc" in name for name in names):
            return False, "missing_s3_geo"
        for band in cfg.S3_BANDS:
            if not any(PurePosixPath(name).name.startswith(band + "_") and name.endswith("radiance.nc") for name in names):
                return False, f"missing_s3_band:{band}"
    else:
        return False, "unknown_sensor"
    return True, "ok"


def _scan_one_zip(path):
    name = product_name_from_path(path)
    sensor = sensor_from_name(name)
    if not sensor:
        return None
    date = product_date(name)
    ok, reason = validate_zip(path, sensor)
    return {
        "name": name,
        "path": str(path),
        "sensor": sensor,
        "date": date,
        "mgrs_tile": extract_mgrs_from_s2_name(name) if sensor == "s2" else "",
        "valid": ok,
        "reason": reason,
        "size": Path(path).stat().st_size if Path(path).exists() else 0,
    }



def scan_product_names(roots, label="scan_product_names"):
    items = []
    total_seen = 0
    rejected_by_name = 0
    years = configured_allowed_years()
    year_text = ",".join(sorted(years)) if years else "all"
    progress(f"{label}: filename-only scan years={year_text}")
    for root in roots:
        root = Path(root)
        progress(f"{label}: scanning root {root}")
        if not root.exists():
            continue
        for path in root.rglob("*.zip"):
            total_seen += 1
            name = product_name_from_path(path)
            sensor = sensor_from_name(name)
            date = product_date(name) if sensor else ""
            if not sensor or not date:
                rejected_by_name += 1
            elif years and date[:4] not in years:
                rejected_by_name += 1
            else:
                items.append({
                    "name": name,
                    "path": str(path),
                    "sensor": sensor,
                    "date": date,
                    "mgrs_tile": extract_mgrs_from_s2_name(name) if sensor == "s2" else "",
                    "valid": True,
                    "reason": "name_prefilter_only",
                    "size": 0,
                })
            if total_seen % 10000 == 0:
                progress(f"{label}: discovered {total_seen} zip files, kept_by_name={len(items)}, rejected_by_name={rejected_by_name}")
    progress(f"{label}: discovered {total_seen} zip files, kept {len(items)} after filename prefilter, rejected {rejected_by_name}")
    return sorted(items, key=lambda item: item["path"])

def scan_local_zips(roots, label="scan_local_zips"):
    paths = []
    total_seen = 0
    rejected_by_name = 0
    years = configured_allowed_years()
    year_text = ",".join(sorted(years)) if years else "all"
    progress(f"{label}: filename prefilter years={year_text}")
    for root in roots:
        root = Path(root)
        progress(f"{label}: scanning root {root}")
        if not root.exists():
            continue
        for path in root.rglob("*.zip"):
            total_seen += 1
            if product_path_allowed_by_name(path):
                paths.append(path)
            else:
                rejected_by_name += 1
            if total_seen % 10000 == 0:
                progress(f"{label}: discovered {total_seen} zip files, kept_by_name={len(paths)}, rejected_by_name={rejected_by_name}")
    total = len(paths)
    progress(f"{label}: discovered {total_seen} zip files, kept {total} after filename prefilter, rejected {rejected_by_name}")
    if not paths:
        return []
    workers = max(1, int(getattr(cfg, "FILE_SCAN_WORKERS", 1)))
    step = progress_every(0, total)
    if workers == 1:
        items = []
        for idx, path in enumerate(paths, 1):
            item = _scan_one_zip(path)
            if item:
                items.append(item)
            if idx == total or idx % step == 0:
                progress(f"{label}: validated {idx}/{total} prefiltered zip files")
        return sorted(items, key=lambda item: item["path"])
    items = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_scan_one_zip, path) for path in paths]
        for idx, future in enumerate(as_completed(futures), 1):
            item = future.result()
            if item:
                items.append(item)
            if idx == total or idx % step == 0:
                progress(f"{label}: validated {idx}/{total} prefiltered zip files")
    return sorted(items, key=lambda item: item["path"])





def _bounds_from_coord_sequence(coords, acc):
    if coords is None:
        return
    if isinstance(coords, (list, tuple)):
        if len(coords) >= 2 and all(isinstance(v, (int, float, np.integer, np.floating)) for v in coords[:2]):
            lon = float(coords[0])
            lat = float(coords[1])
            if math.isfinite(lon) and math.isfinite(lat):
                acc.append((lon, lat))
            return
        for item in coords:
            _bounds_from_coord_sequence(item, acc)


def bounds_from_geometry_like(geom):
    """Return WGS84 bbox from OGR geometry, GeoJSON dict/string, WKT, or coordinates."""
    if geom is None:
        return ()
    if isinstance(geom, (list, tuple)) and len(geom) == 4 and all(isinstance(v, (int, float, np.integer, np.floating)) for v in geom):
        return tuple(float(v) for v in geom)
    try:
        env = geom.GetEnvelope()
        return (float(env[0]), float(env[2]), float(env[1]), float(env[3]))
    except Exception:
        pass
    if isinstance(geom, str):
        text = geom.strip()
        if not text:
            return ()
        try:
            return bounds_from_geometry_like(json.loads(text))
        except Exception:
            pass
        nums = [float(v) for v in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)]
        if len(nums) >= 4:
            pts = list(zip(nums[0::2], nums[1::2]))
            lons = [p[0] for p in pts if math.isfinite(p[0]) and math.isfinite(p[1])]
            lats = [p[1] for p in pts if math.isfinite(p[0]) and math.isfinite(p[1])]
            if lons and lats:
                return (min(lons), min(lats), max(lons), max(lats))
        return ()
    if isinstance(geom, dict):
        if "bbox" in geom and len(geom["bbox"]) >= 4:
            b = geom["bbox"]
            return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        coords = geom.get("coordinates") if "coordinates" in geom else geom.get("geometry", {}).get("coordinates")
        acc = []
        _bounds_from_coord_sequence(coords, acc)
        if acc:
            lons = [pt[0] for pt in acc]
            lats = [pt[1] for pt in acc]
            return (min(lons), min(lats), max(lons), max(lats))
    return ()

def bbox_intersects(a, b):
    if not a or not b:
        return False
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])



def bbox_contains(outer, inner):
    if not outer or not inner:
        return False
    return float(outer[0]) <= float(inner[0]) and float(outer[1]) <= float(inner[1]) and float(outer[2]) >= float(inner[2]) and float(outer[3]) >= float(inner[3])

def bbox_overlap_ratio(reference_bbox, candidate_bbox):
    if not reference_bbox or not candidate_bbox:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in reference_bbox]
    bx1, by1, bx2, by2 = [float(v) for v in candidate_bbox]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    reference_area = max((ax2 - ax1) * (ay2 - ay1), 1e-12)
    return float(intersection / reference_area)


def number_summary(values):
    values = [float(v) for v in values if v not in (None, "")]
    if not values:
        return {"count": 0, "min": "", "max": "", "mean": "", "median": ""}
    values = sorted(values)
    mid = len(values) // 2
    median = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
    return {"count": len(values), "min": values[0], "max": values[-1], "mean": sum(values) / len(values), "median": median}


def histogram(values):
    counts = Counter(str(v) for v in values if v not in (None, ""))
    return dict(sorted(counts.items()))


def read_s2_cloud(zip_path):
    try:
        from osgeo import gdal

        ds = gdal.Open(str(zip_path))
        if ds is None:
            return ""
        metadata = ds.GetMetadata()
        return float(metadata["CLOUD_COVERAGE_ASSESSMENT"])
    except Exception:
        return ""

from osgeo import gdal, ogr, osr
import geojson
def read_s2_data_percentage(zip_file):
    try:
        s = gdal.Open(zip_file)
        metadata = s.GetMetadata()
        src = gdal.Open(s.GetSubDatasets()[0][0])
        projection = src.GetProjection()
        projection_EPSG = int(s.GetSubDatasets()[0][0].split("EPSG_")[1])
    
        gt = src.GetGeoTransform()
        lon1 = gt[0]
        lon2 = lon1 + gt[1] * src.RasterXSize
        lat1 = gt[-3]
        lat2 = lat1 + gt[5] * src.RasterYSize
    
        # 转换坐标
        srs4326 = osr.SpatialReference()
        srs4326.ImportFromEPSG(4326)
        in_srs = osr.SpatialReference()
        in_srs.ImportFromEPSG(projection_EPSG)
        coordinate_transforer = osr.CoordinateTransformation(in_srs, srs4326)
    
        # 注意，这里是反的，如果不对，可能是和gdal2.x 或者 3.x有关
        lat1, lon1 = coordinate_transforer.TransformPoint(lon1, lat1)[:2]
        lat2, lon2 = coordinate_transforer.TransformPoint(lon2, lat2)[:2]
    
        geometry = geojson.geometry.Polygon(
            [[(lon1, lat1), (lon2, lat1), (lon2, lat2), (lon1, lat2), (lon1, lat1)]])
    
        footprint_geo = ogr.CreateGeometryFromJson(str(geometry))
        footprint_with_data_geo = ogr.CreateGeometryFromWkt(metadata['FOOTPRINT'])
    
        # 获得geojson
        footprint = footprint_geo.ExportToJson()
        footprint_with_data = footprint_with_data_geo.ExportToJson()
    
        # 获得数据占比
        data_percentage = footprint_with_data_geo.GetArea() / footprint_geo.GetArea()
        if data_percentage > 1:
            data_percentage = 1
        print(data_percentage)
        return data_percentage
    except Exception as e:
        return ""

        

# def read_s2_data_percentage(zip_path):
#     """Return server-style valid S2 footprint area ratio.

#     The server implementation compares GDAL product metadata FOOTPRINT area with
#     the full 20 m band area. Missing FOOTPRINT is treated as unknown by default,
#     not as valid, because v3 uses this as a hard pre-pairing quality filter.
#     """
#     try:
#         from osgeo import gdal, ogr

#         ds = gdal.Open(str(zip_path))
#         if ds is None:
#             return ""
#         metadata = ds.GetMetadata()
#         footprint = metadata.get("FOOTPRINT") or metadata.get("footprint")
#         if not footprint:
#             return 1.0 if getattr(cfg, "ALLOW_MISSING_S2_FOOTPRINT", False) else ""
#         geom = ogr.CreateGeometryFromWkt(footprint)
#         if geom is None:
#             return ""
#         footprint_area = abs(float(geom.GetArea()))
#         band_paths = find_s2_band_paths(zip_path)
#         with rasterio.open(band_paths["B02"]) as src:
#             bounds = src.bounds
#             full_area = abs(float((bounds.right - bounds.left) * (bounds.top - bounds.bottom)))
#         if full_area <= 0:
#             return ""
#         ratio = footprint_area / full_area
#         if ratio < 0:
#             return ""
#         return min(1.0, float(ratio))
#     except Exception:
#         return ""

def s2_bounds_wgs84(zip_path):
    band_paths = find_s2_band_paths(zip_path)
    with rasterio.open(band_paths["B02"]) as src:
        return tuple(float(v) for v in transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21))


def s3_bounds_wgs84(zip_path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        names = zip_ref.namelist()
        geo_name = [name for name in names if "geo_coordinates.nc" in name][0]
        with zip_ref.open(geo_name) as geo_file:
            geo_data = xr.open_dataset(io.BytesIO(geo_file.read()))
            lat = np.asarray(geo_data["latitude"].values, dtype=np.float64)
            lon = np.asarray(geo_data["longitude"].values, dtype=np.float64)
    valid = np.isfinite(lat) & np.isfinite(lon)
    if not np.any(valid):
        return ()
    return (float(np.min(lon[valid])), float(np.min(lat[valid])), float(np.max(lon[valid])), float(np.max(lat[valid])))

@lru_cache(maxsize=2048)
def find_s2_band_paths(zip_path):
    out = {}
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        names = zip_ref.namelist()
    for band in cfg.S2_BANDS:
        matches = [name for name in names if band in name and "20m" in name and name.endswith(".jp2")]
        if not matches:
            raise RuntimeError(f"Cannot find {band} 20m JP2 in {zip_path}")
        out[band] = "/vsizip/" + str(zip_path) + "/" + matches[0]
    return out


def s2_patch_grid(s2_zip_path):
    band_paths = find_s2_band_paths(s2_zip_path)
    with rasterio.open(band_paths["B02"]) as src:
        rows = src.height // cfg.PATCH_HR_SIZE
        cols = src.width // cfg.PATCH_HR_SIZE
    return rows, cols


def iter_s2_patch_windows(s2_zip_path):
    rows, cols = s2_patch_grid(s2_zip_path)
    for row in range(rows):
        for col in range(cols):
            window = Window(col * cfg.PATCH_HR_SIZE, row * cfg.PATCH_HR_SIZE, cfg.PATCH_HR_SIZE, cfg.PATCH_HR_SIZE)
            yield f"tile_r{row:04d}_c{col:04d}", row, col, window


def read_s2_patch(s2_zip_path, window):
    arrays = []
    for band in cfg.S2_BANDS:
        with rasterio.open(find_s2_band_paths(s2_zip_path)[band]) as src:
            arrays.append(src.read(1, window=window).astype(np.uint16))
    arr = np.stack(arrays, axis=0)
    if has_bad_value(arr, cfg.S2_BAD_VALUE):
        return None
    return arr


def s2_valid_grid_ref(s2_zip_path):
    band_paths = find_s2_band_paths(s2_zip_path)
    with rasterio.open(band_paths["B02"]) as src:
        rows = src.height // cfg.PATCH_HR_SIZE
        cols = src.width // cfg.PATCH_HR_SIZE
        height = rows * cfg.PATCH_HR_SIZE
        width = cols * cfg.PATCH_HR_SIZE
        window = Window(0, 0, width, height)
        bounds = window_bounds(window, src.transform)
        return {
            "rows": rows,
            "cols": cols,
            "height": height,
            "width": width,
            "window": window,
            "crs": src.crs,
            "bounds": bounds,
            "bounds_wgs84": transform_bounds(src.crs, "EPSG:4326", *bounds, densify_pts=21),
        }


def read_s2_valid_grid(s2_zip_path, grid_ref=None):
    grid_ref = grid_ref or s2_valid_grid_ref(s2_zip_path)
    window = grid_ref["window"]
    arrays = []
    for band in cfg.S2_BANDS:
        with rasterio.open(find_s2_band_paths(s2_zip_path)[band]) as src:
            arrays.append(src.read(1, window=window).astype(np.uint16))
    return np.stack(arrays, axis=0)


def s2_patch_ref(s2_zip_path, window):
    band_paths = find_s2_band_paths(s2_zip_path)
    with rasterio.open(band_paths["B02"]) as src:
        b = window_bounds(window, src.transform)
        return src.crs, b, transform_bounds(src.crs, "EPSG:4326", *b, densify_pts=21)


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
            band = PurePosixPath(name).name.split("_")[0]
            if band in cfg.S3_BANDS and name.endswith("radiance.nc"):
                with zip_ref.open(name) as rad_file:
                    bands[band] = np.squeeze(xr.open_dataset(io.BytesIO(rad_file.read()), decode_cf=False).to_dataarray().values)
    missing = [band for band in cfg.S3_BANDS if band not in bands]
    if missing:
        raise RuntimeError(f"Missing S3 bands in {s3_zip_path}: {missing}")
    return latitude, longitude, bands


def area_projection(crs):
    epsg = crs.to_epsg()
    if epsg is None or epsg < 32601 or epsg > 32760:
        raise RuntimeError(f"Expected Sentinel-2 UTM CRS, got EPSG:{epsg}")
    zone = epsg % 100
    projection = {"proj": "utm", "zone": zone, "datum": "WGS84", "units": "m"}
    if epsg >= 32700:
        projection["south"] = True
    return f"utm_{epsg}", projection


def resample_s3_patch(s3_zip_path, s2_crs, s2_bounds, swath_cache):
    key = str(s3_zip_path)
    from pyresample import geometry
    from pyresample.kd_tree import resample_nearest

    if key not in swath_cache:
        with _SWATH_CACHE_LOCK:
            if key not in swath_cache:
                latitude, longitude, bands = load_s3_swath_bands(s3_zip_path)
                swath_cache[key] = {
                    "bands": bands,
                    "swath_def": geometry.SwathDefinition(lons=longitude, lats=latitude),
                }
    cached = swath_cache[key]
    bands = cached["bands"]
    swath_def = cached["swath_def"]

    proj_id, projection = area_projection(s2_crs)
    area_def = geometry.AreaDefinition(
        "s2_patch_512",
        "Sentinel-2 512 patch at 20m",
        proj_id,
        projection,
        cfg.PATCH_LR_SIZE,
        cfg.PATCH_LR_SIZE,
        s2_bounds,
    )
    arrays = []
    for band in cfg.S3_BANDS:
        arr = resample_nearest(swath_def, np.array(bands[band]), area_def, fill_value=cfg.S3_BAD_VALUE, radius_of_influence=500, epsilon=0.5)
        arrays.append(arr.astype(np.uint16))
    out = np.stack(arrays, axis=0)
    if has_bad_value(out, cfg.S3_BAD_VALUE):
        return None
    return out


def resample_s3_grid(s3_zip_path, s2_crs, s2_bounds, lr_width, lr_height, swath_cache):
    key = str(s3_zip_path)
    from pyresample import geometry
    from pyresample.kd_tree import resample_nearest

    if key not in swath_cache:
        with _SWATH_CACHE_LOCK:
            if key not in swath_cache:
                latitude, longitude, bands = load_s3_swath_bands(s3_zip_path)
                swath_cache[key] = {
                    "bands": bands,
                    "swath_def": geometry.SwathDefinition(lons=longitude, lats=latitude),
                }
    cached = swath_cache[key]
    bands = cached["bands"]
    swath_def = cached["swath_def"]

    proj_id, projection = area_projection(s2_crs)
    area_def = geometry.AreaDefinition(
        "s2_valid_grid",
        "Sentinel-2 valid patch grid at LR resolution",
        proj_id,
        projection,
        int(lr_width),
        int(lr_height),
        s2_bounds,
    )
    arrays = []
    for band in cfg.S3_BANDS:
        arr = resample_nearest(swath_def, np.array(bands[band]), area_def, fill_value=cfg.S3_BAD_VALUE, radius_of_influence=500, epsilon=0.5)
        arrays.append(arr.astype(np.uint16))
    return np.stack(arrays, axis=0)


def choose_s3_grid(candidates, s2_crs, s2_bounds, lr_width, lr_height, swath_cache):
    grids = []
    for item in candidates[: cfg.MAX_S3_CANDIDATES_PER_DATE]:
        try:
            arr = resample_s3_grid(Path(item["path"]), s2_crs, s2_bounds, lr_width, lr_height, swath_cache)
        except Exception:
            continue
        grids.append({
            "array": arr,
            "name": item["name"],
            "path": item["path"],
            "score": 1.0,
            "overlap_ratio": float(item.get("s2_overlap_ratio", 0.0)),
        })
    return sorted(grids, key=lambda x: (-x["overlap_ratio"], x["name"]))

def choose_s3_patch(candidates, s2_crs, s2_bounds, swath_cache):
    best = None
    for item in candidates[: cfg.MAX_S3_CANDIDATES_PER_DATE]:
        try:
            arr = resample_s3_patch(Path(item["path"]), s2_crs, s2_bounds, swath_cache)
        except Exception as exc:
            continue
        if arr is None:
            continue
        valid_score = 1.0
        overlap_score = float(item.get("s2_overlap_ratio", 0.0))
        rank_score = (valid_score, overlap_score)
        if best is None or rank_score > best["rank_score"]:
            best = {"array": arr, "name": item["name"], "path": item["path"], "score": valid_score, "overlap_ratio": overlap_score, "rank_score": rank_score}
    return best


def has_bad_value(arr, bad_value):
    if np.any(arr == bad_value):
        return True
    if np.issubdtype(arr.dtype, np.floating):
        return bool(np.any(np.isnan(arr)))
    return False


def pack_arrays(arrays):
    import blosc

    buf = io.BytesIO()
    for arr in arrays:
        np.save(buf, arr, allow_pickle=False)
    return blosc.compress(buf.getvalue(), cname="lz4", clevel=3)


def _ring_area(points):
    total = 0.0
    for idx in range(len(points)):
        x1, y1 = points[idx]
        x2, y2 = points[(idx + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def _read_ocean_rings(path):
    data = Path(path).read_bytes()
    offset = 100
    while offset + 8 <= len(data):
        _, words = struct.unpack(">2i", data[offset:offset + 8])
        offset += 8
        content = data[offset:offset + words * 2]
        offset += words * 2
        if len(content) < 44 or struct.unpack("<i", content[:4])[0] != 5:
            continue
        n_parts, n_points = struct.unpack("<2i", content[36:44])
        parts_off = 44
        pts_off = parts_off + 4 * n_parts
        starts = list(struct.unpack("<" + "i" * n_parts, content[parts_off:pts_off])) + [n_points]
        raw = struct.unpack("<" + "d" * (2 * n_points), content[pts_off:pts_off + 16 * n_points])
        for idx in range(n_parts):
            pts = [(raw[i * 2], raw[i * 2 + 1]) for i in range(starts[idx], starts[idx + 1])]
            if len(pts) >= 4:
                yield pts


def load_ocean_lookup():
    if not cfg.OCEAN_SHP_PATH.exists():
        raise SystemExit(f"Missing ocean shapefile: {cfg.OCEAN_SHP_PATH}")
    exteriors, holes = [], []
    for pts in _read_ocean_rings(cfg.OCEAN_SHP_PATH):
        area = _ring_area(pts)
        if abs(area) < 1e-10:
            continue
        if pts[0] != pts[-1]:
            pts = pts + [pts[0]]
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if area < 0:
            exteriors.append(poly)
        else:
            holes.append(poly)
    tree = STRtree(holes) if holes else None

    def hole_candidates(point):
        if tree is None:
            return []
        items = tree.query(point)
        return [holes[int(item)] if isinstance(item, (int, np.integer)) else item for item in items]

    def is_ocean(bounds_wgs84):
        minx, miny, maxx, maxy = [float(v) for v in bounds_wgs84]
        probes = [((minx + maxx) / 2, (miny + maxy) / 2), (minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)]
        for lon, lat in probes:
            pt = Point(lon, lat)
            if not any(poly.contains(pt) or poly.touches(pt) for poly in exteriors):
                return False
            if any(hole.contains(pt) or hole.touches(pt) for hole in hole_candidates(pt)):
                return False
        return True

    return is_ocean


def scan_landcover_tiles():
    tiles = []
    for path in sorted(cfg.FROMGLC_DIR.glob("*.tif")):
        with rasterio.open(path) as ds:
            tiles.append({"path": path, "bounds": tuple(ds.bounds), "crs": ds.crs})
    if not tiles:
        raise SystemExit(f"No FROMGLC tif files found under {cfg.FROMGLC_DIR}")
    crs_values = {str(tile["crs"]) for tile in tiles}
    if len(crs_values) != 1:
        return {"tiles": tiles, "tree": None, "geoms": [], "geom_index": {}, "crs": None}
    geoms = [box(*tile["bounds"]) for tile in tiles]
    return {
        "tiles": tiles,
        "tree": STRtree(geoms),
        "geoms": geoms,
        "geom_index": {geom.wkb: idx for idx, geom in enumerate(geoms)},
        "crs": tiles[0]["crs"],
    }


def overlap(a, b):
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def _landcover_candidate_tiles(index, bounds):
    tiles = index["tiles"] if isinstance(index, dict) else index
    if not isinstance(index, dict) or index.get("tree") is None:
        return [(tile, bounds if tile["crs"] == index.get("crs") else None) for tile in tiles]
    query_geom = box(*bounds)
    candidates = []
    for item in index["tree"].query(query_geom):
        if isinstance(item, (int, np.integer)):
            idx = int(item)
        else:
            idx = index["geom_index"].get(item.wkb)
        if idx is not None:
            candidates.append(tiles[idx])
    return candidates


def landcover_stats(tiles, src_crs, src_bounds):
    arrays, used = [], []
    if isinstance(tiles, dict) and tiles.get("crs") is not None:
        lc_crs = tiles["crs"]
        query_bounds = transform_bounds(src_crs, lc_crs, *src_bounds, densify_pts=21)
        candidates = _landcover_candidate_tiles(tiles, query_bounds)
        for tile in candidates:
            if not overlap(query_bounds, tile["bounds"]):
                continue
            with rasterio.open(tile["path"]) as ds:
                arr = ds.read(1, window=from_bounds(*query_bounds, transform=ds.transform), out_shape=cfg.FROMGLC_RESAMPLE_SHAPE, boundless=True, fill_value=0, resampling=Resampling.nearest)
            arrays.append(arr)
            used.append(str(tile["path"]))
    else:
        raw_tiles = tiles["tiles"] if isinstance(tiles, dict) else tiles
        for tile in raw_tiles:
            bounds = transform_bounds(src_crs, tile["crs"], *src_bounds, densify_pts=21)
            if not overlap(bounds, tile["bounds"]):
                continue
            with rasterio.open(tile["path"]) as ds:
                arr = ds.read(1, window=from_bounds(*bounds, transform=ds.transform), out_shape=cfg.FROMGLC_RESAMPLE_SHAPE, boundless=True, fill_value=0, resampling=Resampling.nearest)
            arrays.append(arr)
            used.append(str(tile["path"]))
    if not arrays:
        return {"land_fraction": "", "water_fraction": "", "nodata_fraction": "", "landcover_major": "landcover_unavailable", "landcover_tiles_json": "[]"}
    flat = np.concatenate([arr.reshape(-1) for arr in arrays])
    counts = Counter(int(v) for v in flat)
    total = int(flat.size)
    water = sum(counts.get(v, 0) for v in FROMGLC_WATER_CLASSES)
    nodata = sum(counts.get(v, 0) for v in FROMGLC_NODATA_VALUES)
    land_counts = {v: c for v, c in counts.items() if v not in FROMGLC_WATER_CLASSES and v not in FROMGLC_NODATA_VALUES}
    land = sum(land_counts.values())
    major = max(land_counts.items(), key=lambda x: x[1])[0] if land_counts else (60 if water else 0)
    return {
        "land_fraction": land / total,
        "water_fraction": water / total,
        "nodata_fraction": nodata / total,
        "landcover_major": FROMGLC_CLASSES.get(major, f"class_{major}"),
        "landcover_counts_json": json.dumps({FROMGLC_CLASSES.get(k, f"class_{k}"): v for k, v in sorted(counts.items())}, sort_keys=True),
        "landcover_tiles_json": json.dumps(used, sort_keys=True),
    }


def passes_land_filter(stats):
    if stats["land_fraction"] == "":
        return False, "landcover_unavailable"
    if float(stats["land_fraction"]) < cfg.MIN_LAND_FRACTION:
        return False, "low_land_fraction"
    if float(stats["water_fraction"]) > cfg.MAX_WATER_FRACTION:
        return False, "high_water_fraction"
    if float(stats["nodata_fraction"]) > cfg.MAX_NODATA_FRACTION:
        return False, "high_nodata_fraction"
    return True, "ok"


def ndvi(arr, red=2, nir=3):
    r = arr[red].astype(np.float32)
    n = arr[nir].astype(np.float32)
    denom = n + r
    return np.divide(n - r, denom, out=np.zeros_like(n, dtype=np.float32), where=np.abs(denom) > 1e-6)


def sample_metrics(arrays):
    lr1, hr1, lr2, hr2 = arrays
    return {
        "s2_change_mean": float(np.mean(np.abs(hr2.astype(np.float32) / 10000.0 - hr1.astype(np.float32) / 10000.0))),
        "s3_change_mean": float(np.mean(np.abs(lr2.astype(np.float32) - lr1.astype(np.float32)))),
        "s2_ndvi_t1_mean": float(np.mean(ndvi(hr1))),
        "s2_ndvi_t2_mean": float(np.mean(ndvi(hr2))),
        "s3_ndvi_t1_mean": float(np.mean(ndvi(lr1))),
        "s3_ndvi_t2_mean": float(np.mean(ndvi(lr2))),
    }


def estimate_map_size(candidate_samples):
    if cfg.LMDB_MAP_SIZE:
        return cfg.LMDB_MAP_SIZE
    return max(cfg.LMDB_MIN_MAP_SIZE, int(max(1, candidate_samples) * cfg.LMDB_BYTES_PER_SAMPLE_ESTIMATE * cfg.LMDB_MAP_SIZE_MARGIN))


def sleep_retry(attempt):
    time.sleep(min(60, 4 * attempt))

# Notebook analysis helpers consolidated from 05/07/08.
ARRAY_NAMES = ("lr_t1", "hr_t1", "lr_t2", "hr_t2")
S2_RGB_BANDS = (2, 1, 0)
S3_RGB_BANDS = (2, 1, 0)
CHINA_BBOX_WGS84 = (73.5, 18.0, 135.1, 53.6)


def _safe_float(value):
    if value in ("", None):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _safe_int(value):
    value = _safe_float(value)
    return None if value is None else int(value)


def read_csv_rows(path=None):
    path = Path(path or cfg.SAMPLE_INDEX_CSV)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fd:
        return list(csv.DictReader(fd))


def dataframe_from_csv(path=None):
    import pandas as pd

    return pd.DataFrame(read_csv_rows(path))


def parse_json_field(value, default=None):
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def bbox_union(bounds_list):
    bounds = [tuple(float(v) for v in b) for b in bounds_list if b and len(b) == 4]
    if not bounds:
        return []
    return [
        min(b[0] for b in bounds),
        min(b[1] for b in bounds),
        max(b[2] for b in bounds),
        max(b[3] for b in bounds),
    ]


def bbox_center(bounds):
    if not bounds or len(bounds) != 4:
        return "", ""
    return (float(bounds[0]) + float(bounds[2])) / 2.0, (float(bounds[1]) + float(bounds[3])) / 2.0


def bbox_polygon(bounds):
    if not bounds or len(bounds) != 4:
        return None
    minx, miny, maxx, maxy = [float(v) for v in bounds]
    return {"type": "Polygon", "coordinates": [[[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]]}


def sample_rows(path=None):
    return read_csv_rows(path or cfg.SAMPLE_INDEX_CSV)


def sample_dataframe(path=None, add_center=True):
    import pandas as pd

    rows = sample_rows(path)
    df = pd.DataFrame(rows)
    if df.empty or not add_center:
        return df
    lon_values, lat_values = [], []
    for _, row in df.iterrows():
        bounds = parse_json_field(row.get("bounds_wgs84_json"), [])
        lon, lat = bbox_center(bounds)
        lon_values.append(lon)
        lat_values.append(lat)
    df["center_lon"] = lon_values
    df["center_lat"] = lat_values
    for column in ["interval_days", "land_fraction", "water_fraction", "nodata_fraction", "s2_change_mean", "s3_change_mean"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def category_counts(path=None, column="landcover_major", normalize=True, drop_empty=True):
    df = sample_dataframe(path, add_center=False)
    if df.empty or column not in df.columns:
        return []
    values = df[column].fillna("").astype(str)
    if drop_empty:
        values = values[values != ""]
    counts = values.value_counts(dropna=False)
    total = int(counts.sum())
    rows = []
    for key, count in counts.items():
        row = {"category": key, "count": int(count)}
        if normalize:
            row["percent"] = (float(count) * 100.0 / total) if total else 0.0
        rows.append(row)
    return rows


def interval_bin(days):
    days = _safe_int(days)
    if days is None:
        return "missing"
    if days <= 15:
        return "000-015"
    if days <= 30:
        return "016-030"
    if days <= 60:
        return "031-060"
    if days <= 90:
        return "061-090"
    if days <= 180:
        return "091-180"
    if days <= 365:
        return "181-365"
    return "366+"


def numeric_summary(path=None, columns=None, by=None):
    import pandas as pd

    df = sample_dataframe(path)
    if df.empty:
        return pd.DataFrame()
    if columns is None:
        columns = [c for c in ["interval_days", "land_fraction", "water_fraction", "nodata_fraction", "s2_change_mean", "s3_change_mean"] if c in df.columns]
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if by and by in df.columns:
        return df.groupby(by)[columns].describe().reset_index()
    return df[columns].describe().T.reset_index().rename(columns={"index": "metric"})


def landcover_pixel_counts(path=None, max_rows=0):
    counter = Counter()
    rows = sample_rows(path)
    for idx, row in enumerate(rows):
        if max_rows and idx >= int(max_rows):
            break
        counts = parse_json_field(row.get("landcover_counts_json"), {}) or {}
        for key, value in counts.items():
            try:
                counter[str(key)] += int(value)
            except (TypeError, ValueError):
                continue
    total = sum(counter.values())
    return [{"category": key, "pixels": int(value), "percent": (float(value) * 100.0 / total) if total else 0.0} for key, value in counter.most_common()]


def dataset_distribution_summary(folder=None):
    folder = Path(folder or cfg.PIPELINE_OUT)
    samples = sample_rows(folder / "sample_index.csv")
    pair_rows = read_csv_rows(folder / "pair_summary.csv")
    build_report = read_json(folder / "build_report.json", {})
    mgrs = Counter(row.get("mgrs_tile", "") for row in samples if row.get("mgrs_tile"))
    landcover = Counter(row.get("landcover_major", "") for row in samples if row.get("landcover_major"))
    intervals = [_safe_int(row.get("interval_days")) for row in samples]
    interval_counts = Counter(interval_bin(v) for v in intervals)
    return {
        "folder": str(folder),
        "samples": len(samples),
        "pairs": len(pair_rows),
        "build_report_total_samples": build_report.get("total_samples", ""),
        "mgrs_unique": len(mgrs),
        "mgrs_top": mgrs.most_common(20),
        "landcover": landcover.most_common(),
        "interval_bins": dict(sorted(interval_counts.items())),
    }


def plot_sample_distribution_2d(path=None, color_by="landcover_major", ax=None, figsize=(12, 6), s=5, alpha=0.65, title=None, legend=True):
    import matplotlib.pyplot as plt
    import pandas as pd

    df = sample_dataframe(path)
    if df.empty:
        raise ValueError("sample_index.csv is empty or missing")
    df = df[pd.to_numeric(df["center_lon"], errors="coerce").notna() & pd.to_numeric(df["center_lat"], errors="coerce").notna()].copy()
    df["center_lon"] = pd.to_numeric(df["center_lon"], errors="coerce")
    df["center_lat"] = pd.to_numeric(df["center_lat"], errors="coerce")
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    if color_by and color_by in df.columns:
        for value, group in df.groupby(df[color_by].fillna("missing").astype(str)):
            ax.scatter(group["center_lon"], group["center_lat"], s=s, alpha=alpha, label=value)
        if legend:
            ax.legend(markerscale=2, fontsize=8, frameon=False, ncols=1)
    else:
        ax.scatter(df["center_lon"], df["center_lat"], s=s, alpha=alpha)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title or f"Sample distribution ({len(df)} samples)")
    ax.grid(True, linewidth=0.4, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    return ax


def pair_bounds(pair):
    bounds = []
    for time_key in ["t1", "t2"]:
        item = pair.get(time_key, {})
        s2 = item.get("s2", {})
        if s2.get("bounds_wgs84"):
            bounds.append(s2["bounds_wgs84"])
        for s3 in item.get("candidate_s3", []):
            if s3.get("bounds_wgs84"):
                bounds.append(s3["bounds_wgs84"])
    return bbox_union(bounds)


def pair_spatial_rows(pair_index_path=None, reference_bbox=CHINA_BBOX_WGS84):
    pair_index = read_json(pair_index_path or cfg.PAIR_INDEX_JSON, {})
    rows = []
    for pair in pair_index.get("pairs", []):
        bounds = pair_bounds(pair)
        center_lon, center_lat = bbox_center(bounds)
        t1_s2 = pair.get("t1", {}).get("s2", {})
        t2_s2 = pair.get("t2", {}).get("s2", {})
        t1_s3 = (pair.get("t1", {}).get("candidate_s3") or [{}])[0]
        t2_s3 = (pair.get("t2", {}).get("candidate_s3") or [{}])[0]
        rows.append({
            "pair_id": pair.get("pair_id", ""),
            "mgrs_tile": pair.get("mgrs_tile", ""),
            "t1_date": pair.get("t1_date", ""),
            "t2_date": pair.get("t2_date", ""),
            "interval_days": pair.get("interval_days", ""),
            "center_lon": center_lon,
            "center_lat": center_lat,
            "pair_bounds_wgs84": json.dumps(bounds),
            "inside_reference_bbox": bbox_contains(reference_bbox, bounds),
            "intersects_reference_bbox": bbox_intersects(bounds, reference_bbox),
            "t1_s2": t1_s2.get("name", ""),
            "t2_s2": t2_s2.get("name", ""),
            "t1_s3": t1_s3.get("name", ""),
            "t2_s3": t2_s3.get("name", ""),
            "t1_cloud": pair.get("quality", {}).get("t1_cloud", ""),
            "t2_cloud": pair.get("quality", {}).get("t2_cloud", ""),
        })
    return rows


def write_pair_spatial_geojson(path, rows=None):
    rows = rows if rows is not None else pair_spatial_rows()
    features = []
    for row in rows:
        bounds = parse_json_field(row.get("pair_bounds_wgs84"), [])
        geom = bbox_polygon(bounds)
        if geom is None:
            continue
        features.append({"type": "Feature", "geometry": geom, "properties": {k: v for k, v in row.items() if k != "pair_bounds_wgs84"}})
    Path(path).write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2), encoding="utf-8")


def _import_lmdb():
    import lmdb

    return lmdb


def open_lmdb(path=None, **kwargs):
    lmdb = _import_lmdb()
    return lmdb.open(str(path or cfg.LMDB_OUT), readonly=True, lock=False, readahead=False, max_readers=kwargs.pop("max_readers", 128), **kwargs)


def _load_pickle(value):
    import pickle

    return None if value is None else pickle.loads(value)


def read_lmdb_metadata(env):
    with env.begin(write=False) as txn:
        return _load_pickle(txn.get(b"__metadata__"))


def describe_lmdb(env, print_result=True):
    meta = read_lmdb_metadata(env) or {}
    with env.begin(write=False) as txn:
        stat = txn.stat()
        keys = []
        cur = txn.cursor()
        for key, _ in cur:
            key = key.decode("utf-8", errors="ignore")
            if key != "__metadata__":
                keys.append(key)
            if len(keys) >= 10:
                break
    out = {
        "schema": meta.get("schema", ""),
        "dataset_group": meta.get("dataset_group", ""),
        "dataset_name": meta.get("dataset_name", ""),
        "metadata_total_samples": meta.get("total_samples", ""),
        "lmdb_entries": stat.get("entries", ""),
        "sample_entries_estimate": max(0, int(stat.get("entries", 0)) - (1 if meta else 0)),
        "first_keys": keys,
        "hr_shape": meta.get("hr_shape", ""),
        "lr_shape": meta.get("lr_shape", ""),
        "dtype": meta.get("dtype", ""),
    }
    if print_result:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def list_lmdb_keys(env, limit=20, start_key=None):
    keys = []
    start = str(start_key).encode("utf-8") if start_key is not None else None
    with env.begin(write=False) as txn:
        cur = txn.cursor()
        ok = cur.set_range(start) if start is not None else cur.first()
        while ok and len(keys) < limit:
            key = cur.key().decode("utf-8", errors="ignore")
            if key != "__metadata__":
                keys.append(key)
            ok = cur.next()
    return keys


def random_lmdb_keys(env, n=8, seed=0):
    import random

    meta = read_lmdb_metadata(env) or {}
    total = meta.get("total_samples")
    if total in (None, "", 0):
        keys = list_lmdb_keys(env, limit=1000000)
        return random.sample(keys, min(n, len(keys)))
    rng = random.Random(seed)
    candidates = [f"{idx:08d}" for idx in rng.sample(range(int(total)), min(n, int(total)))]
    out = []
    with env.begin(write=False) as txn:
        for key in candidates:
            if txn.get(key.encode("utf-8")) is not None:
                out.append(key)
    return out


def unpack_arrays(packed):
    import blosc

    raw = blosc.decompress(packed)
    buf = io.BytesIO(raw)
    arrays = []
    while buf.tell() < len(raw):
        arrays.append(np.load(buf, allow_pickle=False))
    return arrays


def read_sample(env, key, with_arrays=True):
    if isinstance(key, int):
        key = f"{key:08d}"
    key = str(key)
    with env.begin(write=False) as txn:
        payload = _load_pickle(txn.get(key.encode("utf-8")))
    if payload is None:
        raise KeyError(f"LMDB key not found: {key}")
    sample = {"key": key, "metadata": payload.get("metadata", {})}
    if with_arrays:
        sample["arrays"] = dict(zip(ARRAY_NAMES, unpack_arrays(payload["arrays"])))
    return sample


def read_samples(env, keys):
    return [read_sample(env, key) for key in keys]


def valid_mask(arr, bad_value=None):
    arr = np.asarray(arr)
    mask = np.isfinite(arr)
    if bad_value is not None:
        mask &= arr != bad_value
    return mask


def normalize_image(img, bad_value=None, percentiles=(2, 98)):
    img = np.asarray(img, dtype=np.float32)
    mask = valid_mask(img, bad_value)
    if not np.any(mask):
        return np.zeros_like(img, dtype=np.float32)
    lo, hi = np.percentile(img[mask], percentiles)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(img[mask])), float(np.max(img[mask]))
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    out = (img - lo) / (hi - lo)
    out[~mask] = 0
    return np.clip(out, 0, 1)


def rgb_from_array(arr, sensor="s2", percentiles=(2, 98)):
    arr = np.asarray(arr)
    bands = S2_RGB_BANDS if sensor.lower() == "s2" else S3_RGB_BANDS
    bad_value = cfg.S2_BAD_VALUE if sensor.lower() == "s2" else cfg.S3_BAD_VALUE
    rgb = np.stack([arr[idx] for idx in bands], axis=-1)
    return normalize_image(rgb, bad_value=bad_value, percentiles=percentiles)


def change_image(arr_t1, arr_t2, sensor="s2", percentiles=(2, 98)):
    bad_value = cfg.S2_BAD_VALUE if sensor.lower() == "s2" else cfg.S3_BAD_VALUE
    a = np.asarray(arr_t1, dtype=np.float32)
    b = np.asarray(arr_t2, dtype=np.float32)
    valid = valid_mask(a, bad_value) & valid_mask(b, bad_value)
    diff = np.mean(np.abs(b - a), axis=0)
    diff[~np.all(valid, axis=0)] = 0
    return normalize_image(diff, bad_value=None, percentiles=percentiles)


def metadata_table(samples):
    import pandas as pd

    if isinstance(samples, dict):
        samples = [samples]
    rows = []
    for sample in samples:
        row = dict(sample.get("metadata", {}))
        row["key"] = sample.get("key", row.get("lmdb_key", ""))
        rows.append(row)
    return pd.DataFrame(rows)


def print_sample_summary(sample, fields=None):
    if fields is None:
        fields = ["sample_id", "lmdb_key", "pair_id", "mgrs_tile", "t1_date", "t2_date", "interval_days", "patch_row", "patch_col", "land_fraction", "water_fraction", "nodata_fraction", "landcover_major", "t1_cloud", "t2_cloud", "s2_change_mean", "s3_change_mean"]
    meta = sample.get("metadata", {})
    print(json.dumps({key: meta.get(key, "") for key in fields}, indent=2, ensure_ascii=False))
    for name, arr in sample.get("arrays", {}).items():
        print(f"{name}: shape={arr.shape} dtype={arr.dtype} min={np.nanmin(arr)} max={np.nanmax(arr)}")


def plot_sample(sample, figsize=(13, 8), percentiles=(2, 98), show_summary=True):
    import matplotlib.pyplot as plt

    arrays = sample["arrays"]
    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
    panels = [
        (axes[0, 0], rgb_from_array(arrays["hr_t1"], "s2", percentiles), "S2 t1 RGB"),
        (axes[0, 1], rgb_from_array(arrays["hr_t2"], "s2", percentiles), "S2 t2 RGB"),
        (axes[0, 2], change_image(arrays["hr_t1"], arrays["hr_t2"], "s2", percentiles), "S2 abs change"),
        (axes[1, 0], rgb_from_array(arrays["lr_t1"], "s3", percentiles), "S3 t1 pseudo RGB"),
        (axes[1, 1], rgb_from_array(arrays["lr_t2"], "s3", percentiles), "S3 t2 pseudo RGB"),
        (axes[1, 2], change_image(arrays["lr_t1"], arrays["lr_t2"], "s3", percentiles), "S3 abs change"),
    ]
    for ax, img, name in panels:
        ax.imshow(img, cmap="magma" if img.ndim == 2 else None)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
    meta = sample.get("metadata", {})
    fig.suptitle(f"key={sample.get('key', '')} mgrs={meta.get('mgrs_tile', '')} {meta.get('t1_date', '')}->{meta.get('t2_date', '')}", fontsize=11)
    if show_summary:
        print_sample_summary(sample)
    return fig, axes


def plot_random_samples(env, n=4, seed=0, figsize_per_sample=(5, 5), percentiles=(2, 98)):
    import matplotlib.pyplot as plt

    keys = random_lmdb_keys(env, n=n, seed=seed)
    samples = read_samples(env, keys)
    fig, axes = plt.subplots(len(samples), 2, figsize=(figsize_per_sample[0] * 2, figsize_per_sample[1] * len(samples)), constrained_layout=True)
    if len(samples) == 1:
        axes = np.asarray([axes])
    for row, sample in zip(axes, samples):
        meta = sample.get("metadata", {})
        row[0].imshow(rgb_from_array(sample["arrays"]["hr_t1"], "s2", percentiles))
        row[1].imshow(rgb_from_array(sample["arrays"]["hr_t2"], "s2", percentiles))
        row[0].set_title(f"{sample['key']} S2 t1 {meta.get('t1_date', '')}")
        row[1].set_title(f"S2 t2 {meta.get('t2_date', '')} {meta.get('mgrs_tile', '')}")
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
    return fig, axes, samples


def inspect_key(env, key):
    sample = read_sample(env, key)
    plot_sample(sample)
    return sample

