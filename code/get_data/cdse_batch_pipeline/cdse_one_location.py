import argparse
import io
import json
import os
import pickle
import time
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import numpy as np
import requests
import rasterio
import xarray as xr
from pyproj import Transformer
from pyresample import geometry
from pyresample.kd_tree import resample_nearest
from rasterio.transform import from_origin
from rasterio.windows import Window


CDSE_ODATA = "https://catalogue.dataspace.copernicus.eu/odata/v1"
CDSE_DOWNLOAD = "https://download.dataspace.copernicus.eu/odata/v1"
CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)

S2_BANDS = ["B02", "B03", "B04", "B8A"]
S3_BANDS = ["Oa04", "Oa06", "Oa08", "Oa17"]

HR_SIZE = 1024
LR_FACTOR = 8
LR_SIZE = HR_SIZE // LR_FACTOR
SAMPLE_HR_SIZE = 512
SAMPLE_LR_SIZE = SAMPLE_HR_SIZE // LR_FACTOR

S2_BAD_VALUE = 0
S3_BAD_VALUE = 65535


def extract_s2_date(name):
    return name.split("MSIL2A_")[1][:8]


def extract_s3_date(name):
    return name.split("____")[1][:8]


def get_access_token(username, password):
    response = requests.post(
        CDSE_TOKEN_URL,
        data={
            "client_id": "cdse-public",
            "username": username,
            "password": password,
            "grant_type": "password",
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def build_point_wkt(lon, lat):
    return f"POINT ({lon} {lat})"


def odata_search(collection, name_contains, start, end, wkt, top=100):
    start_str = f"{start}T00:00:00.000Z"
    end_str = f"{end}T23:59:59.999Z"
    filters = [
        f"Collection/Name eq '{collection}'",
        f"contains(Name,'{name_contains}')",
        f"ContentDate/Start ge {start_str}",
        f"ContentDate/Start le {end_str}",
        f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')",
    ]
    url = (
        f"{CDSE_ODATA}/Products?$filter="
        + quote(" and ".join(filters), safe=":/(),';")
        + f"&$top={top}&$expand=Attributes&$orderby=ContentDate/Start asc"
    )
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.json().get("value", [])


def attr_value(product, *names):
    wanted = {name.lower() for name in names}
    for attr in product.get("Attributes", []):
        name = str(attr.get("Name", "")).lower()
        if name in wanted:
            for key in ("Value", "DoubleAttribute", "IntegerAttribute", "StringAttribute"):
                if key in attr and attr[key] is not None:
                    return attr[key]
    return None


def product_cloud(product):
    value = attr_value(
        product,
        "cloudCover",
        "cloudCoverPercentage",
        "cloudyPixelPercentage",
        "CLOUDY_PIXEL_PERCENTAGE",
    )
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def group_candidates(s2_products, s3_products, cloud_limit):
    s3_by_date = defaultdict(list)
    for product in s3_products:
        try:
            s3_by_date[extract_s3_date(product["Name"])].append(product)
        except Exception:
            continue

    pairs = []
    for s2 in s2_products:
        try:
            date_key = extract_s2_date(s2["Name"])
        except Exception:
            continue
        cloud = product_cloud(s2)
        if cloud is not None and cloud > cloud_limit:
            continue
        for s3 in s3_by_date.get(date_key, []):
            covered_s2_ids = s3.get("CoveredS2Ids")
            if covered_s2_ids is not None and s2.get("Id") not in covered_s2_ids:
                continue
            pairs.append({"date": date_key, "cloud": cloud, "s2": s2, "s3": s3})

    pairs.sort(key=lambda item: (9999.0 if item["cloud"] is None else item["cloud"], item["date"]))
    return pairs


def download_product(product, out_dir, token):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / product_zip_name(product)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    url = f"{CDSE_DOWNLOAD}/Products({product['Id']})/$value"
    headers = {"Authorization": f"Bearer {token}"}
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    with requests.get(url, headers=headers, stream=True, timeout=180) as response:
        response.raise_for_status()
        with open(tmp_path, "wb") as fd:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fd.write(chunk)
    tmp_path.replace(out_path)
    return out_path


def product_zip_name(product):
    name = product["Name"]
    if not name.endswith(".zip"):
        name = name + ".zip"
    return name


def product_name_from_zip(zip_path):
    name = zip_path.name
    if name.endswith(".zip"):
        name = name[:-4]
    return name


def find_existing_download_pairs(downloads_dir, pair_count):
    if not downloads_dir.exists():
        raise SystemExit(f"Downloads directory does not exist: {downloads_dir}")

    s2_by_date = defaultdict(list)
    s3_by_date = defaultdict(list)
    for zip_path in sorted(downloads_dir.glob("*.zip")):
        name = product_name_from_zip(zip_path)
        try:
            if "MSIL2A_" in name:
                s2_by_date[extract_s2_date(name)].append((name, zip_path))
            elif "OL_1_EFR" in name:
                s3_by_date[extract_s3_date(name)].append((name, zip_path))
        except Exception:
            continue

    pairs = []
    for date_key in sorted(set(s2_by_date) & set(s3_by_date)):
        for s2_name, s2_path in s2_by_date[date_key]:
            for s3_name, s3_path in s3_by_date[date_key]:
                pairs.append(
                    {
                        "date": date_key,
                        "cloud": None,
                        "s2": {"Id": None, "Name": s2_name, "LocalPath": str(s2_path)},
                        "s3": {"Id": None, "Name": s3_name, "LocalPath": str(s3_path)},
                    }
                )

    if len(pairs) < pair_count:
        raise SystemExit(
            f"Only {len(pairs)} local matched pair(s) found in {downloads_dir}; need {pair_count}."
        )
    return pairs[:pair_count]


def local_product_path(product, downloads_dir):
    if product.get("LocalPath"):
        return Path(product["LocalPath"])
    return downloads_dir / product_zip_name(product)


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


def lonlat_to_pixel(ds, lon, lat):
    transformer = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
    x_geo, y_geo = transformer.transform(lon, lat)
    return ds.index(x_geo, y_geo)


def write_s2_tile(zip_path, out_path, lon, lat):
    band_paths = find_s2_band_paths(zip_path)
    example = rasterio.open(band_paths["B02"])
    row, col = lonlat_to_pixel(example, lon, lat)
    tile_row = row // HR_SIZE
    tile_col = col // HR_SIZE
    xoff = tile_col * HR_SIZE
    yoff = tile_row * HR_SIZE
    if xoff < 0 or yoff < 0 or xoff + HR_SIZE > example.width or yoff + HR_SIZE > example.height:
        raise RuntimeError("Requested point is too close to the S2 product edge for a full 1024 tile")

    window = Window(xoff, yoff, HR_SIZE, HR_SIZE)
    transform = rasterio.windows.transform(window, example.transform)
    profile = example.profile.copy()
    profile.update(driver="GTiff", width=HR_SIZE, height=HR_SIZE, count=len(S2_BANDS), dtype="uint16", transform=transform)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        for band_index, band in enumerate(S2_BANDS, 1):
            with rasterio.open(band_paths[band]) as src:
                arr = src.read(1, window=window).astype(np.uint16)
            dst.write(arr, band_index)
    example.close()
    return tile_row, tile_col


def server_tile_starts(width, height, block_size):
    x_starts = list(range(0, width, block_size))[:-1]
    y_starts = list(range(0, height, block_size))[:-1]
    return y_starts, x_starts


def write_s2_tiles(zip_path, output_folder):
    band_paths = find_s2_band_paths(zip_path)
    output_folder.mkdir(parents=True, exist_ok=True)
    written = []

    with rasterio.open(band_paths["B02"]) as example:
        y_starts, x_starts = server_tile_starts(example.width, example.height, HR_SIZE)
        base_profile = example.profile.copy()

        for yoff in y_starts:
            for xoff in x_starts:
                window = Window(xoff, yoff, HR_SIZE, HR_SIZE)
                transform = rasterio.windows.transform(window, example.transform)
                profile = base_profile.copy()
                profile.update(
                    driver="GTiff",
                    width=HR_SIZE,
                    height=HR_SIZE,
                    count=len(S2_BANDS),
                    dtype="uint16",
                    transform=transform,
                )
                tile_row = yoff // HR_SIZE
                tile_col = xoff // HR_SIZE
                out_path = output_folder / f"tile_{tile_row}_{tile_col}.tif"
                with rasterio.open(out_path, "w", **profile) as dst:
                    for band_index, band in enumerate(S2_BANDS, 1):
                        with rasterio.open(band_paths[band]) as src:
                            arr = src.read(1, window=window).astype(np.uint16)
                        dst.write(arr, band_index)
                written.append(out_path)
    return written


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


def write_s3_tile(s3_zip_path, example_s2_tif, out_path):
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


def write_s3_tiles(s3_zip_path, s2_tile_paths, output_folder):
    output_folder.mkdir(parents=True, exist_ok=True)
    latitude, longitude, bands = load_s3_swath_bands(s3_zip_path)
    swath_def = geometry.SwathDefinition(lons=longitude, lats=latitude)
    written = []

    for example_s2_tif in s2_tile_paths:
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
        area_def = geometry.AreaDefinition(
            area_id, description, proj_id, projection, new_width, new_height, area_extent
        )
        out_path = output_folder / example_s2_tif.name
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
        written.append(out_path)
    return written


def has_bad_value(arr, bad_value):
    return np.any(arr == bad_value) or np.any(np.isnan(arr))


def pack_arrays(arr_list, cname="lz4", clevel=3):
    import blosc

    buf = io.BytesIO()
    for arr in arr_list:
        np.save(buf, arr, allow_pickle=False)
    return blosc.compress(buf.getvalue(), cname=cname, clevel=clevel)


def write_lmdb_from_tiles(base_dir, lmdb_path):
    import lmdb

    tile_sets = []
    t1c_names = {path.name for path in (base_dir / "t1c").glob("*.tif")}
    t1f_names = {path.name for path in (base_dir / "t1f").glob("*.tif")}
    t2c_names = {path.name for path in (base_dir / "t2c").glob("*.tif")}
    t2f_names = {path.name for path in (base_dir / "t2f").glob("*.tif")}
    for tile_name in sorted(t1c_names & t1f_names & t2c_names & t2f_names):
        tile_sets.append(
            [
                base_dir / "t1c" / tile_name,
                base_dir / "t1f" / tile_name,
                base_dir / "t2c" / tile_name,
                base_dir / "t2f" / tile_name,
            ]
        )

    env = lmdb.open(str(lmdb_path), map_size=500 * 1024**2)
    sample_count = 0
    original_total = len(tile_sets) * 4
    with env.begin(write=True) as txn:
        for paths in tile_sets:
            tile_name = paths[0].stem
            for i in range(2):
                for j in range(2):
                    arrays = []
                    windows = [
                        ((i * SAMPLE_LR_SIZE, (i + 1) * SAMPLE_LR_SIZE), (j * SAMPLE_LR_SIZE, (j + 1) * SAMPLE_LR_SIZE)),
                        ((i * SAMPLE_HR_SIZE, (i + 1) * SAMPLE_HR_SIZE), (j * SAMPLE_HR_SIZE, (j + 1) * SAMPLE_HR_SIZE)),
                        ((i * SAMPLE_LR_SIZE, (i + 1) * SAMPLE_LR_SIZE), (j * SAMPLE_LR_SIZE, (j + 1) * SAMPLE_LR_SIZE)),
                        ((i * SAMPLE_HR_SIZE, (i + 1) * SAMPLE_HR_SIZE), (j * SAMPLE_HR_SIZE, (j + 1) * SAMPLE_HR_SIZE)),
                    ]
                    bad_values = [S3_BAD_VALUE, S2_BAD_VALUE, S3_BAD_VALUE, S2_BAD_VALUE]
                    skip = False
                    for path, window, bad_value in zip(paths, windows, bad_values):
                        with rasterio.open(path) as src:
                            arr = src.read(window=window).astype(np.uint16)
                        if has_bad_value(arr, bad_value):
                            skip = True
                            break
                        arrays.append(arr)
                    if skip:
                        continue
                    sample_data = {
                        "arrays": pack_arrays(arrays),
                        "metadata": {"tile": tile_name, "sub_index": (i, j)},
                    }
                    txn.put(f"{sample_count:08d}".encode(), pickle.dumps(sample_data))
                    sample_count += 1

    with env.begin(write=True) as txn:
        txn.put(
            b"__metadata__",
            pickle.dumps(
                {
                    "total_samples": sample_count,
                    "original_total_samples": original_total,
                    "sample_structure": "Each sample has 4 numpy arrays: [LR_t1, HR_t1, LR_t2, HR_t2]",
                }
            ),
        )
    env.sync()
    env.close()
    return sample_count, original_total


def save_manifest(path, selected):
    records = []
    for item in selected:
        s2_record = {"id": item["s2"].get("Id"), "name": item["s2"]["Name"]}
        s3_record = {"id": item["s3"].get("Id"), "name": item["s3"]["Name"]}
        if item["s2"].get("LocalPath"):
            s2_record["local_path"] = item["s2"]["LocalPath"]
        if item["s3"].get("LocalPath"):
            s3_record["local_path"] = item["s3"]["LocalPath"]
        records.append(
            {
                "date": item["date"],
                "cloud": item["cloud"],
                "s2": s2_record,
                "s3": s3_record,
            }
        )
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def load_manifest(path):
    records = json.loads(path.read_text(encoding="utf-8"))
    selected = []
    for record in records:
        s2_product = {"Id": record["s2"].get("id"), "Name": record["s2"]["name"]}
        s3_product = {"Id": record["s3"].get("id"), "Name": record["s3"]["name"]}
        if record["s2"].get("local_path"):
            s2_product["LocalPath"] = record["s2"]["local_path"]
        if record["s3"].get("local_path"):
            s3_product["LocalPath"] = record["s3"]["local_path"]
        selected.append({"date": record["date"], "cloud": record.get("cloud"), "s2": s2_product, "s3": s3_product})
    return selected


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and process one Sentinel-2/Sentinel-3 location with the server-style pipeline."
    )
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--start", default="2025-04-01")
    parser.add_argument("--end", default="2026-12-31")
    parser.add_argument("--cloud-limit", type=float, default=0.1)
    parser.add_argument("--pairs", type=int, default=2, help="Server samples need two dates by default.")
    parser.add_argument("--out", type=Path, default=Path("cdse_one_location_out_2"))
    parser.add_argument("--username", default=os.environ.get("CDSE_USERNAME"))
    parser.add_argument("--password", default=os.environ.get("CDSE_PASSWORD"))
    parser.add_argument("--manifest-in", type=Path, help="Download/process products from an existing manifest.")
    parser.add_argument("--skip-download", action="store_true", help="Only write the manifest.")
    parser.add_argument(
        "--use-existing-downloads",
        action="store_true",
        help="Find already downloaded S2/S3 zip pairs in --out/downloads and skip search/download.",
    )
    parser.add_argument("--skip-lmdb", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    downloads_dir = args.out / "downloads"
    processed_dir = args.out / "processed" / "location_0000"
    manifest_path = args.out / "manifest.json"

    local_pairs = None
    if args.use_existing_downloads:
        selected = find_existing_download_pairs(downloads_dir, args.pairs)
        save_manifest(manifest_path, selected)
        print(f"Loaded local download pairs from: {downloads_dir}")
        print(f"Manifest written: {manifest_path}")
        local_pairs = [
            (local_product_path(item["s2"], downloads_dir), local_product_path(item["s3"], downloads_dir))
            for item in selected
        ]
    elif args.manifest_in:
        selected = load_manifest(args.manifest_in)
        print(f"Loaded manifest: {args.manifest_in}")
        if all(item["s2"].get("LocalPath") and item["s3"].get("LocalPath") for item in selected):
            local_pairs = [
                (local_product_path(item["s2"], downloads_dir), local_product_path(item["s3"], downloads_dir))
                for item in selected
            ]
    else:
        wkt = build_point_wkt(args.lon, args.lat)
        print("Searching Sentinel-2 products...")
        s2_products = odata_search("SENTINEL-2", "MSIL2A", args.start, args.end, wkt, top=200)
        print(f"  found {len(s2_products)} S2 products")

        print("Searching Sentinel-3 OLCI EFR products...")
        s3_products = odata_search("SENTINEL-3", "OL_1_EFR", args.start, args.end, wkt, top=500)
        print(f"  found {len(s3_products)} S3 products")

        pairs = group_candidates(s2_products, s3_products, args.cloud_limit)
        if len(pairs) < args.pairs:
            raise SystemExit(f"Only {len(pairs)} matched pair(s) found; need {args.pairs}.")

        selected = pairs[: args.pairs]
        save_manifest(manifest_path, selected)
        print(f"Manifest written: {manifest_path}")
    for index, item in enumerate(selected, 1):
        print(f"  t{index}: {item['date']} cloud={item['cloud']} S2={item['s2']['Name']} S3={item['s3']['Name']}")

    if args.skip_download:
        return
    if local_pairs is None:
        if not args.username or not args.password:
            raise SystemExit("Set CDSE_USERNAME/CDSE_PASSWORD or pass --username/--password.")

        token = get_access_token(args.username, args.password)
        local_pairs = []
        for index, item in enumerate(selected, 1):
            print(f"Downloading t{index} S2...")
            s2_path = download_product(item["s2"], downloads_dir, token)
            time.sleep(1)
            print(f"Downloading t{index} S3...")
            s3_path = download_product(item["s3"], downloads_dir, token)
            local_pairs.append((s2_path, s3_path))


    
    for index, (s2_path, s3_path) in enumerate(local_pairs, 1):
        fine_dir = processed_dir / f"t{index}f"
        coarse_dir = processed_dir / f"t{index}c"
        print(f"Processing t{index} S2 tiles...")
        s2_tiles = write_s2_tiles(s2_path, fine_dir)
        print(f"  S2 server-grid tiles: {len(s2_tiles)} written to {fine_dir}")
        print(f"Processing t{index} S3 tiles...")
        s3_tiles = write_s3_tiles(s3_path, s2_tiles, coarse_dir)
        print(f"  S3 reprojected tiles: {len(s3_tiles)} written to {coarse_dir}")

    if not args.skip_lmdb:
        lmdb_path = args.out / "lmdb"
        count, original = write_lmdb_from_tiles(processed_dir, lmdb_path)
        print(f"LMDB written: {lmdb_path} ({count}/{original} samples)")


if __name__ == "__main__":
    main()
