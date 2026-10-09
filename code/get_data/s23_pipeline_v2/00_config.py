"""Configuration for the manifest-free S2/S3 v2 data pipeline."""

import os
from pathlib import Path


# ROOT = Path(__file__).resolve().parents[1]
# # Choose one dataset group per run. Override in PowerShell, for example:
# #   $env:S23_DATASET_GROUP = "china"
# #   $env:S23_DATASET_GROUP = "overseas"
# DATASET_GROUP = os.environ.get("S23_DATASET_GROUP", "china").strip().lower()
# if DATASET_GROUP not in {"china", "overseas"}:
#     raise ValueError("S23_DATASET_GROUP must be 'china' or 'overseas'")

MAX_PAIRS_TO_PROCESS = None

ROOT = Path("/data1/code/s23_pipeline_v2")

# DATASET_GROUP = "overseas"
# ALLOWED_YEARS = {"2022","2023","2024","2025","2026"}

DATASET_GROUP = "china"
ALLOWED_YEARS = {"2024"}

CHINA_BBOX_WGS84 = [73.5, 18.0, 135.1, 53.6]
PAIR_SPATIAL_SPLIT_BBOX_WGS84 = CHINA_BBOX_WGS84
PAIR_SPATIAL_SPLIT_MODE = {
    "china": "inside",
    "overseas": "outside",
}.get(DATASET_GROUP, "none")


PIPELINE_ROOT = ROOT / "s23_pipeline_v2_out"
PIPELINE_OUT = PIPELINE_ROOT / DATASET_GROUP

S3_GEO_PICKLE_PATHS = [ROOT / "sentinel3_zip_files_with_geo_all.pickle"]
FILE_INVENTORY_JSON = PIPELINE_OUT / "file_inventory.json"
DOWNLOAD_PLAN_JSON = PIPELINE_OUT / "download_plan.json"
DOWNLOAD_STATUS_JSON = PIPELINE_OUT / "download_status.json"
PAIR_INDEX_JSON = PIPELINE_OUT / "pair_index.json"
PAIR_SUMMARY_CSV = PIPELINE_OUT / "pair_summary.csv"
PAIR_ANALYSIS_JSON = PIPELINE_OUT / "pair_analysis.json"


DATASET_PROFILES = {
    "china": {
        "dataset_name": "s23_china_512",
        "search_roots": [
           Path("/Sentinel/Sentinel2/"),Path("/data2/Sentinel-2"),Path("/Sentinel/Sentinel3/SB3_2014"),Path("/data2/Sentinel-3"),
        ],
        "download_link_files": [
            ROOT / "download_links_china.txt",
        ],
    },
    "overseas": {
        "dataset_name": "s23_overseas_512",
        "search_roots": [
            Path("/data2/Sentinel23_globe"),
        ],
        "download_link_files": [
            ROOT / "download_links_overseas.txt",
        ],
    },
}

PROFILE = DATASET_PROFILES[DATASET_GROUP]

SEARCH_ROOTS = PROFILE["search_roots"]
DOWNLOAD_LINK_FILES = PROFILE["download_link_files"]
DOWNLOAD_ROOT = PIPELINE_OUT / "downloads"

LMDB_OUT = PIPELINE_OUT / "lmdb_512"
SAMPLE_INDEX_CSV = PIPELINE_OUT / "sample_index.csv"
METADATA_JSON = PIPELINE_OUT / "metadata.json"
BUILD_REPORT_JSON = PIPELINE_OUT / "build_report.json"
BUILD_STATE_JSON = PIPELINE_OUT / "build_state.json"

FROMGLC_DIR = ROOT / "fromglc"
OCEAN_SHP_PATH = ROOT / "ne_10m_ocean" / "ne_10m_ocean.shp"

S2_BANDS = ["B02", "B03", "B04", "B8A"]
S3_BANDS = ["Oa04", "Oa06", "Oa08", "Oa17"]
S2_BAD_VALUE = 0
S3_BAD_VALUE = 65535
LR_FACTOR = 8
TILE_SIZE = 512
PATCH_HR_SIZE = TILE_SIZE
PATCH_LR_SIZE = PATCH_HR_SIZE // LR_FACTOR

MIN_LAND_FRACTION = 0.70
MAX_WATER_FRACTION = 0.20
MAX_NODATA_FRACTION = 0.10
FROMGLC_RESAMPLE_SHAPE = (128, 128)

PAIR_STRATEGY = "lowest_cloud_two_dates_per_mgrs"
# Hard filename-level prefilter. This is applied before opening zip files.

MAX_S2_CLOUD_PERCENT = 0.1
MIN_INTERVAL_DAYS = 1
MAX_INTERVAL_DAYS = 730
# Server-style reduction: keep only the best same-day S3 candidate per S2 date.
MAX_S3_CANDIDATES_PER_DATE = 1
# Fast local approximation of server S3 GeoFootprint.Contains(S2 footprint).
REQUIRE_S3_BBOX_CONTAINS_S2 = True
# Keep a few low-cloud S2 observations per MGRS before expensive bounds checks.
S2_TOPK_PER_MGRS = 4


LMDB_MAP_SIZE = None
LMDB_BYTES_PER_SAMPLE_ESTIMATE = 5 * (1 << 20)
LMDB_MAP_SIZE_MARGIN = 2.0
LMDB_MIN_MAP_SIZE = 8 * (1 << 30)
LMDB_BATCH_SIZE = 500

DOWNLOAD_MAX_RETRIES = 4
DOWNLOAD_CHUNK_SIZE = 1 << 20
DOWNLOAD_TIMEOUT_SECONDS = 120

DATASET_NAME = PROFILE["dataset_name"]
SOURCE_DATASET = "local_raw_s23_zip"
STOP_ON_ERROR = False

# Parallelism. Keep LMDB writing single-threaded; BUILD_SAMPLE_WORKERS only
# parallelizes sample computation before serial LMDB commits.
FILE_SCAN_WORKERS = 8
PAIR_METADATA_WORKERS = 8
DOWNLOAD_WORKERS = 2
BUILD_SAMPLE_WORKERS = 2





