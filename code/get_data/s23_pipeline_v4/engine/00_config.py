"""Configuration for the S2/S3 v3 data pipeline.

v3 keeps v2 intact and adds:
- internal / benchmark dataset roles instead of china / overseas semantics;
- S2 data-percentage filtering before pairing;
- first-next-valid observation pairing;
- CDSE candidate queues with same-day S3 GeoFootprint covering S2 GeoFootprint;
- resumable candidate search/download/local build steps.
"""

import os
from pathlib import Path


# Edit ROOT on the server if needed. Environment variables override defaults.
ROOT = Path(os.environ.get("S23_V3_ROOT", "/data1/code/s23_pipeline_v3"))
PIPELINE_ROOT = Path(os.environ.get("S23_V3_OUT", str(ROOT / "s23_pipeline_v3_out")))

# Dataset role per run. Use:
#   S23_DATASET_ROLE=internal
#   S23_DATASET_ROLE=benchmark
DATASET_ROLE = os.environ.get("S23_DATASET_ROLE", "internal").strip().lower()
if DATASET_ROLE not in {"internal", "benchmark"}:
    raise ValueError("S23_DATASET_ROLE must be 'internal' or 'benchmark'")

# Backward-compatible alias for old v2 scripts that still refer to DATASET_GROUP.
DATASET_GROUP = DATASET_ROLE
PIPELINE_OUT = PIPELINE_ROOT / DATASET_ROLE

# Internal is the dense source campaign. Do not treat this as a strict bbox.
SOURCE_YEAR = os.environ.get("S23_SOURCE_YEAR", "2024")
if os.environ.get("S23_ALLOWED_YEARS"):
    ALLOWED_YEARS = {item.strip() for item in os.environ["S23_ALLOWED_YEARS"].split(",") if item.strip()}
else:
    ALLOWED_YEARS = {SOURCE_YEAR} if DATASET_ROLE == "internal" else {"2023", "2024", "2025", "2026"}
SOURCE_APPROX_BBOX_WGS84 = [73.5, 18.0, 135.1, 53.6]
INTERNAL_DOMAIN_JSON = Path(os.environ.get("S23_INTERNAL_DOMAIN_JSON", PIPELINE_ROOT / "internal" / "internal_domain.json"))

DATASET_PROFILES = {
    "internal": {
        "dataset_name": "s23_internal_512_v3",
        "search_roots": [
            Path("/Sentinel/Sentinel2/"),
            Path("/data2/Sentinel-2"),
            Path("/Sentinel/Sentinel3/SB3_2014"),
            Path("/data2/Sentinel-3"),
        ],
        "download_link_files": [ROOT / "download_links_internal.txt"],
    },
    "benchmark": {
        "dataset_name": "s23_benchmark_512_v3",
        "search_roots": [
            PIPELINE_OUT / "downloads",
            Path("/data2/Sentinel23_globe"),
        ],
        "download_link_files": [ROOT / "download_links_benchmark.txt"],
    },
}
PROFILE = DATASET_PROFILES[DATASET_ROLE]
DATASET_NAME = os.environ.get("S23_DATASET_NAME", PROFILE["dataset_name"])
SEARCH_ROOTS = [Path(item) for item in os.environ["S23_SEARCH_ROOTS"].split(os.pathsep) if item.strip()] if os.environ.get("S23_SEARCH_ROOTS") else PROFILE["search_roots"]
DOWNLOAD_LINK_FILES = PROFILE["download_link_files"]
DOWNLOAD_ROOT = PIPELINE_OUT / "downloads"

# Sidecars.
FILE_INVENTORY_JSON = PIPELINE_OUT / "file_inventory.json"
DOWNLOAD_PLAN_JSON = PIPELINE_OUT / "download_plan.json"
DOWNLOAD_STATUS_JSON = PIPELINE_OUT / "download_status.json"
CDSE_CANDIDATE_QUEUE_JSON = PIPELINE_OUT / "cdse_candidate_queue.json"
CDSE_CANDIDATE_STATE_JSON = PIPELINE_OUT / "cdse_candidate_state.json"
CDSE_ACCEPTED_DOWNLOADS_JSON = PIPELINE_OUT / "cdse_accepted_downloads.json"
PAIR_INDEX_JSON = PIPELINE_OUT / "pair_index.json"
PAIR_SUMMARY_CSV = PIPELINE_OUT / "pair_summary.csv"
PAIR_ANALYSIS_JSON = PIPELINE_OUT / "pair_analysis.json"
LMDB_OUT = PIPELINE_OUT / "lmdb_512"
SAMPLE_INDEX_CSV = PIPELINE_OUT / "sample_index.csv"
METADATA_JSON = PIPELINE_OUT / "metadata.json"
BUILD_REPORT_JSON = PIPELINE_OUT / "build_report.json"
BUILD_STATE_JSON = PIPELINE_OUT / "build_state.json"

S3_GEO_PICKLE_PATHS = [ROOT / "sentinel3_zip_files_with_geo_all.pickle"]
FROMGLC_DIR = ROOT / "fromglc"
OCEAN_SHP_PATH = ROOT / "ne_10m_ocean" / "ne_10m_ocean.shp"

# Optional selected benchmark MGRS input. Accepts either a JSON list of strings
# or records with mgrs_tile / mgrs fields.
BENCHMARK_MGRS_JSON = Path(os.environ.get("S23_BENCHMARK_MGRS_JSON", PIPELINE_OUT / "benchmark_mgrs_selected.json"))
BENCHMARK_FINAL_MGRS = int(os.environ.get("S23_BENCHMARK_FINAL_MGRS", "100"))
# Select a larger MGRS pool first. CDSE/S3 coverage filtering is strict, so the
# final benchmark should be drawn from successful tiles rather than assuming
# every land-cover-balanced tile will have valid S2/S3 temporal pairs.
BENCHMARK_TARGET_MGRS = int(os.environ.get("S23_BENCHMARK_TARGET_MGRS", "300"))
BENCHMARK_MAX_MGRS_ATTEMPTS = int(os.environ.get("S23_BENCHMARK_MAX_MGRS_ATTEMPTS", "500"))

# CDSE search settings. Credentials are read only by the download script.
CDSE_ODATA = "https://catalogue.dataspace.copernicus.eu/odata/v1"
CDSE_DOWNLOAD = "https://download.dataspace.copernicus.eu/odata/v1"
CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CDSE_SEARCH_START = os.environ.get("S23_CDSE_SEARCH_START", "2023-01-01" if DATASET_ROLE == "benchmark" else f"{SOURCE_YEAR}-01-01")
CDSE_SEARCH_END = os.environ.get("S23_CDSE_SEARCH_END", "2026-12-31" if DATASET_ROLE == "benchmark" else f"{SOURCE_YEAR}-12-31")
CDSE_S2_COLLECTION = "SENTINEL-2"
CDSE_S3_COLLECTION = "SENTINEL-3"
CDSE_S2_NAME_CONTAINS = ["MSIL2A"]
CDSE_S3_NAME_CONTAINS = ["OL_1_EFR"]
CDSE_S2_PAGE_SIZE = 100
CDSE_S3_PAGE_SIZE = 100
CDSE_MAX_S2_PRODUCTS_PER_MGRS = None
CDSE_MAX_S3_PRODUCTS_PER_DATE = 20
CDSE_SEARCH_WORKERS = 1
CDSE_REQUEST_SLEEP_SECONDS = 0.2
# Query S3 once per date and S2-footprint bbox instead of once per S2 product.
CDSE_S3_QUERY_MODE = "per_date_bbox"
CDSE_QUERY_CACHE_ENABLED = True
CDSE_QUERY_CACHE_DIR = PIPELINE_OUT / "cdse_query_cache"
CDSE_MAX_RETRIES = 4
CDSE_RETRY_SLEEP_SECONDS = 10
CDSE_REQUIRE_S3_FOOTPRINT_COVERS_S2 = True

# Pair selection.
PAIR_STRATEGY = "balanced_month_interval_s2s3_observation_v3"
# Search-stage candidate depth per MGRS. A later global balancing step keeps a
# smaller number for download/build.
CANDIDATE_PAIRS_PER_MGRS = 8
MIN_INTERVAL_DAYS = 5
MAX_INTERVAL_DAYS = 180
FALLBACK_MAX_INTERVAL_DAYS = 365
MAX_PAIRS_TO_PROCESS = None
MAX_S2_CLOUD_PERCENT = 0.1
MIN_S2_DATA_PERCENTAGE = 0.90
# Server-style full 20 m Sentinel-2 tile width/height: 5490 pixels * 20 m.
S2_FULL_TILE_SIZE_M = 109800.0
MAX_S3_CANDIDATES_PER_DATE = 1
REQUIRE_S3_BBOX_CONTAINS_S2 = True
S2_TOPK_PER_MGRS = 144
S2_TOPK_SELECTION = "month_round_robin"
PAIRS_PER_MGRS = int(os.environ.get("S23_PAIRS_PER_MGRS", "8" if DATASET_ROLE == "internal" else "2"))
MIN_CANDIDATE_PAIRS_PER_MGRS = 2
PAIR_TARGET_INTERVAL_DAYS = 60
PAIR_MONTH_BALANCE = True
PAIR_INTERVAL_BINS = [(5, 30), (31, 60), (61, 120), (121, 180), (181, 365)]
MAX_SAMPLES_PER_PAIR = 256
SAMPLE_PATCH_SELECTION = "spatial_round_robin"

# Sample generation.
S2_BANDS = ["B02", "B03", "B04", "B8A"]
S3_BANDS = ["Oa04", "Oa06", "Oa08", "Oa17"]
S2_BAD_VALUE = 0
S3_BAD_VALUE = 65535
LR_FACTOR = 8
TILE_SIZE = 512
PATCH_HR_SIZE = TILE_SIZE
PATCH_LR_SIZE = PATCH_HR_SIZE // LR_FACTOR
MIN_VALID_PATCHES_PER_PAIR = 30

MIN_LAND_FRACTION = 0.70
MAX_WATER_FRACTION = 0.20
MAX_NODATA_FRACTION = 0.10
FROMGLC_RESAMPLE_SHAPE = (128, 128)

LMDB_MAP_SIZE = None
LMDB_BYTES_PER_SAMPLE_ESTIMATE = 5 * (1 << 20)
LMDB_MAP_SIZE_MARGIN = 2.0
LMDB_MIN_MAP_SIZE = 8 * (1 << 30)
LMDB_BATCH_SIZE = 500

DOWNLOAD_MAX_RETRIES = 4
DOWNLOAD_CHUNK_SIZE = 1 << 20
DOWNLOAD_TIMEOUT_SECONDS = 120
DOWNLOAD_WORKERS = 2
DOWNLOAD_VALIDATE_ORDER = "s3_first"
# CDSE credentials are read by 02_download_cdse_candidate_queue.py from
# S23_CDSE_ACCOUNTS_JSON or accounts.local.json. Do not hard-code passwords.
CDSE_ACCOUNTS = []
# Pair input mode for 03: auto uses accepted CDSE JSON for benchmark when present, otherwise scans directories.
PAIR_INPUT_MODE = os.environ.get("S23_PAIR_INPUT_MODE", "auto").strip().lower()  # auto | json | directory
ALLOW_MISSING_S2_FOOTPRINT = False

SOURCE_DATASET = "local_raw_s23_zip_v3"
STOP_ON_ERROR = False
FILE_SCAN_WORKERS = 8
PAIR_METADATA_WORKERS = 8
BUILD_SAMPLE_WORKERS = 2











