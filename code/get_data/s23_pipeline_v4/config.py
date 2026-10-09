"""Configuration for the v4 S23 experiment pipeline.

v4 keeps the validated v3 raw-product readers and LMDB builder, but changes the
experiment organization:

- internal generation is frozen as a local-data workflow;
- benchmark is an evaluation-only collection with explicit subsets;
- local downloaded files are always used before any CDSE query planning;
- final scientific balance is done from sample_index.csv, not from pair_index.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("S23_V4_ROOT", Path(__file__).resolve().parent))
ENGINE_ROOT = Path(os.environ.get("S23_V4_ENGINE_ROOT", ROOT / "engine"))
OUT_ROOT = Path(os.environ.get("S23_V4_OUT", ROOT / "s23_pipeline_v4_out"))
SOURCE_YEAR = os.environ.get("S23_SOURCE_YEAR", "2024")

# The four scientific data subsets. Only internal_2024 is trainable by default;
# the other three are benchmark/evaluation-only subsets.
SUBSETS = {
    "internal_2024": {
        "domain": "internal",
        "year_group": "source_2024",
        "years": [SOURCE_YEAR],
        "usage": "train_val_test",
        "target_samples": 120000,
        "target_pairs": 2000,
    },
    "internal_non2024": {
        "domain": "internal",
        "year_group": "non2024",
        "years": ["2023", "2025", "2026"],
        "usage": "temporal_ood",
        "target_samples": 50000,
        "target_pairs": 1000,
    },
    "external_2024": {
        "domain": "external",
        "year_group": "source_2024",
        "years": [SOURCE_YEAR],
        "usage": "spatial_ood",
        "target_samples": 20000,
        "target_pairs": 400,
    },
    "external_non2024": {
        "domain": "external",
        "year_group": "non2024",
        "years": ["2023", "2025", "2026"],
        "usage": "spatiotemporal_ood",
        "target_samples": 20000,
        "target_pairs": 400,
    },
}

# Balance dimensions used for gap analysis and final split generation.
LANDCOVER_MAIN = ["cropland", "forest", "grassland", "bareland", "shrubland"]
INTERVAL_BINS = [(5, 30), (31, 60), (61, 120), (121, 180), (181, 365)]
MONTHS = [f"{m:02d}" for m in range(1, 13)]
LAT_BANDS = ["south_mid_high", "south_tropical", "north_tropical", "north_mid", "north_high"]

# Local-first benchmark policy. If a subset already satisfies these sample-level
# minima, CDSE query planning should skip it.
MIN_SAMPLES_PER_MAIN_LANDCOVER = int(os.environ.get("S23_MIN_SAMPLES_PER_LC", "1000"))
MIN_SAMPLES_PER_INTERVAL_BIN = int(os.environ.get("S23_MIN_SAMPLES_PER_INTERVAL", "1000"))
MIN_SAMPLES_PER_MONTH = int(os.environ.get("S23_MIN_SAMPLES_PER_MONTH", "500"))

# v4 engine compatibility. The engine is vendored from the validated v3 code,
# but lives inside v4 and is executed with subset-specific output directories.
def subset_root(name: str) -> Path:
    return OUT_ROOT / name


def v3_out_for_subset(name: str) -> Path:
    # v3 writes to S23_V3_OUT / DATASET_ROLE, so keep each subset isolated.
    return subset_root(name) / "v3"


def subset_lmdb_dir(name: str, role: str = "benchmark") -> Path:
    return v3_out_for_subset(name) / role / "lmdb_512"


def subset_sample_index(name: str, role: str = "benchmark") -> Path:
    return v3_out_for_subset(name) / role / "sample_index.csv"


def internal_domain_candidates() -> list[Path]:
    paths = []
    if os.environ.get("S23_INTERNAL_DOMAIN_JSON"):
        paths.append(Path(os.environ["S23_INTERNAL_DOMAIN_JSON"]))
    paths.extend([
        ROOT / "internal_domain.json",
        v3_out_for_subset("internal_2024") / "internal" / "internal_domain.json",
    ])
    return paths


def internal_sample_index_candidates() -> list[Path]:
    paths = []
    if os.environ.get("S23_INTERNAL_SAMPLE_INDEX"):
        paths.append(Path(os.environ["S23_INTERNAL_SAMPLE_INDEX"]))
    paths.extend([
        ROOT / "sample_index.csv",
        subset_sample_index("internal_2024", role="internal"),
    ])
    return paths
