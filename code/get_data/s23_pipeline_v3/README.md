# S23 Pipeline V3

This folder keeps `s23_pipeline_v2` intact and adds a stricter v3 workflow for
RSE-level generalization datasets.

Main changes:

- dataset roles are `internal` and `benchmark`, not `china/overseas` semantics;
- CDSE search creates resumable candidate queues per MGRS;
- each MGRS keeps `CANDIDATE_PAIRS_PER_MGRS = 4` candidate time pairs by default;
- CDSE candidate observations require same-day Sentinel-3 OLCI products whose
  `GeoFootprint` covers the Sentinel-2 product `GeoFootprint`;
- downloading tries candidate pairs in rank order and falls back to the next pair
  when locally downloaded S3 bounds do not cover the S2 GeoFootprint bbox;
- local pairing filters S2 by cloud and `MIN_S2_DATA_PERCENTAGE >= 0.9`;
- local pairing uses `first_next_valid_s2s3_observation_v3` instead of selecting
  the two lowest-cloud dates;
- LMDB still stores raw uint16 arrays without quantization;
- split files are generated as indexes, not separate LMDB copies.

## Dataset Role

Set one role per run:

```bash
export S23_DATASET_ROLE=internal
# or
export S23_DATASET_ROLE=benchmark
```

Outputs go to:

```text
/data1/code/s23_pipeline_v3/s23_pipeline_v3_out/internal
/data1/code/s23_pipeline_v3/s23_pipeline_v3_out/benchmark
```

Override paths if needed:

```bash
export S23_V3_ROOT=/data1/code/s23_pipeline_v3
export S23_V3_OUT=/data1/code/s23_pipeline_v3/s23_pipeline_v3_out
```

## Internal From Existing Local Files

If S2/S3 zip files are already available, you can start from local pairing:

```bash
export S23_DATASET_ROLE=internal
python 03_pair_local_s23.py
python 04_build_lmdb_from_pairs.py
python 09_build_internal_domain.py
python 10_build_split_index.py
```

The local pairing step applies:

```text
S2 cloud <= MAX_S2_CLOUD_PERCENT
S2 data percentage >= MIN_S2_DATA_PERCENTAGE
same-day S3 local bounds contains S2 bounds
first-next-valid t1/t2 interval selection
```

## Benchmark CDSE Candidate Queue

Prepare a selected MGRS file at:

```text
<PIPELINE_OUT>/benchmark_mgrs_selected.json
```

It can be either:

```json
["33UVP", "10SFD"]
```

or records:

```json
[{"mgrs_tile": "33UVP"}, {"mgrs_tile": "10SFD"}]
```

Build the resumable CDSE candidate queue:

```bash
export S23_DATASET_ROLE=benchmark
python 01_build_cdse_candidate_queue.py
```

This writes:

```text
cdse_candidate_queue.json
cdse_candidate_state.json
```

The state file allows rerunning after interruption. S2 search also applies an online server-style data coverage estimate: S2 GeoFootprint projected area divided by the full 20 m tile area must be greater than or equal to `MIN_S2_DATA_PERCENTAGE`. The SVG file is a quick global spatial distribution plot for candidate MGRS with valid candidate pairs.

## Download Candidate Queue

Set CDSE credentials in `00_config.py`:

```python
CDSE_ACCOUNTS = [
    {"username": "account1@example.com", "password": "password1"},
    {"username": "account2@example.com", "password": "password2"},
]
```

Then run:

```bash
export S23_DATASET_ROLE=benchmark
python 02_download_cdse_candidate_queue.py
```

The downloader:

1. tries each MGRS candidate pair by rank;
2. downloads S3 t1 first and validates local S3 bounds against S2 GeoFootprint;
3. downloads S3 t2 and validates it;
4. downloads S2 t1/t2 only after S3 coverage passes;
5. checks S2 data percentage;
6. records accepted/rejected history in `cdse_accepted_downloads.json`.

Rerunning continues from existing accepted/rejected records and existing `.part`
files.

## Rebuild After Pair-Strategy Changes

When changing temporal pairing or sample caps, rebuild from the pairing stage.
Do not reuse old `pair_index.json`, `build_state.json`, `sample_index.csv`, or
`lmdb_512/`, because they encode the previous pair/sample distribution.

For internal data, rerun:

```bash
python 03_pair_local_s23.py
python 04_build_lmdb_from_pairs.py
python 10_build_split_index.py
```

For benchmark data, keep completed downloads if desired, but rerun local pairing
and LMDB construction after cleaning old build outputs.

## Build Benchmark LMDB

After downloads:

```bash
export S23_DATASET_ROLE=benchmark
python 03_pair_local_s23.py
python 04_build_lmdb_from_pairs.py
python 10_build_split_index.py
```

## Transfer

The v2 transfer script is retained:

```bash
python 06_pack_lmdb_for_transfer.py
```

## Important Notes

CDSE `GeoFootprint` is only a download-cost filter. Final sample quality is still
validated locally:

```text
CDSE GeoFootprint coverage -> local S3 bounds coverage -> patch-level S2/S3 validity
```

A pair is scientifically useful only after `04_build_lmdb_from_pairs.py` reports
sufficient valid samples. Use `build_report.json`, `sample_index.csv`, and the
notebook helpers in `s23_common.py` to inspect the final distribution.

Notebook examples:

```python
import sys
sys.path.append("/data1/code/s23_pipeline_v3/get_data/s23_pipeline_v3")
import s23_common as C

C.dataset_distribution_summary()
C.category_counts(column="landcover_major")
C.numeric_summary(by="landcover_major")
C.plot_sample_distribution_2d(color_by="landcover_major")

env = C.open_lmdb(C.cfg.LMDB_OUT)
C.describe_lmdb(env)
sample = C.read_sample(env, 0)
C.plot_sample(sample)
```

## Global Balanced Benchmark MGRS

To avoid manual point selection, build global land MGRS candidates from FROMGLC
and then select a balanced benchmark set:

```bash
export S23_DATASET_ROLE=benchmark
python 00_build_global_mgrs_candidates.py
python 00_select_balanced_benchmark_mgrs.py
```

This writes:

```text
global_mgrs_candidates.json
benchmark_mgrs_selected.json
```

`00_build_global_mgrs_candidates.py` removes ocean/water-dominated candidates
using FROMGLC. `00_select_balanced_benchmark_mgrs.py` targets about 100 MGRS by
default and balances mainly by dominant land cover, with continent and latitude
band as soft constraints.

## Pair Input Mode

`03_pair_local_s23.py` supports three modes via `S23_PAIR_INPUT_MODE`:

```bash
export S23_PAIR_INPUT_MODE=auto       # default
export S23_PAIR_INPUT_MODE=json       # require cdse_accepted_downloads.json
export S23_PAIR_INPUT_MODE=directory  # ignore accepted JSON and scan folders
```

In `auto` mode, benchmark runs use `cdse_accepted_downloads.json` when it exists,
so the candidate queue and fallback history from 01/02 are preserved. Internal
runs, or benchmark runs without accepted JSON, fall back to directory scanning.

## S2 Data Percentage

v3 uses server-style S2 data percentage from GDAL product `FOOTPRINT` metadata:

```text
S2 data percentage = FOOTPRINT area / full 20 m band tile area
```

Missing `FOOTPRINT` is treated as unknown and rejected by default. Set
`ALLOW_MISSING_S2_FOOTPRINT = True` in `00_config.py` only for debugging.

## Valid Data Logic

The server-style valid data check is still the final patch-level check:

```text
S2 patch invalid if any value == 0
S3 patch invalid if resampled patch contains 65535
```

Therefore CDSE GeoFootprint coverage and local S3 bounds coverage are prefilters
for download efficiency, not final sample validity guarantees.

