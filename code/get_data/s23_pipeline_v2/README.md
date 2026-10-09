# S23 Pipeline V2

This folder builds 512-tile S2/S3 LMDB data from local downloaded zip files.
It does not require or read a manifest for pairing.

The pipeline supports two independent dataset groups: `china` and `overseas`.
All generated files are written under separate folders:

- `get_data/s23_pipeline_v2_out/china/`
- `get_data/s23_pipeline_v2_out/overseas/`

Choose the group before running scripts:

```powershell
$env:S23_DATASET_GROUP = "china"
# or
$env:S23_DATASET_GROUP = "overseas"
```

Edit `00_config.py` first. For each profile, put every folder that may contain
downloaded S2/S3 zip files in `DATASET_PROFILES[...]["search_roots"]`. Optional
URL-list text files can be placed in `download_link_files`, one URL per line.

If the S2/S3 zip files have already been downloaded, you can start directly from
`03_pair_local_s23.py`; `01` and `02` are only for inventory/download planning.
You can either edit `DATASET_PROFILES[...]["search_roots"]` in `00_config.py`,
or override roots at runtime:

```powershell
$env:S23_DATASET_GROUP = "china"
$env:S23_SEARCH_ROOTS = "D:\s23_china_downloads;E:\backup_s23_china"
C:\Users\xuchen\AppData\Local\Programs\Python\Python39\python.exe .\get_data\s23_pipeline_v2\03_pair_local_s23.py
```

For a small test run, set the pair limit before running `03` and `04`:

```powershell
$env:S23_DATASET_GROUP = "china"
$env:S23_MAX_PAIRS = "5"
C:\Users\xuchen\AppData\Local\Programs\Python\Python39\python.exe .\get_data\s23_pipeline_v2\03_pair_local_s23.py
C:\Users\xuchen\AppData\Local\Programs\Python\Python39\python.exe .\get_data\s23_pipeline_v2\04_build_lmdb_from_pairs.py
```

Clear the limit for full production:

```powershell
Remove-Item Env:S23_MAX_PAIRS -ErrorAction SilentlyContinue
```
Cloud threshold is controlled in `00_config.py` or at runtime:

```powershell
$env:S23_MAX_S2_CLOUD_PERCENT = "0.1"
```

The default is `0.1`, matching the strict low-cloud server setting. Increase it
only when the strict setting leaves too few valid MGRS pairs.

1. `01_find_download_files.py`

   Scans the selected group's `SEARCH_ROOTS` for valid S2/S3 zip files and
   writes `file_inventory.json`. It also reads URL lists and writes
   `download_plan.json` for files that are not already present.

2. `02_download_pending_files.py`

   Reads `download_plan.json` and downloads pending URLs into the selected
   group's `DOWNLOAD_ROOT`. It uses `.part` files, HTTP Range resume, retries,
   parallel downloads, and zip validation.

3. `03_pair_local_s23.py`

   Scans local zip files only. S2 observations are grouped by `MGRS + date`,
   S3 observations by date, and S2 observations with missing cloud metadata or
   cloud percentage above `MAX_S2_CLOUD_PERCENT` are rejected. Same-day S3
   candidates are ranked by S2 bbox overlap ratio. For each MGRS tile, only the
   two lowest-cloud valid S2 dates are selected to form one fusion pair, matching
   the server pipeline selection style. It writes `pair_index.json`,
   `pair_analysis.json`, and `pair_summary.csv`.

4. `04_build_lmdb_from_pairs.py`

   Reads `pair_index.json`, opens S2 zip files directly, reads each 512x512 S2
   tile into memory, resamples candidate S3 swaths directly to the matching
   64x64 grid, then writes LMDB samples as `[LR_t1, HR_t1, LR_t2, HR_t2]`.
   It writes `sample_index.csv`, `metadata.json`, `build_report.json`, and
   `build_state.json` for resume.

5. `06_pack_lmdb_for_transfer.py`

   Verifies the generated LMDB, estimates whether compact rewriting is worth it,
   and creates a transfer archive using the server style `tar/star + zstd -5 -T0`.
   By default it runs with no arguments, skips compact rewriting when the expected
   saving is below 5% or 1 GiB, and streams archives into 10 GiB parts. If the
   compressed output fits in one part, it is renamed to a single `.tar.zst` file.

   Example:

   ```powershell
   $env:S23_DATASET_GROUP = "china"
   C:\Users\xuchen\AppData\Local\Programs\Python\Python39\python.exe .\get_data\s23_pipeline_v2\06_pack_lmdb_for_transfer.py
   ```

   Force compact rewriting only when needed:

   ```powershell
   C:\Users\xuchen\AppData\Local\Programs\Python\Python39\python.exe .\get_data\s23_pipeline_v2\06_pack_lmdb_for_transfer.py --compact-mode always
   ```

   Main outputs:

   - `lmdb_512_final/`: compact read-only LMDB plus metadata files.
   - `transfer/*.tar.zst` or `transfer/*.tar.zst.part-000`: compressed package.
   - `transfer/*.sha256`: checksum file for transfer verification.
   - `transfer/transfer_manifest.json`: package summary.

   On the target machine, restore a split package with:

   ```bash
   sha256sum -c s23_china_512.sha256
   cat s23_china_512.tar.zst.part-* | zstd -d | tar -xf -
   ```

   For a single archive:

   ```bash
   sha256sum -c s23_china_512.sha256
   zstd -d -c s23_china_512.tar.zst | tar -xf -
   ```
The LMDB stores `uint16` arrays using `np.save` inside Blosc LZ4 compression.
No numeric quantization is applied during LMDB writing. Ocean tiles, low-land
FROMGLC tiles, high-water/nodata tiles, and bad-value samples are skipped.

## Parallelism

Parallelism is controlled in `00_config.py`:

- `FILE_SCAN_WORKERS`: validates local zip files in parallel during file discovery and pairing.
- `PAIR_METADATA_WORKERS`: reads S2 cloud, S2 bounds, and S3 swath bounds in parallel while building `pair_index.json`.
- `DOWNLOAD_WORKERS`: downloads multiple pending URLs concurrently. Keep this modest to avoid server throttling.
- `BUILD_SAMPLE_WORKERS`: computes samples in parallel before serial LMDB writes. LMDB key assignment and commits stay single-threaded for stability.

For large S3 swaths, `BUILD_SAMPLE_WORKERS=2` is a conservative default. Increase only if memory and disk IO are stable.



