# S23 Pipeline v4

v4 is self-contained. The validated v3 raw-product engine has been vendored into
`engine/`, so no external `s23_pipeline_v3` checkout is required.

Folder layout:

- `internal/`: local downloaded internal data workflow.
- `benchmark/`: local-first benchmark inventory and sample-level split reports.
- `engine/`: vendored low-level S2/S3 pairing, LMDB building, packing, and split helpers.

Put the source-domain description here when reusing an existing internal set:

```text
s23_pipeline_v4/internal_domain.json
```

Optionally also put/copy the source sample index here:

```text
s23_pipeline_v4/sample_index.csv
```

## Internal

```bash
cd /data1/code/s23_pipeline_v4/internal
python build_internal.py --subset internal_2024
```

This runs the vendored engine in an isolated v4 output directory:
`03 -> 05 -> 04 -> 09 -> 10`.

## Benchmark

```bash
cd /data1/code/s23_pipeline_v4/benchmark
python build_benchmark.py local-report
python build_benchmark.py split-report --write-splits
```

Benchmark subsets are:

- `internal_non2024`: temporal OOD, internal domain, non-2024 years.
- `external_2024`: spatial OOD, external domain, 2024.
- `external_non2024`: spatiotemporal OOD, external domain, non-2024 years.

`local-report` does not query CDSE. It inventories local zip files and uses
`internal_domain.json` to classify internal/external MGRS. Pass extra local roots
with `--paths` when reusing previously downloaded files.

Final scientific balance should be generated from `sample_index.csv`, not from
pair-level candidate counts.
