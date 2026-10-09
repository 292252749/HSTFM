# Analysis and Evaluation

Use `analysis/run.py` for formal validation/test runs. It reuses the same data module and model builder as `train/run.py`, records the evaluation configuration, and writes `test_metrics.csv` under the evaluation output directory.

Formal evaluation rules:

- Checkpoint loading is strict by default. Use `--allow_partial_load` only for debugging, not for reported metrics.
- Use the same `--data_source`, `--dataset`, `--lmdb_path`, `--bands`, and model arguments as the training run being evaluated.
- For LMDB experiments, pass the training run's `split_indices.json` through `--split_file` so train/val/test membership is identical.
- `--mask` only enables the mask-aware data path. Clean formal evaluation uses `--eval_mask_prob 0`, which is the default.
- Missing-observation/cloud robustness tests must set a fixed `--eval_mask_prob` and should be reported separately from clean architecture comparisons.
- `--ergas_ratio` defaults to 8 for repository LMDB and 16 for native benchmark datasets. Override it only when the dataset scale requires it.

`analysis/utils.py` contains ad-hoc single-image export helpers. It is not the formal metric pipeline.
