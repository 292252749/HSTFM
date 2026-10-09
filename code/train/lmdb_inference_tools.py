"""Notebook helpers for LMDB fusion inference and lightweight model comparison.

Typical notebook usage:
    import sys
    sys.path.insert(0, r"E:/OneDrive/code/HSTFM")  # or /data1/code/HSTFM on server
    from train import lmdb_inference_tools as T

    model = T.load_trained_model("HSTFM", "path/to/best.ckpt", device="cuda")
    sample = T.load_lmdb_sample("/data1/code/s23_pipeline_v2/s23_pipeline_v2_out/china/lmdb_512", 0)
    result = T.run_model_on_sample(model, sample)
    T.plot_prediction_result(result)

    results = T.compare_methods(
        sample,
        models={"HSTFM": model},
        include_baselines=True,
    )
    T.metrics_table(results)
    T.plot_comparison(results)
"""

from pathlib import Path
from types import SimpleNamespace
import inspect
import math
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LibSTFv1.dataset.SlideDataset import SentinelLMDBDataset
from train.run import build_model


DEFAULT_ARGS = {
    "epochs": 1,
    "bands": 4,
    "rgb_c": [2, 1, 0],
    "dataset": "Sentinel",
    "data_source": "lmdb",
    "ergas_ratio": 8.0,
    "mask": False,
    "model": "HSTFM",
}


def make_args(model_name="HSTFM", **kwargs):
    values = dict(DEFAULT_ARGS)
    values.update(kwargs)
    values["model"] = model_name
    return SimpleNamespace(**values)


def _filter_state_dict(state_dict, ignore_cache_keys=True):
    if not ignore_cache_keys:
        return state_dict, []
    filtered = {}
    ignored = []
    for key, value in state_dict.items():
        if "._mask_cache." in key or key.endswith("._mask_cache"):
            ignored.append(key)
        else:
            filtered[key] = value
    return filtered, ignored


def load_trained_model(model_name, ckpt_path, device="cuda", strict=True, return_info=False, ignore_cache_keys=True, **kwargs):
    """Load a model checkpoint produced by train/run.py."""
    device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    args = make_args(model_name, **kwargs)
    cfg = {"band": int(args.bands), "rgb_c": args.rgb_c, "name": args.dataset}
    model = build_model(args, cfg)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict, ignored_keys = _filter_state_dict(state_dict, ignore_cache_keys=ignore_cache_keys)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.eval()
    info = {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "ignored_keys": ignored_keys,
        "checkpoint": str(ckpt_path),
        "device": str(device),
    }
    return (model, info) if return_info else model


def load_lmdb_dataset(lmdb_path, patch_size=512, mask=False, deterministic_mask=True):
    return SentinelLMDBDataset(
        lmdb_path,
        patch_size=patch_size,
        mask=mask,
        deterministic_mask=deterministic_mask,
    )


def load_lmdb_sample(lmdb_path, index_or_key=0, patch_size=512, mask=False):
    """Load one normalized sample as tensors CHW, matching the training batch convention."""
    dataset = load_lmdb_dataset(lmdb_path, patch_size=patch_size, mask=mask, deterministic_mask=True)
    if isinstance(index_or_key, str):
        index = int(index_or_key)
    else:
        index = int(index_or_key)
    item = dataset[index]
    dataset.close()
    item["index"] = index
    return item


def batch_from_sample(sample, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    batch = {}
    for key in ["LR_t1", "LR_t2", "HR_t1", "HR_t2", "mask"]:
        value = sample[key]
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, dtype=torch.float32)
        batch[key] = value.unsqueeze(0).to(device)
    return batch


def _predict_model(model, batch):
    forward_params = inspect.signature(model.forward).parameters
    with torch.no_grad():
        if "mask" in forward_params:
            out = model.forward(batch["LR_t1"], batch["LR_t2"], batch["HR_t1"], batch["mask"])
        else:
            out = model.forward(batch["LR_t1"], batch["LR_t2"], batch["HR_t1"])
    if isinstance(out, dict):
        pred = out.get("pred")
    else:
        pred = out
    if pred is None:
        raise RuntimeError("Model output does not contain 'pred'.")
    return torch.clamp(pred, 0.0, 1.0)


def run_model_on_sample(model, sample, name="model", device=None):
    if device is None:
        device = next(model.parameters()).device
    batch = batch_from_sample(sample, device=device)
    pred = _predict_model(model, batch)
    return make_result(name, pred, batch, metadata=sample.get("metadata"), index=sample.get("index"))


def baseline_predictions(batch):
    """Return simple baselines in normalized 0..1 HR space."""
    s2_copy = torch.clamp(batch["HR_t1"], 0.0, 1.0)
    s3_t2 = torch.clamp(batch["LR_t2"], 0.0, 1.0)
    s3_delta = torch.clamp(batch["HR_t1"] + (batch["LR_t2"] - batch["LR_t1"]), 0.0, 1.0)
    return {
        "S2_t1_copy": s2_copy,
        "S3_t2_upsample": s3_t2,
        "S2_t1_plus_S3_delta": s3_delta,
    }


def compare_methods(sample, models=None, include_baselines=True, device="cuda"):
    """Run several loaded models and optional simple baselines on one sample."""
    batch = batch_from_sample(sample, device=device)
    results = []
    if include_baselines:
        for name, pred in baseline_predictions(batch).items():
            results.append(make_result(name, pred, batch, metadata=sample.get("metadata"), index=sample.get("index")))
    for name, model in (models or {}).items():
        model_device = next(model.parameters()).device
        model_batch = batch if model_device == batch["HR_t2"].device else batch_from_sample(sample, device=model_device)
        pred = _predict_model(model, model_batch)
        results.append(make_result(name, pred, model_batch, metadata=sample.get("metadata"), index=sample.get("index")))
    return results


def make_result(name, pred, batch, metadata=None, index=None):
    gt = batch["HR_t2"]
    pred = pred.detach()
    gt = gt.detach()
    return {
        "name": name,
        "pred": pred.cpu(),
        "gt": gt.cpu(),
        "inputs": {key: value.detach().cpu() for key, value in batch.items()},
        "metrics": compute_metrics(pred, gt),
        "metadata": metadata or {},
        "index": index,
    }


def compute_metrics(pred, gt, eps=1e-8):
    pred = torch.clamp(pred.float(), 0.0, 1.0)
    gt = torch.clamp(gt.float(), 0.0, 1.0)
    diff = pred - gt
    mse = torch.mean(diff ** 2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(diff))
    psnr = 20.0 * torch.log10(torch.tensor(1.0, device=pred.device) / (rmse + eps))
    p = pred.flatten(2)
    g = gt.flatten(2)
    dot = torch.sum(p * g, dim=1)
    norm = torch.linalg.norm(p, dim=1) * torch.linalg.norm(g, dim=1)
    sam = torch.mean(torch.acos(torch.clamp(dot / (norm + eps), -1.0, 1.0)))
    out = {"MAE": float(mae.cpu()), "RMSE": float(rmse.cpu()), "PSNR": float(psnr.cpu()), "SAM": float(sam.cpu())}
    try:
        import torchmetrics.functional.image as MF

        out["SSIM"] = float(MF.structural_similarity_index_measure(pred, gt, data_range=1.0).detach().cpu())
        out["ERGAS"] = float((MF.error_relative_global_dimensionless_synthesis(pred, gt) / 8.0).detach().cpu())
    except Exception:
        pass
    return out


def metrics_table(results):
    rows = []
    for result in results:
        row = {"method": result["name"], "index": result.get("index", "")}
        row.update(result.get("metrics", {}))
        meta = result.get("metadata", {}) or {}
        for key in ["sample_id", "mgrs_tile", "pair_id", "t1_date", "t2_date", "interval_days", "landcover_major"]:
            row[key] = meta.get(key, "")
        rows.append(row)
    return pd.DataFrame(rows)


def _to_numpy_image(tensor, bands=(2, 1, 0), percentiles=(2, 98)):
    if tensor.ndim == 4:
        tensor = tensor[0]
    arr = tensor.detach().cpu().float().numpy()
    rgb = np.stack([arr[idx] for idx in bands], axis=-1)
    lo, hi = np.percentile(rgb[np.isfinite(rgb)], percentiles)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(rgb)), float(np.nanmax(rgb))
    return np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)


def plot_prediction_result(result, bands=(2, 1, 0), figsize=(12, 4)):
    import matplotlib.pyplot as plt

    pred = result["pred"]
    gt = result["gt"]
    err = torch.mean(torch.abs(pred - gt), dim=1, keepdim=True)
    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    axes[0].imshow(_to_numpy_image(pred, bands=bands))
    axes[0].set_title(f"{result['name']} pred")
    axes[1].imshow(_to_numpy_image(gt, bands=bands))
    axes[1].set_title("GT HR_t2")
    axes[2].imshow(err[0, 0].numpy(), cmap="magma")
    axes[2].set_title(f"Abs error RMSE={result['metrics'].get('RMSE', math.nan):.4f}")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    return fig, axes


def plot_comparison(results, bands=(2, 1, 0), figsize_per_row=(9, 3)):
    import matplotlib.pyplot as plt

    rows = len(results)
    fig, axes = plt.subplots(rows, 3, figsize=(figsize_per_row[0], figsize_per_row[1] * rows), constrained_layout=True)
    if rows == 1:
        axes = np.asarray([axes])
    for ax_row, result in zip(axes, results):
        pred = result["pred"]
        gt = result["gt"]
        err = torch.mean(torch.abs(pred - gt), dim=1, keepdim=True)
        ax_row[0].imshow(_to_numpy_image(pred, bands=bands))
        ax_row[0].set_title(f"{result['name']} pred")
        ax_row[1].imshow(_to_numpy_image(gt, bands=bands))
        ax_row[1].set_title("GT")
        ax_row[2].imshow(err[0, 0].numpy(), cmap="magma")
        ax_row[2].set_title(f"RMSE={result['metrics'].get('RMSE', math.nan):.4f}")
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])
    return fig, axes


def evaluate_indices(lmdb_path, methods, indices, device="cuda", include_baselines=False):
    """Evaluate loaded models over selected LMDB sample indices."""
    rows = []
    for idx in indices:
        sample = load_lmdb_sample(lmdb_path, idx)
        results = compare_methods(sample, models=methods, include_baselines=include_baselines, device=device)
        rows.append(metrics_table(results))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


