import inspect
from pathlib import Path

import numpy as np
import rasterio
import torch

from LibSTFv1.dataset.SlideDataset import DEFAULT_SENTINEL_LR_FACTORS, SentinelLMDBDataset


def load_weights_without_prefix(model, checkpoint_path, strict=True):
    """Load a checkpoint for ad-hoc analysis.

    Formal evaluation should use analysis/run.py. This helper defaults to strict
    loading so a mismatched checkpoint cannot silently produce metrics.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("generator."):
            key = key[len("generator."):]
        elif key.startswith("model."):
            key = key[len("model."):]
        cleaned[key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=strict)
    if missing:
        print(f"Warning: missing keys while loading {checkpoint_path}: {len(missing)}")
    if unexpected:
        print(f"Warning: unexpected keys while loading {checkpoint_path}: {len(unexpected)}")
    if (missing or unexpected) and not strict:
        print("Partial checkpoint loading was explicitly allowed; do not use this for formal metrics without justification.")
    return model


def nearest_upsample_chw(array, factor):
    return np.repeat(np.repeat(array, factor, axis=1), factor, axis=2)


def _pad_to_tile(array, tile_size):
    _, height, width = array.shape
    pad_h = (tile_size - height % tile_size) % tile_size
    pad_w = (tile_size - width % tile_size) % tile_size
    if pad_h == 0 and pad_w == 0:
        return array, height, width
    padded = np.pad(array, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
    return padded, height, width


def _model_call(model, lr_t1, lr_t2, hr_t1, mask=None, input_order="wrapper"):
    """Call either the Lightning wrapper order or a raw HSTFM network order.

    input_order="wrapper" uses (LR_t1, LR_t2, HR_t1, mask), matching train.run.
    input_order="raw_hstfm" uses (LR_t1, HR_t1, LR_t2, mask), matching network.HSTFM.
    """
    signature = inspect.signature(model.forward if hasattr(model, "forward") else model)
    has_mask = "mask" in signature.parameters
    if input_order == "raw_hstfm":
        args = (lr_t1, hr_t1, lr_t2)
    elif input_order == "wrapper":
        args = (lr_t1, lr_t2, hr_t1)
    else:
        raise ValueError("input_order must be 'wrapper' or 'raw_hstfm'")
    if has_mask:
        return model(*args, mask)
    return model(*args)


def predict(model, A_path, a_path, b_path, out_path, band_num=4, lr_norm_factors=DEFAULT_SENTINEL_LR_FACTORS,
            hr_scale=10000.0, lr_to_hr_factor=8, tile_size=512, device="cuda", enlarge_lr=True,
            input_order="wrapper"):
    """Run tiled single-image inference with the repository LMDB scaling.

    This helper is for qualitative analysis and product export. Formal metrics
    should be produced by analysis/run.py so test splits and checkpoint loading
    are recorded consistently.
    """
    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = model.to(device).eval()
    lr_factors = np.asarray(lr_norm_factors, dtype=np.float32).reshape(-1, 1, 1)

    with rasterio.open(A_path) as src:
        A = src.read().astype(np.float32)[:band_num]
        profile = src.profile
    with rasterio.open(a_path) as src:
        a = src.read().astype(np.float32)[:band_num]
    with rasterio.open(b_path) as src:
        b = src.read().astype(np.float32)[:band_num]

    if enlarge_lr:
        a = nearest_upsample_chw(a, lr_to_hr_factor)
        b = nearest_upsample_chw(b, lr_to_hr_factor)
    if A.shape[1:] != a.shape[1:] or A.shape[1:] != b.shape[1:]:
        raise ValueError(f"shape mismatch after LR upsampling: A={A.shape}, a={a.shape}, b={b.shape}")

    A_pad, orig_h, orig_w = _pad_to_tile(A, tile_size)
    a_pad, _, _ = _pad_to_tile(a, tile_size)
    b_pad, _, _ = _pad_to_tile(b, tile_size)
    _, height, width = A_pad.shape
    output = np.zeros((band_num, height, width), dtype=np.float32)
    mask = torch.ones((1, 1, tile_size, tile_size), dtype=torch.float32, device=device)

    with torch.no_grad():
        for row in range(0, height, tile_size):
            for col in range(0, width, tile_size):
                a_tile = a_pad[:, row:row + tile_size, col:col + tile_size] * lr_factors
                b_tile = b_pad[:, row:row + tile_size, col:col + tile_size] * lr_factors
                A_tile = A_pad[:, row:row + tile_size, col:col + tile_size] / hr_scale
                lr_t1 = torch.from_numpy(a_tile[None]).to(device)
                lr_t2 = torch.from_numpy(b_tile[None]).to(device)
                hr_t1 = torch.from_numpy(A_tile[None]).to(device)
                pred = _model_call(model, lr_t1, lr_t2, hr_t1, mask, input_order=input_order)
                if isinstance(pred, dict):
                    pred = pred["pred"]
                output[:, row:row + tile_size, col:col + tile_size] = pred[0].detach().cpu().numpy() * hr_scale

    output = output[:, :orig_h, :orig_w]
    profile.update(count=band_num, dtype="float32", height=orig_h, width=orig_w)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(output)
    return out_path


def save_lmdb_sample_to_tif(dataset, idx, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = dataset[idx]
    paths = {}
    for key in ("LR_t1", "HR_t1", "LR_t2", "HR_t2", "mask"):
        array = sample[key].detach().cpu().numpy()
        path = output_dir / f"{key}_{idx}.tif"
        with rasterio.open(path, "w", driver="GTiff", height=array.shape[1], width=array.shape[2],
                           count=array.shape[0], dtype=array.dtype) as dst:
            dst.write(array)
        paths[key] = str(path)
    return paths
