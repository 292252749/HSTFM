"""Notebook helpers for inspecting S2/S3 512 LMDB samples.

Notebook import:
    import importlib.util
    p = "/data1/code/s23_pipeline_v2/get_data/s23_pipeline_v2/05_lmdb_visual_check.py"
    spec = importlib.util.spec_from_file_location("lmdb_vis", p)
    V = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(V)

    env = V.open_lmdb("/data1/code/s23_pipeline_v2/s23_pipeline_v2_out/china/lmdb_512")
    V.describe_lmdb(env)
    sample = V.read_sample(env, 0)
    V.plot_sample(sample)
    V.plot_random_samples(env, n=4)
"""

import io
import json
import pickle
import random

import numpy as np


ARRAY_NAMES = ("lr_t1", "hr_t1", "lr_t2", "hr_t2")
S2_RGB_BANDS = (2, 1, 0)  # B04, B03, B02 from [B02, B03, B04, B8A]
S3_RGB_BANDS = (2, 1, 0)  # pseudo RGB from [Oa04, Oa06, Oa08, Oa17]
S2_BAD_VALUE = 0
S3_BAD_VALUE = 65535


def _import_lmdb():
    import lmdb

    return lmdb


def open_lmdb(path, **kwargs):
    """Open an LMDB read-only for notebook inspection."""
    lmdb = _import_lmdb()
    return lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=kwargs.pop("max_readers", 128),
        **kwargs,
    )


def _load_pickle(value):
    if value is None:
        return None
    return pickle.loads(value)


def read_lmdb_metadata(env):
    """Read the global __metadata__ record written by 04_build_lmdb_from_pairs.py."""
    with env.begin(write=False) as txn:
        return _load_pickle(txn.get(b"__metadata__"))


def describe_lmdb(env, print_result=True):
    """Return a compact summary: metadata, entry count, and first numeric keys."""
    meta = read_lmdb_metadata(env) or {}
    with env.begin(write=False) as txn:
        stat = txn.stat()
        keys = []
        cur = txn.cursor()
        for key, _ in cur:
            text = key.decode("utf-8", errors="ignore")
            if text != "__metadata__":
                keys.append(text)
            if len(keys) >= 10:
                break
    out = {
        "schema": meta.get("schema", ""),
        "dataset_group": meta.get("dataset_group", ""),
        "dataset_name": meta.get("dataset_name", ""),
        "metadata_total_samples": meta.get("total_samples", ""),
        "lmdb_entries": stat.get("entries", ""),
        "sample_entries_estimate": max(0, int(stat.get("entries", 0)) - (1 if meta else 0)),
        "first_keys": keys,
        "hr_shape": meta.get("hr_shape", ""),
        "lr_shape": meta.get("lr_shape", ""),
        "dtype": meta.get("dtype", ""),
    }
    if print_result:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def list_keys(env, limit=20, start_key=None):
    """List sample keys, excluding __metadata__."""
    keys = []
    start = str(start_key).encode("utf-8") if start_key is not None else None
    with env.begin(write=False) as txn:
        cur = txn.cursor()
        ok = cur.set_range(start) if start is not None else cur.first()
        while ok and len(keys) < limit:
            key = cur.key().decode("utf-8", errors="ignore")
            if key != "__metadata__":
                keys.append(key)
            ok = cur.next()
    return keys


def random_keys(env, n=8, seed=0):
    """Sample existing numeric keys using metadata total_samples when available."""
    meta = read_lmdb_metadata(env) or {}
    total = meta.get("total_samples")
    if total in (None, "", 0):
        keys = list_keys(env, limit=1000000)
        return random.sample(keys, min(n, len(keys)))
    rng = random.Random(seed)
    candidates = [f"{idx:08d}" for idx in rng.sample(range(int(total)), min(n, int(total)))]
    out = []
    with env.begin(write=False) as txn:
        for key in candidates:
            if txn.get(key.encode("utf-8")) is not None:
                out.append(key)
    return out


def unpack_arrays(packed):
    """Decompress the packed array blob into [LR_t1, HR_t1, LR_t2, HR_t2]."""
    import blosc

    raw = blosc.decompress(packed)
    buf = io.BytesIO(raw)
    arrays = []
    while buf.tell() < len(raw):
        arrays.append(np.load(buf, allow_pickle=False))
    return arrays


def read_sample(env, key, with_arrays=True):
    """Read one sample by integer index or string LMDB key."""
    if isinstance(key, int):
        key = f"{key:08d}"
    key = str(key)
    with env.begin(write=False) as txn:
        payload = _load_pickle(txn.get(key.encode("utf-8")))
    if payload is None:
        raise KeyError(f"LMDB key not found: {key}")
    sample = {"key": key, "metadata": payload.get("metadata", {})}
    if with_arrays:
        arrays = unpack_arrays(payload["arrays"])
        sample["arrays"] = dict(zip(ARRAY_NAMES, arrays))
    return sample


def read_samples(env, keys):
    return [read_sample(env, key) for key in keys]


def valid_mask(arr, bad_value=None):
    arr = np.asarray(arr)
    mask = np.isfinite(arr)
    if bad_value is not None:
        mask &= arr != bad_value
    return mask


def normalize_image(img, bad_value=None, percentiles=(2, 98)):
    """Scale an image to 0..1 for display using robust percentiles."""
    img = np.asarray(img, dtype=np.float32)
    mask = valid_mask(img, bad_value)
    if not np.any(mask):
        return np.zeros_like(img, dtype=np.float32)
    lo, hi = np.percentile(img[mask], percentiles)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(img[mask])), float(np.max(img[mask]))
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    out = (img - lo) / (hi - lo)
    out[~mask] = 0
    return np.clip(out, 0, 1)


def rgb_from_array(arr, sensor="s2", percentiles=(2, 98)):
    """Create display RGB from a CHW array."""
    arr = np.asarray(arr)
    bands = S2_RGB_BANDS if sensor.lower() == "s2" else S3_RGB_BANDS
    bad_value = S2_BAD_VALUE if sensor.lower() == "s2" else S3_BAD_VALUE
    rgb = np.stack([arr[idx] for idx in bands], axis=-1)
    return normalize_image(rgb, bad_value=bad_value, percentiles=percentiles)


def change_image(arr_t1, arr_t2, sensor="s2", percentiles=(2, 98)):
    """Mean absolute spectral change image for quick quality checks."""
    bad_value = S2_BAD_VALUE if sensor.lower() == "s2" else S3_BAD_VALUE
    a = np.asarray(arr_t1, dtype=np.float32)
    b = np.asarray(arr_t2, dtype=np.float32)
    valid = valid_mask(a, bad_value) & valid_mask(b, bad_value)
    diff = np.mean(np.abs(b - a), axis=0)
    diff[~np.all(valid, axis=0)] = 0
    return normalize_image(diff, bad_value=None, percentiles=percentiles)


def ndvi(arr, red_idx=2, nir_idx=3, bad_value=0):
    """Compute NDVI from a CHW array; default indices match S2 [B02,B03,B04,B8A]."""
    arr = np.asarray(arr, dtype=np.float32)
    red = arr[red_idx]
    nir = arr[nir_idx]
    mask = valid_mask(red, bad_value) & valid_mask(nir, bad_value) & ((nir + red) != 0)
    out = np.full(red.shape, np.nan, dtype=np.float32)
    out[mask] = (nir[mask] - red[mask]) / (nir[mask] + red[mask])
    return out


def metadata_table(samples):
    """Return a pandas DataFrame for a list of samples or one sample."""
    import pandas as pd

    if isinstance(samples, dict):
        samples = [samples]
    rows = []
    for sample in samples:
        row = dict(sample.get("metadata", {}))
        row["key"] = sample.get("key", row.get("lmdb_key", ""))
        rows.append(row)
    return pd.DataFrame(rows)


def print_sample_summary(sample, fields=None):
    """Print key metadata fields and array shapes/dtypes."""
    if fields is None:
        fields = [
            "sample_id", "lmdb_key", "pair_id", "mgrs_tile", "t1_date", "t2_date",
            "interval_days", "patch_row", "patch_col", "land_fraction",
            "water_fraction", "nodata_fraction", "landcover_major",
            "t1_cloud", "t2_cloud", "s2_change_mean", "s3_change_mean",
        ]
    meta = sample.get("metadata", {})
    print(json.dumps({key: meta.get(key, "") for key in fields}, indent=2, ensure_ascii=False))
    arrays = sample.get("arrays", {})
    for name in ARRAY_NAMES:
        arr = arrays.get(name)
        if arr is not None:
            print(f"{name}: shape={arr.shape} dtype={arr.dtype} min={np.nanmin(arr)} max={np.nanmax(arr)}")


def plot_sample(sample, figsize=(13, 8), percentiles=(2, 98), show_summary=True):
    """Plot HR S2, LR S3, and change maps for one sample."""
    import matplotlib.pyplot as plt

    arrays = sample["arrays"]
    fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
    panels = [
        (axes[0, 0], rgb_from_array(arrays["hr_t1"], "s2", percentiles), "S2 t1 RGB"),
        (axes[0, 1], rgb_from_array(arrays["hr_t2"], "s2", percentiles), "S2 t2 RGB"),
        (axes[0, 2], change_image(arrays["hr_t1"], arrays["hr_t2"], "s2", percentiles), "S2 abs change"),
        (axes[1, 0], rgb_from_array(arrays["lr_t1"], "s3", percentiles), "S3 t1 pseudo RGB"),
        (axes[1, 1], rgb_from_array(arrays["lr_t2"], "s3", percentiles), "S3 t2 pseudo RGB"),
        (axes[1, 2], change_image(arrays["lr_t1"], arrays["lr_t2"], "s3", percentiles), "S3 abs change"),
    ]
    for ax, img, title in panels:
        ax.imshow(img, cmap="magma" if img.ndim == 2 else None)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    meta = sample.get("metadata", {})
    fig.suptitle(
        f"key={sample.get('key', '')} mgrs={meta.get('mgrs_tile', '')} "
        f"{meta.get('t1_date', '')}->{meta.get('t2_date', '')} "
        f"land={meta.get('land_fraction', '')} water={meta.get('water_fraction', '')}",
        fontsize=11,
    )
    if show_summary:
        print_sample_summary(sample)
    return fig, axes


def plot_random_samples(env, n=4, seed=0, figsize_per_sample=(5, 5), percentiles=(2, 98)):
    """Plot S2 t1/t2 RGB for several random samples."""
    import matplotlib.pyplot as plt

    keys = random_keys(env, n=n, seed=seed)
    samples = read_samples(env, keys)
    fig, axes = plt.subplots(len(samples), 2, figsize=(figsize_per_sample[0] * 2, figsize_per_sample[1] * len(samples)), constrained_layout=True)
    if len(samples) == 1:
        axes = np.asarray([axes])
    for row, sample in zip(axes, samples):
        meta = sample.get("metadata", {})
        row[0].imshow(rgb_from_array(sample["arrays"]["hr_t1"], "s2", percentiles))
        row[1].imshow(rgb_from_array(sample["arrays"]["hr_t2"], "s2", percentiles))
        row[0].set_title(f"{sample['key']} S2 t1 {meta.get('t1_date', '')}")
        row[1].set_title(f"S2 t2 {meta.get('t2_date', '')} {meta.get('mgrs_tile', '')}")
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
    return fig, axes, samples


def inspect_key(env, key):
    """Read, summarize, and plot one key in a notebook."""
    sample = read_sample(env, key)
    plot_sample(sample)
    return sample



