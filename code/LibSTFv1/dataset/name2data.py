from datetime import datetime
from pathlib import Path


def get_pairs(data_path, num_train, key_Landsat, key_MODIS, val_count=None):
    data_path = Path(data_path)
    lr_dir = data_path / "MODIS"
    hr_dir = data_path / "Landsat"
    if not lr_dir.exists() or not hr_dir.exists():
        raise FileNotFoundError(f"dataset folder must contain MODIS and Landsat directories: {data_path}")

    lr_files = sorted([p.name for p in lr_dir.iterdir() if p.is_file()], key=key_MODIS)
    hr_files = sorted([p.name for p in hr_dir.iterdir() if p.is_file()], key=key_Landsat)
    if len(lr_files) != len(hr_files):
        raise ValueError(f"LR/HR file count mismatch in {data_path}: MODIS={len(lr_files)} Landsat={len(hr_files)}")
    if len(lr_files) < 2:
        raise ValueError(f"dataset needs at least two dates: {data_path}")

    lrs = [str((lr_dir / name).resolve()) for name in lr_files]
    hrs = [str((hr_dir / name).resolve()) for name in hr_files]
    if num_train >= len(lrs):
        raise ValueError(f"num_train={num_train} leaves no validation dates for {data_path}")

    remaining = len(lrs) - num_train
    if val_count is None:
        if remaining >= 4:
            val_count = max(2, remaining // 2)
        else:
            val_count = remaining
    val_lrs = lrs[num_train:num_train + val_count]
    val_hrs = hrs[num_train:num_train + val_count]
    test_lrs = lrs[num_train + val_count:]
    test_hrs = hrs[num_train + val_count:]
    if len(test_lrs) < 2:
        test_lrs, test_hrs = val_lrs, val_hrs
        split_note = "test_reuses_validation_dates_insufficient_remaining_dates"
    else:
        split_note = "standard_non_overlapping_dates"
    if 0 < len(val_lrs) < 2:
        val_lrs = [lrs[num_train - 1]] + val_lrs
        val_hrs = [hrs[num_train - 1]] + val_hrs
        split_note = "val_uses_last_train_date_as_t1"
    if 0 < len(test_lrs) < 2:
        test_lrs = [lrs[num_train - 1]] + test_lrs
        test_hrs = [hrs[num_train - 1]] + test_hrs
        split_note = split_note + ";test_uses_last_train_date_as_t1"
    if len(val_lrs) < 2:
        raise ValueError(f"validation split needs at least one date after training: {data_path}")
    if len(test_lrs) < 2:
        test_lrs, test_hrs = val_lrs, val_hrs
        split_note = split_note + ";test_reuses_validation_dates"

    return {
        "train": {"LRs": lrs[:num_train], "HRs": hrs[:num_train]},
        "val": {"LRs": val_lrs, "HRs": val_hrs},
        "test": {"LRs": test_lrs, "HRs": test_hrs},
        "split_note": split_note,
    }


_DATASET_SPECS = {
    "CIA": {
        "relative_path": "CIA",
        "num_train": 11,
        "band": 6,
        "rgb_c": [3, 2, 1],
        "img_size": (1792, 1280),
        "slide_size": 512,
        "max_value": 10000.0,
        "key_Landsat": lambda x: x[13:21],
        "key_MODIS": lambda x: x[9:16],
    },
    "LGC": {
        "relative_path": "LGC",
        "num_train": 9,
        "band": 6,
        "rgb_c": [3, 2, 1],
        "img_size": (2560, 3072),
        "slide_size": 512,
        "max_value": 10000.0,
        "key_Landsat": lambda x: x[:8],
        "key_MODIS": lambda x: x[9:16],
    },
    "AHB": {
        "relative_path": "Datasets/AHB",
        "num_train": 20,
        "band": 6,
        "rgb_c": [3, 2, 1],
        "img_size": (2480, 2800),
        "slide_size": 512,
        "max_value": 255.0,
        "key_Landsat": lambda f: datetime.strptime(f[2:-4], "%Y-%m-%d"),
        "key_MODIS": lambda f: datetime.strptime(f[2:-4], "%Y-%m-%d"),
    },
    "DX": {
        "relative_path": "Datasets/Daxing",
        "num_train": 20,
        "band": 6,
        "rgb_c": [3, 2, 1],
        "img_size": (1640, 1640),
        "slide_size": 512,
        "max_value": 255.0,
        "key_Landsat": lambda f: datetime.strptime(f[2:-4], "%Y-%m-%d"),
        "key_MODIS": lambda f: datetime.strptime(f[2:-4], "%Y-%m-%d"),
    },
    "TJ": {
        "relative_path": "Datasets/Tianjin",
        "num_train": 20,
        "band": 6,
        "rgb_c": [3, 2, 1],
        "img_size": (2100, 1970),
        "slide_size": 512,
        "max_value": 255.0,
        "key_Landsat": lambda f: datetime.strptime(f[2:-4], "%Y-%m-%d"),
        "key_MODIS": lambda f: datetime.strptime(f[2:-4], "%Y-%m-%d"),
    },
    "WH": {
        "relative_path": "Wuhan",
        "num_train": 5,
        "band": 4,
        "rgb_c": [2, 1, 0],
        "img_size": (1000, 1000),
        "slide_size": 512,
        "max_value": 10000.0,
        "key_Landsat": lambda x: x[2:],
        "key_MODIS": lambda x: x[2:],
    },
    "IC": {
        "relative_path": "IC",
        "num_train": 3,
        "band": 4,
        "rgb_c": [2, 1, 0],
        "img_size": (1500, 1500),
        "slide_size": 512,
        "max_value": 1.0,
        "key_Landsat": lambda x: x[3:11],
        "key_MODIS": lambda x: x[3:11],
    },
    "BC": {
        "relative_path": "BC",
        "num_train": 3,
        "band": 4,
        "rgb_c": [2, 1, 0],
        "img_size": (1500, 1500),
        "slide_size": 512,
        "max_value": 1.0,
        "key_Landsat": lambda x: x[3:11],
        "key_MODIS": lambda x: x[3:11],
    },
}


def available_datasets():
    return sorted(_DATASET_SPECS)


def get_dataset_config(name, data_root="data"):
    if name not in _DATASET_SPECS:
        raise KeyError(f"unknown dataset {name!r}; choices: {available_datasets()}")
    spec = dict(_DATASET_SPECS[name])
    data_path = Path(data_root) / spec.pop("relative_path")
    pairs = get_pairs(
        data_path=data_path,
        num_train=spec["num_train"],
        key_Landsat=spec["key_Landsat"],
        key_MODIS=spec["key_MODIS"],
    )
    return {**pairs, **spec, "name": name, "data_path": str(data_path)}


class _LazyName2Data(dict):
    def __getitem__(self, key):
        return get_dataset_config(key)

    def keys(self):
        return available_datasets()


name2data = _LazyName2Data()
