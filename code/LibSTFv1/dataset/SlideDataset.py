import io
import json
import os
import pickle
import random
import multiprocessing as mp
from pathlib import Path

import blosc
import lmdb
import numpy as np
import pytorch_lightning as pl
import tifffile
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split


DEFAULT_SENTINEL_LR_FACTORS = (0.0113963, 0.0100382, 0.00897469, 0.00605576)


def _slide(x, k):
    _, channels, height, width = x.shape
    if height < k or width < k:
        raise ValueError(f"image size {(height, width)} is smaller than slide size {k}")
    return (
        x.unfold(2, k, k)
        .unfold(3, k, k)
        .permute(0, 2, 3, 1, 4, 5)
        .contiguous()
        .view(-1, channels, k, k)
    )


def _read_tif_chw(path):
    data = np.asarray(tifffile.imread(path))
    if data.ndim != 3:
        raise ValueError(f"unsupported image shape for {path}: {data.shape}")
    if data.shape[-1] <= 16:
        data = np.transpose(data, (2, 0, 1))
    elif data.shape[0] <= 16:
        pass
    else:
        raise ValueError(f"cannot infer channel dimension for {path}: {data.shape}")
    data = data.astype(np.float32, copy=False)
    data[data < 0] = 0
    return data


class SlideDataset(Dataset):
    """Temporal adjacent-pair TIFF dataset for CIA/LGC/AHB/IC/BC style data.

    Native benchmark datasets do not have cloud/no-data masks. A full-one mask is
    returned only to keep a shared batch interface with HSTFM and LMDB datasets.
    """

    def __init__(self, pairs, img_size=None, slide_size=512, max_val=10000.0, return_mask=True):
        self.pairs = pairs
        self.slide_size = slide_size
        self.img_size = img_size
        self.max_val = float(max_val)
        self.return_mask = return_mask

        lr_patches = []
        hr_patches = []
        for lr_path, hr_path in zip(pairs["LRs"], pairs["HRs"]):
            lr_data = _read_tif_chw(lr_path)
            hr_data = _read_tif_chw(hr_path)
            if lr_data.shape != hr_data.shape:
                raise ValueError(f"LR/HR shape mismatch: {lr_path} {lr_data.shape} vs {hr_path} {hr_data.shape}")
            lr_patches.append(_slide(torch.from_numpy(lr_data).unsqueeze(0), slide_size).numpy() / self.max_val)
            hr_patches.append(_slide(torch.from_numpy(hr_data).unsqueeze(0), slide_size).numpy() / self.max_val)

        if len(lr_patches) < 2:
            raise ValueError("at least two temporal images are required to form t1/t2 pairs")

        self.LRs_arr = np.asarray(lr_patches, dtype=np.float32)
        self.HRs_arr = np.asarray(hr_patches, dtype=np.float32)
        self.num_patches = self.LRs_arr.shape[1]
        self.channels = self.LRs_arr.shape[2]

    def __len__(self):
        return (self.LRs_arr.shape[0] - 1) * self.num_patches

    def __getitem__(self, index):
        image_idx = index // self.num_patches
        patch_idx = index % self.num_patches
        lr_t1 = torch.from_numpy(self.LRs_arr[image_idx][patch_idx])
        hr_t1 = torch.from_numpy(self.HRs_arr[image_idx][patch_idx])
        lr_t2 = torch.from_numpy(self.LRs_arr[image_idx + 1][patch_idx])
        hr_t2 = torch.from_numpy(self.HRs_arr[image_idx + 1][patch_idx])
        sample = {
            "LR_t1": lr_t1,
            "HR_t1": hr_t1,
            "LR_t2": lr_t2,
            "HR_t2": hr_t2,
            "metadata": {
                "source": "native_tif",
                "image_idx": image_idx,
                "patch_idx": patch_idx,
            },
        }
        if self.return_mask:
            sample["mask"] = torch.ones((1, self.slide_size, self.slide_size), dtype=torch.float32)
        return sample


class plNBUDataset(pl.LightningDataModule):
    def __init__(self, dataset_dict, batch_size, num_workers=4, pin_memory=True):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        img_size = dataset_dict.get("img_size")
        slide_size = dataset_dict["slide_size"]
        max_val = dataset_dict["max_value"]
        self.dataset_train = SlideDataset(dataset_dict["train"], img_size, slide_size, max_val)
        self.dataset_val = SlideDataset(dataset_dict["val"], img_size, slide_size, max_val)
        self.dataset_test = SlideDataset(dataset_dict.get("test", dataset_dict["val"]), img_size, slide_size, max_val)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, drop_last=False, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=1, shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=1, shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, drop_last=False)


def worker_init_fn(worker_id):
    seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(seed)
    random.seed(seed)


class CurriculumMaskCallback(pl.Callback):
    def __init__(self, mask_prob_shared, max_prob=0.8, start_epoch=5, warmup_epochs=20):
        super().__init__()
        self.mask_prob_shared = mask_prob_shared
        self.max_prob = max_prob
        self.start_epoch = start_epoch
        self.warmup_epochs = warmup_epochs
        self.train_prob = 0.0

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch < self.start_epoch:
            self.train_prob = 0.0
        else:
            progress = min(max((trainer.current_epoch - self.start_epoch) / self.warmup_epochs, 0.0), 1.0)
            self.train_prob = progress * self.max_prob
        self.mask_prob_shared.value = self.train_prob
        if trainer.is_global_zero:
            print(f"[CurriculumMask] epoch={trainer.current_epoch} train_mask_prob={self.train_prob:.4f}")

    def on_validation_epoch_start(self, trainer, pl_module):
        self.mask_prob_shared.value = 0.0

    def on_validation_epoch_end(self, trainer, pl_module):
        self.mask_prob_shared.value = self.train_prob


class SentinelLMDBDataset(Dataset):
    """Reader for the server/CDSE LMDB samples generated in this repository.

    Expected sample payload: pickled dict with compressed arrays in this order:
    [LR_t1, HR_t1, LR_t2, HR_t2]. LR is Sentinel-3 4x64x64 uint16, HR is
    Sentinel-2 4x512x512 uint16.
    """

    def __init__(self, lmdb_path, patch_size=512, mask_prob_shared=None, mask=True,
                 lr_norm_factors=DEFAULT_SENTINEL_LR_FACTORS, hr_scale=10000.0,
                 deterministic_mask=False):
        self.lmdb_path = str(lmdb_path)
        self.env = None
        self.patch_size = int(patch_size)
        self.mask_enabled = bool(mask)
        self.mask_prob_shared = mask_prob_shared if mask_prob_shared is not None else mp.Value("d", 0.0)
        self.norm_factors = torch.tensor(lr_norm_factors, dtype=torch.float32).view(4, 1, 1)
        self.hr_scale = float(hr_scale)
        self.deterministic_mask = bool(deterministic_mask)

        with lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False).begin(write=False) as txn:
            metadata_bytes = txn.get(b"__metadata__")
            if metadata_bytes:
                self.metadata = pickle.loads(metadata_bytes)
                self.base_length = int(self.metadata["total_samples"])
            else:
                self.metadata = None
                self.base_length = txn.stat()["entries"]

        self.tile_size_x = 512
        self.tile_size_y = 512
        if self.tile_size_x % self.patch_size != 0 or self.tile_size_y % self.patch_size != 0:
            raise ValueError("patch_size must divide 512 for repository LMDB samples")
        self.num_patch_x = self.tile_size_x // self.patch_size
        self.num_patch_y = self.tile_size_y // self.patch_size
        self.length = self.base_length * self.num_patch_x * self.num_patch_y

    def _init_env(self):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

    def __len__(self):
        return self.length

    @staticmethod
    def unpack_arrays(compressed_data):
        raw_bytes = blosc.decompress(compressed_data)
        buf = io.BytesIO(raw_bytes)
        arrays = []
        while buf.tell() < len(raw_bytes):
            arrays.append(np.load(buf, allow_pickle=False))
        return arrays

    @staticmethod
    def enlarge_size(x, factor=8):
        return x.repeat_interleave(factor, dim=1).repeat_interleave(factor, dim=2)

    @staticmethod
    def _generate_low_res_mask(h, w, prob, rng=None):
        rng = rng or random
        rand = rng.random
        randint = rng.randint if rng is random else lambda low, high: int(rng.integers(low, high + 1))
        if rand() >= prob:
            return torch.ones((1, h, w), dtype=torch.float32)
        mask = torch.ones((1, h, w), dtype=torch.float32)
        if rand() < 0.5:
            for _ in range(randint(1, 3)):
                hole_w = randint(max(1, w // 4), max(1, w // 2))
                hole_h = randint(max(1, h // 4), max(1, h // 2))
                x1 = randint(0, w - hole_w)
                y1 = randint(0, h - hole_h)
                mask[:, y1:y1 + hole_h, x1:x1 + hole_w] = 0.0
        else:
            if rng is random:
                noise = torch.rand((1, h, w))
            else:
                noise = torch.from_numpy(rng.random((1, h, w))).float()
            mask[noise > 0.6] = 0.0
        return mask

    def __getitem__(self, idx):
        tile_idx = idx // (self.num_patch_x * self.num_patch_y)
        patch_id = idx % (self.num_patch_x * self.num_patch_y)
        px = patch_id // self.num_patch_y
        py = patch_id % self.num_patch_y

        self._init_env()
        key = f"{tile_idx:08d}".encode()
        with self.env.begin(write=False, buffers=True) as txn:
            data_bytes = txn.get(key)
            if data_bytes is None:
                raise KeyError(f"missing LMDB sample key {key!r}")
            sample_data = pickle.loads(bytes(data_bytes))
            arrays = self.unpack_arrays(sample_data["arrays"])
            metadata = sample_data.get("metadata")

        a, A, b, B = [torch.as_tensor(x, dtype=torch.float32) for x in arrays]
        if self.mask_enabled:
            rng = np.random.default_rng(idx) if self.deterministic_mask else None
            mask_lr = self._generate_low_res_mask(b.shape[1], b.shape[2], self.mask_prob_shared.value, rng=rng)
        else:
            mask_lr = torch.ones((1, b.shape[1], b.shape[2]), dtype=torch.float32)
        b = b * mask_lr

        a_up = self.enlarge_size(a, 8)
        b_up = self.enlarge_size(b, 8)
        mask_up = self.enlarge_size(mask_lr, 8)

        x1, x2 = px * self.patch_size, (px + 1) * self.patch_size
        y1, y2 = py * self.patch_size, (py + 1) * self.patch_size
        return {
            "LR_t1": a_up[:, x1:x2, y1:y2] * self.norm_factors,
            "HR_t1": A[:, x1:x2, y1:y2] / self.hr_scale,
            "LR_t2": b_up[:, x1:x2, y1:y2] * self.norm_factors,
            "HR_t2": B[:, x1:x2, y1:y2] / self.hr_scale,
            "mask": mask_up[:, x1:x2, y1:y2],
            "metadata": metadata,
        }

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def __del__(self):
        self.close()


class SentinelLMDBDataModule(pl.LightningDataModule):
    def __init__(self, lmdb_path, batch_size=8, num_workers=4, sample_num=None, random_seed=42,
                 patch_size=512, mask=True, split=(0.9, 0.05, 0.05), pin_memory=True, split_by_group=True,
                 split_file=None, save_split_file=None, eval_mask_prob=0.0, deterministic_eval_mask=True,
                 train_sample_num=None, val_sample_num=None, test_sample_num=None, split_seed=None):
        super().__init__()
        self.lmdb_path = lmdb_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_num = sample_num
        self.train_sample_num = train_sample_num
        self.val_sample_num = val_sample_num
        self.test_sample_num = test_sample_num
        self.seed = random_seed
        self.split_seed = random_seed if split_seed is None else split_seed
        self.patch_size = patch_size
        self.mask = mask
        self.split = split
        self.pin_memory = pin_memory
        self.split_by_group = split_by_group
        self.mask_prob_shared = mp.Value("d", 0.0)
        self.split_file = Path(split_file) if split_file else None
        self.save_split_file = Path(save_split_file) if save_split_file else None
        self.eval_mask_prob = float(eval_mask_prob)
        self.deterministic_eval_mask = bool(deterministic_eval_mask)

    def set_mask_probability(self, prob):
        self.mask_prob_shared.value = float(prob)

    def _metadata_group_indices(self, dataset):
        if not self.split_by_group:
            return None
        groups = {}
        try:
            dataset._init_env()
            patches_per_tile = dataset.num_patch_x * dataset.num_patch_y
            with dataset.env.begin(write=False, buffers=True) as txn:
                for tile_idx in range(dataset.base_length):
                    data_bytes = txn.get(f"{tile_idx:08d}".encode())
                    if data_bytes is None:
                        return None
                    sample_data = pickle.loads(bytes(data_bytes))
                    metadata = sample_data.get("metadata") or {}
                    group_id = (metadata.get("split_group") or metadata.get("mgrs_tile") or metadata.get("pair_id") or metadata.get("group_id") or metadata.get("tile"))
                    if group_id is None:
                        return None
                    groups.setdefault(str(group_id), [])
                    base_idx = tile_idx * patches_per_tile
                    groups[str(group_id)].extend(range(base_idx, base_idx + patches_per_tile))
        except Exception as exc:
            print(f"Warning: group-level split unavailable, fallback to sample split: {exc}")
            return None
        return groups

    def _target_sizes(self, total_size):
        explicit = any(v is not None for v in (self.train_sample_num, self.val_sample_num, self.test_sample_num))
        if explicit:
            if self.train_sample_num is None or self.val_sample_num is None or self.test_sample_num is None:
                raise ValueError("train_sample_num, val_sample_num, and test_sample_num must be set together")
            train_size = int(self.train_sample_num)
            val_size = int(self.val_sample_num)
            test_size = int(self.test_sample_num)
            requested = train_size + val_size + test_size
            if requested > total_size:
                raise ValueError(f"requested split size {requested} exceeds dataset size {total_size}")
            return train_size, val_size, test_size, requested, "explicit_counts"

        use_size = min(self.sample_num, total_size) if self.sample_num is not None else total_size
        train_size = int(use_size * self.split[0])
        val_size = int(use_size * self.split[1])
        test_size = use_size - train_size - val_size
        if val_size == 0 and use_size >= 3:
            val_size = 1
            train_size -= 1
        if test_size == 0 and use_size >= 3:
            test_size = 1
            train_size -= 1
        return train_size, val_size, test_size, use_size, "ratio_counts"

    def _split_indices(self, full_dataset, total_size, generator):
        train_target, val_target, test_target, use_size, target_mode = self._target_sizes(total_size)
        groups = self._metadata_group_indices(full_dataset)
        if not groups:
            indices = torch.randperm(len(full_dataset), generator=generator)[:use_size].tolist()
            return (indices[:train_target],
                    indices[train_target:train_target + val_target],
                    indices[train_target + val_target:train_target + val_target + test_target]), f"sample_random_{target_mode}"
        if len(groups) < 3:
            print("Warning: fewer than 3 metadata groups; fallback to sample-level split for train/val/test.")
            indices = torch.randperm(len(full_dataset), generator=generator)[:use_size].tolist()
            return (indices[:train_target],
                    indices[train_target:train_target + val_target],
                    indices[train_target + val_target:train_target + val_target + test_target]), f"sample_random_few_groups_{target_mode}"

        group_names = sorted(groups)
        group_order = torch.randperm(len(group_names), generator=generator).tolist()
        split_lists = [[], [], []]
        targets = [train_target, val_target, test_target]
        split_id = 0
        for pos in group_order:
            if split_id >= 3:
                break
            group_indices = groups[group_names[pos]]
            split_lists[split_id].extend(group_indices)
            if len(split_lists[split_id]) >= targets[split_id]:
                split_id += 1
        if any(len(items) == 0 for items in split_lists):
            raise ValueError("group-level split produced an empty train/val/test split; reduce requested sizes or use --sample_random_split")
        return tuple(split_lists), f"metadata_group_{target_mode}"

    def _load_split(self):
        if not self.split_file:
            return None
        data = json.loads(self.split_file.read_text(encoding="utf-8"))
        return data["train_indices"], data["val_indices"], data["test_indices"], data.get("split_mode", "loaded")

    def _save_split(self, train_indices, val_indices, test_indices, split_mode):
        if not self.save_split_file:
            return
        self.save_split_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "lmdb_path": str(self.lmdb_path),
            "seed": self.seed,
            "sample_num": self.sample_num,
            "train_sample_num": self.train_sample_num,
            "val_sample_num": self.val_sample_num,
            "test_sample_num": self.test_sample_num,
            "patch_size": self.patch_size,
            "split": list(self.split),
            "split_by_group": self.split_by_group,
            "split_seed": self.split_seed,
            "split_mode": split_mode,
            "actual_counts": {"train": len(train_indices), "val": len(val_indices), "test": len(test_indices)},
            "train_indices": list(map(int, train_indices)),
            "val_indices": list(map(int, val_indices)),
            "test_indices": list(map(int, test_indices)),
        }
        self.save_split_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def setup(self, stage=None):
        pl.seed_everything(self.seed, workers=True)
        full_dataset = SentinelLMDBDataset(
            self.lmdb_path,
            patch_size=self.patch_size,
            mask_prob_shared=self.mask_prob_shared,
            mask=self.mask,
            deterministic_mask=stage in ("test", "validate") and self.deterministic_eval_mask,
        )
        total_size = len(full_dataset)
        generator = torch.Generator().manual_seed(self.split_seed)
        loaded_split = self._load_split()
        if loaded_split is not None:
            train_indices, val_indices, test_indices, split_mode = loaded_split
            self.train_dataset = Subset(full_dataset, train_indices)
            self.val_dataset = Subset(full_dataset, val_indices)
            self.test_dataset = Subset(full_dataset, test_indices)
            self.set_mask_probability(self.eval_mask_prob if stage in ("test", "validate") else 0.0)
            print(
                f"Data Loaded: train={len(self.train_dataset)}, val={len(self.val_dataset)}, "
                f"test={len(self.test_dataset)}, mask={self.mask}, split={split_mode}, "
                f"mask_prob={self.mask_prob_shared.value:.4f}"
            )
            return

        split_result, split_mode = self._split_indices(full_dataset, total_size, generator)
        train_indices, val_indices, test_indices = split_result
        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(full_dataset, val_indices)
        self.test_dataset = Subset(full_dataset, test_indices)

        self._save_split(train_indices, val_indices, test_indices, split_mode)
        self.set_mask_probability(self.eval_mask_prob if stage in ("test", "validate") else 0.0)

        print(
            f"Data Loaded: train={len(self.train_dataset)}, val={len(self.val_dataset)}, "
            f"test={len(self.test_dataset)}, mask={self.mask}, split={split_mode}, "
            f"mask_prob={self.mask_prob_shared.value:.4f}"
        )

    def _loader(self, dataset, shuffle, batch_size=None):
        return DataLoader(
            dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, False, batch_size=1)

    def test_dataloader(self):
        return self._loader(self.test_dataset, False, batch_size=1)

    def get_curriculum_callback(self, max_prob=0.8, start_epoch=5, warmup_epochs=20):
        return CurriculumMaskCallback(self.mask_prob_shared, max_prob=max_prob, start_epoch=start_epoch,
                                      warmup_epochs=warmup_epochs)

