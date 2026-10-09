import argparse
import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

from train.run import SUPPORTED_MODELS, build_model


class DummySTFDataset(Dataset):
    def __init__(self, bands, size, length=1):
        self.bands = bands
        self.size = size
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        shape = (self.bands, self.size, self.size)
        lr_t1 = torch.rand(shape, dtype=torch.float32)
        lr_t2 = torch.rand(shape, dtype=torch.float32)
        hr_t1 = torch.rand(shape, dtype=torch.float32)
        hr_t2 = torch.rand(shape, dtype=torch.float32)
        return {
            "LR_t1": lr_t1,
            "LR_t2": lr_t2,
            "HR_t1": hr_t1,
            "HR_t2": hr_t2,
            "mask": torch.ones((1, self.size, self.size), dtype=torch.float32),
        }


class DummySTFDataModule(pl.LightningDataModule):
    def __init__(self, bands, size, batch_size):
        super().__init__()
        self.dataset = DummySTFDataset(bands=bands, size=size, length=max(batch_size, 1))
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)


def base_args(model, epochs):
    return SimpleNamespace(
        model=model,
        epochs=epochs,
        data_source="lmdb",
        ergas_ratio=8.0,
        rgb_c=[1, 2, 0],
        dataset="SmokeTest",
    )


def run_one_model(model_name, args):
    result = {"model": model_name, "status": "unknown", "stage": None, "error": None}
    cfg = {"band": args.bands, "rgb_c": [1, 2, 0], "name": "SmokeTest"}
    try:
        result["stage"] = "build_model"
        model = build_model(base_args(model_name, args.epochs), cfg)
        result["stage"] = "trainer_fit_one_step"
        dm = DummySTFDataModule(bands=args.bands, size=args.size, batch_size=args.batch_size)
        trainer = pl.Trainer(
            accelerator=args.accelerator,
            devices=args.devices,
            precision=args.precision,
            max_steps=1,
            max_epochs=args.epochs,
            limit_train_batches=1,
            limit_val_batches=0,
            num_sanity_val_steps=0,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            default_root_dir=str(Path(args.output).parent / "tmp" / model_name),
        )
        trainer.fit(model, datamodule=dm)
        result["status"] = "ok"
        result["stage"] = "done"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    return result


def get_args_parser():
    parser = argparse.ArgumentParser(description="Smoke-test model construction and one training step for supported STF models.")
    parser.add_argument("--models", nargs="+", default=SUPPORTED_MODELS, choices=SUPPORTED_MODELS)
    parser.add_argument("--bands", default=4, type=int)
    parser.add_argument("--size", default=512, type=int, help="Patch size for dummy tensors. Use 512 for the real training shape.")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--accelerator", default="cpu", choices=["cpu", "gpu", "auto"])
    parser.add_argument("--devices", default="1")
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--output", default="log/smoke_test_models.json")
    return parser


def main(args):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = [run_one_model(model_name, args) for model_name in args.models]
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for item in results:
        suffix = "" if item["status"] == "ok" else f" at {item['stage']}: {item['error']}"
        print(f"{item['model']}: {item['status']}{suffix}")
    failed = [item for item in results if item["status"] != "ok"]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main(get_args_parser().parse_args())
