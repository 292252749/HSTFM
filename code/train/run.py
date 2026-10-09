import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import os
import json

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning import seed_everything
from sorcery import dict_of

from LibSTFv1.base.base_model import BaseModel
from LibSTFv1.dataset.SlideDataset import SentinelLMDBDataModule, plNBUDataset
from LibSTFv1.dataset.name2data import available_datasets, get_dataset_config
from LibSTFv1.model.HSTFM.network import HSTFM
from LibSTFv1.util.misc import check_and_make


SUPPORTED_MODELS = ["HSTFM", "ECPW_STFN", "EDCSTFN", "FSDFormer", "GAN_STFM", "MLFF_GAN", "SwinSTFM", "STFDiff", "DSDF"]


class TrainableHSTFMModel(BaseModel):
    """HSTFM Lightning wrapper kept outside LibSTFv1/model to avoid editing network files."""

    def __init__(self, epochs, bands, rgb_c, dataname):
        super().__init__(epochs=epochs, bands=bands, rgb_c=rgb_c, dataname=dataname)
        self.model = HSTFM(in_channels=bands)
        self.cri_pix = nn.L1Loss()
        self.save_hyperparameters()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.epochs, eta_min=1e-7)
        return [opt], [scheduler]

    def on_load_checkpoint(self, checkpoint):
        # HSTFM builds attention masks lazily and stores them in _mask_cache.
        # They are deterministic temporary tensors, not trainable state.
        state_dict = checkpoint.get("state_dict", {})
        for key in list(state_dict.keys()):
            if "model._mask_cache." in key:
                state_dict.pop(key)

    def forward(self, LR_t1, LR_t2, HR_t1, mask):
        pred = self.model(LR_t1, HR_t1, LR_t2, mask)
        return dict_of(pred)

    def training_step(self, batch, batch_idx):
        out = self.forward(batch["LR_t1"], batch["LR_t2"], batch["HR_t1"], batch["mask"])
        pred = out["pred"]
        loss = self.cri_pix(pred, batch["HR_t2"])
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        self.log_dict({"train/loss": loss, "train/mse": F.mse_loss(pred.detach(), batch["HR_t2"])},
                      prog_bar=True, logger=True, on_step=True, on_epoch=True)

    def on_train_epoch_end(self):
        self.lr_schedulers().step()


class _SwinSTFMCore(nn.Module):
    def __init__(self, bands, img_size=512):
        super().__init__()
        from LibSTFv1.model.SwinSTFM.fem import Down
        from LibSTFv1.model.SwinSTFM.mfm import FineUp
        self.coarse_down = Down(bands, img_size)
        self.fine_down = Down(bands, img_size)
        self.fine_up = FineUp(img_size=img_size, output_bands=bands)

    def forward(self, c0, f0, c1):
        coarse_fea0 = self.coarse_down(c0)
        coarse_fea1 = self.coarse_down(c1)
        fine_fea = self.fine_down(f0)
        diff_fea1 = [coarse_fea1[i] - coarse_fea0[i] for i in range(5)]
        return self.fine_up(coarse_fea0, diff_fea1, fine_fea, coarse_fea1)


class TrainableSwinSTFMModel(BaseModel):
    def __init__(self, epochs, bands, rgb_c, dataname):
        super().__init__(epochs=epochs, bands=bands, rgb_c=rgb_c, dataname=dataname)
        from LibSTFv1.model.SwinSTFM.loss import GeneratorLoss
        self.model = _SwinSTFMCore(bands, 512)
        self.cri_pix = GeneratorLoss()
        self.save_hyperparameters()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.epochs, eta_min=1e-7)
        return [opt], [scheduler]

    def forward(self, LR_t1, LR_t2, HR_t1, HR_t2=None):
        pred = self.model(LR_t1, HR_t1, LR_t2)
        return dict_of(pred)

    def training_step(self, batch, batch_idx):
        out = self.forward(batch["LR_t1"], batch["LR_t2"], batch["HR_t1"])
        pred = out["pred"]
        loss = self.cri_pix(pred, batch["HR_t2"], is_ds=False)
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        self.log_dict({"train/loss": loss, "train/mse": F.mse_loss(pred.detach(), batch["HR_t2"])},
                      prog_bar=True, logger=True, on_step=True, on_epoch=True)

    def on_train_epoch_end(self):
        self.lr_schedulers().step()


def parse_devices(value):
    if value == "auto":
        return "auto"
    if "," in value:
        return [int(v.strip()) for v in value.split(",") if v.strip()]
    return int(value)


def build_datamodule(args):
    if args.data_source == "native":
        cfg = get_dataset_config(args.dataset, data_root=args.data_root)
        return plNBUDataset(cfg, batch_size=args.batch_size, num_workers=args.num_workers), cfg

    if not args.lmdb_path:
        raise ValueError("--lmdb_path is required when --data_source lmdb")
    cfg = {
        "name": args.dataset,
        "band": args.bands,
        "rgb_c": args.rgb_c,
        "slide_size": args.patch_size,
        "max_value": 1.0,
    }
    dm = SentinelLMDBDataModule(
        args.lmdb_path,
        batch_size=args.batch_size,
        sample_num=args.sample_num,
        train_sample_num=args.train_sample_num,
        val_sample_num=args.val_sample_num,
        test_sample_num=args.test_sample_num,
        random_seed=args.seed,
        split_seed=args.split_seed,
        num_workers=args.num_workers,
        patch_size=args.patch_size,
        mask=args.mask,
        split=(args.train_ratio, args.val_ratio, 1.0 - args.train_ratio - args.val_ratio),
        split_by_group=not args.sample_random_split,
        split_file=args.split_file,
        save_split_file=args.save_split_file,
        eval_mask_prob=args.eval_mask_prob,
        deterministic_eval_mask=not args.random_eval_mask,
    )
    return dm, cfg


def build_model(args, cfg):
    bands = int(cfg["band"])
    rgb_c = cfg.get("rgb_c", args.rgb_c)
    dataname = cfg.get("name", args.dataset)
    if args.model == "HSTFM":
        model = TrainableHSTFMModel(args.epochs, bands, rgb_c, dataname)
    elif args.model == "SwinSTFM":
        model = TrainableSwinSTFMModel(args.epochs, bands, rgb_c, dataname)
    elif args.model == "ECPW_STFN":
        from LibSTFv1.model.ECPW_STFN.ECPW_STFN import ECPW_STFNModel
        model = ECPW_STFNModel(args.epochs, bands, rgb_c, dataname)
    elif args.model == "EDCSTFN":
        from LibSTFv1.model.EDCSTFN.EDCSTFN import EDCSTFNModel
        model = EDCSTFNModel(args.epochs, bands, rgb_c, dataname)
    elif args.model == "FSDFormer":
        from LibSTFv1.model.FSDFormer.FSDFormer import FSDFormerModel
        model = FSDFormerModel(args.epochs, bands, rgb_c, dataname)
    elif args.model == "GAN_STFM":
        from LibSTFv1.model.GAN_STFM.GAN_STFM import GANSTFMModel
        model = GANSTFMModel(args.epochs, bands, rgb_c, dataname)
    elif args.model == "MLFF_GAN":
        from LibSTFv1.model.MLFF_GAN.MLFF_GAN import MLFFGANModel
        model = MLFFGANModel(args.epochs, bands, rgb_c, dataname)
    elif args.model == "STFDiff":
        from LibSTFv1.model.STFDiff.STFDiff import STFDiffModel
        model = STFDiffModel(args.epochs, bands, rgb_c, dataname)
    elif args.model == "DSDF":
        from LibSTFv1.model.DSDF.DSDF import DSDFModel
        model = DSDFModel(args.epochs, bands, rgb_c, dataname)
    else:
        raise ValueError(f"unsupported model: {args.model}")
    model.spatial_ratio = float(args.ergas_ratio if args.ergas_ratio is not None else (8.0 if args.data_source == "lmdb" else 16.0))
    return model


def find_latest_ckpt(ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return None
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    return str(max(ckpts, key=lambda p: p.stat().st_mtime)) if ckpts else None


def get_args_parser():
    parser = argparse.ArgumentParser(description="Train STF models on native benchmark datasets or repository LMDB.")
    parser.add_argument("--model", default="HSTFM", choices=SUPPORTED_MODELS)
    parser.add_argument("--experiment_name", default=None, help="Optional stable experiment tag added to the run directory.")
    parser.add_argument("--data_source", default="lmdb", choices=["native", "lmdb"])
    parser.add_argument("--dataset", default="Sentinel", help=f"Native choices: {available_datasets()}; lmdb label defaults to Sentinel")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--lmdb_path", default=None)
    parser.add_argument("--bands", default=4, type=int, help="Band count for lmdb data. Native datasets use their config.")
    parser.add_argument("--rgb_c", default=[1, 2, 0], type=int, nargs="+")
    parser.add_argument("--mask", action="store_true", help="Enable synthetic LR_t2 masking for repository LMDB. Training uses curriculum; evaluation uses --eval_mask_prob.")
    parser.add_argument("--eval_mask_prob", default=0.0, type=float, help="Fixed mask probability for validation/test on LMDB when --mask is set. Use 0 for clean formal evaluation.")
    parser.add_argument("--random_eval_mask", action="store_true", help="Use random masks during LMDB validation/test instead of deterministic per-sample masks.")
    parser.add_argument("--mask_max_prob", default=0.8, type=float)
    parser.add_argument("--mask_start_epoch", default=5, type=int)
    parser.add_argument("--mask_warmup_epochs", default=20, type=int)
    parser.add_argument("--sample_num", default=None, type=int, help="Legacy LMDB total sample budget split by --train_ratio/--val_ratio.")
    parser.add_argument("--train_sample_num", default=None, type=int, help="Explicit LMDB train sample target. Set with --val_sample_num and --test_sample_num.")
    parser.add_argument("--val_sample_num", default=None, type=int, help="Explicit LMDB validation sample target.")
    parser.add_argument("--test_sample_num", default=None, type=int, help="Explicit LMDB test sample target.")
    parser.add_argument("--split_seed", default=None, type=int, help="Seed for LMDB split construction. Defaults to --seed.")
    parser.add_argument("--patch_size", default=512, type=int)
    parser.add_argument("--train_ratio", default=0.9, type=float)
    parser.add_argument("--val_ratio", default=0.05, type=float)
    parser.add_argument("--sample_random_split", action="store_true", help="For LMDB only: split samples randomly instead of keeping metadata groups separate.")
    parser.add_argument("--split_file", default=None, help="For LMDB only: load train/val/test indices from this JSON file.")
    parser.add_argument("--save_split_file", default=None, help="For LMDB only: save train/val/test indices to this JSON file. Training defaults to the run directory.")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--ergas_ratio", default=None, type=float, help="Spatial resolution ratio for ERGAS. Defaults to 8 for LMDB and 16 for native datasets.")
    parser.add_argument("--strategy", default="auto")
    parser.add_argument("--log_root", default="log")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--test_after_fit", action="store_true")
    return parser


def main(args):
    seed_everything(args.seed, workers=True)
    mask_tag = "mask" if args.mask else "nomask"
    if args.experiment_name:
        run_name = f"{args.experiment_name}_{args.model}_{mask_tag}_seed{args.seed}"
    else:
        size_tag = f"n{args.sample_num}" if args.sample_num is not None else "all"
        if args.train_sample_num is not None:
            size_tag = f"tr{args.train_sample_num}_va{args.val_sample_num}_te{args.test_sample_num}"
        run_name = f"{args.dataset}_{args.data_source}_{args.model}_{mask_tag}_{size_tag}_seed{args.seed}"
    save_dir = Path(args.log_root) / args.dataset / args.model / run_name
    check_and_make(str(save_dir))
    if args.data_source == "lmdb" and args.save_split_file is None and args.split_file is None:
        args.save_split_file = str(save_dir / "split_indices.json")

    dm, cfg = build_datamodule(args)
    dm.setup("fit")
    model = build_model(args, cfg)
    train_config = {
        "model": args.model,
        "data_source": args.data_source,
        "dataset": args.dataset,
        "data_root": args.data_root,
        "lmdb_path": args.lmdb_path,
        "bands": int(cfg["band"]),
        "mask": args.mask,
        "eval_mask_prob": args.eval_mask_prob,
        "mask_max_prob": args.mask_max_prob if args.mask else 0.0,
        "sample_num": args.sample_num,
        "train_sample_num": args.train_sample_num,
        "val_sample_num": args.val_sample_num,
        "test_sample_num": args.test_sample_num,
        "patch_size": args.patch_size,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "sample_random_split": args.sample_random_split,
        "split_file": args.split_file,
        "save_split_file": args.save_split_file,
        "ergas_ratio": model.spatial_ratio,
        "seed": args.seed,
        "split_seed": args.split_seed if args.split_seed is not None else args.seed,
        "experiment_name": args.experiment_name,
        "run_name": run_name,
        "save_dir": str(save_dir),
        "native_split_note": cfg.get("split_note"),
    }
    (save_dir / "train_config.json").write_text(json.dumps(train_config, indent=2), encoding="utf-8")

    ckpt_cb = ModelCheckpoint(
        dirpath=str(save_dir / "checkpoints"),
        monitor="val/RMSE_mean",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="epoch={epoch:03d}",
        auto_insert_metric_name=False,
    )
    callbacks = [ckpt_cb]
    if args.data_source == "lmdb" and args.mask:
        callbacks.append(dm.get_curriculum_callback(args.mask_max_prob, args.mask_start_epoch, args.mask_warmup_epochs))

    logger = TensorBoardLogger(save_dir=str(save_dir), name="tb")
    csv_logger = CSVLogger(save_dir=str(save_dir), name="csv")
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=parse_devices(args.devices),
        strategy=args.strategy,
        precision=args.precision,
        logger=[logger, csv_logger],
        default_root_dir=str(save_dir),
        callbacks=callbacks,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
    )

    ckpt_path = args.ckpt_path
    if ckpt_path is None and args.resume:
        ckpt_path = find_latest_ckpt(save_dir / "checkpoints")
    trainer.fit(model, dm, ckpt_path=ckpt_path)
    if args.test_after_fit:
        if hasattr(dm, "set_mask_probability"):
            dm.set_mask_probability(args.eval_mask_prob if args.mask else 0.0)
        trainer.test(model, datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main(get_args_parser().parse_args())
