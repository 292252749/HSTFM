import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import json

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from train.run import build_datamodule, build_model, get_args_parser as get_train_args_parser, parse_devices


def get_args_parser():
    parser = get_train_args_parser()
    parser.description = "Evaluate/test STF checkpoints with the same data normalization as training."
    parser.set_defaults(test_after_fit=False, resume=False)
    parser.add_argument("--output_dir", default=None, help="Evaluation log directory. Defaults beside checkpoint.")
    parser.add_argument("--allow_partial_load", action="store_true", help="Allow missing/unexpected checkpoint keys. Formal evaluation should not use this.")
    return parser


def load_checkpoint_weights(model, ckpt_path, strict=True):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        print(f"Warning: missing checkpoint keys: {len(missing)}")
        for key in missing[:20]:
            print(f"  missing: {key}")
    if unexpected:
        print(f"Warning: unexpected checkpoint keys: {len(unexpected)}")
        for key in unexpected[:20]:
            print(f"  unexpected: {key}")
    if (missing or unexpected) and not strict:
        print("Partial checkpoint loading was explicitly allowed; do not use this run for formal metrics unless justified.")
    return checkpoint


def main(args):
    if not args.ckpt_path:
        raise ValueError("--ckpt_path is required for evaluation")
    dm, cfg = build_datamodule(args)
    dm.setup("test")
    model = build_model(args, cfg)
    checkpoint = load_checkpoint_weights(model, args.ckpt_path, strict=not args.allow_partial_load)

    ckpt_path = Path(args.ckpt_path)
    output_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent.parent / "evaluation" / ckpt_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_config.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "data_source": args.data_source,
                "dataset": args.dataset,
                "lmdb_path": args.lmdb_path,
                "data_root": args.data_root,
                "mask": args.mask,
                "eval_mask_prob": args.eval_mask_prob,
                "random_eval_mask": args.random_eval_mask,
                "seed": args.seed,
                "sample_num": args.sample_num,
                "train_sample_num": args.train_sample_num,
                "val_sample_num": args.val_sample_num,
                "test_sample_num": args.test_sample_num,
                "patch_size": args.patch_size,
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "sample_random_split": args.sample_random_split,
                "split_file": args.split_file,
                "split_seed": args.split_seed if args.split_seed is not None else args.seed,
                "experiment_name": args.experiment_name,
                "ergas_ratio": args.ergas_ratio if args.ergas_ratio is not None else (8.0 if args.data_source == "lmdb" else 16.0),
                "checkpoint": str(ckpt_path),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "global_step": checkpoint.get("global_step"),
                "strict_load": not args.allow_partial_load,
                "native_split_note": cfg.get("split_note"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger = TensorBoardLogger(save_dir=str(output_dir), name="tb")
    csv_logger = CSVLogger(save_dir=str(output_dir), name="csv")
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=parse_devices(args.devices),
        precision=args.precision,
        logger=[logger, csv_logger],
        default_root_dir=str(output_dir),
        enable_checkpointing=False,
    )
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main(get_args_parser().parse_args())
