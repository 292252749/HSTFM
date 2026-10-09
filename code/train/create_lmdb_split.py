import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LibSTFv1.dataset.SlideDataset import SentinelLMDBDataModule


def get_args_parser():
    parser = argparse.ArgumentParser(description="Create a reusable LMDB train/val/test split JSON without training.")
    parser.add_argument("--lmdb_path", required=True)
    parser.add_argument("--output", required=True, help="Path to split_indices.json")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--split_seed", default=None, type=int)
    parser.add_argument("--sample_num", default=None, type=int)
    parser.add_argument("--train_sample_num", default=None, type=int)
    parser.add_argument("--val_sample_num", default=None, type=int)
    parser.add_argument("--test_sample_num", default=None, type=int)
    parser.add_argument("--train_ratio", default=0.9, type=float)
    parser.add_argument("--val_ratio", default=0.05, type=float)
    parser.add_argument("--patch_size", default=512, type=int)
    parser.add_argument("--sample_random_split", action="store_true")
    parser.add_argument("--num_workers", default=0, type=int)
    return parser


def main(args):
    output = Path(args.output)
    dm = SentinelLMDBDataModule(
        args.lmdb_path,
        batch_size=1,
        num_workers=args.num_workers,
        sample_num=args.sample_num,
        train_sample_num=args.train_sample_num,
        val_sample_num=args.val_sample_num,
        test_sample_num=args.test_sample_num,
        random_seed=args.seed,
        split_seed=args.split_seed,
        patch_size=args.patch_size,
        mask=False,
        split=(args.train_ratio, args.val_ratio, 1.0 - args.train_ratio - args.val_ratio),
        split_by_group=not args.sample_random_split,
        save_split_file=output,
    )
    dm.setup("fit")
    data = json.loads(output.read_text(encoding="utf-8"))
    print(
        f"Split written: {output} "
        f"train={len(data['train_indices'])} val={len(data['val_indices'])} test={len(data['test_indices'])} "
        f"mode={data.get('split_mode')}"
    )


if __name__ == "__main__":
    main(get_args_parser().parse_args())
