import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "train" / "run.py"
SPLIT_SCRIPT = PROJECT_ROOT / "train" / "create_lmdb_split.py"
SUPPORTED_MODELS = ["HSTFM", "ECPW_STFN", "EDCSTFN", "FSDFormer", "GAN_STFM", "MLFF_GAN", "SwinSTFM", "STFDiff", "DSDF"]


def get_args_parser():
    parser = argparse.ArgumentParser(description="Launch multiple one-GPU STF training jobs with a shared LMDB split.")
    parser.add_argument("--lmdb_path", required=True)
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--models", nargs="+", default=SUPPORTED_MODELS, choices=SUPPORTED_MODELS)
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3", "4", "5", "6", "7"], help="GPU ids used round-robin or as concurrent slots.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--log_root", default="log")
    parser.add_argument("--dataset", default="Sentinel")
    parser.add_argument("--bands", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--split_seed", default=None, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--sample_num", default=None, type=int)
    parser.add_argument("--train_sample_num", default=None, type=int)
    parser.add_argument("--val_sample_num", default=None, type=int)
    parser.add_argument("--test_sample_num", default=None, type=int)
    parser.add_argument("--train_ratio", default=0.9, type=float)
    parser.add_argument("--val_ratio", default=0.05, type=float)
    parser.add_argument("--patch_size", default=512, type=int)
    parser.add_argument("--split_file", default=None)
    parser.add_argument("--sample_random_split", action="store_true")
    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--eval_mask_prob", default=0.0, type=float)
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--extra_args", nargs=argparse.REMAINDER, help="Arguments appended to every train/run.py command after --.")
    return parser


def add_optional(cmd, name, value):
    if value is not None:
        cmd.extend([name, str(value)])


def build_split(args, split_file):
    if split_file.exists():
        print(f"Using existing split: {split_file}")
        return
    cmd = [args.python, str(SPLIT_SCRIPT), "--lmdb_path", args.lmdb_path, "--output", str(split_file),
           "--seed", str(args.seed), "--train_ratio", str(args.train_ratio), "--val_ratio", str(args.val_ratio),
           "--patch_size", str(args.patch_size)]
    add_optional(cmd, "--split_seed", args.split_seed)
    add_optional(cmd, "--sample_num", args.sample_num)
    add_optional(cmd, "--train_sample_num", args.train_sample_num)
    add_optional(cmd, "--val_sample_num", args.val_sample_num)
    add_optional(cmd, "--test_sample_num", args.test_sample_num)
    if args.sample_random_split:
        cmd.append("--sample_random_split")
    print("Creating split:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def train_command(args, model, split_file):
    cmd = [args.python, str(TRAIN_SCRIPT), "--data_source", "lmdb", "--dataset", args.dataset,
           "--lmdb_path", args.lmdb_path, "--model", model, "--bands", str(args.bands),
           "--experiment_name", args.experiment_name, "--split_file", str(split_file),
           "--seed", str(args.seed), "--epochs", str(args.epochs), "--batch_size", str(args.batch_size),
           "--num_workers", str(args.num_workers), "--devices", "1", "--accelerator", "gpu",
           "--precision", args.precision, "--log_root", args.log_root, "--eval_mask_prob", str(args.eval_mask_prob)]
    if args.mask:
        cmd.append("--mask")
    if args.extra_args:
        extra = args.extra_args[1:] if args.extra_args and args.extra_args[0] == "--" else args.extra_args
        cmd.extend(extra)
    return cmd


def main(args):
    split_file = Path(args.split_file) if args.split_file else Path(args.log_root) / args.experiment_name / "split_indices.json"
    build_split(args, split_file)
    running = []
    pending = list(args.models)
    free_gpus = list(args.gpus)
    while pending or running:
        while pending and free_gpus:
            gpu = free_gpus.pop(0)
            model = pending.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            cmd = train_command(args, model, split_file)
            log_dir = Path(args.log_root) / args.experiment_name / "launcher_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{model}_seed{args.seed}_gpu{gpu}.log"
            print(f"Launching {model} on GPU {gpu}: {' '.join(cmd)}")
            log_f = log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=log_f, stderr=subprocess.STDOUT)
            running.append({"model": model, "gpu": gpu, "proc": proc, "log": log_f, "log_path": log_path})
        time.sleep(10)
        still_running = []
        for item in running:
            code = item["proc"].poll()
            if code is None:
                still_running.append(item)
                continue
            item["log"].close()
            status = "ok" if code == 0 else f"failed({code})"
            print(f"Finished {item['model']} on GPU {item['gpu']}: {status}; log={item['log_path']}")
            free_gpus.append(item["gpu"])
            if code != 0:
                raise SystemExit(code)
        running = still_running


if __name__ == "__main__":
    main(get_args_parser().parse_args())
