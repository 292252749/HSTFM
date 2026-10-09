#!/usr/bin/env python3
"""Build internal_2024 with the validated v3 local workflow.

This script intentionally does not reimplement S2/S3 reading or LMDB building.
It runs v3 in an isolated v4 output directory so the existing internal logic is
preserved while outputs are reproducible under s23_pipeline_v4_out.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as V4  # noqa: E402


def run(cmd, env, cwd, dry_run=False):
    text = " ".join(str(x) for x in cmd)
    print(f"[v4 internal] {text}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description="Build v4 internal subset from local downloaded files.")
    parser.add_argument("--subset", default="internal_2024", choices=["internal_2024"], help="Internal subset to build.")
    parser.add_argument("--target-pairs", type=int, default=None, help="Pair count for v3 05_balance_pair_index.py.")
    parser.add_argument("--skip-balance", action="store_true", help="Skip pair_index balancing before LMDB build.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    subset = V4.SUBSETS[args.subset]
    out_root = V4.v3_out_for_subset(args.subset)
    out_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "S23_DATASET_ROLE": "internal",
        "S23_PAIR_INPUT_MODE": "directory",
        # The vendored engine still reads these legacy variable names, but both
        # now point inside v4. No external v3 checkout is required.
        "S23_V3_ROOT": str(V4.ENGINE_ROOT),
        "S23_V3_OUT": str(out_root),
    })
    env.setdefault("S23_BALANCE_TARGET_PAIRS", str(args.target_pairs or subset["target_pairs"]))

    py = sys.executable
    engine = V4.ENGINE_ROOT
    steps = [
        [py, "03_pair_local_s23.py"],
    ]
    if not args.skip_balance:
        steps.append([py, "05_balance_pair_index.py"])
    steps.extend([
        [py, "04_build_lmdb_from_pairs.py"],
        [py, "09_build_internal_domain.py"],
        [py, "10_build_split_index.py"],
    ])
    for cmd in steps:
        run(cmd, env, engine, dry_run=args.dry_run)

    print({
        "subset": args.subset,
        "v3_out": str(out_root),
        "lmdb": str(V4.subset_lmdb_dir(args.subset, role="internal")),
        "sample_index": str(V4.subset_sample_index(args.subset, role="internal")),
    })


if __name__ == "__main__":
    main()
