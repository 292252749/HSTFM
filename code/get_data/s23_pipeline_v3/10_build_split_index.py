#!/usr/bin/env python3
"""Build internal split index files from sample_index.csv without copying LMDB data.

The split design is cost-aware and paper-oriented:
- train/val/test are split at pair level to avoid patch leakage from the same
  S2/S3 observation pair;
- the default internal split is 90/5/5 because the source-domain dataset is
  large enough for stable validation/testing at lower proportions;
- all data-scale training splits are nested subsets of the training pool;
- the 10k training split is also used as the default ablation split;
- validation/test keys are fixed across all training sizes and ablations.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

try:
    import s23_common as C
    cfg = C.cfg
except ModuleNotFoundError as exc:
    if exc.name != "osgeo":
        raise
    spec = importlib.util.spec_from_file_location("s23_cfg", Path(__file__).with_name("00_config.py"))
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    class _CommonLite:
        @staticmethod
        def read_json(path, default=None):
            path = Path(path)
            if not path.exists():
                return default
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return default

        @staticmethod
        def write_json(path, data):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(path)

        @staticmethod
        def progress(message):
            print(message, flush=True)

    C = _CommonLite()

SEED = int(getattr(cfg, "SPLIT_RANDOM_SEED", 20260706))
TRAIN_FRACTION = float(getattr(cfg, "SPLIT_TRAIN_FRACTION", 0.90))
VAL_FRACTION = float(getattr(cfg, "SPLIT_VAL_FRACTION", 0.05))
TRAIN_SAMPLE_SIZES = [5000, 10000, 30000, 60000]
ABLATION_SAMPLE_SIZE = 10000
MAIN_LANDCOVERS = {"cropland", "forest", "grassland", "bareland", "shrubland"}


def read_rows(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fd:
        return list(csv.DictReader(fd))


def sample_years(row):
    years = set()
    for key in ("t1_date", "t2_date"):
        date = row.get(key, "")
        if len(date) >= 4:
            years.add(date[:4])
    return sorted(years)


def strict_source_year(row, source_year):
    years = sample_years(row)
    return bool(years) and all(year == source_year for year in years)


def interval_bin(days):
    try:
        value = int(float(days))
    except Exception:
        return "missing"
    if value <= 30:
        return "000_030"
    if value <= 60:
        return "031_060"
    if value <= 90:
        return "061_090"
    if value <= 180:
        return "091_180"
    if value <= 365:
        return "181_365"
    return "366_plus"


def start_month(row):
    date = row.get("t1_date", "")
    return date[4:6] if len(date) >= 6 else "missing"


def landcover_group(row):
    lc = row.get("landcover_major") or "missing"
    return lc if lc in MAIN_LANDCOVERS else "other"


def sample_balance_key(row):
    return (landcover_group(row), interval_bin(row.get("interval_days", "")), start_month(row))


def pair_balance_key(items):
    # Use the dominant stratum among patches from the same pair.
    return Counter(sample_balance_key(row) for row in items).most_common(1)[0][0]


def write_split(out_dir, name, rows, criteria):
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = [row.get("lmdb_key") for row in rows if row.get("lmdb_key")]
    data = {
        "schema": "s23_split_index_v3",
        "name": name,
        "criteria": criteria,
        "count": len(keys),
        "keys": keys,
    }
    C.write_json(out_dir / f"{name}.json", data)
    return {"name": name, "count": len(keys), "path": str(out_dir / f"{name}.json"), "criteria": criteria}


def rows_by_pair(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("pair_id", "missing")].append(row)
    return grouped


def pair_level_balanced_split(rows, rng):
    grouped = rows_by_pair(rows)
    by_stratum = defaultdict(list)
    for pair_id, items in grouped.items():
        by_stratum[pair_balance_key(items)].append(pair_id)

    train_pairs, val_pairs, test_pairs = set(), set(), set()
    for pair_ids in by_stratum.values():
        rng.shuffle(pair_ids)
        n = len(pair_ids)
        n_val = int(round(n * VAL_FRACTION))
        if n >= 20:
            n_val = max(1, n_val)
        remaining = n - n_val
        test_fraction = max(0.0, 1.0 - TRAIN_FRACTION - VAL_FRACTION)
        n_test = int(round(n * test_fraction))
        if remaining >= 20:
            n_test = max(1, n_test)
        n_test = min(n_test, max(0, n - n_val))
        val_pairs.update(pair_ids[:n_val])
        test_pairs.update(pair_ids[n_val:n_val + n_test])
        train_pairs.update(pair_ids[n_val + n_test:])

    # If rounding left val/test too small globally, move pairs from train while
    # preserving as much stratum balance as possible.
    target_val = max(1, int(round(len(grouped) * VAL_FRACTION)))
    target_test = max(1, int(round(len(grouped) * max(0.0, 1.0 - TRAIN_FRACTION - VAL_FRACTION))))
    train_list = list(train_pairs)
    rng.shuffle(train_list)
    while len(val_pairs) < target_val and train_list:
        pid = train_list.pop()
        train_pairs.remove(pid)
        val_pairs.add(pid)
    while len(test_pairs) < target_test and train_list:
        pid = train_list.pop()
        train_pairs.remove(pid)
        test_pairs.add(pid)

    return (
        [row for pid in sorted(train_pairs) for row in grouped[pid]],
        [row for pid in sorted(val_pairs) for row in grouped[pid]],
        [row for pid in sorted(test_pairs) for row in grouped[pid]],
        {"pairs": len(grouped), "train_pairs": len(train_pairs), "val_pairs": len(val_pairs), "test_pairs": len(test_pairs), "unit": "pair", "strategy": "balanced_by_dominant_landcover_interval_month"},
    )


def balanced_order(rows, rng):
    pools = defaultdict(list)
    for row in rows:
        pools[sample_balance_key(row)].append(row)
    for items in pools.values():
        rng.shuffle(items)
        items.sort(key=lambda r: (r.get("pair_id", ""), int(r.get("patch_row", 0) or 0), int(r.get("patch_col", 0) or 0), r.get("lmdb_key", "")))
    strata = list(pools)
    rng.shuffle(strata)

    ordered = []
    used_keys = set()
    pair_counts = Counter()
    total = sum(len(items) for items in pools.values())
    cursor = 0
    idle = 0
    while len(ordered) < total and idle < max(1, len(strata)) * 3:
        key = strata[cursor % len(strata)]
        cursor += 1
        pool = pools[key]
        chosen = None
        dynamic_pair_quota = max(1, len(ordered) // max(1, len(rows_by_pair(rows))) + 1)
        while pool:
            row = pool.pop(0)
            lmdb_key = row.get("lmdb_key")
            if lmdb_key in used_keys:
                continue
            if pair_counts[row.get("pair_id", "")] > dynamic_pair_quota and len(ordered) < total * 0.9:
                pool.append(row)
                break
            chosen = row
            break
        if chosen is None:
            idle += 1
            continue
        idle = 0
        ordered.append(chosen)
        used_keys.add(chosen.get("lmdb_key"))
        pair_counts[chosen.get("pair_id", "")] += 1

    if len(ordered) < total:
        rest = [row for row in rows if row.get("lmdb_key") not in used_keys]
        rng.shuffle(rest)
        ordered.extend(rest)
    return ordered


def summarize_rows(rows):
    return {
        "samples": len(rows),
        "pairs": len({row.get("pair_id") for row in rows}),
        "mgrs": len({row.get("mgrs_tile") for row in rows}),
        "landcover": dict(Counter(row.get("landcover_major") or "missing" for row in rows)),
        "interval": dict(Counter(interval_bin(row.get("interval_days", "")) for row in rows)),
        "month": dict(Counter(start_month(row) for row in rows)),
    }


def source_rows_for_internal(rows, internal_mgrs, source_year):
    strict_rows = [row for row in rows if strict_source_year(row, source_year)]
    if not internal_mgrs:
        return strict_rows, "no_internal_domain_json_all_strict_source_year_rows"
    domain_rows = [row for row in strict_rows if row.get("mgrs_tile") in internal_mgrs]
    if len(domain_rows) < len(strict_rows) * 0.95:
        return strict_rows, "fallback_all_strict_source_year_rows_due_to_incomplete_internal_domain"
    return domain_rows, "internal_domain_mgrs"


def build_internal_splits(rows, internal_mgrs, source_year, out_dir):
    source_rows, domain_policy = source_rows_for_internal(rows, internal_mgrs, source_year)
    train, val, test, pair_stats = pair_level_balanced_split(source_rows, random.Random(SEED))
    ordered_train = balanced_order(train, random.Random(SEED + 17))

    written = []
    written.append(write_split(out_dir, "internal_train_2024_full", train, {"role": "internal", "year": source_year, "split": "train", "unit": "pair", "fraction": TRAIN_FRACTION, "domain_policy": domain_policy}))
    written.append(write_split(out_dir, "internal_val_2024", val, {"role": "internal", "year": source_year, "split": "val", "unit": "pair", "fraction": VAL_FRACTION, "domain_policy": domain_policy}))
    written.append(write_split(out_dir, "internal_test_2024", test, {"role": "internal", "year": source_year, "split": "test", "unit": "pair", "fraction": max(0.0, 1.0 - TRAIN_FRACTION - VAL_FRACTION), "domain_policy": domain_policy}))

    scale_summaries = {}
    available_sizes = [size for size in TRAIN_SAMPLE_SIZES if size < len(ordered_train)]
    for size in available_sizes:
        subset = ordered_train[:size]
        scale_summaries[size] = summarize_rows(subset)
        written.append(write_split(out_dir / "train_scales", f"internal_train_2024_{size:06d}", subset, {"role": "internal", "year": source_year, "split": "train", "samples": size, "selection": "nested_balanced_landcover_interval_month", "parent": "internal_train_2024_full", "nested": True}))

    ablation_size = min(ABLATION_SAMPLE_SIZE, len(ordered_train))
    ablation = ordered_train[:ablation_size]
    written.append(write_split(out_dir, f"internal_ablation_2024_{ablation_size:06d}", ablation, {"role": "internal", "year": source_year, "purpose": "ablation", "samples": ablation_size, "selection": "same_as_nested_train_scale", "parent": f"train_scales/internal_train_2024_{ablation_size:06d}.json" if ablation_size in available_sizes else "internal_train_2024_full"}))

    report = {
        "schema": "s23_internal_split_report_v3",
        "source_year": source_year,
        "seed": SEED,
        "split_policy": "pair_level_balanced_90_5_5_with_nested_train_scales",
        "train_fraction": TRAIN_FRACTION,
        "val_fraction": VAL_FRACTION,
        "test_fraction": max(0.0, 1.0 - TRAIN_FRACTION - VAL_FRACTION),
        "domain_policy": domain_policy,
        "pair_level_split": pair_stats,
        "source_rows": summarize_rows(source_rows),
        "train_full": summarize_rows(train),
        "val": summarize_rows(val),
        "test": summarize_rows(test),
        "train_scale_sizes": available_sizes,
        "train_scale_summaries": {str(k): v for k, v in scale_summaries.items()},
        "ablation_size": ablation_size,
        "ablation_is_train_scale": ablation_size in available_sizes,
        "splits": written,
    }
    C.write_json(out_dir / "internal_split_report.json", report)
    return report


def build_group_splits(rows, out_dir):
    written = []
    by_lc = defaultdict(list)
    by_interval = defaultdict(list)
    for row in rows:
        by_lc[row.get("landcover_major", "missing") or "missing"].append(row)
        by_interval[interval_bin(row.get("interval_days", ""))].append(row)
    for key, items in by_lc.items():
        written.append(write_split(out_dir / "by_landcover", f"landcover_{key}", items, {"landcover_major": key}))
    for key, items in by_interval.items():
        written.append(write_split(out_dir / "by_interval", f"interval_{key}", items, {"interval_bin": key}))
    written.append(write_split(out_dir, "all", rows, {"all": True}))
    return written


def main():
    rows = read_rows(cfg.SAMPLE_INDEX_CSV)
    if not rows:
        raise SystemExit(f"No samples found: {cfg.SAMPLE_INDEX_CSV}")
    internal = C.read_json(cfg.INTERNAL_DOMAIN_JSON, {})
    internal_mgrs = set(internal.get("observed_mgrs_tiles", [])) if isinstance(internal, dict) else set()
    source_year = str(getattr(cfg, "SOURCE_YEAR", "2024"))
    out_dir = cfg.PIPELINE_OUT / "splits"

    internal_report = build_internal_splits(rows, internal_mgrs, source_year, out_dir)
    group_splits = build_group_splits(rows, out_dir)
    C.progress(f"10 splits written under {out_dir}")
    print({
        "samples": len(rows),
        "split_dir": str(out_dir),
        "internal_splits": len(internal_report["splits"]),
        "group_splits": len(group_splits),
        "train_full": internal_report["train_full"]["samples"],
        "val": internal_report["val"]["samples"],
        "test": internal_report["test"]["samples"],
        "train_scale_sizes": internal_report["train_scale_sizes"],
    })


if __name__ == "__main__":
    main()
