#!/usr/bin/env python3
"""Benchmark orchestration and sample-level split reporting for v4.

The benchmark side is intentionally local-first. This script does not query CDSE
by default. It inspects already downloaded files and already built sample_index
files, then writes reports/splits that tell us what still needs online
supplementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as V4  # noqa: E402

S2_RE = re.compile(r"_MSIL2A_(\d{8})T.*_T([0-9]{2}[A-Z]{3})_")
S3_RE = re.compile(r"OL_1_EFR____(\d{8})T")


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def parse_product(path: Path):
    name = path.name
    if name.lower().endswith(".zip"):
        name = name[:-4]
    m = S2_RE.search(name)
    if m:
        date, mgrs = m.groups()
        return {"sensor": "s2", "date": date, "year": date[:4], "month": date[4:6], "mgrs_tile": mgrs, "name": name, "path": str(path)}
    m = S3_RE.search(name)
    if m:
        date = m.group(1)
        return {"sensor": "s3", "date": date, "year": date[:4], "month": date[4:6], "mgrs_tile": "", "name": name, "path": str(path)}
    return None


def default_download_roots():
    roots = []
    # v4 is self-contained. Put/reuse downloads under these folders, or pass
    # additional roots with --paths.
    roots.append(V4.OUT_ROOT / "downloads")
    roots.append(V4.OUT_ROOT)
    return roots


def scan_downloads(paths):
    records = []
    seen = set()
    for root in paths:
        root = Path(root)
        if not root.exists():
            continue
        for path in root.rglob("*.zip"):
            key = str(path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            rec = parse_product(path)
            if rec:
                records.append(rec)
    return records


def load_internal_mgrs(required=False):
    for path in V4.internal_domain_candidates():
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        mgrs = set(data.get("observed_mgrs_tiles", []) or [])
        if not mgrs and isinstance(data.get("observed_mgrs_pair_counts"), dict):
            mgrs = set(data["observed_mgrs_pair_counts"].keys())
        if mgrs:
            return mgrs, path, "internal_domain_json"
    for path in V4.internal_sample_index_candidates():
        if not path.exists():
            continue
        mgrs = set()
        with path.open("r", encoding="utf-8", newline="") as fd:
            for row in csv.DictReader(fd):
                tile = row.get("mgrs_tile")
                if tile:
                    mgrs.add(tile)
        if mgrs:
            return mgrs, path, "internal_sample_index"
    if required:
        checked = [str(p) for p in V4.internal_domain_candidates() + V4.internal_sample_index_candidates()]
        raise SystemExit("Cannot classify benchmark subsets without internal domain. Checked: " + json.dumps(checked, indent=2))
    return set(), None, "missing"


def subset_for_record(rec, internal_mgrs=None):
    year = rec.get("year")
    mgrs = rec.get("mgrs_tile")
    domain = "internal" if internal_mgrs and mgrs in internal_mgrs else "external"
    if domain == "internal" and year == V4.SOURCE_YEAR:
        return "internal_2024"
    if domain == "internal":
        return "internal_non2024"
    if year == V4.SOURCE_YEAR:
        return "external_2024"
    return "external_non2024"


def local_report(args):
    paths = [Path(p) for p in args.paths] if args.paths else default_download_roots()
    internal_mgrs, internal_source, internal_source_type = load_internal_mgrs(required=not args.allow_missing_internal_domain)
    records = scan_downloads(paths)
    for rec in records:
        rec["v4_subset"] = subset_for_record(rec, internal_mgrs) if rec["sensor"] == "s2" else "s3_unassigned"
    by_sensor = Counter(r["sensor"] for r in records)
    by_year = Counter(r["year"] for r in records)
    by_subset = Counter(r.get("v4_subset", "unknown") for r in records if r["sensor"] == "s2")
    by_s2_mgrs = Counter(r["mgrs_tile"] for r in records if r["sensor"] == "s2")
    s3_dates = {r["date"] for r in records if r["sensor"] == "s3"}
    s2_same_day_s3 = [r for r in records if r["sensor"] == "s2" and r["date"] in s3_dates]

    out = {
        "schema": "s23_v4_local_download_report",
        "roots": [str(p) for p in paths],
        "products": len(records),
        "by_sensor": dict(by_sensor),
        "by_year": dict(sorted(by_year.items())),
        "by_subset_s2": dict(sorted(by_subset.items())),
        "internal_domain_source": str(internal_source) if internal_source else "",
        "internal_domain_source_type": internal_source_type,
        "internal_mgrs_count": len(internal_mgrs),
        "s2_mgrs": len(by_s2_mgrs),
        "s2_dates_with_same_day_s3_by_name": len(s2_same_day_s3),
        "top_s2_mgrs": by_s2_mgrs.most_common(50),
        "records": records if args.include_records else [],
        "note": "This is a filename-level local inventory. Spatial S3 coverage and patch validity are still verified by v3 03/04.",
    }
    out_path = V4.OUT_ROOT / "benchmark_local_download_report.json"
    write_json(out_path, out)
    print({k: out[k] for k in ("products", "by_sensor", "by_year", "by_subset_s2", "internal_mgrs_count", "s2_mgrs", "s2_dates_with_same_day_s3_by_name")})
    print(f"wrote {out_path}")


def interval_bin(days):
    try:
        value = int(float(days))
    except Exception:
        return "missing"
    for lo, hi in V4.INTERVAL_BINS:
        if lo <= value <= hi:
            return f"{lo:03d}_{hi:03d}d"
    return "other"


def season(month):
    if month in {"12", "01", "02"}:
        return "winter"
    if month in {"03", "04", "05"}:
        return "spring"
    if month in {"06", "07", "08"}:
        return "summer"
    if month in {"09", "10", "11"}:
        return "autumn"
    return "missing"


def sample_subset(row, internal_mgrs):
    role = row.get("dataset_role") or row.get("dataset_group") or ""
    year = (row.get("t1_date") or "")[:4]
    mgrs = row.get("mgrs_tile", "")
    is_internal = role == "internal" or (bool(internal_mgrs) and mgrs in internal_mgrs)
    if is_internal and year == V4.SOURCE_YEAR:
        return "internal_2024"
    if is_internal:
        return "internal_non2024"
    if year == V4.SOURCE_YEAR:
        return "external_2024"
    return "external_non2024"


def sample_index_paths():
    paths = []
    # Existing v4 subset outputs.
    for subset in V4.SUBSETS:
        for role in ("internal", "benchmark"):
            p = V4.subset_sample_index(subset, role=role)
            if p.exists():
                paths.append((subset, p))
    # De-duplicate while preserving order.
    seen = set(); out = []
    for subset, p in paths:
        key = str(p.resolve()).lower()
        if key not in seen:
            seen.add(key); out.append((subset, p))
    return out


def iter_samples(paths, internal_mgrs):
    for subset_hint, path in paths:
        with path.open("r", encoding="utf-8", newline="") as fd:
            for row in csv.DictReader(fd):
                subset = subset_hint if subset_hint != "auto" else sample_subset(row, internal_mgrs)
                row["v4_subset"] = subset
                row["v4_source_sample_index"] = str(path)
                yield row


def add_group(groups, name, row):
    key = row.get("lmdb_key")
    if key:
        groups[name].append(key)


def split_report(args):
    paths = [(hint, Path(p)) for hint, p in args.sample_index] if args.sample_index else sample_index_paths()
    if not paths:
        raise SystemExit("No sample_index.csv files found. Build at least one subset first.")
    internal_mgrs, internal_source, internal_source_type = load_internal_mgrs(required=not args.allow_missing_internal_domain)

    counts = defaultdict(Counter)
    groups = defaultdict(list)
    total = 0
    for row in iter_samples(paths, internal_mgrs):
        total += 1
        subset = row["v4_subset"]
        month = (row.get("t1_date") or "")[4:6] or "missing"
        lc = row.get("landcover_major") or "missing"
        ibin = interval_bin(row.get("interval_days"))
        seas = season(month)
        counts[subset]["samples"] += 1
        counts[f"{subset}:landcover"][lc] += 1
        counts[f"{subset}:interval"][ibin] += 1
        counts[f"{subset}:month"][month] += 1
        counts[f"{subset}:season"][seas] += 1
        add_group(groups, f"{subset}/all", row)
        add_group(groups, f"{subset}/by_landcover/{lc}", row)
        add_group(groups, f"{subset}/by_interval/{ibin}", row)
        add_group(groups, f"{subset}/by_month/{month}", row)
        add_group(groups, f"{subset}/by_season/{seas}", row)

    report = {
        "schema": "s23_v4_sample_split_report",
        "sample_indexes": [str(p) for _, p in paths],
        "internal_domain_source": str(internal_source) if internal_source else "",
        "internal_domain_source_type": internal_source_type,
        "internal_mgrs_count": len(internal_mgrs),
        "total_samples": total,
        "counts": {k: dict(v) for k, v in sorted(counts.items())},
        "targets": V4.SUBSETS,
        "gap_summary": gap_summary(counts),
    }
    out_dir = V4.OUT_ROOT / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(V4.OUT_ROOT / "sample_split_report.json", report)

    if args.write_splits:
        for name, keys in groups.items():
            data = {"schema": "s23_v4_split_keys", "name": name, "count": len(keys), "keys": keys}
            write_json(out_dir / f"{name}.json", data)
    print({"total_samples": total, "subsets": {k: dict(v) for k, v in counts.items() if ":" not in k}})
    print(f"wrote {V4.OUT_ROOT / 'sample_split_report.json'}")


def gap_summary(counts):
    gaps = {}
    for subset, spec in V4.SUBSETS.items():
        have = counts.get(subset, Counter()).get("samples", 0)
        target = int(spec.get("target_samples", 0))
        subset_gaps = {"target_samples": target, "current_samples": have, "missing_samples": max(0, target - have)}
        if have:
            lc_missing = {lc: max(0, V4.MIN_SAMPLES_PER_MAIN_LANDCOVER - counts.get(f"{subset}:landcover", Counter()).get(lc, 0)) for lc in V4.LANDCOVER_MAIN}
            int_missing = {f"{lo:03d}_{hi:03d}d": max(0, V4.MIN_SAMPLES_PER_INTERVAL_BIN - counts.get(f"{subset}:interval", Counter()).get(f"{lo:03d}_{hi:03d}d", 0)) for lo, hi in V4.INTERVAL_BINS}
            month_missing = {m: max(0, V4.MIN_SAMPLES_PER_MONTH - counts.get(f"{subset}:month", Counter()).get(m, 0)) for m in V4.MONTHS}
            subset_gaps.update({"landcover_missing": lc_missing, "interval_missing": int_missing, "month_missing": month_missing})
        gaps[subset] = subset_gaps
    return gaps


def main():
    parser = argparse.ArgumentParser(description="v4 benchmark local-first reporting and split generation")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_local = sub.add_parser("local-report", help="Inventory already downloaded local S2/S3 zip files; no CDSE query.")
    p_local.add_argument("--paths", nargs="*", default=[], help="Additional/local roots to scan. Defaults include v4 downloads and v4 outputs.")
    p_local.add_argument("--include-records", action="store_true", help="Store every product record in the JSON report.")
    p_local.add_argument("--allow-missing-internal-domain", action="store_true", help="Allow reporting without internal domain; internal/external classification may be wrong.")

    p_split = sub.add_parser("split-report", help="Build sample-level reports/splits from existing sample_index.csv files.")
    p_split.add_argument("--sample-index", nargs=2, action="append", metavar=("SUBSET", "CSV"), default=[], help="Override sample indexes as pairs: subset path.csv")
    p_split.add_argument("--write-splits", action="store_true", help="Write split key JSON files under v4_out/splits.")
    p_split.add_argument("--allow-missing-internal-domain", action="store_true", help="Allow split reporting without internal domain; internal/external classification may be wrong.")

    args = parser.parse_args()
    if args.cmd == "local-report":
        local_report(args)
    elif args.cmd == "split-report":
        split_report(args)


if __name__ == "__main__":
    main()
