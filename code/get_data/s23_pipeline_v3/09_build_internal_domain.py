#!/usr/bin/env python3
"""Build observed internal-domain metadata from final generated samples.

This script should be run after 04_build_lmdb_from_pairs.py. The authoritative
internal domain is the final sample_index.csv, because it records only samples
that actually entered the LMDB. pair_index.json is retained only as auxiliary
provenance and as a fallback when sample_index.csv is unavailable.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from collections import Counter
from datetime import datetime
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


def bounds_union(bounds_list):
    bounds = [b for b in bounds_list if b]
    if not bounds:
        return []
    return [
        min(b[0] for b in bounds),
        min(b[1] for b in bounds),
        max(b[2] for b in bounds),
        max(b[3] for b in bounds),
    ]


def parse_bounds(text):
    if not text:
        return []
    try:
        data = json.loads(text)
    except Exception:
        return []
    if isinstance(data, list) and len(data) == 4:
        try:
            return [float(v) for v in data]
        except Exception:
            return []
    return []


def read_sample_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fd:
        return list(csv.DictReader(fd))


def sample_pair_id(row):
    return row.get("pair_id") or row.get("source_sample_uid", "").split(":", 1)[0]


def count_dates(rows, key):
    years = Counter()
    months = Counter()
    for row in rows:
        date = row.get(key, "")
        if len(date) >= 8:
            years[date[:4]] += 1
            months[date[:6]] += 1
    return years, months


def pair_bounds(pair):
    out = []
    for side in ("t1", "t2"):
        bounds = pair.get(side, {}).get("s2", {}).get("bounds_wgs84")
        if bounds:
            out.append(bounds)
    return out


def build_from_samples(sample_rows, pairs):
    sample_mgrs = Counter(row.get("mgrs_tile", "") for row in sample_rows if row.get("mgrs_tile"))
    sample_pairs = Counter(sample_pair_id(row) for row in sample_rows if sample_pair_id(row))
    pair_mgrs_from_samples = Counter()
    pair_seen = set()
    for row in sample_rows:
        pair_id = sample_pair_id(row)
        if not pair_id or pair_id in pair_seen:
            continue
        pair_seen.add(pair_id)
        if row.get("mgrs_tile"):
            pair_mgrs_from_samples[row["mgrs_tile"]] += 1

    pair_mgrs_from_pair_index = Counter(pair.get("mgrs_tile", "") for pair in pairs if pair.get("mgrs_tile"))
    sample_bounds = [parse_bounds(row.get("bounds_wgs84_json", "")) for row in sample_rows]
    pair_index_bounds = [bounds for pair in pairs for bounds in pair_bounds(pair)]
    t1_years, t1_months = count_dates(sample_rows, "t1_date")
    t2_years, t2_months = count_dates(sample_rows, "t2_date")

    sample_mgrs_set = set(sample_mgrs)
    pair_mgrs_set = set(pair_mgrs_from_pair_index)
    return {
        "domain_source": "sample_index_csv",
        "sample_count": len(sample_rows),
        "productive_pair_count": len(sample_pairs),
        "pair_index_pair_count": len(pairs),
        "mgrs_count": len(sample_mgrs),
        "observed_mgrs_tiles": sorted(sample_mgrs),
        "observed_mgrs_sample_counts": dict(sample_mgrs.most_common()),
        "observed_mgrs_pair_counts": dict(pair_mgrs_from_samples.most_common()),
        "pair_index_mgrs_count": len(pair_mgrs_from_pair_index),
        "pair_index_mgrs_pair_counts": dict(pair_mgrs_from_pair_index.most_common()),
        "sample_mgrs_missing_from_pair_index": sorted(sample_mgrs_set - pair_mgrs_set),
        "pair_index_mgrs_without_samples": sorted(pair_mgrs_set - sample_mgrs_set),
        "observed_sample_bbox_wgs84": bounds_union(sample_bounds),
        "observed_pair_bbox_wgs84": bounds_union(pair_index_bounds),
        "t1_year_counts_from_samples": dict(sorted(t1_years.items())),
        "t2_year_counts_from_samples": dict(sorted(t2_years.items())),
        "t1_month_counts_from_samples": dict(sorted(t1_months.items())),
        "t2_month_counts_from_samples": dict(sorted(t2_months.items())),
        # Backward-compatible aliases. They are intentionally t1-based so the
        # counts sum to sample_count, unlike older versions that counted t1+t2.
        "year_counts_from_samples": dict(sorted(t1_years.items())),
        "month_counts_from_samples": dict(sorted(t1_months.items())),
    }


def build_from_pairs_only(pairs):
    pair_mgrs = Counter(pair.get("mgrs_tile", "") for pair in pairs if pair.get("mgrs_tile"))
    bounds = [bounds for pair in pairs for bounds in pair_bounds(pair)]
    return {
        "domain_source": "pair_index_json_fallback",
        "sample_count": 0,
        "productive_pair_count": 0,
        "pair_index_pair_count": len(pairs),
        "mgrs_count": len(pair_mgrs),
        "observed_mgrs_tiles": sorted(pair_mgrs),
        "observed_mgrs_sample_counts": {},
        "observed_mgrs_pair_counts": dict(pair_mgrs.most_common()),
        "pair_index_mgrs_count": len(pair_mgrs),
        "pair_index_mgrs_pair_counts": dict(pair_mgrs.most_common()),
        "sample_mgrs_missing_from_pair_index": [],
        "pair_index_mgrs_without_samples": sorted(pair_mgrs),
        "observed_sample_bbox_wgs84": [],
        "observed_pair_bbox_wgs84": bounds_union(bounds),
        "t1_year_counts_from_samples": {},
        "t2_year_counts_from_samples": {},
        "t1_month_counts_from_samples": {},
        "t2_month_counts_from_samples": {},
        "year_counts_from_samples": {},
        "month_counts_from_samples": {},
    }


def main():
    sample_rows = read_sample_rows(cfg.SAMPLE_INDEX_CSV)
    pair_index = C.read_json(cfg.PAIR_INDEX_JSON, {})
    pairs = pair_index.get("pairs", []) if isinstance(pair_index, dict) else []

    if sample_rows:
        observed = build_from_samples(sample_rows, pairs)
    else:
        observed = build_from_pairs_only(pairs)

    out = {
        "schema": "s23_internal_domain_v3",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_role": getattr(cfg, "DATASET_ROLE", "internal"),
        "dataset_name": cfg.DATASET_NAME,
        "source_year": getattr(cfg, "SOURCE_YEAR", "2024"),
        "approx_bbox_wgs84": getattr(cfg, "SOURCE_APPROX_BBOX_WGS84", []),
        "approx_bbox_note": "Configuration/provenance only. Use observed_sample_bbox_wgs84 and observed_mgrs_tiles for final sample-domain splits.",
        "sample_index_csv": str(cfg.SAMPLE_INDEX_CSV),
        "pair_index_json": str(cfg.PAIR_INDEX_JSON),
        **observed,
    }
    C.write_json(cfg.INTERNAL_DOMAIN_JSON, out)
    C.progress(f"09 internal domain written: {cfg.INTERNAL_DOMAIN_JSON}")
    print({
        "domain_source": out["domain_source"],
        "sample_count": out["sample_count"],
        "productive_pair_count": out["productive_pair_count"],
        "mgrs": len(out["observed_mgrs_tiles"]),
    })


if __name__ == "__main__":
    main()
