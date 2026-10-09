#!/usr/bin/env python3
"""
Analyze china/overseas S23 LMDB metadata distributions.

This script reads the JSON/CSV sidecar files produced by 03/04 and writes a
compact set of tables plus a Markdown report for deciding whether the dataset
supports downstream generalization experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any


CHINA_BBOX_WGS84 = (73.5, 18.0, 135.1, 53.6)
DEFAULT_GROUPS = ("china", "overseas")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_root = here.parent / "s23_pipeline_v2_out"
    parser = argparse.ArgumentParser(
        description="Analyze S23 pair/sample sidecars for china and overseas datasets."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help="Root directory containing china/ and overseas/ output folders.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=list(DEFAULT_GROUPS),
        help="Dataset group folders to analyze.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Default: <root>/distribution_analysis",
    )
    parser.add_argument(
        "--max-landcover-rows",
        type=int,
        default=0,
        help="Limit rows used to aggregate landcover_counts_json; 0 means all rows.",
    )
    parser.add_argument(
        "--make-plots",
        action="store_true",
        help="Also write simple PNG plots if matplotlib is installed.",
    )
    return parser.parse_args()


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def safe_int(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def date_parts(value: str) -> tuple[str, str] | tuple[None, None]:
    if not value or len(value) < 6:
        return None, None
    return value[:4], value[:6]


def quantiles(values: list[float]) -> dict[str, Any]:
    values = sorted(v for v in values if v is not None and not math.isnan(v))
    if not values:
        return {"count": 0}

    def q(p: float) -> float:
        if len(values) == 1:
            return values[0]
        pos = (len(values) - 1) * p
        low = math.floor(pos)
        high = math.ceil(pos)
        if low == high:
            return values[low]
        return values[low] * (high - pos) + values[high] * (pos - low)

    return {
        "count": len(values),
        "min": values[0],
        "p05": q(0.05),
        "p25": q(0.25),
        "median": q(0.50),
        "mean": mean(values),
        "p75": q(0.75),
        "p95": q(0.95),
        "max": values[-1],
    }


def interval_bin(days: int | None) -> str:
    if days is None:
        return "missing"
    if days <= 15:
        return "000-015"
    if days <= 30:
        return "016-030"
    if days <= 60:
        return "031-060"
    if days <= 90:
        return "061-090"
    if days <= 180:
        return "091-180"
    if days <= 365:
        return "181-365"
    return "366+"


def bbox_center(raw: str) -> tuple[float, float] | None:
    try:
        bounds = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(bounds, list) or len(bounds) != 4:
        return None
    vals = [safe_float(v) for v in bounds]
    if any(v is None for v in vals):
        return None
    minx, miny, maxx, maxy = vals
    return (minx + maxx) / 2.0, (miny + maxy) / 2.0


def in_china_bbox(lon: float, lat: float) -> bool:
    minx, miny, maxx, maxy = CHINA_BBOX_WGS84
    return minx <= lon <= maxx and miny <= lat <= maxy


def pct(part: int | float, total: int | float) -> float:
    if not total:
        return 0.0
    return float(part) * 100.0 / float(total)


def read_csv_rows(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def flatten_counter(counter: Counter, group: str, name: str) -> list[dict[str, Any]]:
    total = sum(counter.values())
    return [
        {
            "group": group,
            "metric": name,
            "key": key,
            "count": count,
            "percent": pct(count, total),
        }
        for key, count in counter.most_common()
    ]


def summarize_pair_summary(path: Path) -> dict[str, Any]:
    rows = list(read_csv_rows(path) or [])
    intervals: list[float] = []
    clouds: list[float] = []
    overlaps: list[float] = []
    s3_candidates: list[float] = []
    mgrs = Counter()
    year_counter = Counter()
    month_counter = Counter()
    interval_bins = Counter()

    for row in rows:
        mgrs[row.get("mgrs_tile", "")] += 1
        interval = safe_int(row.get("interval_days"))
        if interval is not None:
            intervals.append(float(interval))
            interval_bins[interval_bin(interval)] += 1
        for key in ("t1_cloud", "t2_cloud"):
            v = safe_float(row.get(key))
            if v is not None:
                clouds.append(v)
        for key in ("t1_best_s3_overlap_ratio", "t2_best_s3_overlap_ratio"):
            v = safe_float(row.get(key))
            if v is not None:
                overlaps.append(v)
        for key in ("t1_s3_candidates", "t2_s3_candidates"):
            v = safe_float(row.get(key))
            if v is not None:
                s3_candidates.append(v)
        for key in ("t1_date", "t2_date"):
            year, month = date_parts(row.get(key, ""))
            if year:
                year_counter[year] += 1
            if month:
                month_counter[month] += 1

    return {
        "rows": len(rows),
        "mgrs_unique": len(mgrs),
        "mgrs_top": mgrs.most_common(30),
        "years": dict(sorted(year_counter.items())),
        "months": dict(sorted(month_counter.items())),
        "interval_days": quantiles(intervals),
        "interval_bins": dict(sorted(interval_bins.items())),
        "s2_cloud": quantiles(clouds),
        "s3_overlap_ratio": quantiles(overlaps),
        "s3_candidates": quantiles(s3_candidates),
    }


def summarize_samples(path: Path, max_landcover_rows: int = 0) -> dict[str, Any]:
    counters: dict[str, Counter] = {
        "mgrs": Counter(),
        "utm_zone": Counter(),
        "year": Counter(),
        "month": Counter(),
        "interval_bin": Counter(),
        "landcover_major": Counter(),
        "schema": Counter(),
        "s3_resample_mode": Counter(),
        "china_bbox_flag": Counter(),
    }
    numeric: dict[str, list[float]] = defaultdict(list)
    landcover_pixels = Counter()
    total = 0
    centers = []

    for row in read_csv_rows(path) or []:
        total += 1
        mgrs = row.get("mgrs_tile", "")
        if mgrs:
            counters["mgrs"][mgrs] += 1
            counters["utm_zone"][mgrs[:2]] += 1
        counters["landcover_major"][row.get("landcover_major", "missing") or "missing"] += 1
        counters["schema"][row.get("schema", "missing") or "missing"] += 1
        counters["s3_resample_mode"][row.get("s3_resample_mode", "missing") or "missing"] += 1

        interval = safe_int(row.get("interval_days"))
        counters["interval_bin"][interval_bin(interval)] += 1

        for key in ("t1_date", "t2_date"):
            year, month = date_parts(row.get(key, ""))
            if year:
                counters["year"][year] += 1
            if month:
                counters["month"][month] += 1

        for key in (
            "interval_days",
            "land_fraction",
            "water_fraction",
            "nodata_fraction",
            "s2_change_mean",
            "s3_change_mean",
            "s2_ndvi_t1_mean",
            "s2_ndvi_t2_mean",
            "s3_ndvi_t1_mean",
            "s3_ndvi_t2_mean",
            "t1_s3_overlap_ratio",
            "t2_s3_overlap_ratio",
            "t1_s3_score",
            "t2_s3_score",
        ):
            v = safe_float(row.get(key))
            if v is not None:
                numeric[key].append(v)

        center = bbox_center(row.get("bounds_wgs84_json", ""))
        if center:
            lon, lat = center
            centers.append(center)
            counters["china_bbox_flag"]["inside" if in_china_bbox(lon, lat) else "outside"] += 1

        if max_landcover_rows <= 0 or total <= max_landcover_rows:
            raw_counts = row.get("landcover_counts_json")
            if raw_counts:
                try:
                    counts = json.loads(raw_counts)
                except json.JSONDecodeError:
                    counts = {}
                if isinstance(counts, dict):
                    for key, value in counts.items():
                        n = safe_int(value)
                        if n:
                            landcover_pixels[key] += n

    if centers:
        lons = [v[0] for v in centers]
        lats = [v[1] for v in centers]
        spatial = {
            "sample_centers": len(centers),
            "lon": quantiles(lons),
            "lat": quantiles(lats),
            "bbox_wgs84": [min(lons), min(lats), max(lons), max(lats)],
        }
    else:
        spatial = {"sample_centers": 0}

    return {
        "samples": total,
        "unique_mgrs": len(counters["mgrs"]),
        "unique_utm_zones": len(counters["utm_zone"]),
        "counters": {k: dict(v.most_common()) for k, v in counters.items()},
        "counter_totals": {k: sum(v.values()) for k, v in counters.items()},
        "numeric": {k: quantiles(v) for k, v in sorted(numeric.items())},
        "landcover_pixels": dict(landcover_pixels.most_common()),
        "spatial": spatial,
        "top_mgrs": counters["mgrs"].most_common(50),
    }


def summarize_build_report(path: Path) -> dict[str, Any]:
    data = load_json(path, {})
    pairs = data.get("pairs", []) if isinstance(data, dict) else []
    skipped = Counter()
    status = Counter()
    samples_written = []
    zero_sample_pairs = 0

    for item in pairs:
        status[item.get("status", "missing")] += 1
        written = safe_int(item.get("samples_written")) or 0
        samples_written.append(float(written))
        if written == 0:
            zero_sample_pairs += 1
        for reason, count in (item.get("skipped") or {}).items():
            n = safe_int(count)
            if n:
                skipped[reason] += n

    return {
        "pairs_in_report": len(pairs),
        "status": dict(status.most_common()),
        "zero_sample_pairs": zero_sample_pairs,
        "zero_sample_pair_percent": pct(zero_sample_pairs, len(pairs)),
        "samples_written_per_pair": quantiles(samples_written),
        "skipped": dict(skipped.most_common()),
        "skipped_total": sum(skipped.values()),
    }


def summarize_group(group: str, folder: Path, max_landcover_rows: int) -> dict[str, Any]:
    metadata = load_json(folder / "metadata.json", {})
    pair_analysis = load_json(folder / "pair_analysis.json", {})
    pair_index = load_json(folder / "pair_index.json", {})

    return {
        "group": group,
        "folder": str(folder),
        "metadata": metadata,
        "pair_analysis": pair_analysis,
        "pair_index_counts": pair_index.get("counts", {}) if isinstance(pair_index, dict) else {},
        "pair_summary": summarize_pair_summary(folder / "pair_summary.csv"),
        "sample_index": summarize_samples(folder / "sample_index.csv", max_landcover_rows),
        "build_report": summarize_build_report(folder / "build_report.json"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compact_stats(stats: dict[str, Any]) -> str:
    if not stats or stats.get("count", 0) == 0:
        return "n=0"
    return (
        f"n={stats['count']}, min={stats['min']:.4g}, "
        f"median={stats['median']:.4g}, mean={stats['mean']:.4g}, "
        f"p95={stats['p95']:.4g}, max={stats['max']:.4g}"
    )


def assess_group(summary: dict[str, Any]) -> list[str]:
    group = summary["group"]
    sample = summary["sample_index"]
    pair = summary["pair_summary"]
    build = summary["build_report"]
    notes = []

    n_samples = sample["samples"]
    n_pairs = pair["rows"]
    n_mgrs = sample["unique_mgrs"]
    n_years = len(sample["counters"]["year"])
    zero_pair_pct = build["zero_sample_pair_percent"]
    inside = sample["counters"]["china_bbox_flag"].get("inside", 0)
    outside = sample["counters"]["china_bbox_flag"].get("outside", 0)

    if group == "china":
        if n_samples >= 50000 and n_pairs >= 1000 and n_mgrs >= 500:
            notes.append("China 主集规模充足，可作为主训练/主消融数据。")
        else:
            notes.append("China 主集规模偏低或空间覆盖不足，需要优先检查过滤条件。")
        if outside:
            notes.append(f"China 样本中心有 {outside} 个落在中国框外，应复查 S2 空间筛选。")
        if n_years <= 1:
            notes.append("China 当前主要是单年数据，适合国内泛化和空间泛化，但不足以单独证明多年时间泛化。")
    else:
        if n_samples >= 5000 and n_mgrs >= 50:
            notes.append("Overseas 外推测试集有基本规模，可用于跨区域泛化验证。")
        else:
            notes.append("Overseas 样本规模偏小，适合外部验证，但支撑细分气候/地类实验会偏弱。")
        if inside:
            notes.append(f"Overseas 样本中心有 {inside} 个落在中国框内，应复查 S2 空间筛选。")
        if n_years >= 3:
            notes.append("Overseas 具备多年分布，可用于补充时间外推分析。")

    if zero_pair_pct > 20:
        notes.append(f"零样本 pair 比例较高({zero_pair_pct:.1f}%)，说明 patch 级无效值/地类过滤仍是主要瓶颈。")
    if len(sample["counters"]["landcover_major"]) < 4:
        notes.append("主导地类种类偏少，地类泛化实验支撑不足。")
    return notes


def write_markdown(path: Path, summaries: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# S23 Dataset Distribution Analysis",
        "",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overall",
        "",
        "| group | samples | pairs | candidate samples | skipped samples | unique MGRS | years | zero-sample pairs | inside China bbox | outside China bbox |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for group, s in summaries.items():
        meta_summary = (s.get("metadata") or {}).get("summary", {})
        sample = s["sample_index"]
        pair = s["pair_summary"]
        build = s["build_report"]
        flags = sample["counters"]["china_bbox_flag"]
        lines.append(
            "| {group} | {samples} | {pairs} | {candidate} | {skipped} | {mgrs} | {years} | {zero} ({zero_pct:.1f}%) | {inside} | {outside} |".format(
                group=group,
                samples=sample["samples"],
                pairs=pair["rows"],
                candidate=meta_summary.get("candidate_samples", ""),
                skipped=meta_summary.get("skipped_samples", ""),
                mgrs=sample["unique_mgrs"],
                years=len(sample["counters"]["year"]),
                zero=build["zero_sample_pairs"],
                zero_pct=build["zero_sample_pair_percent"],
                inside=flags.get("inside", 0),
                outside=flags.get("outside", 0),
            )
        )

    for group, s in summaries.items():
        sample = s["sample_index"]
        pair = s["pair_summary"]
        build = s["build_report"]
        pair_analysis = s.get("pair_analysis") or {}
        cloud = pair_analysis.get("cloud", {})
        file_counts = pair_analysis.get("file_counts", {})
        spatial_filter = pair_analysis.get("space", {}).get("dataset_spatial_split", [])

        lines.extend(
            [
                "",
                f"## {group}",
                "",
                "### Assessment",
            ]
        )
        for note in assess_group(s):
            lines.append(f"- {note}")

        lines.extend(
            [
                "",
                "### Pair And File Filtering",
                f"- name_prefilter_s2: {file_counts.get('name_prefilter_s2', '')}",
                f"- s2_after_cloud_filter: {file_counts.get('s2_after_cloud_filter', '')}",
                f"- s2_selected_for_bounds: {file_counts.get('s2_selected_for_bounds', '')}",
                f"- s3_selected_for_bounds: {file_counts.get('s3_selected_for_bounds', '')}",
                f"- cloud threshold: {cloud.get('max_s2_cloud_percent', '')}",
                f"- rejected_by_cloud: {cloud.get('rejected_by_cloud', '')}",
                f"- pair interval days: {compact_stats(pair['interval_days'])}",
                f"- S2 cloud: {compact_stats(pair['s2_cloud'])}",
                f"- S3 overlap ratio: {compact_stats(pair['s3_overlap_ratio'])}",
                f"- spatial split: `{json.dumps(spatial_filter, ensure_ascii=False)}`",
                "",
                "### Sample Distributions",
                f"- sample interval days: {compact_stats(sample['numeric'].get('interval_days', {}))}",
                f"- land fraction: {compact_stats(sample['numeric'].get('land_fraction', {}))}",
                f"- water fraction: {compact_stats(sample['numeric'].get('water_fraction', {}))}",
                f"- nodata fraction: {compact_stats(sample['numeric'].get('nodata_fraction', {}))}",
                f"- S2 change mean: {compact_stats(sample['numeric'].get('s2_change_mean', {}))}",
                f"- S3 change mean: {compact_stats(sample['numeric'].get('s3_change_mean', {}))}",
                f"- S2 NDVI t1: {compact_stats(sample['numeric'].get('s2_ndvi_t1_mean', {}))}",
                f"- S2 NDVI t2: {compact_stats(sample['numeric'].get('s2_ndvi_t2_mean', {}))}",
                f"- S3 NDVI t1: {compact_stats(sample['numeric'].get('s3_ndvi_t1_mean', {}))}",
                f"- S3 NDVI t2: {compact_stats(sample['numeric'].get('s3_ndvi_t2_mean', {}))}",
                f"- spatial bbox of sample centers: `{sample['spatial'].get('bbox_wgs84', '')}`",
                "",
                "### Top Landcover Major",
            ]
        )
        total_samples = sample["samples"]
        for key, count in list(sample["counters"]["landcover_major"].items())[:20]:
            lines.append(f"- {key}: {count} ({pct(count, total_samples):.2f}%)")

        lines.extend(["", "### Top Skip Reasons"])
        for key, count in list(build["skipped"].items())[:20]:
            lines.append(f"- {key}: {count} ({pct(count, build['skipped_total']):.2f}%)")

        lines.extend(["", "### Years"])
        for key, count in sample["counters"]["year"].items():
            lines.append(f"- {key}: {count}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tables(out_dir: Path, summaries: dict[str, dict[str, Any]]) -> None:
    overview = []
    landcover_major = []
    landcover_pixels = []
    months = []
    intervals = []
    skips = []
    top_mgrs = []
    numeric_rows = []

    for group, s in summaries.items():
        sample = s["sample_index"]
        pair = s["pair_summary"]
        build = s["build_report"]
        flags = sample["counters"]["china_bbox_flag"]
        overview.append(
            {
                "group": group,
                "samples": sample["samples"],
                "pairs": pair["rows"],
                "unique_mgrs": sample["unique_mgrs"],
                "unique_utm_zones": sample["unique_utm_zones"],
                "years": ";".join(sample["counters"]["year"].keys()),
                "zero_sample_pairs": build["zero_sample_pairs"],
                "zero_sample_pair_percent": build["zero_sample_pair_percent"],
                "inside_china_bbox": flags.get("inside", 0),
                "outside_china_bbox": flags.get("outside", 0),
            }
        )

        landcover_major.extend(flatten_counter(Counter(sample["counters"]["landcover_major"]), group, "landcover_major"))
        landcover_pixels.extend(flatten_counter(Counter(sample["landcover_pixels"]), group, "landcover_pixels"))
        months.extend(flatten_counter(Counter(sample["counters"]["month"]), group, "sample_month"))
        intervals.extend(flatten_counter(Counter(sample["counters"]["interval_bin"]), group, "sample_interval_bin"))
        skips.extend(flatten_counter(Counter(build["skipped"]), group, "skip_reason"))
        top_mgrs.extend(
            {
                "group": group,
                "mgrs_tile": tile,
                "samples": count,
                "percent": pct(count, sample["samples"]),
            }
            for tile, count in sample["top_mgrs"]
        )

        for name, stats in sample["numeric"].items():
            row = {"group": group, "metric": name}
            row.update(stats)
            numeric_rows.append(row)

    write_csv(out_dir / "overview.csv", overview)
    write_csv(out_dir / "landcover_major.csv", landcover_major)
    write_csv(out_dir / "landcover_pixels.csv", landcover_pixels)
    write_csv(out_dir / "monthly_counts.csv", months)
    write_csv(out_dir / "interval_bins.csv", intervals)
    write_csv(out_dir / "skip_reasons.csv", skips)
    write_csv(out_dir / "top_mgrs.csv", top_mgrs)
    write_csv(out_dir / "numeric_stats.csv", numeric_rows)


def make_plots(out_dir: Path, summaries: dict[str, dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib is unavailable, skip plots: {exc}")
        return

    for group, s in summaries.items():
        sample = s["sample_index"]

        lc = sample["counters"]["landcover_major"]
        if lc:
            labels = list(lc.keys())[:15]
            values = [lc[k] for k in labels]
            plt.figure(figsize=(10, 5))
            plt.bar(labels, values)
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("samples")
            plt.title(f"{group} landcover major")
            plt.tight_layout()
            plt.savefig(out_dir / f"{group}_landcover_major.png", dpi=180)
            plt.close()

        months = sample["counters"]["month"]
        if months:
            labels = sorted(months.keys())
            values = [months[k] for k in labels]
            plt.figure(figsize=(12, 4))
            plt.bar(labels, values)
            plt.xticks(rotation=60, ha="right")
            plt.ylabel("sample-date counts")
            plt.title(f"{group} month distribution")
            plt.tight_layout()
            plt.savefig(out_dir / f"{group}_monthly_counts.png", dpi=180)
            plt.close()


def main() -> None:
    args = parse_args()
    out_dir = args.out or (args.root / "distribution_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for group in args.groups:
        folder = args.root / group
        if not folder.exists():
            raise FileNotFoundError(f"dataset group folder not found: {folder}")
        print(f"analyze {group}: {folder}")
        summaries[group] = summarize_group(group, folder, args.max_landcover_rows)

    (out_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_tables(out_dir, summaries)
    write_markdown(out_dir / "analysis_report.md", summaries)
    if args.make_plots:
        make_plots(out_dir, summaries)

    print(f"done: {out_dir}")
    print(f"report: {out_dir / 'analysis_report.md'}")


if __name__ == "__main__":
    main()
