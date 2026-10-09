"""
Summarize a 512-sample S2/S3 sample_index.csv.

Edit the CONFIG section and run after build_lmdb_512.py. Outputs are lightweight
CSV/JSON files that help decide whether the sample distribution is suitable for
the planned experiments.
"""

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


# =========================
# CONFIG: edit values here.
# =========================
SAMPLE_INDEX_CSV = Path("cdse_batch_pipeline_out/v2_512/sample_index.csv")
REPORT_DIR = Path("cdse_batch_pipeline_out/v2_512/reports")
NUMERIC_FIELDS = [
    "land_fraction",
    "water_fraction",
    "nodata_fraction",
    "dominant_landcover_ratio",
    "s2_change_mean",
    "s3_change_mean",
    "s2_ndvi_change_mean",
    "s3_ndvi_change_mean",
    "s3_t1_t2_shift_magnitude",
    "s3_t1_t2_registration_peak",
    "interval_days",
]
CATEGORY_FIELDS = [
    "source_dataset",
    "region",
    "continent",
    "country",
    "landform",
    "stratum",
    "mgrs_tile",
    "t1_year",
    "t2_year",
    "t1_month",
    "t2_month",
    "t1_season",
    "t2_season",
    "landcover_major",
]


def load_rows(path):
    with path.open("r", newline="", encoding="utf-8") as fd:
        return list(csv.DictReader(fd))


def to_float(value):
    if value in ("", None):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def quantiles(values):
    if not values:
        return {}
    values = sorted(values)

    def q(p):
        if len(values) == 1:
            return values[0]
        pos = (len(values) - 1) * p
        low = int(math.floor(pos))
        high = int(math.ceil(pos))
        if low == high:
            return values[low]
        return values[low] * (high - pos) + values[high] * (pos - low)

    return {
        "count": len(values),
        "min": values[0],
        "p05": q(0.05),
        "p25": q(0.25),
        "median": q(0.50),
        "p75": q(0.75),
        "p95": q(0.95),
        "max": values[-1],
        "mean": sum(values) / len(values),
    }


def write_counter(counter, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fd:
        writer = csv.writer(fd)
        writer.writerow(["value", "count"])
        for value, count in counter.most_common():
            writer.writerow([value, count])


def write_group_table(rows, group_field, numeric_fields, path):
    groups = defaultdict(list)
    for row in rows:
        groups[row.get(group_field, "")].append(row)
    with path.open("w", newline="", encoding="utf-8") as fd:
        fieldnames = [group_field, "count"]
        for field in numeric_fields:
            fieldnames.extend([f"{field}_mean", f"{field}_median"])
        writer = csv.DictWriter(fd, fieldnames=fieldnames)
        writer.writeheader()
        for group_value, group_rows in sorted(groups.items(), key=lambda item: str(item[0])):
            out = {group_field: group_value, "count": len(group_rows)}
            for field in numeric_fields:
                values = [to_float(row.get(field)) for row in group_rows]
                values = [value for value in values if value is not None]
                qs = quantiles(values)
                out[f"{field}_mean"] = qs.get("mean", "")
                out[f"{field}_median"] = qs.get("median", "")
            writer.writerow(out)


def main():
    rows = load_rows(SAMPLE_INDEX_CSV)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "s23_lmdb_512_summary_v1",
        "sample_index_csv": str(SAMPLE_INDEX_CSV),
        "sample_count": len(rows),
        "unique_groups": len({row.get("group_id", "") for row in rows}),
        "unique_mgrs_tiles": len({row.get("mgrs_tile", "") for row in rows}),
        "numeric": {},
        "categories": {},
    }

    for field in NUMERIC_FIELDS:
        values = [to_float(row.get(field)) for row in rows]
        values = [value for value in values if value is not None]
        summary["numeric"][field] = quantiles(values)

    for field in CATEGORY_FIELDS:
        counter = Counter(row.get(field, "") for row in rows)
        summary["categories"][field] = dict(counter.most_common())
        write_counter(counter, REPORT_DIR / f"count_by_{field}.csv")

    for group_field in ("source_dataset", "region", "continent", "landcover_major", "t1_year", "t2_year"):
        if group_field in rows[0] if rows else False:
            write_group_table(rows, group_field, NUMERIC_FIELDS, REPORT_DIR / f"metrics_by_{group_field}.csv")

    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary written to {REPORT_DIR / 'summary.json'}")
    print(f"Tables written under {REPORT_DIR}")


if __name__ == "__main__":
    main()

