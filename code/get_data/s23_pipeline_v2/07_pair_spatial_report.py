"""Create spatial QA files from pair_index.json.

Outputs a CSV, GeoJSON, and summary JSON that make it easy to inspect whether
pairs fall inside a target lon/lat bbox.
"""

import argparse
import csv
import json
from pathlib import Path

import s23_common as C

cfg = C.cfg

CHINA_BBOX = [73.5, 18.0, 135.1, 53.6]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize pair spatial locations from pair_index.json.")
    parser.add_argument("--pair-index", type=Path, default=cfg.PAIR_INDEX_JSON, help="Input pair_index.json.")
    parser.add_argument("--out-dir", type=Path, default=cfg.PIPELINE_OUT / "pair_spatial_report", help="Output directory.")
    parser.add_argument("--bbox", nargs=4, type=float, default=CHINA_BBOX, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"), help="Reference bbox in WGS84.")
    return parser.parse_args()


def bbox_union(bounds_list):
    bounds = [tuple(float(v) for v in b) for b in bounds_list if b]
    if not bounds:
        return []
    return [
        min(b[0] for b in bounds),
        min(b[1] for b in bounds),
        max(b[2] for b in bounds),
        max(b[3] for b in bounds),
    ]


def bbox_center(bounds):
    if not bounds:
        return "", ""
    return (float(bounds[0]) + float(bounds[2])) / 2.0, (float(bounds[1]) + float(bounds[3])) / 2.0


def bbox_inside(inner, outer):
    return C.bbox_contains(outer, inner)


def bbox_polygon(bounds):
    if not bounds:
        return None
    minx, miny, maxx, maxy = [float(v) for v in bounds]
    return {
        "type": "Polygon",
        "coordinates": [[
            [minx, miny],
            [maxx, miny],
            [maxx, maxy],
            [minx, maxy],
            [minx, miny],
        ]],
    }


def pair_bounds(pair):
    bounds = []
    for time_key in ["t1", "t2"]:
        item = pair.get(time_key, {})
        s2 = item.get("s2", {})
        if s2.get("bounds_wgs84"):
            bounds.append(s2["bounds_wgs84"])
        for s3 in item.get("candidate_s3", []):
            if s3.get("bounds_wgs84"):
                bounds.append(s3["bounds_wgs84"])
    return bbox_union(bounds)


def row_for_pair(pair, bbox):
    bounds = pair_bounds(pair)
    center_lon, center_lat = bbox_center(bounds)
    t1_s2 = pair.get("t1", {}).get("s2", {})
    t2_s2 = pair.get("t2", {}).get("s2", {})
    t1_s3 = (pair.get("t1", {}).get("candidate_s3") or [{}])[0]
    t2_s3 = (pair.get("t2", {}).get("candidate_s3") or [{}])[0]
    return {
        "pair_id": pair.get("pair_id", ""),
        "mgrs_tile": pair.get("mgrs_tile", ""),
        "t1_date": pair.get("t1_date", ""),
        "t2_date": pair.get("t2_date", ""),
        "interval_days": pair.get("interval_days", ""),
        "center_lon": center_lon,
        "center_lat": center_lat,
        "pair_bounds_wgs84": json.dumps(bounds),
        "inside_bbox": bbox_inside(bounds, bbox),
        "intersects_bbox": C.bbox_intersects(bounds, bbox),
        "t1_s2_bounds_wgs84": json.dumps(t1_s2.get("bounds_wgs84", [])),
        "t2_s2_bounds_wgs84": json.dumps(t2_s2.get("bounds_wgs84", [])),
        "t1_s3_bounds_wgs84": json.dumps(t1_s3.get("bounds_wgs84", [])),
        "t2_s3_bounds_wgs84": json.dumps(t2_s3.get("bounds_wgs84", [])),
        "t1_s2": t1_s2.get("name", ""),
        "t2_s2": t2_s2.get("name", ""),
        "t1_s3": t1_s3.get("name", ""),
        "t2_s3": t2_s3.get("name", ""),
        "t1_cloud": pair.get("quality", {}).get("t1_cloud", ""),
        "t2_cloud": pair.get("quality", {}).get("t2_cloud", ""),
    }


def write_csv(path, rows):
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fd:
        writer = csv.DictWriter(fd, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_geojson(path, rows):
    features = []
    for row in rows:
        bounds = json.loads(row["pair_bounds_wgs84"]) if row["pair_bounds_wgs84"] else []
        geom = bbox_polygon(bounds)
        if geom is None:
            continue
        features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": {k: v for k, v in row.items() if k != "pair_bounds_wgs84"},
        })
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    pair_index = C.read_json(args.pair_index, {})
    pairs = pair_index.get("pairs", [])
    bbox = [float(v) for v in args.bbox]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = [row_for_pair(pair, bbox) for pair in pairs]
    inside = sum(1 for row in rows if row["inside_bbox"])
    intersects = sum(1 for row in rows if row["intersects_bbox"])
    summary = {
        "schema": "s23_pair_spatial_report_v1",
        "pair_index": str(args.pair_index),
        "bbox_wgs84": bbox,
        "pairs": len(rows),
        "inside_bbox": inside,
        "outside_bbox": len(rows) - inside,
        "intersects_bbox": intersects,
        "not_intersects_bbox": len(rows) - intersects,
        "pair_union_bounds_wgs84": bbox_union([json.loads(row["pair_bounds_wgs84"]) for row in rows if row["pair_bounds_wgs84"]]),
    }

    csv_path = args.out_dir / "pair_spatial_report.csv"
    geojson_path = args.out_dir / "pair_spatial_report.geojson"
    summary_path = args.out_dir / "pair_spatial_summary.json"
    write_csv(csv_path, rows)
    write_geojson(geojson_path, rows)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    C.progress(f"07 finish: pairs={len(rows)}, inside={inside}, intersects={intersects}")
    print(f"CSV: {csv_path}")
    print(f"GeoJSON: {geojson_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
