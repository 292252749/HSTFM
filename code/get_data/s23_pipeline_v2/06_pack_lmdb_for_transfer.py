"""Compact, verify, and package an S23 LMDB for transfer.

This script is intentionally separate from LMDB generation. Build with a
generous map_size in 04, then run this script once to create a smaller final
read-only LMDB and a zstd-compressed transfer package.
"""

import argparse
import hashlib
import json
import math
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import lmdb

import s23_common as C

cfg = C.cfg

SIDECARE_NAMES = [
    "metadata.json",
    "sample_index.csv",
    "build_report.json",
    "pair_index.json",
    "pair_analysis.json",
    "pair_summary.csv",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Create a transfer package from an S23 LMDB.")
    parser.add_argument("--lmdb-dir", type=Path, default=cfg.LMDB_OUT, help="Source LMDB directory.")
    parser.add_argument("--pipeline-dir", type=Path, default=cfg.PIPELINE_OUT, help="Directory containing sidecar metadata files.")
    parser.add_argument("--final-dir", type=Path, default=cfg.PIPELINE_OUT / "lmdb_512_final", help="Compact final LMDB directory.")
    parser.add_argument("--archive-dir", type=Path, default=cfg.PIPELINE_OUT / "transfer", help="Directory for archive parts and checksums.")
    parser.add_argument("--name", default=cfg.DATASET_NAME, help="Archive base name.")
    parser.add_argument("--force", action="store_true", default=True, help="Overwrite old outputs. This is the default.")
    parser.add_argument("--no-force", dest="force", action="store_false", help="Fail instead of overwriting existing final/package outputs.")
    parser.add_argument("--compact-mode", choices=["auto", "always", "never"], default="auto", help="When to rewrite a compact LMDB. Default: auto.")
    parser.add_argument("--skip-compact", action="store_true", help="Alias for --compact-mode never.")
    parser.add_argument("--skip-archive", action="store_true", help="Only prepare/verify the package directory; do not compress.")
    parser.add_argument("--compact-min-saving-ratio", type=float, default=0.05, help="Auto compact only if reclaimable map space is at least this ratio.")
    parser.add_argument("--compact-min-saving-gb", type=float, default=1.0, help="Auto compact only if reclaimable map space is at least this many GiB.")
    parser.add_argument("--map-margin", type=float, default=1.08, help="Destination LMDB map_size margin over used source pages.")
    parser.add_argument("--min-map-gb", type=float, default=1.0, help="Minimum destination LMDB map_size in GiB.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Key/value pairs per write transaction during compact copy.")
    parser.add_argument("--split-size-gb", type=float, default=10.0, help="Archive part size in GiB. Use 0 to disable splitting.")
    parser.add_argument("--zstd-level", type=int, default=5, help="zstd compression level; server reference used 5.")
    parser.add_argument("--packer", choices=["auto", "tar", "star"], default="auto", help="Directory packer.")
    args = parser.parse_args()
    if args.skip_compact:
        args.compact_mode = "never"
    return args


def byte_size(path):
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def human_size(n):
    n = float(n)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if n < 1024 or unit == "TiB":
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TiB"


def ensure_empty_dir(path, force):
    path = Path(path)
    if path.exists():
        if not force:
            raise FileExistsError(f"{path} already exists. Use --force to overwrite it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def open_readonly_lmdb(path):
    return lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=512,
        subdir=True,
    )


def source_used_bytes(env):
    info = env.info()
    stat = env.stat()
    page_size = int(stat["psize"])
    used_pages = int(info["last_pgno"]) + 1
    return used_pages * page_size


def planned_map_size(env, margin, min_map_gb):
    used = source_used_bytes(env)
    minimum = int(min_map_gb * (1 << 30))
    return max(minimum, int(math.ceil(used * margin)))


def lmdb_sample_status(env):
    keys = []
    with env.begin() as txn:
        with txn.cursor() as cur:
            for key, _ in cur:
                text = key.decode("utf-8", errors="replace")
                if text == "__metadata__":
                    continue
                if text.isdigit():
                    keys.append(int(text))
                else:
                    keys.append(text)
    numeric = all(isinstance(key, int) for key in keys)
    status = {"entries": len(keys), "numeric_keys": numeric}
    if numeric:
        keys.sort()
        gaps = []
        expected = 0
        for key in keys:
            if key != expected:
                gaps.append({"expected": expected, "actual": key})
                expected = key
            expected += 1
            if len(gaps) >= 20:
                break
        status.update(
            {
                "min_key": keys[0] if keys else None,
                "max_key": keys[-1] if keys else None,
                "contiguous_from_zero": not gaps and (not keys or keys[0] == 0),
                "first_gaps": gaps,
            }
        )
    return status


def read_lmdb_metadata(env):
    with env.begin() as txn:
        raw = txn.get(b"__metadata__")
    if raw is None:
        return None
    try:
        return pickle.loads(raw)
    except Exception as exc:
        return {"decode_error": str(exc)}


def verify_lmdb(path):
    env = open_readonly_lmdb(path)
    try:
        info = env.info()
        stat = env.stat()
        metadata = read_lmdb_metadata(env)
        sample_status = lmdb_sample_status(env)
        expected = None
        if isinstance(metadata, dict):
            expected = metadata.get("total_samples") or metadata.get("summary", {}).get("samples")
        ok = bool(metadata) and sample_status.get("contiguous_from_zero", False)
        if expected is not None:
            ok = ok and int(expected) == int(sample_status["entries"])
        return {
            "ok": ok,
            "path": str(Path(path).resolve()),
            "map_size_bytes": info.get("map_size"),
            "used_bytes": source_used_bytes(env),
            "stat": stat,
            "metadata_total_samples": expected,
            "sample_status": sample_status,
        }
    finally:
        env.close()


def copy_lmdb_compact_once(src_env, dst_dir, map_size, batch_size):
    dst_env = lmdb.open(str(dst_dir), map_size=map_size, subdir=True)
    copied = 0
    t0 = time.time()
    try:
        with src_env.begin() as src_txn:
            cursor = src_txn.cursor()
            dst_txn = dst_env.begin(write=True)
            try:
                for key, value in cursor:
                    dst_txn.put(key, value)
                    copied += 1
                    if copied % batch_size == 0:
                        dst_txn.commit()
                        C.progress(f"06 compact: copied {copied} records")
                        dst_txn = dst_env.begin(write=True)
                dst_txn.commit()
            except Exception:
                dst_txn.abort()
                raise
        dst_env.sync()
        C.progress(f"06 compact: copied {copied} records in {time.time() - t0:.1f}s")
        return copied
    finally:
        dst_env.close()


def copy_lmdb_compact(src_dir, dst_dir, map_margin, min_map_gb, batch_size):
    src_env = open_readonly_lmdb(src_dir)
    try:
        map_size = planned_map_size(src_env, map_margin, min_map_gb)
        used = source_used_bytes(src_env)
        for attempt in range(1, 5):
            if Path(dst_dir).exists():
                shutil.rmtree(dst_dir)
            Path(dst_dir).mkdir(parents=True, exist_ok=True)
            C.progress(
                f"06 compact: attempt={attempt}, source used={human_size(used)}, "
                f"dest map_size={human_size(map_size)}"
            )
            try:
                copied = copy_lmdb_compact_once(src_env, dst_dir, map_size, batch_size)
                return {"records_copied": copied, "map_size_bytes": map_size, "attempts": attempt}
            except lmdb.MapFullError:
                C.progress("06 compact: map full, retrying with larger map_size")
                map_size = int(math.ceil(map_size * 1.5))
        raise RuntimeError("Compact copy failed after repeated LMDB map_size expansion.")
    finally:
        src_env.close()


def copy_sidecars(pipeline_dir, final_dir):
    copied = []
    for name in SIDECARE_NAMES:
        src = Path(pipeline_dir) / name
        if src.exists():
            dst = Path(final_dir) / name
            shutil.copy2(src, dst)
            copied.append(name)
    return copied


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as fd:
        for chunk in iter(lambda: fd.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_packer(packer):
    if packer == "auto":
        if shutil.which("star"):
            return "star"
        return "tar"
    return packer


def build_pack_command(packer, parent, names):
    names = [str(name) for name in names]
    if packer == "star":
        if not shutil.which("star"):
            raise RuntimeError("star was requested but was not found in PATH.")
        return ["star", "-c"] + names, parent
    if not shutil.which("tar"):
        raise RuntimeError("tar was requested but was not found in PATH.")
    return ["tar", "-cf", "-"] + names, parent


def build_zstd_command(level):
    if not shutil.which("zstd"):
        raise RuntimeError("zstd was not found in PATH.")
    return ["zstd", f"-{level}", "-T0", "--progress", "-c"]


def write_stream_parts(stream, archive_dir, archive_base, split_bytes):
    part_paths = []
    part_index = 0
    current = None
    current_size = 0

    def open_part():
        nonlocal part_index, current, current_size
        suffix = f".part-{part_index:03d}" if split_bytes else ""
        path = archive_dir / f"{archive_base}.tar.zst{suffix}"
        current = path.open("wb")
        current_size = 0
        part_paths.append(path)
        part_index += 1
        C.progress(f"06 archive: writing {path.name}")

    open_part()
    try:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            offset = 0
            while offset < len(chunk):
                if split_bytes and current_size >= split_bytes:
                    current.close()
                    open_part()
                if split_bytes:
                    take = min(len(chunk) - offset, split_bytes - current_size)
                else:
                    take = len(chunk) - offset
                current.write(chunk[offset : offset + take])
                current_size += take
                offset += take
    finally:
        if current is not None:
            current.close()

    if split_bytes and len(part_paths) == 1:
        single = archive_dir / f"{archive_base}.tar.zst"
        part_paths[0].replace(single)
        part_paths = [single]
    return part_paths


def existing_sidecar_names(pipeline_dir):
    return [name for name in SIDECARE_NAMES if (Path(pipeline_dir) / name).exists()]

def archive_name_under(path, parent):
    path = Path(path).resolve()
    parent = Path(parent).resolve()
    try:
        return path.relative_to(parent).as_posix()
    except ValueError:
        return None


def create_archive(package_parent, package_names, archive_dir, archive_base, split_size_gb, zstd_level, packer):
    package_parent = Path(package_parent).resolve()
    package_names = [str(name) for name in package_names]
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)

    split_bytes = int(split_size_gb * (1 << 30)) if split_size_gb and split_size_gb > 0 else 0
    selected_packer = choose_packer(packer)
    pack_cmd, cwd = build_pack_command(selected_packer, package_parent, package_names)
    zstd_cmd = build_zstd_command(zstd_level)

    C.progress(
        f"06 archive: packer={selected_packer}, zstd_level={zstd_level}, "
        f"split={human_size(split_bytes) if split_bytes else 'disabled'}, inputs={package_names}"
    )
    pack_proc = subprocess.Popen(pack_cmd, cwd=str(cwd), stdout=subprocess.PIPE)
    zstd_proc = subprocess.Popen(zstd_cmd, stdin=pack_proc.stdout, stdout=subprocess.PIPE)
    pack_proc.stdout.close()
    try:
        part_paths = write_stream_parts(zstd_proc.stdout, archive_dir, archive_base, split_bytes)
    finally:
        if zstd_proc.stdout is not None:
            zstd_proc.stdout.close()
    zstd_rc = zstd_proc.wait()
    pack_rc = pack_proc.wait()
    if zstd_rc != 0:
        raise RuntimeError(f"zstd failed with exit code {zstd_rc}: {' '.join(zstd_cmd)}")
    if pack_rc != 0:
        raise RuntimeError(f"packer command failed with exit code {pack_rc}: {' '.join(pack_cmd)}")
    return part_paths


def write_checksums(paths, output_path):
    lines = []
    for path in paths:
        digest = sha256_file(path)
        lines.append(f"{digest}  {Path(path).name}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def compact_plan(args, source_check):
    map_size = int(source_check.get("map_size_bytes") or 0)
    used = int(source_check.get("used_bytes") or 0)
    saving = max(0, map_size - used)
    ratio = saving / map_size if map_size > 0 else 0.0
    min_saving = int(args.compact_min_saving_gb * (1 << 30))
    if args.compact_mode == "always":
        should = True
        reason = "forced_always"
    elif args.compact_mode == "never":
        should = False
        reason = "forced_never"
    elif saving >= min_saving and ratio >= args.compact_min_saving_ratio:
        should = True
        reason = "saving_above_threshold"
    else:
        should = False
        reason = "saving_below_threshold"
    return {
        "should_compact": should,
        "reason": reason,
        "reclaimable_bytes": saving,
        "reclaimable_ratio": ratio,
        "min_saving_bytes": min_saving,
        "min_saving_ratio": args.compact_min_saving_ratio,
    }


def main():
    args = parse_args()
    src_lmdb = args.lmdb_dir.resolve()
    final_dir = args.final_dir.resolve()
    archive_dir = args.archive_dir.resolve()
    pipeline_dir = args.pipeline_dir.resolve()

    if not src_lmdb.exists():
        raise FileNotFoundError(f"Source LMDB does not exist: {src_lmdb}")

    C.progress(f"06 start: source={src_lmdb}")
    source_check = verify_lmdb(src_lmdb)
    if not source_check["ok"]:
        raise RuntimeError(f"Source LMDB verification failed: {json.dumps(source_check, indent=2)}")
    C.progress(
        "06 source verified: "
        f"samples={source_check['sample_status']['entries']}, "
        f"map={human_size(source_check['map_size_bytes'])}, "
        f"used={human_size(source_check['used_bytes'])}"
    )

    plan = compact_plan(args, source_check)
    C.progress(
        "06 compact plan: "
        f"mode={args.compact_mode}, should_compact={plan['should_compact']}, "
        f"reason={plan['reason']}, reclaimable={human_size(plan['reclaimable_bytes'])}, "
        f"ratio={plan['reclaimable_ratio']:.4f}"
    )

    compact_result = None
    package_parent = None
    package_names = None
    package_lmdb = None
    final_check = None
    sidecars = []

    if plan["should_compact"]:
        ensure_empty_dir(final_dir, args.force)
        compact_result = copy_lmdb_compact(src_lmdb, final_dir, args.map_margin, args.min_map_gb, args.batch_size)
        sidecars = copy_sidecars(pipeline_dir, final_dir)
        C.progress(f"06 sidecars: copied {sidecars}")
        final_check = verify_lmdb(final_dir)
        if not final_check["ok"]:
            raise RuntimeError(f"Final LMDB verification failed: {json.dumps(final_check, indent=2)}")
        package_parent = final_dir.parent
        package_names = [final_dir.name]
        package_lmdb = final_dir
        C.progress(
            "06 final verified: "
            f"samples={final_check['sample_status']['entries']}, "
            f"map={human_size(final_check['map_size_bytes'])}, "
            f"used={human_size(final_check['used_bytes'])}, "
            f"dir_size={human_size(byte_size(final_dir))}"
        )
    else:
        sidecars = existing_sidecar_names(pipeline_dir)
        src_archive_name = archive_name_under(src_lmdb, pipeline_dir)
        if src_archive_name is None:
            package_parent = src_lmdb.parent
            package_names = [src_lmdb.name]
            sidecars = []
        else:
            package_parent = pipeline_dir
            package_names = [src_archive_name] + sidecars
        package_lmdb = src_lmdb
        final_check = source_check
        C.progress(f"06 compact skipped: packaging source LMDB with sidecars={sidecars}")

    manifest = {
        "schema": "s23_lmdb_transfer_manifest_v1",
        "dataset_group": cfg.DATASET_GROUP,
        "dataset_name": cfg.DATASET_NAME,
        "source_lmdb": str(src_lmdb),
        "package_lmdb": str(package_lmdb),
        "final_dir": str(final_dir) if plan["should_compact"] else "",
        "package_parent": str(package_parent),
        "package_names": package_names,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_check": source_check,
        "final_check": final_check,
        "compact_plan": plan,
        "compact_result": compact_result,
        "package_lmdb_size_bytes": byte_size(package_lmdb),
        "sidecars": sidecars,
        "archive": None,
    }

    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_paths = []
    if not args.skip_archive:
        if archive_dir.exists() and args.force:
            for item in archive_dir.glob(f"{args.name}.tar.zst*"):
                if item.is_file():
                    item.unlink()
        archive_paths = create_archive(
            package_parent=package_parent,
            package_names=package_names,
            archive_dir=archive_dir,
            archive_base=args.name,
            split_size_gb=args.split_size_gb,
            zstd_level=args.zstd_level,
            packer=args.packer,
        )
        checksum_path = archive_dir / f"{args.name}.sha256"
        checksum_lines = write_checksums(archive_paths, checksum_path)
        archive_info = [
            {"path": str(path.resolve()), "name": path.name, "bytes": path.stat().st_size, "sha256": line.split()[0]}
            for path, line in zip(archive_paths, checksum_lines)
        ]
        manifest["archive"] = {
            "archive_dir": str(archive_dir),
            "parts": archive_info,
            "checksum_file": str(checksum_path.resolve()),
            "split_size_bytes": int(args.split_size_gb * (1 << 30)) if args.split_size_gb and args.split_size_gb > 0 else 0,
            "zstd_level": args.zstd_level,
        }
        C.progress(f"06 archive complete: {len(archive_paths)} file(s), total={human_size(sum(p.stat().st_size for p in archive_paths))}")

    manifest_path = archive_dir / "transfer_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if plan["should_compact"]:
        shutil.copy2(manifest_path, final_dir / "transfer_manifest.json")

    C.progress("06 finish")
    print(f"Packaged LMDB: {package_lmdb}")
    if archive_paths:
        print(f"Archive files: {[str(path) for path in archive_paths]}")
        print(f"Checksums: {archive_dir / (args.name + '.sha256')}")
    else:
        print("Archive skipped")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise






