"""Audit the displacement exports and write separate even/odd NumPy maps."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from contextlib import contextmanager

import numpy as np

TIME = re.compile(rb"Time\s*[,=:]?\s*(\d+)\s*", re.I)
CASE = re.compile(r"([A-Z]+Force(?:Only)?)[-_ ]+(\d+(?:\.\d+)?)倍", re.I)
ROW = np.dtype([("id", "<i8"), ("dx", "<f8"), ("dy", "<f8"), ("dz", "<f8")])
TEST_INDICES = np.arange(1, 101, 2)


def save_json(path, data):
    partial = path.with_suffix(".json.partial")
    partial.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
                       encoding="utf-8")
    partial.replace(path)


def read_time_map(path):
    if path.suffix.lower() == ".xlsx":
        from openpyxl import load_workbook
        book = load_workbook(path, read_only=True, data_only=True)
        try:
            pairs = list(book.active.values)[1:]
        finally:
            book.close()
    else:
        with path.open(encoding="utf-8-sig", newline="") as f:
            pairs = list(csv.reader(f))[1:]
    labels, times = [], []
    for label, seconds in pairs:
        match = TIME.fullmatch(str(label).strip().encode())
        if match is None:
            raise ValueError(f"{path}: invalid time label {label!r}")
        labels.append(int(match[1]))
        times.append(float(seconds))
    times = np.asarray(times)
    if len(set(labels)) != 101 or times.shape != (101,) or not np.allclose(
            times, np.arange(101) * 0.001, rtol=0, atol=1e-10):
        raise ValueError(f"{path}: expected 101 unique labels at 0..0.100 s in 0.001 s steps")
    return labels, times


def table(lines):
    rows = np.loadtxt(lines, delimiter=",", dtype=ROW, comments=None, ndmin=1)
    xyz = np.column_stack([rows[c] for c in ("dx", "dy", "dz")])
    if not np.isfinite(xyz).all():
        row, axis = np.argwhere(~np.isfinite(xyz))[0]
        raise ValueError(f"NaN/Inf at node {rows['id'][row]}, component {'XYZ'[axis]}")
    if np.max(np.abs(xyz)) > np.finfo(np.float32).max:
        raise ValueError("Values exceed the float32 range")
    return rows["id"], xyz


def read_coordinates(path):
    with path.open("rb") as f:
        columns = f.readline().removeprefix(b"\xef\xbb\xbf").lower().strip().split(b",")
        columns = [col.strip() for col in columns]
        if columns not in ([b"nodeid", b"dx", b"dy", b"dz"], [b"nodeid", b"x", b"y", b"z"]):
            raise ValueError(f"{path}: unrecognized initial-coordinate header")
        ids, xyz = table(f)
    if np.any(ids <= 0) or len(np.unique(ids)) != len(ids):
        raise ValueError(f"{path}: nonpositive or duplicate coordinate node ID")
    return ids, xyz


def split_members(lines):
    cases = {}
    for member in lines:
        name = member.replace("\\", "/").rsplit("/", 1)[-1]
        match = CASE.search(name)
        if name.endswith("-位移结果.txt") and match:
            key = f"{match[1]}_{match[2]}x"
            if key in cases and cases[key] != member:
                raise ValueError(f"Two displacement exports for {key}: {cases[key]}, {member}")
            cases[key] = member
    return dict(sorted(cases.items()))


def archive_commands(archive):
    local_unrar = Path(".tools/usr/bin/unrar-nonfree").resolve()
    unrar = shutil.which("unrar") or (str(local_unrar) if local_unrar.is_file() else None)
    if unrar:
        args = [unrar, "lb", "-v", "-c-", str(archive)]
        members = subprocess.check_output(args, encoding="utf-8").splitlines()
        return members, [unrar, "p", "-inul", str(archive)]
    seven = next((command for command in ("7zz", "7z", "7za") if shutil.which(command)), None)
    if seven:
        listing = subprocess.check_output([seven, "l", "-slt", str(archive)], encoding="utf-8")
        members = [line[7:] for line in listing.splitlines() if line.startswith("Path = ")]
        return members, [seven, "x", "-so", "-bd", str(archive)]
    raise RuntimeError("Install unrar or a 7-Zip executable with RAR support")


@contextmanager
def open_member(command, member):
    process = subprocess.Popen([*command, member], stdout=subprocess.PIPE)
    try:
        with process.stdout:
            yield process.stdout
    except BaseException:
        if process.poll() is None:
            process.terminate()
        process.wait()
        raise
    if process.wait():
        raise ValueError(f"Archive extraction failed for {member}")


def frame_statistics(index, label, xyz, same_order, same_previous):
    norm = np.linalg.norm(xyz, axis=1)
    return dict(frame_index=index, time_label=label, node_count=len(xyz),
                same_order_as_coordinates=same_order, same_order_as_previous=same_previous,
                min_dx_dy_dz=xyz.min(axis=0).tolist(), max_dx_dy_dz=xyz.max(axis=0).tolist(),
                vector_rms=float(np.sqrt(np.mean(norm**2))), max_vector=float(norm.max()))


def audit_statistics(frames, max_rounding):
    max_values = np.array([frame["max_vector"] for frame in frames])
    nonzero = max_values[max_values > 0]
    threshold = 100 * np.median(nonzero) if len(nonzero) else 0.0
    return dict(frames=frames, max_rounding_abs=max_rounding, nan_inf=0,
                missing_frames=0, missing_or_duplicate_node_records=0,
                reordered_frames=sum(not frame["same_order_as_coordinates"] for frame in frames),
                scale_outlier_threshold=float(threshold),
                scale_outlier_frames=np.flatnonzero(max_values > threshold).tolist())


def binary_maps(destination, count):
    return tuple(np.lib.format.open_memmap(destination / name, mode="w+", dtype="float32",
                                         shape=(frames, count, 3))
                 for name, frames in (("known_even.npy", 51), ("truth_odd.npy", 50)))


def original_id_orders(source):
    """Read only marker and ID columns; the established parser already validated values."""
    ids = []
    label = None
    with source.open("rb", buffering=4 * 1024**2) as stream:
        for line in stream:
            field = line.partition(b",")[0].strip().removeprefix(b"\xef\xbb\xbf")
            if field[:4].lower() == b"time":
                if label is not None:
                    yield label, np.asarray(ids, dtype=np.int64)
                label = int(TIME.fullmatch(field)[1])
                ids.clear()
            elif field.isdigit():
                ids.append(int(field))
    if label is not None:
        yield label, np.asarray(ids, dtype=np.int64)


def reuse_previous(previous, source, destination, ids, labels):
    """Reuse the older strict parser's arrays after checking their source and mapping."""
    meta = json.loads((previous / "snapshots_manifest.json").read_text(encoding="utf-8"))
    raw_stat = source.stat()
    field_path = previous / "snapshots.npy"
    field_stat = field_path.stat()
    expected = {"path": str(source.resolve()), "size_bytes": raw_stat.st_size,
                "mtime_ns": raw_stat.st_mtime_ns}
    if (meta["status"] != "validated" or meta["source"] != expected or meta["time_labels"] != labels
            or meta["snapshots"]["size_bytes"] != field_stat.st_size
            or meta["snapshots"]["mtime_ns"] != field_stat.st_mtime_ns):
        return None
    old_ids = np.load(previous.parent / "node_ids.npy")
    order = np.argsort(old_ids)
    positions = np.searchsorted(old_ids[order], ids)
    if (len(ids) != len(old_ids) or np.any(positions == len(old_ids))
            or not np.array_equal(old_ids[order[positions]], ids)):
        raise ValueError("Previous cache node IDs differ from the supplied coordinate file")
    aligned = order[positions]
    snapshots = np.load(field_path, mmap_mode="r")
    if snapshots.shape != (len(labels), len(ids), 3) or snapshots.dtype != np.float32:
        raise ValueError("Previous cache shape or dtype differs from the audit record")
    outputs = binary_maps(destination, len(ids))
    frames = []
    previous_order = None
    for index, (label, raw_ids) in enumerate(original_id_orders(source)):
        if index >= len(labels) or label != labels[index] or len(raw_ids) != len(ids):
            raise ValueError(f"Original ID scan disagrees with the cached Time {label}")
        if not np.array_equal(np.sort(raw_ids), old_ids[order]):
            raise ValueError(f"Unknown, duplicate, or missing ID at Time {label}")
        xyz = snapshots[index, aligned].astype(np.float64)
        if not np.isfinite(xyz).all():
            raise ValueError(f"Nonfinite values in the previous cache at Time {label}")
        outputs[index % 2][index // 2] = xyz
        frames.append(frame_statistics(index, label, xyz, np.array_equal(raw_ids, ids),
                                       previous_order is None or np.array_equal(raw_ids, previous_order)))
        previous_order = raw_ids
    if len(frames) != len(labels):
        raise ValueError("Previous cache disagrees with the source frame count")
    for output in outputs:
        output.flush()
    return dict(**audit_statistics(frames, meta["max_float32_conversion_abs_error"]),
                reused_snapshot=str(field_path.resolve()))


def parse_case(stream, destination, ids, labels):
    count = len(ids)
    lookup = np.argsort(ids)
    sorted_ids = ids[lookup]
    known, truth = binary_maps(destination, count)
    rows, frames = [], []
    max_rounding = 0.0
    previous_order = None

    def finish():
        nonlocal max_rounding, previous_order
        index = len(frames)
        if len(rows) != count:
            raise ValueError(f"Time {labels[index]}: expected {count} nodes, got {len(rows)}")
        frame_ids, xyz = table(rows)
        same_order = np.array_equal(frame_ids, ids)
        if not same_order:
            positions = np.searchsorted(sorted_ids, frame_ids)
            if (np.any(positions == count)
                    or not np.array_equal(sorted_ids[positions], frame_ids)):
                raise ValueError(f"Time {labels[index]}: unknown node ID")
            if len(np.unique(positions)) != count:
                raise ValueError(f"Time {labels[index]}: duplicated/missing node ID")
            aligned = np.empty_like(xyz)
            aligned[lookup[positions]] = xyz
            xyz = aligned
        stored = xyz.astype(np.float32)
        max_rounding = max(max_rounding, float(np.max(np.abs(xyz - stored))))
        output = truth if index % 2 else known
        output[index // 2] = stored
        frames.append(frame_statistics(index, labels[index], xyz, same_order,
                                       previous_order is None or np.array_equal(frame_ids, previous_order)))
        previous_order = frame_ids
        rows.clear()

    active = False
    for raw in stream:
        line = raw.strip().removeprefix(b"\xef\xbb\xbf")
        if not line:
            continue
        if line[:4].lower() == b"time":
            if active:
                finish()
            match = TIME.fullmatch(line)
            index = len(frames)
            if match is None or index >= len(labels) or int(match[1]) != labels[index]:
                raise ValueError(f"Missing, duplicate or out-of-order time block {index}: {line!r}")
            active = True
        elif active and not rows and line.lower().replace(b" ", b"").endswith(b",dx,dy,dz"):
            continue
        elif active:
            rows.append(line)
            if len(rows) > count:
                raise ValueError(f"Time {labels[len(frames)]}: more rows than coordinate nodes")
        else:
            raise ValueError(f"Content before first Time marker: {line[:100]!r}")
    if active:
        finish()
    if len(frames) != len(labels):
        raise ValueError(f"Missing time blocks {labels[len(frames):]}")
    known.flush()
    truth.flush()
    return audit_statistics(frames, max_rounding)


def cached_dataset(path):
    ids = np.load(path / "node_ids.npy")
    coords = np.load(path / "coordinates.npy")
    times = np.load(path / "times_s.npy")
    metadata = json.loads((path / "dataset.json").read_text(encoding="utf-8"))
    return ids, coords, times, tuple(metadata["cases"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="first RAR volume")
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--time-map", type=Path, required=True)
    parser.add_argument("--extracted", type=Path, help="reuse exact matching exported TXT files")
    parser.add_argument("--previous-cache", type=Path, help="reuse prepared_v1 arrays with matching provenance")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    labels, times = read_time_map(args.time_map)
    ids, xyz = read_coordinates(args.coordinates)
    archive = args.archive.resolve()
    volumes = sorted(archive.parent.glob(archive.name.rsplit(".part", 1)[0] + ".part*.rar"))
    members, command = archive_commands(archive)
    cases = split_members(members)
    if not cases:
        raise ValueError("No displacement exports identified in the archive")
    args.out.mkdir(parents=True, exist_ok=True)
    audit_path = args.out / "audit.json"
    audit = dict(archive=str(archive), volumes={v.name: v.stat().st_size for v in volumes},
                 coordinates=str(args.coordinates.resolve()), time_map=str(args.time_map.resolve()),
                 time_labels=labels, times_s=times.tolist(), node_count=len(ids),
                 coordinate_range=[xyz.min(axis=0).tolist(), xyz.max(axis=0).tolist()],
                 files=members, cases={})
    old_cases = {}
    if audit_path.exists():
        old = json.loads(audit_path.read_text(encoding="utf-8"))
        same_source = all(old.get(key) == audit[key] for key in
                          ("volumes", "archive", "coordinates", "time_map", "time_labels", "times_s"))
        if not same_source or not np.array_equal(np.load(args.out / "node_ids.npy"), ids) or \
                not np.array_equal(np.load(args.out / "coordinates.npy"), xyz):
            raise ValueError("Source changed; select a fresh output directory")
        old_cases = old["cases"]
    np.save(args.out / "node_ids.npy", ids)
    np.save(args.out / "coordinates.npy", xyz)
    np.save(args.out / "times_s.npy", times)
    audit["cases"] = {key: old_cases.get(key, {"status": "pending", "member": member})
                      for key, member in cases.items()}
    save_json(audit_path, audit)
    extracted = {p.name: p for p in args.extracted.rglob("*.txt")} if args.extracted else {}
    previous_cases = {}
    if args.previous_cache:
        for manifest in args.previous_cache.glob("*/snapshots_manifest.json"):
            source_name = Path(json.loads(manifest.read_text(encoding="utf-8"))["source"]["path"]).name
            previous_cases[source_name] = manifest.parent
    errors = []
    for key, member in cases.items():
        destination = args.out / key
        destination.mkdir(exist_ok=True)
        if audit["cases"][key]["status"] == "validated" and all(
                (destination / name).is_file() for name in ("known_even.npy", "truth_odd.npy")):
            print(f"{key}: reuse validated cache", flush=True)
            continue
        source = extracted.get(member.replace("\\", "/").rsplit("/", 1)[-1])
        print(f"{key}: reading {source or member}", flush=True)
        try:
            previous = previous_cases.get(source.name) if source else None
            details = reuse_previous(previous, source, destination, ids, labels) if previous else None
            if details is None:
                if source:
                    with source.open("rb", buffering=4 * 1024**2) as stream:
                        details = parse_case(stream, destination, ids, labels)
                else:
                    with open_member(command, member) as stream:
                        details = parse_case(stream, destination, ids, labels)
            audit["cases"][key] = dict(status="validated", member=member,
                                       extracted_source=str(source.resolve()) if source else None,
                                       **details)
            print(f"{key}: {len(details['frames'])} frames × {len(ids)} nodes; "
                  f"{details['reordered_frames']} reordered frames", flush=True)
        except (ValueError, OSError) as error:
            audit["cases"][key] = dict(status="failed", member=member, error=str(error))
            errors.append(key)
            print(f"{key}: {error}", flush=True)
        save_json(audit_path, audit)
    if errors:
        raise SystemExit(f"Data audit failed for: {', '.join(errors)}")
    save_json(args.out / "dataset.json", {"cases": list(cases)})


if __name__ == "__main__":
    main()
