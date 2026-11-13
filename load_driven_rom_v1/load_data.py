"""Read the project's aligned displacement and original force tables."""
from __future__ import annotations

import io
from pathlib import Path, PurePosixPath
import shutil
import subprocess

import numpy as np


CACHED_LOADS = {
    "XForceOnly_1x": "X",
    "YForceOnly_1x": "Y",
    "ZForceOnly_1x": "Z",
}
ARCHIVE_CASES = {
    "YXZForce_2.5x": "YXZForce-2.5倍幅值",
    "XYZForce_2x": "XYZForce-2倍幅值",
}
AXES = "XYZ"


class Snapshots:
    """Expose the 1 ms sequence from the existing odd/even maps without copying it."""

    def __init__(self, even, odd, node_count):
        if (even.shape != (51, node_count, 3)
                or odd.shape != (50, node_count, 3)
                or even.dtype != np.float32 or odd.dtype != np.float32):
            raise ValueError("Unexpected odd/even displacement cache shape or dtype")
        self._fields = (even, odd)
        self.shape = (101, node_count, 3)
        self.dtype = even.dtype

    def __getitem__(self, index):
        if isinstance(index, tuple):
            frame_index, *other_axes = index
        else:
            frame_index, other_axes = index, []
        frames = np.arange(self.shape[0])[frame_index]
        if np.ndim(frames) == 0:
            frame = int(frames)
            return self._fields[frame % 2][(frame // 2, *other_axes)]
        if np.ndim(frames) != 1:
            raise ValueError("Use a scalar, slice or 1D frame index")
        if len(frames) == 0:
            return self._fields[0][(frames, *other_axes)]
        values = [self._fields[int(frame % 2)][(int(frame // 2), *other_axes)]
                  for frame in frames]
        return np.stack(values)


def load_displacements(cache, names):
    ids = np.load(cache / "node_ids.npy", allow_pickle=False)
    times = np.load(cache / "times_s.npy", allow_pickle=False)
    if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer)
            or len(np.unique(ids)) != len(ids)):
        raise ValueError("Invalid displacement node IDs")
    if times.shape != (101,) or not np.allclose(times, np.arange(101) * .001,
                                                rtol=0, atol=1e-10):
        raise ValueError("Expected 101 displacement frames on the 0–100 ms grid")
    snapshots = {}
    for case in names:
        folder = cache / case
        even = np.load(folder / "known_even.npy", mmap_mode="r", allow_pickle=False)
        odd = np.load(folder / "truth_odd.npy", mmap_mode="r", allow_pickle=False)
        snapshots[case] = Snapshots(even, odd, len(ids))
    return ids, times, snapshots


def cached_force(path):
    with np.load(path, allow_pickle=False) as raw:
        names = [str(value) for value in raw["axes"]]
        times = raw["times_s"]
        values = raw["values"]
    if (len(names) != len(set(names)) or any(name not in AXES or len(name) != 1 for name in names)
            or times.shape != (len(names), 51) or values.shape != times.shape):
        raise ValueError(f"{path}: malformed force cache")
    return {name: np.column_stack((t, f)) for name, t, f in zip(names, times, values)}


def archived_force(root):
    archive_root = root / "data/raw/updated_cases_2026-11-01"
    volumes = list(archive_root.glob("*.part01.rar"))
    if len(volumes) != 1:
        raise ValueError(f"Expected exactly one first RAR volume under {archive_root}")
    archive = volumes[0]
    bundled = root / ".tools/usr/bin/unrar-nonfree"
    unrar = str(bundled) if bundled.is_file() else shutil.which("unrar")
    if unrar is None:
        raise FileNotFoundError("unrar is needed to read the archived load tables")
    paths = dict.fromkeys(subprocess.check_output([unrar, "lb", "-v", str(archive)]).decode().splitlines())
    tables, sources = {}, {}
    for case, suffix in ARCHIVE_CASES.items():
        tables[case], sources[case] = {}, []
        for axis in AXES:
            member, = [path for path in paths if PurePosixPath(path).parent.name.endswith(f"-{suffix}")
                       and PurePosixPath(path).name == f"{axis}方向载荷曲线.txt"]
            data = subprocess.check_output([unrar, "p", "-inul", str(archive), member])
            tables[case][axis] = np.loadtxt(io.BytesIO(data), delimiter=",", skiprows=1)
            sources[case].append(f"{archive.relative_to(root)}::{member}")
    return tables, sources


def align_force(tables, times):
    aligned = np.zeros((len(times), 3), dtype=np.float64)
    expected_knots = np.arange(51) * .002
    for axis, table in tables.items():
        if (table.shape != (51, 2) or not np.isfinite(table).all()
                or not np.allclose(table[:, 0], expected_knots, rtol=0, atol=1e-10)):
            raise ValueError(f"{axis}: expected finite force knots on the 0–100 ms / 2 ms grid")
        aligned[:, AXES.index(axis)] = np.interp(times, table[:, 0], table[:, 1])
    return aligned


def load_cases(root: Path):
    """Return node IDs, displacement times, fields, aligned forces and data provenance."""
    root = Path(root).resolve()
    cache = root / "data/prepared_neural_time_v1"
    load_cache = root / "data/prepared_v1"
    names = tuple(CACHED_LOADS) + tuple(ARCHIVE_CASES)
    ids, times, snapshots = load_displacements(cache, names)
    tables, sources = archived_force(root)
    for case, name in CACHED_LOADS.items():
        path = load_cache / name / "loads.npz"
        tables[case] = cached_force(path)
        sources[case] = [str(path.relative_to(root))]
    xyz_1x = cached_force(load_cache / "XYZ/loads.npz")
    xyz_2x = tables["XYZForce_2x"]
    if not all(np.array_equal(xyz_1x[a], xyz_2x[a]) for a in AXES):
        raise ValueError("The original XYZ 1x/2x input-table duplication has changed")
    forces = {case: align_force(tables[case], times) for case in names}
    metadata = {
        "displacement_cache": str(cache.relative_to(root)),
        "nodes": int(ids.size),
        "frames": int(times.size),
        "displacement_step_s": .001,
        "force_table_step_s": .002,
        "force_axes": list(AXES),
        "force_alignment": "Piecewise linear on supplied force tables",
        "force_values": "Values as supplied; no multiplication based on the case label",
        "cases": {
            case: {
                "force_sources": sources[case],
                "supplied_axes": list(tables[case]),
                "abs_max_raw": np.max(np.abs(forces[case]), axis=0).tolist(),
            } for case in names
        },
        "excluded_cases": {
            "XYZForce_1x": "Same supplied XYZ force tables as XYZForce_2x, with different displacement."
        },
        "case_label_note": "The 2x and 2.5x names are source labels, and solver-side force scaling is unresolved.",
    }
    return ids, times, snapshots, forces, metadata
