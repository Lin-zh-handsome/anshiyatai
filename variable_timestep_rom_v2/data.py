"""Load and audit the aligned X-only finite-element trajectory."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT / "load_driven_rom_v1"), str(ROOT / "time_rom_v1")])
from load_data import align_force, cached_force
from field import fit_pod, project

OUT = ROOT / "outputs/variable_timestep_rom_v2"
CASE = "XForceOnly_1x"
LAG = 4
CHUNK = 4096
SEEN = (1, 2, 4, 6, 8)
UNSEEN = (3, 5, 7)
RANKS = (4, 8, 12, 16)
POD_TOLERANCE = .002
SEED = 20261113


def load_case():
    path = ROOT / "data/prepared_v1"
    case = path / "X"
    info = json.loads((case / "case_info.json").read_text(encoding="utf-8"))
    ids = np.load(path / "node_ids.npy")
    fields = np.load(case / "snapshots.npy", mmap_mode="r")
    times = np.asarray(info["times_s"])
    with (ROOT / "time_rom_v1/time_mapping.csv").open(encoding="utf-8-sig", newline="") as stream:
        mapped = np.array([float(row["time_s"]) for row in csv.DictReader(stream)])
    if (times.shape != (101,) or mapped.shape != times.shape
            or not np.allclose(times, np.arange(101) * .001, rtol=0, atol=1e-10)
            or not np.allclose(times, mapped, rtol=0, atol=1e-10)):
        raise ValueError("Case times and project time mapping disagree")
    if (ids.shape != (267801,) or not np.issubdtype(ids.dtype, np.integer)
            or len(np.unique(ids)) != len(ids) or fields.shape != (101, len(ids), 3)
            or fields.dtype != np.float32):
        raise ValueError("Snapshot dimensions, type, or node ID order are invalid")
    raw = cached_force(case / "loads.npz")
    if set(raw) != {"X"} or info["force_channels"] != ["X"]:
        raise ValueError("Expected the original X-only force table")
    force = align_force(raw, times)[:, 0]
    return ids, times, fields, force, raw["X"], info


def audit(ids, times, fields, force, raw, info):
    source = Path(info["sources"]["loads"][0]["path"])
    source_table = np.loadtxt(source, delimiter=",", skiprows=1)
    if not np.array_equal(raw, source_table):
        raise ValueError("Cached force differs from the original table")
    if not np.array_equal(force[::2], raw[:, 1]):
        raise ValueError("Aligned loads changed the source knots")
    if not all(np.isfinite(frame).all() for frame in fields):
        raise ValueError("Nonfinite displacement field")
    return dict(case=CASE, frames=len(times), nodes=len(ids), components=["DX","DY","DZ"],
                time_range_ms=[0,100], displacement_sampling_ms=1, original_force_knots=len(raw),
                original_force_sampling_ms=2, force_min=float(force.min()),
                force_max=float(force.max()), source_load=str(source.relative_to(ROOT)),
                aligned_force_method="Piecewise linear; all 51 source knots preserved",
                node_mapping="data/prepared_v1/node_ids.npy",
                displacement_source="data/prepared_v1/X/snapshots.npy",
                force_cache_exact_original_match=True, nonfinite_displacement_frames=0,
                units="Source displacement and force units, no conversion",
                coordinate_system="Source XYZ ordering, no transformation")


def windows(step, count=101):
    if not isinstance(step, int) or not 1 <= step <= 8 or count < LAG + step:
        raise ValueError("Expected step in 1–8 ms with a valid measured history")
    anchors = np.arange(LAG - 1, count - step)
    history = anchors[:, None] + np.arange(1 - LAG, 1)
    return history, anchors + step


def decode(q, basis, reference):
    rank, nodes, _ = basis.shape
    return (q @ basis.reshape(rank, -1)).reshape(-1, nodes, 3) + reference


def dump(path, value):
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def prepare(out):
    ids, times, fields, force, raw, info = load_case()
    dump(out / "data_audit.json", audit(ids, times, fields, force, raw, info))
    print(f"Audited {CASE}: {len(times)} frames, {len(ids)} nodes, {len(raw)} raw load knots", flush=True)
    basis, _, energy = fit_pod([fields], np.arange(len(times)), out, max(RANKS), CHUNK)
    scores = []
    reference = fields[0]
    truth_energy = sum(float(np.einsum("ij,ij->", field, field, dtype=np.float64)) for field in fields)
    for rank in RANKS:
        residual = float(energy[rank:].sum())
        scores.append(dict(rank=rank, projection_relative_l2=np.sqrt(residual / truth_energy).item(),
                           projection_rmse=np.sqrt(residual / fields.size).item()))
    qualified = [row["rank"] for row in scores
                 if row["projection_relative_l2"] <= POD_TOLERANCE and row["rank"] <= len(basis)]
    rank = qualified[0] if qualified else len(basis)
    with (out / "pod_ranks.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(scores)
    selected = basis[:rank]
    q = project(selected, fields, np.arange(len(times)), CHUNK)
    rms = np.sqrt(np.mean(q**2, axis=0))
    scale = np.maximum(rms, rms.max() * .02)
    np.save(out / "node_ids.npy", ids)
    np.save(out / "reference_field.npy", reference)
    np.save(out / "pod_coefficients.npy", q)
    np.save(out / "pod_eigenvalues.npy", energy)
    return q, force, dict(case=CASE, nodes=len(ids), frames=len(times), history_length=LAG, width=32,
                          step_scale_ms=8, seed=SEED, rank=int(rank), pod_fit_ms=[0,100],
                          pod_rank_candidates=scores, pod_tolerance_relative_l2=POD_TOLERANCE,
                          coefficient_scale=scale.tolist(), coefficient_rms_floor_fraction=.02,
                          force_scale=float(np.abs(force).max()), train_steps_ms=SEEN,
                          final_test_steps_ms=UNSEEN,
                          windows_by_step={dt: len(windows(dt)[1]) for dt in range(1,9)},
                          selection="Lowest macro-average normalized MSE on training spans only")
