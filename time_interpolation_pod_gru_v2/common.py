"""Shared paths and the fixed six-case target split."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "time_interpolation_mlp_v1"))
from cache import cached_dataset, save_json
from predict import cubic_window

CACHE = ROOT / "data/prepared_neural_time_v1"
OUT = ROOT / "outputs/neural_time_interpolation_v2"
V1 = ROOT / "outputs/neural_time_interpolation_v1"
RANKS = (8, 16, 32, 64)
SPLITS = {
    "train": np.r_[1:20:2, 41:60:2, 81:100:2],
    "validation": np.arange(21, 40, 2),
    "test": np.arange(61, 80, 2),
}
SEED = 20261111


def windows(indices, times):
    slots = np.stack([cubic_window(int(index), times)[0] for index in indices])
    times_context = times[2 * slots]
    targets = times[indices]
    offsets = (times_context - targets[:, None]) / .002
    intervals = np.diff(times_context, axis=1)
    if (not np.allclose(intervals, .002, rtol=0, atol=1e-10)
            or not np.all((offsets[:, 0] < 0) & (offsets[:, -1] > 0))):
        raise ValueError("The context must bracket the target using 2 ms spacing")
    nearest_left = np.max(np.where(offsets < 0, offsets, -np.inf), axis=1)
    nearest_right = np.min(np.where(offsets > 0, offsets, np.inf), axis=1)
    query = np.column_stack((targets / .1, -nearest_left, nearest_right,
                             nearest_right - nearest_left))
    return slots, offsets.astype(np.float32), query.astype(np.float32)


def displacement_maps(cache, cases, parity, nodes):
    n_frames = 51 if parity == "known_even" else 50
    result = []
    for case in cases:
        field = np.load(cache / case / f"{parity}.npy", mmap_mode="r")
        if field.shape != (n_frames, nodes, 3) or field.dtype != np.float32:
            raise ValueError(f"{case}: unexpected {parity} cache")
        result.append(field.reshape(n_frames, -1))
    return result


def protocol(cache, out):
    ids, _, times, cases = cached_dataset(cache)
    indices = np.concatenate(list(SPLITS.values()))
    if not np.array_equal(np.sort(indices), np.arange(1, 100, 2)):
        raise ValueError("The odd target partition is incomplete or overlaps")
    if len(cases) != 6 or len(np.unique(ids)) != len(ids) or not np.allclose(
            times, np.arange(101) * .001, rtol=0, atol=1e-10):
        raise ValueError("Unexpected six-case cache mapping")
    descriptions = {}
    for split, target in SPLITS.items():
        slots, _, _ = windows(target, times)
        descriptions[split] = [dict(target_index=int(index), target_time_s=float(times[index]),
                                    known_indices=(2 * context).tolist(),
                                    known_times_s=times[2 * context].tolist())
                               for index, context in zip(target, slots)]
    save_json(out / "protocol.json", dict(seed=SEED, cases=list(cases), node_count=len(ids),
              test_target_range_ms=[61, 79], pod_fit_targets=SPLITS["train"].tolist(),
              coefficient_normalization_targets=SPLITS["train"].tolist(),
              same_protocol_for_all_cases=True, frames=descriptions,
              selection="Minimum full-field validation RMSE; one fixed seed per rank",
              nominal_input_step_s=.002, loads_used=False))
    return ids, times, cases
