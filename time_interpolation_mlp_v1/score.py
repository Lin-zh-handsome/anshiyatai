"""Evaluate all methods against the same held-out finite-element fields."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
import math

import numpy as np

from cache import TEST_INDICES, cached_dataset, save_json

METHODS = ("linear", "cubic", "mlp")
NEAR_ZERO = 1e-12


@dataclass
class ErrorTotals:
    node_count: int = 0
    squared: np.ndarray = field(default_factory=lambda: np.zeros(3))
    absolute: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vector_sum: float = 0.0
    truth_squared: float = 0.0
    max_vector: float = -1.0
    max_node_id: int = -1
    max_index: int = -1

    def add(self, other):
        self.node_count += other.node_count
        self.squared += other.squared
        self.absolute += other.absolute
        self.vector_sum += other.vector_sum
        self.truth_squared += other.truth_squared
        if other.max_vector > self.max_vector:
            self.max_vector = other.max_vector
            self.max_node_id = other.max_node_id
            self.max_index = other.max_index

    def metrics(self):
        n = self.node_count
        total_squared = float(self.squared.sum())
        relative = (math.sqrt(total_squared / self.truth_squared)
                    if self.truth_squared > n * NEAR_ZERO**2 else None)
        return dict(rmse=float(math.sqrt(total_squared / (3 * n))),
                    **{f"rmse_d{axis}": float(value) for axis, value in
                       zip("xyz", np.sqrt(self.squared / n))},
                    vector_rmse=float(math.sqrt(total_squared / n)),
                    mae=float(self.absolute.sum() / (3 * n)),
                    **{f"mae_d{axis}": float(value) for axis, value in zip("xyz", self.absolute / n)},
                    vector_mae=self.vector_sum / n, max_node_vector_error=self.max_vector,
                    max_error_node_id=self.max_node_id, max_error_index=self.max_index,
                    relative_l2=relative, truth_vector_rms=math.sqrt(self.truth_squared / n),
                    node_samples=n)


def frame_errors(error, truth_squared, ids, index):
    squared = error**2
    vector = np.sqrt(squared.sum(axis=1))
    worst = vector.argmax()
    return ErrorTotals(len(ids), squared.sum(axis=0), np.abs(error).sum(axis=0),
                       float(vector.sum()), truth_squared, float(vector[worst]),
                       int(ids[worst]), index)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(cache, predictions):
    ids, coords, times, cases = cached_dataset(cache)
    for name, expected in (("node_ids", ids), ("target_indices", TEST_INDICES),
                           ("target_times_s", times[TEST_INDICES])):
        if not np.array_equal(np.load(predictions / f"{name}.npy"), expected):
            raise ValueError(f"Prediction mapping mismatch: {name}")
    timing_data = [json.loads((predictions / file).read_text(encoding="utf-8"))
                   for file in ("baseline_timing.json", "mlp_timing.json")]
    timing = {(row["case"], row["method"], row["index"]): row
              for info in timing_data for row in info["predictions"]}
    results, case_totals, diagnostics = [], [], []
    overall = {method: ErrorTotals() for method in METHODS}
    for case in cases:
        reference = np.load(cache / case / "truth_odd.npy", mmap_mode="r")
        fields = {method: np.load(predictions / case / f"{method}.npy", mmap_mode="r")
                  for method in METHODS}
        expected_shape = (len(TEST_INDICES), len(ids), 3)
        if reference.shape != expected_shape or any(value.shape != expected_shape for value in fields.values()):
            raise ValueError(f"{case}: incomplete prediction or truth array")
        totals = {method: ErrorTotals() for method in METHODS}
        desired_squared, correction_squared, alignment = 0.0, 0.0, 0.0
        wins_linear, wins_cubic = 0, 0
        for row, index in enumerate(TEST_INDICES.tolist()):
            truth = reference[row].astype(np.float64)
            truth_squared = float(np.sum(truth**2))
            errors = {method: fields[method][row] - truth for method in METHODS}
            for method, error in errors.items():
                if not np.isfinite(error).all():
                    raise ValueError(f"{case} {method} {index}: nonfinite error")
                frame = frame_errors(error, truth_squared, ids, index)
                totals[method].add(frame)
                overall[method].add(frame)
                info = timing[case, method, index]
                results.append(dict(case=case, method=method, index=index, time_s=times[index],
                                    input_indices=" ".join(map(str, info["input_indices"])),
                                    inference_ms=info["inference_ms"], **frame.metrics()))
            desired = -errors["linear"]
            correction = errors["mlp"] + desired
            desired_squared += float(np.sum(desired**2))
            correction_squared += float(np.sum(correction**2))
            alignment += float(np.sum(desired * correction))
            wins_linear += np.sum(errors["mlp"]**2) < np.sum(errors["linear"]**2)
            wins_cubic += np.sum(errors["mlp"]**2) < np.sum(errors["cubic"]**2)
        diagnostics.append(dict(case=case, mlp_wins_vs_linear=int(wins_linear), mlp_wins_vs_cubic=int(wins_cubic),
                                correction_to_ideal_norm=math.sqrt(correction_squared / desired_squared)
                                if desired_squared else None,
                                correction_cosine=alignment / math.sqrt(desired_squared * correction_squared)
                                if desired_squared * correction_squared else None))
        for method in METHODS:
            case_totals.append(summary_row(case, method, totals[method], results))
        print(f"{case}: evaluated all {len(TEST_INDICES)} odd frames", flush=True)
    case_totals.extend(summary_row("ALL", method, overall[method], results) for method in METHODS)
    return results, case_totals, diagnostics, timing_data[0]["hardware"], timing_data[1]


def summary_row(case, method, totals, rows):
    selected = [row for row in rows if row["method"] == method and
                (case == "ALL" or row["case"] == case)]
    ms = [row["inference_ms"] for row in selected]
    return dict(case=case, method=method, frames=len(selected),
                inference_mean_ms=float(np.mean(ms)), inference_median_ms=float(np.median(ms)),
                inference_p95_ms=float(np.percentile(ms, 95)),
                near_zero_truth_frames=sum(row["relative_l2"] is None for row in selected),
                **totals.metrics())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--error-ms", type=int, default=73)
    parser.add_argument("--nodes", nargs="+", type=int, default=[56877, 173465])
    args = parser.parse_args()
    if args.error_ms not in TEST_INDICES:
        parser.error("--error-ms must be an odd held-out millisecond from 1 to 99")
    args.out.mkdir(parents=True, exist_ok=True)
    frame_rows, summary, diagnostics, baseline_hardware, model_info = evaluate(args.cache, args.predictions)
    write_csv(args.out / "per_frame.csv", frame_rows)
    write_csv(args.out / "summary.csv", summary)
    write_csv(args.out / "residual_diagnostics.csv", diagnostics)
    save_json(args.out / "summary.json", dict(summary=summary, diagnostics=diagnostics,
              baseline_hardware=baseline_hardware, mlp_hardware=model_info["hardware"],
              selected_epoch=model_info["checkpoint_epoch"], near_zero_truth_vector_rms=NEAR_ZERO))
    from reporting import report_and_plots
    report_and_plots(args, frame_rows, summary, diagnostics, baseline_hardware, model_info)


if __name__ == "__main__":
    main()
