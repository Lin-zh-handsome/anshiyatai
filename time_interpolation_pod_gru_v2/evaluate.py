"""Select the model on validation only, then score the four methods on identical test fields."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from common import CACHE, OUT, RANKS, SPLITS, V1, cached_dataset, displacement_maps, save_json, windows
from predict_gru import Predictor
from score import ErrorTotals, frame_errors, summary_row, write_csv


METHODS = ("linear", "cubic", "mlp", "pod_bigru")


def select(out):
    options = [json.loads((out / "training" / f"rank{rank}" / "result.json").read_text()) for rank in RANKS]
    best = min(options, key=lambda row: (row["validation_field_rmse"], row["rank"]))
    save_json(out / "selection.json",
              dict(**best, criterion="Minimum validation full-field RMSE", candidates=options))
    write_csv(out / "rank_selection.csv", options)
    return best


def load_baselines(v1, ids, times):
    predictions = v1 / "predictions"
    indices = np.load(predictions / "target_indices.npy")
    if (not np.array_equal(np.load(predictions / "node_ids.npy"), ids)
            or not np.array_equal(np.load(predictions / "target_times_s.npy"), times[indices])):
        raise ValueError("V1 prediction node or time mapping disagrees with the cache")
    if not set(SPLITS["test"]).issubset(indices.tolist()):
        raise ValueError("V1 predictions do not cover the V2 test targets")
    with (v1 / "per_frame.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = {(row["case"], row["method"], int(row["index"])): row for row in csv.DictReader(f)}
    return predictions, {int(index): slot for slot, index in enumerate(indices)}, rows


def frame_diagnostics(predictor, truth, oracle, z, target_z, frame):
    error = oracle - truth
    floor = float(np.einsum("i,i->", error, error))
    coefficient_error = z.astype(np.float64) - target_z
    coefficient_sse = float(coefficient_error @ coefficient_error)
    physical = float(frame.squared.sum())
    return dict(pod_floor_sse=floor, temporal_coefficient_sse=coefficient_sse,
                field_sse=physical,
                pod_projection_rmse=math.sqrt(floor / len(truth)),
                latent_rmse=math.sqrt(coefficient_sse / len(z)),
                latent_normalized_mse=float(np.mean((coefficient_error / predictor.scale)**2)),
                temporal_field_rmse=math.sqrt(coefficient_sse / len(truth)),
                closure_relative=abs(physical - floor - coefficient_sse) / max(physical, 1e-30))


def evaluate(cache, out, v1, device, field_case, field_ms):
    selected = select(out)
    rank = selected["rank"]
    print(f"Selected rank {rank}, epoch {selected['selected_epoch']} using validation", flush=True)
    predictor = Predictor(out, rank, device)
    ids, coords, times, cases = cached_dataset(cache)
    baseline_path, baseline_slots, old_rows = load_baselines(v1, ids, times)
    fields = displacement_maps(cache, cases, "known_even", len(ids))
    truth = displacement_maps(cache, cases, "truth_odd", len(ids))
    targets = SPLITS["test"]
    slots, offsets, query = windows(targets, times)
    output = out / "predictions"
    output.mkdir(exist_ok=True)
    np.save(output / "node_ids.npy", ids)
    np.save(output / "target_indices.npy", targets)
    np.save(output / "target_times_s.npy", times[targets])
    rows, summary, latent, diagnostics, components = [], [], [], [], []
    totals = {method: ErrorTotals() for method in METHODS}
    oracle_totals = ErrorTotals()
    node_ids = [56877, 173465]
    node_slots = [int(np.flatnonzero(ids == node)[0]) for node in node_ids]
    node_traces = []
    field_image = None
    predictor.predict(fields[0], slots[0], offsets[0], query[0])
    for case_idx, case in enumerate(cases):
        per_case = {method: ErrorTotals() for method in METHODS}
        case_oracle = ErrorTotals()
        baselines = {method: np.load(baseline_path / case / f"{method}.npy", mmap_mode="r")
                     for method in METHODS[:-1]}
        if any(field.shape[1:] != (len(ids), 3) for field in baselines.values()):
            raise ValueError(f"{case}: V1 field shape disagrees with cache")
        file = output / f"{case}.npy"
        partial = file.with_suffix(".partial.npy")
        result = np.lib.format.open_memmap(partial, mode="w+", dtype="float32",
                                         shape=(len(targets), len(ids), 3))
        for pos, original in enumerate(targets.tolist()):
            reference = truth[case_idx][original // 2].astype(np.float64)
            norm2 = float(reference @ reference)
            field, z_pred, elapsed = predictor.predict(fields[case_idx], slots[pos], offsets[pos], query[pos])
            if not np.isfinite(field).all():
                raise ValueError(f"{case}: nonfinite prediction at {original} ms")
            result[pos] = field.reshape(len(ids), 3)
            full = {method: baselines[method][baseline_slots[original]] for method in METHODS[:-1]}
            full["pod_bigru"] = result[pos]
            errors = {}
            for method, predicted in full.items():
                diff = predicted - reference.reshape(len(ids), 3)
                errors[method] = frame_errors(diff, norm2, ids, original)
                info = old_rows.get((case, method, original))
                per_case[method].add(errors[method])
                totals[method].add(errors[method])
                rows.append(dict(case=case, method=method, index=original, time_s=float(times[original]),
                                 input_indices=(info["input_indices"] if info else
                                                " ".join(map(str, 2 * slots[pos]))),
                                 inference_ms=float(info["inference_ms"]) if info else elapsed,
                                 **errors[method].metrics()))
            z_true = predictor.project(reference)
            oracle = predictor.reconstruct(z_true)
            oracle_error = frame_errors((oracle - reference).reshape(len(ids), 3), norm2, ids, original)
            case_oracle.add(oracle_error)
            oracle_totals.add(oracle_error)
            diagnostic = frame_diagnostics(predictor, reference, oracle, z_pred, z_true, errors["pod_bigru"])
            diagnostics.append(dict(case=case, index=original, time_s=float(times[original]), **diagnostic))
            latent.extend(dict(case=case, index=original, time_s=float(times[original]),
                               mode=mode + 1, true=float(z_true[mode]), predicted=float(z_pred[mode]))
                          for mode in range(rank))
            for nid, slot in zip(node_ids, node_slots):
                for method, values in [("truth", reference.reshape(len(ids), 3)), *full.items()]:
                    dx, dy, dz = values[slot]
                    node_traces.append(dict(case=case, index=original, node_id=nid, method=method,
                                            dx=float(dx), dy=float(dy), dz=float(dz)))
            if case == field_case and original == field_ms:
                field_image = [(pred - reference.reshape(len(ids), 3)) for pred in
                               (oracle.reshape(len(ids), 3), full["pod_bigru"], full["cubic"])]
        result.flush()
        del result, full
        partial.replace(file)
        summary.extend(summary_row(case, method, per_case[method], rows) for method in METHODS)
        components.append(component_summary(case, case_oracle, diagnostics, rank))
        print(f"{case}: scored {len(targets)} shared test targets", flush=True)
    summary.extend(summary_row("ALL", method, totals[method], rows) for method in METHODS)
    components.append(component_summary("ALL", oracle_totals, diagnostics, rank))
    write_csv(out / "summary.csv", summary)
    write_csv(out / "per_frame.csv", rows)
    write_csv(out / "latent_coefficients.csv", latent)
    write_csv(out / "error_decomposition.csv", components)
    write_csv(out / "per_frame_decomposition.csv", diagnostics)
    write_csv(out / "key_nodes.csv", node_traces)
    save_json(out / "summary.json", dict(methods=summary, decomposition=components,
              selected=selected, test_indices=targets.tolist(),
              timing=dict(baselines="Original per-frame timings for the exact reused V1 predictions",
                          pod_bigru="Four full input frames, CPU projection, neural model, full reconstruction",
                          model_device=str(device))))
    if field_image is None:
        raise ValueError("Missing requested error-field image")
    from figures import report
    report(out, selected, summary, components, rows, latent, node_traces, coords, field_image, field_case, field_ms)


def component_summary(case, oracle, diagnostics, rank):
    values = [row for row in diagnostics if case == "ALL" or row["case"] == case]
    sse = sum(row["field_sse"] for row in values)
    pod_sse = sum(row["pod_floor_sse"] for row in values)
    latent_sse = sum(row["temporal_coefficient_sse"] for row in values)
    field_metrics = oracle.metrics()
    return dict(case=case, pod_projection_rmse=field_metrics["rmse"],
                pod_projection_relative_l2=field_metrics["relative_l2"],
                temporal_field_rmse=math.sqrt(latent_sse / (oracle.node_count * 3)),
                latent_rmse=math.sqrt(latent_sse / (len(values) * rank)),
                latent_normalized_mse=float(np.mean([row["latent_normalized_mse"] for row in values])),
                pod_sse_fraction=pod_sse / sse, temporal_sse_fraction=latent_sse / sse,
                closure_relative=abs(sse - pod_sse - latent_sse) / sse)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=CACHE)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--v1", type=Path, default=V1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--field-case", default="XYZForce_1x")
    p.add_argument("--field-ms", type=int, default=73)
    args = p.parse_args()
    torch.set_num_threads(1)
    evaluate(args.cache, args.out, args.v1, args.device, args.field_case, args.field_ms)
