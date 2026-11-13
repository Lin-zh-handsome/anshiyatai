"""Compare the trained model and load-driven baselines on a whole held-out case."""
import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from metrics import evaluate
from predict import ROM
from train import NODE_CHUNK, OUTPUT, ROOT, TEST, VAL, dump, load_cases, project


def diagnostic(rom, case, ids, times, snapshots, drive, out):
    steps = np.arange(rom.lag, len(times))
    tic = perf_counter()
    history = rom.encode_history(snapshots)
    encode_s = perf_counter() - tic
    # The predictor only sees its initialization history and the aligned load trace.
    start = perf_counter()
    tcn = rom.run_latent(history, drive)
    rollout_s = perf_counter() - start
    linear = rom.run_linear(history, drive)
    no_force = rom.run_latent(history, np.zeros_like(drive))
    load_sensitivity = float(np.linalg.norm(tcn - no_force) / max(np.linalg.norm(tcn), 1e-12))

    # Holdout displacement is read only after forecast generation, for scoring.
    oracle = project(rom.basis, snapshots, steps, NODE_CHUNK)
    true_q = np.concatenate((history, oracle))
    history_rows = steps[:, None] - rom.lag + np.arange(rom.lag)
    drive_rows = steps[:, None] - rom.lag + np.arange(rom.lag + 1)
    with torch.inference_mode():
        one = rom.model(
            torch.as_tensor(true_q[history_rows] / rom.scale, dtype=torch.float32, device=rom.device),
            torch.as_tensor(drive[drive_rows] / rom.force_scale, dtype=torch.float32, device=rom.device)
        ).cpu().numpy() * rom.scale
    result = evaluate(out, case, ids, times, snapshots, rom.basis,
                      dict(tcn=tcn, linear=linear,
                           persistence=np.broadcast_to(history[-1], tcn.shape),
                           pod_projection=oracle),
                      steps,
                      dict(history_indices=list(range(rom.lag)), encode_s=encode_s,
                           tcn_recursive_forecast_s=rollout_s, epoch=rom.epoch,
                           single_step_with_measured_history_latent_relative_l2=
                           float(np.linalg.norm(one - oracle) / np.linalg.norm(oracle)),
                           zero_load_counterfactual_relative_latent_change=load_sensitivity))
    methods = result["methods"]
    result["diagnostics"]["encode_rollout_reconstruct_s"] = (
        encode_s + rollout_s + methods["tcn"]["field_reconstruction_s"])
    with (out / case / "per_frame.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["method"] == "tcn"]
    result["diagnostics"]["rmse_at_horizon_ms"] = {
        str(horizon): float(rows[horizon - 1]["rmse"]) for horizon in (1, 10, 20, 50, len(steps))
    }
    dump(out / case / "evaluation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(2)
    ids, times, fields, forces, _ = load_cases(ROOT)
    rom = ROM(args.model_dir, args.device)
    if not np.array_equal(ids, rom.ids):
        raise ValueError("Node IDs disagree with the trained POD basis")
    results = {}
    for case in (VAL, TEST):
        results[case] = diagnostic(rom, case, ids, times, fields[case], forces[case], args.model_dir)
        print(case, {k: f"{v['relative_l2']:.4%}" for k, v in results[case]["methods"].items()}, flush=True)
    dump(args.model_dir / "summary.json", results)


if __name__ == "__main__":
    main()
