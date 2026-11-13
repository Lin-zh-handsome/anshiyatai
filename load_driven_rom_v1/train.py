"""Fit a single load-driven displacement ROM, selecting by whole-case validation."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from dynamics import Dynamics, windows
from load_data import load_cases

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "time_rom_v1"))
from field import fit_pod, project
from history_model import ridge, rollout, rollout_ridge

TRAIN = ("XForceOnly_1x", "YForceOnly_1x", "ZForceOnly_1x")
VAL = "YXZForce_2.5x"
TEST = "XYZForce_2x"
SEED = 20261113
LAG = 8
WIDTH = 32
NODE_CHUNK = 8192
CANDIDATE_RANKS = (4, 8, 12, 16)
OUTPUT = ROOT / "outputs/load_driven_rom_v1"


def dump(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def fit_dynamics(q, forces, val_q, val_force, device, out, epochs):
    scale = np.sqrt(np.mean(q**2, axis=(0, 1)))
    scale = np.maximum(scale, scale.max() * .02)
    force_scale = np.max(np.abs(forces), axis=(0, 1))
    if np.any(force_scale <= 0):
        raise ValueError("All force axes must be represented in training")
    train_z = q / scale
    train_force = forces / force_scale
    val_z = val_q / scale
    val_drive = val_force / force_scale
    rank = q.shape[-1]
    model = Dynamics(rank, LAG, WIDTH).to(device)
    criterion_weight = torch.as_tensor(scale / scale.max(), device=device, dtype=torch.float32)
    val_history = torch.as_tensor(val_z[None, :LAG], device=device, dtype=torch.float32)
    val_input = torch.as_tensor(val_drive[None], device=device, dtype=torch.float32)
    val_horizon = len(val_q) - LAG

    def validation_linear(weight):
        pred = rollout_ridge(weight, val_z[None, :LAG], val_drive[None], val_horizon)[0] * scale
        return np.linalg.norm(pred - val_q[LAG:]) / np.linalg.norm(val_q[LAG:])

    linear_options = [(float(validation_linear(w)), p, w) for p in (1e-4, 1e-3, 1e-2)
                      for w in (ridge(train_z, train_force, LAG, p),)]
    _, penalty, weight = min(linear_options, key=lambda item: item[0])
    with torch.no_grad():
        model.linear.weight.copy_(torch.as_tensor(weight.T, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    phase = [(max(1, epochs // 4), 1, 1e-3),
             (max(1, epochs // 3), 8, 5e-4),
             (max(1, epochs - epochs // 4 - epochs // 3), 24, 2e-4)]
    best = float("inf")
    history = []
    cursor = 0

    def check(stage, loss):
        nonlocal best
        model.eval()
        with torch.inference_mode():
            pred = rollout(model, val_history, val_input, val_horizon).cpu().numpy()[0] * scale
        error = float(np.linalg.norm(pred - val_q[LAG:]) / np.linalg.norm(val_q[LAG:]))
        if not np.isfinite(error):
            raise ValueError(f"Nonfinite validation prediction at epoch {cursor}")
        improved = error < best
        if improved:
            best = error
            target = out / "model.tmp"
            torch.save(dict(model=model.state_dict(), epoch=cursor, validation_relative_l2=error,
                            rank=rank, lag=LAG, width=WIDTH, scale=scale.tolist(),
                            force_scale=force_scale.tolist(), ridge_weight=weight.tolist(),
                            ridge_penalty=penalty), target)
            target.replace(out / "model.pt")
        history.append(dict(epoch=cursor, stage=stage, train_loss=loss, validation_relative_l2=error))
        print(f"epoch={cursor:03d} horizon={stage:2d} loss={loss} validation_latent_relL2={error:.4%}"
              + (" selected" if improved else ""), flush=True)

    # Choose among trained networks. The ridge initialization remains a separate baseline.
    rng = np.random.default_rng(SEED)
    for duration, horizon, lr in phase:
        for group in opt.param_groups:
            group["lr"] = lr
        h, f, target = windows(train_z, train_force, LAG, horizon)
        bank = [torch.as_tensor(arr.reshape(-1, *arr.shape[2:]), device=device, dtype=torch.float32)
                for arr in (h, f, target)]
        for ep in range(duration):
            model.train()
            batch_losses = []
            for batch in np.array_split(rng.permutation(len(bank[0])), max(1, (len(bank[0]) + 47)//48)):
                states, drive, targets = (arr[batch] for arr in bank)
                pred = rollout(model, states, drive, horizon)
                sq = ((pred - targets) * criterion_weight).square()
                loss = .4 * sq[:, 0].mean() + .6 * sq.mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                batch_losses.append(loss.detach().item())
            cursor += 1
            if cursor == 1 or ep + 1 == duration or (ep + 1) % 10 == 0:
                check(horizon, float(np.mean(batch_losses)))
    with (out / "training.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=90)
    args = p.parse_args()
    if args.epochs < 4:
        p.error("epochs must be at least 4")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    ids, times, snapshots, forces, metadata = load_cases(ROOT)
    indices = np.arange(len(times))
    tic = perf_counter()
    basis, _, values = fit_pod([snapshots[k] for k in TRAIN], indices, args.output,
                                max(CANDIDATE_RANKS), NODE_CHUNK)
    ranks = [k for k in CANDIDATE_RANKS if k <= len(basis)]
    if not ranks or ranks[-1] != len(basis):
        ranks.append(len(basis))
    remaining = {k: float(np.sqrt(values[k:].sum() / values.sum())) for k in ranks}
    rank = next((k for k in ranks if remaining[k] <= 0.002), ranks[-1])
    np.save(args.output / "node_ids.npy", ids)
    np.save(args.output / "times_s.npy", times)
    np.save(args.output / "pod_eigenvalues.npy", values)
    q = np.stack([project(basis[:rank], snapshots[k], indices, NODE_CHUNK) for k in TRAIN])
    val_q = project(basis[:rank], snapshots[VAL], indices, NODE_CHUNK)
    dump(args.output / "protocol.json", dict(train=TRAIN, validation=VAL, test=TEST,
         lag=LAG, first_forecast_index=LAG, rank=rank, available_basis_rank=len(basis),
         pod_train_relative_l2_by_rank=remaining, pod_selection_train_relative_l2_target=.002,
         model_selection="Minimum uninterrupted validation latent relative L2 from first eight true fields",
         rank_selection="First candidate meeting train displacement-increment projection relative L2 target",
         loss="0.4 * first-step plus 0.6 * rollout MSE, mode weighted by train RMS",
         seed=SEED, device=args.device, epochs=args.epochs, pod_and_projection_s=perf_counter()-tic,
         data=metadata))
    print(f"POD rank {rank}, training reconstruction relL2={remaining[rank]:.4%}; train={TRAIN}; "
          f"validation={VAL}; held-out={TEST}; device={args.device}", flush=True)
    fit_dynamics(q, np.stack([forces[k] for k in TRAIN]), val_q, forces[VAL],
                 torch.device(args.device), args.output, args.epochs)


if __name__ == "__main__":
    main()
