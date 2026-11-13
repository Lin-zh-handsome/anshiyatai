"""Train fixed 1 ms first, then train and lock the variable-step model."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from data import OUT, SEED, SEEN, dump, prepare, windows
from model import ConditionalTCN


def fit(label, steps, q, force, out, epochs, device):
    torch.manual_seed(SEED)
    model = ConditionalTCN(q.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=.00002)
    bank = []
    for step in steps:
        h, y = (torch.as_tensor(array, device=device) for array in windows(step, len(q)))
        bank.append((q[h], h, y, q[y]))
    # Each complete epoch includes every legal window, weighted equally by span.
    history, indices, targets, truth = [torch.cat(items) for items in zip(*bank)]
    sizes = [len(group[0]) for group in bank]
    weights = torch.cat([q.new_full((n,), 1/(len(steps)*n)) for n in sizes])
    best = float("inf")
    path = out / f"{label}_best.pt"
    rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        mse = (model(history, indices, force, targets) - truth).square().mean(dim=1)
        loss = (mse * weights).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            squared = (model(history, indices, force, targets) - truth).square().mean(dim=1)
            step_mse = torch.stack([group.mean() for group in squared.split(sizes)])
            score = float(step_mse.mean())
        if not np.isfinite(score):
            raise ValueError("Nonfinite training monitor")
        row = dict(epoch=epoch, macro_normalized_mse=score, lr=optimizer.param_groups[0]["lr"],
                   **{f"mse_{step}ms": float(mse) for step,mse in zip(steps,step_mse)})
        rows.append(row)
        if score < best:
            best = score
            selected = dict(row)
            torch.save(model.state_dict(), path.with_suffix(".partial"))
            path.with_suffix(".partial").replace(path)
        if epoch == 1 or epoch % 250 == 0:
            print(f"{label} epoch={epoch} training normalized MSE={score:.8g}", flush=True)
        scheduler.step()
    with (out / f"{label}_training.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selected.update(epochs=epochs, training_steps=list(steps), samples_by_step=dict(zip(steps,sizes)),
                    parameters=sum(p.numel() for p in model.parameters()))
    print(f"{label} selected epoch {selected['epoch']}/{epochs}, MSE={best:.8g}", flush=True)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fixed-epochs", type=int, default=1000)
    parser.add_argument("--variable-epochs", type=int, default=1800)
    args = parser.parse_args()
    if min(args.fixed_epochs, args.variable_epochs) < 1:
        parser.error("Epoch budgets must be positive")
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "selection.json").exists():
        raise FileExistsError("A completed run exists; choose a new output directory")
    q, force, protocol = prepare(args.out)
    device = torch.device(args.device)
    protocol.update(device=str(device), fixed_epochs=args.fixed_epochs,
                    variable_epochs=args.variable_epochs, optimizer="AdamW",
                    learning_rate=.001, min_lr=.00002, gradient_clip_norm=1.,
                    scheduler="cosine annealing", training_weight_decay=.0001,
                    hardware=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU")
    dump(args.out / "protocol.json", protocol)
    print(f"POD selected rank {protocol['rank']} with all 101 frames; device {protocol['hardware']}", flush=True)
    state = torch.as_tensor(q / protocol["coefficient_scale"], device=device, dtype=torch.float32)
    drive = torch.as_tensor(force / protocol["force_scale"], device=device, dtype=torch.float32)
    fixed = fit("fixed", (1,), state, drive, args.out, args.fixed_epochs, device)
    variable = fit("variable", SEEN, state, drive, args.out, args.variable_epochs, device)
    dump(args.out / "selection.json", dict(fixed=fixed, variable=variable))


if __name__ == "__main__":
    main()
