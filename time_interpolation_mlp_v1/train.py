"""Fit and select a checkpoint using only even frames."""
from __future__ import annotations

import argparse
import csv
from itertools import islice
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cache import cached_dataset, save_json
from model import ResidualMLP, inputs, load_checkpoint


TRAIN_BLOCKS = ((0, 16), (30, 46), (60, 76), (90, 100))
VALIDATION_BLOCKS = ((18, 28), (48, 58), (78, 88))
WINDOW_STEPS = (4, 6, 8)


def windows(blocks):
    return [(left, target, left + span)
            for start, end in blocks
            for left in range(start, end, 2)
            for span in WINDOW_STEPS if left + span <= end
            for target in range(left + 2, left + span, 2)]


def normalization(coordinates, fields, training_windows):
    frames = sorted({index // 2 for window in training_windows for index in window})
    squares, count = 0.0, 0
    for field in fields:
        for frame in frames:
            value = field[frame].astype(np.float64)
            squares += np.sum(value**2)
            count += value.size
    coordinate_std = coordinates.std(axis=0)
    return dict(coordinate_mean=coordinates.mean(axis=0).tolist(),
                coordinate_std=np.where(coordinate_std > 0, coordinate_std, 1.0).tolist(),
                displacement_scale=math.sqrt(squares / count) or 1.0)


def batches(fields, coordinates, times, window_list, batch_windows, nodes, rng):
    pairs = [(case, window) for case in range(len(fields)) for window in window_list]
    order = iter(rng.permutation(len(pairs)))
    while group := list(islice(order, batch_windows)):
        data, target = [], []
        for index in group:
            case, (left, mid, right) = pairs[index]
            field = fields[case]
            pick = rng.integers(len(coordinates), size=nodes)
            dt = times[right] - times[left]
            data.append(inputs(coordinates[pick], field[left // 2, pick], field[right // 2, pick],
                               (times[mid] - times[left]) / dt, dt))
            target.append(field[mid // 2, pick])
        yield np.concatenate(data), np.concatenate(target)


def run_epoch(model, batch_iter, device, scale, amp_dtype, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    sums = torch.zeros(2, dtype=torch.float64, device=device)
    total = 0
    with torch.set_grad_enabled(training):
        for data, target in batch_iter:
            data = torch.from_numpy(data).to(device)
            target = torch.from_numpy(target).to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=training and device.type == "cuda"):
                predicted = model(data)
            loss = ((predicted - target) / scale).square().mean()
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                alpha = data[:, 9:10]
                linear = (1 - alpha) * data[:, 3:6] + alpha * data[:, 6:9]
                sums[1] += ((linear - target) / scale).square().sum()
            sums[0] += loss.detach() * target.numel()
            total += target.numel()
    mse, linear_mse = (sums / total).tolist()
    if not math.isfinite(mse):
        raise RuntimeError("Training or validation produced a nonfinite loss")
    return mse, linear_mse


def write_history(path, history):
    partial = path.with_suffix(".csv.partial")
    with partial.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    partial.replace(path)


def save_checkpoint(path, checkpoint):
    partial = path.with_suffix(".pt.partial")
    torch.save(checkpoint, partial)
    partial.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=32)
    p.add_argument("--nodes-per-window", type=int, default=8192)
    p.add_argument("--validation-nodes", type=int, default=8192)
    p.add_argument("--batch-windows", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=20261111)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--gpus", type=int, choices=(1, 2), default=1)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if min(args.epochs, args.nodes_per_window, args.validation_nodes, args.batch_windows) <= 0:
        p.error("epoch and batch counts must be positive")
    device = torch.device(args.device)
    if args.gpus == 2 and device.type != "cuda":
        p.error("two GPUs require CUDA")
    if device.type == "cuda" and torch.cuda.device_count() < args.gpus:
        p.error("insufficient visible GPUs")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    _, coords, times, cases = cached_dataset(args.cache)
    fields = [np.load(args.cache / case / "known_even.npy", mmap_mode="r") for case in cases]
    train_windows, val_windows = windows(TRAIN_BLOCKS), windows(VALIDATION_BLOCKS)
    train_indices = {index for w in train_windows for index in w}
    validation_indices = {index for w in val_windows for index in w}
    if train_indices & validation_indices or any(index % 2 for index in train_indices | validation_indices):
        raise ValueError("Overlapping or odd indices in the training and validation protocol")
    protocol = dict(cases=list(cases), seed=args.seed, training_windows=[list(w) for w in train_windows],
                    validation_windows=[list(w) for w in val_windows],
                    training_indices=sorted(train_indices), validation_indices=sorted(validation_indices),
                    test_indices=list(range(1, 101, 2)), learning_rate=args.lr,
                    nodes_per_window=args.nodes_per_window, validation_nodes=args.validation_nodes,
                    batch_windows=args.batch_windows, gpu_count=args.gpus, device=args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    latest = args.out / "latest.pt"
    if latest.exists() != args.resume:
        p.error("use --resume for an existing run, or select a new output directory")
    state = load_checkpoint(latest) if args.resume else None
    if state and state["protocol"] != protocol:
        p.error("resume configuration differs from saved training protocol")
    stats = state["normalization"] if state else normalization(coords, fields, train_windows)
    model = ResidualMLP(**stats).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bf16 = device.type == "cuda" and all(
        torch.cuda.get_device_capability(index)[0] >= 8 for index in range(args.gpus))
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    scale_gradients = device.type == "cuda" and not bf16
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=scale_gradients)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=scale_gradients)
    if state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
    network = nn.DataParallel(model, device_ids=list(range(args.gpus))) if args.gpus == 2 else model
    history = state["history"] if state else []
    best = min((row["validation_mse"] for row in history), default=math.inf)
    scale = stats["displacement_scale"]
    save_json(args.out / "protocol.json", {**protocol, "normalization": stats})
    gpu_names = [torch.cuda.get_device_name(index) for index in range(args.gpus)] if device.type == "cuda" else []
    print(f"{len(cases)} cases, {len(train_windows)} train / {len(val_windows)} validation "
          f"windows per case; {gpu_names or ['CPU']}; AMP={amp_dtype if gpu_names else 'off'}", flush=True)
    for epoch in range(len(history) + 1, args.epochs + 1):
        train = batches(fields, coords, times, train_windows, args.batch_windows,
                        args.nodes_per_window, np.random.default_rng([args.seed, epoch]))
        validation = batches(fields, coords, times, val_windows, args.batch_windows,
                             args.validation_nodes, np.random.default_rng([args.seed, 0]))
        train_mse, _ = run_epoch(network, train, device, scale, amp_dtype, optimizer, scaler)
        val_mse, linear_mse = run_epoch(network, validation, device, scale, amp_dtype)
        history.append(dict(epoch=epoch, train_mse=train_mse, validation_mse=val_mse,
                            validation_linear_mse=linear_mse))
        checkpoint = dict(epoch=epoch, model=model.state_dict(), normalization=stats,
                          optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                          protocol=protocol, history=history)
        if val_mse < best:
            best = val_mse
            save_checkpoint(args.out / "best.pt", checkpoint)
        save_checkpoint(latest, checkpoint)
        write_history(args.out / "history.csv", history)
        print(f"epoch={epoch:02d} train={train_mse:.6e} val={val_mse:.6e} "
              f"val_linear={linear_mse:.6e} best={best:.6e}", flush=True)


if __name__ == "__main__":
    main()
