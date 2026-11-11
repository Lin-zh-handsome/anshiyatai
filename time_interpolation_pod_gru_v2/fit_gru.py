"""Train a small conditional bidirectional GRU in POD coefficient space."""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common import CACHE, OUT, RANKS, SEED, SPLITS, cached_dataset, displacement_maps, save_json, windows
from model import load_checkpoint


class ConditionalBiGRU(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.frame = nn.Sequential(nn.Linear(rank + 1, 64), nn.GELU())
        self.gru = nn.GRU(64, 64, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(128 + 4, 128), nn.GELU(), nn.Linear(128, rank))

    def forward(self, sequence, query):
        embedding = self.frame(sequence)
        # Use full precision for the recurrent kernel on older CUDA/cuDNN builds.
        with torch.cuda.amp.autocast(enabled=False):
            _, state = self.gru(embedding.float())
        state = state.transpose(0, 1).reshape(len(sequence), 128)
        return self.head(torch.cat((state, query), dim=1)).float()


def features(z, slots, offsets, query, center, scale):
    z = (z[slots] - center) / scale
    return np.concatenate((z, offsets[..., None]), axis=-1).astype(np.float32), query


def arrays(pod, times, split, rank, center, scale):
    known = np.load(pod / "known_z.npy")[..., :rank]
    target = np.load(pod / f"{split}_z.npy")[..., :rank]
    slots, offsets, query = windows(SPLITS[split], times)
    sequence = np.concatenate([features(case, slots, offsets, query, center, scale)[0]
                               for case in known])
    normalized = ((target.reshape(-1, rank) - center) / scale).astype(np.float32)
    return sequence, np.tile(query, (len(known), 1)), normalized


def history_csv(path, history):
    partial = path.with_suffix(".partial")
    with partial.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    partial.replace(path)


def save_checkpoint(path, checkpoint):
    partial = path.with_suffix(".partial")
    torch.save(checkpoint, partial)
    partial.replace(path)


def load_model(path, device):
    checkpoint = load_checkpoint(path)
    model = ConditionalBiGRU(checkpoint["rank"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint


@torch.no_grad()
def confirm_full_field(model, checkpoint, cache, out, directory, device):
    rank = checkpoint["rank"]
    ids, _, times, cases = cached_dataset(cache)
    pod = out / "pod"
    center = np.asarray(checkpoint["center"], dtype=np.float32)
    scale = np.asarray(checkpoint["scale"], dtype=np.float32)
    sequence, query, _ = arrays(pod, times, "validation", rank, center, scale)
    prediction = model(torch.from_numpy(sequence).to(device), torch.from_numpy(query).to(device))
    prediction = prediction.cpu().numpy() * scale + center
    mean = np.load(pod / "mean.npy", mmap_mode="r")
    basis = np.load(pod / "basis.npy", mmap_mode="r")[:rank]
    truth = displacement_maps(cache, cases, "truth_odd", len(ids))
    sse = 0.
    cursor = 0
    for case in truth:
        for frame in SPLITS["validation"]:
            error = (prediction[cursor] @ basis + mean).astype(np.float64) - case[frame // 2]
            sse += np.dot(error, error)
            cursor += 1
    measured = math.sqrt(sse / (cursor * len(mean)))
    saved = checkpoint["validation_field_rmse"]
    if not math.isclose(measured, saved, rel_tol=1e-4, abs_tol=1e-7):
        raise RuntimeError(f"Validation field metric mismatch: {measured} vs {saved}")
    save_json(directory / "validation_full_field.json",
              dict(frames=cursor, rmse=measured, coefficient_space_rmse=saved))
    print(f"Rank {rank} validation closed loop: {cursor} full fields, RMSE={measured:.6g}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=CACHE)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--rank", type=int, choices=RANKS, required=True)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=45)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--confirm-full-field", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    device = torch.device(args.device)
    _, _, times, _ = cached_dataset(args.cache)
    pod = args.out / "pod"
    directory = args.out / "training" / f"rank{args.rank}"
    directory.mkdir(parents=True, exist_ok=True)
    rank = args.rank
    z = np.load(pod / "train_z.npy")[..., :rank].reshape(-1, rank)
    center = z.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = z.std(axis=0, dtype=np.float64)
    scale = np.maximum(std, std.max() * 1e-8).astype(np.float32)
    np.savez(directory / "normalization.npz", center=center, scale=scale)
    stats = dict(rank=rank, seed=SEED, lr=args.lr, batch=args.batch,
                 center=center.tolist(), scale=scale.tolist(), device=args.device)
    save_json(directory / "config.json", stats)
    training = [torch.from_numpy(value).to(device) for value in
                arrays(pod, times, "train", rank, center, scale)]
    validation = [torch.from_numpy(value).to(device) for value in
                  arrays(pod, times, "validation", rank, center, scale)]
    validation_floor = float(np.load(pod / "validation_floor_sse.npy")[RANKS.index(rank)].sum())
    dimension = len(np.load(pod / "mean.npy", mmap_mode="r"))
    scale_t = torch.from_numpy(scale).to(device)
    model = ConditionalBiGRU(rank).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    latest = directory / "latest.pt"
    best_path = directory / "best.pt"
    if latest.exists() != args.resume:
        p.error("Use --resume with an existing run")
    history = []
    if args.resume:
        state = load_checkpoint(latest)
        if state["config"] != stats:
            raise ValueError("Training settings changed on resume")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        history = state["history"]
    best = min((row["validation_field_rmse"] for row in history), default=math.inf)
    best_epoch = min(history, key=lambda row: row["validation_field_rmse"])["epoch"] if history else 0
    count = len(training[0])
    print(f"Rank {rank}: {count} training targets, {len(validation[0])} validation targets, "
          f"device={device}, mixed precision={amp}", flush=True)
    for epoch in range(len(history) + 1, args.epochs + 1):
        if epoch - best_epoch > args.patience:
            break
        model.train()
        order = np.random.default_rng([SEED, epoch]).permutation(count)
        running = 0.
        for batch in range(0, count, args.batch):
            selected = torch.from_numpy(order[batch:batch + args.batch]).to(device)
            sequence, query, target = (item[selected] for item in training)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                prediction = model(sequence, query)
            loss = (prediction - target).square().mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.detach().item() * len(selected)
        model.eval()
        with torch.no_grad():
            sequence, query, target = validation
            error = model(sequence, query) - target
            val_mse = error.square().mean().item()
            sse_z = (error * scale_t).double().square().sum().item()
            val_field_rmse = math.sqrt((sse_z + validation_floor) / (len(target) * dimension))
        row = dict(epoch=epoch, training_z_mse=running / count, validation_z_mse=val_mse,
                   validation_field_rmse=val_field_rmse)
        if not all(math.isfinite(value) for value in row.values()):
            raise RuntimeError("Nonfinite training loss")
        history.append(row)
        state = dict(**stats, config=stats, epoch=epoch, model=model.state_dict(),
                     optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), history=history,
                     validation_field_rmse=val_field_rmse)
        if val_field_rmse < best:
            best, best_epoch = val_field_rmse, epoch
            save_checkpoint(best_path, state)
        save_checkpoint(latest, state)
        history_csv(directory / "history.csv", history)
        if epoch == 1 or epoch % 20 == 0:
            print(f"Rank {rank} epoch {epoch}: train={row['training_z_mse']:.6g} "
                  f"val={val_mse:.6g} field={val_field_rmse:.6g}", flush=True)
    save_json(directory / "result.json",
              dict(rank=rank, selected_epoch=best_epoch, epochs=len(history),
                   validation_field_rmse=best, training_z_mse_at_best=history[best_epoch - 1]["training_z_mse"],
                   validation_z_mse_at_best=history[best_epoch - 1]["validation_z_mse"],
                   last_training_z_mse=history[-1]["training_z_mse"],
                   last_validation_field_rmse=history[-1]["validation_field_rmse"]))
    print(f"Rank {rank}: selected epoch {best_epoch}/{len(history)}, validation RMSE={best:.6g}", flush=True)
    if args.confirm_full_field:
        model, checkpoint = load_model(best_path, device)
        confirm_full_field(model, checkpoint, args.cache, args.out, directory, device)


if __name__ == "__main__":
    main()
