"""Fit one global POD basis using a blocked tall-skinny SVD."""
import argparse
import math
from pathlib import Path

import numpy as np

from common import CACHE, OUT, RANKS, SPLITS, displacement_maps, protocol, save_json
from score import write_csv


def fields_block(maps, frame_indices, start, end):
    return np.concatenate([field[frame_indices, start:end] for field in maps]).astype(np.float64)


def blocks(size, width):
    return [(start, min(start + width, size)) for start in range(0, size, width)]


def fit(cache, out, width):
    out.mkdir(parents=True, exist_ok=True)
    ids, times, cases = protocol(cache, out)
    folder = out / "pod"
    folder.mkdir(exist_ok=True)
    dim = len(ids) * 3
    odd = displacement_maps(cache, cases, "truth_odd", len(ids))
    even = displacement_maps(cache, cases, "known_even", len(ids))
    train_frames = SPLITS["train"] // 2
    val_frames = SPLITS["validation"] // 2
    n_train = len(cases) * len(train_frames)
    n_val = len(cases) * len(val_frames)
    print(f"Global POD: {n_train} training snapshots, dimension {dim}", flush=True)
    parts = blocks(dim, width)
    mean = np.empty(dim, dtype=np.float32)
    triangular = np.empty((0, n_train), dtype=np.float64)
    training_truth_sse = 0.
    for start, end in parts:
        raw = fields_block(odd, train_frames, start, end)
        training_truth_sse += np.sum(raw**2)
        mean[start:end] = raw.mean(axis=0)
        raw -= mean[start:end]
        local_r = np.linalg.qr(raw.T, mode="r")
        triangular = np.linalg.qr(np.vstack((triangular, local_r)), mode="r")
    _, singular_values, vectors = np.linalg.svd(triangular, full_matrices=False)
    eigenvalues = singular_values**2
    eigenvectors = vectors.T
    rank = max(RANKS)
    if singular_values[rank - 1] <= singular_values[0] * np.finfo(np.float64).eps * n_train:
        raise ValueError("Training snapshots do not support the requested POD rank")
    anchor = np.argmax(np.abs(eigenvectors[:, :rank]), axis=0)
    basis_map = eigenvectors[:, :rank] / np.sqrt(eigenvalues[:rank])
    basis_map *= np.where(eigenvectors[anchor, np.arange(rank)] < 0, -1, 1)
    basis = np.lib.format.open_memmap(folder / "basis.npy", mode="w+", dtype="float32",
                                     shape=(rank, dim))
    np.save(folder / "mean.npy", mean)
    for start, end in parts:
        centered = fields_block(odd, train_frames, start, end) - mean[start:end]
        basis[:, start:end] = basis_map.T @ centered
    basis.flush()

    observed = np.zeros((len(cases) * 51, rank), dtype=np.float64)
    validation = np.zeros((n_val, rank), dtype=np.float64)
    train = np.zeros((n_train, rank), dtype=np.float64)
    gram_basis = np.zeros((rank, rank), dtype=np.float64)
    for start, end in parts:
        modes = basis[:, start:end].astype(np.float64)
        mu = mean[start:end]
        gram_basis += modes @ modes.T
        observed += (fields_block(even, np.arange(51), start, end) - mu) @ modes.T
        train += (fields_block(odd, train_frames, start, end) - mu) @ modes.T
        validation += (fields_block(odd, val_frames, start, end) - mu) @ modes.T
    np.save(folder / "known_z.npy", observed.reshape(len(cases), 51, rank).astype(np.float32))
    np.save(folder / "train_z.npy", train.reshape(len(cases), len(train_frames), rank).astype(np.float32))
    np.save(folder / "validation_z.npy", validation.reshape(len(cases), len(val_frames), rank).astype(np.float32))

    floor_sse = np.zeros((len(RANKS), n_val))
    val_truth_sse = 0.
    for start, end in parts:
        raw = fields_block(odd, val_frames, start, end)
        val_truth_sse += np.sum(raw**2)
        centered = raw - mean[start:end]
        modes = basis[:, start:end].astype(np.float64)
        for row, r in enumerate(RANKS):
            delta = validation[:, :r] @ modes[:r] - centered
            floor_sse[row] += np.einsum("ij,ij->i", delta, delta)
    np.save(folder / "validation_floor_sse.npy", floor_sse)
    retained = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    write_csv(out / "pod_spectrum.csv",
              [dict(mode=i + 1, eigenvalue=float(value), energy=float(retained[i]))
               for i, value in enumerate(eigenvalues)])
    rank_metrics = []
    for row, r in enumerate(RANKS):
        tr_sse = float(eigenvalues[r:].sum())
        va_sse = float(floor_sse[row].sum())
        rank_metrics.append(dict(rank=r, cumulative_energy=float(retained[r-1]),
                                 train_projection_rmse=math.sqrt(tr_sse / (n_train * dim)),
                                 train_projection_relative_l2=math.sqrt(tr_sse / training_truth_sse),
                                 validation_projection_rmse=math.sqrt(va_sse / (n_val * dim)),
                                 validation_projection_relative_l2=math.sqrt(va_sse / val_truth_sse)))
    write_csv(out / "pod_rank_metrics.csv", rank_metrics)
    save_json(folder / "fit.json", dict(train_snapshots=n_train, dim=dim,
              max_basis_orthonormality_error=float(np.max(np.abs(gram_basis - np.eye(rank)))),
              ranks=rank_metrics))
    for row in rank_metrics:
        print(f"POD {row['rank']}: train={row['train_projection_rmse']:.6g}, "
              f"validation projection={row['validation_projection_rmse']:.6g}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=CACHE)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--feature-block", type=int, default=16384)
    args = p.parse_args()
    fit(args.cache, args.out, args.feature_block)
