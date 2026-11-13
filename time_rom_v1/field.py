"""Shared POD basis from observed displacement frames, with chunked field access."""
from __future__ import annotations

import numpy as np


def fit_pod(snapshots, indices, output, requested_rank, node_chunk):
    n_cases = len(snapshots)
    n_obs = len(indices)
    n_nodes = snapshots[0].shape[1]
    size = n_cases * n_obs
    gram = np.zeros((size, size), dtype=np.float64)

    def block(start, end):
        return np.concatenate([
            (np.asarray(x[indices, start:end], dtype=np.float64) - x[0, start:end]).reshape(n_obs, -1)
            for x in snapshots
        ], axis=0)

    for start in range(0, n_nodes, node_chunk):
        x = block(start, min(start + node_chunk, n_nodes))
        gram += x @ x.T
    values, vectors = np.linalg.eigh(0.5 * (gram + gram.T))
    values = np.maximum(values[::-1], 0)
    vectors = vectors[:, ::-1]
    rank = min(requested_rank, int(np.count_nonzero(values > values[0] * 1e-12)))
    if rank == 0:
        raise ValueError("训练帧没有有效位移能量")
    vectors = vectors[:, :rank]
    signs = np.sign(vectors[np.argmax(np.abs(vectors), axis=0), np.arange(rank)])
    vectors = vectors * signs
    singular = np.sqrt(values[:rank])
    basis = np.lib.format.open_memmap(output / "pod_modes.npy", mode="w+", dtype="float64",
                                      shape=(rank, n_nodes, 3))
    for start in range(0, n_nodes, node_chunk):
        end = min(start + node_chunk, n_nodes)
        b = (vectors.T @ block(start, end)) / singular[:, None]
        basis[:, start:end] = b.reshape(rank, end - start, 3)
    basis.flush()
    q = (vectors * singular).reshape(n_cases, n_obs, rank)
    return basis, q, values


def project(basis, snapshots, frames, node_chunk):
    r, nodes, _ = basis.shape
    result = np.zeros((len(frames), r))
    for start in range(0, nodes, node_chunk):
        end = min(start + node_chunk, nodes)
        x = (np.asarray(snapshots[frames, start:end], dtype=np.float64) - snapshots[0, start:end]).reshape(len(frames), -1)
        result += x @ basis[:, start:end].reshape(r, -1).T
    return result
