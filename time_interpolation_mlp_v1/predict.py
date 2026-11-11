"""Generate complete odd-frame predictions or infer one bracketed query."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from cache import TEST_INDICES, cached_dataset, save_json


def linear(left, right, alpha):
    return (1 - alpha) * left + alpha * right


def cubic_window(index, times):
    # Four even indices, bracketing the query. Use a one-sided window at the boundaries.
    left = max(0, min(index // 2 - 1, 47))
    even_slots = np.arange(left, left + 4)
    abscissas = times[2 * even_slots]
    weights = []
    for j, x in enumerate(abscissas):
        others = np.delete(abscissas, j)
        weights.append(np.prod((times[index] - others) / (x - others)))
    return even_slots, weights


def cubic(field, slots, weights):
    result = field[slots[0]] * weights[0]
    for slot, weight in zip(slots[1:], weights[1:]):
        result += field[slot] * weight
    return result.astype(np.float32, copy=False)


class Predictor:
    def __init__(self, checkpoint, device, chunk_size):
        import torch
        from model import inputs, load_model
        self.torch = torch
        self.inputs = inputs
        self.model, self.checkpoint = load_model(checkpoint, device)
        self.device = torch.device(device)
        self.chunk_size = chunk_size

    def sync(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def __call__(self, coordinates, left, right, alpha, delta_t):
        if alpha == 0:
            return np.array(left, copy=True)
        if alpha == 1:
            return np.array(right, copy=True)
        result = np.empty(left.shape, dtype=np.float32)
        with self.torch.inference_mode():
            for start in range(0, len(coordinates), self.chunk_size):
                nodes = slice(start, start + self.chunk_size)
                data = self.inputs(coordinates[nodes], left[nodes], right[nodes], alpha, delta_t)
                predicted = self.model(self.torch.from_numpy(data).to(self.device))
                result[nodes] = predicted.cpu().numpy()
        if not np.isfinite(result).all():
            raise RuntimeError("Nonfinite MLP prediction")
        return result


def generate(cache, out, predictor):
    ids, coordinates, times, cases = cached_dataset(cache)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "node_ids.npy", ids)
    np.save(out / "target_indices.npy", TEST_INDICES)
    np.save(out / "target_times_s.npy", times[TEST_INDICES])
    methods = ("mlp",) if predictor else ("linear", "cubic")
    timings = []
    for case in cases:
        field = np.load(cache / case / "known_even.npy", mmap_mode="r")
        destination = out / case
        destination.mkdir(exist_ok=True)
        for method in methods:
            def frame(index):
                lower = index // 2
                if method == "cubic":
                    slots, weights = cubic_window(index, times)
                    return cubic(field, slots, weights)
                delta = times[index + 1] - times[index - 1]
                alpha = (times[index] - times[index - 1]) / delta
                if method == "linear":
                    return linear(field[lower], field[lower + 1], alpha)
                return predictor(coordinates, field[lower], field[lower + 1], alpha, delta)

            frame(int(TEST_INDICES[0]))  # Warm up the full-frame inference path.
            temp = destination / f"{method}.partial.npy"
            predicted = np.lib.format.open_memmap(temp, mode="w+", dtype="float32",
                                                   shape=(len(TEST_INDICES), len(ids), 3))
            for row, index in enumerate(TEST_INDICES):
                if predictor:
                    predictor.sync()
                start = perf_counter()
                values = frame(int(index))
                if predictor:
                    predictor.sync()
                elapsed_ms = (perf_counter() - start) * 1000
                predicted[row] = values
                input_frames = ([int(2 * j) for j in cubic_window(index, times)[0]] if method == "cubic"
                                else [int(index - 1), int(index + 1)])
                timings.append(dict(case=case, method=method, index=int(index),
                                    time_s=float(times[index]), input_indices=input_frames,
                                    inference_ms=elapsed_ms))
            predicted.flush()
            del predicted
            temp.replace(destination / f"{method}.npy")
            print(f"{case} {method}: {len(TEST_INDICES)} × {len(ids)} predictions", flush=True)
    info = dict(hardware=(predictor.torch.cuda.get_device_name(predictor.device)
                          if predictor and predictor.device.type == "cuda" else "CPU"),
                precision="float32 output",
                checkpoint_epoch=predictor.checkpoint["epoch"] if predictor else None,
                chunk_size=predictor.chunk_size if predictor else None,
                predictions=timings)
    save_json(out / ("mlp_timing.json" if predictor else "baseline_timing.json"), info)


def query(args):
    ids, coordinates, _, _ = cached_dataset(args.cache)
    if not all(math.isfinite(t) for t in (args.t0, args.target, args.t1)) or \
            not args.t0 <= args.target <= args.t1 or args.t0 >= args.t1:
        raise ValueError("Require finite t0 <= target <= t1 and t0 < t1")
    left = np.load(args.left, mmap_mode="r")
    right = np.load(args.right, mmap_mode="r")
    if left.shape != (len(ids), 3) or right.shape != left.shape:
        raise ValueError("Endpoint files must have shape (number of cached nodes, 3) in cached node order")
    if not (np.isfinite(left).all() and np.isfinite(right).all()):
        raise ValueError("Endpoint fields contain nonfinite values")
    predictor = Predictor(args.checkpoint, args.device, args.chunk_size)
    delta_t = args.t1 - args.t0
    predicted = predictor(coordinates, left, right, (args.target - args.t0) / delta_t, delta_t)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, node_ids=ids, time_s=args.target, displacement=predicted)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    batch_parser = sub.add_parser("batch")
    single_parser = sub.add_parser("query")
    for cmd in (batch_parser, single_parser):
        cmd.add_argument("--cache", type=Path, required=True)
        cmd.add_argument("--out", type=Path, required=True)
        cmd.add_argument("--checkpoint", type=Path, required=cmd is single_parser)
        cmd.add_argument("--device", default="cuda")
        cmd.add_argument("--chunk-size", type=int, default=32768)
    for field in ("left", "right"):
        single_parser.add_argument(f"--{field}", type=Path, required=True)
    for field in ("t0", "t1", "target"):
        single_parser.add_argument(f"--{field}", type=float, required=True)
    args = p.parse_args()
    if args.chunk_size <= 0:
        p.error("--chunk-size must be positive")
    if args.mode == "query":
        query(args)
    else:
        predictor = Predictor(args.checkpoint, args.device, args.chunk_size) if args.checkpoint else None
        generate(args.cache, args.out, predictor)


if __name__ == "__main__":
    main()
