"""Run the learned displacement ROM with an initial history and a supplied load."""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from load_data import load_cases
from train import Dynamics, LAG, NODE_CHUNK, OUTPUT, ROOT, project, rollout, rollout_ridge


class ROM:
    def __init__(self, model_directory, device):
        self.device = torch.device(device)
        state = torch.load(model_directory / "model.pt", map_location="cpu")
        self.lag = state["lag"]
        self.rank = state["rank"]
        self.model = Dynamics(self.rank, self.lag, state["width"]).to(self.device)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.scale = np.asarray(state["scale"])
        self.force_scale = np.asarray(state["force_scale"])
        self.linear_weight = np.asarray(state["ridge_weight"])
        self.basis = np.load(model_directory / "pod_modes.npy", mmap_mode="r")[:self.rank]
        self.ids = np.load(model_directory / "node_ids.npy")
        self.epoch = state["epoch"]

    @torch.inference_mode()
    def run_latent(self, history, drive):
        if history.shape != (self.lag, self.rank) or drive.ndim != 2 or drive.shape[1] != 3:
            raise ValueError("Unexpected history or aligned load shape")
        if len(drive) <= self.lag or not np.isfinite(drive).all() or not np.isfinite(history).all():
            raise ValueError("Finite initial states and loads extending beyond the history are required")
        q = torch.as_tensor(history[None] / self.scale, dtype=torch.float32, device=self.device)
        f = torch.as_tensor(drive[None] / self.force_scale, dtype=torch.float32, device=self.device)
        result = rollout(self.model, q, f, len(drive) - self.lag)[0].cpu().numpy()
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite recursive displacement forecast")
        return result * self.scale

    def run_linear(self, history, drive):
        return rollout_ridge(self.linear_weight, history[None] / self.scale,
                             drive[None] / self.force_scale, len(drive) - self.lag)[0] * self.scale

    def encode_history(self, fields):
        return project(self.basis, fields, np.arange(self.lag), NODE_CHUNK)

    def decode(self, coefficients, initial, output):
        nodes = len(self.ids)
        partial = output.with_suffix(".partial.npy")
        target = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32,
                                          shape=(len(coefficients), nodes, 3))
        for begin in range(0, nodes, NODE_CHUNK):
            end = min(begin + NODE_CHUNK, nodes)
            modes = self.basis[:, begin:end].reshape(self.rank, -1)
            block = coefficients @ modes
            target[:, begin:end] = block.reshape(len(coefficients), end - begin, 3) + initial[begin:end]
        target.flush()
        del target
        partial.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--case", help="Use the first eight frames and the load table of a cached case")
    inputs.add_argument("--input", type=Path, help="NPZ containing initial fields and aligned load")
    parser.add_argument("--model-dir", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = ROM(args.model_dir, args.device)
    if args.case:
        ids, times, fields, loads, _ = load_cases(ROOT)
        frames, drive = fields[args.case], loads[args.case]
    else:
        with np.load(args.input, allow_pickle=False) as inputs:
            ids, times = inputs["node_ids"], inputs["times_s"]
            frames, drive = inputs["history"], inputs["loads"]
        if frames.shape != (model.lag, len(model.ids), 3) or not np.isfinite(frames).all():
            raise ValueError("history must contain exactly eight finite initial fields in model node order")
    if not np.array_equal(ids, model.ids):
        raise ValueError("Input node IDs differ from the model mesh")
    if (times.shape != (len(drive),) or not np.isfinite(times).all() or
            not np.allclose(np.diff(times), .001, atol=1e-10, rtol=0)):
        raise ValueError("Expected matching 1 ms time/load grids")
    args.out.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    q = model.run_latent(model.encode_history(frames), drive)
    model.decode(q, frames[0], args.out / "displacement.npy")
    np.savez(args.out / "mapping.npz", node_ids=ids, times_s=times[model.lag:])
    print(f"Predicted {len(q)} frames, total encode/forecast/decode/write: {perf_counter()-started:.2f} s")


if __name__ == "__main__":
    main()
