"""Project four observed displacement fields, predict POD coefficients, and reconstruct."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from common import CACHE, OUT, cached_dataset, windows
from fit_gru import load_model


class Predictor:
    def __init__(self, out, rank, device):
        self.device = torch.device(device)
        self.model, ckpt = load_model(out / "training" / f"rank{rank}" / "best.pt", self.device)
        self.center = np.asarray(ckpt["center"], dtype=np.float32)
        self.scale = np.asarray(ckpt["scale"], dtype=np.float32)
        self.mean = np.load(out / "pod" / "mean.npy").astype(np.float64)
        self.basis = np.load(out / "pod" / "basis.npy", mmap_mode="r")[:rank].astype(np.float64)

    def project(self, field):
        return (field - self.mean) @ self.basis.T

    def reconstruct(self, z):
        return z @ self.basis + self.mean

    @torch.inference_mode()
    def latent(self, z, offsets, query):
        normalized = (z - self.center) / self.scale
        sequence = np.column_stack((normalized, offsets)).astype(np.float32)[None]
        query = query.astype(np.float32, copy=False)[None]
        prediction = self.model(torch.from_numpy(sequence).to(self.device),
                                torch.from_numpy(query).to(self.device))
        return prediction.cpu().numpy()[0] * self.scale + self.center

    def predict(self, observed, slots, offsets, query):
        start = perf_counter()
        context = self.project(observed[slots])
        z = self.latent(context, offsets, query)
        field = self.reconstruct(z).astype(np.float32)
        return field, z, (perf_counter() - start) * 1000


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=CACHE)
    p.add_argument("--run", type=Path, default=OUT)
    p.add_argument("--case", required=True)
    p.add_argument("--target-ms", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    torch.set_num_threads(1)
    ids, _, times, cases = cached_dataset(args.cache)
    if args.case not in cases or args.target_ms not in range(1, 100, 2):
        p.error("Specify a cached case and an odd target from 1 to 99 ms")
    rank = json.loads((args.run / "selection.json").read_text())["rank"]
    predictor = Predictor(args.run, rank, args.device)
    fields = np.load(args.cache / args.case / "known_even.npy", mmap_mode="r").reshape(51, -1)
    slots, offsets, query = windows(np.array([args.target_ms]), times)
    field, z, elapsed = predictor.predict(fields, slots[0], offsets[0], query[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, node_ids=ids, time_s=times[args.target_ms],
             displacement=field.reshape(len(ids), 3), pod_coefficients=z)
    print(f"Saved {len(ids)} nodes; complete frame inference {elapsed:.2f} ms")


if __name__ == "__main__":
    main()
