"""Forecast independent requested targets from four measured displacement fields."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import CHUNK, LAG, OUT, decode
from model import ConditionalTCN


class ROM:
    def __init__(self, folder, device, model_type="variable"):
        self.device = torch.device(device)
        self.model_type = model_type
        self.config = json.loads((folder / "protocol.json").read_text(encoding="utf-8"))
        self.rank = self.config["rank"]
        self.basis = np.load(folder / "pod_modes.npy", mmap_mode="r")[:self.rank]
        self.reference = np.load(folder / "reference_field.npy", mmap_mode="r")
        self.ids = np.load(folder / "node_ids.npy")
        self.scale = np.asarray(self.config["coefficient_scale"])
        self.model = ConditionalTCN(self.rank, self.config["width"]).to(self.device)
        self.model.load_state_dict(torch.load(folder / f"{model_type}_best.pt", map_location=self.device))
        self.model.eval()

    def pack(self, coefficients, history, load, targets):
        return (torch.as_tensor(coefficients / self.scale, device=self.device, dtype=torch.float32),
                torch.as_tensor(history, device=self.device, dtype=torch.int64),
                torch.as_tensor(load / self.config["force_scale"], device=self.device, dtype=torch.float32),
                torch.as_tensor(targets, device=self.device, dtype=torch.int64))

    @torch.inference_mode()
    def predict(self, fields, history_ms, loads, target_ms):
        if fields.shape != (LAG, len(self.ids), 3) or not np.isfinite(fields).all():
            raise ValueError("Expected four finite measured fields in the model node order")
        if (history_ms.shape != (LAG,) or not np.issubdtype(history_ms.dtype, np.integer)
                or np.any(np.diff(history_ms) != 1) or history_ms[0] < 0):
            raise ValueError("Expected four consecutive integer millisecond history indices")
        if (target_ms.ndim != 1 or not len(target_ms) or not np.issubdtype(target_ms.dtype, np.integer)
                or np.any((target_ms - history_ms[-1] < 1) | (target_ms - history_ms[-1] > 8))
                or (self.model_type == "fixed" and np.any(target_ms - history_ms[-1] != 1))
                or loads.ndim != 1 or not np.isfinite(loads).all() or target_ms.max() >= len(loads)):
            raise ValueError("Invalid target span or X load coverage")
        encoded = np.zeros((LAG, self.rank))
        for left in range(0, len(self.ids), CHUNK):
            right = min(left+CHUNK, len(self.ids))
            mode = self.basis[:, left:right].reshape(self.rank,-1)
            block = (fields[:,left:right].astype(np.float64) - self.reference[left:right]).reshape(LAG,-1)
            encoded += block @ mode.T
        count = len(target_ms)
        history = np.broadcast_to(history_ms, (count,LAG)).copy()
        q = np.broadcast_to(encoded, (count,LAG,self.rank))
        return self.model(*self.pack(q,history,loads,target_ms)).cpu().numpy() * self.scale

    def write_fields(self, path, prediction):
        result = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                           shape=(len(prediction),len(self.ids),3))
        for left in range(0,len(self.ids),CHUNK):
            right = min(left+CHUNK,len(self.ids))
            result[:,left:right] = decode(prediction,self.basis[:,left:right],self.reference[left:right])
        result.flush()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=OUT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    rom = ROM(args.model_dir,args.device)
    with np.load(args.input, allow_pickle=False) as data:
        if not np.array_equal(data["node_ids"],rom.ids):
            raise ValueError("Node IDs must match the saved model")
        target = data["target_ms"]
        q = rom.predict(data["history"],data["history_ms"],data["loads_x"],target)
    args.out.mkdir(parents=True,exist_ok=True)
    rom.write_fields(args.out / "displacement.npy",q)
    np.savez(args.out / "mapping.npz",node_ids=rom.ids,times_s=target*.001)
    print(f"Forecast {len(target)} independent target(s) from one measured history")


if __name__ == "__main__":
    main()
