"""Node-wise, time-conditioned residual MLP."""
from __future__ import annotations

from inspect import signature

import numpy as np
import torch
from torch import nn


class ResidualMLP(nn.Module):
    def __init__(self, coordinate_mean, coordinate_std, displacement_scale):
        super().__init__()
        self.register_buffer("coordinate_mean", torch.as_tensor(coordinate_mean, dtype=torch.float32))
        self.register_buffer("coordinate_std", torch.as_tensor(coordinate_std, dtype=torch.float32))
        self.register_buffer("displacement_scale", torch.as_tensor(displacement_scale, dtype=torch.float32))
        layers = []
        for width in (11, 256, 256, 256):
            layers.extend((nn.Linear(width, 256), nn.GELU()))
        output = nn.Linear(256, 3)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        self.network = nn.Sequential(*layers, output)

    def forward(self, data):
        coordinates, left, right, alpha, delta_t = data.split((3, 3, 3, 1, 1), dim=1)
        features = torch.cat(((coordinates - self.coordinate_mean) / self.coordinate_std,
                              left / self.displacement_scale, right / self.displacement_scale,
                              alpha, delta_t / 0.004), dim=1)
        linear = (1 - alpha) * left + alpha * right
        correction = self.network(features).float() * self.displacement_scale
        return linear + alpha * (1 - alpha) * correction


def inputs(coordinates, left, right, alpha, delta_t):
    data = np.empty((len(coordinates), 11), dtype=np.float32)
    data[:, :3] = coordinates
    data[:, 3:6] = left
    data[:, 6:9] = right
    data[:, 9] = alpha
    data[:, 10] = delta_t
    return data


def load_checkpoint(path):
    options = {"weights_only": True} if "weights_only" in signature(torch.load).parameters else {}
    return torch.load(path, map_location="cpu", **options)


def load_model(path, device):
    checkpoint = load_checkpoint(path)
    model = ResidualMLP(**checkpoint["normalization"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint
