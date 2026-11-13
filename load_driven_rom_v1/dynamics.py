"""Compact load-driven temporal convolution with a linear autoregressive skip."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class CausalConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, 3, dilation=dilation)

    def forward(self, sequence):
        return self.conv(F.pad(sequence, (self.padding, 0)))


class Dynamics(nn.Module):
    def __init__(self, rank, lag=8, width=32):
        super().__init__()
        self.rank = rank
        self.lag = lag
        self.linear = nn.Linear(lag * rank + (lag + 1) * 3, rank, bias=False)
        self.temporal = nn.Sequential(CausalConv(rank + 6, width, 1), nn.Tanh(),
                                      CausalConv(width, width, 2), nn.Tanh(),
                                      nn.Conv1d(width, rank, 1))
        nn.init.zeros_(self.temporal[-1].weight)
        nn.init.zeros_(self.temporal[-1].bias)

    def forward(self, history, force):
        sequence = torch.cat((history, force[:, :-1], force[:, 1:]), dim=-1)
        direct = self.linear(torch.cat((history.flatten(1), force.flatten(1)), dim=1))
        correction = self.temporal(sequence.transpose(1, 2))[:, :, -1]
        return history[:, -1] + direct + correction


def windows(state, force, lag, horizon):
    length = state.shape[1]
    start = np.arange(length - lag - horizon + 1)
    history = start[:, None] + np.arange(lag)
    driver = start[:, None] + np.arange(lag + horizon)
    target = start[:, None] + lag + np.arange(horizon)
    return state[:, history], force[:, driver], state[:, target]
