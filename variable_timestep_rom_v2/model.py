"""Two causal temporal convolutions with a separate prediction-interval encoder."""
import torch
from torch import nn

from data import LAG
from dynamics import CausalConv

STEP_SCALE = 8.


def force_features(force, current, target):
    dt = target - current
    offsets = torch.arange(1, 9, device=force.device)
    valid = offsets[None] <= dt[:, None]
    samples = force[current[:, None] + torch.minimum(offsets[None], dt[:, None])]
    start, end = force[current], force[target]
    mean = ((samples * valid).sum(dim=1) + .5 * (start - end)) / dt
    return torch.stack((start, end, mean), dim=1)


class ConditionalTCN(nn.Module):
    def __init__(self, rank, width=32):
        super().__init__()
        self.temporal = nn.Sequential(CausalConv(rank + 2, width, 1), nn.Tanh(),
                                      CausalConv(width, width, 2), nn.Tanh())
        self.timestep = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh())
        self.load = nn.Sequential(nn.Linear(3, 16), nn.Tanh())
        self.rate = nn.Sequential(nn.Linear(width * LAG + 32, width), nn.Tanh(), nn.Linear(width, rank))
        nn.init.zeros_(self.rate[-1].weight)
        nn.init.zeros_(self.rate[-1].bias)

    def forward(self, history, history_idx, force, target):
        previous_gaps = history_idx[:, 1:] - history_idx[:, :-1]
        gap = torch.cat((torch.zeros_like(history_idx[:, :1]), previous_gaps), dim=1)
        current = history_idx[:, -1]
        dt = (target - current).unsqueeze(1) / STEP_SCALE
        seq = torch.cat((history, (gap / STEP_SCALE).unsqueeze(2), force[history_idx].unsqueeze(2)), dim=2)
        feature = self.temporal(seq.transpose(1, 2)).flatten(1)
        forcing = self.load(force_features(force, current, target))
        update = self.rate(torch.cat((feature, self.timestep(dt), forcing), dim=1))
        speed = (history[:, -1] - history[:, -2]) * (STEP_SCALE / previous_gaps[:, -1, None])
        return history[:, -1] + dt * (speed + update)
