"""History-initialized neural displacement dynamics for forecasting past the observed interval."""
from __future__ import annotations

from copy import deepcopy
import math
import time

import numpy as np
from scipy.interpolate import CubicSpline
import torch
from torch import nn


class DelayDynamics(nn.Module):
    def __init__(self, rank, lag, width):
        super().__init__()
        self.lag = lag
        features = lag * rank + (lag + 1) * 3
        self.linear = nn.Linear(features, rank, bias=False, dtype=torch.float64)
        self.residual = nn.Sequential(
            nn.Linear(features, width, bias=False, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(width, rank, bias=False, dtype=torch.float64),
        )
        nn.init.zeros_(self.residual[-1].weight)

    def forward(self, history, loads):
        values = torch.cat((history.flatten(-2), loads.flatten(-2)), dim=-1)
        delta = self.linear(values) + self.residual(values)
        return history[..., -1, :] + delta


def force_at(times, knots, values):
    if times[0] < knots[0] - 1e-10 or times[-1] > knots[-1] + 1e-10:
        raise ValueError("外推时刻超过已提供载荷")
    return np.stack([
        np.column_stack([np.interp(times, knots, case[:, i]) for i in range(3)])
        for case in values
    ])


def scale_q(q):
    rms = np.sqrt(np.mean(q**2, axis=(0, 1)))
    return np.maximum(rms, rms.max() * 0.005)


def ridge(q, f, lag, penalty):
    obs = np.arange(lag, q.shape[1])
    u_history = obs[:, None] - lag + np.arange(lag)
    f_history = obs[:, None] - lag + np.arange(lag + 1)
    x = np.concatenate((q[:, u_history].reshape(q.shape[0], len(obs), -1),
                        f[:, f_history].reshape(q.shape[0], len(obs), -1)), axis=-1)
    x = x.reshape(-1, x.shape[-1])
    target = (q[:, obs] - q[:, obs - 1]).reshape(-1, q.shape[-1])
    gram = x.T @ x / len(x)
    return np.linalg.solve(gram + penalty * np.eye(gram.shape[0]), x.T @ target / len(x))


def rollout(model, history, force, steps):
    lag = model.lag
    result = []
    for offset in range(steps):
        q_next = model(history, force[..., offset:offset + lag + 1, :])
        result.append(q_next)
        history = torch.cat((history[..., 1:, :], q_next.unsqueeze(-2)), dim=-2)
    return torch.stack(result, dim=-2)


def rollout_ridge(weight, history, force, steps):
    lag = history.shape[1]
    result = []
    for offset in range(steps):
        x = np.concatenate((history.reshape(len(history), -1), force[:, offset:offset+lag+1].reshape(len(history), -1)), axis=1)
        q = history[:, -1, :] + x @ weight
        result.append(q)
        history = np.concatenate((history[:, 1:], q[:, None]), axis=1)
    return np.stack(result, axis=1)


def select(q, forces, lags):
    horizon = max(2, q.shape[1] // 5)
    starts = [q.shape[1] - 2 * horizon, q.shape[1] - horizon]
    scores = []
    for lag in lags:
        if min(starts) <= lag:
            continue
        for penalty in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
            per_split = []
            for cut in starts:
                scale = scale_q(q[:, :cut])
                weight = ridge(q[:, :cut] / scale, forces[:, :cut], lag, penalty)
                predicted = rollout_ridge(weight, q[:, cut-lag:cut] / scale,
                                          forces[:, cut-lag:cut+horizon], horizon) * scale
                actual = q[:, cut:cut+horizon]
                error = np.sqrt(np.sum((predicted - actual)**2, axis=(1, 2))
                                / np.maximum(np.sum(actual**2, axis=(1, 2)), 1e-20))
                per_split.extend(error.tolist())
            scores.append((float(np.mean(per_split)), lag, penalty))
    if not scores:
        raise ValueError("历史长度超过可用于训练内验证的帧数")
    return min(scores), scores


def fit_delay(q, forces, *, seed, width=32, epochs=260, lags=(2, 3, 4, 6)):
    began = time.perf_counter()
    torch.manual_seed(seed)
    (validation, lag, penalty), scores = select(q, forces, lags)
    scale = scale_q(q)
    weight = ridge(q / scale, forces, lag, penalty)
    model = DelayDynamics(q.shape[-1], lag, width)
    with torch.no_grad():
        model.linear.weight.copy_(torch.as_tensor(weight.T))
    linear = deepcopy(model).eval()
    x = torch.as_tensor(q / scale)
    f = torch.as_tensor(forces)
    steps = min(8, q.shape[1] // 5)
    starts = np.arange(q.shape[1] - lag - steps + 1)
    history = x[:, starts[:, None] + np.arange(lag)]
    drive = f[:, starts[:, None] + np.arange(lag + steps)]
    targets = x[:, starts[:, None] + lag + np.arange(steps)]
    weights = torch.as_tensor(np.sqrt(scale / scale.max()))
    opt = torch.optim.Adam([
        {"params": model.linear.parameters(), "lr": 0.0003},
        {"params": model.residual.parameters(), "lr": 0.001},
    ])
    best, best_state = math.inf, None
    for step in range(epochs + 1):
        pred = rollout(model, history, drive, steps)
        loss = ((pred - targets) * weights).square().mean()
        loss += 1e-5 * model.residual[-1].weight.square().mean()
        loss_value = float(loss.detach())
        if loss_value < best:
            best, best_state = loss_value, deepcopy(model.state_dict())
        if step < epochs:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.load_state_dict(best_state)
    info = {"lag": lag, "ridge": penalty, "internal_validation_relative_l2": validation,
            "fit_s": time.perf_counter() - began, "width": width, "epochs": epochs}
    print(f"delay dynamics lag={lag}, ridge={penalty:g}, internal validation={validation:.3%}", flush=True)
    return {"linear_delay": linear, "neural_delay": model.eval()}, scale, info


def forecast(models, q, observed_t, query_t, knots, force, force_scale, q_scale):
    step = np.median(np.diff(observed_t))
    ahead = int(np.ceil((query_t[-1] - observed_t[-1]) / step - 1e-9))
    future = observed_t[-1] + step * np.arange(1, ahead + 1)
    lag = next(iter(models.values())).lag
    drive_times = np.concatenate([observed_t[-lag:], future])
    drive = torch.as_tensor(force_at(drive_times, knots, force) / force_scale)
    state = torch.as_tensor(q[:, -lag:] / q_scale)
    predicted, inference_s = {}, {}
    for name, model in models.items():
        t0 = time.perf_counter()
        with torch.no_grad():
            modelled = rollout(model, state, drive, ahead).numpy() * q_scale
        interp_times = np.concatenate([observed_t[-2:], future])
        interp_q = np.concatenate([q[:, -2:], modelled], axis=1)
        predicted[name] = CubicSpline(interp_times, interp_q, axis=1)(query_t)
        inference_s[name] = time.perf_counter() - t0
    return predicted, inference_s
