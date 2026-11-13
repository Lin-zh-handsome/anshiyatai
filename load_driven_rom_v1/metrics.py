"""Measure recursively predicted displacements against finite-element trajectories."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np


NODE_CHUNK = 4096
METHOD_LABELS = {
    "tcn": "TCN（时序卷积网络）",
    "linear": "载荷驱动线性推进",
    "persistence": "保持初始状态",
    "pod_projection": "POD 真值投影",
}


def write_csv(path: Path, columns: list[str], rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def series_metrics(squared, absolute, truth_squared, max_vector, peak, peak_node, times, ids):
    frame, count = len(times), len(ids)
    scale = np.sqrt(truth_squared)
    relative = np.divide(np.sqrt(squared), scale,
                         out=np.full(frame, np.nan), where=scale > 1e-12)
    pooled = float(np.sqrt(squared.sum() / truth_squared.sum())) if truth_squared.sum() > 1e-24 else None
    peak_step = int(peak.argmax())
    return {
        "rmse": float(np.sqrt(squared.sum() / (frame * count * 3))),
        "relative_l2": pooled,
        "mae": float(absolute.sum() / (frame * count * 3)),
        "max_node_vector_error": float(max_vector.max()),
        "max_error_time_ms": float(times[max_vector.argmax()] * 1000),
        "peak_displacement": float(peak[peak_step]),
        "peak_time_ms": float(times[peak_step] * 1000),
        "peak_node_id": int(ids[peak_node[peak_step]]),
    }, relative


def plot_results(out: Path, case: str, horizon, times, methods, per_frame, node_id, traces):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt

    families = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in ("Noto Sans CJK JP", "Noto Sans CJK SC")
                 if name in families), None)
    if font:
        plt.rcParams["font.family"] = font
        plt.rcParams["axes.unicode_minus"] = False
    colors = {"tcn": "#007F75", "linear": "#496681", "persistence": "#AD613B", "pod_projection": "#697581"}

    fig, axes = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True, constrained_layout=True)
    for name in methods:
        stats = per_frame[name]
        label = METHOD_LABELS.get(name, name) if font else name
        axes[0].plot(horizon, stats["rmse"], label=label, color=colors.get(name))
        axes[1].plot(horizon, stats["relative_l2"] * 100, color=colors.get(name))
    axes[0].set_title(case, loc="left")
    axes[0].set_ylabel("全场均方根误差（原始单位）" if font else "Field RMSE (native units)")
    axes[1].set_ylabel("相对 L2（%）" if font else "Relative L2 (%)")
    axes[1].set_xlabel("推进时间（毫秒）" if font else "Horizon (ms)")
    axes[0].legend(loc="upper left", frameon=False)
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    fig.savefig(out / "error_vs_horizon.png", dpi=155)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
    shown = [name for name in ("truth", "tcn", "linear") if name in traces]
    for direction, ax in enumerate(axes):
        for name in shown:
            if name == "truth":
                label = "有限元真值" if font else "FE truth"
                color, style = "#23252D", "-"
            else:
                label = METHOD_LABELS.get(name, name) if font else name
                color, style = colors.get(name), "--"
            ax.plot(times * 1000, traces[name][:, direction], color=color, linestyle=style, label=label)
        ax.grid(axis="y", alpha=.2)
        ax.set_ylabel(("DX", "DY", "DZ")[direction])
    axes[0].set_title(f"{case}  /  节点 {node_id}" if font else f"{case} / node {node_id}", loc="left")
    axes[0].legend(loc="upper left", frameon=False)
    axes[-1].set_xlabel("仿真时间（毫秒）" if font else "Simulation time (ms)")
    fig.savefig(out / "key_node.png", dpi=155)
    plt.close(fig)


def evaluate(out: Path, case: str, ids, times, snapshots, basis, forecast, steps, diagnostics):
    """Score latent forecasts decoded as initial field plus POD basis times coefficients."""
    directory = Path(out) / case
    directory.mkdir(parents=True, exist_ok=True)
    ids, times = np.asarray(ids), np.asarray(times)
    steps = np.asarray(steps, dtype=np.intp)
    nodes, rank, horizon = len(ids), len(basis), len(steps)
    if not nodes or not horizon or steps[0] < 1 or np.any(np.diff(steps) <= 0) or steps[-1] >= len(times):
        raise ValueError("Evaluation requires a valid strictly increasing noninitial frame sequence")
    if basis.shape != (rank, nodes, 3):
        raise ValueError("POD basis and node mapping are inconsistent")
    if any(z.shape != (horizon, rank) or not np.isfinite(z).all() for z in forecast.values()):
        raise ValueError("Latent forecasts must be finite arrays with shape (horizon, rank)")

    target_times = times[steps]
    elapsed = (target_times - times[steps[0] - 1]) * 1000
    method_names = list(forecast)
    names = ["truth", *method_names]
    truth_squared = np.zeros(horizon)
    squared = {name: np.zeros(horizon) for name in method_names}
    absolute = {name: np.zeros(horizon) for name in method_names}
    max_vector = {name: np.zeros(horizon) for name in method_names}
    peak = {name: np.full(horizon, -1.) for name in names}
    peak_node = {name: np.zeros(horizon, dtype=int) for name in names}
    decode_time = dict.fromkeys(method_names, 0.)
    frame_index = np.arange(horizon)
    node_ids = np.flatnonzero(ids == 56877)
    tcn_displacement = directory / "tcn_displacement.npy"
    tmp = tcn_displacement.with_suffix(".partial.npy")
    prediction = None
    if "tcn" in forecast:
        prediction = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(horizon, nodes, 3))

    def record_peak(name, values, offset):
        maxima = np.linalg.norm(values, axis=-1)
        node = maxima.argmax(axis=1)
        amplitude = maxima[frame_index, node]
        update = amplitude > peak[name]
        peak[name][update] = amplitude[update]
        peak_node[name][update] = (node + offset)[update]

    for start in range(0, nodes, NODE_CHUNK):
        end = min(start + NODE_CHUNK, nodes)
        truth = np.asarray(snapshots[steps, start:end], dtype=np.float64)
        initial = np.asarray(snapshots[0, start:end], dtype=np.float64)
        spatial = np.asarray(basis[:, start:end], dtype=np.float64).reshape(rank, -1)
        truth_squared += np.einsum("hij,hij->h", truth, truth)
        record_peak("truth", truth, start)
        for name, coefficients in forecast.items():
            then = perf_counter()
            values = (coefficients @ spatial).reshape(horizon, end - start, 3) + initial
            decode_time[name] += perf_counter() - then
            if name == "tcn":
                prediction[:, start:end] = values
            record_peak(name, values, start)
            error = values - truth
            squared[name] += np.einsum("hij,hij->h", error, error)
            absolute[name] += np.abs(error).sum(axis=(1, 2))
            max_vector[name] = np.maximum(max_vector[name], np.linalg.norm(error, axis=-1).max(axis=1))

    if prediction is not None:
        prediction.flush()
        del prediction
        tmp.replace(tcn_displacement)
        np.savez(directory / "tcn_mapping.npz", node_ids=ids, frame_indices=steps, times_s=target_times)

    methods, per_frame, frame_rows = {}, {}, []
    for name in method_names:
        metrics, relative = series_metrics(squared[name], absolute[name], truth_squared, max_vector[name],
                                           peak[name], peak_node[name], target_times, ids)
        metrics["field_reconstruction_s"] = float(decode_time[name])
        methods[name] = metrics
        rmse = np.sqrt(squared[name] / (nodes * 3))
        mae = absolute[name] / (nodes * 3)
        per_frame[name] = {"rmse": rmse, "relative_l2": relative}
        frame_rows.extend((name, int(step), time * 1000, length, err,
                           rel if np.isfinite(rel) else "", ae, vec)
                          for step, time, length, err, rel, ae, vec in
                          zip(steps, target_times, elapsed, rmse, relative, mae, max_vector[name]))
    write_csv(directory / "per_frame.csv",
              ["method", "frame", "time_ms", "horizon_ms", "rmse", "relative_l2", "mae",
               "max_node_vector_error"], frame_rows)

    truth_peak_step = int(peak["truth"].argmax())
    truth_peak_node = int(peak_node["truth"][truth_peak_step])
    key = int(node_ids[0]) if len(node_ids) else truth_peak_node
    basis_node = np.asarray(basis[:, key], dtype=np.float64)
    origin = np.asarray(snapshots[0, key:key + 1], dtype=np.float64)[0]
    traces = {"truth": np.asarray(snapshots[steps, key:key + 1], dtype=np.float64)[:, 0, :]}
    traces.update({name: coeff @ basis_node + origin for name, coeff in forecast.items()})
    write_csv(directory / "key_node.csv", ["node_id", "method", "time_ms", "dx", "dy", "dz"],
              ((int(ids[key]), name, time * 1000, *values)
               for name, trace in traces.items() for time, values in zip(target_times, trace)))

    summary = {
        "case": case, "horizon_frames": horizon,
        "first_target_ms": float(target_times[0] * 1000),
        "last_target_ms": float(target_times[-1] * 1000),
        "diagnostics": diagnostics, "methods": methods, "key_node_id": int(ids[key]),
        "truth_peak_displacement": float(peak["truth"][truth_peak_step]),
        "truth_peak_time_ms": float(target_times[truth_peak_step] * 1000),
        "truth_peak_node_id": int(ids[truth_peak_node]),
    }
    with (directory / "evaluation.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
    plot_results(directory, case, elapsed, target_times, method_names, per_frame, int(ids[key]), traces)
    return summary
