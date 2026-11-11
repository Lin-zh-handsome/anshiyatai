"""Make the data audit, experiment report, and figures."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, ft2font
import matplotlib.pyplot as plt
import numpy as np

from cache import TEST_INDICES, cached_dataset
from score import METHODS


LABELS = {"linear": "线性", "cubic": "局部三次", "mlp": "残差 MLP"}
COLORS = {"linear": "#55748C", "cubic": "#118872", "mlp": "#CF613D"}


def save(fig, path):
    fig.savefig(path, dpi=155, facecolor="white")
    plt.close(fig)


def curves(args, figures, rows, summaries):
    with (args.training / "history.csv").open(newline="", encoding="utf-8") as f:
        history = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(f)]
    best = min(history, key=lambda row: row["validation_mse"])
    fig, ax = plt.subplots(figsize=(7.5, 4.2), layout="constrained")
    ax.semilogy([r["epoch"] for r in history], [r["train_mse"] for r in history], label="训练损失")
    ax.semilogy([r["epoch"] for r in history], [r["validation_mse"] for r in history], label="验证损失")
    ax.axhline(best["validation_linear_mse"], color=COLORS["linear"], linestyle="--", label="验证集线性基线")
    ax.axvline(best["epoch"], color="0.45", linestyle=":", label=f"选中第 {best['epoch']:.0f} 轮")
    ax.set(xlabel="训练轮次", ylabel="均方误差（归一化位移）")
    ax.legend()
    save(fig, figures / "training.png")

    cases = list(dict.fromkeys(row["case"] for row in rows))
    fig, panels = plt.subplots((len(cases) + 2) // 3, 3, figsize=(13, 3.3 * ((len(cases) + 2) // 3)),
                               layout="constrained", squeeze=False)
    for ax, case in zip(panels.flat, cases):
        for method in METHODS:
            subset = [row for row in rows if row["case"] == case and row["method"] == method]
            ax.plot([row["time_s"] * 1000 for row in subset], [row["rmse"] for row in subset],
                    label=LABELS[method], color=COLORS[method])
        ax.set(title=case, xlabel="时间（毫秒）", ylabel="均方根误差（原始位移单位）")
    for ax in list(panels.flat)[len(cases):]:
        ax.set_visible(False)
    panels[0, 0].legend()
    save(fig, figures / "per_frame_rmse.png")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), layout="constrained")
    labels = cases + ["全部"]
    ticks = np.arange(len(labels))
    keys = {(row["case"], row["method"]): row for row in summaries}
    for i, method in enumerate(METHODS):
        selection = [keys[key, method] for key in cases + ["ALL"]]
        ax1.bar(ticks + (i - 1) * .24, [row["rmse"] for row in selection],
                width=.24, color=COLORS[method], label=LABELS[method])
        ax2.bar(ticks + (i - 1) * .24, [row["inference_median_ms"] for row in selection],
                width=.24, color=COLORS[method])
    ax1.set(ylabel="均方根误差（原始位移单位）", xticks=ticks, xticklabels=labels)
    ax2.set(ylabel="全帧推理中位数（毫秒）", xticks=ticks, xticklabels=labels, yscale="log")
    ax1.legend()
    save(fig, figures / "overall_comparison.png")
    return history, best


def field_plots(args, figures):
    ids, coords, times, cases = cached_dataset(args.cache)
    node_ids = [key for key in args.nodes if np.any(ids == key)]
    if not node_ids:
        raise ValueError("No requested plot node ID was found")
    node_rows = [int(np.flatnonzero(ids == key)[0]) for key in node_ids]
    target_row = (args.error_ms - 1) // 2
    for case in cases:
        truth = np.load(args.cache / case / "truth_odd.npy", mmap_mode="r")
        known = np.load(args.cache / case / "known_even.npy", mmap_mode="r")
        predictions = {method: np.load(args.predictions / case / f"{method}.npy", mmap_mode="r")
                       for method in METHODS}
        fig, panels = plt.subplots(3, len(node_rows), figsize=(5.4 * len(node_rows), 7.0),
                                   sharex=True, squeeze=False, layout="constrained")
        for col, (node, row) in enumerate(zip(node_ids, node_rows)):
            for axis, label in enumerate(("DX", "DY", "DZ")):
                ax = panels[axis, col]
                trace = np.empty(len(times))
                trace[::2], trace[1::2] = known[:, row, axis], truth[:, row, axis]
                ax.plot(times * 1000, trace, color="#17202A", linewidth=1.25, label="有限元真值")
                for method, field in predictions.items():
                    ax.plot(times[TEST_INDICES] * 1000, field[:, row, axis], marker=".", markersize=2.5,
                            linewidth=0.9, linestyle="--", color=COLORS[method], label=LABELS[method])
                ax.set_ylabel(label)
                if axis == 0:
                    ax.set_title(f"节点 {node}")
                if axis == 2:
                    ax.set_xlabel("时间（毫秒）")
        panels[0, 0].legend(ncol=2, fontsize=8)
        fig.suptitle(f"{case} | 关键节点位移（原始位移单位）")
        save(fig, figures / f"nodes_{case}.png")

        errors = [np.linalg.norm(values[target_row] - truth[target_row], axis=1)
                  for values in predictions.values()]
        random = np.random.default_rng(20261111)
        sample = random.choice(len(ids), min(len(ids), 15000), replace=False)
        peaks = [np.argsort(error)[-200:] for error in errors]
        pick = np.unique(np.concatenate([sample, *peaks]))
        coordinates = coords[pick]
        limit = max(float(value.max()) for value in errors) or 1.0
        fig, panels = plt.subplots(1, 3, figsize=(13.8, 4.6), subplot_kw={"projection": "3d"},
                                   layout="constrained")
        extent = np.ptp(coords, axis=0)
        for ax, method, error in zip(panels, METHODS, errors):
            scatter = ax.scatter(*coordinates.T, c=error[pick], s=1.8, cmap="magma", vmin=0,
                                 vmax=limit, alpha=0.82, depthshade=False)
            ax.set(xlabel="X", ylabel="Y", zlabel="Z", title=LABELS[method])
            ax.set_box_aspect(np.where(extent > 0, extent, 1))
        fig.colorbar(scatter, ax=panels, shrink=.62, label="三维位移向量误差（原始位移单位）")
        fig.suptitle(f"{case} | {args.error_ms} 毫秒的全场误差分布")
        save(fig, figures / f"field_{args.error_ms:03d}ms_{case}.png")


def data_report(args):
    audit = json.loads((args.cache / "audit.json").read_text(encoding="utf-8"))
    lines = ["# 数据检查报告", "",
             f"从一个 RAR 分卷档案的实际成员识别 {len(audit['cases'])} 个位移工况。",
             f"读取 {len(audit['volumes'])} 个分卷。每个工况有 {audit['node_count']:,} 个节点，"
             "101 帧，0–0.100 s，间隔 0.001 s。",
             f"初始坐标来自独立文件 {Path(audit['coordinates']).name}。"
             f"物理时间取自 {Path(audit['time_map']).name} 中的 Time 标签映射。",
             "节点以初始坐标文件的 ID 顺序对齐。位移缓存为单精度，节点 ID 为 64 位整数。",
             "载荷倍率沿用供应方文件名。载荷和约束反力文件保留在原档案中，"
             "模型使用初始坐标、位移和时间条件。", "",
             "| 工况 | 帧数 | 重排帧 | 原始顺序变更帧 | 缺帧 | 缺失或重复节点 | NaN/Inf | 最大节点位移 | "
             "异常幅值帧 | 最大缓存转换误差 | 数据来源 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for case, info in audit["cases"].items():
        frames = info["frames"]
        order_changes = sum(not frame["same_order_as_previous"] for frame in frames)
        source = "既有严格缓存，原始 ID 顺序复核" if info.get("reused_snapshot") else "原档案流式解析"
        lines.append(f"| {case} | {len(frames)} | {info['reordered_frames']} | {order_changes} | "
                     f"{info['missing_frames']} | {info['missing_or_duplicate_node_records']} | {info['nan_inf']} | "
                     f"{max(frame['max_vector'] for frame in frames):.6g} | {len(info['scale_outlier_frames'])} | "
                     f"{info['max_rounding_abs']:.3g} | {source} |")
    lines.extend(["", "异常幅值筛查以每帧最大位移向量范数为统计量，阈值为同工况非零最大值中位数的 100 倍。"
                  "完整逐帧极值、均方根和重排状态在缓存目录的 audit.json 中。"])
    (args.out / "data_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def experiment_report(args, summaries, diagnostics, hardware, model_info, history, best):
    pooled = {row["method"]: row for row in summaries if row["case"] == "ALL"}
    linear, cubic, mlp = [pooled[method] for method in METHODS]
    improvement = 100 * (1 - mlp["rmse"] / linear["rmse"]) if linear["rmse"] else 0
    winner = min(pooled, key=lambda method: pooled[method]["rmse"])
    direction = "降低" if improvement >= 0 else "升高"
    conclusion = f"残差 MLP（多层感知机）的合并全场均方根误差较线性插值{direction} {abs(improvement):.2f}%，{LABELS[winner]}的误差最低。"
    protocol = json.loads((args.training / "protocol.json").read_text(encoding="utf-8"))
    lines = ["# 瞬态有限元时间步长内插", "", conclusion, "",
             f"全部工况合并均方根误差分别为线性 {linear['rmse']:.6g}、"
             f"局部三次 {cubic['rmse']:.6g}、残差 MLP {mlp['rmse']:.6g}（原始位移单位）。",
             f"全帧推理时间中位数分别为 {linear['inference_median_ms']:.3f}、"
             f"{cubic['inference_median_ms']:.3f}、{mlp['inference_median_ms']:.3f} 毫秒。",
             f"传统插值运行于 {hardware}，残差 MLP 运行于 {model_info['hardware']}。"
             "计时包含输入取片、计算和模型主机/显卡传输，排除加载模型、写盘与误差统计。", "",
             "## 划分与比较", "",
             f"- 用同一套偶数帧时间块训练与验证，每工况 {len(protocol['training_windows'])} 个训练窗口、"
             f"{len(protocol['validation_windows'])} 个验证窗口。两者的输入与监督帧均完全分离。",
             "- 训练和验证使用 4、6、8 毫秒端点间隔，最终评测只在 2 毫秒端点间隔的中点进行。"
             "每个工况的 50 个奇数毫秒帧均为保留标签，检验同工况从较粗间隔迁移到较细间隔。",
             "- 线性与模型使用相邻两个偶数端点。局部三次采用四个偶数帧的拉格朗日多项式，"
             "内部通常为目标前后各两帧，首尾分别用 [0, 2, 4, 6] 与 [94, 96, 98, 100] 毫秒。",
             f"- 最低验证损失出现在第 {best['epoch']:.0f}/{len(history)} 轮。"
             f"该轮验证均方误差 {best['validation_mse']:.6g}，验证线性插值 {best['validation_linear_mse']:.6g}。"
             f"末轮训练/验证均方误差为 {history[-1]['train_mse']:.6g}/{history[-1]['validation_mse']:.6g}。"
             + ("训练损失继续下降，而验证损失高于最佳轮，表明后续训练出现时间窗口过拟合。"
                if best["epoch"] < len(history) and history[-1]["train_mse"] < best["train_mse"] else ""),
             f"- 位移真值三维向量均方根不超过 1e-12 时，单帧相对 L2（欧氏范数误差比）留空。"
             f"本次近零真值帧数为 {linear['near_zero_truth_frames']}。", "",
             "## 分工况结果", "",
             "| 工况 | 线性 RMSE | 三次 RMSE | MLP RMSE | 线性相对 L2 | 三次相对 L2 | MLP 相对 L2 | MLP 优于线性帧数 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    keys = {(row["case"], row["method"]): row for row in summaries}
    for d in diagnostics:
        case = d["case"]
        methods = [keys[case, method] for method in METHODS]
        values = " | ".join(f"{row['rmse']:.6g}" for row in methods)
        relative = " | ".join(f"{100 * row['relative_l2']:.4f}%" if row["relative_l2"] is not None else "—"
                              for row in methods)
        lines.append(f"| {case} | {values} | {relative} | {d['mlp_wins_vs_linear']}/50 |")
    lines += ["", "## 残差分析", "",
              "理想修正量为真值减线性插值。下表比较模型修正与理想修正的全场范数比例和余弦相似度。",
              "| 工况 | 修正范数比 | 余弦相似度 |", "|---|---:|---:|"]
    for d in diagnostics:
        norm = d["correction_to_ideal_norm"]
        cosine = d["correction_cosine"]
        lines.append(f"| {d['case']} | {f'{norm:.3f}' if norm is not None else '—'} | "
                     f"{f'{cosine:.3f}' if cosine is not None else '—'} |")
    lines += ["", "当修正范数比大于两倍余弦相似度时，残差修正会增大全场平方误差。"
              "该条件来自修正后平方误差的展开。结果表中的分量误差、三维向量误差和最大误差均使用原始位移单位。",
              "", "## 产物", "",
              "- summary.csv 与 per_frame.csv 包含全局、工况、逐帧指标与实际输入帧。",
              "- residual_diagnostics.csv 包含残差量级、方向和逐帧胜率。",
              f"- figures 包含训练损失、逐帧均方根误差、关键节点曲线和 {args.error_ms} 毫秒三维点云误差。"
              "误差散点图绘制固定随机采样节点与各方法误差最大的节点，全部节点已参与评测。",
              "- predictions 保存三个方法的全部保留帧，以及节点 ID、帧索引和时间戳。"]
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def configure_plot_font():
    preferred = ("Noto Sans CJK SC", "Microsoft YaHei", "Noto Sans CJK JP",
                 "Noto Serif CJK SC", "Noto Serif CJK JP", "WenQuanYi Zen Hei")
    required = set(map(ord, "训练轮次均方误差位移有限元图例（毫秒）"))
    available = font_manager.fontManager.ttflist
    for family in preferred:
        match = next((font for font in available if font.name == family), None)
        if match and required <= ft2font.FT2Font(match.fname).get_charmap().keys():
            plt.rcParams["font.family"] = [family]
            plt.rcParams["axes.unicode_minus"] = False
            return family
    raise RuntimeError("Install a CJK font for the Chinese figure labels, such as Noto Sans CJK")


def report_and_plots(args, rows, summaries, diagnostics, hardware, model_info):
    configure_plot_font()
    figures = args.out / "figures"
    figures.mkdir(exist_ok=True)
    data_report(args)
    history, best = curves(args, figures, rows, summaries)
    field_plots(args, figures)
    experiment_report(args, summaries, diagnostics, hardware, model_info, history, best)
