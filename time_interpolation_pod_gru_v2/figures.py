"""Plot the training, comparison, latent, node, and spatial results."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import RANKS
from reporting import configure_plot_font


LABEL = {"truth": "有限元真值", "linear": "线性", "cubic": "局部三次",
         "mlp": "第一版残差 MLP", "pod_bigru": "全局 POD + BiGRU"}
COLOR = {"truth": "#1C2430", "linear": "#55748C", "cubic": "#118872",
         "mlp": "#9F5B9B", "pod_bigru": "#CF613D"}
METHODS = ("linear", "cubic", "mlp", "pod_bigru")


def save(fig, destination):
    fig.savefig(destination, dpi=135, facecolor="white")
    plt.close(fig)


def pod_figures(out, folder, selected):
    with (out / "pod_spectrum.csv").open(encoding="utf-8-sig", newline="") as f:
        spectrum = list(csv.DictReader(f))
    mode = np.array([int(row["mode"]) for row in spectrum])
    tail = np.array([float(row["eigenvalue"]) for row in spectrum])
    cumulative = np.cumsum(tail) / tail.sum()
    residual = np.r_[np.cumsum(tail[:0:-1])[::-1] / tail.sum(), 0.]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2), layout="constrained")
    axes[0].plot(mode[:16], 100 * cumulative[:16], marker=".", color="#176773")
    axes[0].set(xlabel="POD 阶数", ylabel="累计能量占比（%）", title="前 16 阶累计能量")
    axes[1].semilogy(mode[:80], residual[:80], color="#176773", linewidth=2)
    for rank in RANKS:
        axes[1].scatter(rank, residual[rank - 1], color=COLOR["pod_bigru"])
        axes[1].annotate(str(rank), (rank, residual[rank - 1]), xytext=(5, 6), textcoords="offset points")
    axes[1].set(xlabel="POD 阶数", ylabel="剩余能量占比", title="训练谱尾能量")
    save(fig, folder / "pod_energy.png")

    with (out / "pod_rank_metrics.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(7.3, 4.3), layout="constrained")
    for key, name in (("train_projection_rmse", "训练投影"), ("validation_projection_rmse", "验证投影")):
        ax.semilogy(RANKS, [float(row[key]) for row in rows], marker="o", label=name)
    ax.axvline(selected["rank"], color="0.5", linestyle=":", label="所选阶数")
    ax.set(xticks=RANKS, xlabel="POD 阶数", ylabel="全场均方根误差（原始单位）")
    ax.legend()
    save(fig, folder / "pod_reconstruction.png")


def loss_figure(out, folder, selected):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), layout="constrained")
    for rank in RANKS:
        path = out / "training" / f"rank{rank}" / "history.csv"
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        epoch = [int(row["epoch"]) for row in rows]
        axes[1].plot(epoch, [float(row["validation_field_rmse"]) for row in rows], label=f"{rank} 阶")
        if rank == selected["rank"]:
            for key, name in (("training_z_mse", "训练"), ("validation_z_mse", "验证")):
                axes[0].semilogy(epoch, [float(row[key]) for row in rows], label=name)
            axes[0].axvline(selected["selected_epoch"], color="0.5", linestyle=":", label="所选轮次")
    axes[0].set(xlabel="训练轮次", ylabel="归一化系数均方误差", title=f"所选 {selected['rank']} 阶 BiGRU")
    axes[1].set(xlabel="训练轮次", ylabel="验证全场均方根误差", title="各阶数验证曲线")
    for ax in axes:
        ax.legend()
    save(fig, folder / "training.png")


def comparison(folder, rows, summaries, cases):
    fig, ax = plt.subplots(figsize=(11.2, 4.4), layout="constrained")
    ticks = np.arange(len(cases) + 1)
    lookup = {(row["case"], row["method"]): row for row in summaries}
    for i, method in enumerate(METHODS):
        ax.bar(ticks + (i - 1.5) * .19, [lookup[case, method]["rmse"] for case in cases + ["ALL"]],
               width=.19, label=LABEL[method], color=COLOR[method])
    ax.set(xticks=ticks, xticklabels=cases + ["全部"], yscale="log",
           ylabel="全场均方根误差（原始单位）")
    ax.legend(ncol=2)
    save(fig, folder / "rmse_comparison.png")

    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.2), layout="constrained")
    for ax, case in zip(axes.flat, cases):
        for method in METHODS:
            selection = [row for row in rows if row["case"] == case and row["method"] == method]
            ax.plot([row["index"] for row in selection], [row["rmse"] for row in selection],
                    marker=".", color=COLOR[method], label=LABEL[method])
        ax.set(title=case, xlabel="时间（毫秒）", ylabel="全场均方根误差")
    axes[0, 0].legend(fontsize=8)
    save(fig, folder / "per_frame_rmse.png")


def coefficient_figure(folder, coefficients, cases):
    fig, axes = plt.subplots(len(cases), 3, figsize=(11.2, 11.8),
                             sharex=True, layout="constrained")
    for row, case in enumerate(cases):
        for col in range(3):
            ax = axes[row, col]
            values = [entry for entry in coefficients if entry["case"] == case and entry["mode"] == col + 1]
            for key, method in (("true", "truth"), ("predicted", "pod_bigru")):
                ax.plot([entry["index"] for entry in values], [entry[key] for entry in values],
                        marker=".", color=COLOR[method], label=LABEL[method])
            if row == 0:
                ax.set_title(f"第 {col + 1} 个 POD 系数")
            if col == 0:
                ax.set_ylabel(case)
            if row == len(cases) - 1:
                ax.set_xlabel("时间（毫秒）")
    axes[0, 0].legend(fontsize=8)
    save(fig, folder / "coefficients.png")


def node_figure(folder, traces, case):
    node_ids = sorted({row["node_id"] for row in traces})
    fig, axes = plt.subplots(3, len(node_ids), figsize=(10.8, 7.3), sharex=True, layout="constrained")
    for col, node in enumerate(node_ids):
        for row, component in enumerate(("dx", "dy", "dz")):
            ax = axes[row, col]
            for method in ("truth", *METHODS):
                values = [item for item in traces if item["node_id"] == node and
                          item["case"] == case and item["method"] == method]
                ax.plot([item["index"] for item in values], [item[component] for item in values],
                        color=COLOR[method], linestyle="-" if method == "truth" else "--",
                        marker=".", label=LABEL[method])
            if row == 0:
                ax.set_title(f"节点 {node}")
            if row == 2:
                ax.set_xlabel("时间（毫秒）")
            ax.set_ylabel(component.upper())
    axes[0, 0].legend(ncol=2, fontsize=8)
    fig.suptitle(f"{case} | 关键节点位移（原始单位）")
    save(fig, folder / "key_nodes.png")


def field_figure(folder, coords, deltas, case, time_ms):
    errors = [np.linalg.norm(delta, axis=1) for delta in deltas]
    rng = np.random.default_rng(20261111)
    sample = rng.choice(len(coords), min(8500, len(coords)), replace=False)
    picked = np.unique(np.concatenate([sample, *(np.argsort(error)[-160:] for error in errors)]))
    vmax = max(float(error.max()) for error in errors) or 1.
    extent = np.ptp(coords, axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), subplot_kw={"projection": "3d"},
                             layout="constrained")
    for ax, error, label in zip(axes, errors, ("POD 真值投影", LABEL["pod_bigru"], LABEL["cubic"])):
        points = ax.scatter(*coords[picked].T, c=error[picked], s=2., cmap="magma",
                            vmin=0, vmax=vmax, depthshade=False)
        ax.set(xlabel="X", ylabel="Y", zlabel="Z", title=label)
        ax.set_box_aspect(np.where(extent > 0, extent, 1.))
    fig.colorbar(points, ax=axes, shrink=.64, label="节点三维位移向量误差")
    fig.suptitle(f"{case} | {time_ms} 毫秒误差场（原始单位）")
    save(fig, folder / "field_error.png")


def write_report(out, selected, summary, components, coefficients, field_case, field_ms):
    overall = {row["method"]: row for row in summary if row["case"] == "ALL"}
    v2 = overall["pod_bigru"]
    v1 = overall["mlp"]
    reduction = 100 * (1 - v2["rmse"] / v1["rmse"])
    best_traditional = min(overall[method]["rmse"] for method in ("linear", "cubic"))
    winner = min(overall, key=lambda method: overall[method]["rmse"])
    diag = components[-1]
    lines = ["# 第二版全局 POD + 条件 BiGRU 时间内插", "",
             f"在六工况相同的 60 个测试全场上，第二版 RMSE（均方根误差）为 {v2['rmse']:.6g}，"
             f"第一版残差 MLP（多层感知机）为 {v1['rmse']:.6g}。"
             f"相对第一版{'降低' if reduction >= 0 else '升高'} {abs(reduction):.2f}%。"
             f"{LABEL[winner]} 的合并 RMSE 最低。", "",
             "| 方法 | RMSE | MAE（平均绝对误差） | 相对 L2（欧氏范数误差比） | 最大节点向量误差 | 单帧耗时中位数（毫秒） |",
             "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        row = overall[method]
        lines.append(f"| {LABEL[method]} | {row['rmse']:.6g} | {row['mae']:.6g} | "
                     f"{100 * row['relative_l2']:.4f}% | {row['max_node_vector_error']:.6g} | "
                     f"{row['inference_median_ms']:.3f} |")
    lines.extend(["", "绝对误差使用原始位移单位。前三种方法复用第一版相同测试帧的预测和对应时间记录，"
                   "统一从有限元真值重算误差。第二版计时包括四个全场观测的读取和投影、显卡时序预测与全场重构，排除写盘。",
                   "", "## 数据与选择", "",
                   "- 六工况共用一套全局 POD（本征正交分解）。训练奇数目标 180 个用于拟合均值和基，"
                   "同一集合计算系数的逐维均值与标准差。模型以四个偶数观测和时间特征为输入。",
                   "- 每个工况按相同的奇数目标时间块划分为训练 30 帧、验证 10 帧、测试 10 帧。"
                   "四帧观测在训练、验证与测试均按 2 毫秒间隔采样。",
                   "- 32 阶先验证完整全场闭环，再以同一种子训练 8、16、64 阶。"
                   "选择以验证全场均方根误差为准，训练损失为归一化系数均方误差。",
                   f"- 所选维度 {selected['rank']} 阶，最佳权重为第 {selected['selected_epoch']}/{selected['epochs']} 轮。"
                   f"最佳验证全场误差 {selected['validation_field_rmse']:.6g}，末轮 {selected['last_validation_field_rmse']:.6g}。",
                   "", "## 分工况全场均方根误差", "",
                   "| 工况 | 线性 | 局部三次 | 第一版残差 MLP | 第二版 BiGRU |",
                   "|---|---:|---:|---:|---:|"])
    cases = [row["case"] for row in components if row["case"] != "ALL"]
    lookup = {(row["case"], row["method"]): row for row in summary}
    for case in cases:
        result = " | ".join(f"{lookup[case, method]['rmse']:.6g}" for method in METHODS)
        lines.append(f"| {case} | {result} |")
    overfit = (selected["last_training_z_mse"] < selected["training_z_mse_at_best"] and
               selected["last_validation_field_rmse"] > 1.02 * selected["validation_field_rmse"])
    error_by_mode = np.bincount([row["mode"] for row in coefficients],
                                weights=[(row["predicted"] - row["true"])**2 for row in coefficients])
    top_modes = sorted(range(1, len(error_by_mode)), key=lambda mode: -error_by_mode[mode])[:2]
    mode_share = 100 * error_by_mode[top_modes].sum() / error_by_mode.sum()
    lines.extend(["", "## 误差来源", "",
                   f"测试真值投影后的 POD 重构 RMSE 为 **{diag['pod_projection_rmse']:.6g}**。"
                   f"低维系数预测 RMSE 为 **{diag['latent_rmse']:.6g}**，换算为全场时序分量 RMSE 为 "
                   f"**{diag['temporal_field_rmse']:.6g}**。截断和时序项分别约占全场平方误差的 "
                   f"**{100 * diag['pod_sse_fraction']:.3g}%** 和 **{100 * diag['temporal_sse_fraction']:.5g}%**。",
                   f"系数误差最集中的两个模态为第 {top_modes[0]} 和第 {top_modes[1]} 模态，"
                   f"合计贡献 {mode_share:.2f}% 的系数平方误差。",
                   f"最佳轮的归一化系数均方误差为训练 {selected['training_z_mse_at_best']:.6g}、"
                   f"验证 {selected['validation_z_mse_at_best']:.6g}、测试 {diag['latent_normalized_mse']:.6g}。",
                   f"相对最优传统插值，第二版的合并 RMSE 比值为 {v2['rmse'] / best_traditional:.3g}。"
                   + ("选中轮次之后训练损失继续下降而验证误差上升，表现出时间窗口过拟合。"
                      if overfit else "训练与验证曲线记录了选中轮次和末轮的误差变化。"),
                   "", "## 附件", "",
                   "- [数据划分协议](protocol.json)记录每个目标的四个观测帧。",
                   "- [降维指标](pod_rank_metrics.csv)、[维度选择](rank_selection.csv)、"
                   "[误差分解](error_decomposition.csv)、[逐帧结果](per_frame.csv)及[可视化图表](figures/)。",
                   f"- 全场误差图使用 {field_case} 的 {field_ms} 毫秒帧。全场预测数组和权重存放在服务器。"])
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def report(out, selected, summary, components, rows, coefficients, traces, coords, deltas, case, time_ms):
    configure_plot_font()
    folder = out / "figures"
    folder.mkdir(exist_ok=True)
    cases = [row["case"] for row in components if row["case"] != "ALL"]
    write_report(out, selected, summary, components, coefficients, case, time_ms)
    pod_figures(out, folder, selected)
    loss_figure(out, folder, selected)
    comparison(folder, rows, summary, cases)
    coefficient_figure(folder, coefficients, cases)
    node_figure(folder, traces, case)
    field_figure(folder, coords, deltas, case, time_ms)
