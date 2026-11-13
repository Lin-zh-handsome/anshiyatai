"""Create the actual-results report and figures from finalized metrics."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

LABEL = {"truth":"有限元真值","variable_tcn":"可变步长 TCN","fixed_1ms":"固定 1 ms TCN",
         "persistence":"保持位移","velocity":"常速度","pod_projection":"POD 真值投影"}
COLOR = {"truth":"#202631","variable_tcn":"#0e8073","fixed_1ms":"#92538b",
         "persistence":"#a19387","velocity":"#d1753e","pod_projection":"#7b98a8"}


def read_csv(path):
    with path.open(encoding="utf-8-sig",newline="") as stream:
        return list(csv.DictReader(stream))


def figures(out, metrics, seen, unseen):
    font = next((f for f in ("Noto Sans CJK SC","Noto Sans CJK JP")
                 if any(item.name==f for item in font_manager.fontManager.ttflist)),None)
    if font:
        plt.rcParams.update({"font.family":font,"axes.unicode_minus":False})
    dest = out/"figures"
    dest.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1,2,figsize=(11,4.4),constrained_layout=True)
    for ax,field,scale,title in zip(axes,("rmse","relative_l2"),(1.,100.),
                                   ("全场均方根误差（原位移单位）","全场相对 L2（%）")):
        for dt in unseen:
            ax.axvspan(dt-.21,dt+.21,color="#e7ecef",zorder=0)
        for method in ("persistence","velocity","variable_tcn","fixed_1ms","pod_projection"):
            rows = [r for r in metrics if r["method"]==method]
            x = [r["step_ms"] for r in rows]
            y = [r[field]*scale for r in rows]
            ax.plot(x,y,marker="o",color=COLOR[method],label=LABEL[method])
        ax.set(xticks=list(range(1,9)),xlabel="预测跨度（毫秒）",ylabel=title,yscale="log")
        ax.grid(axis="y",alpha=.22)
    axes[1].legend(fontsize=9)
    fig.suptitle("同轨迹单步预测 · 灰色带为未见跨度 3、5、7 ms")
    fig.savefig(dest/"error_vs_step.png",dpi=170)
    plt.close(fig)

    trace = read_csv(out/"key_node.csv")
    for group, steps in (("seen",seen),("unseen",unseen)):
        fig, axes = plt.subplots(3,len(steps),figsize=(3.4*len(steps),7.4),sharex="col",constrained_layout=True)
        for col,dt in enumerate(steps):
            selected = [row for row in trace if int(row["step_ms"])==dt]
            for i,component in enumerate(("dx","dy","dz")):
                ax=axes[i,col]
                for method in ("truth","variable_tcn","velocity","fixed_1ms"):
                    rows=[r for r in selected if r["method"]==method]
                    if rows:
                        ax.plot([int(r["target_ms"]) for r in rows],
                                [float(r[component]) for r in rows],color=COLOR[method],
                                linestyle="-" if method=="truth" else "--",
                                label=LABEL[method],linewidth=1.45)
                if col==0:
                    ax.set_ylabel(component.upper())
                if i==0:
                    ax.set_title(f"{dt} 毫秒")
                if i==2:
                    ax.set_xlabel("目标时刻（毫秒）")
                ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=8)
        fig.suptitle("节点 56877 · 位移使用原始单位")
        fig.savefig(dest/f"key_node_{group}.png",dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(9.4,4.3),constrained_layout=True)
    for ax,label,method in zip(axes,("fixed","variable"),("fixed_1ms","variable_tcn")):
        rows=read_csv(out/f"{label}_training.csv")
        ax.semilogy([int(r["epoch"]) for r in rows],
                    [float(r["macro_normalized_mse"]) for r in rows],color=COLOR[method])
        ax.set(title=LABEL[method],xlabel="轮次",ylabel="已见跨度归一化均方误差")
        ax.grid(alpha=.2)
    fig.savefig(dest/"training.png",dpi=160)
    plt.close(fig)


def generate(out):
    protocol=json.loads((out/"protocol.json").read_text())
    run=json.loads((out/"selection.json").read_text())
    results=json.loads((out/"summary.json").read_text())
    metrics=results["results"]
    lookup={(r["step_ms"],r["method"]):r for r in metrics}
    seen,unseen=protocol["train_steps_ms"],protocol["final_test_steps_ms"]
    pct=lambda value:f"{100*value:.5g}%"
    relative=lambda dt,m:lookup[dt,m]["relative_l2"]
    improved=[dt for dt in unseen if relative(dt,"variable_tcn")<relative(dt,"velocity")]
    fixed=lookup[1,"fixed_1ms"]
    lines=["# X 单轴工况时间步长内插 V2","",
           "训练和评估均以四帧真实历史初始化。每个目标只进行一次直接预测。"
           "所有时间段用于拟合，测试保留的是预测跨度 3、5、7 毫秒。","",
           "## 数据、模型与权重","",
           "- 复用 XForceOnly_1x 的 101 帧、267,801 个节点和三向位移。原始 51 点 X 向载荷每 2 毫秒采样，"
           "用分段线性插值对齐，保留原载荷值、位移单位和节点顺序。数据审计见 data_audit.json。",
           "- 以 0 毫秒为参考拟合一套全轨迹 POD（本征正交分解）基。所有预测方法使用相同目标时刻。"
           "状态保持和常速度由实际全场历史直接计算。",
           "- 条件 TCN（时序卷积网络）用两层因果卷积编码低维状态、历史时间间隔和 X 向载荷，"
           "用独立多层感知机编码 Δt，并融合未来区间起点、终点与平均载荷。更新由当前状态加上 Δt 乘变化率组成。",
           "- 每轮覆盖所有合法窗口，各已见跨度的归一化均方误差等权。固定模型先训练，仅使用 1 毫秒。"
           "可变模型只使用 1、2、4、6、8 毫秒，AdamW（解耦权重衰减优化器）、梯度裁剪与余弦学习率调度。"
           "以已见跨度归一化均方误差最低的权重定型，之后才计算 3、5、7 毫秒指标。","",
           "## POD 重构误差","",
           "| 模态数 | 全轨迹相对 L2 | 全轨迹 RMSE |","|---:|---:|---:|"]
    for row in protocol["pod_rank_candidates"]:
        lines.append(f"| {row['rank']} | {pct(row['projection_relative_l2'])} | {row['projection_rmse']:.6g} |")
    lines+=["",f"按事先选定的 {pct(protocol['pod_tolerance_relative_l2'])} 重构误差门槛，"
                f"选用 {protocol['rank']} 阶。POD 真值投影用于量化空间截断误差。","",
            "## 各跨度单步全场相对 L2","",
            "相对 L2 为误差场欧氏范数与真值场欧氏范数之比。RMSE（均方根误差）、最大节点向量误差、"
            "真值和预测的最大位移及其时刻误差均见 summary.csv。","",
            "| Δt（毫秒） | 分组 | 窗口 | 状态保持 | 常速度 | 固定 1 ms TCN | 可变步长 TCN | POD 投影 |",
            "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for dt in range(1,9):
        fixed_text=pct(relative(1,"fixed_1ms")) if dt==1 else "—"
        r=lookup[dt,"variable_tcn"]
        lines.append(f"| {dt} | {'已见' if dt in seen else '未见'} | {r['windows']} | "
                     f"{pct(relative(dt,'persistence'))} | {pct(relative(dt,'velocity'))} | "
                     f"{fixed_text} | {pct(r['relative_l2'])} | {pct(relative(dt,'pod_projection'))} |")
    floor=[relative(dt,"pod_projection") for dt in unseen]
    motion=[lookup[dt,"variable_tcn"]["dynamics_sse_fraction"] for dt in unseen]
    speed=[lookup[dt,"variable_tcn"]["latent_forward_median_ms"] for dt in range(1,9)]
    lines+=["","## 结论","",
            f"- 固定 1 毫秒模型在 97 个训练窗口上的全场 RMSE 为 {fixed['rmse']:.6g}，相对 L2 为 {pct(fixed['relative_l2'])}，"
            f"选中第 {run['fixed']['epoch']} 轮，归一化系数均方误差为 {run['fixed']['macro_normalized_mse']:.6g}。",
            f"- 可变步长模型选中第 {run['variable']['epoch']} 轮。在未见跨度中，优于常速度的跨度为 "
            + ("、".join(map(str,improved))+" 毫秒。" if improved else "零个。"),
            f"- 未见跨度的 POD 真值重构相对误差为 {pct(min(floor))}–{pct(max(floor))}。"
            f"可变模型的低维预测项占全场平方误差 {100*min(motion):.4f}%–{100*max(motion):.4f}%。",
            f"- {results['hardware']} 上，变步长模型的单样本低维前向中位数为 {min(speed):.3f}–{max(speed):.3f} 毫秒。"
            "计时使用已在设备上的张量，包含设备同步，编码、解码和文件读取未计入。各步长和固定模型计时见 summary.csv。",
            "- 结果只量化同一条已学习有限元轨迹的未见预测跨度。各窗口相互重叠。"
            "这些结果不代表独立轨迹、跨工况或有限元求解器数值积分步长的泛化。","",
            "## 结果文件","",
            "- summary.csv 与 per_frame.csv 分别记录汇总和逐窗口结果。key_node.csv 保存节点 56877 的三向位移。",
            "- pod_ranks.csv、protocol.json、selection.json 与训练 CSV 保存拟合与选模信息。",
            "- figures/ 保存步长误差、关键节点和训练曲线。"]
    (out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    figures(out,metrics,seen,unseen)


if __name__=="__main__":
    import sys
    generate(Path(sys.argv[1]))
