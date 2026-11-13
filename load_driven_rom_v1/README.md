# 载荷驱动位移降阶模型

按《安世亚泰最终敲定版本》第 6–9 页实现。POD（本征正交分解）将 267,801 个节点的三向位移增量压缩为 8 个系数。TCN（时序卷积网络）读取最近 8 个低维状态及对应三向载荷，加入下一时间点的已知载荷，预测下一低维状态。模型将预测状态送回历史窗口，按 1 毫秒推进。POD 解码将预测系数恢复到全场位移。

## 训练与验证

- 训练工况为 X、Y、Z 三个单向工况，验证为 YXZ 2.5 倍，独立测试为 XYZ 2 倍。POD 基、系数尺度和载荷尺度均来自训练工况。选权重只用验证工况 8–100 毫秒的完整滚动误差。
- 所有载荷均使用原始表值。原表间隔 2 毫秒，以分段线性插值对齐 1 毫秒的位移时间轴。未施加的单向工况轴填零。组合工况倍率沿用供应方标签，载荷表未体现倍率。XYZ 1 倍与 2 倍载荷表相同，且位移不同，本版排除 XYZ 1 倍。
- 以训练增量场的相对 L2（欧氏范数误差比）不超过 0.2% 作为本次选阶目标，首个满足的候选为 8 阶。模型以训练工况的带载荷线性推进权重初始化跳连，训练时依次使用 1、8、24 步损失。最终验证选中第 10 轮。
- 整条测试工况只在 0–7 毫秒提供真实位移作为初始历史，之后 93 步均用预测状态继续推进。载荷时程已知。真值投影与逐步更新真实历史的单步结果仅用于评估诊断。

## 运行

从 `/home/robot/Project/anshiyatai` 执行。使用既有 `data/prepared_neural_time_v1`、`data/prepared_v1` 与原始载荷压缩包。环境为 `/home/robot/miniconda3/envs/shiyi`，依赖 NumPy、SciPy、PyTorch、Matplotlib、unrar。

```bash
export CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=4
PY=/home/robot/miniconda3/envs/shiyi/bin/python
$PY load_driven_rom_v1/train.py --output outputs/load_driven_rom_reproduce
$PY load_driven_rom_v1/evaluate_run.py --model-dir outputs/load_driven_rom_reproduce
$PY load_driven_rom_v1/predict.py --case XYZForce_2x --out /tmp/xyz_rom_prediction
```

外部推理用 `predict.py --input input.npz --out 输出目录`。输入包含 `node_ids`（训练网格顺序）、`times_s`（1 毫秒等距时间戳）、`history`（最初 8 帧，形状 [8, 267801, 3]）和 `loads`（对齐到时间轴的 [帧数, 3] 三向载荷）。位移和载荷均沿用源数据单位、坐标系和方向约定。输出 `displacement.npy` 为 [预测帧数, 节点数, 3]，`mapping.npz` 提供节点和时间对应关系。

## 本次交付

`outputs/load_driven_rom_v1/` 包含 POD 基、模型权重、训练协议、训练日志、预测全场、逐帧误差、关键节点曲线和 `report.md`。本版验证范围是固定网格的三向位移，现有数据仅支持这一输出。

## 模型结构与实测

训练工况的位移增量经 POD 投影得到低维状态 \(q_t\)。每次推进输入 8 帧低维历史及 9 个时刻的三轴载荷（历史 8 帧和下一步）。线性自回归支路接收展开的历史，因果卷积支路按时间读取低维状态和相邻时刻载荷，输出下一步的低维增量。将增量加到当前状态后作为下一步输入，再以 POD 基和首帧位移恢复全场。

| 部分 | 实现 |
|---|---|
| 空间编码与重构 | 267,801 个节点 × 3 个分量，保留 8 个 POD 模态 |
| 状态与载荷 | 8 × 8 个低维系数，9 × 3 个三轴载荷值 |
| 低维时序网络 | 两层 32 通道因果卷积，卷积核为 3，膨胀率分别为 1、2，随后用 1 × 1 卷积输出 8 个增量分量 |
| 时间推进 | 初始位移为 0–7 毫秒的 8 个时刻，之后每 1 毫秒推进，连续预测 8–100 毫秒的 93 帧 |

[实测报告](../outputs/load_driven_rom_v1/report.md) · [训练与数据协议](../outputs/load_driven_rom_v1/protocol.json) · [测试工况关键节点图](../outputs/load_driven_rom_v1/XYZForce_2x/key_node.png)

此提交记录首次实测所用的原始载荷和工况划分。POD 基、模型权重与全场位移结果保存在服务器。
