# 单工况时间步长内插 V2

在 XForceOnly_1x 的 101 帧上拟合一套全局 POD（本征正交分解）。用四个连续真实历史状态与已知 X 向载荷直接预测目标时刻。条件 TCN（时序卷积网络）由两层 32 通道因果卷积构成，卷积核 3、膨胀率 1 与 2。预测跨度除以 8 毫秒后单独输入 MLP（多层感知机），与历史特征和目标区间载荷融合。输出为当前 POD 系数加归一化跨度乘变化率，变化率包含最近两帧的常速度项与学习修正。POD 解码恢复节点位移。

## 协议

- 全轨迹 0–100 毫秒，位移每 1 毫秒一帧。采用原始 X 向载荷的 51 个采样点，分段线性对齐。数据源、节点映射和单位约定见输出目录的 data_audit.json。
- 先训练固定 1 毫秒模型，再训练 1、2、4、6、8 毫秒的条件模型。每轮覆盖全部合法窗口，各跨度归一化系数均方误差等权。
- 3、5、7 毫秒仅用于定型后评分。固定模型只在 1 毫秒报告。各样本使用真实历史，单独直接预测对应目标。所有窗口来自同一条训练轨迹。
- 速度和保持基线直接使用全场真实历史计算。POD 真值投影量化空间重构误差。

## 复现

依赖项目现有 data/prepared_v1/X 缓存、data/extracted_v1 的原始 X 向载荷，以及 time_rom_v1 和 load_driven_rom_v1 中的通用模块。从项目根目录运行。

```bash
export CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_NUM_THREADS=2
PY=/home/robot/miniconda3/envs/shiyi/bin/python
"$PY" variable_timestep_rom_v2/train.py
"$PY" variable_timestep_rom_v2/evaluate.py
"$PY" variable_timestep_rom_v2/predict.py --input forecast_input.npz --out prediction
```

训练和评测默认写入 outputs/variable_timestep_rom_v2，可用 --out 指定新目录。独立推理支持 --model-dir。NPZ 输入包含 node_ids、history（4 × 节点数 × 3，原位移单位）、history_ms（连续四个整数毫秒索引）、loads_x（从 0 毫秒起每毫秒一个原单位载荷）及 target_ms（一维整数索引数组）。每个目标均独立使用所给的同一份历史。输出 displacement.npy 和 mapping.npz。

## 交付

[实验报告](../outputs/variable_timestep_rom_v2/report.md)及汇总、逐帧误差、关键节点数据、训练日志和图表位于 outputs/variable_timestep_rom_v2。POD 基、参考状态、节点 ID 与最佳权重存于服务器原目录。
