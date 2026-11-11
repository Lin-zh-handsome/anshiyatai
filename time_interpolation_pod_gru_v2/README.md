# 全局 POD + 条件 BiGRU 瞬态位移内插

六个工况共享全局 POD（本征正交分解）空间基。分块 QR 与小矩阵 SVD（奇异值分解）只从 180 个训练目标位移场拟合均值和基。1 层双向 GRU（门控循环单元）读取四个已知位移帧的投影系数与相对时间偏移，融合目标时间和实际端点间隔，直接预测归一化 POD 系数。再将系数反归一化并重构位移场。

## 数据与划分

使用第一版 `data/prepared_neural_time_v1/` 的六工况缓存。每工况有 51 帧已知偶数毫秒观测和 50 帧可监督奇数目标，原始位移数组维度为 [时间, 节点, 3]。节点顺序与 `node_ids.npy` 一致。时间戳来自 `times_s.npy`。POD、归一化和网络训练使用奇数目标 1–19、41–59、81–99 毫秒，验证为 21–39 毫秒，最终保留测试为 61–79 毫秒。已知观测在所有阶段按 2 毫秒等距取四帧，首尾采用单侧四帧窗口。实际窗口写入 `protocol.json`。

评测需要第一版 `outputs/neural_time_interpolation_v1/predictions/` 中三种方法的完整预测及节点、时间映射，以及第一版的 `per_frame.csv` 记录。对三种预测均以相同的 60 个测试场重算误差。位移缓存或第一版预测缺失时，先按第一版 README 在服务器重建。

## 运行

依赖 Python、NumPy、Matplotlib 和支持 CUDA 的 PyTorch（张量计算框架）。绘图使用服务器已有的中文字体。下列命令从仓库根目录运行，示例指定一张可用卡。

```bash
export PYTHON=/home/robot/miniconda3/envs/shiyi/bin/python
export CUDA_VISIBLE_DEVICES=0
bash time_interpolation_pod_gru_v2/run.sh first
bash time_interpolation_pod_gru_v2/run.sh all
```

`first` 拟合全局 POD，训练 32 阶，并逐帧重构 60 个验证场验证全场闭环。`all` 复用完成的阶段，用同一固定种子补齐 8、16 和 64 阶。训练以归一化系数均方误差为损失，使用 AdamW（解耦权重衰减优化器）与显卡混合精度，依据验证全场误差早停并保存最佳轮。阶段中断后，重复 `all` 从当前检查点续训。最终用同一验证标准选阶数，再读取奇数测试目标。

单帧预测示例，输出包含时间、节点 ID、位移和系数。

```bash
"$PYTHON" time_interpolation_pod_gru_v2/predict_gru.py \
  --case XYZForce_1x --target-ms 73 --output /tmp/pod_gru_73ms.npz
```

## 结果

服务器目录为 `outputs/neural_time_interpolation_v2/`。POD 基、均值、权重和全场数组保存在服务器。预测数组形状为 [10, 节点数, 3]，对应 `predictions/target_indices.npy`、`target_times_s.npy`、`node_ids.npy`。

- [实验总结](../outputs/neural_time_interpolation_v2/summary.md)
- [数据划分协议](../outputs/neural_time_interpolation_v2/protocol.json)
- [各阶重构与选择](../outputs/neural_time_interpolation_v2/rank_selection.csv)
- [测试逐帧指标](../outputs/neural_time_interpolation_v2/per_frame.csv)
- [POD 截断与时序误差](../outputs/neural_time_interpolation_v2/error_decomposition.csv)
