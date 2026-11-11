# 瞬态有限元时间步长内插

用同工况的两个完整节点位移场推断中间帧。输入是初始坐标、两端三分量位移、目标比例和端点时间差。残差 MLP（多层感知机）在线性插值上添加 `alpha * (1-alpha)` 加权的修正，端点与输入位移一致。模型使用 256 维输入投影、三层 256 维隐藏层、GELU（高斯误差线性单元）和零初始化输出层。训练使用 PyTorch、AdamW（解耦权重衰减优化器）、CUDA（显卡计算）混合精度和节点抽样。

## 数据

在现有 Ubuntu 项目根目录运行。原始 RAR 的第一分卷位于 `data/raw/updated_cases_2026-11-01/`。准备程序从实际成员名识别工况。将初始坐标 TXT 放到 `data/prepared_neural_time_v1/source_coordinates.txt`，每行 `NodeId,X,Y,Z`。供应方坐标表头 `NodeId,Dx,Dy,Dz` 也可读取。位移 TXT 中每个 `Time k` 标签后是 `NodeId,DX,DY,DZ`。`time_rom_v1/time_mapping.csv` 将标签映射到秒，与项目原始 XLSX 的映射一致。两类 TXT 的来源和含义各自独立。

准备程序复用来源一致的旧缓存并扫描其原始节点顺序。其余工况从压缩包流式解析一次。新缓存按初始坐标顺序存储，`known_even.npy` 为 [51, N, 3]，`truth_odd.npy` 为 [50, N, 3]。首维顺序对应 0、2、…、100 毫秒与 1、3、…、99 毫秒。根目录 `node_ids.npy` 与 `coordinates.npy` 对应第二维，`times_s.npy` 保存实际秒值。审计文件保存时间、节点数量、排列和极值检查。载荷与约束反力保存在原始档案中。

## 执行

依赖 Python 3.9+、NumPy、支持 CUDA 的 PyTorch 1.10+、Matplotlib 和 unrar（解压工具）或含 RAR 支持的 7-Zip。Ubuntu 工程内置的 `.tools/usr/bin/unrar-nonfree` 可直接使用。以 XLSX 替代已有时间 CSV 时需要 openpyxl。

```bash
cd /home/robot/Project/anshiyatai
export PYTHON=/home/robot/miniconda3/envs/shiyi/bin/python
bash time_interpolation_mlp_v1/run.sh prepare
CUDA_VISIBLE_DEVICES=0 bash time_interpolation_mlp_v1/run.sh experiment
```

`experiment` 先计算传统插值预测，再训练并以最低验证损失选择模型，生成模型预测后统一评测保留标签。重复执行会复用已完成的基线并从最近训练轮继续。训练中断可在相同运行目录重跑。可另行调用 `train.py --help` 查看配置，双卡可用时显式传入 `--gpus 2` 并限定两张可见显卡。

模型训练和验证只读取偶数帧位移，按原时间索引划分互斥时间块。训练时间块为 [0,16]、[30,46]、[60,76]、[90,100] 毫秒，验证为 [18,28]、[48,58]、[78,88] 毫秒。所有窗口的端点和目标都在同一块内。输入端点间隔为 4、6、8 毫秒。最终测试使用 2 毫秒端点间隔的中点。位移尺度从训练帧计算。按工况分别报告同工况内插结果。

线性与 MLP 使用目标两侧相邻偶数帧。局部三次通过四个偶数帧拟合拉格朗日插值多项式，首尾目标使用最前或最后四个偶数帧。逐帧 CSV 记录每种方法的实际输入索引。

## 单帧推理

传入与缓存节点顺序对应的两个 [N, 3] NumPy 位移数组。时间单位为秒。

```bash
"$PYTHON" time_interpolation_mlp_v1/predict.py query \
  --cache data/prepared_neural_time_v1 \
  --checkpoint outputs/neural_time_interpolation_v1/training/best.pt \
  --left u0.npy --right u2.npy --t0 0 --t1 0.002 --target 0.001 --out predicted_001ms.npz
```

## 输出

`outputs/neural_time_interpolation_v1/` 包含数据检查与实验报告、逐帧与汇总 CSV、图表、训练记录与权重。每个工况的三种全场预测保存在 `predictions/<工况>/<方法>.npy`，形状 [50, N, 3]。时间行对应 `target_indices.npy` 与 `target_times_s.npy`，节点列对应 `node_ids.npy`，最后一维顺序为 DX、DY、DZ。全场预测和指标使用原始位移单位。
