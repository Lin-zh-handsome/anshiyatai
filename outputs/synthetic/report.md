# 安世亚太机械臂 V0.1 位移降阶与重建

**SYNTHETIC/合成测试**

**仅同工况训练内压缩与重建；不是跨载荷工况预测；不证明已解决强非线性 AI-ROM（人工智能降阶模型）。**

节点数：5；时间点：6；自由度：15；时间：0～0.05 s（秒）。位移：数据原单位，待客户确认。

## 输入与校验

初始坐标表Dx/Dy/Dz解释为X0/Y0/Z0。位移TXT按Time块逐行读取，NodeId（节点编号）与坐标表对齐。已检查每帧节点唯一、完整、有限数值、时间标签与实际CSV映射一致。未修改原始数据。

解析30条节点记录。float32（单精度浮点数）快照相对原TXT的最大分量绝对转换误差：1.004147339e-07。后续POD指标均相对该快照计算；这个转换误差单独列出。

```json

{
  "coordinates": {
    "path": "./outputs/synthetic_fixture/raw/坐标_合成测试.txt",
    "size_bytes": 67,
    "mtime_ns": 1792643064617628874
  },
  "time_map": {
    "path": "./outputs/synthetic_fixture/raw/time_mapping.csv",
    "size_bytes": 96,
    "mtime_ns": 1792643064616628864
  },
  "displacement": {
    "path": "./outputs/synthetic_fixture/raw/位移_合成测试.txt",
    "size_bytes": 2165,
    "mtime_ns": 1792643064617628874
  },
  "labels": [
    1,
    2,
    3,
    4,
    5,
    6
  ],
  "times_s": [
    0.0,
    0.01,
    0.02,
    0.03,
    0.04,
    0.05
  ]
}

```

## 方法与数值定义

X为时间×空间矩阵。每个空间分量沿时间求均值μ，Xc=X−μ。分块累积float64（双精度浮点数）时间Gram矩阵（内积矩阵）G=Xc Xcᵀ；G=QΛQᵀ，Φ=Λ⁻¹ᐟ²QᵀXc，A=QΛ¹ᐟ²，重建Xhat=μ+AᵣΦᵣ。模态按欧氏内积归一；模态无量纲，均值和系数保持位移数据原单位。无质量矩阵，不采用质量加权。

累计能量针对中心化快照；relative L2（相对L2误差）=‖Xhat−X‖F/‖X‖F，包含全部输入帧；RMSE（均方根误差）=sqrt(误差平方和/(T×N×3))。最大节点误差为三个分量误差的欧氏范数最大值。原始全零帧的逐帧相对误差留空，不硬除零。

总位移是sqrt(DX²+DY²+DZ²)。全场峰值在全部节点和时间上搜索；表中峰值时间误差比较原始全局峰值时刻与重建全局峰值时刻，峰值节点可能不同；并列峰值取首次出现。关键节点表另列各节点峰值。

存储比=原始float32快照字节数/[float64均值+截断模态+系数字节数]，不含文件头与元数据，也不代表求解加速比。磁盘保存全部有效模态供全量重建检查；该存储比表示仅保存指定阶数模型的理论存储比。

有效秩=2（特征值相对阈值1e-12）；选定重建阶数=2。零中心化能量时只保存均值，阶数0，能量比例无定义；逐帧绝对误差仍有效。

## 实测多阶结果

|阶数|中心化累计能量|RMSE（均方根误差）|相对L2误差|最大节点向量误差|峰值幅值绝对误差|峰值时间误差(s)|理论存储比|
|---:|---:|---:|---:|---:|---:|---:|---:|
|1|0.9409238754|0.1624324212|0.139600199|0.5944830298|0.3157627847|0|1.25|
|2|1|1.695356129e-08|1.457049345e-08|8.739852797e-08|1.817842943e-08|0|0.7894736842|

## 关键节点

实际全场峰值节点=9，幅值=4.344159093，时刻=0.05秒。

```json

[
  {
    "node_id": 2,
    "rank": 2,
    "actual_peak": 3.733070759399696,
    "reconstructed_peak": 3.733070746457852,
    "peak_amplitude_abs_error": 1.2941844040881278e-08,
    "actual_peak_time_s": 0.05,
    "reconstructed_peak_time_s": 0.05,
    "peak_time_abs_error_s": 0.0,
    "rmse": 1.3132261852600973e-08
  },
  {
    "node_id": 17,
    "rank": 2,
    "actual_peak": 2.4476229390908375,
    "reconstructed_peak": 2.447622947286913,
    "peak_amplitude_abs_error": 8.196075373945177e-09,
    "actual_peak_time_s": 0.05,
    "reconstructed_peak_time_s": 0.05,
    "peak_time_abs_error_s": 0.0,
    "rmse": 1.0780703100422268e-08
  },
  {
    "node_id": 9,
    "rank": 2,
    "actual_peak": 4.3441590925192815,
    "reconstructed_peak": 4.344159110697711,
    "peak_amplitude_abs_error": 1.8178429428417076e-08,
    "actual_peak_time_s": 0.05,
    "reconstructed_peak_time_s": 0.05,
    "peak_time_abs_error_s": 0.0,
    "rmse": 3.15607339165503e-08
  }
]

```

请求但不存在的节点：[]。合成样本不应包含或冒充真实机械臂节点。

## 实际资源与耗时

```json

{
  "stages": {
    "parse_or_reuse_s": 0.8145291149849072,
    "pod_s": 0.03147649299353361,
    "evaluation_s": 0.007062979973852634,
    "plot_s": 1.846618258045055,
    "total_s": 3.1133348789298907
  },
  "peak_rss_mib": 113.59765625,
  "process_cpu_seconds": 2.226580212
}

```

硬件、内存、磁盘、依赖与线程配置见environment.txt；锁定依赖见requirements.lock.txt。

## 图表和数组

01_pod_energy.png：累计能量；02_reconstruction_rmse.png：多阶逐帧误差；03_key_node_curves.png：关键节点三个分量及总位移；04_rank_errors.png：多阶全场误差。CSV表保留数值，NPY数组保存节点ID、快照、均值、模态和时间系数。

## 局限与下一阶段

客户资料描述22零件、线弹性、绑定连接、267801节点、169736单元、Newmark（纽马克时间积分）0.1秒。求解器类型不证明存在强非线性效应。未提供精确载荷、应力、速度、加速度，本文不推定这些数值。未执行跨工况或动态预测，也未依赖截图提取约束节点诊断。后续需要明确位移单位、Fx/Fy/Fz（X/Y/Z方向力）及单位和分配、多工况位移、独立留出工况；应力恢复另需单元连接、材料分配与数值应力标签。

## 复现命令

```bash

./.venv/bin/python src/rom_pod_v1.py --coordinates 'outputs/synthetic_fixture/raw/坐标_合成测试.txt' --displacements 'outputs/synthetic_fixture/raw/位移_合成测试.txt' --time-map outputs/synthetic_fixture/raw/time_mapping.csv --output outputs/synthetic --synthetic --rank 2 --node-ids 2,17 --feature-chunk 4

```

原始输入保持不变。可加--reuse-snapshots复用已完成快照；复用核对路径/大小/修改时间、节点ID顺序、时间映射、快照形状和类型，不使用哈希。
