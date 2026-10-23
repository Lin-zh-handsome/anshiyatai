# Ubuntu运行说明

建议使用Python 3.10及以上。先执行`bash scripts/bootstrap_ubuntu.sh`，依赖安装在项目`.venv`内；锁定依赖见`outputs/v01/requirements.lock.txt`。

## 输入文件（不随仓库提供）

```text
data/
  raw/
    初始时刻各节点的坐标值.txt
    各时刻各节点的位移结果.txt
    时间点对应关系.xlsx
  time_mapping.csv
```

坐标表首行`NodeId,Dx,Dy,Dz`，后三列解释为初始空间坐标，不能当作初始位移。位移按`Time 1`等时间标签分块，每行`节点ID,DX,DY,DZ`；可有表头、科学计数法及UTF-8 BOM（字节顺序标记），同帧节点行允许乱序。缺帧、重复/缺失节点、未知节点及非有限数值均会失败退出。

时间映射CSV（逗号分隔文本）使用`time_label,time_s`两列，内容必须从原始时间对应表取得。本次实际映射为Time 1至Time 101，0至0.1秒，间隔0.001秒；本仓库未上传任何工程输入表。

## 执行

```bash
bash scripts/smoke_test.sh
# 可选：同时核对真实坐标和原始Excel工作簿/CSV时间映射
.venv/bin/python scripts/check_inputs.py

# 指定新输出目录，可保留仓库内的已发表结果
OUTPUT_DIR=outputs/local_run bash run_v01.sh

# 在同一输出目录复用已经完成且来源未变化的快照
OUTPUT_DIR=outputs/local_run bash run_v01.sh --reuse-snapshots
```

直接调用CLI（命令行接口）：

```bash
.venv/bin/python src/rom_pod_v1.py \
  --coordinates 'data/raw/初始时刻各节点的坐标值.txt' \
  --displacements 'data/raw/各时刻各节点的位移结果.txt' \
  --time-map data/time_mapping.csv \
  --output outputs/local_run --expected-nodes 267801 --expected-times 101 --rank 10
```

默认分块12000自由度，快照使用float32（单精度浮点数），内积累积、均值、模态与系数使用float64（双精度浮点数）。快照数据部分324574812字节；本次31个有效模态对应的均值、模态、系数共205696216字节。原始TXT（文本）1888223355字节，运行时另需临时空间，建议预留至少4GB磁盘。

无图形界面也能生成PNG（位图）。重新生成中文图表时，可将有使用权限的中文字体放在`assets/chinese.ttc`；字体不随仓库提供。已生成的四张真实结果图可直接下载。

`.gitignore`已排除data目录、参考资料、快照/模态/系数数组、运行环境、字体及临时归档。
