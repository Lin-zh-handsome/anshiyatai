#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可验证的小型合成机械位移样本；绝不冒充真实CAE结果。"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import math


def make_toy_data(out: Path, *, corrupt: str = "") -> tuple[Path, Path]:
    out.mkdir(parents=True, exist_ok=True)
    coord = out / '坐标_合成测试.txt'
    disp = out / '位移_合成测试.txt'
    ids = [12, 2, 17, 4, 9]
    xyz = [(1, 2, 3), (2, 3, -1), (4, 5, 0), (-1, 6, 3), (2, -5, 7)]
    b1 = [(0,0,0), (1,2,0), (0,1,-1), (1,0,1), (-1,2,3)]
    b2 = [(0,0,0), (2,0,1), (1,1,2), (0,-2,2), (2,1,0)]
    with (out/'time_mapping.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f); w.writerow(['time_label','time_s'])
        for k in range(6): w.writerow([f'Time {k+1}',k*0.01])
    with coord.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['NodeId','Dx','Dy','Dz'])  # 注意这里是坐标而非位移
        for i,x in zip(ids,xyz):
            w.writerow([i,*x])
    with disp.open('w',encoding='utf-8',newline='') as f:
        w = csv.writer(f)
        for k in range(6):
            if corrupt == 'missing_time' and k==4:
                continue
            w.writerow([f'Time {k+1}'])
            w.writerow(['节点ID', 'DX', 'DY', 'DZ'])
            a = math.sin(k/3)
            b = (k/5)**2
            order=list(range(len(ids)))
            if k % 2:
                order.reverse()  # 故意改变每帧节点行顺序
            if corrupt == 'missing_node' and k==3:
                order = order[:-1]
            if corrupt == 'duplicate_node' and k==3:
                order[-1] = order[0]
            for j in order:
                vals = [a*b1[j][d] + b*b2[j][d] for d in range(3)]
                if corrupt == 'nan' and k==3 and j==2:
                    vals[0]=float('nan')
                w.writerow([ids[j],*[f'{v:.14e}' for v in vals]])
    print(f'[SYNTHETIC/合成测试] 坐标：{coord}；位移：{disp}')
    return coord,disp

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('tests/_toy_data'))
    p.add_argument('--corrupt',choices=['','missing_time','missing_node','duplicate_node','nan'],default='')
    a=p.parse_args()
    make_toy_data(a.output,corrupt=a.corrupt)
