"""Ubuntu只读检查：坐标与XLSX/CSV时间映射一致性。"""
import sys
import json
from pathlib import Path
import numpy as np
import openpyxl
sys.path.insert(0,str(Path.cwd()))
from src.pod_engine import read_coordinates,read_time_map
ids,xyz=read_coordinates(Path('data/raw/初始时刻各节点的坐标值.txt'))
labels,times=read_time_map(Path('data/time_mapping.csv'))
book=openpyxl.load_workbook('data/raw/时间点对应关系.xlsx',read_only=True,data_only=True)
rows=list(book.active.values)
print('XLSX前5行',rows[:5])
print('XLSX末3行',rows[-3:])
print('坐标节点数',len(ids),'坐标范围',xyz.min(axis=0).tolist(),xyz.max(axis=0).tolist())
print('CSV帧数',len(labels),'时间范围',times[0],times[-1])
numeric=[]
for row in rows:
    if len(row)>=2 and isinstance(row[1],(int,float)):
        label=str(row[0]).strip()
        if label.startswith('Time'): numeric.append((int(label.split()[-1]),float(row[1])))
if len(numeric)!=len(labels): raise ValueError('XLSX有效时间行数不匹配，需检查表格布局')
assert [r[0] for r in numeric]==labels
np.testing.assert_allclose([r[1] for r in numeric],times,rtol=0,atol=1e-12)
assert len(ids)==267801 and len(labels)==101
assert 56877 in ids and 173465 in ids
print('坐标与实际XLSX/CSV映射校验通过；关键节点存在。')
Path('outputs/input_check.json').write_text(json.dumps(dict(n_nodes=len(ids),n_time_steps=len(labels),xlsx_csv_match=True,key_ids_present=[56877,173465]),indent=2),encoding='utf-8')
