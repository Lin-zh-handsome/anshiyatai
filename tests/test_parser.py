"""验证合成数据解析/低秩基线；非真实机械臂计算。"""
from pathlib import Path
import sys
import numpy as np
import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from scripts.make_toy_data import make_toy_data
from src.rom_pod_v1 import read_coordinates, parse_displacements, snapshot_pod


def test_reads_id_permutation_and_rank_two(tmp_path):
    coord, disp = make_toy_data(tmp_path/'raw')
    ids, xyz = read_coordinates(coord)
    assert len(ids) == 5 and xyz.shape == (5,3)
    out=tmp_path/'run';out.mkdir()
    snapfile, labels = parse_displacements(disp,ids,out,6)
    snap=np.load(snapfile,mmap_mode='r')
    assert labels == [1,2,3,4,5,6]
    assert snap.shape==(6,5,3)
    assert np.max(np.abs(snap[0])) == 0
    eigenvalues, rank, mean, energy = snapshot_pod(snapfile,out,2,5)
    assert rank==2
    assert energy[1]>0.999999


@pytest.mark.parametrize('corrupt,reason', [
 ('missing_time','时间块缺失或顺序不匹配'),
 ('missing_node','节点记录不完整'),
 ('duplicate_node','节点ID重复'),
])
def test_reject_bad_data(tmp_path,corrupt,reason):
    coord, disp = make_toy_data(tmp_path/'raw',corrupt=corrupt)
    ids,_=read_coordinates(coord)
    out=tmp_path/'run';out.mkdir()
    with pytest.raises(ValueError,match=reason):
        parse_displacements(disp,ids,out,6)


def test_nan_is_rejected(tmp_path):
    """必须拒绝NaN/Inf，防止生成看似正常的POD输出。"""
    coord, disp = make_toy_data(tmp_path/'raw',corrupt='nan')
    ids,_=read_coordinates(coord)
    out=tmp_path/'run';out.mkdir()
    with pytest.raises(ValueError,match='NaN/Inf'):
        parse_displacements(disp,ids,out,6)
