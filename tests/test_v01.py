"""数值正确性与必要的数据失败路径；不使用真实工程数据生成替代真值。"""
from pathlib import Path
import sys
import csv
import json
import subprocess
import argparse
import numpy as np
import pytest
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from src.pod_engine import read_coordinates,read_time_map,parse_displacements,snapshot_pod,evaluate,ensure_snapshots
from scripts.make_toy_data import make_toy_data


def fixture(tmp_path,corrupt=''):
    coord,disp=make_toy_data(tmp_path/'raw',corrupt=corrupt)
    ids,_=read_coordinates(coord); out=tmp_path/'out'; out.mkdir()
    return coord,disp,ids,out


def test_known_rank_and_projection(tmp_path):
    coord,disp,ids,out=fixture(tmp_path)
    snap,_=parse_displacements(disp,ids,out,6)
    a=np.load(snap).reshape(6,-1).astype(float)
    vals,rank,mu,energy=snapshot_pod(snap,out,2,5)
    assert rank==2
    modes=np.load(out/'pod_modes.npy').reshape(rank,-1); coef=np.load(out/'pod_coefficients.npy')
    assert np.allclose(modes@modes.T,np.eye(2),atol=1e-9)
    assert np.allclose((a-mu)@modes.T,coef,atol=1e-9)
    _,singular,_=np.linalg.svd(a-a.mean(axis=0),full_matrices=False)
    np.testing.assert_allclose(vals[:2],singular[:2]**2,rtol=1e-10)
    assert np.linalg.norm(mu+coef@modes-a)/np.linalg.norm(a)<1e-7
    # 每帧倒序或乱序都必须以ID还原，与解析的数学真值逐项一致。
    for k in range(6):
        import math
        b1=np.array([(0,0,0),(1,2,0),(0,1,-1),(1,0,1),(-1,2,3)])
        b2=np.array([(0,0,0),(2,0,1),(1,1,2),(0,-2,2),(2,1,0)])
        expected=math.sin(k/3)*b1+(k/5)**2*b2
        np.testing.assert_allclose(a[k].reshape(5,3),expected,rtol=1e-7,atol=1e-7)
    labels,ts=read_time_map(coord.parent/'time_mapping.csv')
    summary,frames,curves=evaluate(snap,out,ids,labels,ts,2,rank,energy,[2,17],2)
    assert summary['rank_metrics'][-1]['relative_L2']<1e-7
    # rank1的误差应等于丢弃的奇异值平方和。
    expected=np.sqrt(np.sum(singular[1:]**2)/np.sum(a*a))
    assert summary['rank_metrics'][0]['relative_L2']==pytest.approx(expected,rel=1e-9)
    assert frames[0][4] is None  # 首帧为零，不能硬除零


@pytest.mark.parametrize('corrupt,reason',[
    ('missing_node','节点记录不完整'),('duplicate_node','节点ID重复'),
    ('missing_time','时间块缺失或顺序不匹配'),('nan','NaN/Inf')])
def test_required_rejections(tmp_path,corrupt,reason):
    _,disp,ids,out=fixture(tmp_path,corrupt)
    with pytest.raises(ValueError,match=reason): parse_displacements(disp,ids,out,6)
    assert not (out/'snapshots_manifest.json').exists()
    assert not (out/'snapshots.partial.npy').exists()


@pytest.mark.parametrize('replacement,reason', [('inf','NaN/Inf'),('1e100','超出float32'),('abc','数值解析失败')])
def test_invalid_numeric(tmp_path,replacement,reason):
    _,disp,ids,out=fixture(tmp_path)
    text=disp.read_text(); disp.write_text(text.replace('0.00000000000000e+00',replacement,1),encoding='utf-8')
    with pytest.raises(ValueError,match=reason): parse_displacements(disp,ids,out,6)


@pytest.mark.parametrize('constant', [0.,3.25])
def test_zero_energy(tmp_path,constant):
    _,disp,ids,out=fixture(tmp_path)
    with disp.open('w',encoding='utf-8-sig') as f:
        for k in range(6):
            f.write(f'Time {k+1}\n')
            for nid in ids: f.write(f'{nid},{constant},{constant},{constant}\n')
    snap,_=parse_displacements(disp,ids,out,6)
    vals,rank,mu,energy=snapshot_pod(snap,out,2,5)
    assert rank==0
    result,_,_=evaluate(snap,out,ids,list(range(1,7)),np.arange(6)*.01,2,rank,energy,[2],2)
    r=result['rank_metrics'][0]
    assert r['rmse']==0 and r['max_node_vector_error']==0
    assert r['relative_L2']==(None if constant==0 else 0.)
    assert r['worst_node_id'] in ids


def test_actual_time_mapping_bom_and_reuse(tmp_path):
    coord,disp,ids,out=fixture(tmp_path)
    mapping=coord.parent/'time_mapping.csv'
    mapping.write_text('time_label,time_s\n'+''.join(f'Time {i+1},{.5+i*.03}\n' for i in range(6)),encoding='utf-8-sig')
    labels,ts=read_time_map(mapping)
    assert ts[0]==.5 and ts[-1]==.65
    args=argparse.Namespace(coordinates=coord,displacements=disp,time_map=mapping,output=out,reuse_snapshots=False)
    snap,_=ensure_snapshots(args,ids,labels,ts); args.reuse_snapshots=True
    assert ensure_snapshots(args,ids,labels,ts)[0]==snap
    with disp.open('a') as f: f.write('\n')
    with pytest.raises(ValueError,match='来源'): ensure_snapshots(args,ids,labels,ts)


def test_unknown_id_and_missing_last_frame(tmp_path):
    _,disp,ids,out=fixture(tmp_path)
    text=disp.read_text(encoding='utf-8'); disp.write_text(text.replace('\n12,','\n999,',1),encoding='utf-8')
    with pytest.raises(ValueError,match='未知节点ID'): parse_displacements(disp,ids,out,6)
    disp.write_text(text.split('Time 6')[0],encoding='utf-8')
    with pytest.raises(ValueError,match='时间块数量不匹配'): parse_displacements(disp,ids,out,6)


def test_cli_failure_no_success_report(tmp_path):
    coord,disp,ids,out=fixture(tmp_path,'duplicate_node')
    result=subprocess.run([sys.executable,str(BASE/'src/rom_pod_v1.py'),'--coordinates',str(coord),
        '--displacements',str(disp),'--time-map',str(coord.parent/'time_mapping.csv'),'--output',str(out),'--synthetic'],capture_output=True,text=True)
    assert result.returncode==1
    assert not (out/'summary.json').exists() and not (out/'report.md').exists()
    assert '节点ID重复' in json.loads((out/'failure.json').read_text())['reason']
