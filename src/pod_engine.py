"""位移快照的严格流式解析与时间快照法；只评价同工况训练内重建。"""
from __future__ import annotations
import argparse
import csv
import json
import logging
import math
import os
from pathlib import Path
import platform
import re
import resource
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np

LOG = logging.getLogger('rom')
TIME = re.compile(r'^Time\s*[,=:]?\s*(\d+)\s*$', re.I)
WARNING = '仅同工况训练内压缩与重建；不是跨载荷工况预测；不证明已解决强非线性 AI-ROM（人工智能降阶模型）。'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f); w.writerow(fields); w.writerows(rows)


def source_info(path):
    s = path.stat()
    return dict(path=str(path.resolve()), size_bytes=s.st_size, mtime_ns=s.st_mtime_ns)


def read_coordinates(path):
    ids, xyz = [], []
    with path.open(encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        if [v.strip().lower() for v in next(reader)] != ['nodeid', 'dx', 'dy', 'dz']:
            raise ValueError('坐标表头应为 NodeId,Dx,Dy,Dz（后三列为初始坐标）')
        for lineno, row in enumerate(reader, 2):
            if not row: continue
            if len(row) != 4: raise ValueError(f'坐标第{lineno}行不是4列')
            try: nid, vals = int(row[0]), [float(v) for v in row[1:]]
            except ValueError as e: raise ValueError(f'坐标第{lineno}行数值无效') from e
            if not all(math.isfinite(v) for v in vals): raise ValueError('坐标含NaN/Inf')
            if nid <= 0 or nid > np.iinfo(np.int64).max: raise ValueError('无效节点ID')
            ids.append(nid); xyz.append(vals)
    ids = np.array(ids, dtype=np.int64)
    if not len(ids) or len(np.unique(ids)) != len(ids): raise ValueError('坐标节点为空或节点ID重复')
    return ids, np.array(xyz, dtype=np.float64)


def read_time_map(path):
    labels, times = [], []
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['time_label', 'time_s']: raise ValueError('时间映射列名应为time_label,time_s')
        for row in reader:
            m = TIME.fullmatch(row['time_label'].strip())
            if not m: raise ValueError(f'无效时间标签 {row}')
            labels.append(int(m[1])); times.append(float(row['time_s']))
    ts = np.array(times)
    if len(labels) < 2 or len(set(labels)) != len(labels): raise ValueError('时间映射至少2帧且不得重复')
    if not np.isfinite(ts).all() or not np.all(np.diff(ts) > 0): raise ValueError('时间映射必须有限且严格递增')
    return labels, ts


def parse_displacements(path, ids, output, expected_times, labels=None):
    labels = list(range(1, expected_times + 1)) if labels is None else labels
    n = len(ids); index = {int(v): i for i, v in enumerate(ids)}
    partial = output / 'snapshots.partial.npy'; final = output / 'snapshots.npy'
    manifest = output / 'snapshots_manifest.json'
    manifest.unlink(missing_ok=True)
    source_before = source_info(path)
    snapshots = np.lib.format.open_memmap(partial, mode='w+', dtype='float32', shape=(expected_times, n, 3))
    seen = np.zeros(n, dtype=bool); block = -1; count = 0; actual = []; header_seen = False
    max_quantization = 0.; rows = 0
    frame64 = np.empty((n,3), dtype=np.float64)
    float32_limit = float(np.finfo(np.float32).max)
    def finish():
        nonlocal max_quantization
        if block < 0: return
        if count != n:
            raise ValueError(f'Time {actual[-1]} 节点记录不完整：已读{count}/{n}，缺失ID示例{ids[~seen][:10].tolist()}')
        # 只暂存一帧双精度数值，整帧转换可避免2700万次逐分量量化误差循环。
        converted = frame64.astype(np.float32)
        max_quantization = max(max_quantization, float(np.max(np.abs(frame64-converted))))
        snapshots[block] = converted
        LOG.info('解析 Time %s: %s节点（%s/%s帧）', actual[-1], count, block+1, expected_times)
    try:
        with path.open(encoding='utf-8-sig', errors='strict', buffering=4*1024**2) as f:
            for lineno, raw in enumerate(f, 1):
                line = raw.strip()
                if not line: continue
                m = TIME.fullmatch(line)
                if m:
                    finish(); label = int(m[1]); block += 1
                    if label in actual: raise ValueError(f'第{lineno}行时间块重复 Time {label}')
                    if block >= expected_times: raise ValueError(f'时间块数量超过映射{expected_times}帧')
                    if label != labels[block]: raise ValueError(f'时间块缺失或顺序不匹配：期望Time {labels[block]}，实际Time {label}，第{lineno}行')
                    actual.append(label); seen.fill(False); count = 0; header_seen = False
                    continue
                parts = [v.strip() for v in line.split(',')]
                if len(parts) == 4 and [v.lower().replace(' ', '') for v in parts] in (
                        ['节点id', 'dx', 'dy', 'dz'], ['nodeid', 'dx', 'dy', 'dz']):
                    if block < 0 or count or header_seen: raise ValueError(f'第{lineno}行表头位置错误')
                    header_seen = True; continue
                if block < 0: raise ValueError(f'第{lineno}行在Time标签之前含未知内容')
                if len(parts) != 4: raise ValueError(f'第{lineno}行应有4列，实际{len(parts)}列')
                try:
                    nid = int(parts[0]); vals = tuple(map(float, parts[1:]))
                except ValueError as e: raise ValueError(f'第{lineno}行数值解析失败') from e
                if not all(math.isfinite(v) for v in vals): raise ValueError(f'第{lineno}行 NaN/Inf，Time {actual[-1]}，节点{nid}')
                if max(abs(v) for v in vals) > float32_limit: raise ValueError(f'第{lineno}行超出float32（单精度浮点数）范围')
                j = index.get(nid)
                if j is None: raise ValueError(f'第{lineno}行未知节点ID {nid}')
                if seen[j]: raise ValueError(f'第{lineno}行节点ID重复：Time {actual[-1]}，节点{nid}')
                seen[j] = True; frame64[j] = vals; count += 1; rows += 1
        finish()
        if len(actual) != expected_times: raise ValueError(f'时间块数量不匹配：已读{len(actual)}/{expected_times}，缺失{labels[len(actual):]}')
        if source_info(path) != source_before: raise ValueError('解析期间输入文件发生变化')
        snapshots.flush(); del snapshots
        partial.replace(final)
        result = dict(source=source_before, node_count=n, time_labels=actual,
                      npy_shape=[expected_times,n,3], dtype='float32', rows=rows,
                      max_float32_conversion_abs_error=max_quantization,
                      snapshots=source_info(final), status='validated')
        write_json(manifest,result)
        return final, actual
    except Exception:
        del snapshots
        partial.unlink(missing_ok=True)
        raise


def ensure_snapshots(args, ids, labels, times):
    out = args.output
    provenance = dict(coordinates=source_info(args.coordinates), time_map=source_info(args.time_map),
                      displacement=source_info(args.displacements), labels=labels, times_s=times.tolist())
    if not args.reuse_snapshots:
        snap, _ = parse_displacements(args.displacements, ids, out, len(labels), labels)
        manifest = json.loads((out/'snapshots_manifest.json').read_text())
        manifest['provenance'] = provenance
        np.save(out/'node_ids.npy', ids)
        write_json(out/'snapshots_manifest.json', manifest)
        return snap, manifest
    manifest = json.loads((out/'snapshots_manifest.json').read_text())
    snap = out/'snapshots.npy'
    if manifest.get('status') != 'validated' or manifest.get('provenance') != provenance:
        raise ValueError('快照来源记录已变化，必须重新解析')
    if manifest['snapshots'] != source_info(snap): raise ValueError('快照文件大小或修改时间已变化')
    if not np.array_equal(np.load(out/'node_ids.npy'), ids): raise ValueError('快照节点ID顺序不匹配')
    a = np.load(snap,mmap_mode='r')
    if a.shape != (len(labels),len(ids),3) or a.dtype != np.float32: raise ValueError('快照形状或类型不匹配')
    return snap, manifest


def snapshot_pod(snap_file, output, n_modes, feature_chunk):
    x = np.load(snap_file, mmap_mode='r'); nt,n,_ = x.shape; x = x.reshape(nt,-1); d=x.shape[1]
    mu = np.empty(d, dtype=np.float64); gram=np.zeros((nt,nt))
    for start in range(0,d,feature_chunk):
        end=min(d,start+feature_chunk); b=np.array(x[:,start:end],dtype=np.float64)
        if not np.isfinite(b).all(): raise ValueError('快照含NaN/Inf')
        mu[start:end]=b.mean(axis=0); b-=mu[start:end]; gram+=b@b.T
    vals,q=np.linalg.eigh((gram+gram.T)*.5); vals=vals[::-1]; q=q[:,::-1]
    scale=max(float(np.max(np.abs(vals))),np.finfo(float).tiny)
    if vals.min() < -scale*1e-10: raise ValueError('时间Gram矩阵（内积矩阵）出现显著负特征值')
    vals=np.maximum(vals,0); tol=scale*1e-12
    effective=min(int(np.count_nonzero(vals>tol)),nt-1) if vals[0]>0 else 0
    # 模态的符号固定为时间特征向量最大绝对值分量为正，便于复现。
    q*=np.where(q[np.argmax(np.abs(q),axis=0),np.arange(nt)]<0,-1,1)
    sigma=np.sqrt(vals[:effective]); coeff=q[:,:effective]*sigma
    basis=np.lib.format.open_memmap(output/'pod_modes.npy',mode='w+',dtype='float64',shape=(effective,n,3)) if effective else np.empty((0,n,3))
    if effective:
        flat=basis.reshape(effective,d)
        for start in range(0,d,feature_chunk):
            end=min(d,start+feature_chunk); b=np.array(x[:,start:end],dtype=np.float64)-mu[start:end]
            flat[:,start:end]=(q[:,:effective].T@b)/sigma[:,None]
        basis.flush()
    else: np.save(output/'pod_modes.npy',basis)
    np.save(output/'mean_displacement.npy',mu.reshape(n,3)); np.save(output/'pod_coefficients.npy',coeff)
    np.save(output/'eigenvalues.npy',vals)
    energy=np.cumsum(vals)/vals.sum() if vals.sum()>0 else np.zeros(nt)
    write_csv(output/'pod_spectrum.csv',['rank','eigenvalue','singular_value','cumulative_energy'],
              [[i+1,float(v),float(np.sqrt(v)),float(energy[i])] for i,v in enumerate(vals)])
    LOG.info('POD（本征正交分解）有效秩=%s，特征值相对截断阈值=1e-12',effective)
    return vals,effective,mu,energy


def evaluate(snap_file, out, ids, labels, ts, requested_rank, effective, energy, node_ids, chunk_nodes):
    u=np.load(snap_file,mmap_mode='r'); nt,n,_=u.shape
    modes=np.load(out/'pod_modes.npy',mmap_mode='r'); coeff=np.load(out/'pod_coefficients.npy')
    mu=np.load(out/'mean_displacement.npy'); chosen=min(requested_rank,effective)
    ranks=sorted({r for r in [1,2,5,10,20,40,chosen,effective] if 0<=r<=effective})
    stats={r:dict(sse=np.zeros(nt),norm=np.zeros(nt),maxerr=np.full(nt,-1.),maxids=np.zeros(nt,dtype=np.int64),peak=np.full(nt,-1.),peakids=np.zeros(nt,dtype=np.int64)) for r in ranks}
    actual_peak=np.full(nt,-1.); actual_peak_ids=np.zeros(nt,dtype=np.int64)
    for start in range(0,n,chunk_nodes):
        end=min(n,start+chunk_nodes); a=np.array(u[:,start:end],dtype=np.float64)
        norm=np.sum(a*a,axis=(1,2)); total=np.linalg.norm(a,axis=2); js=np.argmax(total,axis=1); values=total[np.arange(nt),js]
        mask=values>actual_peak; actual_peak[mask]=values[mask]; actual_peak_ids[mask]=ids[start+js[mask]]
        reconstructed=np.broadcast_to(mu[start:end],a.shape).copy()
        for r in range(effective+1):
            if r: reconstructed+=coeff[:,r-1,None,None]*modes[r-1,start:end][None,:,:]
            if r not in stats: continue
            s=stats[r]; diff=reconstructed-a; s['sse']+=np.sum(diff*diff,axis=(1,2)); s['norm']+=norm
            err=np.linalg.norm(diff,axis=2); jj=np.argmax(err,axis=1); vv=err[np.arange(nt),jj]; better=vv>s['maxerr']
            s['maxerr'][better]=vv[better]; s['maxids'][better]=ids[start+jj[better]]
            total_r=np.linalg.norm(reconstructed,axis=2); jj=np.argmax(total_r,axis=1); vv=total_r[np.arange(nt),jj]; better=vv>s['peak']
            s['peak'][better]=vv[better]; s['peakids'][better]=ids[start+jj[better]]
        if start==0 or end==n or start%(chunk_nodes*10)==0: LOG.info('重建评估 %s/%s 节点',end,n)
    peak_t=int(np.argmax(actual_peak)); maxnode=int(actual_peak_ids[peak_t]); index={int(v):i for i,v in enumerate(ids)}
    selected=list(dict.fromkeys([v for v in node_ids if v in index]+[maxnode])); unavailable=[v for v in node_ids if v not in index]
    metrics=[]; per_time=[]; key_rows=[]; key_summary=[]
    for r,s in stats.items():
        norm=float(s['norm'].sum()); sse=float(s['sse'].sum()); pred_t=int(np.argmax(s['peak'])); err_t=int(np.argmax(s['maxerr']))
        compressed_bytes=8*(n*3*(r+1)+nt*r); original_bytes=4*nt*n*3
        row=dict(rank=r,cumulative_energy=float(energy[r-1]) if r else None,
                 rmse=float(np.sqrt(sse/(nt*n*3))),relative_L2=float(np.sqrt(sse/norm)) if norm>0 else None,
                 max_node_vector_error=float(s['maxerr'][err_t]),worst_node_id=int(s['maxids'][err_t]),worst_time_s=float(ts[err_t]),
                 actual_peak_amplitude=float(actual_peak[peak_t]),reconstructed_peak_amplitude=float(s['peak'][pred_t]),
                 peak_amplitude_abs_error=float(abs(s['peak'][pred_t]-actual_peak[peak_t])),
                 peak_time_abs_error_s=float(abs(ts[pred_t]-ts[peak_t])),actual_peak_time_s=float(ts[peak_t]),
                 reconstructed_peak_time_s=float(ts[pred_t]),original_float32_bytes=original_bytes,
                 compressed_float64_bytes=compressed_bytes,theoretical_storage_ratio=original_bytes/compressed_bytes)
        metrics.append(row)
        for k in range(nt):
            per_time.append([r,labels[k],float(ts[k]),float(np.sqrt(s['sse'][k]/(n*3))),
                             float(np.sqrt(s['sse'][k]/s['norm'][k])) if s['norm'][k]>0 else None,
                             float(s['maxerr'][k]),int(s['maxids'][k]),float(actual_peak[k]),float(s['peak'][k])])
    for nid in selected:
        j=index[nid]; a=np.array(u[:,j],dtype=np.float64)
        b=mu[j]+coeff[:,:chosen]@modes[:chosen,j] if chosen else np.broadcast_to(mu[j],a.shape).copy()
        at=np.linalg.norm(a,axis=1); bt=np.linalg.norm(b,axis=1); ia=int(np.argmax(at)); ib=int(np.argmax(bt))
        key_summary.append(dict(node_id=nid,rank=chosen,actual_peak=float(at[ia]),reconstructed_peak=float(bt[ib]),
                                peak_amplitude_abs_error=float(abs(bt[ib]-at[ia])),actual_peak_time_s=float(ts[ia]),
                                reconstructed_peak_time_s=float(ts[ib]),peak_time_abs_error_s=float(abs(ts[ia]-ts[ib])),
                                rmse=float(np.sqrt(np.mean((b-a)**2)))))
        key_rows.extend([[labels[k],float(ts[k]),nid,*a[k].tolist(),float(at[k]),*b[k].tolist(),float(bt[k])] for k in range(nt)])
    write_csv(out/'rank_metrics.csv',list(metrics[0]),[list(r.values()) for r in metrics])
    write_csv(out/'per_time_metrics.csv',['rank','time_label','time_s','RMSE_components','relative_L2','max_node_vector_error','worst_node_id','max_total_actual','max_total_POD'],per_time)
    write_csv(out/'key_node_curves.csv',['time_label','time_s','node_id','DX_actual','DY_actual','DZ_actual','total_actual','DX_POD','DY_POD','DZ_POD','total_POD'],key_rows)
    write_csv(out/'key_node_metrics.csv',list(key_summary[0]),[list(r.values()) for r in key_summary])
    return dict(selected_rank=chosen,rank_metrics=metrics,key_nodes=key_summary,missing_requested_node_ids=unavailable,
                global_peak_node_id=maxnode,global_peak_amplitude=float(actual_peak[peak_t]),global_peak_time_s=float(ts[peak_t])),per_time,key_rows


def create_plots(out,energy,evaluation,per_time,key_rows,synthetic):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font=Path(__file__).resolve().parents[1]/'assets'/'chinese.ttc'
    if font.exists(): font_manager.fontManager.addfont(str(font)); plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'savefig.dpi':150})
    stamp='SYNTHETIC/合成测试' if synthetic else '真实输入/训练内重建'
    def save(fig,name): fig.tight_layout(); fig.savefig(out/name); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4)); ax.plot(np.arange(1,len(energy)+1),energy*100,'.-'); ax.set(xlabel='模态数',ylabel='累计中心化能量 (%)',title=stamp+'：POD（本征正交分解）能量'); ax.grid(alpha=.3); save(fig,'01_pod_energy.png')
    fig,ax=plt.subplots(figsize=(8,4))
    for r in evaluation['rank_metrics']:
        rows=[v for v in per_time if v[0]==r['rank']]; ax.plot([v[2] for v in rows],[v[3] for v in rows],label=f"r={r['rank']}")
    ax.set(xlabel='时间 (s/秒)',ylabel='RMSE（均方根误差，数据原单位）',title=stamp+'：逐帧误差'); ax.legend(); ax.grid(alpha=.3); save(fig,'02_reconstruction_rmse.png')
    fig,ax=plt.subplots(figsize=(8,4)); rows=evaluation['rank_metrics']; ax.semilogy([v['rank'] for v in rows],[max(v['rmse'],1e-30) for v in rows],'o-'); ax.set(xlabel='模态数',ylabel='RMSE（均方根误差）',title=stamp+'：多阶重建误差'); ax.grid(alpha=.3); save(fig,'04_rank_errors.png')
    fig,axs=plt.subplots(len(evaluation['key_nodes']),4,figsize=(17,3.1*len(evaluation['key_nodes'])),squeeze=False)
    for axes,info in zip(axs,evaluation['key_nodes']):
        a=np.array([v for v in key_rows if v[2]==info['node_id']])
        for j,ax in enumerate(axes):
            ax.plot(a[:,1],a[:,3+j],label='原始位移'); ax.plot(a[:,1],a[:,7+j],'--',label='POD（本征正交分解）重建')
            ax.set(xlabel='时间 (s/秒)',ylabel='数据原单位（待客户确认）',title=f"节点{info['node_id']} / "+['DX（X向位移）','DY（Y向位移）','DZ（Z向位移）','总位移'][j]); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.suptitle(stamp+f"：关键节点，r={evaluation['selected_rank']}"); save(fig,'03_key_node_curves.png')


def environment_text():
    import importlib
    lines=[platform.platform(),sys.version, f'CPU（中央处理器）逻辑核数: {os.cpu_count()}',f'工作路径: {Path.cwd()}']
    for command in [['free','-h'],['df','-h','.'],['lscpu']]:
        p=subprocess.run(command,capture_output=True,text=True); lines.append(p.stdout)
    for name in ['numpy','scipy','matplotlib','pytest','openpyxl']:
        m=importlib.import_module(name); lines.append(f'{name}=={m.__version__}')
    lines.extend(f'{k}={os.environ.get(k,"未指定")}' for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'])
    return '\n'.join(lines)


def main():
    ap=argparse.ArgumentParser(description='训练内位移场POD（本征正交分解）与重建')
    for flag in ['coordinates','displacements','time-map']: ap.add_argument('--'+flag,required=True,type=Path)
    ap.add_argument('--output',type=Path,default=Path('outputs/v01'))
    ap.add_argument('--rank',type=int,default=10); ap.add_argument('--feature-chunk',type=int,default=12000)
    ap.add_argument('--node-ids',default='56877,173465'); ap.add_argument('--reuse-snapshots',action='store_true')
    ap.add_argument('--synthetic',action='store_true'); ap.add_argument('--expected-nodes',type=int); ap.add_argument('--expected-times',type=int)
    args=ap.parse_args(); out=args.output.resolve(); args.output=out
    if args.rank<1 or args.feature_chunk<3: ap.error('rank（阶数）必须大于0，分块自由度数至少3')
    for path in [args.coordinates,args.displacements,args.time_map]:
        if out==path.resolve() or out in path.resolve().parents: ap.error('输入文件必须位于输出目录之外')
    out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(out/'run.log',encoding='utf-8')])
    # 新运行开始即清理旧成功标记；失败时保留诊断，不保留旧成功报告。
    for filename in ['summary.json','report.md','failure.json']: (out/filename).unlink(missing_ok=True)
    start=time.perf_counter(); stages={}
    try:
        (out/'environment.txt').write_text(environment_text(),encoding='utf-8')
        freeze=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True)
        (out/'requirements.lock.txt').write_text(freeze.stdout,encoding='utf-8')
        ids,_=read_coordinates(args.coordinates); labels,ts=read_time_map(args.time_map)
        if args.expected_nodes is not None and len(ids)!=args.expected_nodes: raise ValueError(f'节点数{len(ids)}不符预期{args.expected_nodes}')
        if args.expected_times is not None and len(labels)!=args.expected_times: raise ValueError('映射帧数不符预期')
        if shutil.disk_usage(out).free < len(ids)*len(labels)*3*16: raise ValueError('可用磁盘不足以保存快照和全部模态')
        LOG.info('输入：%s节点 × %s帧，%.9g至%.9g秒',len(ids),len(labels),ts[0],ts[-1]); t=time.perf_counter()
        snap,manifest=ensure_snapshots(args,ids,labels,ts); stages['parse_or_reuse_s']=time.perf_counter()-t
        t=time.perf_counter(); vals,effective,mu,energy=snapshot_pod(snap,out,args.rank,args.feature_chunk); stages['pod_s']=time.perf_counter()-t
        t=time.perf_counter(); evaluation,per_time,key_rows=evaluate(snap,out,ids,labels,ts,args.rank,effective,energy,[int(v) for v in args.node_ids.split(',')],max(1,args.feature_chunk//3)); stages['evaluation_s']=time.perf_counter()-t
        t=time.perf_counter(); create_plots(out,energy,evaluation,per_time,key_rows,args.synthetic); stages['plot_s']=time.perf_counter()-t
        stages['total_s']=time.perf_counter()-start
        chosen=next(v for v in evaluation['rank_metrics'] if v['rank']==evaluation['selected_rank'])
        summary=dict(status='success',data_kind='SYNTHETIC/合成测试' if args.synthetic else 'REAL/真实输入',warning=WARNING,
                     n_nodes=len(ids),n_time_steps=len(labels),total_displacement_dofs=len(ids)*3,
                     time_start_s=float(ts[0]),time_end_s=float(ts[-1]),time_labels=labels,time_s=ts.tolist(),
                     displacement_unit='数据原单位，待客户确认',centering='逐空间分量对所有时间帧求时间均值',
                     effective_rank=effective,eigenvalue_relative_tolerance=1e-12,centered_energy=float(vals.sum()),
                     stored_pod_modes=effective,
                     overall_relative_L2_reconstruction_error=chosen['relative_L2'],
                     full_effective_rank_check=evaluation['rank_metrics'][-1],
                     mode_count_for_energy={str(v):int(np.searchsorted(energy,v)+1) if effective and np.any(energy>=v) else None for v in [.99,.999,.9999]},
                     peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,stage_seconds=stages,
                     process_cpu_seconds=time.process_time(),input_validation=manifest,**evaluation)
        thread_env=' '.join(k+'='+shlex.quote(os.environ[k]) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'] if k in os.environ)
        command='cd '+shlex.quote(str(Path.cwd()))+' && '+thread_env+' '+' '.join(shlex.quote(x) for x in [sys.executable,*sys.argv]); summary['reproduction_command']=command
        write_json(out/'summary.json',summary)
        report=['# 安世亚太机械臂 V0.1 位移降阶与重建',f"**{summary['data_kind']}**",'**'+WARNING+'**',
                f"节点数：{len(ids):,}；时间点：{len(labels)}；自由度：{len(ids)*3:,}；时间：{ts[0]:.9g}～{ts[-1]:.9g} s（秒）。位移：数据原单位，待客户确认。",
                '## 输入与校验', '初始坐标表Dx/Dy/Dz解释为X0/Y0/Z0。位移TXT按Time块逐行读取，NodeId（节点编号）与坐标表对齐。已检查每帧节点唯一、完整、有限数值、时间标签与实际CSV映射一致。未修改原始数据。',
                f"解析{manifest['rows']:,}条节点记录。float32（单精度浮点数）快照相对原TXT的最大分量绝对转换误差：{manifest['max_float32_conversion_abs_error']:.10g}。后续POD指标均相对该快照计算；这个转换误差单独列出。",
                '```json',json.dumps(manifest.get('provenance',{}),ensure_ascii=False,indent=2),'```',
                '## 方法与数值定义',
                'X为时间×空间矩阵。每个空间分量沿时间求均值μ，Xc=X−μ。分块累积float64（双精度浮点数）时间Gram矩阵（内积矩阵）G=Xc Xcᵀ；G=QΛQᵀ，Φ=Λ⁻¹ᐟ²QᵀXc，A=QΛ¹ᐟ²，重建Xhat=μ+AᵣΦᵣ。模态按欧氏内积归一；模态无量纲，均值和系数保持位移数据原单位。无质量矩阵，不采用质量加权。',
                '累计能量针对中心化快照；relative L2（相对L2误差）=‖Xhat−X‖F/‖X‖F，包含全部输入帧；RMSE（均方根误差）=sqrt(误差平方和/(T×N×3))。最大节点误差为三个分量误差的欧氏范数最大值。原始全零帧的逐帧相对误差留空，不硬除零。',
                '总位移是sqrt(DX²+DY²+DZ²)。全场峰值在全部节点和时间上搜索；表中峰值时间误差比较原始全局峰值时刻与重建全局峰值时刻，峰值节点可能不同；并列峰值取首次出现。关键节点表另列各节点峰值。',
                '存储比=原始float32快照字节数/[float64均值+截断模态+系数字节数]，不含文件头与元数据，也不代表求解加速比。磁盘保存全部有效模态供全量重建检查；该存储比表示仅保存指定阶数模型的理论存储比。',
                f'有效秩={effective}（特征值相对阈值1e-12）；选定重建阶数={evaluation["selected_rank"]}。零中心化能量时只保存均值，阶数0，能量比例无定义；逐帧绝对误差仍有效。',
                '## 实测多阶结果', '|阶数|中心化累计能量|RMSE（均方根误差）|相对L2误差|最大节点向量误差|峰值幅值绝对误差|峰值时间误差(s)|理论存储比|','|---:|---:|---:|---:|---:|---:|---:|---:|']
        for row in evaluation['rank_metrics']:
            def fmt(v): return 'N/A（不适用）' if v is None else f'{v:.10g}'
            report.append('|'+ '|'.join(fmt(row[k]) for k in ['rank','cumulative_energy','rmse','relative_L2','max_node_vector_error','peak_amplitude_abs_error','peak_time_abs_error_s','theoretical_storage_ratio'])+'|')
        report+=['## 关键节点',f"实际全场峰值节点={evaluation['global_peak_node_id']}，幅值={evaluation['global_peak_amplitude']:.10g}，时刻={evaluation['global_peak_time_s']:.9g}秒。",'```json',json.dumps(evaluation['key_nodes'],ensure_ascii=False,indent=2),'```',f"请求但不存在的节点：{evaluation['missing_requested_node_ids']}。合成样本不应包含或冒充真实机械臂节点。",
                 '## 实际资源与耗时','```json',json.dumps(dict(stages=stages,peak_rss_mib=summary['peak_rss_mib'],process_cpu_seconds=summary['process_cpu_seconds']),ensure_ascii=False,indent=2),'```','硬件、内存、磁盘、依赖与线程配置见environment.txt；锁定依赖见requirements.lock.txt。',
                 '## 图表和数组','01_pod_energy.png：累计能量；02_reconstruction_rmse.png：多阶逐帧误差；03_key_node_curves.png：关键节点三个分量及总位移；04_rank_errors.png：多阶全场误差。CSV表保留数值，NPY数组保存节点ID、快照、均值、模态和时间系数。',
                 '## 局限与下一阶段','客户资料描述22零件、线弹性、绑定连接、267801节点、169736单元、Newmark（纽马克时间积分）0.1秒。求解器类型不证明存在强非线性效应。未提供精确载荷、应力、速度、加速度，本文不推定这些数值。未执行跨工况或动态预测，也未依赖截图提取约束节点诊断。后续需要明确位移单位、Fx/Fy/Fz（X/Y/Z方向力）及单位和分配、多工况位移、独立留出工况；应力恢复另需单元连接、材料分配与数值应力标签。',
                 '## 复现命令','```bash',command,'```','原始输入保持不变。可加--reuse-snapshots复用已完成快照；复用核对路径/大小/修改时间、节点ID顺序、时间映射、快照形状和类型，不使用哈希。']
        paragraphs=[]
        for part in report:
            if part.startswith('|') and paragraphs and paragraphs[-1].startswith('|'):
                paragraphs[-1]+='\n'+part
            else: paragraphs.append(part)
        (out/'report.md').write_text('\n\n'.join(paragraphs)+'\n',encoding='utf-8')
        LOG.info('完成：%s；选择阶数%s，相对L2误差=%s，RMSE（均方根误差）=%.9g',out,chosen['rank'],chosen['relative_L2'],chosen['rmse'])
    except Exception as e:
        (out/'summary.json').unlink(missing_ok=True); (out/'report.md').unlink(missing_ok=True)
        write_json(out/'failure.json',dict(status='failed',error_type=type(e).__name__,reason=str(e)))
        LOG.exception('运行失败：%s',e); return 1
    return 0


if __name__=='__main__': sys.exit(main())
