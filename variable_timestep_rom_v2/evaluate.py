"""Compare field forecasts at each span using real measured histories."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from data import CHUNK, OUT, SEEN, decode, dump, load_case, windows
from metrics import series_metrics
from predict import ROM


@torch.inference_mode()
def latent_predictions(rom, coefficients, history, targets, load):
    inputs = rom.pack(coefficients[history], history, load, targets)
    prediction = rom.model(*inputs).cpu().numpy() * rom.scale
    results = []
    for i in range(len(history)):
        args = tuple(value if n==2 else value[i:i+1] for n,value in enumerate(inputs))
        if rom.device.type == "cuda":
            torch.cuda.synchronize(rom.device)
        start = perf_counter()
        rom.model(*args)
        if rom.device.type == "cuda":
            torch.cuda.synchronize(rom.device)
        results.append((perf_counter()-start)*1000)
    return prediction, np.asarray(results)


def score(step, rom, fixed, fields, times, force, coefficients):
    h,y = windows(step,len(times))
    current = h[:,-1]
    method = "variable_tcn"
    preds, ms = latent_predictions(rom,coefficients,h,y,force)
    forecasts = {method: preds}
    latency = {method: ms}
    if step==1:
        forecasts["fixed_1ms"],latency["fixed_1ms"] = latent_predictions(fixed,coefficients,h,y,force)
    forecasts["pod_projection"] = coefficients[y]

    names = ("persistence","velocity",*forecasts)
    n = len(y)
    rows = np.arange(n)
    truth_square = np.zeros(n)
    squared = {m:np.zeros(n) for m in names}
    absolute = {m:np.zeros(n) for m in names}
    worst = {m:np.zeros(n) for m in names}
    peaks = {m:np.full(n,-1.) for m in ("truth",*names)}
    nodes = {m:np.zeros(n,dtype=int) for m in peaks}
    temporal = {m:np.zeros(n) for m in forecasts}

    def peak(name,block,left):
        norm = np.einsum("ijk,ijk->ij",block,block)
        index = norm.argmax(axis=1)
        value = np.sqrt(norm[rows,index])
        update = value>peaks[name]
        peaks[name][update] = value[update]
        nodes[name][update] = index[update]+left

    for left in range(0,len(rom.ids),CHUNK):
        right = min(left+CHUNK,len(rom.ids))
        truth = fields[y,left:right].astype(np.float64)
        latest = fields[current,left:right].astype(np.float64)
        previous = fields[current-1,left:right].astype(np.float64)
        basis = rom.basis[:,left:right]
        reference = rom.reference[left:right]
        oracle = decode(forecasts["pod_projection"],basis,reference)
        peak("truth",truth,left)
        truth_square += np.einsum("ijk,ijk->i",truth,truth)
        for m in names:
            if m=="persistence":
                predicted=latest
            elif m=="velocity":
                predicted=latest+step*(latest-previous)
            elif m=="pod_projection":
                predicted=oracle
            else:
                predicted=decode(forecasts[m],basis,reference)
                delta=predicted-oracle
                temporal[m]+=np.einsum("ijk,ijk->i",delta,delta)
            peak(m,predicted,left)
            error=predicted-truth
            norms=np.einsum("ijk,ijk->ij",error,error)
            squared[m]+=norms.sum(axis=1)
            absolute[m]+=np.abs(error).sum(axis=(1,2))
            worst[m]=np.maximum(worst[m],np.sqrt(norms.max(axis=1)))

    key = int(np.flatnonzero(rom.ids==56877)[0])
    cur = fields[current,key].astype(np.float64)
    trace = {"truth":fields[y,key],"persistence":cur,
             "velocity":cur+step*(cur-fields[current-1,key]),
             **{m:q @ rom.basis[:,key]+rom.reference[key] for m,q in forecasts.items()}}
    traces = [dict(step_ms=step,method=m,node_id=int(rom.ids[key]),target_ms=int(target),
                   dx=float(vec[0]),dy=float(vec[1]),dz=float(vec[2]))
              for m, values in trace.items() for target,vec in zip(y,values)]
    t_idx = int(peaks["truth"].argmax())
    stats, per_frame = [], []
    for m in names:
        row, relative = series_metrics(squared[m],absolute[m],truth_square,worst[m],peaks[m],
                                       nodes[m],times[y],rom.ids)
        ms = latency.get(m)
        row.update(step_ms=step, span_group="seen" if step in SEEN else "unseen", method=m,
                   windows=n,true_peak_displacement=float(peaks["truth"][t_idx]),
                   true_peak_time_ms=int(y[t_idx]),
                   peak_amplitude_abs_error=abs(row["peak_displacement"]-peaks["truth"][t_idx]).item(),
                   peak_time_abs_error_ms=abs(row["peak_time_ms"]-int(y[t_idx])),
                   latent_forward_median_ms=float(np.median(ms)) if ms is not None else None,
                   latent_forward_p95_ms=float(np.percentile(ms,95)) if ms is not None else None,
                   dynamics_rmse=float(np.sqrt(temporal[m].sum()/(n*len(rom.ids)*3))) if m in latency else None,
                   dynamics_sse_fraction=float(temporal[m].sum()/squared[m].sum())
                   if m in latency and squared[m].sum()>0 else None)
        stats.append(row)
        per_frame.extend(dict(step_ms=step,method=m,anchor_ms=int(anchor),target_ms=int(target),
                              rmse=float(np.sqrt(sq/(len(rom.ids)*3))),
                              relative_l2=float(rel) if np.isfinite(rel) else None,
                              max_node_vector_error=float(vec),peak_displacement=float(peak_value),
                              peak_node_id=int(rom.ids[peak_id]))
                         for anchor,target,sq,rel,vec,peak_value,peak_id in
                         zip(current,y,squared[m],relative,worst[m],peaks[m],nodes[m]))
    return stats,per_frame,traces


def write_csv(path, rows):
    with path.open("w",newline="",encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--device",default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if not (args.out/"selection.json").is_file():
        raise ValueError("Train and select both models before final scoring")
    rom = ROM(args.out,args.device)
    fixed = ROM(args.out,args.device,"fixed")
    ids,times,fields,force,_,_ = load_case()
    if not np.array_equal(ids,rom.ids):
        raise ValueError("Evaluation data node IDs differ from trained model")
    coeff = np.load(args.out/"pod_coefficients.npy")
    # One warm-up call per model, outside the latency measurements.
    h,y = windows(1)
    with torch.inference_mode():
        for predictor in (rom,fixed):
            predictor.model(*predictor.pack(coeff[h[:1]],h[:1],force,y[:1]))
    metrics, frames, trace = [], [], []
    for step in range(1,9):
        result = score(step,rom,fixed,fields,times,force,coeff)
        for target, source in zip((metrics,frames,trace),result):
            target.extend(source)
        row = next(r for r in result[0] if r["method"]=="variable_tcn")
        print(f"{step} ms: {row['windows']} windows; TCN relL2={row['relative_l2']:.5%}",flush=True)
    for path,rows in (("summary.csv",metrics),("per_frame.csv",frames),("key_node.csv",trace)):
        write_csv(args.out/path,rows)
    dump(args.out/"summary.json",dict(timing="Batch 1 model forward, tensors resident, includes CUDA synchronization; excludes POD and I/O",
                                      hardware=torch.cuda.get_device_name() if rom.device.type=="cuda" else "CPU",
                                      results=metrics))
    from report import generate
    generate(args.out)


if __name__ == "__main__":
    main()
