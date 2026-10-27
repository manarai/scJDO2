#!/usr/bin/env python3
"""Frozen orient/S18b, including planner Amendment Revision 1.

One detached, timed unit at a time. Original S18 inputs remain read-only.
Embedding, score models, tangent directions and states use controls only.
Target-specific H1 mixture weights are the explicitly prescribed diagnostic.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import heapq
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "perturb_k562_occupancy"
LOGS = OUT / "run_logs"
CACHE = OUT / "cache"
FIG = ROOT / "benchmarks" / "figures" / "perturb_k562_occupancy"
S18 = ROOT / "benchmarks" / "perturb_k562"
RAW = S18 / "data" / "K562_essential_raw_singlecell_01.h5ad"
SHA = "3e5a63a9e892b21029bb55fca4e12517a49aad7af6c14133ca63d12cf68c6cee"
BUDGET = 12.0
KS = (20, 15, 30)
PLANNER_COMMIT = "56e9c0045387bf6049dbd40c8b049ef2d55a4cae"


def now():
    return dt.datetime.now(dt.UTC).isoformat()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def cpu_hours():
    return sum(float(json.loads(p.read_text()).get("cpu_hours", 0))
               for p in LOGS.glob("*.checkpoint.json"))


def tree_cpu(pid):
    """Live process-tree CPU, including exited descendants already waited for."""
    seen, pending, ticks = set(), [pid], 0
    while pending:
        p = pending.pop()
        if p in seen:
            continue
        seen.add(p)
        try:
            stat = Path(f"/proc/{p}/stat").read_text().rsplit(")", 1)[1].split()
            ticks += sum(int(stat[i]) for i in (11, 12, 13, 14))
            for task in Path(f"/proc/{p}/task").iterdir():
                try:
                    pending.extend(map(int, (task / "children").read_text().split()))
                except (FileNotFoundError, ProcessLookupError):
                    pass
        except (FileNotFoundError, ProcessLookupError):
            pass
    return ticks / os.sysconf("SC_CLK_TCK")


def run_unit(name, command):
    LOGS.mkdir(parents=True, exist_ok=True)
    checkpoint = LOGS / f"{name}.checkpoint.json"
    if checkpoint.exists():
        raise RuntimeError("Checkpoint exists: inspect, never overwrite or automatically retry")
    used = cpu_hours()
    threads = min(6, os.cpu_count() or 1)
    monitoring_reserve = threads * 30 / 3600
    if used + monitoring_reserve >= BUDGET:
        raise RuntimeError("S18b CPU budget exhausted")
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT", "NUMBA_NUM_THREADS"):
        env[key] = str(threads)
    env["PATH"] = str(ROOT / ".venv/bin") + os.pathsep + env.get("PATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    started, t0 = now(), time.monotonic()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    log_path = LOGS / f"{name}.log"
    print(f"[{started}] {name}: {command}", flush=True)
    with log_path.open("w") as log:
        p = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True)
        while True:
            try:
                code = p.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                live = tree_cpu(p.pid) / 3600
                print(f"[{now()}] {name}: wall {time.monotonic()-t0:.0f}s; "
                      f"unit CPU {live:.4f}h; total {used+live:.4f}/{BUDGET}h", flush=True)
                if used + live + monitoring_reserve >= BUDGET:
                    os.killpg(p.pid, signal.SIGTERM)
                    try:
                        p.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(p.pid, signal.SIGKILL)
                        p.wait()
                    code = 124
                    break
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
    receipt = {"unit": name, "command": command, "started_utc": started,
               "finished_utc": now(), "exit_code": code, "wall_seconds": time.monotonic()-t0,
               "cpu_hours": cpu/3600, "cpu_hours_before": used,
               "max_rss_kib": after.ru_maxrss, "threads": threads,
               "cpu_count": os.cpu_count(),
               "ram_bytes": os.sysconf("SC_PAGE_SIZE")*os.sysconf("SC_PHYS_PAGES"),
               "log": str(log_path.relative_to(ROOT))}
    dump(checkpoint, receipt)
    print("\n".join(log_path.read_text(errors="replace").splitlines()[-24:]), flush=True)
    print(json.dumps(receipt, indent=2), flush=True)
    return code


def verify_input():
    actual = digest(RAW)
    r = {"raw_path": str(RAW.relative_to(ROOT)), "raw_sha256": actual,
         "required_sha256": SHA, "matches": actual == SHA,
         "raw_bytes": RAW.stat().st_size, "raw_mtime_ns": RAW.stat().st_mtime_ns,
         "verified_utc": now(), "raw_open_mode": "rb"}
    dump(OUT / "input_verification.json", r)
    if actual != SHA:
        raise RuntimeError(f"STOP: original K562 SHA differs: {actual}")
    accepted = json.loads((ROOT / "coord/orient/reports/S18.json").read_text())
    expected = {a["path"]: a for a in accepted["artifacts"]}
    paths = [S18 / "targets.csv", S18 / "embedding.json", *sorted((S18 / "cache").glob("*.npz"))]
    reused = []
    for path in paths:
        rel, h = str(path.relative_to(ROOT)), digest(path)
        if rel in expected and h != expected[rel]["sha256"]:
            raise RuntimeError(f"STOP: accepted S18 input changed: {rel}")
        reused.append({"path": rel, "bytes": path.stat().st_size, "sha256": h})
    with (S18 / "targets.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    n = sum(row["significant"] == "True" for row in rows)
    e = json.loads((S18 / "embedding.json").read_text())
    if n != 865 or e["n_controls"] != 10691:
        raise RuntimeError("STOP: required target/control cohort differs")
    r.update(n_significant_targets=n, n_eligible_targets=len(rows),
             n_controls_from_verified_S18_manifest=e["n_controls"], S18_outputs_and_caches=reused,
             fitted_network_state_present=False,
             missing_fit_note="Network state was not saved in S18; exact refit permitted by planner.")
    dump(OUT / "input_verification.json", r)
    print(f"Original SHA verified: {actual}; controls 10691; significant targets {n}", flush=True)
    return 0


def ensure_inputs():
    r = json.loads((OUT / "input_verification.json").read_text())
    source = json.loads((S18 / "data_source.json").read_text())
    if not r["matches"] or r["raw_sha256"] != SHA:
        raise RuntimeError("STOP: initial original-file SHA barrier not passed")
    stat = RAW.stat()
    verified_mtime = r.get("raw_mtime_ns", source["verified_stat_mtime_ns"])
    if stat.st_size != r["raw_bytes"] or stat.st_mtime_ns != verified_mtime:
        raise RuntimeError("STOP: original input changed")
    for a in r["S18_outputs_and_caches"]:
        if digest(ROOT / a["path"]) != a["sha256"]:
            raise RuntimeError(f"STOP: S18 source/cache changed: {a['path']}")


def load_z(k):
    import numpy as np
    import pandas as pd
    ensure_inputs()
    CACHE.mkdir(parents=True, exist_ok=True)
    with np.load(S18 / "cache/embedding.npz", allow_pickle=False) as a:
        z, labels, ci = a["z"][:, :k], a["labels"], a["control_indices"]
    targets = pd.read_csv(S18 / "targets.csv")
    targets = targets[targets.significant].sort_values("target").reset_index(drop=True)
    if len(targets) != 865 or len(ci) != 10691:
        raise RuntimeError("STOP: cohort mismatch")
    return z, labels, ci, targets


def frozen_config():
    from scjdo2.energy_drift import EnergyDriftConfig
    cfg = EnergyDriftConfig(hidden=96, depth=3, n_epochs=1500, seed=0,
                            batch_size=256, loss="esm", lr=5e-4, beta=1.0,
                            esm_laplacian="exact")
    saved = json.loads((S18 / "fit_k20.json").read_text())["S02_config"]
    if json.loads(json.dumps(vars(cfg))) != saved:
        raise RuntimeError("STOP: exact S18 fit configuration changed")
    return cfg


def refit(k):
    import numpy as np
    import torch
    from scjdo2.energy_drift import EnergyDriftAdapter
    path = CACHE / f"model_k{k}.pt"
    if path.exists():
        raise RuntimeError("Model state already saved: reuse it, do not refit")
    z, _, ci, _ = load_z(k)
    controls = np.asarray(z[ci], dtype=np.float32)
    torch.set_num_threads(min(6, os.cpu_count() or 1))
    cfg = frozen_config()
    print(f"[{now()}] S18 exact EnergyDriftAdapter({k},cfg).fit on 10691 controls", flush=True)
    adapter = EnergyDriftAdapter(k, cfg).fit(controls, verbose=True)
    H = adapter.jacobian_autodiff(controls)
    with np.load(S18 / f"cache/H_k{k}.npz", allow_pickle=False) as a:
        old = a["H"]
        np.testing.assert_array_equal(a["control_indices"], ci)
    denominator = np.linalg.norm(old, axis=(1, 2))
    if (denominator == 0).any() or not np.isfinite(H).all():
        raise RuntimeError("STOP: undefined/nonfinite refit comparison")
    relative = np.linalg.norm(H-old, axis=(1, 2)) / denominator
    torch.save({"state_dict": adapter._model.state_dict(),
                "config": vars(cfg), "dim": k, "device": adapter._device}, path)
    np.savez_compressed(CACHE / f"refit_k{k}.npz", H=H, control_indices=ci, relative_difference=relative)
    info = {"k": k, "n_controls": len(ci), "configuration": vars(cfg),
            "device": adapter._device, "median_relative_Frobenius_difference": float(np.median(relative)),
            "max_relative_Frobenius_difference": float(relative.max()),
            "required_primary_max_median": 0.05, "primary_refit_check_pass": bool(np.median(relative)<=0.05) if k==20 else None,
            "fit_path": "EnergyDriftAdapter(k, exact S18 cfg).fit(np.asarray(z[ci],dtype=float32))"}
    dump(OUT / f"refit_k{k}.json", info)
    print(json.dumps(info, indent=2), flush=True)
    if k == 20 and np.median(relative) > 0.05:
        raise RuntimeError("STOP: refit-check median difference exceeds 0.05; do not use the model")
    return 0


def load_model(k):
    import torch
    from scjdo2.energy_drift import EnergyDriftAdapter, EnergyDriftConfig, _EnergyMLP
    ensure_inputs()
    primary = json.loads((OUT / "refit_k20.json").read_text())
    if not primary["primary_refit_check_pass"]:
        raise RuntimeError("STOP: primary refit barrier failed")
    saved = torch.load(CACHE / f"model_k{k}.pt", map_location="cpu", weights_only=True)
    cfg = EnergyDriftConfig(**saved["config"])
    adapter = EnergyDriftAdapter(k, cfg)
    adapter._device = saved["device"]
    adapter._model = _EnergyMLP(k, cfg.hidden, cfg.depth).to(adapter._device)
    adapter._model.load_state_dict(saved["state_dict"])
    adapter._model.eval()
    adapter._fitted = True
    torch.set_num_threads(min(6, os.cpu_count() or 1))
    return adapter


def merge_small(endpoint, initial):
    """Literal iterative nearest-endpoint-centroid merging; no threshold change."""
    import numpy as np
    _, initial = np.unique(initial, return_inverse=True)
    n = np.bincount(initial).astype(np.int64)
    sums = np.zeros((len(n), endpoint.shape[1]), dtype=np.float64)
    np.add.at(sums, initial, endpoint)
    centers = sums/n[:, None]
    parent, generation = np.arange(len(n)), np.zeros(len(n), dtype=np.int64)
    active = np.ones(len(n), dtype=bool)
    heap = [(int(size), b, 0) for b, size in enumerate(n) if size < 100]
    heapq.heapify(heap)
    merges = 0
    while heap:
        _, a, gen = heapq.heappop(heap)
        if not active[a] or generation[a] != gen or n[a] >= 100:
            continue
        candidates = np.flatnonzero(active)
        candidates = candidates[candidates != a]
        if not len(candidates):
            break
        distance2 = np.sum((centers[candidates]-centers[a])**2, axis=1)
        b = int(candidates[np.argmin(distance2)])
        centers[b] = (centers[b]*n[b]+centers[a]*n[a])/(n[a]+n[b])
        n[b] += n[a]
        active[a], parent[a] = False, b
        generation[b] += 1
        merges += 1
        if n[b] < 100:
            heapq.heappush(heap, (int(n[b]), b, int(generation[b])))
    def representative(a):
        chain=[]
        while parent[a] != a:
            chain.append(a)
            a=int(parent[a])
        for j in chain:
            parent[j]=a
        return a
    roots=np.array([representative(int(a)) for a in initial])
    _, labels=np.unique(roots, return_inverse=True)
    return labels, merges


def matched_leiden(controls, B):
    import anndata as ad
    import numpy as np
    import scanpy as sc
    a=ad.AnnData(X=controls)
    # Unspecified library arguments retain Scanpy defaults; seed is fixed at 0.
    sc.pp.neighbors(a, use_rep="X", random_state=0)
    attempts=[]
    def trial(resolution):
        sc.tl.leiden(a, resolution=float(resolution), random_state=0, key_added="state")
        labels=a.obs["state"].astype(str).to_numpy(dtype=str)
        _, labels=np.unique(labels, return_inverse=True)
        count=int(labels.max()+1)
        attempts.append({"resolution":float(resolution),"n_states":count})
        print(f"[{now()}] Leiden resolution={resolution:.9g}: {count} states; requested {B}",flush=True)
        return labels,count
    res=1.0
    labels,count=trial(res)
    if count == B:
        return labels,res,attempts
    if count < B:
        low,high=res,res*2
        for _ in range(60):
            labels,count=trial(high)
            if count==B:
                return labels,high,attempts
            if count>B:
                break
            low,high=high,high*2
        else:
            raise RuntimeError("STOP: cannot bracket the exact requested Leiden state count")
    else:
        high,low=res,res/2
        for _ in range(60):
            labels,count=trial(low)
            if count==B:
                return labels,low,attempts
            if count<B:
                break
            high,low=low,low/2
        else:
            raise RuntimeError("STOP: cannot bracket the exact requested Leiden state count")
    for _ in range(60):
        mid=(low+high)/2
        labels,count=trial(mid)
        if count==B:
            return labels,mid,attempts
        if count<B:
            low=mid
        else:
            high=mid
    raise RuntimeError("STOP: Leiden skipped the exact number of states; no nearest-count substitution")


def build_states(k):
    import numpy as np
    import pandas as pd
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.neighbors import NearestNeighbors
    if (CACHE / f"states_k{k}.npz").exists():
        raise RuntimeError("Completed state cache exists: no repeat")
    z,_,ci,_=load_z(k)
    controls=np.asarray(z[ci],dtype=np.float32)
    model=load_model(k)
    endpoint=controls.copy()
    stopped=np.zeros(len(ci),dtype=bool)
    steps=np.zeros(len(ci),dtype=np.int32)
    for step in range(400):
        idx=np.flatnonzero(~stopped)
        if not len(idx):
            break
        move=0.05*model.score(endpoint[idx])
        if not np.isfinite(move).all():
            raise RuntimeError("STOP: nonfinite fixed-step drift; no step-size substitution")
        endpoint[idx]+=move
        steps[idx]+=1
        stopped[idx[np.linalg.norm(move,axis=1)<1e-4]]=True
        if (step+1)%50==0:
            print(f"[{now()}] k={k} Euler step {step+1}/400; stopped {stopped.sum()}/10691",flush=True)
    distances=NearestNeighbors(n_neighbors=2).fit(endpoint).kneighbors(endpoint)[0][:,1]
    median_nn=float(np.median(distances))
    threshold=0.5*median_nn
    initial=AgglomerativeClustering(n_clusters=None,distance_threshold=threshold).fit_predict(endpoint)
    basin,merges=merge_small(endpoint,initial)
    B=int(basin.max()+1)
    if (np.bincount(basin)<100).any():
        raise RuntimeError("STOP: endpoint merging failed the minimum-100 rule")
    with np.load(CACHE / f"refit_k{k}.npz",allow_pickle=False) as a:
        H=a["H"]
    neighbors=NearestNeighbors(n_neighbors=30).fit(controls).kneighbors(controls,return_distance=False)
    stiffness=np.empty(len(ci))
    for lo in range(0,len(ci),512):
        hi=min(lo+512,len(ci))
        A=np.asarray(controls[neighbors[lo:hi]],dtype=np.float64)
        A-=A.mean(axis=1,keepdims=True)
        covariance=np.swapaxes(A,1,2)@A/29
        _,V=np.linalg.eigh(covariance)
        V=V[:,:,-3:]
        tangent=np.einsum("nia,nij,njb->nab",V,H[lo:hi],V)
        tangent=(tangent+np.swapaxes(tangent,1,2))/2
        stiffness[lo:hi]=-np.linalg.eigvalsh(tangent)[:,-1]
    energy=model.energy(controls)
    endpoint_energy=model.energy(endpoint)
    leiden,res,attempts=matched_leiden(controls,B)
    records=[]
    for scheme,state in (("score_J",basin),("leiden",leiden)):
        for b in range(B):
            mask=state==b
            C=np.asarray(controls[mask],dtype=np.float64)
            spread=float(np.trace(np.cov(C,rowvar=False)))
            if not np.isfinite(spread) or spread<=0:
                raise RuntimeError("STOP: state covariance spread/depth undefined")
            record={"k":k,"scheme":scheme,"basin":b,"n_controls":int(mask.sum()),
                    "spread_trace_covariance":spread,"stiffness":None,"energy_depth":None,
                    "covariance_depth":1/spread if scheme=="leiden" else None}
            if scheme=="score_J":
                record.update(stiffness=float(np.median(stiffness[mask])),
                              energy_depth=float(np.median(energy[mask])-np.min(endpoint_energy[mask])),
                              endpoint_minimum_energy=float(np.min(endpoint_energy[mask])),
                              fraction_drift_stopped=float(stopped[mask].mean()))
            records.append(record)
    pd.DataFrame(records).to_csv(OUT/f"basins_k{k}.csv",index=False,float_format="%.12g")
    np.savez_compressed(CACHE/f"states_k{k}.npz",control_indices=ci,score_basin=basin,
                        leiden_state=leiden,endpoints=endpoint,stiffness=stiffness,
                        energy=energy,endpoint_energy=endpoint_energy,steps=steps,stopped=stopped)
    dump(OUT/f"states_k{k}.json",{"k":k,"n_controls":len(ci),"n_basins":B,
         "n_initial_endpoint_clusters":int(np.unique(initial).size),"n_small_cluster_merges":merges,
         "median_endpoint_nearest_neighbor_distance":median_nn,"agglomerative_distance_threshold":threshold,
         "agglomerative_linkage":"sklearn default ward","merge_algorithm":"Smallest active cluster first, then source label; nearest endpoint centroid; stable lowest-label distance ties; repeat until all groups >=100",
         "euler_step":0.05,"max_steps":400,"step_stop_norm":1e-4,"fraction_stopped":float(stopped.mean()),
         "tangent":"30 neighbors, no explicit self removal; centered; denominator29; largest3 covariance eigendirections",
         "energy_endpoint_minimum":"Lowest energy among the final fixed-integration endpoints in the basin",
         "leiden_resolution":res,"leiden_attempts":attempts,"leiden_neighbors":"Scanpy defaults (n_neighbors15), use_rep X, random_state0; Leiden defaults with seed0",
         "control_only_state_fit":True})
    print(f"k={k}: initial {np.unique(initial).size}, merged basins {B}, Leiden {np.unique(leiden).size}",flush=True)
    return 0


def mark_large(path):
    path=Path(path)
    if path.stat().st_size>10*1024*1024:
        rel=str(path.relative_to(ROOT))
        exclude=ROOT/".git/info/exclude"
        content=exclude.read_text()
        if rel not in content.splitlines():
            with exclude.open("a") as f:
                f.write(rel+"\n")


def occupancy(k):
    import numpy as np
    import pandas as pd
    from sklearn.neighbors import NearestNeighbors
    z,labels,ci,targets=load_z(k)
    with np.load(CACHE/f"states_k{k}.npz",allow_pickle=False) as a:
        states={"score_J":a["score_basin"],"leiden":a["leiden_state"]}
    controls=np.asarray(z[ci],dtype=np.float32)
    target_names=targets.target.to_numpy(dtype=str)
    selected=np.flatnonzero((labels=="non-targeting")|np.isin(labels,target_names))
    control_mean=controls.mean(axis=0,dtype=np.float64)
    nearest=NearestNeighbors(n_neighbors=15,algorithm="brute",n_jobs=1).fit(controls)
    B=int(states["score_J"].max()+1)
    predictions={scheme:np.empty(len(selected),dtype=np.int32) for scheme in states}
    for lo in range(0,len(selected),2048):
        hi=min(lo+2048,len(selected))
        idx=nearest.kneighbors(z[selected[lo:hi]],return_distance=False)
        for scheme,state in states.items():
            votes=np.zeros((hi-lo,B),dtype=np.int16)
            np.add.at(votes,(np.arange(hi-lo)[:,None],state[idx]),1)
            predictions[scheme][lo:hi]=votes.argmax(axis=1)
        if lo%40960==0:
            print(f"[{now()}] k={k} vote assigned {hi}/{len(selected)} cells",flush=True)
    rows,summary=[],[]
    arrays={"obs_indices":selected,"targets":target_names}
    selected_labels=labels[selected]
    for scheme,state in states.items():
        pred=predictions[scheme]
        nc=np.bincount(pred[selected_labels=="non-targeting"],minlength=B)
        original_nc=np.bincount(state,minlength=B)
        counts=np.stack([np.bincount(pred[selected_labels==t],minlength=B) for t in target_names])
        nt=counts.sum(axis=1)
        if not np.array_equal(nt,targets.n_cells.to_numpy()):
            raise RuntimeError("STOP: occupancy target counts differ")
        fc=np.log2((counts+1)/(nt[:,None]+B))-np.log2((nc+1)/(len(ci)+B))[None,:]
        arrays.update({f"{scheme}_pred":pred,f"{scheme}_counts":counts,f"{scheme}_controls_counts":nc,
                       f"{scheme}_log2_fc":fc,f"{scheme}_sensitivity":np.median(np.abs(fc),axis=0)})
        for t,target in enumerate(target_names):
            shift=z[labels==target].mean(axis=0,dtype=np.float64)-control_mean
            summary.append({"k":k,"scheme":scheme,"target":target,"n_cells":int(nt[t]),
                            "centroid_shift_norm_all_controls":float(np.linalg.norm(shift))})
            for b in range(B):
                rows.append({"k":k,"scheme":scheme,"target":target,"basin":b,"n_target_basin":int(counts[t,b]),
                             "n_target":int(nt[t]),"n_controls_assigned":int(nc[b]),
                             "n_controls_original_basin":int(original_nc[b]),"log2_fc":float(fc[t,b])})
    path=OUT/f"occupancy_k{k}.csv"
    pd.DataFrame(rows).to_csv(path,index=False,float_format="%.9g")
    mark_large(path)
    pd.DataFrame(summary).to_csv(OUT/f"centroid_shifts_k{k}.csv",index=False,float_format="%.12g")
    np.savez_compressed(CACHE/f"occupancy_k{k}.npz",**arrays)
    dump(OUT/f"occupancy_k{k}.json",{"k":k,"n_basins":B,"n_targets":865,"n_controls":10691,
          "vote_k":15,"vote_ties":"lowest stable basin ID","controls_self":"Included normally in same-data query",
          "state_depth_and_H1_membership":"Original endpoint/Leiden partition of controls",
          "occupancy_control_membership":"Controls classified by the same 15-neighbor vote as targets; original and classified counts both retained",
          "centroid_reference":"All 10691 controls, never nearest controls","n_cells_classified":len(selected)})
    return 0


def correlation_test(depth,sensitivity,size=None,spread=None):
    import numpy as np
    from scipy.stats import rankdata
    x,y=rankdata(depth),rankdata(sensitivity)
    x=x.astype(np.float64); y=y.astype(np.float64)
    partial=size is not None
    if partial:
        Z=np.column_stack([np.ones(len(x)),rankdata(size),rankdata(spread)])
        pinv=np.linalg.pinv(Z)
        def residual(a):
            return a-(a@pinv.T)@Z.T
        x0=residual(x[None,:])[0]; y0=residual(y[None,:])[0]
        if np.linalg.matrix_rank(np.column_stack([Z,x]))==np.linalg.matrix_rank(Z) or np.linalg.matrix_rank(np.column_stack([Z,y]))==np.linalg.matrix_rank(Z):
            raise RuntimeError("STOP: primary/energy partial Spearman has no independent residual variation")
    else:
        def residual(a):
            return a-a.mean(axis=1,keepdims=True)
        x0=x-x.mean(); y0=y-y.mean()
    if np.linalg.norm(x0)==0 or np.linalg.norm(y0)==0:
        raise RuntimeError("STOP: Spearman correlation undefined")
    observed=float(x0@y0/(np.linalg.norm(x0)*np.linalg.norm(y0)))
    rng=np.random.default_rng(0)
    perms=np.stack([rng.permutation(len(x)) for _ in range(10000)])
    xp=residual(x[perms])
    denominator=np.linalg.norm(xp,axis=1)*np.linalg.norm(y0)
    if (denominator==0).any():
        raise RuntimeError("STOP: an exact prescribed depth permutation has undefined correlation")
    rp=(xp@y0)/denominator
    left=int(np.count_nonzero(rp<=observed))
    two=int(np.count_nonzero(np.abs(rp)>=abs(observed)))
    return {"rho":observed,"p_one_sided_negative":(1+left)/10001,
            "p_two_sided_absolute":(1+two)/10001,"n_basins":len(x),"permutations":10000,
            "negative_tail_count":left,"two_sided_count":two,"rng":"default_rng(0)",
            "partial":partial,"nuisance":"intercept and ranks of original basin size and covariance trace" if partial else None,
            "permutation_rule":"Permute depth across states, re-rank (equivalently permute depth ranks), re-residualize against fixed nuisances"}


def h2(k):
    import numpy as np
    import pandas as pd
    ensure_inputs()
    basin=pd.read_csv(OUT/f"basins_k{k}.csv")
    with np.load(CACHE/f"occupancy_k{k}.npz",allow_pickle=False) as a:
        score_sensitivity=a["score_J_sensitivity"]
        leiden_sensitivity=a["leiden_sensitivity"]
    score=basin[basin.scheme=="score_J"].sort_values("basin")
    leiden=basin[basin.scheme=="leiden"].sort_values("basin")
    size,spread=score.n_controls.to_numpy(),score.spread_trace_covariance.to_numpy()
    result={"k":k,"n_basins":len(score),"n_targets":865,
            "stiffness_ordinary":correlation_test(score.stiffness.to_numpy(),score_sensitivity),
            "stiffness_partial":correlation_test(score.stiffness.to_numpy(),score_sensitivity,size,spread),
            "energy_depth_ordinary":correlation_test(score.energy_depth.to_numpy(),score_sensitivity),
            "energy_depth_partial":correlation_test(score.energy_depth.to_numpy(),score_sensitivity,size,spread),
            "leiden_ordinary":correlation_test(leiden.covariance_depth.to_numpy(),leiden_sensitivity),
            "leiden_partial":None,
            "leiden_partial_reason":"Planner Revision1: depth1/trace is rank-collinear with nuisance spread trace; undefined by construction"}
    r=result["stiffness_partial"]
    result["gate_H2"]={"rho":r["rho"],"p_one_sided_negative":r["p_one_sided_negative"],
                       "required_rho_less_than":0,"required_p_less_than":0.05,
                       "pass":bool(r["rho"]<0 and r["p_one_sided_negative"]<0.05)}
    basin["sensitivity_median_abs_log2_fc"]=np.concatenate([score_sensitivity,leiden_sensitivity])
    basin.to_csv(OUT/f"basins_k{k}.csv",index=False,float_format="%.17g")
    dump(OUT/f"h2_k{k}.json",result)
    print(json.dumps(result,indent=2),flush=True)
    return 0


def simplex_projection(v):
    import numpy as np
    u=np.sort(v,axis=1)[:,::-1]
    cumulative=np.cumsum(u,axis=1)-1
    j=np.arange(1,v.shape[1]+1,dtype=np.float64)
    rho=(u-cumulative/j>0).sum(axis=1)-1
    theta=cumulative[np.arange(len(v)),rho]/(rho+1)
    return np.maximum(v-theta[:,None],0)


def solve_weights(A,C,q_target,start):
    """Batched monotone-restart FISTA; exact convex simplex objective, KKT gap."""
    import numpy as np
    C=(C+C.T)/2
    centered=C-C.mean(0)[None,:]-C.mean(1)[:,None]+C.mean()
    eigen=np.linalg.eigvalsh(-2*centered)
    fp=256*np.finfo(float).eps*max(1,np.linalg.norm(-2*centered,ord=np.inf))
    if eigen[0]<-fp:
        raise RuntimeError("STOP: all-control distance kernel is not conditionally negative definite")
    L=np.nextafter(max(eigen[-1],fp),np.inf)
    R=len(A)
    output=np.empty_like(A); gaps=np.empty(R); iterations=np.empty(R,dtype=np.int32)
    # Numerical optimization accuracy only: no regularizer/statistical threshold.
    tolerance=1e-10*(1+np.abs(q_target)+np.max(np.abs(A),axis=1)+np.max(np.abs(C)))
    for lo in range(0,R,2048):
        hi=min(lo+2048,R)
        active=np.arange(lo,hi)
        W=np.broadcast_to(start,(hi-lo,len(start))).copy()
        Y=W.copy(); tau=np.ones(len(active)); WC=W@C
        for it in range(100001):
            a=A[active]
            objective=2*(a*W).sum(1)-(W*WC).sum(1)
            gradient=2*(a-WC)
            gap=np.maximum((W*gradient).sum(1)-gradient.min(1),0)
            done=gap<=tolerance[active]
            if done.any():
                output[active[done]]=W[done]
                gaps[active[done]]=gap[done]
                iterations[active[done]]=it
            keep=~done
            if not keep.any():
                break
            if it==100000:
                raise RuntimeError("STOP: exact H1 optimizer did not certify convergence; no uncertified values accepted")
            W,WC,Y,tau,objective=W[keep],WC[keep],Y[keep],tau[keep],objective[keep]
            active=active[keep]; a=A[active]
            V=simplex_projection(Y-2*(a-Y@C)/L)
            VC=V@C
            objV=2*(a*V).sum(1)-(V*VC).sum(1)
            restart=objV>objective
            new_tau=(1+np.sqrt(1+4*tau*tau))/2
            newY=V+((tau-1)/new_tau)[:,None]*(V-W)
            if restart.any():
                vp=simplex_projection(W[restart]-2*(a[restart]-WC[restart])/L)
                V[restart],VC[restart]=vp,vp@C
                new_tau[restart],newY[restart]=1,vp
            W,WC,Y,tau=V,VC,newY,new_tau
            if it and it%5000==0:
                print(f"[{now()}] H1 optimizer iteration {it}; unresolved {len(active)}; max gap {gap.max():.3g}",flush=True)
    np.testing.assert_allclose(output.sum(1),1,rtol=0,atol=1e-12)
    if (output<0).any():
        raise RuntimeError("STOP: negative mixture weights")
    e2=2*np.sum(A*output,axis=1)-np.sum((output@C)*output,axis=1)-q_target
    baseline=2*(A@start)-float(start@C@start)-q_target
    roundoff=512*np.finfo(float).eps*(1+np.abs(q_target)+np.max(np.abs(A),axis=1)+np.max(np.abs(C)))
    if (e2 < -roundoff).any() or (baseline<=0).any():
        raise RuntimeError("STOP: materially negative/undefined energy-distance squared")
    e2=np.maximum(e2,0)
    ef=1-np.sqrt(e2/baseline)
    ef_upper=1-np.sqrt(np.maximum(e2-gaps,0)/baseline)
    if (ef < -1e-8).any():
        raise RuntimeError("STOP: optimized energy worse than feasible all-control reference")
    return {"weights":output,"energy_control":np.sqrt(baseline),"energy_reweighted":np.sqrt(e2),
            "explained_fraction":ef,"explained_fraction_upper":ef_upper,"kkt_gap":gaps,
            "iterations":iterations,"lipschitz_L":float(L)}


def h1(k):
    import numpy as np
    import pandas as pd
    from scipy.spatial.distance import cdist
    if (OUT/f"h1_k{k}.csv").exists():
        raise RuntimeError("Completed H1 results exist: do not rerun/re-seed")
    z,labels,ci,targets=load_z(k)
    Ctr=np.asarray(z[ci],dtype=np.float64)
    with np.load(CACHE/f"states_k{k}.npz",allow_pickle=False) as a:
        basin=a["score_basin"]
    B=int(basin.max()+1)
    counts=np.bincount(basin,minlength=B).astype(np.float64)
    P=np.zeros((len(ci),B),dtype=np.float64)
    P[np.arange(len(ci)),basin]=1/counts[basin]
    # One exact all-control distance matrix per unit, never subsample controls.
    D=cdist(Ctr,Ctr,metric="euclidean")
    U=D@P
    C=P.T@U
    C=(C+C.T)/2
    prior=counts/len(ci)
    A=np.empty((len(targets),B)); q=np.empty(len(targets))
    for t,target in enumerate(targets.target):
        X=np.asarray(z[labels==target],dtype=np.float64)
        cross=cdist(X,Ctr).mean(axis=0)
        A[t]=cross@P
        q[t]=cdist(X,X).mean()
        if (t+1)%100==0:
            print(f"[{now()}] k={k} exact all-control target statistics {t+1}/865",flush=True)
    actual=solve_weights(A,C,q,prior)
    rng=np.random.default_rng(0)
    noise={}
    unique_sizes=sorted(set(map(int,targets.n_cells)))
    noise_root=CACHE/f"null_k{k}"
    noise_root.mkdir(parents=True,exist_ok=True)
    max_null_gap=0.0; max_uncertainty=0.0; max_iterations=0
    null_A=np.empty((200*len(unique_sizes),B))
    null_q=np.empty(200*len(unique_sizes))
    sample_banks=[]
    for number,n in enumerate(unique_sizes,1):
        file=noise_root/f"n{n}.npz"
        if file.exists():
            raise RuntimeError("Null bank already exists from an incomplete unit: inspect; no automatic re-seeding/retry")
        samples=np.stack([rng.choice(len(ci),size=n,replace=False) for _ in range(200)])
        lo,hi=(number-1)*200,number*200
        null_A[lo:hi]=np.stack([U[idx].mean(0) for idx in samples])
        null_q[lo:hi]=np.array([D[np.ix_(idx,idx)].mean() for idx in samples])
        sample_banks.append(samples)
        if number%20==0 or number==len(unique_sizes):
            print(f"[{now()}] k={k} exact null sufficient statistics {number}/{len(unique_sizes)} sizes",flush=True)
    null_result=solve_weights(null_A,C,null_q,prior)
    for number,(n,samples) in enumerate(zip(unique_sizes,sample_banks),1):
        lo,hi=(number-1)*200,number*200
        result={key:value[lo:hi] for key,value in null_result.items() if key!="lipschitz_L"}
        noise[n]=result["explained_fraction"]
        max_null_gap=max(max_null_gap,float(result["kkt_gap"].max()))
        max_uncertainty=max(max_uncertainty,float((result["explained_fraction_upper"]-result["explained_fraction"]).max()))
        max_iterations=max(max_iterations,int(result["iterations"].max()))
        file=noise_root/f"n{n}.npz"
        np.savez_compressed(file,indices=samples,explained_fraction=result["explained_fraction"],
                            explained_fraction_upper=result["explained_fraction_upper"],kkt_gap=result["kkt_gap"],
                            iterations=result["iterations"],weights=result["weights"])
        if number%10==0 or number==len(unique_sizes):
            print(f"[{now()}] k={k} null banks {number}/{len(unique_sizes)}; 200 subsets per size; max iterations {max_iterations}",flush=True)
    p95=np.array([np.percentile(noise[int(n)],95) for n in targets.n_cells])
    frame=targets[["target","n_cells","effect_size_energy_distance","p_adjusted_BH"]].copy()
    frame.insert(0,"k",k)
    for field in ("energy_control","energy_reweighted","explained_fraction","explained_fraction_upper","kkt_gap","iterations"):
        frame[field]=actual[field]
    frame["noise_mean"]=[float(noise[int(n)].mean()) for n in targets.n_cells]
    frame["noise_p95"]=p95
    frame["exceeds_noise_p95"]=actual["explained_fraction"]>p95
    frame.to_csv(OUT/f"h1_k{k}.csv",index=False,float_format="%.12g")
    np.savez_compressed(CACHE/f"h1_k{k}.npz",targets=targets.target.to_numpy(dtype=str),weights=actual["weights"],
                        basin_distance=C,control_prior=prior,cross_mean=A,target_within_mean=q)
    median=float(frame.explained_fraction.median())
    fraction=float(frame.exceeds_noise_p95.mean())
    dump(OUT/f"h1_k{k}.json",{"k":k,"n_targets":865,"n_controls":10691,"n_basins":B,
         "gate_H1":{"median_explained_fraction":median,"required_median":0.5,
                    "n_exceeds_noise_p95":int(frame.exceeds_noise_p95.sum()),"fraction_exceeds_noise_p95":fraction,
                    "required_fraction":0.5,"pass":bool(median>=0.5 and fraction>=0.5)},
         "energy_definition":"sqrt(2 mean cross Euclidean distance - mean within each empirical population), all ordered pairs including diagonals",
         "mixture_weights":"Nonnegative, sum1; each control in basin b receives w_b/n_b. All-control baseline is n_b/10691, not equal basin weights.",
         "optimizer":"Convex simplex monotone-restart FISTA, no penalty; numerical tolerance1e-10*(1+q+max|A|+max|C|), KKT/Frank-Wolfe primal-dual gap certificate",
         "n_distinct_target_sizes":len(unique_sizes),"n_null_optimizations":200*len(unique_sizes),
         "null":"200 random control subsets per observed target size; shared bank for identical sizes; without replacement within draw; all controls retained in every mixture and denominator",
         "null_rng":"default_rng(0), ascending unique sizes, 200 draws each",
         "max_null_kkt_gap":max_null_gap,"max_null_explained_fraction_uncertainty":max_uncertainty,
         "max_target_explained_fraction_uncertainty":float((actual["explained_fraction_upper"]-actual["explained_fraction"]).max()),
         "max_null_iterations":max_iterations,"max_target_iterations":int(actual["iterations"].max()),
         "control_distance_matrix":"Exact full10691x10691 float64 in memory for this unit; temporary, not serialized"})
    print(frame[["target","explained_fraction","noise_p95","exceeds_noise_p95"]].tail().to_string(index=False),flush=True)
    print(f"k={k} H1 median {median:.9g}; above noise {fraction:.9g}",flush=True)
    return 0


def aggregate():
    import pandas as pd
    import numpy as np
    for name in ("basins","h1","occupancy"):
        frame=pd.concat([pd.read_csv(OUT/f"{name}_k{k}.csv",float_precision="round_trip") for k in KS],ignore_index=True)
        path=OUT/f"{name}.csv"
        frame.to_csv(path,index=False,float_format="%.17g")
        mark_large(path)
    h2s={str(k):json.loads((OUT/f"h2_k{k}.json").read_text()) for k in KS}
    dump(OUT/"h2.json",{"primary_k":20,"by_k":h2s})
    make_figures()
    write_report("partial")
    return 0


def make_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.stats import rankdata
    FIG.mkdir(parents=True,exist_ok=True)
    t=pd.read_csv(OUT/"h1_k20.csv")
    b=pd.read_csv(OUT/"basins_k20.csv",float_precision="round_trip")
    h=json.loads((OUT/"h2_k20.json").read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    axes[0].hist(t.explained_fraction,bins=30,color="#326778")
    axes[0].axvline(0.5,ls="--",color="#ab633d")
    axes[0].set(xlabel="H1 explained energy-distance fraction",ylabel="Significant targets",title=f"All controls; median {t.explained_fraction.median():.3f}")
    axes[1].scatter(t.noise_p95,t.explained_fraction,s=9,alpha=.5,color="#326778")
    axes[1].plot([0,1],[0,1],ls="--",color="gray")
    axes[1].set(xlabel="Size-matched control-null 95th percentile",ylabel="Target explained fraction",title=f"Above noise: {t.exceeds_noise_p95.sum()}/865",xlim=(0,1),ylim=(0,1))
    fig.savefig(FIG/"h1_reweighting.png",dpi=180);plt.close(fig)
    s=b[b.scheme=="score_J"].sort_values("basin");l=b[b.scheme=="leiden"].sort_values("basin")
    X=np.column_stack([np.ones(len(s)),rankdata(s.n_controls),rankdata(s.spread_trace_covariance)])
    rx=rankdata(s.stiffness);ry=rankdata(s.sensitivity_median_abs_log2_fc)
    rx=rx-X@np.linalg.lstsq(X,rx,rcond=None)[0];ry=ry-X@np.linalg.lstsq(X,ry,rcond=None)[0]
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    axes[0].scatter(rx,ry,s=35,color="#326778")
    a=h["stiffness_partial"]
    axes[0].set(xlabel="Stiffness rank residual | size, spread",ylabel="Sensitivity rank residual | size, spread",title=f"Score J: partial rho={a['rho']:.3f}, p-={a['p_one_sided_negative']:.4g}")
    a=h["leiden_ordinary"]
    axes[1].scatter(l.covariance_depth,l.sensitivity_median_abs_log2_fc,s=35,color="#a76642")
    axes[1].set(xlabel="Leiden covariance depth = 1 / spread",ylabel="Median target |log2 occupancy FC|",title=f"Leiden: ordinary rho={a['rho']:.3f}, p-={a['p_one_sided_negative']:.4g}")
    fig.savefig(FIG/"h2_depth_sensitivity.png",dpi=180);plt.close(fig)
    with np.load(CACHE/"occupancy_k20.npz",allow_pickle=False) as a:
        names=a["targets"];fc=a["score_J_log2_fc"]
    examples=t.sort_values(["energy_control","target"],ascending=[False,True]).head(3).target.tolist()
    order=np.argsort(s.stiffness.to_numpy())
    idx=[int(np.flatnonzero(names==name)[0]) for name in examples]
    fig,ax=plt.subplots(figsize=(11,3.4),constrained_layout=True)
    extent=max(float(np.abs(fc[idx][:,order]).max()),1e-8)
    image=ax.imshow(fc[idx][:,order],aspect="auto",cmap="RdBu_r",vmin=-extent,vmax=extent)
    ax.set(yticks=range(len(examples)),yticklabels=examples,xlabel="Score-J basins sorted shallow to deep",title="Largest all-control energy-distance targets: occupancy log2 FC")
    fig.colorbar(image,ax=ax,label="log2 occupancy FC")
    fig.savefig(FIG/"example_occupancy.png",dpi=180);plt.close(fig)


def repair_precision():
    import numpy as np
    import pandas as pd
    from scipy.stats import rankdata
    ensure_inputs()
    receipt=OUT/"serialization_precision_repair.json"
    if receipt.exists():
        raise RuntimeError("Precision repair receipt already exists; no automatic repeat")
    immutable=[]
    for k in KS:
        immutable += [OUT/f"h1_k{k}.json",OUT/f"h2_k{k}.json",CACHE/f"model_k{k}.pt",
                      CACHE/f"states_k{k}.npz",CACHE/f"refit_k{k}.npz",CACHE/f"occupancy_k{k}.npz",CACHE/f"h1_k{k}.npz"]
    before={str(p.relative_to(ROOT)):digest(p) for p in immutable}
    changes={}
    for k in KS:
        path=OUT/f"basins_k{k}.csv"
        frame=pd.read_csv(path)
        changes[str(k)]={}
        with np.load(CACHE/f"occupancy_k{k}.npz",allow_pickle=False) as cache:
            for scheme in ("score_J","leiden"):
                mask=frame.scheme==scheme
                values=cache[f"{scheme}_sensitivity"]
                np.testing.assert_array_equal(frame.loc[mask,"basin"],np.arange(len(values)))
                old=frame.loc[mask,"sensitivity_median_abs_log2_fc"].to_numpy()
                changes[str(k)][scheme]={"ranks_coalesced_by_csv_rounding":int(np.count_nonzero(rankdata(old)!=rankdata(values))),
                     "max_serialization_absolute_difference":float(np.max(np.abs(old-values)))}
                frame.loc[mask,"sensitivity_median_abs_log2_fc"]=values
        frame.to_csv(path,index=False,float_format="%.17g")
        restored=pd.read_csv(path,float_precision="round_trip")
        with np.load(CACHE/f"occupancy_k{k}.npz",allow_pickle=False) as cache:
            for scheme in ("score_J","leiden"):
                np.testing.assert_array_equal(restored.loc[restored.scheme==scheme,"sensitivity_median_abs_log2_fc"],cache[f"{scheme}_sensitivity"])
    after={str(p.relative_to(ROOT)):digest(p) for p in immutable}
    assert before==after, "STOP: a scientific result/cache changed during serialization repair"
    ensure_inputs()
    dump(receipt,{"status":"pass","csv_format":"17 significant digits; float_precision=round_trip",
         "changes":changes,"scientific_jsons_and_caches_unchanged":True,"immutable_sha256":before,
         "refits":0,"resampling":0,"statistical_parameter_changes":0,"repaired_utc":now()})
    print(json.dumps(changes,indent=2),flush=True)
    return 0


def final_audit():
    """Validate published counts/statistics, source scope and completed units."""
    import fnmatch
    import re
    import numpy as np
    import pandas as pd
    from scipy.stats import rankdata
    ensure_inputs()
    task=ROOT/"coord/orient/tasks/S18b.md"
    contract=json.loads(re.findall(r"```json\n(.*?)\n```",task.read_text(),re.S)[-1])
    planner_task=subprocess.check_output(["git","show",f"{PLANNER_COMMIT}:coord/orient/tasks/S18b.md"],cwd=ROOT)
    assert task.read_bytes()==planner_task, "Authorized verbatim task amendment changed"
    changed=subprocess.check_output(["git","diff","--name-only","orientation...HEAD"],cwd=ROOT,text=True).splitlines()
    exception="coord/orient/tasks/S18b.md"
    outside=[p for p in changed if p!=exception and not any(fnmatch.fnmatchcase(p,g) for g in contract["allowed_paths"])]
    assert not outside, f"Outside authorized paths: {outside}"
    for path in contract["required_files"]:
        subprocess.run(["git","cat-file","-e",f"HEAD:{path}"],cwd=ROOT,check=True)
    for name in ("tests_before","tests_after"):
        assert json.loads((LOGS/f"{name}.checkpoint.json").read_text())["exit_code"]==0
    assert cpu_hours()<BUDGET
    occupancy_frame=pd.read_csv(OUT/"occupancy.csv")
    all_basins=pd.read_csv(OUT/"basins.csv",float_precision="round_trip")
    all_h1=pd.read_csv(OUT/"h1.csv")
    details=[]; qc={}
    for k in KS:
        states=json.loads((OUT/f"states_k{k}.json").read_text())
        B=states["n_basins"]
        target=pd.read_csv(OUT/f"h1_k{k}.csv").sort_values("target")
        recorded_h1=json.loads((OUT/f"h1_k{k}.json").read_text())
        recorded_h2=json.loads((OUT/f"h2_k{k}.json").read_text())
        assert len(target)==865 and target.target.nunique()==865
        assert len(all_h1[all_h1.k==k])==865
        np.testing.assert_allclose(target.explained_fraction.median(),recorded_h1["gate_H1"]["median_explained_fraction"],rtol=0,atol=1e-11)
        assert int((target.explained_fraction>target.noise_p95).sum())==recorded_h1["gate_H1"]["n_exceeds_noise_p95"]
        group=all_basins[all_basins.k==k]
        sensitivity={}
        qc[str(k)]={}
        with np.load(CACHE/f"occupancy_k{k}.npz",allow_pickle=False) as cache:
            for scheme in ("score_J","leiden"):
                b=group[group.scheme==scheme].sort_values("basin")
                assert len(b)==B and int(b.n_controls.sum())==10691
                if scheme=="score_J":
                    assert int(b.n_controls.min())>=100
                rows=occupancy_frame[(occupancy_frame.k==k)&(occupancy_frame.scheme==scheme)]
                assert len(rows)==865*B
                counts=rows.pivot(index="target",columns="basin",values="n_target_basin").loc[target.target].to_numpy()
                nc=rows.groupby("basin").n_controls_assigned.first().sort_index().to_numpy()
                nt=target.n_cells.to_numpy()
                np.testing.assert_array_equal(counts.sum(1),nt)
                assert int(nc.sum())==10691
                fc=np.log2((counts+1)/(nt[:,None]+B))-np.log2((nc+1)/(10691+B))[None,:]
                sensitivity[scheme]=np.median(np.abs(fc),axis=0)
                np.testing.assert_allclose(fc,cache[f"{scheme}_log2_fc"],rtol=0,atol=1e-12)
                np.testing.assert_array_equal(rankdata(sensitivity[scheme]),rankdata(b.sensitivity_median_abs_log2_fc))
                qc[str(k)][scheme]={"n_zero_vote_assigned_controls":int((nc==0).sum()),"min_vote_assigned_controls":int(nc.min())}
        for name,depth in (("stiffness",group[group.scheme=="score_J"].sort_values("basin").stiffness.to_numpy()),
                           ("energy_depth",group[group.scheme=="score_J"].sort_values("basin").energy_depth.to_numpy())):
            s=group[group.scheme=="score_J"].sort_values("basin")
            x,y=rankdata(depth),rankdata(sensitivity["score_J"])
            for partial in (False,True):
                if partial:
                    Z=np.column_stack([np.ones(B),rankdata(s.n_controls),rankdata(s.spread_trace_covariance)])
                    xr=x-Z@np.linalg.lstsq(Z,x,rcond=None)[0]
                    yr=y-Z@np.linalg.lstsq(Z,y,rcond=None)[0]
                else:
                    xr=x-x.mean();yr=y-y.mean()
                rho=float(xr@yr/(np.linalg.norm(xr)*np.linalg.norm(yr)))
                key=name+("_partial" if partial else "_ordinary")
                np.testing.assert_allclose(rho,recorded_h2[key]["rho"],rtol=0,atol=1e-12)
        for key in ("stiffness_ordinary","stiffness_partial","energy_depth_ordinary","energy_depth_partial","leiden_ordinary"):
            r=recorded_h2[key]
            assert r["permutations"]==10000
            assert r["p_one_sided_negative"]==(1+r["negative_tail_count"])/10001
            assert r["p_two_sided_absolute"]==(1+r["two_sided_count"])/10001
        assert recorded_h2["leiden_partial"] is None
        banks=sorted((CACHE/f"null_k{k}").glob("n*.npz"))
        assert len(banks)==recorded_h1["n_distinct_target_sizes"]
        for file in banks:
            with np.load(file,allow_pickle=False) as bank:
                assert bank["indices"].shape==(200,int(file.stem[1:]))
                assert len(bank["explained_fraction"])==200
                assert bank["indices"].min()>=0 and bank["indices"].max()<10691
                assert (np.diff(np.sort(bank["indices"],axis=1),axis=1)>0).all()
                np.testing.assert_allclose(bank["weights"].sum(1),1,rtol=0,atol=1e-12)
                assert (bank["weights"]>=0).all()
        details.append({"k":k,"n_basins":B,"n_targets":865,"n_null_banks":len(banks),"published_statistics_reproduce":True})
    dump(OUT/"final_audit.json",{"status":"pass","source_sha256":SHA,"full_scope_base":"orientation",
         "authorized_exception":exception,"planner_task_blob_unchanged":True,"outside_allowed_paths":outside,
         "checks":details,"assignment_qc":qc,"recorded_cpu_hours_before_audit":cpu_hours(),"checked_utc":now()})
    print("Final audit PASS: published counts, ranks, gates, null banks, required files and full authorized scope",flush=True)
    return 0


def delivery_prepare():
    import zipfile
    stage=ROOT.parent/"S18b_benchmarks.partial.zip"
    final=ROOT.parent/"S18b_benchmarks.zip"
    if stage.exists() or final.exists():
        raise RuntimeError("Delivery archive exists: inspect instead of repeating")
    paths=[]
    for folder in (OUT,FIG):
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p.name!="RESULT_S18b.md" and not p.name.startswith("delivery_prepare.") and not p.name.endswith(".tmp"):
                paths.append(p)
    paths.append(ROOT/"scripts/s18b_occupancy.py")
    with zipfile.ZipFile(stage,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as archive:
        for i,p in enumerate(paths,1):
            compression=zipfile.ZIP_STORED if p.suffix in (".npz",".pt",".png") else zipfile.ZIP_DEFLATED
            archive.write(p,str(p.relative_to(ROOT)),compress_type=compression)
            if i%100==0:
                print(f"[{now()}] archive preparation {i}/{len(paths)} files",flush=True)
        assert archive.testzip() is None
    dump(OUT/"delivery_prepare.json",{"stage_path":str(stage),"staged_bytes":stage.stat().st_size,
         "staged_files":len(paths),"crc_check":"pass","excluded_until_final_accounting":["RESULT_S18b.md","coord/orient/reports/S18b.*","delivery_prepare metadata/log/checkpoint"],
         "original_raw_and_S18_cache_duplicated":False,"prepared_utc":now()})
    return 0


def delivery_finalize():
    """Operational finalization after the timed preparation receipt is complete."""
    import zipfile
    stage=ROOT.parent/"S18b_benchmarks.partial.zip"
    final=ROOT.parent/"S18b_benchmarks.zip"
    bundle=ROOT.parent/"S18b_result.bundle"
    assert stage.is_file() and not final.exists() and not bundle.exists()
    report=json.loads((ROOT/"coord/orient/reports/S18b.json").read_text())
    assert report["status"]=="done" and report["compute"]["cpu_hours"]<BUDGET
    pending=[OUT/"RESULT_S18b.md",ROOT/"coord/orient/reports/S18b.md",ROOT/"coord/orient/reports/S18b.json",
             OUT/"delivery_prepare.json",LOGS/"delivery_prepare.log",LOGS/"delivery_prepare.checkpoint.json"]
    with zipfile.ZipFile(stage,"a",compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as archive:
        names=set(archive.namelist())
        for p in pending:
            relative=str(p.relative_to(ROOT))
            assert relative not in names
            archive.write(p,relative)
            assert archive.read(relative)==p.read_bytes()
            names.add(relative)
        assert len(names)==len(archive.namelist())
        for name in ("basins.csv","occupancy.csv","h1.csv","h2.json","RESULT_S18b.md"):
            assert f"benchmarks/perturb_k562_occupancy/{name}" in names
    stage.replace(final)
    subprocess.run(["git","bundle","create",str(bundle),"manus/orient-S18b"],cwd=ROOT,check=True)
    subprocess.run(["git","bundle","verify",str(bundle)],cwd=ROOT,check=True)
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    ref=subprocess.check_output(["git","bundle","list-heads",str(bundle)],cwd=ROOT,text=True).strip()
    assert ref==f"{head} refs/heads/manus/orient-S18b"
    for p in (bundle,final,ROOT/"coord/orient/reports/S18b.md",ROOT/"coord/orient/reports/S18b.json"):
        print(str(p),p.stat().st_size,"bytes",digest(p),flush=True)
    return 0


def regeneration_command(path):
    import re
    path=Path(path)
    cli=".venv/bin/python scripts/s18b_occupancy.py"
    relative=str(path.relative_to(ROOT))
    match=re.search(r"k(20|15|30)",relative)
    if match:
        k=match.group(1)
        if any(word in relative for word in ("model_k","refit_k")):
            return f"{cli} unit refit_k{k} -- {cli} refit --k {k} (new output directory; retain existing successful states)"
        if any(word in relative for word in ("null_k","h1_k")):
            return f"{cli} unit h1_k{k} -- {cli} h1 --k {k} (new output directory, same frozen control inputs and seeds)"
        if "h2_k" in relative:
            return f"{cli} unit h2_k{k} -- {cli} h2 --k {k}"
        if any(word in relative for word in ("occupancy_k","centroid_shifts_k")):
            return f"{cli} unit occupancy_k{k} -- {cli} occupancy --k {k}"
        return f"{cli} unit states_k{k} -- {cli} states --k {k}; then recorded occupancy and h2 units for the final basin table"
    if path.name=="pip-freeze.txt":
        return ".venv/bin/python -m pip freeze > benchmarks/perturb_k562_occupancy/pip-freeze.txt"
    if path.name=="input_verification.json":
        return f"{cli} unit input_verification -- {cli} verify-input"
    if path.name in ("source_archive.json","environment.json"):
        return f"git show manus/orient-S18b:{relative} > {relative} (restore recorded source/run provenance, not simulated current metadata)"
    if path.name=="final_audit.json":
        return f"{cli} unit final_audit -- {cli} audit"
    if path.name=="delivery_prepare.json":
        return f"{cli} unit delivery_prepare -- {cli} delivery-prepare"
    if path.name=="serialization_precision_repair.json":
        return f"{cli} unit precision_repair -- {cli} repair-precision (only when restoring old rounded CSVs; never repeat on a completed repair)"
    if "run_logs" in relative:
        name=path.name.split(".")[0]
        checkpoint=LOGS/f"{name}.checkpoint.json"
        if checkpoint.exists():
            return " ".join(json.loads(checkpoint.read_text())["command"])+" (timed with the corresponding unit name in a fresh output directory)"
    return f"{cli} unit aggregate -- {cli} aggregate (after all recorded k20,15,30 units; metadata sources/provenance are retained)"


def write_report(status="partial",question=None):
    environment=json.loads((OUT/"environment.json").read_text()) if (OUT/"environment.json").exists() else {}
    units=[json.loads(p.read_text()) for p in sorted(LOGS.glob("*.checkpoint.json"))]
    versions={name:importlib.metadata.version(name) for name in ("numpy","scipy","pandas","anndata","scanpy","scikit-learn","torch","leidenalg","igraph","matplotlib","pytest")}
    versions["python"]=platform.python_version()
    def optional(name):
        p=OUT/name
        return json.loads(p.read_text()) if p.exists() else None
    primary_states=optional("states_k20.json");H1=optional("h1_k20.json");H2=optional("h2_k20.json");ref=optional("refit_k20.json")
    sensitivity={str(k):{"states":optional(f"states_k{k}.json"),"h1":optional(f"h1_k{k}.json"),"h2":optional(f"h2_k{k}.json")} for k in KS}
    for k in (15,30):
        entry=sensitivity[str(k)]
        for category,gatename in (("h1","gate_H1"),("h2","gate_H2")):
            if entry[category]:
                numbers=entry[category].pop(gatename)
                numbers.pop("pass",None)
                entry[category]["report_only_statistics"]=numbers
    metrics={"n_basins":primary_states["n_basins"] if primary_states else None,
             "gate_H1_reweighting":H1["gate_H1"] if H1 else None,
             "gate_H2_depth_sensitivity":H2["gate_H2"] if H2 else None,
             "leiden_comparison":{"ordinary":H2["leiden_ordinary"],"partial":None,
                                   "partial_reason":H2["leiden_partial_reason"]} if H2 else None,
             "versions":versions,"cpu_hours":cpu_hours(),"refit_check":ref,
             "report_only_by_k":sensitivity}
    observations=["Original SHA matches; 10,691 controls and the same 865 significant targets; original inputs and caches are read-only."]
    if ref:
        observations.append(f"Primary exact refit median relative Frobenius difference: {ref['median_relative_Frobenius_difference']:.9g} (stop threshold0.05).")
    if primary_states:
        observations.append(f"Primary endpoints formed {primary_states['n_initial_endpoint_clusters']} initial clusters; nearest-centroid minimum-size merging produced {primary_states['n_basins']} basins. Fraction drift-stopped before/at400 steps: {primary_states['fraction_stopped']:.6f}.")
        observations.append("Most primary controls reached the fixed 400-step cap rather than the step-norm convergence criterion. The report-only energy depth uses the lowest observed terminal endpoint energy in each basin, not an additionally optimized stationary minimum. No extra integration or endpoint optimization was run.")
    if H1:
        observations.append(f"H1 median explained fraction {H1['gate_H1']['median_explained_fraction']:.6f}; above size-matched null95 for {H1['gate_H1']['n_exceeds_noise_p95']}/865 targets.")
        observations.append(f"Primary H1 has {H1['n_distinct_target_sizes']} distinct target sizes and {H1['n_null_optimizations']} certified null optimizations. Maximum target explained-fraction numerical bracket width is {H1['max_target_explained_fraction_uncertainty']:.3g}; maximum null bracket width is {H1['max_null_explained_fraction_uncertainty']:.3g}.")
    if H2:
        ordinary=H2["stiffness_ordinary"];leiden=H2["leiden_ordinary"]
        operatorfree=(leiden["rho"]<0 and leiden["p_one_sided_negative"]<.05)
        observations.append(f"Operator-free Leiden spread-only comparison {'also meets the negative-association significance criterion' if operatorfree else 'does not meet the negative-association significance criterion'}: ordinary rho={leiden['rho']:.6f}, p-={leiden['p_one_sided_negative']:.6g}. This is not adjusted equivalently to score J because its partial correlation is undefined.")
        observations.append(f"Unadjusted score-J stiffness rho={ordinary['rho']:.6f}, p-={ordinary['p_one_sided_negative']:.6g}.")
        if not H2["gate_H2"]["pass"] and not operatorfree:
            observations.append("Both the primary score-J partial comparison and the operator-free Leiden ordinary comparison fail the predicted negative-association criterion. These requested tests do not demonstrate a predictive advantage for score J; they are not an equivalently adjusted paired superiority test.")
    audit=optional("final_audit.json")
    if audit:
        observations.append(f"Completed output audit: {audit['status']}; assigned-control QC by k and scheme: {json.dumps(audit['assignment_qc'])}.")
    validation={}
    for name in ("tests_before","tests_after","final_input_verification","final_audit","coord_check_reviews"):
        p=LOGS/f"{name}.checkpoint.json"
        validation[name]=json.loads(p.read_text()) if p.exists() else None
    later=LOGS/"final_audit_after_precision.checkpoint.json"
    if later.exists():
        validation["final_audit_initial"]=validation["final_audit"]
        validation["final_audit"]=json.loads(later.read_text())
    if optional("serialization_precision_repair.json"):
        deviations_note="An additional artifact audit caught four k30 sensitivity ranks coalesced by 12-significant-digit CSV formatting. Columns were restored from unchanged cached full-precision arrays, written with 17 digits and read round-trip. Original H1/H2 scientific JSONs, model/state/occupancy caches, correlations and permutation p-values are unchanged; no refits or resampling."
    else:
        deviations_note=None
    deviations=["Initial pre-result block resolved by the user's verbatim planner Amendment Revision1, committed as56e9c0045387bf6049dbd40c8b049ef2d55a4cae; Leiden partial is null, one-sided negative p is gated, two-sided absolute p is reported.",
                "Missing S18 model weights refitted only by the exact S18 adapter/configuration; embedding, cohort and accepted S18 caches were reused unchanged.",
                "Leiden graph/clustering arguments not specified by the task retain Scanpy defaults; resolution searched on controls only for exact matched state count, with seed0 and no nearest-count substitution.",
                "Small endpoint clusters merged iteratively into nearest endpoint-centroid cluster; smallest size/source-label order and stable tie rules are recorded. Original memberships define depth, size/spread and uniform H1 mixtures; all cells including controls use the prescribed15-vote classifier for occupancy.",
                "For identical target sizes, the same200 random-control null subsets are reused; each target has the exact prescribed size-conditional null. Full control pool remains in every mixture and energy denominator.",
                "H1 minimized exact squared energy by monotone-restart accelerated simplex projection; no statistical regularization. KKT gaps certify numerical optimization, and explained-fraction uncertainty is recorded.",
                "Only primary k=20 is gated; k=15/30 repeat the corresponding control-only PCA-k pipeline as report-only sensitivities. All centroid shifts use every control.",
                "Workstream mentions 3 xfails, but the supplied protected test file contains none; tests are executed unchanged and all 48 pass before and after analysis.",
                "Mechanical scientific-diff check uses the user-authorized planner-amendment commit as baseline because its expressly requested task-file edit lies outside scientific allowed_paths."]
    if deviations_note:
        deviations.append(deviations_note)
    if question:
        deviations.append(question)
    script=ROOT/"scripts/s18b_occupancy.py"
    artifacts=[{"path":"scripts/s18b_occupancy.py","bytes":script.stat().st_size,"sha256":digest(script),"committed":True,
                "regenerate":"Restore scripts/s18b_occupancy.py from the returned S18b_result.bundle"}]
    for folder in (OUT,FIG):
        if not folder.exists():
            continue
        for p in sorted(folder.rglob("*")):
            if not p.is_file() or p.name=="RESULT_S18b.md" or p.name.endswith(".tmp"):
                continue
            rel=str(p.relative_to(ROOT))
            ignored=subprocess.run(["git","check-ignore","-q",rel],cwd=ROOT).returncode==0
            artifacts.append({"path":rel,"bytes":p.stat().st_size,"sha256":digest(p),"committed":not ignored,
                              "regenerate":regeneration_command(p)})
    report={"task":"orient/S18b","status":status,"started_utc":min(u["started_utc"] for u in units),"finished_utc":now(),
            "compute":{"cpu_hours":cpu_hours(),"gpu_hours":0,"hardware":environment,"units":units},"metrics":metrics,
            "artifacts":artifacts,"deviations":deviations,"questions":[question] if question else [],
            "validation":validation,"observations":observations,"suggestions":["In a separately approved follow-up, assess endpoint-clustering and drift-convergence stability because most controls reached the 400-step cap and 7,901 initial endpoint clusters were merged to 59 basins; also assess original versus vote-classified memberships. No such sensitivity was run here."],
            "input_verification":json.loads((OUT/"input_verification.json").read_text()),"planner_amendment_commit":PLANNER_COMMIT}
    dump(ROOT/"coord/orient/reports/S18b.json",report)
    lines=["# orient/S18b — K562 control-basin occupancy","","## Summary","",f"Status: **{status}**. All predictor/state fits use only 10,691 non-targeting controls; the 865 targets are unchanged. Every energy/reference population and centroid shift uses all controls.",f"Measured recorded-unit CPU hours: {cpu_hours():.6f} / 12."]
    if ref:
        lines.append(f"Refit-check median relative Frobenius difference: {ref['median_relative_Frobenius_difference']:.9g}; required <=0.05 before use.")
    for key in ("gate_H1_reweighting","gate_H2_depth_sensitivity"):
        if metrics[key]:
            lines.append(f"{key}: {'PASS' if metrics[key]['pass'] else 'FAIL'}; `{json.dumps(metrics[key])}`.")
    if H2:
        lines.append("The primary score-J and operator-free Leiden comparisons do not show the predicted negative association; neither demonstrates a predictive advantage in these requested tests." if not H2["gate_H2"]["pass"] and not operatorfree else f"Leiden ordinary rho={leiden['rho']:.6f}, p-negative={leiden['p_one_sided_negative']:.6g}; partial null by construction.")
    lines.append(f"Original K562 SHA-256: `{SHA}` (verified unchanged). Cloud-only execution; no Mac access, merge or push.")
    lines += ["","## What ran",""]
    lines += [f"- `{' '.join(u['command'])}` — exit {u['exit_code']}, wall {u['wall_seconds']:.3f}s, CPU {u['cpu_hours']:.6f}h." for u in sorted(units,key=lambda x:x['started_utc'])]
    lines += ["",f"Fresh `.venv`; pip freeze committed. Machine/environment: `{json.dumps(environment)}`. Per-unit RAM/RSS and thread counts are in JSON checkpoints.","","## Gate numbers","","| Metric | Value | Requirement/result |","|---|---:|---|"]
    if primary_states:
        lines.append(f"| Primary basins | {primary_states['n_basins']} | Report |")
    if ref:
        lines.append(f"| Refit median relative Frobenius difference | {ref['median_relative_Frobenius_difference']:.9g} | <=0.05 before use |")
    if H1:
        g=H1["gate_H1"]
        lines += [f"| H1 median explained fraction | {g['median_explained_fraction']:.9g} | >=0.5 |",f"| H1 fraction above target-size null95 | {g['fraction_exceeds_noise_p95']:.9g} ({g['n_exceeds_noise_p95']}/865) | >=0.5; {'PASS' if g['pass'] else 'FAIL'} together |"]
    if H2:
        for name in ("stiffness_partial","stiffness_ordinary","energy_depth_partial","energy_depth_ordinary","leiden_ordinary"):
            a=H2[name]
            lines.append(f"| {name} rho | {a['rho']:.9g} | p-={a['p_one_sided_negative']:.9g}; p-two={a['p_two_sided_absolute']:.9g} |")
        lines.append("| Leiden partial rho/p | null | Undefined by construction; planner amendment |")
    if all(optional(f"h1_k{k}.json") and optional(f"h2_k{k}.json") for k in KS):
        lines += ["","### PCA dimension sensitivity (report-only for k=15/30)","","| k | Basins | H1 median | H1 above noise | Partial stiffness rho | p-negative | Leiden ordinary rho | p-negative |","|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for k in KS:
            a=optional(f"h1_k{k}.json");b=optional(f"h2_k{k}.json")
            lines.append(f"| {k} | {b['n_basins']} | {a['gate_H1']['median_explained_fraction']:.6f} | {a['gate_H1']['fraction_exceeds_noise_p95']:.6f} | {b['stiffness_partial']['rho']:.6f} | {b['stiffness_partial']['p_one_sided_negative']:.6g} | {b['leiden_ordinary']['rho']:.6f} | {b['leiden_ordinary']['p_one_sided_negative']:.6g} |")
    lines += ["","H1 uses E=sqrt(2 cross mean minus within means), not squared E in the explained-fraction ratio. Weight optimization uses its monotone-equivalent squared objective, nonnegative weights summing to 1, and equal cell mass within each original basin. The baseline weights are basin occupancy n_b/10,691. No control subsampling or removal is performed. Control null draws only supply pseudo-targets. Partial Spearman residualizes ranks against an intercept, basin-size rank and covariance-trace rank; depth is permuted 10,000 times and re-residualized. Negative-tail p=(1+count(rho_perm<=rho_obs))/10,001; two-sided p uses absolute rho.","","## Deviations",""]+[f"- {d}" for d in deviations]
    for section,items in (("Observations",observations),("Suggestions",report["suggestions"]),("Questions",report["questions"] or ["None."])):
        lines += ["",f"## {section}",""]+[f"- {a}" for a in items]
    lines += ["","## Artifacts","","`basins.csv`, `occupancy.csv`, `h1.csv`, `h2.json`, per-k tables, state/model/null caches, figures, unit logs/checkpoints and this report are in the returned benchmark download. Large/ignored files remain uncommitted and are identified by artifact size/hash in JSON. Temporary full-control distance matrices are not serialized. Original raw input/S18 caches are not duplicated; their exact checksums are in `input_verification.json`.","","## Revision 1","","The authorized planner amendment is appended verbatim to `coord/orient/tasks/S18b.md`. The initial blocked report is preserved in commitc87e793. No scientific result existed when the amendment was fixed.",""]
    lines += ["## Reproduction and reuse","","Clone the returned bundle on `manus/orient-S18b`, restore the benchmark ZIP at the repository root, and retain the original S18 inputs/caches at the paths listed in `input_verification.json`. The source file must match the stated SHA-256; restore S18 caches from the S18 benchmark download, not reconstructed or imputed arrays. Create a fresh `.venv` and install `-e \".[dev]\"` as recorded. Saved S18b network weights and completed caches are reused; successful refits or null banks are never overwritten.","","The ordered timed commands are in the checkpoints. In a genuinely fresh output directory, execute `verify-input`, then for k=20,15,30 the `refit`, `states`, `occupancy`, `h1`, and `h2` subcommands; finally `aggregate`, the required tests, `audit`, and report generation. Each numerical command uses `unit <name> -- .venv/bin/python scripts/s18b_occupancy.py <command>`. For an already restored result, use the saved state/model/occupancy caches for reporting and inspection rather than repeating fits. Exact file-specific regeneration/restoration commands accompany each JSON artifact.",""]
    text="\n".join(lines)
    (ROOT/"coord/orient/reports/S18b.md").write_text(text)
    (OUT/"RESULT_S18b.md").write_text(text)
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    unit=sub.add_parser("unit");unit.add_argument("name");unit.add_argument("command",nargs=argparse.REMAINDER)
    sub.add_parser("verify-input")
    for name in ("refit","states","occupancy","h1","h2"):
        p=sub.add_parser(name);p.add_argument("--k",type=int,choices=KS,required=True)
    sub.add_parser("aggregate")
    for name in ("audit","repair-precision","delivery-prepare","delivery-finalize"):
        sub.add_parser(name)
    report=sub.add_parser("report");report.add_argument("--status",choices=("partial","done","blocked"),default="partial");report.add_argument("--question")
    args=parser.parse_args()
    if args.action=="unit":
        cmd=args.command[1:] if args.command and args.command[0]=="--" else args.command
        if not cmd:
            parser.error("unit requires a command")
        return run_unit(args.name,cmd)
    if args.action=="verify-input":
        return verify_input()
    if args.action=="aggregate":
        return aggregate()
    if args.action=="repair-precision":
        return repair_precision()
    if args.action=="audit":
        return final_audit()
    if args.action=="delivery-prepare":
        return delivery_prepare()
    if args.action=="delivery-finalize":
        return delivery_finalize()
    if args.action=="report":
        return write_report(args.status,args.question)
    return {"refit":refit,"states":build_states,"occupancy":occupancy,"h1":h1,"h2":h2}[args.action](args.k)


if __name__=="__main__":
    sys.exit(main())
