#!/usr/bin/env python3
"""orient/S13 — checkpointed whitened OT flow-J comparison.

Operator fits consume only observed coordinates plus sampling day/time. Synthetic
truth is read only by score_synthetic(), after all fits in a unit are complete.
Run every command through a detached /usr/bin/time -l wrapper.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, itertools, json, os, re, resource, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path: sys.path.insert(0, str(SCRIPTS))
OUT = ROOT / "benchmarks/synthetic_multimodal/s13"
ARR, CK, LOG, FIG = OUT / "arrays", OUT / "checkpoints", OUT / "logs", ROOT / "benchmarks/figures/orient/s13"
S12ARR = ROOT / "benchmarks/synthetic_multimodal/data/s12"
S12CK = ROOT / "benchmarks/synthetic_multimodal/s12/checkpoints"
S10REAL = ROOT / "benchmarks/synthetic_multimodal/data/s10/real"
SYN = tuple((p, s) for p in ("v3_transition", "v3_default") for s in range(3))
EMB = ("pca10", "pca20", "pca50", "fa20")
LEMB = ("pca30", "fa30")
MEMSTOP = 24 * 1024**3
TASK = "orient/S13"

def now(t: float | None = None) -> str:
    return datetime.fromtimestamp(time.time() if t is None else t, timezone.utc).isoformat().replace("+00:00", "Z")
def sid(p: str, s: int) -> str: return f"{p}_s{s}"
def hash_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for x in iter(lambda: f.read(1 << 20), b""): h.update(x)
    return h.hexdigest()
def jdump(x: Any) -> Any:
    if isinstance(x, dict): return {str(k): jdump(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [jdump(v) for v in x]
    if isinstance(x, np.ndarray): return jdump(x.tolist())
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x)
    if isinstance(x, float) and not np.isfinite(x): return None
    return x
def wjson(p: Path, x: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True); q=p.with_suffix(p.suffix+".tmp")
    q.write_text(json.dumps(jdump(x),indent=2,sort_keys=True)+"\n"); q.replace(p)
def rjson(p: Path) -> dict[str,Any]: return json.loads(p.read_text())
def cpu() -> float:
    a,b=resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime)
def rss() -> int:
    v=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss); return v if sys.platform=="darwin" else v*1024
def chk(name: str, started: float, c0: float, data: dict[str,Any]) -> dict[str,Any]:
    if rss()>MEMSTOP: raise RuntimeError(f"MEMORY STOP: {rss()/1024**3:.3f} GiB > 24 GiB")
    rec={"task":TASK,"unit":name,"started_utc":now(started),"finished_utc":now(),"wall_seconds":round(time.time()-started,3),"in_process_cpu_seconds":round(cpu()-c0,3),"peak_rss_bytes":rss(),**data}
    wjson(CK/f"{name}.json",rec); return rec
def module(name: str, path: Path) -> Any:
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError(f"cannot import {path}")
    m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m
def s12() -> Any: return module("s13_readonly_s12",SCRIPTS/"s12_step0.py")
def s11() -> Any: return module("s13_readonly_s11",SCRIPTS/"s11_velocities.py")
def s10real() -> Any: return module("s13_readonly_s10real",SCRIPTS/"s10_real_transitions.py")
def set_threads() -> None:
    n=min(12,int(os.environ.get("OMP_NUM_THREADS","8")))
    import torch
    torch.set_num_threads(n)
    try: torch.set_num_interop_threads(min(4,n))
    except RuntimeError: pass
def req_preflight() -> dict[str,Any]:
    p=CK/"preflight.json"
    if not p.exists() or not rjson(p).get("passed"): raise RuntimeError("run preflight successfully first")
    return rjson(p)
def arr_path(dataset: str, emb: str) -> Path: return ARR/f"{dataset}_{emb}.npz"
def s12_embed(p: str,s: int) -> Path: return S12ARR/sid(p,s)/"embeddings.npz"
def s12_raw(p: str,s: int,e: str) -> Path: return S12ARR/sid(p,s)/f"step0_ot_{e}.npz"
def unit_name(dataset: str,e: str) -> str: return f"unit_{dataset}_{e}"

def whitening(z: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray,dict[str,Any]]:
    mean=z.mean(0); cov=np.cov(z,rowvar=False); val,V=np.linalg.eigh(cov); floor=1e-6*float(val.max()); clip=np.maximum(val,floor)
    W=(V*(1/np.sqrt(clip)))@V.T; WI=(V*np.sqrt(clip))@V.T; zw=(z-mean)@W
    return zw,W,WI,{"eigen_min":float(val.min()),"eigen_max":float(val.max()),"eigen_floor":floor,"n_floored":int((val<floor).sum()),"whiten_cov_max_abs_error":float(np.max(abs(np.cov(zw,rowvar=False)-np.eye(z.shape[1]))))}

def fit_model(z: np.ndarray, v: np.ndarray, rows: np.ndarray) -> tuple[Any,dict[str,Any]]:
    """Exactly MeasuredDrift(d, seed=0): no S13 tuning or alternate config."""
    from scjdo2.measured_drift import MeasuredDrift
    z=np.asarray(z,np.float32); rows=np.asarray(rows,int); vv=np.asarray(v,np.float32)[rows]; good=np.isfinite(vv).all(1)
    if good.sum()<10: raise RuntimeError("fewer than ten finite velocity rows")
    t=time.time(); model=MeasuredDrift(z.shape[1],seed=0); model.fit(z[rows][good],vv[good],verbose=False)
    device=str(next(model._model.parameters()).device); model._model.to("cpu")
    return model,{"n_fit_cells":int(good.sum()),"fit_device":device,"jacobian_device":"cpu","seconds":round(time.time()-t,3),"seed":0,"defaults":True}
def jac(model: Any,z: np.ndarray) -> np.ndarray: return np.asarray(model.jacobian_autodiff(np.asarray(z,np.float32)),np.float32)
def velocity(model: Any,z: np.ndarray) -> np.ndarray: return np.asarray(model.velocity(np.asarray(z,np.float32)),np.float64)

def energy(x: np.ndarray,y: np.ndarray,block: int=256) -> float:
    from scipy.spatial.distance import cdist
    def mean(a: np.ndarray,b: np.ndarray) -> float:
        return sum(float(cdist(a[i:min(i+block,len(a))],b).sum()) for i in range(0,len(a),block))/(len(a)*len(b))
    return float(np.sqrt(max(0.,2*mean(x,y)-mean(x,x)-mean(y,y))))

def heldout(s: Any,z: np.ndarray,zw: np.ndarray,WI: np.ndarray,mean: np.ndarray,days: np.ndarray,hours: np.ndarray,L: np.ndarray,Lcommon: np.ndarray) -> tuple[list[dict[str,Any]],dict[str,Any]]:
    u=np.unique(days)
    if len(u)<3: raise RuntimeError("need three sampling days for heldout prediction")
    d0,d1,d2=u[:3]; pair=(days==d0)|(days==d1); first,mid,last=days==d0,days==d1,days==d2
    vp,pairs=s.ot_velocity(zw[pair],days[pair],hours[pair]); model,info=fit_model(zw[pair],vp,np.flatnonzero(np.isfinite(vp).all(1)))
    gap=float(np.median(hours[last])-np.median(hours[mid]))
    if gap<=0 or abs(gap-round(gap))>1e-8: raise RuntimeError(f"invalid Euler gap {gap}")
    pred=np.asarray(zw[mid],float).copy()
    for _ in range(int(round(gap))): pred += velocity(model,pred)
    pred=pred@WI+mean; nomove=np.asarray(z[mid],float); meanmove=nomove+(z[mid].mean(0)-z[first].mean(0))
    inv=np.linalg.pinv(np.asarray(L,float)); observed=(np.asarray(z[last],float)@inv)@Lcommon
    rows=[]
    for method,x in (("whitened_flow",pred),("no_movement",nomove),("mean_displacement",meanmove)):
        rows.append({"method":method,"energy_distance":energy((x@inv)@Lcommon,observed),"n_middle":int(mid.sum()),"n_last":int(last.sum())})
    return rows,{"first_day":float(d0),"middle_day":float(d1),"last_day":float(d2),"gap_hours":gap,"ot_pairs":pairs,"measured_drift":info}

def score_synthetic(s: Any,J: np.ndarray,truth: dict[str,Any],axis: np.ndarray,WI: np.ndarray|None) -> tuple[dict[str,Any],np.ndarray]:
    """The only synthetic truth-consuming function; called after all fits."""
    classes=s.truth_classes(truth["z"],truth["z1_crit"]); sp=s.spectra(J); vec=np.asarray(sp["vec1"],float)
    if WI is not None:
        vec=vec@WI; vec/=np.maximum(np.linalg.norm(vec,axis=1,keepdims=True),1e-12)
    sad=classes["saddle"]; al=np.abs(vec[sad]@axis) if sad.any() else np.array([])
    return {"auc_lam1":float(s.auc(sp["lam1"],classes)),"n_saddle":int(sad.sum()),"n_committed":int(classes["committed"].sum()),"fate_axis_abs_cos_at_saddle":float(np.median(al)) if len(al) else float("nan"),"median_lam1_saddle":float(np.median(sp["lam1"][sad])) if sad.any() else float("nan"),"median_lam1_committed":float(np.median(sp["lam1"][classes["committed"]])) if classes["committed"].any() else float("nan")},np.asarray(sp["lam1"],np.float32)

def preflight() -> None:
    import anndata as ad
    started,c0=time.time(),cpu(); s=s12(); synth={}
    for p,seed in SYN:
        row=s.manifest_row(p,seed,verify=True); ep=s12_embed(p,seed); eck=rjson(S12CK/f"embed_{p}_s{seed}.json")
        if hash_file(ep)!=eck["sha256"]: raise RuntimeError(f"CHECKSUM FAILURE: {ep}")
        raw={}
        for e in EMB:
            rp=s12_raw(p,seed,e); rck=rjson(S12CK/f"ot_{p}_s{seed}_{e}.json")
            if hash_file(rp)!=rck["sha256"]: raise RuntimeError(f"CHECKSUM FAILURE: {rp}")
            raw[e]={"path":str(rp.relative_to(ROOT)),"sha256":rck["sha256"]}
        synth[sid(p,seed)]={"input_sha256":row["sha256"],"embedding_sha256":eck["sha256"],"raw":raw}
    m=s11(); raw=Path(m.LARRY_RAW); rawhash=hash_file(raw); old=rjson(ROOT/"benchmarks/real_flow/checkpoints/prepare_larry.json")
    if rawhash!=old["input_sha256"]: raise RuntimeError("CHECKSUM FAILURE: LARRY raw input")
    pp=S10REAL/"arrays/embeddings/larry_x_pca30.npz"
    with np.load(pp,allow_pickle=False) as x: ix=np.asarray(x["sample_index"],int); days=np.asarray(x["day"],float)
    rng=np.random.default_rng(0); expect=np.concatenate([rng.choice(np.flatnonzero(days==d),4000,replace=False) for d in (2.,4.,6.)])
    if not np.array_equal(ix,expect): raise RuntimeError("frozen default_rng(0) LARRY sample mismatch")
    # Conservative admission plan: five S12-equivalents per synthetic unit
    vals={}
    with (ROOT/"benchmarks/synthetic_multimodal/s12/cpu_ledger.csv").open() as f:
        for r in csv.DictReader(f):
            for e in EMB:
                if "/ot_" in r["log"] and f"_{e}.log" in r["log"]: vals.setdefault(e,[]).append(float(r["cpu_s"]))
    med={e:float(np.median(vals[e])) for e in EMB}; synthcpu=5*sum(6*med[e] for e in EMB); hold=sum(3*med[e] for e in EMB)
    rv=[]
    with (ROOT/"benchmarks/synthetic_multimodal/s10/cpu_ledger.csv").open() as f:
        for r in csv.DictReader(f):
            if "real_fit_larry_" in r["log"]: rv.append(float(r["cpu_s"]))
    plan={"basis":"S12/S10 completed timed units; conservative five-equivalent synthetic multiplier","s12_median_cpu_seconds_by_embedding":med,"projected_cpu_hours":(synthcpu+hold+8*np.median(rv)+.35*3600)/3600,"budget_cpu_hours":8.0}
    plan["admitted"]=plan["projected_cpu_hours"]<8
    if not plan["admitted"]: raise RuntimeError(f"COMPUTE STOP: projected {plan['projected_cpu_hours']:.3f} CPU-h")
    OUT.mkdir(parents=True,exist_ok=True); wjson(OUT/"admission_plan.json",plan)
    chk("preflight",started,c0,{"passed":True,"synthetic":synth,"larry":{"raw_path":str(raw),"raw_sha256":rawhash,"sample_path":str(pp.relative_to(ROOT)),"sample_sha256":hash_file(pp),"sample_per_day":4000,"sample_size":int(len(ix))},"admission_plan":plan,"threads_limit":12,"memory_stop_gib":24.0})

def prepare_larry() -> None:
    import anndata as ad
    from sklearn.decomposition import PCA
    req_preflight(); started,c0=time.time(),cpu(); m=s11(); q=s10real(); pp=S10REAL/"arrays/embeddings/larry_x_pca30.npz"
    with np.load(pp,allow_pickle=False) as p, np.load(S10REAL/"larry.npz",allow_pickle=False) as l:
        ix=np.asarray(p["sample_index"],int); hvg=np.asarray(p["hvg_index"],int); target=float(p["normalization_target"]); pmap=np.asarray(p["linear_map"],float); pmean=np.asarray(p["pca_mean"],float); zfa=np.asarray(l["z_fa"],float)
    a=ad.read_h5ad(Path(m.LARRY_RAW)); lib=np.asarray(a.X.sum(axis=1)).ravel().astype(float); X=m._transform_larry_rows(a.X[:,hvg][ix],lib[ix],target)
    fa,fmap,fmean=q.fa_map(X,30); err=float(np.max(abs(fa-zfa)))
    if err>1e-5: raise RuntimeError(f"S10 FA-30 reconstruction mismatch {err}")
    cp=PCA(n_components=50,random_state=0).fit(X); ARR.mkdir(parents=True,exist_ok=True); dest=ARR/"larry_maps.npz"
    np.savez_compressed(dest,pca_map=pmap,pca_mean=pmean,fa_map=np.asarray(fmap,float),fa_mean=np.asarray(fmean,float),common_map=np.asarray(cp.components_.T,float),common_mean=np.asarray(cp.mean_,float),sample_index=ix,hvg_index=hvg,normalization_target=np.float64(target))
    chk("prepare_larry",started,c0,{"path":str(dest.relative_to(ROOT)),"sha256":hash_file(dest),"n_cells":12000,"n_hvg":2000,"sample_per_day":4000,"sample_seed":0,"fa30_reconstruction_max_abs_diff_vs_s10":err,"common_reference":"PCA-50 fitted on all 12,000 fixed-sample log-normalised HVG cells"})

def synthetic_data(p: str,seed: int,e: str) -> dict[str,Any]:
    import anndata as ad
    s=s12(); row=s.manifest_row(p,seed,verify=False); src=ad.read_h5ad(ROOT/row["path"],backed="r")
    days=src.obs["day"].to_numpy(int); hours=src.obs["time_hours"].to_numpy(float)
    if e=="pca20": meta=src.uns["embedding_pca"]; z=np.asarray(src.obsm["X_pca"],float); L=np.asarray(meta["linear_map"],float)
    elif e=="fa20": meta=src.uns["embedding_fa"]; z=np.asarray(src.obsm["X_fa"],float); L=np.asarray(meta["linear_map"],float)
    else:
        with np.load(s12_embed(p,seed),allow_pickle=False) as a: z=np.asarray(a[f"z_{e}"],float); L=np.asarray(a[f"linear_map_{e}"],float)
    with np.load(s12_embed(p,seed),allow_pickle=False) as a: common=np.asarray(a["linear_map_pca50"],float)
    return {"s":s,"src":src,"row":row,"days":days,"hours":hours,"z":z,"L":L,"common":common}

def run_synth(p: str,seed: int,e: str) -> dict[str,Any]:
    d=synthetic_data(p,seed,e); s,src,z,days,hours=d["s"],d["src"],d["z"],d["days"],d["hours"]
    try:
        zw,W,WI,win=whitening(z); vr,ir=s.ot_velocity(z,days,hours); vw,iw=s.ot_velocity(zw,days,hours); rr=np.flatnonzero(np.isfinite(vr).all(1)); rw=np.flatnonzero(np.isfinite(vw).all(1))
        m,fi=fit_model(zw,vw,rw); Jfull=jac(m,zw); del m
        m,fr=fit_model(z,vw@WI,rw); Jwr=jac(m,z); del m
        m,fw=fit_model(zw,vr@W,rr); Jrw=jac(m,zw); del m
        with np.load(s12_raw(p,seed,e),allow_pickle=False) as a: Jraw=np.asarray(a["J"],np.float32)
        # Truth scoring begins only here, after raw/full/partial fits have completed.
        truth=s.truth_for_scoring(src); axis=s.fate_axis(z,truth["day"],truth["fate"]); scores={}; lam={}
        for name,J,inv in (("raw_s12",Jraw,None),("whitened_full",Jfull,WI),("whitened_ot_raw_drift",Jwr,None),("raw_ot_whitened_drift",Jrw,WI)):
            scores[name],lam[name]=score_synthetic(s,J,truth,axis,inv)
        if p == "v3_transition":
            hp,hi=heldout(s,z,zw,WI,z.mean(0),days,hours,d["L"],d["common"])
        else:
            hp,hi=[],{"status":"not required: S13 held-out prediction is v3_transition and LARRY only"}
        dataset=sid(p,seed); dest=arr_path(dataset,e); ARR.mkdir(parents=True,exist_ok=True); np.savez_compressed(dest,**lam)
        return {"dataset":dataset,"preset":p,"seed":seed,"embedding":e,"input_sha256":d["row"]["sha256"],"n_cells":int(len(z)),"dimension":int(z.shape[1]),"whitening":win,"raw_ot_pairs":ir,"whitened_ot_pairs":iw,"fits":{"whitened_full":fi,"whitened_ot_raw_drift":fr,"raw_ot_whitened_drift":fw},"scores":scores,"heldout_prediction":hp,"heldout_details":hi,"array":{"path":str(dest.relative_to(ROOT)),"sha256":hash_file(dest)}}
    finally: src.file.close()

def larry_maps() -> dict[str,np.ndarray]:
    p=ARR/"larry_maps.npz"
    if not p.exists(): raise RuntimeError("run prepare-larry first")
    with np.load(p,allow_pickle=False) as a: return {k:np.asarray(a[k]) for k in a.files}
def run_larry(e: str) -> dict[str,Any]:
    s=s12(); M=larry_maps()
    with np.load(S10REAL/"larry.npz",allow_pickle=False) as a: z=np.asarray(a["z_pca" if e=="pca30" else "z_fa"],float); days=np.asarray(a["day"],float)
    L=np.asarray(M["pca_map" if e=="pca30" else "fa_map"],float); common=np.asarray(M["common_map"],float); zw,W,WI,win=whitening(z); hours=days*24
    vr,ir=s.ot_velocity(z,days,hours); vw,iw=s.ot_velocity(zw,days,hours); rr=np.flatnonzero(np.isfinite(vr).all(1)); rw=np.flatnonzero(np.isfinite(vw).all(1))
    m,fi=fit_model(zw,vw,rw); Jfull=jac(m,zw); del m
    m,fr=fit_model(z,vw@WI,rw); Jwr=jac(m,z); del m
    m,fw=fit_model(zw,vr@W,rr); Jrw=jac(m,zw); del m
    rawfile=S10REAL/f"fit_larry_{'pca' if e=='pca30' else 'fa'}_seed0.npz"
    with np.load(rawfile,allow_pickle=False) as a: raw=np.asarray(s.spectra(a["J_flow_ot"])["lam1"],np.float32)
    lam={"raw_s10":raw,"whitened_full":np.asarray(s.spectra(Jfull)["lam1"],np.float32),"whitened_ot_raw_drift":np.asarray(s.spectra(Jwr)["lam1"],np.float32),"raw_ot_whitened_drift":np.asarray(s.spectra(Jrw)["lam1"],np.float32)}
    hp,hi=heldout(s,z,zw,WI,z.mean(0),days,hours,L,common); dest=arr_path("larry",e); ARR.mkdir(parents=True,exist_ok=True); np.savez_compressed(dest,**lam)
    scores={k:{"median_lam1":float(np.median(v)),"mean_lam1":float(np.mean(v))} for k,v in lam.items()}
    return {"dataset":"larry","preset":"larry","seed":0,"embedding":e,"input_sha256":req_preflight()["larry"]["raw_sha256"],"n_cells":int(len(z)),"dimension":int(z.shape[1]),"sample_note":"S11 fixed default_rng(0) sample: 4,000 cells per day, exactly as S10 Step 5","whitening":win,"raw_ot_pairs":ir,"whitened_ot_pairs":iw,"fits":{"whitened_full":fi,"whitened_ot_raw_drift":fr,"raw_ot_whitened_drift":fw},"scores":scores,"heldout_prediction":hp,"heldout_details":hi,"array":{"path":str(dest.relative_to(ROOT)),"sha256":hash_file(dest)}}

def unit(dataset: str,e: str) -> None:
    req_preflight(); set_threads(); started,c0=time.time(),cpu()
    if dataset=="larry":
        if e not in LEMB: raise RuntimeError("LARRY requires pca30 or fa30")
        out=run_larry(e)
    else:
        m=re.fullmatch(r"(v3_(?:transition|default))_s([012])",dataset)
        if m is None or e not in EMB: raise RuntimeError("invalid synthetic dataset or embedding")
        out=run_synth(m.group(1),int(m.group(2)),e)
    chk(unit_name(dataset,e),started,c0,out)

def records() -> list[dict[str,Any]]:
    names=[unit_name(sid(p,s),e) for p,s in SYN for e in EMB]+[unit_name("larry",e) for e in LEMB]
    missing=[x for x in names if not (CK/f"{x}.json").exists()]
    if missing: raise RuntimeError("missing units: "+", ".join(missing))
    return [rjson(CK/f"{x}.json") for x in names]
def markdown_table(frame: Any) -> str:
    """Render a small report table without pandas' optional tabulate dependency."""
    headers = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            values.append(f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)

def summarize() -> None:
    import pandas as pd
    from scipy.stats import spearmanr
    req_preflight(); started,c0=time.time(),cpu(); recs=records(); arows=[]; vrows=[]; hrows=[]; data={}
    for r in recs:
        ds,e=r["dataset"],r["embedding"]
        with np.load(ROOT/r["array"]["path"],allow_pickle=False) as x: data.setdefault(ds,{})[e]={k:np.asarray(x[k]) for k in x.files}
        for variant,score in r["scores"].items():
            row={"dataset":ds,"preset":r["preset"],"seed":r["seed"],"embedding":e,"variant":variant,"n_cells":r["n_cells"],"dimension":r["dimension"],**score}; vrows.append(row)
            if ds!="larry": arows.append(row)
        for q in r["heldout_prediction"]: hrows.append({"dataset":ds,"preset":r["preset"],"seed":r["seed"],"embedding":e,**q,"fit_rule":"first sampling pair only; Euler 1 h from middle to final day"})
    auc=pd.DataFrame(arows).sort_values(["preset","seed","embedding","variant"]); variants=pd.DataFrame(vrows).sort_values(["dataset","embedding","variant"]); held=pd.DataFrame(hrows).sort_values(["dataset","embedding","method"])
    OUT.mkdir(parents=True,exist_ok=True); auc.to_csv(OUT/"auc.csv",index=False); variants.to_csv(OUT/"variants.csv",index=False); held.to_csv(OUT/"heldout_prediction.csv",index=False)
    grows=[]
    for ds,per in sorted(data.items()):
        if ds=="larry": continue
        p,s=ds.rsplit("_s",1)
        for a,b in itertools.combinations(EMB,2): grows.append({"dataset":ds,"preset":p,"seed":int(s),"embedding_left":a,"embedding_right":b,"variant":"whitened_full","spearman_lambda1":float(spearmanr(per[a]["whitened_full"],per[b]["whitened_full"]).statistic),"n_cells":len(per[a]["whitened_full"])})
    agree=pd.DataFrame(grows).sort_values(["preset","seed","embedding_left","embedding_right"]); agree.to_csv(OUT/"lambda1_agreement.csv",index=False)
    full=auc[(auc.variant=="whitened_full")&(auc.preset=="v3_transition")]; tab=full.groupby("embedding").auc_lam1.agg(["count","min","median","max"]).reindex(EMB); passing=[e for e,x in tab["min"].items() if np.isfinite(x) and x>=.85 and tab.loc[e,"count"]==3]
    gate={"rule":"AUC >= 0.85 on every v3_transition seed in at least 3 linear embeddings","passing_embeddings":passing,"n_passing_embeddings":len(passing),"pass":len(passing)>=3,"by_embedding":{str(i):{k:(int(v) if k=="count" else float(v)) for k,v in r.items()} for i,r in tab.iterrows()}}; wjson(OUT/"gate.json",gate)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    FIG.mkdir(parents=True,exist_ok=True); order=[sid(p,s) for p,s in SYN]; colors={"transition":"#2369a6","default":"#bc5a45"}
    fig,ax=plt.subplots(figsize=(9,4.3))
    for j,e in enumerate(EMB):
        raw=auc[(auc.embedding==e)&(auc.variant=="raw_s12")].set_index("dataset"); white=auc[(auc.embedding==e)&(auc.variant=="whitened_full")].set_index("dataset")
        for k,ds in enumerate(order):
            x=j+(k-2.5)*.065; c=colors["transition" if "transition" in ds else "default"]; ax.plot([x-.017,x+.017],[raw.loc[ds,"auc_lam1"],white.loc[ds,"auc_lam1"]],color=c,lw=.8); ax.scatter(x-.017,raw.loc[ds,"auc_lam1"],color=c,s=18); ax.scatter(x+.017,white.loc[ds,"auc_lam1"],color=c,marker="s",s=18)
    ax.axhline(.85,color="black",ls="--",lw=.9); ax.set_xticks(range(4),["PCA-10","PCA-20","PCA-50","FA-20"]); ax.set_ylim(0,1.03); ax.set_ylabel("fork AUC (λ₁)"); ax.set_title("S13: raw circles, fully whitened squares (blue transition; red default)"); fig.tight_layout(); fig.savefig(FIG/"auc_raw_vs_whitened.png",dpi=160); plt.close(fig)
    mat=np.eye(4)
    for i,a in enumerate(EMB):
        for j,b in enumerate(EMB):
            if i<j:
                v=float(agree[(agree.embedding_left==a)&(agree.embedding_right==b)].spearman_lambda1.median()); mat[i,j]=mat[j,i]=v
    fig,ax=plt.subplots(figsize=(5.3,4.4)); im=ax.imshow(mat,vmin=-1,vmax=1,cmap="coolwarm"); labels=["PCA-10","PCA-20","PCA-50","FA-20"]; ax.set_xticks(range(4),labels,rotation=35,ha="right"); ax.set_yticks(range(4),labels)
    for i in range(4):
        for j in range(4): ax.text(j,i,f"{mat[i,j]:.2f}",ha="center",va="center",fontsize=9)
    fig.colorbar(im,ax=ax,label="median per-cell Spearman(λ₁), six datasets"); ax.set_title("S13: whitened λ₁ agreement"); fig.tight_layout(); fig.savefig(FIG/"lambda1_agreement_matrix.png",dpi=160); plt.close(fig)
    best=held.loc[held.groupby(["dataset","embedding"]).energy_distance.idxmin(),["dataset","embedding","method","energy_distance"]]
    lines=["# orient/S13 — whitened OT flow J","","## Fixed gate",f"* **{'PASS' if gate['pass'] else 'FAIL'}** — {gate['n_passing_embeddings']} of 4 linear embeddings pass every v3_transition seed (required ≥3).","","| embedding | n | min AUC | median AUC | max AUC | all seeds pass |","|---|---:|---:|---:|---:|---|"]
    for e,r in tab.iterrows(): lines.append(f"| {e} | {int(r['count'])} | {r['min']:.4f} | {r['median']:.4f} | {r['max']:.4f} | {'yes' if e in passing else 'no'} |")
    lines += ["","## Held-out prediction (truth-free)","Flow fit: first sampling pair only; integrate middle-day cells to final day in 1-hour Euler steps. Best method for each dataset × embedding:","",markdown_table(best),"","## Scope","Synthetic units use every S09/S10 cell. LARRY uses S11’s fixed `default_rng(0)` sample of 4,000 cells per time point as amended in S13 Revision 1.","","## CPU","`ledger` adds the closed /usr/bin/time total here after all unit logs are available."]
    (OUT/"RESULT_S13.md").write_text("\n".join(lines)+"\n"); chk("summarize",started,c0,{"gate":gate,"n_auc_rows":len(auc),"n_agreement_rows":len(agree),"n_variant_rows":len(variants),"n_heldout_rows":len(held),"figures":[str((FIG/"auc_raw_vs_whitened.png").relative_to(ROOT)),str((FIG/"lambda1_agreement_matrix.png").relative_to(ROOT))]})
def ledger() -> None:
    req_preflight(); started,c0=time.time(),cpu(); rows=[]
    for p in sorted(LOG.glob("*.log")):
        t=p.read_text(errors="ignore"); tm=re.search(r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys",t); st=re.search(r"^START (\S+)",t,re.M); ex=re.search(r"^EXIT (\d+) (\S+)",t,re.M); rm=re.search(r"(\d+)\s+maximum resident set size",t)
        rows.append({"log":str(p.relative_to(ROOT)),"start_utc":st.group(1) if st else "","end_utc":ex.group(2) if ex else "","exit":int(ex.group(1)) if ex else "","wall_s":float(tm.group(1)) if tm else "","cpu_s":round(float(tm.group(2))+float(tm.group(3)),3) if tm else "","peak_rss_gb":round(int(rm.group(1))/1e9,3) if rm else ""})
    if not rows: raise RuntimeError("no closed S13 logs")
    with (OUT/"cpu_ledger.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n"); w.writeheader(); w.writerows(rows)
    total=sum(float(r["cpu_s"]) for r in rows if r["cpu_s"]!=""); result=OUT/"RESULT_S13.md"
    if result.exists(): result.write_text(result.read_text().replace("`ledger` adds the closed /usr/bin/time total here after all unit logs are available.",f"Timed unit CPU total from `cpu_ledger.csv`: **{total/3600:.3f} CPU-h** (task budget: 8.000 CPU-h)."))
    chk("ledger",started,c0,{"n_logs":len(rows),"timed_cpu_hours":total/3600,"budget_cpu_hours":8.0,"within_budget":total/3600<=8.0,"ledger":str((OUT/"cpu_ledger.csv").relative_to(ROOT))})
def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("stage",choices=("preflight","prepare-larry","unit","summarize","ledger")); p.add_argument("--dataset"); p.add_argument("--embedding"); a=p.parse_args()
    if a.stage=="preflight": preflight()
    elif a.stage=="prepare-larry": prepare_larry()
    elif a.stage=="unit":
        if not a.dataset or not a.embedding: p.error("unit requires --dataset and --embedding")
        unit(a.dataset,a.embedding)
    elif a.stage=="summarize": summarize()
    else: ledger()
if __name__=="__main__": main()
