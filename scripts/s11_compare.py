#!/usr/bin/env python3
"""S11 direct and null comparisons; never reads truth or mutates real data."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks/real_flow"
ARRAYS = OUT / "arrays"
OPS = ARRAYS / "operators"
VEL = ARRAYS / "velocities"
BLOCKS = OUT / "comparison_blocks"
CHECKPOINTS = OUT / "checkpoints"
FIG = ROOT / "benchmarks/figures/real_flow"

PAIRS = {"scnt_hema": (("labelling", "scvelo"), ("labelling", "dynamo_gene"), ("scvelo", "dynamo_gene")), "larry": (("ot", "lineage"),)}
SOURCES = {"scnt_hema": ("labelling", "scvelo", "dynamo_gene"), "larry": ("ot", "lineage")}


def stamp() -> str: return datetime.now(timezone.utc).isoformat()
def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(data, indent=2, sort_keys=True)+"\n")
def checkpoint(unit: str, started: float, **payload: Any) -> None:
    write_json(CHECKPOINTS / f"{unit}.json", {"task":"orient/S11","unit":unit,"started_utc":datetime.fromtimestamp(started,timezone.utc).isoformat(),"finished_utc":stamp(),"wall_seconds":round(time.time()-started,3),**payload})
def block_path(kind: str, dataset: str, a: str, b: str | None = None) -> Path:
    return BLOCKS / (f"{kind}_{dataset}_{a}" + (f"_{b}" if b else "") + ".csv")
def velocity(dataset: str, source: str) -> dict[str,np.ndarray]:
    p=VEL/f"{dataset}_{source}.npz"; d={k:v for k,v in np.load(p,allow_pickle=False).items()}
    return d
def op(dataset: str, source: str, estimator: str, seed: int|None=None) -> dict[str,np.ndarray]:
    suffix=f"_s{seed}" if seed is not None else ""; p=OPS/f"{dataset}_{source}_{estimator}{suffix}.npz"
    return {k:v for k,v in np.load(p,allow_pickle=False).items()}

def align(a: dict[str,np.ndarray], b: dict[str,np.ndarray]) -> tuple[np.ndarray,np.ndarray]:
    pos={str(x):i for i,x in enumerate(b["cell_id"].astype(str))}
    ia=[];ib=[]
    for i,x in enumerate(a["cell_id"].astype(str)):
        j=pos.get(str(x))
        if j is not None: ia.append(i);ib.append(j)
    if not ia: raise RuntimeError("no overlapping source/operator cells")
    return np.asarray(ia),np.asarray(ib)

def eig_features(j: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    vals, vecs=np.linalg.eig(np.asarray(j,dtype=np.float64))
    ix=np.argmax(vals.real,axis=1)
    lam=np.take_along_axis(vals.real,ix[:,None],axis=1)[:,0]
    vec=np.take_along_axis(vecs,ix[:,None,None],axis=2)[:,:,0]
    vec=vec/np.maximum(np.linalg.norm(vec,axis=1,keepdims=True),1e-12)
    return lam,vec

def metric(j1: np.ndarray,j2: np.ndarray) -> dict[str,float]:
    l1,e1=eig_features(j1);l2,e2=eig_features(j2)
    denom=.5*(np.linalg.norm(j1,axis=(1,2))+np.linalg.norm(j2,axis=(1,2)))
    rel=np.linalg.norm(j1-j2,axis=(1,2))/np.maximum(denom,1e-12)
    cosine=np.abs(np.sum(np.conj(e1)*e2,axis=1))
    rho=spearmanr(l1,l2).statistic
    return {"n_cells":int(len(l1)),"relative_frobenius_median":float(np.median(rel)),"lambda1_spearman":float(rho),"leading_eigenvector_abs_cosine_median":float(np.median(cosine))}

def cosine(v1:np.ndarray,v2:np.ndarray)->np.ndarray:
    return np.sum(v1*v2,axis=1)/np.maximum(np.linalg.norm(v1,axis=1)*np.linalg.norm(v2,axis=1),1e-12)

def measured(z:np.ndarray,v:np.ndarray):
    from scjdo2.measured_drift import MeasuredDrift
    return MeasuredDrift(30,seed=0).fit(z,v,verbose=True).jacobian_autodiff(z)
def local(z:np.ndarray,v:np.ndarray):
    from scjdo2.orient import OrientConfig
    from scjdo2.orient.base import local_jacobian,neighbours
    cfg=OrientConfig();return local_jacobian(z,v,neighbours(z,30),cfg.ridge)

def direct(dataset:str)->None:
    started=time.time(); BLOCKS.mkdir(parents=True,exist_ok=True)
    vrows=[];orows=[]
    for a,b in PAIRS[dataset]:
        va,vb=velocity(dataset,a),velocity(dataset,b);ia,ib=align(va,vb)
        cc=cosine(va["velocity"][ia],vb["velocity"][ib])
        for i,c in enumerate(cc): vrows.append({"dataset":dataset,"source_a":a,"source_b":b,"cell_id":str(va["cell_id"][ia[i]]),"velocity_cosine":float(c)})
        ja,jb=op(dataset,a,"measured",0),op(dataset,b,"measured",0);oa,ob=align(ja,jb)
        orows.append({"dataset":dataset,"comparison_kind":"independent_sources","left":a,"right":b,"left_estimator":"measured_s0","right_estimator":"measured_s0",**metric(ja["jacobian"][oa],jb["jacobian"][ob])})
    for source in SOURCES[dataset]:
        base=op(dataset,source,"measured",0)
        for estimator in ("local","dynamo"):
            other=op(dataset,source,estimator);ia,ib=align(base,other)
            orows.append({"dataset":dataset,"comparison_kind":"estimators","left":source,"right":source,"left_estimator":"measured_s0","right_estimator":estimator,**metric(base["jacobian"][ia],other["jacobian"][ib])})
        for seed in (1,2):
            other=op(dataset,source,"measured",seed);ia,ib=align(base,other)
            orows.append({"dataset":dataset,"comparison_kind":"seed_variability","left":source,"right":source,"left_estimator":"measured_s0","right_estimator":f"measured_s{seed}",**metric(base["jacobian"][ia],other["jacobian"][ib])})
    pd.DataFrame(vrows).to_csv(BLOCKS/f"velocity_direct_{dataset}.csv",index=False)
    pd.DataFrame(orows).to_csv(BLOCKS/f"operator_direct_{dataset}.csv",index=False)
    checkpoint(f"comparison_direct_{dataset}",started,dataset=dataset,n_velocity_rows=len(vrows),n_operator_rows=len(orows))

def null_source(dataset:str,a:str,b:str)->None:
    started=time.time();va,vb=velocity(dataset,a),velocity(dataset,b);ia,ib=align(va,vb);z=va["z"][ia];xa=va["velocity"][ia];xb=vb["velocity"][ib];rows=[]
    for seed in range(5):
        rng=np.random.default_rng(seed);ja=measured(z,xa[rng.permutation(len(xa))]);jb=measured(z,xb[rng.permutation(len(xb))]);rows.append({"dataset":dataset,"null_kind":"independent_sources","left":a,"right":b,"seed":seed,**metric(ja,jb)})
    BLOCKS.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(block_path("null_source",dataset,a,b),index=False);checkpoint(f"null_source_{dataset}_{a}_{b}",started,dataset=dataset,left=a,right=b,n_seeds=5)

def null_local(dataset:str,source:str)->None:
    started=time.time();d=velocity(dataset,source);z=d["z"];v=d["velocity"];rows=[]
    for seed in range(5):
        perm=v[np.random.default_rng(seed).permutation(len(v))];rows.append({"dataset":dataset,"null_kind":"estimator_local","left":source,"right":source,"seed":seed,**metric(measured(z,perm),local(z,perm))})
    BLOCKS.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(block_path("null_local",dataset,source),index=False);checkpoint(f"null_local_{dataset}_{source}",started,dataset=dataset,source=source,n_seeds=5)

def null_dynamo_prepare(dataset:str,source:str)->None:
    started=time.time();d=velocity(dataset,source);z=d["z"];v=d["velocity"];js=[];vs=[]
    for seed in range(5):
        perm=v[np.random.default_rng(seed).permutation(len(v))];vs.append(perm.astype(np.float32));js.append(measured(z,perm).astype(np.float32))
    p=ARRAYS/"null_dynamo";p.mkdir(parents=True,exist_ok=True);np.savez_compressed(p/f"{dataset}_{source}.npz",z=z.astype(np.float32),velocity=np.stack(vs),measured=np.stack(js))
    checkpoint(f"null_dynamo_prepare_{dataset}_{source}",started,dataset=dataset,source=source,n_seeds=5)

def null_dynamo_run(dataset:str,source:str)->None:
    started=time.time();import anndata as ad;import dynamo
    x=np.load(ARRAYS/"null_dynamo"/f"{dataset}_{source}.npz",allow_pickle=False);z=x["z"];rows=[]
    for seed in range(5):
        a=ad.AnnData(X=np.zeros((len(z),1),dtype=np.float32));a.obsm["X_pca"]=z;a.obsm["velocity_pca"]=x["velocity"][seed]
        dynamo.vf.VectorField(a,basis="pca",normalize=False,grid_velocity=False,cores=1);dynamo.vf.jacobian(a,basis="pca",method="analytical",store_in_adata=True)
        jd=np.transpose(np.asarray(a.uns["jacobian_pca"]["jacobian"]),(2,0,1));rows.append({"dataset":dataset,"null_kind":"estimator_dynamo","left":source,"right":source,"seed":seed,**metric(x["measured"][seed],jd)})
    BLOCKS.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(block_path("null_dynamo",dataset,source),index=False);checkpoint(f"null_dynamo_run_{dataset}_{source}",started,dataset=dataset,source=source,n_seeds=5,dynamo_version=getattr(dynamo,"__version__","1.5.3"))

def final()->None:
    started=time.time();import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    vr=pd.concat([pd.read_csv(p) for p in sorted(BLOCKS.glob("velocity_direct_*.csv"))],ignore_index=True)
    od=pd.concat([pd.read_csv(p) for p in sorted(BLOCKS.glob("operator_direct_*.csv"))],ignore_index=True)
    nr=pd.concat([pd.read_csv(p) for p in sorted(BLOCKS.glob("null_*.csv"))],ignore_index=True)
    vr.to_csv(OUT/"velocity_agreement.csv",index=False);od.to_csv(OUT/"operator_agreement.csv",index=False);nr.to_csv(OUT/"null.csv",index=False)
    # Score-versus-flow is reported at per-cell resolution, with the global lambda correlation retained per source.
    score_rows=[];br=[]
    all_operators=[]
    for dataset,sources in SOURCES.items():
        score=op(dataset,"score","score")
        all_operators.append((dataset,"score","score",score))
        for source in sources:
            flow=op(dataset,source,"measured",0);ia,ib=align(score,flow);ls,es=eig_features(score["jacobian"][ia]);lf,ef=eig_features(flow["jacobian"][ib])
            evcos=np.abs(np.sum(np.conj(es)*ef,axis=1));rho=float(spearmanr(ls,lf).statistic)
            for i in range(len(ia)):
                score_rows.append({"dataset":dataset,"source":source,"cell_id":str(score["cell_id"][ia[i]]),"cell_type":str(score["cell_type"][ia[i]]),"lambda1_score":float(ls[i]),"lambda1_flow":float(lf[i]),"leading_eigenvector_abs_cosine":float(evcos[i]),"lambda1_spearman_all_cells":rho})
        for source in sources:
            for est,seed in (("measured",0),("local",None),("dynamo",None)):
                all_operators.append((dataset,source,est,op(dataset,source,est,seed)))
    pd.DataFrame(score_rows).to_csv(OUT/"score_vs_flow.csv",index=False)
    for dataset,source,est,o in all_operators:
        lam,_=eig_features(o["jacobian"]);df=pd.DataFrame({"cell_type":o["cell_type"].astype(str),"lambda1":lam})
        for ct,g in df.groupby("cell_type",sort=True): br.append({"dataset":dataset,"source":source,"operator":est,"cell_type":ct,"n_cells":len(g),"lambda1_median":float(g.lambda1.median()),"lambda1_fraction_positive":float((g.lambda1>0).mean())})
    pd.DataFrame(br).to_csv(OUT/"by_cell_type.csv",index=False)
    criteria=[]
    for _,r in od.iterrows():
        if r.comparison_kind == "seed_variability":
            continue
        nk="independent_sources" if r.comparison_kind=="independent_sources" else ("estimator_"+str(r.right_estimator))
        q=nr[(nr.null_kind==nk)&(nr.left==r.left)&(nr.right==r.right)]
        if len(q):
            criteria.append({"comparison_kind":r.comparison_kind,"left":r.left,"right":r.right,"right_estimator":r.right_estimator,"lambda1_spearman":r.lambda1_spearman,"null_lambda1_spearman_max":q.lambda1_spearman.max(),"eigenvector_abs_cosine":r.leading_eigenvector_abs_cosine_median,"null_eigenvector_abs_cosine_max":q.leading_eigenvector_abs_cosine_median.max(),"pass":bool(r.lambda1_spearman>q.lambda1_spearman.max() and r.leading_eigenvector_abs_cosine_median>q.leading_eigenvector_abs_cosine_median.max())})
    cdf=pd.DataFrame(criteria)
    FIG.mkdir(parents=True,exist_ok=True)
    for fp in FIG.glob("*.png"): fp.unlink()
    if len(cdf):
        fig,axes=plt.subplots(1,2,figsize=(15,5));x=np.arange(len(cdf));labels=[f"{a}/{b}:{e}" for a,b,e in zip(cdf.left,cdf.right,cdf.right_estimator)]
        axes[0].scatter(x,cdf.lambda1_spearman,label="direct");axes[0].scatter(x,cdf.null_lambda1_spearman_max,label="max null",marker="x");axes[0].set_ylabel("λ1 Spearman")
        axes[1].scatter(x,cdf.eigenvector_abs_cosine,label="direct");axes[1].scatter(x,cdf.null_eigenvector_abs_cosine_max,label="max null",marker="x");axes[1].set_ylabel("median |cos(leading eigenvector)|")
        for ax in axes: ax.set_xticks(x,labels,rotation=55,ha="right");ax.legend()
        fig.tight_layout();fig.savefig(FIG/"agreement_vs_null.png",dpi=160);plt.close(fig)
    for dataset in SOURCES:
        df=pd.DataFrame([r for r in br if r["dataset"]==dataset])
        if len(df):
            fig,ax=plt.subplots(figsize=(12,5));
            groups=list(df.groupby(["source","operator"],sort=True))
            for i,(name,g) in enumerate(groups): ax.scatter(np.full(len(g),i),g.lambda1_median,s=24)
            ax.set_xticks(range(len(groups)),["/".join(x) for x,_ in groups],rotation=45,ha="right");ax.set_ylabel("median λ1 by cell type");fig.tight_layout();fig.savefig(FIG/f"lambda1_by_cell_type_{dataset}.png",dpi=160);plt.close(fig)
    for dataset,source,est,o in all_operators:
        lam,_=eig_features(o["jacobian"]);fig,ax=plt.subplots(figsize=(6,5));im=ax.scatter(o["z"][:,0],o["z"][:,1],c=lam,s=4,cmap="coolwarm");fig.colorbar(im,ax=ax,label="leading real λ1");ax.set_xlabel("PC1");ax.set_ylabel("PC2");ax.set_title(f"{dataset}: {source}/{est}");fig.tight_layout();fig.savefig(FIG/f"embedding_lambda1_{dataset}_{source}_{est}.png",dpi=160);plt.close(fig)
    def markdown_table(frame: pd.DataFrame) -> str:
        if frame.empty: return "No criterion rows."
        cols=list(frame.columns)
        def cell(value: Any) -> str: return f"{value:.6g}" if isinstance(value,float) else str(value)
        rows=["| "+" | ".join(cols)+" |","| "+" | ".join(["---"]*len(cols))+" |"]
        rows += ["| "+" | ".join(cell(row[c]) for c in cols)+" |" for _,row in frame.iterrows()]
        return "\n".join(rows)
    score_summary=pd.DataFrame(score_rows).groupby(["dataset","source"],sort=True).agg(lambda1_spearman_all_cells=("lambda1_spearman_all_cells","first"),leading_eigenvector_abs_cosine_median=("leading_eigenvector_abs_cosine","median"),n_cells=("cell_id","size")).reset_index()
    lines=["# S11 real-flow results","","## What ran","- Read-only scNT-seq and LARRY sources in their fixed common PCA-30 embeddings.","- Measured flow (seeds 0–2), local ridge, SparseVFC analytic, and score operators; 5 deterministic permutation nulls per comparison block.","","## Criteria (verbatim)","- Reproducible: independent-source measured-flow λ1 Spearman and median |cos| exceed the largest of five nulls.","- Estimator-robust: measured vs local and measured vs Dynamo agree above their matching nulls likewise.","","## Criterion table","",markdown_table(cdf),"","## Score-vs-flow summary","",markdown_table(score_summary),"","## Notes","- Stored processed Dynamo VecFld_pca is report-only in its own PCA basis and is never cross-basis compared."]
    (OUT/"RESULT_S11.md").write_text("\n".join(lines)+"\n")
    checkpoint("comparison_finalize",started,n_velocity_rows=len(vr),n_operator_rows=len(od),n_null_rows=len(nr),n_score_rows=len(score_rows),n_criteria=len(cdf))

def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--action",choices=("direct","null_source","null_local","null_dynamo_prepare","null_dynamo_run","finalize"),required=True);p.add_argument("--dataset",choices=("scnt_hema","larry"));p.add_argument("--source-a");p.add_argument("--source-b");p.add_argument("--source");a=p.parse_args()
    if a.action=="direct":
        if not a.dataset: p.error("direct requires --dataset")
        direct(a.dataset)
    elif a.action=="null_source":
        if not (a.dataset and a.source_a and a.source_b): p.error("null_source requires dataset/source-a/source-b")
        null_source(a.dataset,a.source_a,a.source_b)
    elif a.action=="null_local":
        if not (a.dataset and a.source): p.error("null_local requires dataset/source")
        null_local(a.dataset,a.source)
    elif a.action=="null_dynamo_prepare":
        if not (a.dataset and a.source): p.error("null_dynamo_prepare requires dataset/source")
        null_dynamo_prepare(a.dataset,a.source)
    elif a.action=="null_dynamo_run":
        if not (a.dataset and a.source): p.error("null_dynamo_run requires dataset/source")
        null_dynamo_run(a.dataset,a.source)
    else: final()
if __name__=="__main__": main()
