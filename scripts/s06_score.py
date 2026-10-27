#!/usr/bin/env python3
"""Score orient/S06 v2 outputs without ever exposing simulator truth to orient()."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from scjdo2.data.synthetic_multimodal import forward_developmental_truth
from scjdo2.orient import OrientConfig
from scjdo2.orient.base import get_operator, operator_hash
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from orient_end_to_end_s03 import call_orient, chord_truth, expected_flow_truth, score_group, source_addition

ROOT = SCRIPTS_DIR.parent
MANIFEST = ROOT / "benchmarks/synthetic_multimodal/MANIFEST_v2.csv"
FIT_MANIFEST = ROOT / "benchmarks/synthetic_multimodal/s06/fit_manifest.json"
USER_MANIFEST = ROOT / "benchmarks/synthetic_multimodal/s06/user_run_manifest.json"
FIT_DIR = ROOT / "benchmarks/synthetic_multimodal/data/s06"
OUT = ROOT / "benchmarks/synthetic_multimodal/s06"
FIG = ROOT / "benchmarks/figures/orient/s06"
PRESETS = ("v2_default", "v2_stress", "v2_batch", "v2_stationary_cycle")
NON_STATIONARY = ("v2_default", "v2_stress", "v2_batch")
METHODS = ("pca", "fa")
COMBINATIONS = {
    "flow_labelling": (["labelling"], "flow", "labelling"),
    "flow_velocity": (["velocity"], "flow", "velocity"),
    "flow_combined": (["labelling", "velocity"], "flow", "labelling+velocity"),
    "dev_lineage": (["lineage"], "dev", "lineage"),
    "dev_pseudotime": (["pseudotime"], "dev", "pseudotime"),
    "dev_combined": (["lineage", "pseudotime"], "dev", "lineage+pseudotime"),
    "all_four": (["labelling", "velocity", "lineage", "pseudotime"], "both", "all_four"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def utc(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray): return json_safe(value.tolist())
    if isinstance(value, np.generic): return value.item()
    return value


def input_rows() -> list[dict[str, str]]:
    with MANIFEST.open(newline="") as f: rows = list(csv.DictReader(f))
    expected = {(p, s) for p in PRESETS for s in range(3)}
    if {(r["preset"], int(r["seed"])) for r in rows} != expected: raise RuntimeError("unexpected v2 manifest")
    for r in rows:
        p = ROOT / r["path"]
        if not p.is_file() or sha256(p) != r["sha256"]: raise RuntimeError(f"input hash mismatch: {p}")
    return sorted(rows, key=lambda r: (PRESETS.index(r["preset"]), int(r["seed"])))


def read_json(path: Path) -> dict[str, Any]: return json.loads(path.read_text())


def manifests() -> tuple[dict[str, Any], dict[str, Any]]:
    fit, user = read_json(FIT_MANIFEST), read_json(USER_MANIFEST)
    if len(fit.get("files", [])) != 24: raise RuntimeError("incomplete fit manifest")
    if len(user.get("scvelo_files", [])) != 9 or len(user.get("palantir_runs", [])) != 180: raise RuntimeError("incomplete user manifest")
    return fit, user


def observed_only(score: ad.AnnData) -> ad.AnnData:
    """Return an orient input with all task-listed simulator truth removed."""
    data = score.copy()
    for key in list(data.layers):
        if key.endswith("_true") or key == "ds_dt_true": del data.layers[key]
    for key in ("z_true",):
        if key in data.obsm: del data.obsm[key]
    for key in ("clone_true", "fate_true"):
        if key in data.obs: del data.obs[key]
    for key in ("gene_kind", "theta", "delta", "fate_sign", "nb_theta"):
        if key in data.var: del data.var[key]
    return data


def fit_record(records: list[dict[str, Any]], preset: str, seed: int, method: str) -> dict[str, Any]:
    unit = f"{preset}_s{seed}_{method}"
    found = [r for r in records if r["unit"] == unit]
    if len(found) != 1: raise RuntimeError(f"missing fit record {unit}")
    return found[0]


def scvelo_record(records: list[dict[str, Any]], preset: str, seed: int) -> dict[str, Any]:
    unit = f"{preset}_s{seed}"
    found = [r for r in records if r["unit"] == unit]
    if len(found) != 1: raise RuntimeError(f"missing scVelo record {unit}")
    return found[0]


def read_fit(row: dict[str, str], method: str, fit_records: list[dict[str, Any]], scv_records: list[dict[str, Any]]) -> tuple[ad.AnnData, ad.AnnData, dict[str, Any]]:
    preset, seed = row["preset"], int(row["seed"])
    fr = fit_record(fit_records, preset, seed, method)
    fp = ROOT / fr["path"]
    if not fp.is_file() or sha256(fp) != fr["sha256"] or fr["input_sha256"] != row["sha256"]: raise RuntimeError("fit provenance failed")
    score = ad.read_h5ad(fp)
    scv = scvelo_record(scv_records, preset, seed)
    sp = ROOT / scv["path"]
    if not sp.is_file() or sha256(sp) != scv["sha256"] or scv["input_sha256"] != row["sha256"]: raise RuntimeError("scVelo provenance failed")
    user = ad.read_h5ad(sp)
    if not score.obs_names.equals(user.obs_names): raise RuntimeError("fit/scVelo cell-order mismatch")
    data = observed_only(score)
    data.obsm[f"velocity_{method}"] = np.asarray(user.obsm[f"velocity_{method}"], dtype=np.float32)
    return data, score, scv


def read_pseudotime(record: dict[str, Any], names: pd.Index) -> tuple[np.ndarray, dict[str, Any]]:
    path = ROOT / record["path"]
    if not path.is_file() or sha256(path) != record["sha256"]: raise RuntimeError(f"pseudotime hash mismatch: {record['unit']}")
    a = ad.read_h5ad(path)
    if not a.obs_names.equals(names): raise RuntimeError("pseudotime cell order mismatch")
    pseudo = a.obs["palantir_pseudotime"].to_numpy(dtype=float)
    if not np.isfinite(pseudo).all(): raise RuntimeError("nonfinite pseudotime")
    return pseudo, {"actual_start_cell": str(a.uns["palantir_start_cell"]), "given_cell": str(a.uns["palantir_given_cell"]),
                    "mode": str(a.uns["palantir_mode"]), "root_group": str(a.uns["palantir_root_group"])}


def configs(method: str, labelling_variant: str) -> dict[str, OrientConfig]:
    labelling_key = "velocity_labelling_raw" if labelling_variant == "raw" else "velocity_labelling"
    base = {"labelling_key": labelling_key}
    return {"on": OrientConfig(**base), "off": OrientConfig(**base, reliability_min={}),
            "lineage_pseudotime_check": OrientConfig(**base, reliability_min={"velocity": 0.6, "pseudotime": 0.0})}


def reliability_text(outcome: dict[str, Any]) -> str:
    return json.dumps(json_safe(outcome.get("meta", {}).get("reliability", {})), sort_keys=True)


def base(row: dict[str, str], method: str, pt: dict[str, Any], reliability: str, variant: str) -> dict[str, Any]:
    return {"preset": row["preset"], "seed": int(row["seed"]), "method": method, "root_group": pt["root_group"], "palantir_mode": pt["mode"],
            "root_index": int(pt["root_index"]), "given_cell": pt["given_cell"], "actual_start_cell": pt["actual_start_cell"],
            "actual_start_day": int(pt["actual_start_day"]), "given_differs_from_actual": bool(pt["given_differs_from_actual"]),
            "reliability_mode": reliability, "labelling_variant": variant}


def score_outcome(outcome: dict[str, Any], requested_group: str, flow_truth: np.ndarray, dev_truth: np.ndarray, days: np.ndarray, common: dict[str, Any], combo: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]]=[]; by: list[dict[str, Any]]=[]
    groups=("flow","dev") if requested_group=="both" else (requested_group,)
    for group in groups:
        metric, per_day=score_group(outcome,group,flow_truth if group=="flow" else dev_truth,days)
        rows.append({"row_type":"combination",**common,"group":group,"combination":combo,"reliability":reliability_text(outcome),**metric})
        by += [{"row_type":"combination",**common,"group":group,"combination":combo,"day":x["day"],**{k:v for k,v in x.items() if k!="day"}} for x in per_day]
    return rows,by


def forward_and_chord(score: ad.AnnData, method: str) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    # Scoring-only copy: no object passed to orient contains these values.
    scoring = score.copy()
    forward=forward_developmental_truth(scoring,method,horizon_hours=24.0,n_paths=8,seed=0)
    scoring.obsm[f"dev_truth_forward_{method}"]=forward
    return expected_flow_truth(scoring,method),forward,chord_truth(scoring,method)


def j_rows(score: ad.AnnData, row: dict[str,str], method: str) -> list[dict[str,Any]]:
    z=np.asarray(score.obsm["z_true"],dtype=float); j=np.asarray(score.uns["scjdo2"]["J"],dtype=float)
    lead=np.linalg.eigvalsh(j)[:,-1]
    mu=z[:,1]-1.0
    saddle=(mu>=0.1)&(np.abs(z[:,0])<0.3*np.sqrt(np.maximum(mu,0)))
    committed=(mu>=0.3)&(np.abs(np.abs(z[:,0])-np.sqrt(np.maximum(mu,0)))<0.3*np.sqrt(np.maximum(mu,0)))
    progenitor=z[:,1]<0.8
    day=score.obs["day"].to_numpy(dtype=int); fate=score.obs["fate_true"].astype(str).to_numpy()
    positive=np.asarray(score.obsm[f"X_{method}"],float)[(day==4)&(fate=="positive")].mean(axis=0)
    negative=np.asarray(score.obsm[f"X_{method}"],float)[(day==4)&(fate=="negative")].mean(axis=0)
    axis=positive-negative; axis/=max(np.linalg.norm(axis),1e-12)
    eigvec=np.linalg.eigh(j)[1][:,:,-1]
    align=np.abs(eigvec[saddle]@axis)
    auc=float(roc_auc_score(np.r_[np.ones(saddle.sum()),np.zeros(committed.sum())],np.r_[lead[saddle],lead[committed]])) if saddle.any() and committed.any() else float("nan")
    out=[]
    for label,mask in (("saddle",saddle),("committed",committed),("progenitor",progenitor)):
        out.append({"preset":row["preset"],"seed":int(row["seed"]),"method":method,"truth_class":label,"n":int(mask.sum()),"median_lead":float(np.median(lead[mask])) if mask.any() else float("nan"),"fraction_lead_positive":float(np.mean(lead[mask]>0)) if mask.any() else float("nan"),"auc_saddle_vs_committed":auc,"median_abs_cosine_fate_axis_at_saddle":float(np.median(align)) if align.size else float("nan")})
    return out


def run_unit(row: dict[str,str], method: str, fit_records: list[dict[str,Any]], user: dict[str,Any]) -> tuple[list[dict[str,Any]],list[dict[str,Any]],list[dict[str,Any]],list[dict[str,Any]]]:
    data,score,scv=read_fit(row,method,fit_records,user["scvelo_files"])
    days=data.obs["day"].to_numpy(dtype=int)
    runs=[r for r in user["palantir_runs"] if r["preset"]==row["preset"] and int(r["seed"])==int(row["seed"])]
    if len(runs)!=20: raise RuntimeError(f"expected 20 runs for {row['preset']} s{row['seed']}")
    results=[];byday=[]; diagnostics=[];ptledger=[]
    truths: tuple[np.ndarray,np.ndarray,np.ndarray]|None=None
    for r in sorted(runs,key=lambda x:x["unit"]):
        pseudo,meta=read_pseudotime(r,data.obs_names)
        d=data.copy(); d.obs["palantir_pseudotime"]=pseudo
        # Saved Palantir actual start is an observed Step 2 output, not simulator truth.
        d.uns["palantir_start_cell"] = meta["actual_start_cell"]
        # Always perform the dedicated pseudotime guard call and record its disposition.
        pt_spearman=float(spearmanr(pseudo,days).statistic)
        guard_outcome=None; guard_error=None
        try:
            guard_outcome=call_orient(d,["pseudotime"],method,configs(method,"raw")["on"],[])
        except ValueError as exc:
            guard_error=str(exc)
        common_pt={**r,**meta,"palantir_mode":meta["mode"],"method":method,"spearman_pseudotime_day":pt_spearman,"guard_status":"refused" if guard_error else "accepted","guard_error":guard_error}
        if guard_outcome is not None:
            if truths is None: truths=forward_and_chord(score,method)
            _,forward,_=truths
            metric,_=score_group(guard_outcome,"dev",forward,days)
            common_pt["forward_agreement"] = metric["sign_agreement"]
        else: common_pt["forward_agreement"]=float("nan")
        ptledger.append(common_pt)
        # Combinations not using pseudotime run regardless of guard. Pseudotime combinations require acceptance.
        allowed=["flow_labelling","flow_velocity","flow_combined","dev_lineage"]
        if guard_outcome is not None: allowed += ["dev_pseudotime","dev_combined","all_four"]
        outcomes: dict[tuple[str,str,str],dict[str,Any]]={}
        for variant in ("raw","smoothed"):
            cfgs=configs(method,variant)
            for reliability in ("on","off"):
                for name in allowed:
                    outcome=call_orient(d,COMBINATIONS[name][0],method,cfgs[reliability],[])
                    outcomes[(variant,reliability,name)]=outcome
                    if truths is None: truths=forward_and_chord(score,method)
                    flow,forward,chord=truths
                    common=base(row,method,common_pt,reliability,variant)
                    got,days_rows=score_outcome(outcome,COMBINATIONS[name][1],flow,forward,days,common,COMBINATIONS[name][2])
                    results += got; byday += days_rows
            # Additional task-prescribed lineage/pseudotime reliability check.
            if guard_outcome is not None:
                name="dev_combined"; outcome=call_orient(d,COMBINATIONS[name][0],method,cfgs["lineage_pseudotime_check"],[])
                outcomes[(variant,"lineage_pseudotime_check",name)]=outcome
                if truths is None: truths=forward_and_chord(score,method)
                flow,forward,chord=truths; common=base(row,method,common_pt,"lineage_pseudotime_check",variant)
                got,days_rows=score_outcome(outcome,"dev",flow,forward,days,common,"lineage+pseudotime")
                results += got; byday += days_rows
        # Add coverage/ranking and exact independence diagnostics where sources were accepted.
        for variant in ("raw","smoothed"):
            for reliability in ("on","off"):
                if truths is None: truths=forward_and_chord(score,method)
                flow,forward,chord=truths; common=base(row,method,common_pt,reliability,variant)
                fcombo=outcomes[(variant,reliability,"flow_combined")]; flab=outcomes[(variant,reliability,"flow_labelling")]
                diagnostics.append({"row_type":"source_addition",**common,"group":"flow","combination":"labelling+velocity",**source_addition(fcombo,flab,"flow","labelling","velocity",flow)})
                if guard_outcome is not None:
                    dcombo=outcomes[(variant,reliability,"dev_combined")]; dlin=outcomes[(variant,reliability,"dev_lineage")]
                    diagnostics.append({"row_type":"source_addition",**common,"group":"dev","combination":"lineage+pseudotime",**source_addition(dcombo,dlin,"dev","lineage","pseudotime",forward)})
                    allfour=outcomes[(variant,reliability,"all_four")]
                    diagnostics.append({"row_type":"independence",**common,"group":"both","combination":"all_four","flow_identical":bool(np.array_equal(fcombo["flow"]["sign"],allfour["flow"]["sign"]) and np.array_equal(fcombo["flow"]["source"],allfour["flow"]["source"])),"dev_identical":bool(np.array_equal(dcombo["dev"]["sign"],allfour["dev"]["sign"]) and np.array_equal(dcombo["dev"]["source"],allfour["dev"]["source"])),"hash_unchanged":bool(allfour["meta"]["hash_unchanged"])})
                    # Required flow-vs-development disagreement by day from the all-four call's written observed output.
                    if "flow_vs_dev_disagree" in d.obs:
                        for day in sorted(np.unique(days)):
                            diagnostics.append({"row_type":"flow_vs_dev_by_day",**common,"group":"both","combination":"all_four","day":int(day),"disagreement_fraction":float(d.obs.loc[days==day,"flow_vs_dev_disagree"].mean())})
    return results,byday,diagnostics,ptledger


def flags(row:dict[str,str],method:str,fit_records:list[dict[str,Any]])->dict[str,Any]:
    data,score,_=read_fit(row,method,fit_records,[]) if False else (None,None,None)
    fr=fit_record(fit_records,row["preset"],int(row["seed"]),method); score=ad.read_h5ad(ROOT/fr["path"]); data=observed_only(score)
    # Primary raw labelling is the frozen flags readout.
    outcome=call_orient(data,["labelling"],method,configs(method,"raw")["on"],[])
    eq=np.asarray(outcome["equilibrium_flag"],str)
    return {"row_type":"flags_only","preset":row["preset"],"seed":int(row["seed"]),"method":method,"fraction_non_equilibrium":float(np.mean(eq=="non_equilibrium")),"fraction_equilibrium":float(np.mean(eq=="equilibrium")),"fraction_not_identified":float(np.mean(eq=="not_identified")),"hash_unchanged":bool(outcome["meta"]["hash_unchanged"])}


def summary(frame: pd.DataFrame)->pd.DataFrame:
    x=frame[frame.row_type.eq("combination")].copy()
    if x.empty:return x
    g=x.groupby(["preset","method","root_group","palantir_mode","reliability_mode","labelling_variant","group","combination"],dropna=False)
    return g.agg(n=("seed","size"),agreement_mean=("sign_agreement","mean"),agreement_min=("sign_agreement","min"),oriented_fraction_mean=("oriented_fraction","mean"),oriented_fraction_min=("oriented_fraction","min")).reset_index()


def gate_values(results:pd.DataFrame,diag:pd.DataFrame,pt:pd.DataFrame,j:pd.DataFrame,flags_frame:pd.DataFrame)->dict[str,Any]:
    sel=results[(results.preset=="v2_default")&(results.method=="pca")&(results.labelling_variant=="raw")&(results.reliability_mode=="on")&(results.root_group=="day1")&(results.palantir_mode=="default")]
    def vals(group,combo,col): return sel[(sel.group==group)&(sel.combination==combo)][col].to_numpy(float)
    lab_a,lab_o=vals("flow","labelling","sign_agreement"),vals("flow","labelling","oriented_fraction")
    comb_a,comb_o=vals("flow","labelling+velocity","sign_agreement"),vals("flow","labelling+velocity","oriented_fraction")
    lin_a=vals("dev","lineage","sign_agreement")
    pt_a=vals("dev","pseudotime","sign_agreement")
    dev_o=vals("dev","lineage+pseudotime","oriented_fraction")
    ind=diag[(diag.row_type=="independence")&(diag.preset=="v2_default")&(diag.method=="pca")&(diag.labelling_variant=="raw")&(diag.reliability_mode=="on")&(diag.root_group=="day1")&(diag.palantir_mode=="default")]
    neg=pt[pt.root_group.eq("day4")]; day1=pt[(pt.root_group=="day1")&(pt.palantir_mode=="default")]
    accepted=neg[neg.guard_status.eq("accepted")]
    jsel=j[(j.preset=="v2_default")&(j.truth_class=="saddle")]
    stationary=flags_frame[flags_frame.preset.eq("v2_stationary_cycle")]
    return {
      "gate_flow_labelling":{"agreement_min":float(np.nanmin(lab_a)),"oriented_min":float(np.nanmin(lab_o)),"pass":bool(np.all(lab_a>=.85) and np.all(lab_o>=.3))},
      "gate_flow_check_guard":{"agreement_margin_min":float(np.nanmin(comb_a-lab_a)),"oriented_margin_min":float(np.nanmin(comb_o-lab_o)),"pass":bool(np.all(comb_a>=lab_a-.03) and np.all(comb_o>=lab_o))},
      "gate_dev_lineage":{"agreement_min":float(np.nanmin(lin_a)),"pass":bool(np.all(lin_a>=.90))},
      "gate_dev_pseudotime":{"median":float(np.nanmedian(pt_a)),"minimum":float(np.nanmin(pt_a)),"pass":bool(np.nanmedian(pt_a)>=.85 and np.nanmin(pt_a)>=.80)},
      "gate_dev_coverage":{"oriented_min":float(np.nanmin(dev_o)),"pass":bool(np.all(dev_o>=.80))},
      "gate_independence_hash":{"independence_all":bool((ind.flow_identical & ind.dev_identical).all()) if len(ind) else False,"hash_all":bool(results.hash_unchanged.dropna().all()),"pass":bool(len(ind) and (ind.flow_identical & ind.dev_identical).all() and results.hash_unchanged.dropna().all())},
      "gate_flags_stationary":{"minimum_non_equilibrium":float(stationary.fraction_non_equilibrium.min()),"pass":bool((stationary.fraction_non_equilibrium>=.8).all())},
      "gate_j_auc":{"minimum":float(jsel.auc_saddle_vs_committed.min()),"pass":bool((jsel.auc_saddle_vs_committed>=.75).all())},
      "gate_negative_control": {
          "n_day4": int(len(neg)), "n_day1_default": int(len(day1)),
          "accepted_day4_n": int(len(accepted)),
          "day4_accepted_bad_start_n": int((~accepted.actual_start_day.isin([1, 2])).sum()),
          "day4_accepted_low_forward_n": int((accepted.forward_agreement < 0.5).sum()),
          "day1_default_refused_n": int((day1.guard_status != "accepted").sum()),
          "pass": bool(
              ((~accepted.actual_start_day.isin([1, 2])).sum() == 0)
              and ((accepted.forward_agreement < 0.5).sum() == 0)
              and bool((day1.guard_status == "accepted").all())
          ),
      },
    }


def figures(results:pd.DataFrame,summary_frame:pd.DataFrame,j:pd.DataFrame)->None:
    FIG.mkdir(parents=True,exist_ok=True)
    primary=results[(results.row_type=="combination")&(results.labelling_variant=="raw")&(results.reliability_mode=="on")&(results.root_group=="day1")&(results.palantir_mode=="default")]
    fig,ax=plt.subplots(figsize=(12,6))
    for i,((preset,combo,group),d) in enumerate(primary.groupby(["preset","combination","group"])):
        ax.scatter(np.full(len(d),i),d.sign_agreement,s=15,alpha=.45)
        ax.plot([i-.25,i+.25],[d.sign_agreement.median()]*2,lw=3)
    labels=[f"{p}\n{c}:{g}" for p,c,g in primary.groupby(["preset","combination","group"]).groups]
    ax.set_xticks(range(len(labels)),labels,rotation=60,ha="right");ax.set_ylabel("Sign agreement");ax.set_ylim(0,1.02);ax.set_title("S06 agreement by combination and preset")
    fig.tight_layout();fig.savefig(FIG/"agreement_by_combination_preset.png",dpi=180);plt.close(fig)
    x=summary_frame[(summary_frame.labelling_variant=="raw")&(summary_frame.root_group=="day1")&(summary_frame.palantir_mode=="default")&(summary_frame.combination=="labelling+velocity")&(summary_frame.group=="flow")]
    fig,ax=plt.subplots(figsize=(8,5));
    for i,mode in enumerate(("on","off")):
        y=x[x.reliability_mode==mode].agreement_mean.to_numpy(float);ax.bar(np.arange(len(y))+i*.38,y,.36,label=f"check {mode}")
    ax.set_xticks(np.arange(len(x.preset.unique()))+.19,sorted(x.preset.unique()));ax.set_ylim(0,1.02);ax.set_ylabel("Mean agreement");ax.set_title("Reliability check: combined flow");ax.legend();fig.tight_layout();fig.savefig(FIG/"check_on_vs_off.png",dpi=180);plt.close(fig)
    x=results[(results.row_type=="combination")&(results.labelling_variant=="raw")&(results.reliability_mode=="on")&(results.root_group=="day1")&(results.group=="dev")&(results.combination=="pseudotime")]
    fig,ax=plt.subplots(figsize=(8,5));
    for mode,d in x.groupby("palantir_mode"): ax.scatter(d.seed+(0 if mode=="default" else .15),d.sign_agreement,label=mode,alpha=.7)
    ax.set_ylim(0,1.02);ax.set_xlabel("Dataset seed");ax.set_ylabel("Developmental agreement");ax.set_title("Pseudotime agreement by Palantir mode");ax.legend();fig.tight_layout();fig.savefig(FIG/"pseudotime_by_mode.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));
    for i,(label,d) in enumerate(j.groupby("truth_class")): ax.scatter(np.full(len(d),i),d.median_lead,alpha=.65,label=label)
    ax.axhline(0,color="black",lw=.8);ax.set_xticks(range(j.truth_class.nunique()),sorted(j.truth_class.unique()));ax.set_ylabel("Median lead eigenvalue");ax.set_title("J lead eigenvalue by truth class");fig.tight_layout();fig.savefig(FIG/"j_lead_by_truth_class.png",dpi=180);plt.close(fig)


def write_result_markdown(gates:dict[str,Any],results:pd.DataFrame,pt:pd.DataFrame,j:pd.DataFrame,flags_frame:pd.DataFrame)->None:
    lines=["# orient/S06 executed result", "", "## What ran", "Observed-data fitted units, fixed scVelo/Palantir runs, and truth-separated scoring calls completed. Simulator truth was absent from every `orient()` input and used only in a separate scoring copy after orientation.", "", "## Gates"]
    for name,value in gates.items(): lines.append(f"- **{name}**: {'PASS' if value.get('pass') else 'FAIL'} — `{json.dumps(json_safe(value),sort_keys=True)}`")
    lines += ["", "## Counts",f"- scored rows: {len(results)}",f"- Palantir guard records: {len(pt)}",f"- J class records: {len(j)}",f"- flags rows: {len(flags_frame)}"]
    (OUT/"RESULT_S06.md").write_text("\n".join(lines)+"\n")


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--preset",choices=PRESETS);parser.add_argument("--seed",type=int,choices=range(3));args=parser.parse_args()
    if (args.preset is None)!=(args.seed is None):parser.error("--preset and --seed are paired")
    started=time.time(); rows=input_rows();fit,user=manifests()
    selected=[r for r in rows if args.preset is None or (r["preset"]==args.preset and int(r["seed"])==args.seed)]
    all_results=[];all_by=[];all_diag=[];all_pt=[];all_j=[];all_flags=[]
    for row in selected:
        print(f"[S06 score] start {row['preset']} s{row['seed']}", flush=True)
        for method in METHODS:
            print(f"[S06 score] {row['preset']} s{row['seed']} {method}", flush=True)
            if row["preset"]=="v2_stationary_cycle":
                # Task-specified stationary-cycle controls are flags only.
                all_flags.append(flags(row,method,fit["files"]))
                continue
            # J is a scoring-only truth check independent of orientation inputs.
            fr=fit_record(fit["files"],row["preset"],int(row["seed"]),method);score=ad.read_h5ad(ROOT/fr["path"]);all_j+=j_rows(score,row,method)
            results,by,diag,pt=run_unit(row,method,fit["files"],user);all_results+=results;all_by+=by;all_diag+=diag;all_pt+=pt
        print(f"[S06 score] complete {row['preset']} s{row['seed']}", flush=True)
    # For a filtered preflight output, write self-contained tables. Full task invokes no filter.
    results=pd.DataFrame(all_results);by=pd.DataFrame(all_by);diag=pd.DataFrame(all_diag);pt=pd.DataFrame(all_pt);j=pd.DataFrame(all_j);flags_frame=pd.DataFrame(all_flags)
    OUT.mkdir(parents=True,exist_ok=True)
    results.to_csv(OUT/"results.csv",index=False);by.to_csv(OUT/"by_day.csv",index=False);diag.to_csv(OUT/"diagnostics.csv",index=False);pt.to_csv(OUT/"palantir_runs.csv",index=False);j.to_csv(OUT/"j_check.csv",index=False);flags_frame.to_csv(OUT/"flags.csv",index=False)
    s=summary(results);s.to_csv(OUT/"summary.csv",index=False)
    if args.preset is None:
        gates=gate_values(results,diag,pt,j,flags_frame);write_result_markdown(gates,results,pt,j,flags_frame);figures(results,s,j)
        (OUT/"scoring_manifest.json").write_text(json.dumps({"task":"orient/S06","started_utc":utc(started),"finished_utc":utc(time.time()),"wall_seconds":round(time.time()-started,3),"versions":{x:version(x) for x in ("numpy","scikit-learn","torch","scvelo","palantir","scanpy")},"gates":json_safe(gates),"n_results":len(results),"n_by_day":len(by),"n_palantir":len(pt),"n_j":len(j)},indent=2,sort_keys=True)+"\n")
    print(json.dumps({"preset":args.preset or "all","seed":args.seed,"results":len(results),"palantir":len(pt),"wall_seconds":round(time.time()-started,3)},indent=2))

if __name__=="__main__":main()
