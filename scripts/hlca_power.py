#!/usr/bin/env python3
"""EXPLORATORY HX02 power analysis on real H03 donor cells.

Synthetic, known-size effects are injected into stored H03 drift-subsample cells.
No model is fitted and no frozen H04 output is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import zlib
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scjdo2.compare import anticipation_trend, centroid_operator, donor_operator, isotropy_trend, stiffness_trend

ROOT = Path(__file__).resolve().parents[1]
HLCA = ROOT / "benchmarks" / "aging_hlca"
OUT = HLCA / "power"
FIG = ROOT / "benchmarks" / "figures" / "aging_hlca" / "power"
INPUT = HLCA / "data" / "h03_primary_drift.h5ad"
PRIMARY = HLCA / "tests_primary.csv"
H03_REPORT = ROOT / "coord" / "hlca" / "reports" / "H03.json"
H04_REPORT = ROOT / "coord" / "hlca" / "reports" / "H04.json"
GRID = OUT / "power_grid.csv"
MDE = OUT / "mde.csv"
RESULT = OUT / "RESULT_HX02.md"
CHECKSUMS = OUT / "input_checksums.json"
METRICS = OUT / "metrics.json"

SEED = 0
N_COMMON, N_DRAWS, N_CENTROID = 50, 100, 30
N_PERM, N_REPLICATES, DONOR_FRACTION = 200, 50, 0.80
FOOTER = "EXPLORATORY · hlca/HX02 · not part of the frozen outcome"
KINDS = ("single_axis", "uniform", "drift")
SIZES = {"single_axis": (1.25, 1.5, 2., 3., 5., 8.), "uniform": (1.1, 1.25, 1.5, 2., 3.), "drift": (.5, 1., 2., 4., 8., 16.)}
NULL = {"single_axis": 1., "uniform": 1., "drift": 0.}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verified_inputs() -> list[dict[str, object]]:
    checks = ((H03_REPORT, "benchmarks/aging_hlca/data/h03_primary_drift.h5ad"), (H04_REPORT, "benchmarks/aging_hlca/tests_primary.csv"))
    rows = []
    for report_path, rel in checks:
        report = json.loads(report_path.read_text())
        found = next((x for x in report["artifacts"] if x["path"] == rel), None)
        if found is None:
            raise RuntimeError(f"{report_path} does not inventory {rel}")
        path = ROOT / rel
        actual = sha256(path)
        if path.stat().st_size != int(found["bytes"]) or actual != found["sha256"]:
            raise RuntimeError(f"input checksum mismatch: {rel}")
        rows.append({"path": rel, "bytes": path.stat().st_size, "sha256": actual, "source_report": str(report_path.relative_to(ROOT))})
    return rows


def load_data() -> tuple[dict[str, dict[str, object]], tuple[str, ...]]:
    primary = pd.read_csv(PRIMARY)
    if len(primary) != 13 or primary["ann_level_3"].duplicated().any() or {"ann_level_3", "n_donors"} - set(primary):
        raise RuntimeError("H04 primary table must contain the frozen unique 13-type set")
    types = tuple(primary["ann_level_3"].astype(str))
    expected_n = primary.set_index("ann_level_3")["n_donors"].astype(int).to_dict()
    adata = ad.read_h5ad(INPUT)
    z_all = np.asarray(adata.obsm.get("X_scVI_primary"), dtype=np.float64)
    j_all = np.asarray(adata.uns.get("scjdo2", {}).get("J_latent"), dtype=np.float64)
    obs = adata.obs.copy()
    if z_all.shape != (adata.n_obs, 15) or j_all.shape != (adata.n_obs, 15, 15):
        raise RuntimeError(f"unexpected H03 primary schema: z={z_all.shape}; J={j_all.shape}")
    for column in ("ann_level_3", "donor_id", "age"):
        if column not in obs:
            raise RuntimeError(f"H03 drift observation column missing: {column}")
    labels = obs["ann_level_3"].astype(str).to_numpy(); donors = obs["donor_id"].astype(str).to_numpy(); ages_all = pd.to_numeric(obs["age"], errors="raise").to_numpy(float)
    data: dict[str, dict[str, object]] = {}
    for cell_type in types:
        take = np.flatnonzero(labels == cell_type)
        z, jacobian, donor, ages = z_all[take], j_all[take], donors[take], ages_all[take]
        donor_names, donor_ages, donor_positions = [], [], []
        for donor_id in sorted(set(donor)):
            pos = np.flatnonzero(donor == donor_id)
            if len(pos) >= N_COMMON:
                values = np.unique(ages[pos])
                if len(values) != 1: raise RuntimeError(f"age varies within {cell_type}/{donor_id}")
                donor_names.append(donor_id); donor_ages.append(float(values[0])); donor_positions.append(pos)
        if len(donor_names) != expected_n[cell_type]:
            raise RuntimeError(f"H04 contributor mismatch for {cell_type}: {len(donor_names)} != {expected_n[cell_type]}")
        n_sub = int(np.floor(DONOR_FRACTION * len(donor_names)))
        if n_sub < 4: raise RuntimeError(f"insufficient 80% donor subset for {cell_type}")
        reps = []
        for r in range(N_REPLICATES):
            rng = np.random.default_rng(zlib.crc32(f"{cell_type}|rep{r}".encode()))
            reps.append(np.sort(rng.choice(len(donor_names), n_sub, replace=False)))
        young = ages < 45
        ref = centroid_operator(z[young], jacobian[young], n_cells_centroid=N_CENTROID)
        ev, vec = np.linalg.eigh(0.5 * (ref + ref.T)); u = vec[:, int(np.argmax(ev))]
        sigma = float(np.std(z[young] @ u))
        if not np.isfinite(sigma) or sigma <= 0: raise RuntimeError(f"invalid young soft-axis σ for {cell_type}")
        data[cell_type] = {"z":z, "j":jacobian, "t":(ages - 10.) / 66., "donor_names":tuple(donor_names), "donor_ages":np.array(donor_ages), "donor_positions":tuple(donor_positions), "reps":tuple(reps), "ref":ref, "u":u, "sigma":sigma, "n_sub":n_sub}
    return data, types


def injected(item: dict[str, object], kind: str, size: float) -> tuple[np.ndarray, np.ndarray]:
    z, j, t = item["z"], item["j"], item["t"]
    if kind == "single_axis":
        if size == 1.: return z, j
        u = item["u"]
        curvature = np.einsum("i,nij,j->n", u, j, u, optimize=True)
        return z, j + ((np.power(size, -t) - 1.) * curvature)[:, None, None] * np.outer(u, u)[None, :, :]
    if kind == "uniform":
        return (z, j) if size == 1. else (z, np.power(size, -t)[:, None, None] * j)
    if kind == "drift":
        return (z, j) if size == 0. else (z + (size * item["sigma"] * t)[:, None] * item["u"][None, :], j)
    raise ValueError(kind)


def build_operators(cell_type: str, item: dict[str, object], z: np.ndarray, j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ops, centroids = [], []
    for donor, pos in zip(item["donor_names"], item["donor_positions"], strict=True):
        op, centroid = donor_operator(z[pos], j[pos], n_common=N_COMMON, n_draws=N_DRAWS, n_cells_centroid=N_CENTROID, seed=zlib.crc32(f"{cell_type}|{donor}".encode()))
        ops.append(op); centroids.append(centroid)
    return np.stack(ops), np.stack(centroids)


def test_power(item: dict[str, object], kind: str, ops: np.ndarray, centroids: np.ndarray) -> float:
    hits = []
    for selection in item["reps"]:
        ages = item["donor_ages"][selection]
        if kind == "single_axis":
            result = isotropy_trend(ops[selection], ages, item["ref"], n_perm=N_PERM, seed=SEED); hits.append(bool(result.passes))
        elif kind == "uniform":
            result = stiffness_trend(ops[selection], ages, n_perm=N_PERM, seed=SEED); hits.append(bool(result.passes and result.statistic < 0.))
        else:
            result = anticipation_trend(centroids[selection], ages, item["ref"], n_perm=N_PERM, seed=SEED); hits.append(bool(result.passes))
    return float(np.mean(hits))


def mde(values: pd.DataFrame) -> float | str:
    found = values.loc[(~values["is_null"]) & (values["power"] >= .80), "size"]
    return float(found.min()) if len(found) else "> max"


def plot(grid: pd.DataFrame, mde_table: pd.DataFrame, types: tuple[str, ...]) -> list[Path]:
    FIG.mkdir(parents=True, exist_ok=True); paths = []
    titles = {"single_axis":"Single-axis softening power", "uniform":"Uniform softening power", "drift":"Soft-axis drift power"}
    for kind in KINDS:
        fig, ax = plt.subplots(figsize=(11.8, 6.4)); frame = grid[(grid.kind == kind) & (~grid.is_null)]
        for cell_type in types:
            p = frame[frame.ann_level_3 == cell_type].sort_values("size")
            ax.plot(p["size"], p["power"], marker="o", markersize=3.5, linewidth=1.1, label=cell_type)
        ax.axhline(.8, color="#c94c4c", linestyle="--", linewidth=1.1, label="80% power")
        ax.set_xscale("log", base=2); ax.set_ylim(-.02, 1.04); ax.grid(color=".9")
        ax.set_xlabel("fold F" if kind != "drift" else "drift D (young soft-axis σ)"); ax.set_ylabel("power across 50 donor subsamples")
        ax.set_title(titles[kind], loc="left", weight="bold"); ax.legend(ncol=3, fontsize=6.1, frameon=False, loc="lower right")
        fig.text(.5, .01, FOOTER, ha="center", fontsize=7, color=".35"); fig.tight_layout(rect=(0,.04,1,.98))
        for extension in ("pdf", "png"):
            path = FIG / f"power_curve_{kind}.{extension}"; fig.savefig(path, dpi=200, bbox_inches="tight"); paths.append(path)
        plt.close(fig)
    cols = (("mde_single_axis_F", "single-axis\nF"), ("mde_uniform_F", "uniform\nF"), ("mde_drift_D", "drift\nD σ"))
    values = np.full((len(types),3), np.nan); labels = np.empty((len(types),3), object)
    for i, cell_type in enumerate(types):
        row = mde_table[mde_table.ann_level_3 == cell_type].iloc[0]
        for j, (col, _) in enumerate(cols):
            x = row[col]; labels[i,j] = str(x)
            if str(x) != "> max": values[i,j] = float(x)
    cmap = plt.cm.viridis.copy(); cmap.set_bad("#d9dde3")
    fig, ax = plt.subplots(figsize=(8.8,8.7)); im = ax.imshow(values, aspect="auto", cmap=cmap)
    median = np.nanmedian(values)
    for i in range(len(types)):
        for j in range(3): ax.text(j,i,labels[i,j],ha="center",va="center",fontsize=8,color="white" if np.isfinite(values[i,j]) and values[i,j]<median else "#19232c")
    ax.set_yticks(range(len(types)),types,fontsize=8); ax.set_xticks(range(3),[x[1] for x in cols]); ax.set_title("MDE at 80% power\ngrey = no grid size reached 80%",loc="left",weight="bold")
    fig.colorbar(im,ax=ax,shrink=.8,label="MDE value (F or D σ; column scales differ)")
    fig.text(.5,.01,FOOTER,ha="center",fontsize=7,color=".35"); fig.tight_layout(rect=(0,.04,1,.98))
    for extension in ("pdf","png"):
        path=FIG/f"mde_heatmap.{extension}"; fig.savefig(path,dpi=200,bbox_inches="tight"); paths.append(path)
    plt.close(fig); return paths


def table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines=["| "+" | ".join(columns)+" |","|"+"|".join(["---"]*len(columns))+"|"]
    for values in frame[columns].itertuples(index=False,name=None):
        line=[]
        for x in values: line.append(f"{float(x):.3g}" if isinstance(x,(float,np.floating)) else str(x))
        lines.append("| "+" | ".join(line)+" |")
    return "\n".join(lines)


def median_value(series: pd.Series, maximum_grid: float) -> str:
    """Median respecting right-censored (> max) MDE values."""
    censored = sum(str(x) == "> max" for x in series)
    if censored >= (len(series) + 1) // 2:
        return f">{maximum_grid:g}"
    numbers=[float(x) for x in series if str(x) != "> max"]
    return f"{np.median(numbers):g}" if numbers else f">{maximum_grid:g}"


def fold_phrase(value: str) -> str:
    return f">{value[1:]}×" if value.startswith(">") else f"×{value}"


def write_result(mde_table: pd.DataFrame) -> None:
    definitions = """For each type: `ref` = the H04 reference operator (pooled cells of donors aged < 45, `compare.centroid_operator`, 30 cells); `u` = its soft axis (eigenvector of its largest eigenvalue); `σ_u` = standard deviation of the young pooled cells' projections on `u`. Age is mapped to `t = (age − 10) / 66` (0 at 10 years, 1 at 76). Injections apply to **every cell** of the type, using that cell's donor age:
1. **Single-axis softening** (target: `isotropy_trend`), fold `F`:
   `J' = J + (F^(−t) − 1) · (uᵀ J u) · u uᵀ` — the curvature along `u`
   is divided by `F^t`; other directions unchanged.
   `F ∈ {1.25, 1.5, 2, 3, 5, 8}`.
2. **Uniform softening** (target: `stiffness_trend`), fold `F`:
   `J' = F^(−t) · J`. `F ∈ {1.1, 1.25, 1.5, 2, 3}`.
3. **Drift along the young soft axis** (target: `anticipation_trend`),
   size `D` (in units of `σ_u` over the age range):
   `z' = z + D · σ_u · t · u`. `D ∈ {0.5, 1, 2, 4, 8, 16}`.
Plus a **null row** (no injection, `F = 1`, `D = 0`) for the baseline pass rate."""
    cols=["ann_level_3","mde_single_axis_F","mde_uniform_F","mde_drift_D","null_pass_rate_iso","null_pass_rate_stiff","null_pass_rate_ant"]
    null_cols=["ann_level_3","null_pass_rate_iso","null_pass_rate_stiff","null_pass_rate_ant"]
    high=[]
    for row in mde_table.itertuples(index=False):
        for label,value in (("iso",row.null_pass_rate_iso),("stiff",row.null_pass_rate_stiff),("ant",row.null_pass_rate_ant)):
            if value > .20: high.append(f"{row.ann_level_3}/{label}={value:.2f}")
    text="# EXPLORATORY — hlca/HX02 power analysis\n\n## What ran\nReal H03 drift-subsample cells were used with synthetic, known-size injections only; no model was fitted. Each grid row rebuilt deterministic equal-count donor operators (`n_common=50`, 100 draws, 30-cell centroid) and used 50 seeded 80%-donor subsamples with 200 age permutations per target test.\n\n## Definitions\n"+definitions+"\n\n## MDE table\n"+table(mde_table,cols)+"\n\n"
    single = median_value(mde_table.mde_single_axis_F, max(SIZES["single_axis"]))
    uniform = median_value(mde_table.mde_uniform_F, max(SIZES["uniform"]))
    drift = median_value(mde_table.mde_drift_D, max(SIZES["drift"]))
    text+=f"Across the 13 types, the median MDE was {fold_phrase(single)} for single-axis softening, {fold_phrase(uniform)} for uniform softening, and {drift} σ for drift along the soft axis, over ages 10–76.\n\n"
    text+="## Null rows\n"+table(mde_table,null_cols)+"\n\n"
    text+=("Null pass rates exceeding 0.20: "+"; ".join(high)+".\n" if high else "No null pass rate exceeded 0.20.\n")
    RESULT.write_text(text)


def run(dry_run: bool) -> dict[str, object]:
    checks=verified_inputs(); data,types=load_data()
    n_rows=len(types)*sum(1+len(SIZES[k]) for k in KINDS)
    n_builds=sum(len(x["donor_names"])*(1+sum(len(SIZES[k]) for k in KINDS)) for x in data.values())
    plan={"n_types":len(types),"n_grid_rows_expected":n_rows,"n_operator_builds_expected":n_builds,"n_test_evaluations_expected":n_rows*N_REPLICATES,"n_permutations_per_test":N_PERM,"n_donors_per_replicate":{name:int(x["n_sub"]) for name,x in data.items()},"seed":SEED}
    if dry_run:
        print(json.dumps({**plan,"dry_run":True},indent=2,sort_keys=True)); return plan
    OUT.mkdir(parents=True,exist_ok=True); CHECKSUMS.write_text(json.dumps(checks,indent=2)+"\n")
    rows=[]; start=time.monotonic(); done=0
    for cell_type,item in data.items():
        null_ops=None
        for kind in KINDS:
            for size in (NULL[kind],)+SIZES[kind]:
                done+=1; is_null=size==NULL[kind]
                if is_null and null_ops is not None: ops,cents=null_ops
                else:
                    z,j=injected(item,kind,float(size)); ops,cents=build_operators(cell_type,item,z,j)
                    if is_null: null_ops=(ops,cents)
                power=test_power(item,kind,ops,cents)
                rows.append({"ann_level_3":cell_type,"kind":kind,"size":float(size),"power":power,"n_donors_per_replicate":int(item["n_sub"]),"is_null":bool(is_null)})
                print(f"[{done:03d}/{n_rows}] {cell_type} {kind} size={size:g} power={power:.2f} elapsed={time.monotonic()-start:.1f}s",flush=True)
    grid=pd.DataFrame(rows); grid["_t"]=grid.ann_level_3.map({x:i for i,x in enumerate(types)}); grid["_k"]=grid.kind.map({x:i for i,x in enumerate(KINDS)}); grid=grid.sort_values(["_t","_k","size"],kind="stable").drop(columns=["_t","_k"]).reset_index(drop=True); grid.loc[:, ["ann_level_3", "kind", "size", "power", "n_donors_per_replicate"]].to_csv(GRID,index=False)
    mde_rows=[]
    for cell_type in types:
        one=grid[grid.ann_level_3==cell_type]
        mde_rows.append({"ann_level_3":cell_type,"mde_single_axis_F":mde(one[one.kind=="single_axis"]),"mde_uniform_F":mde(one[one.kind=="uniform"]),"mde_drift_D":mde(one[one.kind=="drift"]),"null_pass_rate_iso":float(one[(one.kind=="single_axis")&one.is_null].power.iloc[0]),"null_pass_rate_stiff":float(one[(one.kind=="uniform")&one.is_null].power.iloc[0]),"null_pass_rate_ant":float(one[(one.kind=="drift")&one.is_null].power.iloc[0])})
    mde_table=pd.DataFrame(mde_rows); mde_table.to_csv(MDE,index=False); figure_paths=plot(grid,mde_table,types); write_result(mde_table)
    out={**plan,"n_grid_rows":len(grid),"n_figures":len(figure_paths)//2,"median_mde_single_axis_F":median_value(mde_table.mde_single_axis_F,max(SIZES["single_axis"])),"median_mde_uniform_F":median_value(mde_table.mde_uniform_F,max(SIZES["uniform"])),"median_mde_drift_D":median_value(mde_table.mde_drift_D,max(SIZES["drift"])),"n_types_mde_beyond_grid":int((mde_table[["mde_single_axis_F","mde_uniform_F","mde_drift_D"]]=="> max").to_numpy().sum()),"max_null_pass_rate":float(mde_table[["null_pass_rate_iso","null_pass_rate_stiff","null_pass_rate_ant"]].to_numpy(float).max()),"mde_by_type":mde_table.set_index("ann_level_3").to_dict(orient="index"),"figure_paths":[str(p.relative_to(ROOT)) for p in figure_paths]}
    METRICS.write_text(json.dumps(out,indent=2,default=str)+"\n"); print(json.dumps(out,indent=2,default=str,sort_keys=True)); return out


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--seed",type=int,default=SEED); parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args()
    if args.seed != SEED: raise ValueError("HX02 is frozen at seed 0")
    run(args.dry_run); return 0

if __name__ == "__main__": raise SystemExit(main())
