"""Setty + scVelo dynamical — measured-flow drift on the same CD34+
dataset as the DSM snapshot run.

Three-way comparison on human CD34+ bone marrow:
  * Setty DSM snapshot   — ``benchmarks/v2/setty_cd34/`` (DSM drift)
  * Setty scVelo (this)  — ``benchmarks/v2/setty_scvelo/`` (splicing flow)
  * Dynamo scNT-seq      — ``benchmarks/v2/dynamo_hematopoiesis/`` (labelling flow)

scVelo dynamical mode uses the Setty spliced / unspliced layers to
estimate per-cell velocity, then project to PCA via scVelo's
``velocity_pca``. We regress the velocity_pca onto a scJDO2
MeasuredDrift (MLP(z → v)) exactly as in Dynamo §7, then run Phase C
anchor + opspace + the gate (behaviour-by-cluster + decision archetype).

Clock is relative (scVelo's "time" is not calibrated to a physical
clock), so eigenvalues are reported in ``(relative)`` units; this is
analogous to Setty DSM, not to Dynamo scNT-seq.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv

from scjdo2 import io as scjdo2_io
from scjdo2 import opspace
from scjdo2.data import load_setty_cd34, PROGENITOR_CLUSTERS, TERMINAL_CLUSTERS
from scjdo2.measured_drift import MeasuredDrift, MeasuredDriftConfig
from scjdo2.pipeline import fit_rate_and_gene_field


REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "benchmarks" / "v2" / "setty_scvelo"


def _run_scvelo(a, n_top_genes: int, n_pcs: int):
    """Preprocess + scVelo dynamical. Returns the processed adata with
    ``velocity_pca`` populated. scVelo 0.3.x doesn't expose
    ``filter_genes_dispersion`` directly; we use scanpy's HVG
    before scVelo's filter_and_normalize."""
    # Pre-filter to HVG via scanpy on a log-normalised copy, keeping
    # GATA1 / SPI1 regardless.
    tmp = a.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=n_top_genes, flavor="seurat")
    hv = tmp.var["highly_variable"].to_numpy()
    for g in ("GATA1", "SPI1"):
        if g in a.var_names:
            hv[a.var_names.get_loc(g)] = True
    a = a[:, hv].copy()
    print(f"[setty-scvelo]   HVG kept {a.n_vars} genes")
    scv.pp.filter_and_normalize(a, min_shared_counts=20)
    scv.pp.moments(a, n_pcs=30, n_neighbors=30)
    scv.tl.recover_dynamics(a, n_jobs=4)
    scv.tl.velocity(a, mode="dynamical")
    scv.tl.velocity_graph(a, n_jobs=4)
    # Project velocity to PCA.
    scv.tl.velocity_embedding(a, basis="pca")
    # Make sure X_pca and velocity_pca are trimmed to n_pcs.
    if "X_pca" not in a.obsm:
        sc.tl.pca(a, n_comps=n_pcs)
    a.obsm["X_pca"] = np.asarray(a.obsm["X_pca"])[:, :n_pcs].astype(np.float32)
    # Dynamical mode writes "velocity_pca" (not "velocity_embedding"); fall
    # back to velocity_embedding if the dynamical path isn't picked up.
    v = a.obsm.get("velocity_pca", a.obsm.get("velocity_embedding"))
    if v is None:
        raise RuntimeError("scVelo did not produce velocity_pca or velocity_embedding")
    v = np.asarray(v)
    # Pad or trim to n_pcs.
    if v.shape[1] != n_pcs:
        if v.shape[1] < n_pcs:
            pad = np.zeros((v.shape[0], n_pcs - v.shape[1]), dtype=v.dtype)
            v = np.concatenate([v, pad], axis=1)
        else:
            v = v[:, :n_pcs]
    a.obsm["velocity_pca"] = v.astype(np.float32)
    return a


def _report(a) -> dict:
    scj = a.uns.get("scjdo2", {})
    beh = a.obs["op_behaviour"].astype(str).to_numpy()
    rows_by_cluster = []
    for cl in sorted(set(a.obs["cluster"].astype(str))):
        m = a.obs["cluster"].astype(str).to_numpy() == cl
        c = pd.Series(beh[m]).value_counts()
        rec = {"cluster": cl, "n": int(m.sum()),
               "is_terminal": cl in TERMINAL_CLUSTERS,
               "is_progenitor": cl in PROGENITOR_CLUSTERS}
        for k in ("fork", "holder", "drift", "damped_oscillator",
                  "growing_oscillator"):
            rec[k] = int(c.get(k, 0))
        rec["frac_fork"] = rec["fork"] / rec["n"] if rec["n"] else 0.0
        rec["frac_holder"] = rec["holder"] / rec["n"] if rec["n"] else 0.0
        rows_by_cluster.append(rec)
    return {"behaviour_by_cluster": pd.DataFrame(rows_by_cluster)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-pcs", type=int, default=10)
    p.add_argument("--n-top-genes", type=int, default=2000)
    p.add_argument("--drift-epochs", type=int, default=1500)
    p.add_argument("--rate-epochs", type=int, default=120)
    p.add_argument("--n-anchors", type=int, default=200)
    p.add_argument("--anchor-max-live", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print("[setty-scvelo] load")
    a = load_setty_cd34()
    print(f"[setty-scvelo]   shape={a.shape}  layers={list(a.layers.keys())}")

    print("[setty-scvelo] scVelo dynamical")
    t0 = time.monotonic()
    a = _run_scvelo(a, args.n_top_genes, args.n_pcs)
    print(f"[setty-scvelo]   scVelo in {time.monotonic()-t0:.1f}s  "
          f"shape={a.shape}  (velocity_pca stored)")

    # Measured drift.
    print(f"[setty-scvelo] MeasuredDrift fit ({args.drift_epochs} epochs)")
    t0 = time.monotonic()
    z = np.asarray(a.obsm["X_pca"]).astype(np.float32)
    v = np.asarray(a.obsm["velocity_pca"]).astype(np.float32)
    # Guard against NaN rows from scVelo's dynamical fit.
    bad = ~np.all(np.isfinite(v), axis=1)
    if bad.any():
        print(f"[setty-scvelo]   dropping {int(bad.sum())} cells with NaN velocity")
        a = a[~bad].copy()
        z = np.asarray(a.obsm["X_pca"]).astype(np.float32)
        v = np.asarray(a.obsm["velocity_pca"]).astype(np.float32)
    drift = MeasuredDrift(dim=z.shape[1], config=MeasuredDriftConfig(
        hidden=(96, 96, 96), n_epochs=args.drift_epochs,
        weight_decay=0.0, seed=args.seed, clock_calibrated=False,
    )).fit(z, v, verbose=False)
    print(f"[setty-scvelo]   drift fit {time.monotonic()-t0:.1f}s")
    J = drift.jacobian_autodiff(z)
    v_pred = drift.velocity(z)
    scj = a.uns.setdefault("scjdo2", {})
    scj["J"] = J
    scj["meta"] = {
        "clock_calibrated": False, "use_rep": "X_pca", "time_key": None,
        "jacobian_method": "autodiff",
    }
    scj["field_reason_summary"] = {"ok": 1.0, "autodiff": True}
    a.obsm["X_drift_v"] = v_pred
    a.obs["pseudotime"] = a.obsm["X_pca"][:, 0].astype(float)

    # Phase C.
    print(f"[setty-scvelo] Phase C (anchor max_live={args.anchor_max_live})")
    t0 = time.monotonic()
    fit_rate_and_gene_field(
        a, use_rep="X_pca", n_epochs_rate=args.rate_epochs,
        k=30, cv_sample_cells=20, seed=args.seed,
        anchor_fit=True, n_anchors=args.n_anchors, anchor_top_k=10,
        anchor_max_live=args.anchor_max_live,
    )
    print(f"[setty-scvelo]   Phase C in {time.monotonic()-t0:.1f}s")

    print("[setty-scvelo] opspace.compute")
    opspace.compute(a)
    arch = a.uns["scjdo2"].get("archetypes") or {}

    rep = _report(a)
    rep["behaviour_by_cluster"].to_csv(OUT / "behaviour_by_cluster.csv", index=False)
    print("\n[setty-scvelo] BEHAVIOUR BY CLUSTER:")
    print(rep["behaviour_by_cluster"].to_string(index=False))

    # Decision-region archetype + GATA1/SPI1 ranks.
    W = a.obsm["archetype_weights"]
    clusters = a.obs["cluster"].astype(str).to_numpy()
    cl_names = sorted(set(clusters))
    acw_rows = []
    for k in range(W.shape[1]):
        w = W[:, k]
        tot = float(w.sum()) or 1.0
        rec = {"archetype": k, "n_cells_hard": int((W.argmax(axis=1) == k).sum())}
        for cl in cl_names:
            m = clusters == cl
            rec[cl] = float(w[m].sum() / tot)
        acw_rows.append(rec)
    acw = pd.DataFrame(acw_rows)
    acw.to_csv(OUT / "archetype_by_cluster.csv", index=False)
    prog_cols = [c for c in acw.columns if c in PROGENITOR_CLUSTERS]
    decision_arch = int(acw.loc[acw[prog_cols].sum(axis=1).idxmax(), "archetype"])
    print(f"\n[setty-scvelo] decision-region archetype: {decision_arch}")

    region_rows = []
    arch_info = arch.get(decision_arch, {})
    for table_name in ("responders", "drivers_left"):
        df = arch_info.get(table_name)
        if df is None:
            continue
        for g in ("GATA1", "SPI1"):
            hit = df.index[df["gene"].astype(str) == g].tolist()
            if not hit:
                continue
            i = int(hit[0])
            r = df.loc[i].to_dict()
            region_rows.append({
                "archetype": decision_arch, "table": table_name, "gene": g,
                "rank": i, "lead_re": float(arch_info["eigenvalues"][0].real),
                "behaviour": arch_info["behaviour"],
                "n_cells": int(arch_info["n_cells"]),
                "fisher": float(r["fisher"]),
                "snr": float(r.get("snr", float("nan"))),
                "sensitivity": float(r.get("sensitivity", float("nan"))),
                "reason": str(r["reason"]),
            })
    region_df = pd.DataFrame(region_rows)
    region_df.to_csv(OUT / "decision_region_archetype.csv", index=False)
    print("\n[setty-scvelo] decision-archetype GATA1/SPI1:")
    print(region_df.to_string(index=False))

    # Overall behaviour summary.
    beh = a.obs["op_behaviour"].astype(str).value_counts()
    beh.to_csv(OUT / "overall_behaviour.csv")
    print("\n[setty-scvelo] overall behaviour:")
    print(beh.to_string())
    print(f"\n[setty-scvelo] outputs → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
