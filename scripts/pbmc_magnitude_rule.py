"""PBMC magnitude-rule classifier at d_latent=15 (post-5e7ad2f spec).

Replaces the r-1 rule with the user's final magnitude rule
(2026-10): basin iff all tangent eigenvalues < +tol AND some eigenvalue
< -tol; fork iff any tangent eigenvalue > +tol; drift iff all |λ| < tol.
Also reports the continuous basin index
``bi = -max(ev_r) / median|ev_r_neg|``.

Uses ``scjdo2.opspace.classify_basin_latent`` directly (which was
patched to the magnitude rule + Σ-based tangent in this commit).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

from scjdo2.data import (
    CYCLE_GENES,
    RESTING_TYPES,
    load_pbmc3k,
)
from scjdo2.energy_drift import EnergyDriftAdapter, EnergyDriftConfig
from scjdo2.opspace import classify_basin_latent


REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "benchmarks" / "v2" / "pbmc_basin_magnitude"


def _fit_scvi(adata, n_latent: int, n_epochs: int, seed: int = 0):
    import scvi
    scvi.settings.seed = seed
    adata = adata.copy()
    adata.layers["counts"] = adata.X.copy()
    scvi.model.SCVI.setup_anndata(adata, layer="counts")
    m = scvi.model.SCVI(adata, n_latent=int(n_latent))
    m.train(max_epochs=int(n_epochs), check_val_every_n_epoch=10,
            early_stopping=False, enable_progress_bar=False)
    z = np.asarray(m.get_latent_representation(adata)).astype(np.float32)
    return z


def _oracle_tol(z: np.ndarray, cluster: np.ndarray,
                resting_types) -> float:
    leads = []
    for cl in sorted(set(cluster)):
        if cl not in set(resting_types):
            continue
        m = cluster == cl
        if m.sum() < 10:
            continue
        Sigma = np.cov(z[m], rowvar=False, bias=False) + 1e-4 * np.eye(z.shape[1])
        ev = np.linalg.eigvalsh(-np.linalg.inv(Sigma))
        leads.append(abs(float(ev.max())))
    return 0.1 * float(np.median(leads)) if leads else 0.1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--d-latent", type=int, default=15)
    p.add_argument("--scvi-epochs", type=int, default=180)
    p.add_argument("--drift-epochs", type=int, default=1500)
    p.add_argument("--n-top-genes", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print("[pbmc-mag] load pbmc3k")
    a = load_pbmc3k()
    sc.pp.filter_genes(a, min_cells=3)
    tmp = a.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=args.n_top_genes, flavor="seurat")
    hv = tmp.var["highly_variable"].to_numpy()
    for g in list(CYCLE_GENES) + ["CD3D","CD4","CD8A","CD8B","MS4A1","CD79A",
                                   "NKG7","GNLY","CD14","FCGR3A","CST3","LYZ",
                                   "FCER1A","PPBP"]:
        if g in a.var_names:
            hv[a.var_names.get_loc(g)] = True
    a = a[:, hv].copy()
    print(f"[pbmc-mag]   shape={a.shape}")

    t0 = time.monotonic()
    z = _fit_scvi(a, args.d_latent, args.scvi_epochs, args.seed)
    a.obsm["X_scVI"] = z
    print(f"[pbmc-mag] scVI {time.monotonic()-t0:.1f}s")

    t0 = time.monotonic()
    drift = EnergyDriftAdapter(dim=z.shape[1], config=EnergyDriftConfig(
        hidden=96, depth=3, n_epochs=args.drift_epochs, seed=args.seed,
        batch_size=256, loss="esm", lr=5e-4, beta=1.0,
        esm_laplacian="exact" if z.shape[1] <= 20 else "hutchinson",
    )).fit(z, verbose=False)
    print(f"[pbmc-mag] ESM drift {time.monotonic()-t0:.1f}s")

    sym = drift.symmetry_check(z, n_cells=100, seed=args.seed)
    J = drift.jacobian_autodiff(z)
    a.uns.setdefault("scjdo2", {})["J_latent"] = J
    a.uns["scjdo2"]["J"] = J
    cluster = a.obs["cluster"].astype(str).to_numpy()
    tol = _oracle_tol(z, cluster, RESTING_TYPES)
    print(f"[pbmc-mag] oracle_tol = {tol:.4f}  sym_median = {sym['median_asym']:.3e}")

    classify_basin_latent(a, tol=tol, use_rep="X_scVI")
    beh = a.obs["op_behaviour_basin"].astype(str).to_numpy()
    bi = a.obs["op_basin_index"].to_numpy(dtype=np.float64)
    rr = a.obs["op_tangent_r"].to_numpy(dtype=np.int32)
    ns = a.obs["op_n_soft"].to_numpy(dtype=np.int32)

    rows = []
    for cl in sorted(set(cluster)):
        m = cluster == cl
        n = int(m.sum())
        c = pd.Series(beh[m]).value_counts()
        bi_cl = bi[m]
        bi_cl = bi_cl[np.isfinite(bi_cl)]
        rr_cl = rr[m]
        ns_cl = ns[m]
        rec = {
            "cluster": cl, "n": n,
            "is_resting": cl in set(RESTING_TYPES),
        }
        for k in ("holder", "fork", "drift"):
            rec[k] = int(c.get(k, 0))
        rec["frac_holder"] = rec["holder"] / n if n else 0.0
        rec["frac_fork"] = rec["fork"] / n if n else 0.0
        rec["basin_index_median"] = float(np.median(bi_cl)) if bi_cl.size else float("nan")
        rec["basin_index_mean"] = float(np.mean(bi_cl)) if bi_cl.size else float("nan")
        rec["r_median"] = int(np.median(rr_cl)) if rr_cl.size else 0
        rec["n_soft_median"] = int(np.median(ns_cl)) if ns_cl.size else 0
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "behaviour_by_cluster.csv", index=False)
    print("\n[pbmc-mag] BEHAVIOUR BY CLUSTER (magnitude rule):")
    print(df.to_string(index=False))

    # Resting summary.
    r = df[df["is_resting"]]
    n_rest = int(r["n"].sum())
    n_hold = int(r["holder"].sum())
    print(f"\n[pbmc-mag]   resting: {n_hold}/{n_rest} holder "
          f"({n_hold/n_rest:.2%})")

    gate = {
        "classifier": "magnitude_rule_tangent_restricted (opspace.classify_basin_latent, post-5e7ad2f)",
        "d_latent": int(args.d_latent),
        "oracle_tol": tol,
        "symmetry": sym,
        "n_resting_cells": n_rest,
        "n_resting_holders": n_hold,
        "frac_resting_holder": n_hold / n_rest if n_rest else 0.0,
        "basin_index_median_resting": float(df[df["is_resting"]]["basin_index_median"].median()),
    }
    (OUT / "gate.json").write_text(json.dumps(gate, indent=2, default=str))
    print("\n[pbmc-mag] GATE:")
    print(json.dumps(gate, indent=2, default=str))
    print(f"[pbmc-mag] outputs → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
