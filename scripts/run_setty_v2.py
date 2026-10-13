"""v2 Phase C field on the whole Setty CD34+ bone marrow dataset.

No branch split (v1 preprint used per-Palantir-branch slicing;
v2 fits one field on all 5780 cells). HVG → 2000 + PCA(10) +
Palantir pseudotime (already in obs). The v2 pipeline:

  * ``fit_latent_field`` (DSM-residualised drift, no clock
    calibration — Setty is a snapshot without labelling), autodiff
    Jacobian;
  * Phase C anchor fit (``max_live=200``) on the full HVG set;
  * ``opspace.compute`` for behaviour counts, archetype
    responders / drivers_left tables.

Report sections (CSVs under benchmarks/v2/setty_cd34/):

  * behaviour counts by cluster (holder vs fork vs ...) — the
    terminal-lineage holder check.
  * behaviour at the ery/myelo/lymphoid decision region
    (HSC_1 + HSC_2 + Precursors).
  * GATA1 / SPI1 ranks per archetype (both tables).
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

from scjdo2 import io as scjdo2_io
from scjdo2 import opspace
from scjdo2.data import load_setty_cd34, PROGENITOR_CLUSTERS, TERMINAL_CLUSTERS
from scjdo2.drift import DriftConfig
from scjdo2.pipeline import fit_latent_field, fit_rate_and_gene_field


REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "benchmarks" / "v2" / "setty_cd34"
MUST_KEEP = ("GATA1", "SPI1")


def _hvg(adata, n_top_genes: int):
    tmp = adata.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=n_top_genes, flavor="seurat")
    hv = tmp.var["highly_variable"].to_numpy()
    for g in MUST_KEEP:
        if g in adata.var_names:
            hv[adata.var_names.get_loc(g)] = True
    kept = adata[:, hv].copy()
    kept.layers["counts"] = kept.X.copy()
    return kept


def _pca(adata, n_comps):
    tmp = adata.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.scale(tmp, max_value=10)
    sc.tl.pca(tmp, n_comps=n_comps)
    adata.obsm["X_pca"] = tmp.obsm["X_pca"].copy()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-pcs", type=int, default=10)
    p.add_argument("--n-top-genes", type=int, default=2000)
    p.add_argument("--n-epochs", type=int, default=800)
    p.add_argument("--rate-epochs", type=int, default=150)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--n-anchors", type=int, default=300)
    p.add_argument("--anchor-max-live", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print("[setty] loading")
    a = load_setty_cd34()
    print(f"[setty]   shape={a.shape}  clusters={sorted(set(a.obs['cluster']))}")
    print(f"[setty]   pseudotime: min={a.obs['pseudotime'].min():.3f} "
          f"max={a.obs['pseudotime'].max():.3f}")

    print(f"[setty] HVG → top-{args.n_top_genes}")
    a = _hvg(a, args.n_top_genes)
    print(f"[setty]   kept {a.n_vars} genes (GATA1 in={('GATA1' in a.var_names)}  "
          f"SPI1 in={('SPI1' in a.var_names)})")

    print(f"[setty] PCA(n={args.n_pcs})")
    _pca(a, args.n_pcs)

    print(f"[setty] drift fit (n_epochs={args.n_epochs}, DSM — Setty is a snapshot)")
    t0 = time.monotonic()
    fit_latent_field(
        a, use_rep="X_pca", time_key="pseudotime",
        drift_config=DriftConfig(
            hidden=96, depth=3, n_epochs=args.n_epochs,
            seed=args.seed, use_spectral_norm=False, batch_size=256,
        ),
    )
    print(f"[setty]   drift+latent J in {time.monotonic()-t0:.1f}s")

    print(f"[setty] Phase C (anchor max_live={args.anchor_max_live})")
    t0 = time.monotonic()
    fit_rate_and_gene_field(
        a, use_rep="X_pca", n_epochs_rate=args.rate_epochs,
        k=args.k, cv_sample_cells=20, seed=args.seed,
        anchor_fit=True, n_anchors=args.n_anchors, anchor_top_k=10,
        anchor_max_live=args.anchor_max_live,
    )
    print(f"[setty]   Phase C in {time.monotonic()-t0:.1f}s")

    print("[setty] opspace.compute")
    opspace.compute(a)

    # Behaviour counts by cluster.
    beh = a.obs["op_behaviour"].astype(str).to_numpy()
    rows = []
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
        rows.append(rec)
    bc = pd.DataFrame(rows)
    bc.to_csv(OUT / "behaviour_by_cluster.csv", index=False)
    print("\n[setty] BEHAVIOUR BY CLUSTER:")
    print(bc.to_string(index=False))

    # Decision-region (HSC_1 + HSC_2 + Precursors) breakdown.
    decision_mask = a.obs["is_progenitor"].to_numpy()
    dec_c = pd.Series(beh[decision_mask]).value_counts()
    dec_rec = {"region": "ery–myelo–lymphoid_decision",
               "n": int(decision_mask.sum())}
    for k in ("fork", "holder", "drift", "damped_oscillator",
              "growing_oscillator"):
        dec_rec[k] = int(dec_c.get(k, 0))
    pd.DataFrame([dec_rec]).to_csv(OUT / "behaviour_at_decision.csv", index=False)
    print("\n[setty] BEHAVIOUR AT DECISION REGION (progenitors):")
    print(pd.DataFrame([dec_rec]).to_string(index=False))

    # Archetype driver tables for GATA1 / SPI1.
    arch = a.uns["scjdo2"].get("archetypes") or {}
    rows_r = []
    for ak, info in arch.items():
        for table_name in ("responders", "drivers_left"):
            df = info.get(table_name)
            if df is None:
                continue
            for g in MUST_KEEP:
                hit = df.index[df["gene"].astype(str) == g].tolist()
                if not hit:
                    continue
                i = int(hit[0])
                r = df.loc[i].to_dict()
                rows_r.append({
                    "archetype": int(ak), "behaviour": info["behaviour"],
                    "lead_re": float(info["eigenvalues"][0].real),
                    "n_cells": int(info["n_cells"]),
                    "table": table_name, "gene": g, "rank": i,
                    "sensitivity": float(r.get("sensitivity", float("nan"))),
                    "magnitude": float(r["magnitude"]),
                    "weight": float(r["weight"]),
                    "fisher": float(r["fisher"]),
                    "snr": float(r.get("snr", float("nan"))),
                    "reason": str(r["reason"]),
                })
    drv = pd.DataFrame(rows_r)
    drv.to_csv(OUT / "archetype_drivers_GATA1_SPI1.csv", index=False)
    print("\n[setty] GATA1 / SPI1 ARCHETYPE RANKS:")
    print(drv.to_string(index=False))

    # Save adata.
    try:
        scjdo2_io.save(a, OUT)
    except Exception as err:
        print(f"[setty] save skipped: {err}")
    print(f"\n[setty] outputs under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
