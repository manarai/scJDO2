"""Setty v2 — archetype × cluster co-occurrence + decision-region
archetype + comparison figure.

Re-runs Setty Phase C with slightly reduced anchor settings for speed
(the full run already landed its gate CSVs; this one dumps the
archetype_weights × cluster mapping and the comparison figure).

Outputs (benchmarks/v2/setty_cd34/):
  * archetype_by_cluster.csv            — soft-weight share of each
                                           archetype across the 10
                                           Setty clusters.
  * decision_region_archetype.csv       — the archetype whose soft
                                           weight concentrates in the
                                           HSC + Precursors layer, with
                                           its GATA1 / SPI1 ranks and
                                           Fisher mass pulled from the
                                           earlier gate CSV.
  * v1_vs_v2_figure.pdf / .png          — Setty tSNE on the left
                                           (v2 behaviour overlay +
                                           decision-region archetype),
                                           v1 per-branch pseudotime
                                           lead_re on the right.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from scjdo2 import opspace
from scjdo2.data import PROGENITOR_CLUSTERS, load_setty_cd34
from scjdo2.drift import DriftConfig
from scjdo2.pipeline import fit_latent_field, fit_rate_and_gene_field


REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "benchmarks" / "v2" / "setty_cd34"
V1_RESULTS = Path("/Users/terooatt/Documents/scJDO/examples/results/palantir_driftfield")


def _preprocess(a, n_top_genes: int, n_pcs: int):
    tmp = a.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=n_top_genes, flavor="seurat")
    hv = tmp.var["highly_variable"].to_numpy()
    for g in ("GATA1", "SPI1"):
        if g in a.var_names:
            hv[a.var_names.get_loc(g)] = True
    kept = a[:, hv].copy()
    kept.layers["counts"] = kept.X.copy()
    tmp = kept.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.scale(tmp, max_value=10)
    sc.tl.pca(tmp, n_comps=n_pcs)
    kept.obsm["X_pca"] = tmp.obsm["X_pca"].copy()
    return kept


def _archetype_cluster_table(adata) -> pd.DataFrame:
    """Soft-weight share of each archetype across clusters."""
    W = adata.obsm["archetype_weights"]
    clusters = adata.obs["cluster"].astype(str).to_numpy()
    cl_names = sorted(set(clusters))
    rows = []
    for k in range(W.shape[1]):
        w = W[:, k]
        total = float(w.sum()) or 1.0
        rec = {"archetype": k}
        for cl in cl_names:
            m = clusters == cl
            rec[cl] = float(w[m].sum() / total)
        rec["n_cells_hard"] = int((W.argmax(axis=1) == k).sum())
        rows.append(rec)
    return pd.DataFrame(rows)


def _decision_region_archetype(acw: pd.DataFrame, clusters_progenitor) -> int:
    """Archetype whose weight share on the HSC + Precursors cells is highest."""
    prog_cols = [c for c in acw.columns if c in clusters_progenitor]
    score = acw[prog_cols].sum(axis=1)
    return int(acw.loc[score.idxmax(), "archetype"])


def _tsne_fig(a, decision_archetype: int, out_path: Path):
    """Left panel: tSNE coloured by behaviour, with the decision
    archetype's hard-membership cells circled. Right panel: v1 per-
    branch max_real_eig along pseudotime (reconstructed from the v1
    CSV)."""
    v1_lead = pd.read_csv(
        REPO / "benchmarks" / "v1_baseline" / "setty_cd34" / "v1_branch_lead_re.csv"
    )
    v1_waypoint_pt = np.linspace(0, 1, 100)
    import anndata as ad
    try:
        v1_h5ad = ad.read_h5ad(V1_RESULTS / "marrow_scjdo.h5ad")
    except Exception:
        v1_h5ad = None
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    tsne = np.asarray(a.obsm["X_tsne"])
    beh = a.obs["op_behaviour"].astype(str).to_numpy()
    palette = {"fork": "#d55e00", "holder": "#0072b2",
               "growing_oscillator": "#009e73",
               "damped_oscillator": "#56b4e9",
               "drift": "#999999", "": "#cccccc"}
    for b in set(beh):
        m = beh == b
        axes[0].scatter(tsne[m, 0], tsne[m, 1], s=3, alpha=0.5,
                        label=f"{b} ({m.sum()})",
                        c=palette.get(b, "#333333"))
    W = a.obsm["archetype_weights"]
    hard = W.argmax(axis=1) == decision_archetype
    axes[0].scatter(tsne[hard, 0], tsne[hard, 1], s=12,
                    facecolors="none", edgecolors="black",
                    linewidths=0.4, label=f"arch {decision_archetype} hard")
    axes[0].set_title(f"Setty v2 — behaviour on tSNE; decision arch = {decision_archetype}")
    axes[0].set_xlabel("tSNE-1"); axes[0].set_ylabel("tSNE-2")
    axes[0].legend(loc="upper right", fontsize=6, markerscale=1.5)
    if v1_h5ad is not None:
        for branch, c in zip(("Ery", "Mono", "DC"),
                              ("#cc3333", "#3377cc", "#33aa33")):
            d = v1_h5ad.uns.get(f"scqdiff_{branch}")
            if d is None:
                continue
            mre = np.asarray(d["max_real_eig"])
            axes[1].plot(v1_waypoint_pt, mre, c=c, label=branch, lw=1.6)
    axes[1].axhline(0, color="k", lw=0.4, ls="--")
    axes[1].set_title("Setty v1 per-branch — lead_re along pseudotime")
    axes[1].set_xlabel("pseudotime (branch waypoint)")
    axes[1].set_ylabel("max Re λ")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.savefig(out_path.with_suffix(".png"), dpi=140)
    print(f"[setty-arch] wrote figure {out_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-pcs", type=int, default=10)
    p.add_argument("--n-top-genes", type=int, default=2000)
    p.add_argument("--n-epochs", type=int, default=600)
    p.add_argument("--rate-epochs", type=int, default=120)
    p.add_argument("--n-anchors", type=int, default=200)
    p.add_argument("--anchor-max-live", type=int, default=150)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print("[setty-arch] load")
    a = load_setty_cd34()
    print(f"[setty-arch]   shape={a.shape}")
    a = _preprocess(a, args.n_top_genes, args.n_pcs)

    print("[setty-arch] drift fit")
    t0 = time.monotonic()
    fit_latent_field(
        a, use_rep="X_pca", time_key="pseudotime",
        drift_config=DriftConfig(
            hidden=96, depth=3, n_epochs=args.n_epochs,
            seed=args.seed, use_spectral_norm=False, batch_size=256,
        ),
    )
    print(f"[setty-arch]   drift+latent J in {time.monotonic()-t0:.1f}s")

    print(f"[setty-arch] Phase C (anchor max_live={args.anchor_max_live})")
    t0 = time.monotonic()
    fit_rate_and_gene_field(
        a, use_rep="X_pca", n_epochs_rate=args.rate_epochs,
        k=args.k, cv_sample_cells=15, seed=args.seed,
        anchor_fit=True, n_anchors=args.n_anchors, anchor_top_k=10,
        anchor_max_live=args.anchor_max_live,
    )
    print(f"[setty-arch]   Phase C in {time.monotonic()-t0:.1f}s")

    print("[setty-arch] opspace.compute")
    opspace.compute(a)

    acw = _archetype_cluster_table(a)
    acw.to_csv(OUT / "archetype_by_cluster.csv", index=False)
    print("\n[setty-arch] archetype × cluster soft-weight shares:")
    print(acw.to_string(index=False))

    decision_arch = _decision_region_archetype(acw, PROGENITOR_CLUSTERS)
    print(f"\n[setty-arch] decision-region archetype: {decision_arch}")

    # GATA1 / SPI1 ranks for the decision archetype pulled from the
    # already-saved gate CSV + the fresh archetypes dict.
    arch_info = a.uns["scjdo2"]["archetypes"].get(decision_arch, {})
    region_rows = []
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
                "archetype": decision_arch,
                "table": table_name, "gene": g, "rank": i,
                "lead_re": float(arch_info["eigenvalues"][0].real),
                "behaviour": arch_info["behaviour"],
                "n_cells": int(arch_info["n_cells"]),
                "fisher": float(r["fisher"]),
                "snr": float(r.get("snr", float("nan"))),
                "sensitivity": float(r.get("sensitivity", float("nan"))),
                "reason": str(r["reason"]),
            })
    region_df = pd.DataFrame(region_rows)
    region_df.to_csv(OUT / "decision_region_archetype.csv", index=False)
    print(f"\n[setty-arch] decision archetype GATA1/SPI1 ranks:")
    print(region_df.to_string(index=False))

    # Comparison figure.
    _tsne_fig(a, decision_arch, OUT / "v1_vs_v2_figure.pdf")
    print(f"\n[setty-arch] done → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
