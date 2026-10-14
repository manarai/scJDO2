"""Rerun the basin classifier (SUMMARY §5.5 lock) across every real-
data benchmark + the equilibrated pitchfork.

For each dataset:
  * load the AnnData / build the latent (reusing each dataset's
    established embedding),
  * fit the ESM energy drift,
  * classify with ``opspace.classify_basin_latent``,
  * report per-cluster behaviour counts + overall holder fraction,
  * compare to the biological expectation (progenitor compartments
    should NOT be majority holder; the PBMC homeostatic compartment
    IS, which is §5.5).

Outputs under ``benchmarks/v2/basin_rerun/<dataset>/``:
  * behaviour_by_cluster.csv
  * meta.json (oracle_tol, symmetry, timing)

Does NOT refit the gene-space Phase C. The user's gate at this stage
is latent-only; the gene-space archetype work resumes under §7c
after the lock.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from basin_classify_latent import classify, fit_energy_drift     # noqa: E402

from scjdo2.data import (
    PROGENITOR_CLUSTERS,
    TERMINAL_CLUSTERS,
    generate_pitchfork_equilibrated,
    load_larry,
    load_setty_cd34,
)


OUT = REPO / "benchmarks" / "v2" / "basin_rerun"


# -----------------------------------------------------------------
# Pitchfork
# -----------------------------------------------------------------


def run_pitchfork(n_cells: int = 1200, n_alpha: int = 20, seed: int = 0,
                  drift_epochs: int = 1500) -> dict:
    """Equilibrated pitchfork at τ=0.3 (α-slice with saddle in the
    middle). "Resting-like" cells = those far from the pitchfork
    saddle in pseudotime, "fork-like" = cells near the saddle."""
    t0 = time.monotonic()
    adata = generate_pitchfork_equilibrated(
        tau_crit=0.3, seed=seed, n_cells=n_cells, n_alpha=n_alpha,
        n_genes=200, n_latent=20, noise_sigma=0.3,
        T_burn=20.0, T_sample=5.0, dt=0.02, sigma_sde=0.20,
        init_box=2.5,
    )
    # Use α-slice as a surrogate cluster: 20 slices → 20 clusters.
    alpha = adata.obs["alpha"].to_numpy()
    adata.obs["cluster"] = pd.Categorical(
        [f"a{int(round(a * 100))}" for a in alpha])
    # Resting = α far below saddle (deep in basin, α ≈ 0); fork = α near
    # saddle. We just mark everything as "resting" for the gate — the
    # pitchfork gate is more about dynamic ground truth than basin fraction.
    drift = fit_energy_drift(np.asarray(adata.obsm["X_pca"]),
                              n_epochs=drift_epochs, seed=seed)
    out = classify(adata, "X_pca", drift)
    out["dataset"] = "pitchfork_equilibrated_tau0.3"
    out["seconds"] = time.monotonic() - t0
    return out


# -----------------------------------------------------------------
# Setty
# -----------------------------------------------------------------


def run_setty(seed: int = 0, drift_epochs: int = 1500,
              n_top_genes: int = 2000, n_pcs: int = 10) -> dict:
    t0 = time.monotonic()
    a = load_setty_cd34()
    # HVG + PCA (match the Setty primary comparison).
    tmp = a.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=int(n_top_genes),
                                flavor="seurat")
    hv = tmp.var["highly_variable"].to_numpy()
    for g in ("GATA1", "SPI1"):
        if g in a.var_names:
            hv[a.var_names.get_loc(g)] = True
    kept = a[:, hv].copy()
    tmp2 = kept.copy()
    sc.pp.normalize_total(tmp2, target_sum=1e4)
    sc.pp.log1p(tmp2)
    sc.pp.scale(tmp2, max_value=10)
    sc.tl.pca(tmp2, n_comps=int(n_pcs))
    kept.obsm["X_pca"] = tmp2.obsm["X_pca"].copy()
    # Use existing cluster column.
    kept.obs["cluster"] = a.obs["cluster"].astype(str)
    drift = fit_energy_drift(np.asarray(kept.obsm["X_pca"]),
                              n_epochs=drift_epochs, seed=seed)
    out = classify(kept, "X_pca", drift, resting_types=())  # progenitor compartment
    out["dataset"] = "setty_cd34"
    out["seconds"] = time.monotonic() - t0
    return out


# -----------------------------------------------------------------
# Paul15
# -----------------------------------------------------------------


def run_paul15(seed: int = 0, drift_epochs: int = 1500,
               n_top_genes: int = 2000, n_pcs: int = 10) -> dict:
    t0 = time.monotonic()
    adata = sc.datasets.paul15()
    adata.X = adata.X.astype(np.float32)
    tmp = adata.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=int(n_top_genes), flavor="seurat")
    hv = tmp.var["highly_variable"].to_numpy()
    for g in ("Gata1", "Sfpi1"):
        if g in adata.var_names:
            hv[adata.var_names.get_loc(g)] = True
    kept = adata[:, hv].copy()
    tmp2 = kept.copy()
    sc.pp.normalize_total(tmp2, target_sum=1e4)
    sc.pp.log1p(tmp2)
    sc.pp.scale(tmp2, max_value=10)
    sc.tl.pca(tmp2, n_comps=int(n_pcs))
    kept.obsm["X_pca"] = tmp2.obsm["X_pca"].copy()
    kept.obs["cluster"] = adata.obs["paul15_clusters"].astype(str)
    drift = fit_energy_drift(np.asarray(kept.obsm["X_pca"]),
                              n_epochs=drift_epochs, seed=seed)
    out = classify(kept, "X_pca", drift, resting_types=())
    out["dataset"] = "paul15"
    out["seconds"] = time.monotonic() - t0
    return out


# -----------------------------------------------------------------
# LARRY
# -----------------------------------------------------------------


def run_larry(n_cells: int = 10000, seed: int = 0, drift_epochs: int = 1500,
              n_top_genes: int = 2000, n_pcs: int = 10) -> dict:
    t0 = time.monotonic()
    a = load_larry()
    # Clone-stratified subsample for speed (reuse the existing script's
    # logic in spirit: just random subsample here).
    rng = np.random.default_rng(seed)
    if a.n_obs > n_cells:
        idx = rng.choice(a.n_obs, size=n_cells, replace=False)
        a = a[idx].copy()
    tmp = a.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=int(n_top_genes), flavor="seurat")
    hv = tmp.var["highly_variable"].to_numpy()
    kept = a[:, hv].copy()
    tmp2 = kept.copy()
    sc.pp.normalize_total(tmp2, target_sum=1e4)
    sc.pp.log1p(tmp2)
    sc.pp.scale(tmp2, max_value=10)
    sc.tl.pca(tmp2, n_comps=int(n_pcs))
    kept.obsm["X_pca"] = tmp2.obsm["X_pca"].copy()
    # Use the fate_state column as cluster label (produced by load_larry).
    for key in ("Cell type annotation", "cell_type", "annotation", "state"):
        if key in a.obs:
            kept.obs["cluster"] = a.obs[key].astype(str).to_numpy()
            break
    else:
        kept.obs["cluster"] = "unknown"
    drift = fit_energy_drift(np.asarray(kept.obsm["X_pca"]),
                              n_epochs=drift_epochs, seed=seed)
    out = classify(kept, "X_pca", drift, resting_types=())
    out["dataset"] = "larry_inVitro"
    out["seconds"] = time.monotonic() - t0
    return out


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------


DATASETS = {
    "pitchfork": run_pitchfork,
    "setty":     run_setty,
    "paul15":    run_paul15,
    "larry":     run_larry,
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+",
                   choices=list(DATASETS.keys()),
                   default=list(DATASETS.keys()))
    p.add_argument("--drift-epochs", type=int, default=1500)
    p.add_argument("--larry-cells", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for name in args.datasets:
        print(f"\n[rerun] === {name} ===")
        fn = DATASETS[name]
        try:
            if name == "larry":
                out = fn(n_cells=args.larry_cells, seed=args.seed,
                         drift_epochs=args.drift_epochs)
            elif name == "pitchfork":
                out = fn(seed=args.seed, drift_epochs=args.drift_epochs)
            else:
                out = fn(seed=args.seed, drift_epochs=args.drift_epochs)
        except Exception as err:
            print(f"[rerun] {name}: FAILED  {err}")
            all_rows.append({"dataset": name, "status": "fail", "err": str(err)})
            continue
        beh = out["behaviour"]
        total = int(beh["n"].sum())
        holder = int(beh["holder"].sum())
        print(f"[rerun]   oracle_tol = {out['oracle_tol']:.4f}")
        print(f"[rerun]   symmetry median = {out['symmetry']['median_asym']:.3e}")
        print(f"[rerun]   total holders / n = {holder}/{total} ({holder/total:.1%})")
        print(f"[rerun]   fit {out['seconds']:.1f}s")
        print(beh.to_string(index=False))
        dout = OUT / name
        dout.mkdir(parents=True, exist_ok=True)
        beh.to_csv(dout / "behaviour_by_cluster.csv", index=False)
        with open(dout / "meta.json", "w") as f:
            json.dump({"dataset": out["dataset"],
                        "oracle_tol": out["oracle_tol"],
                        "symmetry": out["symmetry"],
                        "seconds": out["seconds"],
                        "n_total": total,
                        "n_holder": holder,
                        "frac_holder": holder / total if total else 0.0,
                        }, f, indent=2, default=str)
        all_rows.append({"dataset": name, "n_total": total,
                         "n_holder": holder, "frac_holder": holder / total,
                         "oracle_tol": out["oracle_tol"],
                         "sym_median_asym": out["symmetry"]["median_asym"]})

    summary = pd.DataFrame(all_rows)
    summary.to_csv(OUT / "summary.csv", index=False)
    print("\n[rerun] SUMMARY across datasets:")
    print(summary.to_string(index=False))
    print(f"[rerun] outputs → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
