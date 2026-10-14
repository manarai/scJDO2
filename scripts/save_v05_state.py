"""Save v0.5-basin-lock state for PBMC + equilibrated pitchfork.

Writes two self-contained AnnData objects (``adata.h5ad``) with:
  * ``obsm``: embedding (``X_scVI`` for PBMC, ``X_pca`` for pitchfork)
    + tSNE/UMAP where present.
  * ``obs``: cluster labels + classifier output (``op_behaviour_basin``,
    ``op_basin_index``, plus ``lead_re``/``lead_im`` for figure Panel B).
  * ``uns["scjdo2"]``: the latent J tensor + a small meta dict.
  * ``uns["v05_meta"]``: tag + commit + rule name + oracle_tol.

Notebooks load these h5ads and render without refitting.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

from scjdo2.data import (
    CYCLE_GENES,
    RESTING_TYPES,
    generate_pitchfork_equilibrated,
    load_pbmc3k,
)
from scjdo2.energy_drift import EnergyDriftAdapter, EnergyDriftConfig
from scjdo2.opspace import classify_basin_latent, invariants, behaviour, spectrum


REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "benchmarks" / "v0.5_lock"


def _oracle_tol(z: np.ndarray, cluster: np.ndarray, resting) -> float:
    leads = []
    for cl in sorted(set(cluster)):
        if resting and cl not in set(resting):
            continue
        m = cluster == cl
        if m.sum() < 10:
            continue
        Sigma = np.cov(z[m], rowvar=False, bias=False) + 1e-4 * np.eye(z.shape[1])
        ev = np.linalg.eigvalsh(-np.linalg.inv(Sigma))
        leads.append(abs(float(ev.max())))
    return 0.1 * float(np.median(leads)) if leads else 0.1


def _fit_scvi(adata, n_latent: int, n_epochs: int, seed: int = 0):
    import scvi
    scvi.settings.seed = seed
    a = adata.copy()
    a.layers["counts"] = a.X.copy()
    scvi.model.SCVI.setup_anndata(a, layer="counts")
    m = scvi.model.SCVI(a, n_latent=int(n_latent))
    m.train(max_epochs=int(n_epochs), check_val_every_n_epoch=10,
            early_stopping=False, enable_progress_bar=False)
    return np.asarray(m.get_latent_representation(a)).astype(np.float32)


def save_pbmc(seed: int = 0, d_latent: int = 15, scvi_epochs: int = 180,
               drift_epochs: int = 1500, n_top_genes: int = 3000) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[save-pbmc] load")
    a = load_pbmc3k()
    sc.pp.filter_genes(a, min_cells=3)
    tmp = a.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=n_top_genes, flavor="seurat")
    hv = tmp.var["highly_variable"].to_numpy()
    for g in list(CYCLE_GENES) + ["CD3D","CD4","CD8A","CD8B","MS4A1","CD79A",
                                   "NKG7","GNLY","CD14","FCGR3A","CST3","LYZ"]:
        if g in a.var_names:
            hv[a.var_names.get_loc(g)] = True
    a = a[:, hv].copy()

    print(f"[save-pbmc] scVI d_latent={d_latent}")
    t0 = time.monotonic()
    z = _fit_scvi(a, d_latent, scvi_epochs, seed)
    a.obsm["X_scVI"] = z
    print(f"[save-pbmc]   scVI {time.monotonic()-t0:.1f}s")

    print(f"[save-pbmc] ESM energy drift")
    t0 = time.monotonic()
    drift = EnergyDriftAdapter(dim=z.shape[1], config=EnergyDriftConfig(
        hidden=96, depth=3, n_epochs=drift_epochs, seed=seed,
        batch_size=256, loss="esm", lr=5e-4, beta=1.0,
        esm_laplacian="exact" if z.shape[1] <= 20 else "hutchinson",
    )).fit(z, verbose=False)
    sym = drift.symmetry_check(z, n_cells=100, seed=seed)
    J = drift.jacobian_autodiff(z)
    print(f"[save-pbmc]   drift+J {time.monotonic()-t0:.1f}s  "
          f"SYM median {sym['median_asym']:.2e}")

    cluster = a.obs["cluster"].astype(str).to_numpy()
    tol = _oracle_tol(z, cluster, RESTING_TYPES)
    a.uns.setdefault("scjdo2", {})["J"] = J.astype(np.float32)
    a.uns["scjdo2"]["J_latent"] = J.astype(np.float32)
    a.uns["scjdo2"]["meta"] = {"use_rep": "X_scVI", "clock_calibrated": False,
                                "jacobian_method": "autodiff",
                                "drift_param": "energy_scalar_mlp",
                                "loss": "esm", "beta": 1.0}

    print("[save-pbmc] opspace.spectrum / invariants / behaviour (for Panel B)")
    spectrum(a, n_eig=5, fisher_mask=False, smooth_k=0)
    invariants(a, tol=tol)
    behaviour(a, tol=tol)

    print("[save-pbmc] classify_basin_latent (lead_eig rule, locked)")
    classify_basin_latent(a, tol=tol, rule="lead_eig", use_rep="X_scVI")

    # Carry over tSNE/UMAP from pbmc3k_processed if available (already done
    # in load_pbmc3k) for Panel A embedding.
    for key in ("X_umap", "X_tsne"):
        if key in a.obsm:
            print(f"[save-pbmc]   embedding present: {key} shape {a.obsm[key].shape}")

    a.uns["v05_meta"] = {
        "tag": "v0.5-basin-lock",
        "rule": "lead_eig",
        "oracle_tol_neg": tol,
        "symmetry_median": sym["median_asym"],
        "d_latent": int(d_latent),
    }
    out_h5 = OUT / "pbmc_v05.h5ad"
    a.write_h5ad(out_h5)
    print(f"[save-pbmc] wrote {out_h5}  shape={a.shape}")
    return out_h5


def save_pitchfork(seed: int = 0, tau_crit: float = 0.3,
                    n_cells: int = 1200, n_alpha: int = 20,
                    drift_epochs: int = 1500) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[save-pitch] generate τ_crit={tau_crit}")
    a = generate_pitchfork_equilibrated(
        tau_crit=tau_crit, seed=seed,
        n_cells=n_cells, n_alpha=n_alpha,
        n_genes=200, n_latent=20, noise_sigma=0.3,
        T_burn=20.0, T_sample=5.0, dt=0.02, sigma_sde=0.20,
        init_box=2.5,
    )
    z = np.asarray(a.obsm["X_pca"], dtype=np.float32)
    a.obs["cluster"] = pd.Categorical(
        [f"a{int(round(p*100))}" for p in a.obs["pseudotime"].to_numpy()])

    print(f"[save-pitch] ESM energy drift ({drift_epochs} ep)")
    t0 = time.monotonic()
    drift = EnergyDriftAdapter(dim=z.shape[1], config=EnergyDriftConfig(
        hidden=128, depth=3, n_epochs=drift_epochs, seed=seed,
        batch_size=256, loss="esm", lr=5e-4, beta=1.0,
        esm_laplacian="exact" if z.shape[1] <= 20 else "hutchinson",
    )).fit(z, verbose=False)
    sym = drift.symmetry_check(z, n_cells=100, seed=seed)
    J = drift.jacobian_autodiff(z)
    print(f"[save-pitch]   drift+J {time.monotonic()-t0:.1f}s  "
          f"SYM median {sym['median_asym']:.2e}")

    cluster = a.obs["cluster"].astype(str).to_numpy()
    tol = _oracle_tol(z, cluster, resting=None)
    a.uns.setdefault("scjdo2", {})["J"] = J.astype(np.float32)
    a.uns["scjdo2"]["J_latent"] = J.astype(np.float32)
    a.uns["scjdo2"]["meta"] = {"use_rep": "X_pca", "clock_calibrated": False,
                                "jacobian_method": "autodiff",
                                "drift_param": "energy_scalar_mlp",
                                "loss": "esm", "beta": 1.0}

    print("[save-pitch] opspace spectrum / invariants / behaviour")
    spectrum(a, n_eig=5, fisher_mask=False, smooth_k=0)
    invariants(a, tol=tol)
    behaviour(a, tol=tol)

    print("[save-pitch] classify_basin_latent (lead_eig)")
    classify_basin_latent(a, tol=tol, rule="lead_eig", use_rep="X_pca")

    a.uns["v05_meta"] = {
        "tag": "v0.5-basin-lock",
        "rule": "lead_eig",
        "oracle_tol_neg": tol,
        "symmetry_median": sym["median_asym"],
        "tau_crit": float(tau_crit),
    }
    out_h5 = OUT / "pitchfork_v05.h5ad"
    a.write_h5ad(out_h5)
    print(f"[save-pitch] wrote {out_h5}  shape={a.shape}")
    return out_h5


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--targets", nargs="+",
                   choices=["pbmc", "pitchfork"],
                   default=["pbmc", "pitchfork"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if "pbmc" in args.targets:
        save_pbmc(seed=args.seed)
    if "pitchfork" in args.targets:
        save_pitchfork(seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
