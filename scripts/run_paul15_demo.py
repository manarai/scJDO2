"""Run scJDO2 opspace/plot end-to-end on the Paul15 v1 baseline.

Reads the v1 baseline AnnData (which already has ``X_pca`` from scVI/PCA and
``pseudotime`` from DPT), runs scJDO2's pipeline + opspace, and writes the
two publication figures under ``benchmarks/figures/paul15/``.

Usage:
    python scripts/run_paul15_demo.py --n-epochs 1500

By default we use 1500 drift epochs for a quick demo pass; the Paul15 v1
baseline was fit at 5000.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import anndata as ad
import numpy as np

from scjdo2 import opspace, plot
from scjdo2.drift import DriftConfig
from scjdo2.field import FieldConfig
from scjdo2.pipeline import fit_latent_field


REPO = Path(__file__).resolve().parent.parent
V1_BASELINE = REPO / "benchmarks" / "v1_baseline" / "paul15"
OUT_DIR = REPO / "benchmarks" / "figures" / "paul15"
RUN_DIR = REPO / "benchmarks" / "v2" / "paul15"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-epochs", type=int, default=1500)
    p.add_argument("--k", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(V1_BASELINE / "adata.h5ad")
    print(f"[paul15 demo] loaded adata: {adata.shape}")
    print(f"  obsm keys: {list(adata.obsm.keys())}")
    for key in ("X_pca", "X_umap"):
        if key in adata.obsm:
            print(f"  {key}: {adata.obsm[key].shape}")
    print(f"  pseudotime range: [{adata.obs['pseudotime'].min():.3f}, "
          f"{adata.obs['pseudotime'].max():.3f}]")

    # Drop v1's uns["scjdo"] block so our writes to uns["scjdo2"] don't collide.
    if "scjdo" in adata.uns:
        v1_block = adata.uns.pop("scjdo")
        print(f"  stashed v1 uns['scjdo'] with keys: {list(v1_block.keys())[:5]} …")
        adata.uns["scjdo_v1_present"] = True

    t0 = time.monotonic()
    fit_latent_field(
        adata,
        use_rep="X_pca",
        time_key="pseudotime",
        drift_config=DriftConfig(
            hidden=128, depth=4, n_epochs=args.n_epochs, seed=args.seed,
            use_spectral_norm=False, batch_size=256,
        ),
        field_config=FieldConfig(k=args.k, ridge=1e-3, seed=args.seed),
        verbose=True,
    )
    dt_fit = time.monotonic() - t0
    print(f"[paul15 demo] fit_latent_field complete in {dt_fit:.1f} s")

    opspace.spectrum(adata, n_eig=5, fisher_mask=False)
    opspace.invariants(adata)
    opspace.behaviour(adata)
    opspace.field_gradient(adata, k=15)

    # Lineage paths — pick start cells near pseudotime=0 as the trajectory origin.
    t = adata.obs["pseudotime"].to_numpy()
    origin = np.argsort(t)[:5]
    paths = opspace.lineage_path(adata, start_cells=origin.tolist(), n_steps=120)

    # Export tables + save processed AnnData.
    opspace.export_tables(adata, RUN_DIR)
    adata.write_h5ad(RUN_DIR / "adata.h5ad")

    # Figures
    print("[paul15 demo] rendering figure_operator_space")
    with plot.style():
        plot.figure_operator_space(
            adata, lineages=paths, use_rep="X_umap" if "X_umap" in adata.obsm else "X_pca",
            out=OUT_DIR / "operator_space",
        )
    print("[paul15 demo] rendering figure_v1_vs_v2")
    with plot.style():
        plot.figure_v1_vs_v2(
            adata, v1_results=V1_BASELINE,
            use_rep="X_umap" if "X_umap" in adata.obsm else "X_pca",
            out=OUT_DIR / "v1_vs_v2",
        )

    # Summary
    beh = adata.obs["op_behaviour"].value_counts()
    print("\n[paul15 demo] behaviour counts:")
    for b, n in beh.items():
        print(f"  {b or '(invalid)':20s}  {n}")
    print(f"\n[paul15 demo] figures:")
    for name in ("operator_space.pdf", "v1_vs_v2.pdf"):
        p = OUT_DIR / name
        if p.exists():
            print(f"  {p}  ({p.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
