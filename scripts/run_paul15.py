"""Residual 2+4 driver — scjdo2 run on Paul15 (HVG subset).

Reads the v1 baseline AnnData (X_pca + pseudotime already in place),
subsets to the top ``--n-top`` highly-variable genes (keeping Gata1
and Sfpi1 — the fork TFs — regardless), writes a staging h5ad, then
runs `scjdo2.pipeline.fit_latent_field` + `fit_rate_and_gene_field` +
`opspace.compute`, saves the run dir, and renders the full figure
set.

Deliberately small by default (500 genes, 400/400 epochs) so the full
pipeline + figures land in a few minutes rather than tens of minutes.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import anndata as ad
import numpy as np
import scanpy as sc

from scjdo2 import io as scjdo2_io
from scjdo2 import opspace
from scjdo2.drift import DriftConfig
from scjdo2.pipeline import fit_latent_field, fit_rate_and_gene_field
from scjdo2.plot import (
    figure_archetype_sheet,
    figure_ledger_summary,
    figure_operator_space,
    figure_v1_vs_v2,
)


REPO = Path(__file__).resolve().parent.parent
V1_BASELINE = REPO / "benchmarks" / "v1_baseline" / "paul15"
OUT_RUN = REPO / "benchmarks" / "v2" / "paul15"
OUT_FIG = REPO / "benchmarks" / "figures" / "paul15"
MUST_KEEP = ["Gata1", "Sfpi1"]


def _subset_hvg(adata, n_top: int):
    """Always compute a top-``n_top`` HVG subset (the v1 baseline marks all
    2000 genes as highly_variable, so the pre-existing mask is useless for
    further subsetting). Falls back to raw variance if scanpy's HVG errors
    on the already-log-normalised matrix."""
    try:
        sc.pp.highly_variable_genes(
            adata.copy(), n_top_genes=n_top, flavor="seurat_v3",
            inplace=False,
        )
        # Compute HVGs out-of-place and copy the mask back.
        hvg = sc.pp.highly_variable_genes(
            adata, n_top_genes=n_top, flavor="seurat_v3", inplace=False,
        )
        hvg_mask = hvg["highly_variable"].to_numpy()
    except Exception as err:
        print(f"[paul15]   scanpy HVG failed ({err}); using raw variance.")
        X = adata.X
        var = np.asarray(X.var(axis=0)).ravel() if not hasattr(X, "toarray") \
            else np.asarray(X.toarray().var(axis=0)).ravel()
        hvg_mask = np.zeros(adata.n_vars, dtype=bool)
        hvg_mask[np.argsort(-var)[: int(n_top)]] = True
    for g in MUST_KEEP:
        if g in adata.var_names:
            hvg_mask[adata.var_names.get_loc(g)] = True
    kept = adata[:, hvg_mask].copy()
    print(f"[paul15]   kept {kept.n_vars} HVGs (incl. {MUST_KEEP})")
    return kept


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-epochs", type=int, default=1500)
    p.add_argument("--rate-epochs", type=int, default=400)
    p.add_argument("--n-top-genes", type=int, default=500)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--cv-sample-cells", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--full-genes", action="store_true",
                   help="skip HVG subsetting; fit rate model + gene-space J "
                        "on every gene (recommended per the R15 driver "
                        "contract — HVGs are for the embedding only)")
    p.add_argument("--anchor-fit", action="store_true",
                   help="use the anchor-based Step 5 variant")
    p.add_argument("--n-anchors", type=int, default=300)
    p.add_argument("--anchor-top-k", type=int, default=10)
    p.add_argument("--anchor-batch", type=int, default=25)
    p.add_argument("--out-run", type=Path, default=OUT_RUN)
    p.add_argument("--out-fig", type=Path, default=OUT_FIG)
    args = p.parse_args()
    out_run = Path(args.out_run)
    out_fig = Path(args.out_fig)

    out_run.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)

    print(f"[paul15] loading v1 baseline from {V1_BASELINE}/adata.h5ad")
    adata = ad.read_h5ad(V1_BASELINE / "adata.h5ad")
    print(f"[paul15]   shape={adata.shape}; obsm={list(adata.obsm.keys())}")

    if args.full_genes:
        print(f"[paul15]   keeping all {adata.n_vars} genes (--full-genes)")
    else:
        adata = _subset_hvg(adata, args.n_top_genes)

    print(f"[paul15] fitting latent drift + field (n_epochs={args.n_epochs})")
    t0 = time.monotonic()
    fit_latent_field(
        adata, use_rep="X_pca", time_key="pseudotime",
        drift_config=DriftConfig(
            hidden=96, depth=3, n_epochs=int(args.n_epochs),
            seed=int(args.seed), use_spectral_norm=False, batch_size=256,
        ),
    )
    print(f"[paul15]   drift+latent J in {time.monotonic() - t0:.1f}s")

    variant = "anchor" if args.anchor_fit else "per_cell"
    print(f"[paul15] fitting NB rate model + gene-space J "
          f"(variant={variant}, rate_epochs={args.rate_epochs}, k={args.k})")
    t0 = time.monotonic()
    fit_rate_and_gene_field(
        adata, use_rep="X_pca",
        n_epochs_rate=int(args.rate_epochs),
        k=int(args.k),
        cv_sample_cells=int(args.cv_sample_cells),
        seed=int(args.seed),
        anchor_fit=bool(args.anchor_fit),
        n_anchors=int(args.n_anchors),
        anchor_top_k=int(args.anchor_top_k),
        anchor_batch=int(args.anchor_batch),
    )
    print(f"[paul15]   Phase C fit in {time.monotonic() - t0:.1f}s")

    print("[paul15] opspace.compute (R15 orchestration)")
    opspace.compute(adata)

    print(f"[paul15] saving run dir to {out_run}")
    saved = scjdo2_io.save(adata, out_run)
    print(f"[paul15]   wrote {saved['h5ad'].name}"
          + (f" + {saved['sidecar'].name}" if saved.get("sidecar") else ""))

    print("[paul15] rendering operator_space")
    figure_operator_space(adata, out=out_fig / "operator_space")
    print("[paul15] rendering archetype_sheet")
    try:
        figure_archetype_sheet(adata, out=out_fig / "archetype_sheet")
    except Exception as err:
        print(f"[paul15]   archetype_sheet skipped: {err}")
    print("[paul15] rendering ledger_summary")
    figure_ledger_summary(adata, out=out_fig / "ledger_summary")
    try:
        print("[paul15] rendering v1_vs_v2")
        figure_v1_vs_v2(adata, v1_results=V1_BASELINE, out=out_fig / "v1_vs_v2")
    except Exception as err:
        print(f"[paul15]   v1_vs_v2 skipped: {err}")

    try:
        opspace.export_tables(adata, out_run)
    except Exception as err:
        print(f"[paul15]   export_tables skipped: {err}")

    # Fork-archetype driver table (Gata1 / Sfpi1 highlight).
    scj = adata.uns.get("scjdo2", {})
    arch = scj.get("archetypes") or {}
    fork_ids = [a for a, info in arch.items() if info.get("behaviour") == "fork"]
    def _rank_in(df, g):
        idx = df.index[df["gene"].astype(str) == g].tolist()
        return (int(idx[0]), df.loc[idx[0]].to_dict()) if idx else (None, None)
    drivers_path = out_fig / "fork_archetype_drivers.txt"
    with open(drivers_path, "w") as fh:
        for aid in fork_ids:
            info = arch[aid]
            df = info["drivers"]
            header = (
                f"archetype {aid} — behaviour=fork, "
                f"n_cells={info['n_cells']}, "
                f"lead_re={info['eigenvalues'][0].real:+.3f}, "
                f"henrici={info['henrici']:.3f}"
            )
            print(f"\n[paul15] {header}")
            top = df.head(20)
            print(top.to_string(index=False))
            fh.write(header + "\n")
            fh.write(top.to_string(index=False) + "\n")
            for g in MUST_KEEP:
                rank, row = _rank_in(df, g)
                if rank is None:
                    msg = f"  {g}: absent from driver table"
                else:
                    msg = (f"  {g}: rank {rank}, weight={row['weight']:+.4f}, "
                           f"magnitude={row['magnitude']:.4f}, "
                           f"fisher={row['fisher']:.2f}, reason={row['reason']}")
                print(msg)
                fh.write(msg + "\n")
            fh.write("\n")
    print(f"\n[paul15] wrote driver tables to {drivers_path}")
    if not fork_ids:
        print("[paul15] no archetype classified as 'fork'; archetype summary:")
        for aid, info in arch.items():
            print(f"  archetype {aid}: {info['behaviour']} "
                  f"lead=({info['eigenvalues'][0].real:+.3f}"
                  f"{info['eigenvalues'][0].imag:+.3f}j) n={info['n_cells']}")

    print(f"\n[paul15] done — run dir {out_run}; figures {out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
