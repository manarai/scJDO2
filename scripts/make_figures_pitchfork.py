"""Task 41 §6 — produce the pitchfork figure set from one adata.

Runs a lightweight Phase C pipeline on the equilibrated pitchfork
(single τ, smaller cell count than the gate) so the figure CLI has
something to render, then calls `figure_operator_space`,
`figure_archetype_sheet` (if archetypes are present), and
`figure_ledger_summary`.

Output lands under ``benchmarks/figures/pitchfork/``.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                    # headless
import numpy as np

from scjdo2 import io as scjdo2_io, opspace
from scjdo2.data import generate_pitchfork_equilibrated
from scjdo2.drift import DriftConfig
from scjdo2.field import (
    FieldConfig,
    GeneFieldConfig,
    fit_gene_space_jacobians,
    fit_jacobian_tangent_projectors,
)
from scjdo2.fisher import compute as fisher_compute
from scjdo2.pipeline import fit_latent_field
from scjdo2.plot import (
    figure_ledger_summary,
    figure_operator_space,
)
from scjdo2.rates import GeneRateModel, RatesConfig


def make_pitchfork_adata(
    tau_crit: float = 0.3,
    n_cells: int = 600,
    n_alpha: int = 20,
    n_genes: int = 50,
    seed: int = 0,
):
    """Build + fit a pitchfork adata with everything the §1 compute()
    wrapper needs: latent J (dense), gene-space J (list of CSR), the
    Fisher ledger (with ``off_tangent`` projector + reason_per_cell),
    ``grad_eta`` for the R15 lift, and the latent kNN embedding."""
    print(f"[figures] generating pitchfork (τ={tau_crit}, n={n_cells})")
    adata = generate_pitchfork_equilibrated(
        tau_crit=tau_crit, seed=seed,
        n_cells=n_cells, n_alpha=n_alpha,
        n_genes=n_genes, n_latent=10,
        noise_sigma=0.3,
        T_burn=15.0, T_sample=5.0, dt=0.02, sigma_sde=0.2,
        init_box=2.5, counts=True, counts_depth=10.0,
    )
    # Fit latent drift + J_latent via the Phase B pipeline (equilibrated DSM
    # off — simpler for a figure dataset).
    print("[figures] fitting latent drift + J_latent")
    t0 = time.monotonic()
    fit_latent_field(
        adata, use_rep="X_pca", time_key="pseudotime",
        drift_config=DriftConfig(hidden=96, depth=3, n_epochs=1500,
                                 seed=seed, use_spectral_norm=False,
                                 batch_size=256),
        field_config=FieldConfig(k=30, ridge=1e-3, seed=seed),
    )
    print(f"[figures]   drift+latent J in {time.monotonic() - t0:.1f}s")
    # The pipeline wrote adata.uns["scjdo2"]["J"] = latent dense J.
    uns = adata.uns.setdefault("scjdo2", {})
    uns["J_latent"] = np.asarray(uns["J"])

    # Fit NB rate model, then gene-space J + Fisher ledger + tangent + grad_eta.
    print("[figures] fitting NB rate model")
    counts = np.asarray(adata.X, dtype=np.float32)
    rm = GeneRateModel(RatesConfig(
        hidden=(128, 128), n_epochs=400, lr=1e-3,
        batch_size=256, held_out_frac=0.10, patience=30,
        seed=seed,
    ))
    rm.fit(counts, np.asarray(adata.obsm["X_pca"]))
    print(f"[figures]   held-out NB NLL = {rm.best_val_nll:.3f}")

    print("[figures] fitting gene-space J + tangent + ledger")
    gs = fit_gene_space_jacobians(
        np.asarray(adata.obsm["X_pca"]),
        np.asarray(adata.obsm["X_drift_v"]),
        rm, counts=counts,
        config=GeneFieldConfig(
            k=30, cv_sample_cells=30, cv_metric="nb_nll", seed=seed,
        ),
    )
    tangent = fit_jacobian_tangent_projectors(
        np.asarray(adata.obsm["X_pca"]), rm,
        n_perm=30, quantile=0.5, seed=seed,
    )
    ledger = fisher_compute(gs, tangent=tangent)

    # Store the gene-space J (R15: compute() will swap in J_latent for the
    # behavioural pass, then back to this for the drivers pass).
    uns["J"] = gs["J"]
    uns["fisher_ledger"] = ledger
    uns["grad_eta"] = rm.grad_eta(np.asarray(adata.obsm["X_pca"]))   # (n, G, d)
    return adata


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tau-crit", type=float, default=0.3)
    p.add_argument("--n-cells", type=int, default=600)
    p.add_argument("--n-alpha", type=int, default=20)
    p.add_argument("--n-genes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=Path("benchmarks/figures/pitchfork"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    adata = make_pitchfork_adata(
        tau_crit=args.tau_crit,
        n_cells=args.n_cells, n_alpha=args.n_alpha,
        n_genes=args.n_genes, seed=args.seed,
    )

    print("[figures] running opspace.compute (R15 orchestration)")
    opspace.compute(adata)

    # Persist run directory first (adata.h5ad + phase_c.npz sidecar) so
    # `scjdo2 figure *` can read it back without re-fitting.
    print(f"[figures] saving run dir to {args.out}")
    saved = scjdo2_io.save(adata, args.out)
    print(f"[figures]   wrote {saved['h5ad'].name}"
          + (f" + {saved['sidecar'].name}" if saved.get("sidecar") else ""))

    print("[figures] rendering operator_space")
    figure_operator_space(adata, out=args.out / "operator_space")
    print("[figures] rendering ledger_summary")
    figure_ledger_summary(adata, out=args.out / "ledger_summary")

    print("[figures] exporting per_cell_invariants.csv")
    try:
        opspace.export_tables(adata, args.out)
    except Exception as err:
        print(f"[figures]   export_tables skipped: {err}")

    print(f"[figures] done — see {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
