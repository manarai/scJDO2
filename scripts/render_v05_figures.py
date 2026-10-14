"""Render the six v0.5-basin-lock figures from stored state.

Reads:
  * benchmarks/v0.5_lock/pbmc_v05.h5ad
  * benchmarks/v0.5_lock/pitchfork_v05.h5ad
  * benchmarks/perturbseq_k562/per_knockdown.csv +
    generic_control_comparison.csv

Writes under benchmarks/figures/v0.5_lock/:
  * PBMC:      {1} spectrum_ladder, {2} soft_axis_map,
               {3} ledger_map, {6} basin_index_by_type
  * K562:      {3} ledger_map (NOT built — no stored adata),
               {5} perturb_paired
  * Pitchfork: {4} oracle_agreement

No refitting; everything reads stored artefacts.
"""

from __future__ import annotations

import time
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import pandas as pd

from scjdo2.data import RESTING_TYPES
from scjdo2.plot import (
    basin_index_by_type,
    ledger_map,
    oracle_agreement,
    perturb_paired,
    soft_axis_map,
    spectrum_ladder,
)


REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "benchmarks" / "figures" / "v0.5_lock"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    produced = []

    # PBMC: 1, 2, 3, 6
    pbmc_h5 = REPO / "benchmarks" / "v0.5_lock" / "pbmc_v05.h5ad"
    if pbmc_h5.exists():
        print(f"[render] load {pbmc_h5}")
        a = ad.read_h5ad(pbmc_h5)
        for name, fn, kwargs in [
            ("pbmc_spectrum_ladder", spectrum_ladder,
             dict(adata=a, by="cluster", oracle=True)),
            ("pbmc_soft_axis_map", soft_axis_map,
             dict(adata=a, by="cluster", use_rep="X_umap")),
            ("pbmc_ledger_map", ledger_map,
             dict(adata=a, use_rep="X_umap")),
            ("pbmc_basin_index_by_type", basin_index_by_type,
             dict(adata=a, by="cluster", resting_types=RESTING_TYPES)),
        ]:
            t0 = time.monotonic()
            fn(**kwargs, out=OUT / name)
            print(f"[render]   {name} {time.monotonic()-t0:.1f}s")
            produced.extend([OUT / f"{name}.pdf", OUT / f"{name}.png"])
    else:
        print(f"[render] skip PBMC: {pbmc_h5} not found")

    # Pitchfork: 4
    pf_h5 = REPO / "benchmarks" / "v0.5_lock" / "pitchfork_v05.h5ad"
    if pf_h5.exists():
        print(f"[render] load {pf_h5}")
        p = ad.read_h5ad(pf_h5)
        t0 = time.monotonic()
        tau = float(p.uns.get("v05_meta", {}).get("tau_crit", 0.3))
        oracle_agreement(p, tau_crit=tau, out=OUT / "pitchfork_oracle_agreement")
        print(f"[render]   pitchfork_oracle_agreement {time.monotonic()-t0:.1f}s")
        produced.extend([OUT / "pitchfork_oracle_agreement.pdf",
                          OUT / "pitchfork_oracle_agreement.png"])
    else:
        print(f"[render] skip pitchfork: {pf_h5} not found")

    # K562: 5 (perturb_paired)
    pk_csv = REPO / "benchmarks" / "perturbseq_k562" / "per_knockdown.csv"
    gen_csv = REPO / "benchmarks" / "perturbseq_k562" / "generic_control_comparison.csv"
    if pk_csv.exists():
        print(f"[render] load K562 tables")
        pk = pd.read_csv(pk_csv)
        gen = pd.read_csv(gen_csv) if gen_csv.exists() else None
        t0 = time.monotonic()
        perturb_paired({"per_kd": pk, "generic_table": gen},
                        out=OUT / "k562_perturb_paired")
        print(f"[render]   k562_perturb_paired {time.monotonic()-t0:.1f}s")
        produced.extend([OUT / "k562_perturb_paired.pdf",
                          OUT / "k562_perturb_paired.png"])
    else:
        print(f"[render] skip K562: {pk_csv} not found")

    # K562 ledger_map requires a saved adata; the K562 pipeline did not
    # store one (310 k × 3 559 is too big for the gitignore pattern).
    # Skip gracefully; the pbmc_ledger_map covers the figure type.
    print(f"[render] K562 ledger_map skipped (no stored K562 adata)")

    print("\n[render] produced:")
    for p in produced:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
