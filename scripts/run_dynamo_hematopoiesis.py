"""Phase D §7 — Dynamo scNT-seq hematopoiesis gate on measured flow.

Replaces Step 2's residual-free DSM drift with ``MeasuredDrift``
regressed on dynamo's labelled velocity_pca (per-hour units from the
5 h one-shot label). Then runs scjdo2 Phase C (rate model, anchor J,
Fisher ledger, archetype operators) and reports the full gate:

  * behaviour counts by annotated fate (holders expected on terminals),
  * fork near the erythroid–myeloid split (GMP-like / MEP-like),
  * Gata1 / Spi1 ranks in both responders and drivers_left per archetype,
  * eigenvalues in per-hour units with ``clock_calibrated = True``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from scjdo2 import io as scjdo2_io
from scjdo2 import opspace
from scjdo2.data import load_dynamo_hematopoiesis
from scjdo2.measured_drift import MeasuredDrift, MeasuredDriftConfig
from scjdo2.pipeline import fit_rate_and_gene_field


REPO = Path(__file__).resolve().parent.parent
MUST_KEEP = ("GATA1", "SPI1")
TERMINALS = {"Ery", "Meg", "Mon", "Neu", "Bas"}
SPLIT_CELL_TYPES = {"MEP-like", "GMP-like"}


def _fit_measured_drift_field(
    adata, n_pcs: int, n_epochs: int, seed: int, verbose: bool,
):
    """Phase B/§7 analogue of fit_latent_field. Fits MeasuredDrift
    on dynamo's measured velocity_pca, writes obsm['X_drift_v'] and
    uns['scjdo2']['J'] (dense autodiff Jacobian), and marks
    clock_calibrated=True because the velocity is in per-hour units."""
    z = np.asarray(adata.obsm["X_pca"])
    v = np.asarray(adata.obsm["velocity_pca"])
    assert z.shape == v.shape, f"X_pca {z.shape} vs velocity_pca {v.shape}"
    drift = MeasuredDrift(dim=z.shape[1], config=MeasuredDriftConfig(
        hidden=(96, 96, 96), n_epochs=n_epochs, seed=seed, clock_calibrated=True,
    )).fit(z, v, verbose=verbose)
    v_pred = drift.velocity(z)
    J = drift.jacobian_autodiff(z)
    scj = adata.uns.setdefault("scjdo2", {})
    scj["J"] = J
    scj["meta"] = {
        "clock_calibrated": True,
        "use_rep": "X_pca",
        "time_key": None,
        "jacobian_method": "autodiff",
        "drift": {
            "method": "measured_mlp_regression",
            "source": adata.uns.get("labelling", {}).get("source"),
            "n_epochs": int(n_epochs),
        },
        "field": {"method": "autodiff"},
    }
    scj["field_reason_summary"] = {"ok": 1.0, "autodiff": True}
    adata.obsm["X_drift_v"] = v_pred
    return drift


def behaviour_counts_by_fate(adata) -> pd.DataFrame:
    beh = adata.obs["op_behaviour"].astype(str).to_numpy()
    fate = adata.obs["cell_type"].astype(str).to_numpy()
    rows = []
    for f in sorted(set(fate)):
        if f == "nan":
            continue
        mask = fate == f
        c = pd.Series(beh[mask]).value_counts()
        rec = {"fate": f, "n": int(mask.sum())}
        for k in ("fork", "holder", "drift", "damped_oscillator", "growing_oscillator"):
            rec[k] = int(c.get(k, 0))
        rec["frac_fork"] = rec["fork"] / rec["n"] if rec["n"] else 0.0
        rec["frac_holder"] = rec["holder"] / rec["n"] if rec["n"] else 0.0
        rows.append(rec)
    return pd.DataFrame(rows)


def behaviour_at_split(adata) -> pd.DataFrame:
    """Report behaviour counts restricted to MEP-like + GMP-like
    (the annotated ery/myelo bifurcation progenitors). If any cells
    in these annotations get holder, that's the saddle approach;
    fork remains the expected classification *at* the split."""
    beh = adata.obs["op_behaviour"].astype(str).to_numpy()
    fate = adata.obs["cell_type"].astype(str).to_numpy()
    out = []
    for f in sorted(SPLIT_CELL_TYPES):
        mask = fate == f
        c = pd.Series(beh[mask]).value_counts()
        rec = {"fate": f, "n": int(mask.sum())}
        for k in ("fork", "holder", "drift", "damped_oscillator", "growing_oscillator"):
            rec[k] = int(c.get(k, 0))
        out.append(rec)
    return pd.DataFrame(out)


def archetype_driver_report(adata) -> pd.DataFrame:
    arch = adata.uns["scjdo2"].get("archetypes") or {}
    rows = []
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
                rows.append({
                    "archetype": int(ak),
                    "behaviour": info["behaviour"],
                    "lead_re_per_h": float(info["eigenvalues"][0].real),
                    "lead_im_per_h": float(info["eigenvalues"][0].imag),
                    "n_cells": int(info["n_cells"]),
                    "table": table_name,
                    "gene": g,
                    "rank": i,
                    "sensitivity": float(r.get("sensitivity", float("nan"))),
                    "magnitude": float(r["magnitude"]),
                    "weight": float(r["weight"]),
                    "fisher": float(r["fisher"]),
                    "snr": float(r.get("snr", float("nan"))),
                    "reason": str(r["reason"]),
                })
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-pcs", type=int, default=10)
    p.add_argument("--drift-epochs", type=int, default=1500)
    p.add_argument("--rate-epochs", type=int, default=200)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--n-anchors", type=int, default=200)
    p.add_argument("--anchor-top-k", type=int, default=10)
    p.add_argument("--anchor-max-live", type=int, default=300)
    p.add_argument("--cv-sample-cells", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-run", type=Path,
                   default=REPO / "benchmarks" / "v2" / "dynamo_hematopoiesis")
    args = p.parse_args()
    args.out_run.mkdir(parents=True, exist_ok=True)

    print("[dynamo-hemat] loading processed dataset")
    adata = load_dynamo_hematopoiesis(n_pcs=args.n_pcs)
    print(f"[dynamo-hemat]   shape={adata.shape}  cell_types={sorted(set(adata.obs['cell_type']))}")

    # Pseudotime proxy for parts of scjdo2 that peek at obs['pseudotime'].
    # The MeasuredDrift doesn't need it; it's required by fit_rate_and_gene_field's
    # size-factor path only when size_factor=None but we also need it for figure
    # layer defaults. Set to the labelled potential (first PC) so it's monotone.
    adata.obs["pseudotime"] = adata.obsm["X_pca"][:, 0].astype(float)

    print(f"[dynamo-hemat] measured-drift fit ({args.drift_epochs} epochs)")
    t0 = time.monotonic()
    _fit_measured_drift_field(adata, args.n_pcs, args.drift_epochs,
                              args.seed, verbose=True)
    print(f"[dynamo-hemat]   drift fit {time.monotonic()-t0:.1f}s")

    print(f"[dynamo-hemat] Phase C (anchor fit, max_live={args.anchor_max_live})")
    t0 = time.monotonic()
    fit_rate_and_gene_field(
        adata, use_rep="X_pca",
        n_epochs_rate=args.rate_epochs,
        k=args.k, cv_sample_cells=args.cv_sample_cells,
        seed=args.seed,
        anchor_fit=True, n_anchors=args.n_anchors,
        anchor_top_k=args.anchor_top_k,
        anchor_max_live=args.anchor_max_live,
    )
    print(f"[dynamo-hemat]   Phase C in {time.monotonic()-t0:.1f}s")

    # Mark the stored run as clock-calibrated per R18 — the drift came from
    # labelling, so eigenvalues are per-hour.
    adata.uns.setdefault("scjdo2", {}).setdefault("meta", {})
    adata.uns["scjdo2"]["meta"]["clock_calibrated"] = True

    print("[dynamo-hemat] opspace.compute")
    opspace.compute(adata)

    print("\n[dynamo-hemat] BEHAVIOUR COUNTS BY FATE:")
    bc = behaviour_counts_by_fate(adata)
    print(bc.to_string(index=False))
    bc.to_csv(args.out_run / "behaviour_counts_by_fate.csv", index=False)

    print("\n[dynamo-hemat] BEHAVIOUR AT ANNOTATED SPLIT (MEP-like + GMP-like):")
    bs = behaviour_at_split(adata)
    print(bs.to_string(index=False))
    bs.to_csv(args.out_run / "behaviour_at_split.csv", index=False)

    print("\n[dynamo-hemat] GATA1 / SPI1 ARCHETYPE DRIVER REPORT:")
    drv = archetype_driver_report(adata)
    print(drv.to_string(index=False))
    drv.to_csv(args.out_run / "archetype_drivers_GATA1_SPI1.csv", index=False)

    # Behaviour summary across the whole dataset.
    summ = adata.obs["op_behaviour"].astype(str).value_counts()
    print(f"\n[dynamo-hemat] overall behaviour counts:")
    print(summ.to_string())
    summ.to_csv(args.out_run / "overall_behaviour.csv")

    # Save run dir (best-effort).
    try:
        scjdo2_io.save(adata, args.out_run)
        print(f"[dynamo-hemat] saved {args.out_run}")
    except Exception as err:
        print(f"[dynamo-hemat] save skipped: {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
