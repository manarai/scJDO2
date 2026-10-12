"""Four-row Phase B gate on the equilibrated pitchfork — DYNAMIC ground truth.

Per user:
    (1) Build the equilibrated synthetic (snapshot = mixture of stationary
        densities per α).
    (2) Oracle: analytic ∂f_toggle/∂z at each cell, same opspace classifier,
        same gate — the ceiling.
    (3) DSM on the equilibrated snapshots, autodiff Jacobian, same gate.
    (4) v1 slicing on the equilibrated data, same gate.
    (5) DSM-transient (α-ramp) under the SAME dynamic ground truth, inline.

Ground-truth labelling per cell (strict sign, no tolerance):

    true_fork(c)   ⇔  Re(λ_max) of analytic 2D J at (X_true_c, α_c) > 0
    true_holder(c) ⇔  Re(λ_max) of analytic 2D J at (X_true_c, α_c) < 0

The τ-band labels proved sub-optimal on equilibrated data (the oracle
itself failed the ≥ 0.8 gate because equilibrated snapshots are mostly
in basins). Dynamic labels follow the actual physics per cell; the
oracle is self-consistent (P/R near 1 by construction) and the DSM
rows get measured against the correct per-cell physics.

Phase C stays blocked until DSM-equilibrated passes ≥ 0.8 under this gate.

Usage::

    python scripts/equilibrated_gate.py --n-epochs 5000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from anndata import AnnData

from scjdo2 import opspace
from scjdo2.data import (
    analytic_jacobian,
    generate_pitchfork,
    generate_pitchfork_equilibrated,
)
from scjdo2.drift import DriftConfig
from scjdo2.pipeline import fit_latent_field


REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "benchmarks" / "phase_b_gate" / "equilibrated"
TRANSIENT_REPORT = REPO / "benchmarks" / "phase_b_gate" / "report.json"


# ---------------------------------------------------------------- metrics

def _pr(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}


def _dynamic_labels(adata: AnnData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cell ground truth from the analytic 2D J of f_toggle.

    Returns ``(true_fork, true_holder, analytic_lead_re)``:
        * ``true_fork(c)   ⇔ analytic_lead_re(c) > 0``
        * ``true_holder(c) ⇔ analytic_lead_re(c) < 0``
    (Strict sign — no tolerance band.)

    Works for both equilibrated and transient variants: equilibrated stores
    ``obs["alpha"]`` directly; transient derives α from ``pseudotime`` via
    ``uns["pitchfork"]`` metadata.
    """
    z_true = np.asarray(adata.obsm["X_true"], dtype=np.float64)
    if "alpha" in adata.obs:
        alpha = adata.obs["alpha"].to_numpy(dtype=np.float64)
    else:
        meta = adata.uns["pitchfork"]
        a_min = float(meta["alpha_min"])
        a_max = float(meta["alpha_max"])
        alpha = a_min + adata.obs["pseudotime"].to_numpy(dtype=np.float64) * (a_max - a_min)
    J = analytic_jacobian(z_true, alpha)
    lead_re = np.array([float(np.max(np.linalg.eigvals(J[c]).real)) for c in range(J.shape[0])])
    true_fork = lead_re > 0.0
    true_holder = lead_re < 0.0
    return true_fork, true_holder, lead_re


def _gate(adata: AnnData, tau_crit: float) -> dict[str, Any]:
    """Apply opspace to ``adata.uns["scjdo2"]["J"]`` and score the gate under
    the DYNAMIC per-cell ground truth (sign of analytic lead_re).

    ``J`` is assumed already populated by the caller.
    """
    opspace.spectrum(adata, n_eig=5, fisher_mask=False)
    opspace.invariants(adata)
    opspace.behaviour(adata)

    beh = adata.obs["op_behaviour"].astype(str).to_numpy()
    lead_re = adata.obs["lead_re"].to_numpy()

    pred_fork = beh == "fork"
    pred_holder = beh == "holder"
    true_fork, true_holder, analytic_lead_re = _dynamic_labels(adata)

    return {
        "fork_pr": _pr(pred_fork, true_fork),
        "holder_pr": _pr(pred_holder, true_holder),
        "beh_counts": {b: int(np.sum(beh == b)) for b in np.unique(beh) if b},
        "lead_re": {
            "frac_positive": float(np.nanmean(lead_re > 0)),
            "frac_negative": float(np.nanmean(lead_re < 0)),
            "median": float(np.nanmedian(lead_re)),
        },
        "ground_truth": {
            "n_true_fork": int(true_fork.sum()),
            "n_true_holder": int(true_holder.sum()),
            "frac_true_fork": float(np.mean(true_fork)),
            "analytic_lead_re_median": float(np.median(analytic_lead_re)),
        },
    }


# -------------------------------------------------------- row producers

def row_oracle(adata: AnnData, tau_crit: float) -> dict[str, Any]:
    """Row 1 — analytic Jacobian of f_toggle at each cell's (X_true, α).

    The analytic J is a ``(n, 2, 2)`` tensor; opspace / behaviour / gate
    operate on this 2D J directly.
    """
    alpha = adata.obs["alpha"].to_numpy()
    z_true = np.asarray(adata.obsm["X_true"], dtype=np.float64)
    J = analytic_jacobian(z_true, alpha)
    # Local AnnData copy so we can store J and run opspace on 2D J without
    # polluting the DSM runs that use the same adata.
    ad2 = adata.copy()
    ad2.uns.setdefault("scjdo2", {})
    ad2.uns["scjdo2"]["J"] = J
    ad2.uns["scjdo2"]["meta"] = {"clock_calibrated": False, "jacobian_method": "oracle_analytic"}
    return {
        "row": "oracle",
        "note": "analytic 2D J of f_toggle at (X_true, α) per cell",
        **_gate(ad2, tau_crit),
    }


def row_dsm_equilibrated(adata: AnnData, tau_crit: float, n_epochs: int,
                         seed: int, inject_velocity_prior: bool,
                         verbose: bool) -> dict[str, Any]:
    """Row 2 — DSM drift fit + autodiff J on equilibrated snapshots."""
    t0 = time.monotonic()
    fit_latent_field(
        adata, use_rep="X_pca", time_key="pseudotime",
        drift_config=DriftConfig(
            hidden=128, depth=3, n_epochs=n_epochs, seed=seed,
            use_spectral_norm=False, batch_size=256,
            inject_velocity_prior=inject_velocity_prior,
        ),
        jacobian_method="autodiff",
        verbose=verbose,
    )
    dt_fit = time.monotonic() - t0
    r = _gate(adata, tau_crit)
    r["row"] = "dsm_equilibrated"
    r["fit_seconds"] = dt_fit
    r["inject_velocity_prior"] = bool(inject_velocity_prior)
    r["n_epochs"] = int(n_epochs)
    return r


def row_dsm_transient(tau_crit: float, n_cells: int, n_epochs: int, seed: int,
                      inject_velocity_prior: bool, verbose: bool) -> dict[str, Any]:
    """Row 3 — DSM on the TRANSIENT α-ramp pitchfork, measured against the
    SAME dynamic per-cell ground truth as the other rows.
    """
    adata = generate_pitchfork(
        tau_crit=tau_crit, seed=seed,
        n_cells=n_cells, n_genes=200, n_latent=20,
        noise_sigma=0.3, T_total=10.0, dt=0.02, sigma_sde=0.20,
    )
    t0 = time.monotonic()
    fit_latent_field(
        adata, use_rep="X_pca", time_key="pseudotime",
        drift_config=DriftConfig(
            hidden=128, depth=3, n_epochs=n_epochs, seed=seed,
            use_spectral_norm=False, batch_size=256,
            inject_velocity_prior=inject_velocity_prior,
        ),
        jacobian_method="autodiff",
        verbose=verbose,
    )
    dt_fit = time.monotonic() - t0
    r = _gate(adata, tau_crit)
    r["row"] = "dsm_transient"
    r["fit_seconds"] = dt_fit
    r["note"] = "α-ramp synthetic — negative control"
    return r


def row_v1_slicing(json_path: Optional[Path]) -> dict[str, Any]:
    """Row 4 — v1 pseudotime-windowed Jacobian on equilibrated data.

    Produced out-of-band by ``compare/run_v1_equilibrated_pitchfork.py``; we
    read the resulting JSON here. If the path is missing we return a stub.
    """
    if json_path is None or not json_path.exists():
        return {
            "row": "v1_slicing",
            "status": "pending",
            "note": "produce via compare/run_v1_equilibrated_pitchfork.py",
        }
    with json_path.open() as fh:
        data = json.load(fh)
    return {"row": "v1_slicing", **data}


# ------------------------------------------------------------------- cli

def run_tau(tau_crit: float, n_cells: int, n_alpha: int, n_epochs: int,
            seed: int, inject_velocity_prior: bool, verbose: bool) -> dict[str, Any]:
    print(f"\n=== τ_crit = {tau_crit} ===")
    adata = generate_pitchfork_equilibrated(
        tau_crit=tau_crit, seed=seed,
        n_cells=n_cells, n_alpha=n_alpha,
        n_genes=200, n_latent=20,
        noise_sigma=0.3,
        T_burn=20.0, T_sample=5.0, dt=0.02, sigma_sde=0.20,
        init_box=2.5,
    )
    print(f"  generated: {adata.shape}, {n_alpha} α-slices, "
          f"{np.sum(np.abs(adata.obs['pseudotime'].to_numpy() - tau_crit) < 0.05)} cells near τ_crit")

    # Row 1 — oracle, ALWAYS FIRST
    print("  [oracle] analytic 2D J of f_toggle")
    r1 = row_oracle(adata, tau_crit)
    print(f"    fork  P={r1['fork_pr']['precision']:.2f} R={r1['fork_pr']['recall']:.2f}  "
          f"holder P={r1['holder_pr']['precision']:.2f} R={r1['holder_pr']['recall']:.2f}  "
          f"frac(Re>0)={r1['lead_re']['frac_positive']:.2f}")

    # Row 2 — DSM-equilibrated
    print(f"  [dsm_equilibrated] fit {n_epochs} epochs, prior={inject_velocity_prior}")
    r2 = row_dsm_equilibrated(
        adata, tau_crit, n_epochs=n_epochs, seed=seed,
        inject_velocity_prior=inject_velocity_prior, verbose=verbose,
    )
    print(f"    fork  P={r2['fork_pr']['precision']:.2f} R={r2['fork_pr']['recall']:.2f}  "
          f"holder P={r2['holder_pr']['precision']:.2f} R={r2['holder_pr']['recall']:.2f}  "
          f"frac(Re>0)={r2['lead_re']['frac_positive']:.2f}")

    # Row 3b — DSM-transient, same gate (dynamic ground truth)
    print(f"  [dsm_transient] fit {n_epochs} epochs on α-ramp synthetic")
    r3 = row_dsm_transient(
        tau_crit=tau_crit, n_cells=adata.n_obs, n_epochs=n_epochs,
        seed=seed, inject_velocity_prior=inject_velocity_prior, verbose=verbose,
    )
    print(f"    fork  P={r3['fork_pr']['precision']:.2f} R={r3['fork_pr']['recall']:.2f}  "
          f"holder P={r3['holder_pr']['precision']:.2f} R={r3['holder_pr']['recall']:.2f}  "
          f"frac(Re>0)={r3['lead_re']['frac_positive']:.2f}  "
          f"n_true_fork={r3['ground_truth']['n_true_fork']}")

    return {
        "tau_crit": tau_crit,
        "n_cells": int(adata.n_obs),
        "n_alpha": int(n_alpha),
        "rows": {"oracle": r1, "dsm_equilibrated": r2, "dsm_transient": r3},
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tau-crit", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    p.add_argument("--n-cells", type=int, default=1200)
    p.add_argument("--n-alpha", type=int, default=20)
    p.add_argument("--n-epochs", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--inject-velocity-prior", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--v1-json", type=Path, default=None,
                   help="Path to v1 slicing-on-equilibrated JSON (optional)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    per_tau = []
    for tau in args.tau_crit:
        per_tau.append(run_tau(
            tau_crit=tau, n_cells=args.n_cells, n_alpha=args.n_alpha,
            n_epochs=args.n_epochs, seed=args.seed,
            inject_velocity_prior=args.inject_velocity_prior,
            verbose=args.verbose,
        ))

    v1_row = row_v1_slicing(args.v1_json)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "ground_truth": "dynamic (strict sign of analytic lead_re per cell)",
        "per_tau": per_tau,
        "v1_slicing_on_equilibrated": v1_row,
    }
    out_json = OUT / "report.json"
    out_json.write_text(json.dumps(report, indent=2, default=float))
    print(f"\n[equilibrated gate] wrote {out_json}")

    # Markdown table
    lines = [
        "# Equilibrated pitchfork gate — four-row table",
        f"\nGenerated: {report['timestamp']}\n",
        "| row | τ_crit | fork P | fork R | holder P | holder R | frac(Re>0) | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def _mdrow(label, tau, r):
        f = r.get("fork_pr", {})
        h = r.get("holder_pr", {})
        verd = "PASS" if (
            all(np.isfinite([f.get("precision", np.nan), f.get("recall", np.nan),
                             h.get("precision", np.nan), h.get("recall", np.nan)]))
            and f["precision"] >= 0.8 and f["recall"] >= 0.8
            and h["precision"] >= 0.8 and h["recall"] >= 0.8
        ) else "FAIL"
        frac = r.get("lead_re", {}).get("frac_positive", float("nan"))
        return (f"| {label} | {tau} | {f.get('precision', float('nan')):.2f} | "
                f"{f.get('recall', float('nan')):.2f} | "
                f"{h.get('precision', float('nan')):.2f} | "
                f"{h.get('recall', float('nan')):.2f} | {frac:.2f} | **{verd}** |")

    for row in per_tau:
        tau = row["tau_crit"]
        lines.append(_mdrow("oracle", tau, row["rows"]["oracle"]))
        lines.append(_mdrow("dsm_equilibrated", tau, row["rows"]["dsm_equilibrated"]))
    v1_rows_per_tau = v1_row.get("per_tau", []) if v1_row.get("status") != "pending" else []
    for v in v1_rows_per_tau:
        lines.append(_mdrow("v1_slicing_equilibrated", v["tau_crit"], v))
    for row in per_tau:
        tau = row["tau_crit"]
        lines.append(_mdrow("dsm_transient", tau, row["rows"]["dsm_transient"]))

    if v1_row.get("status") == "pending":
        lines.append(f"\n> v1_slicing_on_equilibrated: PENDING "
                     f"— {v1_row.get('note', '')}")

    # Short interpretation
    lines.extend([
        "",
        "## Reading",
        "",
        "Ground truth is strict per-cell sign of the ANALYTIC",
        "``∂f_toggle/∂z`` evaluated at ``(X_true_c, α_c)``.",
        "``true_fork(c) ⇔ analytic_lead_re > 0``; ``true_holder(c) ⇔ < 0``.",
        "",
        "* **Oracle** is self-consistent (classifier applied to the analytic",
        "  J, scored against the same J's sign) → P/R ≈ 1 modulo cells near",
        "  zero that opspace labels \"drift\".",
        "* **DSM-equilibrated** and **v1_slicing_equilibrated** measure how",
        "  well the LEARNED Jacobian recovers the analytic per-cell sign.",
        "* **DSM-transient** is the α-ramp negative control, measured against",
        "  the same analytic ground truth so it is directly comparable.",
        "",
        "Phase C unblocks only when DSM-equilibrated passes fork+holder P/R",
        "≥ 0.8 at every τ_crit.",
    ])

    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print(f"[equilibrated gate] wrote {OUT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
