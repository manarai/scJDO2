"""Four-row Phase B gate on the equilibrated pitchfork.

Per user:
    (1) Build the equilibrated synthetic (snapshot = mixture of stationary
        densities per α).
    (2) Oracle: fit the Jacobian from the TRUE velocity by autograd of the
        known f_toggle, run the same classifier and the same gate, and
        report it first — that is the ceiling.
    (3) DSM on the equilibrated snapshots, autodiff Jacobian, same gate.
    (4) v1 slicing on the equilibrated data, same gate (loaded from a
        side JSON produced by compare/run_v1.py; see --v1-json).

Transient-α-ramp row is pulled from benchmarks/phase_b_gate/report.json for
side-by-side (negative-control) context.

Phase C stays blocked until the DSM-equilibrated row meets fork+holder
P/R ≥ 0.8 at every τ_crit.

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


def _gate(adata: AnnData, tau_crit: float, fork_half_width: float = 0.05,
          holder_margin: float = 0.15) -> dict[str, Any]:
    """Apply opspace to ``adata.uns["scjdo2"]["J"]`` and score the gate.

    ``J`` is assumed already populated by the caller.
    """
    opspace.spectrum(adata, n_eig=5, fisher_mask=False)
    opspace.invariants(adata)
    opspace.behaviour(adata)

    beh = adata.obs["op_behaviour"].astype(str).to_numpy()
    lead_re = adata.obs["lead_re"].to_numpy()
    t = adata.obs["pseudotime"].to_numpy()

    pred_fork = beh == "fork"
    pred_holder = beh == "holder"
    true_fork = np.abs(t - tau_crit) < fork_half_width
    true_holder = np.abs(t - tau_crit) >= holder_margin

    out = {
        "fork_pr": _pr(pred_fork, true_fork),
        "holder_pr": _pr(pred_holder, true_holder),
        "beh_counts": {b: int(np.sum(beh == b)) for b in np.unique(beh) if b},
        "lead_re": {
            "frac_positive": float(np.nanmean(lead_re > 0)),
            "frac_negative": float(np.nanmean(lead_re < 0)),
            "median": float(np.nanmedian(lead_re)),
        },
    }
    return out


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


def row_dsm_transient_from_report() -> list[dict[str, Any]]:
    """Row 3 — DSM on the TRANSIENT α-ramp pitchfork. Pulled from the
    standing ``benchmarks/phase_b_gate/report.json`` so we don't re-run it.
    """
    if not TRANSIENT_REPORT.exists():
        return [{"row": "dsm_transient", "error": "no transient report on disk"}]
    with TRANSIENT_REPORT.open() as fh:
        data = json.load(fh)
    out = []
    for p in data.get("pitchfork", []):
        out.append({
            "row": "dsm_transient",
            "tau_crit": p["tau_crit"],
            "fork_pr": p["fork_precision_recall"],
            "holder_pr": p["holder_precision_recall"],
            "beh_counts": p["behaviour_counts"],
            "lead_re": p.get("lead_re_summary"),
            "note": "snapshot from non-stationary α-ramp — see V2_PLAN R11",
        })
    return out


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

    return {
        "tau_crit": tau_crit,
        "n_cells": int(adata.n_obs),
        "n_alpha": int(n_alpha),
        "rows": {"oracle": r1, "dsm_equilibrated": r2},
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

    transient_rows = row_dsm_transient_from_report()
    v1_row = row_v1_slicing(args.v1_json)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "per_tau": per_tau,
        "dsm_transient_from_report": transient_rows,
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
    for t_row in transient_rows:
        if "tau_crit" in t_row:
            lines.append(_mdrow("dsm_transient", t_row["tau_crit"], t_row))

    if v1_row.get("status") == "pending":
        lines.append(f"\n> v1_slicing_on_equilibrated: PENDING "
                     f"— {v1_row.get('note', '')}")
    else:
        lines.append("\n## v1_slicing_on_equilibrated\n")
        lines.append("```json")
        lines.append(json.dumps(v1_row, indent=2, default=float))
        lines.append("```")

    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print(f"[equilibrated gate] wrote {OUT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
