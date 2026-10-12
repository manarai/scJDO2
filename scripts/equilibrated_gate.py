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
from scjdo2.field import smooth_jacobians
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
    """Per-cell ground truth from the analytic 2D J of f_toggle, quantized by
    the SAME 0.1·std tolerance that ``opspace.behaviour`` applies.

    Returns ``(true_fork, true_holder, analytic_lead_re)``:
        tol              = 0.1 * np.nanstd(analytic_lead_re_finite)
        true_fork(c)     ⇔ analytic_lead_re(c) >  tol
        true_holder(c)   ⇔ analytic_lead_re(c) < -tol
        (cells with |analytic_lead_re| ≤ tol are "drift" and belong to
         neither class.)

    Using the same tol logic on both sides lets the oracle hit ~1.0 by
    construction and makes "fork P/R within 0.10 of oracle" a well-defined
    relative gate (Phase B gate v2).

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
    finite = lead_re[np.isfinite(lead_re)]
    tol = float(0.1 * np.nanstd(finite)) if finite.size else 0.1
    true_fork = lead_re > tol
    true_holder = lead_re < -tol
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
    finite = analytic_lead_re[np.isfinite(analytic_lead_re)]
    analytic_tol = float(0.1 * np.nanstd(finite)) if finite.size else 0.1

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
            "tol": analytic_tol,
            "n_true_fork": int(true_fork.sum()),
            "n_true_holder": int(true_holder.sum()),
            "frac_true_fork": float(np.mean(true_fork)),
            "analytic_lead_re_median": float(np.median(analytic_lead_re)),
        },
    }


def _gate_smoothed(adata: AnnData, use_rep: str, k: int,
                   bandwidth: Optional[float] = None) -> dict[str, Any]:
    """Apply Gaussian-kernel smoothing (Step 5 post-hoc) to J, re-run opspace,
    rescore the gate.

    Smoothing uses the SAME latent representation ``use_rep`` the drift was
    fit on (so neighbourhoods are consistent with the drift's geometry).
    """
    z = np.asarray(adata.obsm[use_rep], dtype=np.float64)
    J = np.asarray(adata.uns["scjdo2"]["J"], dtype=np.float64)
    J_smooth = smooth_jacobians(J, z, k=int(k), bandwidth=bandwidth, include_self=True)
    adata.uns["scjdo2"]["J_raw"] = J
    adata.uns["scjdo2"]["J"] = J_smooth
    adata.uns["scjdo2"].setdefault("meta", {})
    adata.uns["scjdo2"]["meta"]["smoothed"] = True
    adata.uns["scjdo2"]["meta"]["smooth_k"] = int(k)
    adata.uns["scjdo2"]["meta"]["smooth_bandwidth"] = (
        float(bandwidth) if (bandwidth is not None and np.isfinite(bandwidth)) else "median_nn"
    )
    out = _gate(adata, tau_crit=0.0)  # tau_crit is unused inside _gate
    adata.uns["scjdo2"]["J"] = J  # restore raw so caller can keep going
    adata.uns["scjdo2"].pop("J_raw", None)
    out["smooth_k"] = int(k)
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
                         dsm_sigma_range: Optional[tuple],
                         use_residual: bool, beta: float,
                         verbose: bool,
                         smooth_k: Optional[int] = None) -> dict[str, Any]:
    """Row 2 — DSM drift fit + autodiff J on equilibrated snapshots.

    With ``use_residual=False`` and ``inject_velocity_prior=False`` the drift
    reduces to ``u = β · score(x, t)`` whose autodiff Jacobian is exactly
    ``β · ∂score/∂z``, the Hessian of log density (up to the β scaling).
    Setting ``β = σ_sde²`` puts the drift in the SDE's units for a reversible
    (gradient) flow.
    """
    t0 = time.monotonic()
    fit_latent_field(
        adata, use_rep="X_pca", time_key="pseudotime",
        drift_config=DriftConfig(
            hidden=128, depth=3, n_epochs=n_epochs, seed=seed,
            use_spectral_norm=False, batch_size=256,
            inject_velocity_prior=inject_velocity_prior,
            dsm_sigma_range=dsm_sigma_range,
            use_residual=use_residual,
            beta=beta,
        ),
        jacobian_method="autodiff",
        verbose=verbose,
    )
    dt_fit = time.monotonic() - t0
    r = _gate(adata, tau_crit)
    r["row"] = "dsm_equilibrated"
    r["fit_seconds"] = dt_fit
    r["inject_velocity_prior"] = bool(inject_velocity_prior)
    r["use_residual"] = bool(use_residual)
    r["beta"] = float(beta)
    r["dsm_sigma_range"] = list(dsm_sigma_range) if dsm_sigma_range else None
    r["n_epochs"] = int(n_epochs)
    if smooth_k is not None and smooth_k > 0:
        r_sm = _gate_smoothed(adata, use_rep="X_pca", k=int(smooth_k))
        r_sm["row"] = "dsm_equilibrated_smoothed"
        r_sm["smooth_k"] = int(smooth_k)
        r["smoothed"] = r_sm
    return r


def row_dsm_transient(tau_crit: float, n_cells: int, n_epochs: int, seed: int,
                      inject_velocity_prior: bool,
                      dsm_sigma_range: Optional[tuple],
                      use_residual: bool, beta: float,
                      verbose: bool,
                      smooth_k: Optional[int] = None) -> dict[str, Any]:
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
            dsm_sigma_range=dsm_sigma_range,
            use_residual=use_residual,
            beta=beta,
        ),
        jacobian_method="autodiff",
        verbose=verbose,
    )
    dt_fit = time.monotonic() - t0
    r = _gate(adata, tau_crit)
    r["row"] = "dsm_transient"
    r["fit_seconds"] = dt_fit
    r["use_residual"] = bool(use_residual)
    r["beta"] = float(beta)
    r["dsm_sigma_range"] = list(dsm_sigma_range) if dsm_sigma_range else None
    r["note"] = "α-ramp synthetic — negative control"
    if smooth_k is not None and smooth_k > 0:
        r_sm = _gate_smoothed(adata, use_rep="X_pca", k=int(smooth_k))
        r_sm["row"] = "dsm_transient_smoothed"
        r_sm["smooth_k"] = int(smooth_k)
        r["smoothed"] = r_sm
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
            seed: int, inject_velocity_prior: bool,
            dsm_sigma_range: Optional[tuple],
            use_residual: bool, beta: float,
            verbose: bool,
            smooth_k: Optional[int] = None) -> dict[str, Any]:
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
    print(f"  [dsm_equilibrated] fit {n_epochs} epochs, prior={inject_velocity_prior}, "
          f"residual={use_residual}, β={beta}, sigma_range={dsm_sigma_range}")
    r2 = row_dsm_equilibrated(
        adata, tau_crit, n_epochs=n_epochs, seed=seed,
        inject_velocity_prior=inject_velocity_prior,
        dsm_sigma_range=dsm_sigma_range,
        use_residual=use_residual, beta=beta,
        verbose=verbose, smooth_k=smooth_k,
    )
    print(f"    fork  P={r2['fork_pr']['precision']:.2f} R={r2['fork_pr']['recall']:.2f}  "
          f"holder P={r2['holder_pr']['precision']:.2f} R={r2['holder_pr']['recall']:.2f}  "
          f"frac(Re>0)={r2['lead_re']['frac_positive']:.2f}")
    if "smoothed" in r2:
        s = r2["smoothed"]
        print(f"    [smoothed k={smooth_k}] fork P={s['fork_pr']['precision']:.2f} "
              f"R={s['fork_pr']['recall']:.2f}  "
              f"holder P={s['holder_pr']['precision']:.2f} R={s['holder_pr']['recall']:.2f}  "
              f"frac(Re>0)={s['lead_re']['frac_positive']:.2f}")

    # Row 3b — DSM-transient, same gate (dynamic ground truth)
    print(f"  [dsm_transient] fit {n_epochs} epochs on α-ramp synthetic, "
          f"residual={use_residual}, β={beta}, sigma_range={dsm_sigma_range}")
    r3 = row_dsm_transient(
        tau_crit=tau_crit, n_cells=adata.n_obs, n_epochs=n_epochs,
        seed=seed, inject_velocity_prior=inject_velocity_prior,
        dsm_sigma_range=dsm_sigma_range,
        use_residual=use_residual, beta=beta,
        verbose=verbose, smooth_k=smooth_k,
    )
    print(f"    fork  P={r3['fork_pr']['precision']:.2f} R={r3['fork_pr']['recall']:.2f}  "
          f"holder P={r3['holder_pr']['precision']:.2f} R={r3['holder_pr']['recall']:.2f}  "
          f"frac(Re>0)={r3['lead_re']['frac_positive']:.2f}  "
          f"n_true_fork={r3['ground_truth']['n_true_fork']}")
    if "smoothed" in r3:
        s = r3["smoothed"]
        print(f"    [smoothed k={smooth_k}] fork P={s['fork_pr']['precision']:.2f} "
              f"R={s['fork_pr']['recall']:.2f}  "
              f"holder P={s['holder_pr']['precision']:.2f} R={s['holder_pr']['recall']:.2f}  "
              f"frac(Re>0)={s['lead_re']['frac_positive']:.2f}")

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
    p.add_argument("--use-residual", action=argparse.BooleanOptionalAction, default=True,
                   help="If False, zero out v1's residual MLP (drift = β·score + vel). "
                        "Combined with --no-inject-velocity-prior gives DSM-only drift "
                        "= β·score, Jacobian = β·∂²log p/∂z².")
    p.add_argument("--beta", type=float, default=0.1,
                   help="Scale on the score term in the drift. For the equilibrated "
                        "pitchfork (σ_sde=0.20), set β=σ²=0.04 so drift = σ²·∇log p.")
    p.add_argument("--sigma-min", type=float, default=None,
                   help="If set with --sigma-max, use multi-σ DSM log-uniform in [σ_min, σ_max].")
    p.add_argument("--sigma-max", type=float, default=None)
    p.add_argument("--smooth-k", type=int, default=30,
                   help="k for post-hoc Gaussian smoothing of J (0 to disable). "
                        "Fixed-k default per user: 30.")
    p.add_argument("--v1-json", type=Path, default=None,
                   help="Path to v1 slicing-on-equilibrated JSON (optional)")
    p.add_argument("--out-tag", type=str, default=None,
                   help="Suffix for output files; default uses the σ schedule.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    dsm_range: Optional[tuple] = None
    if args.sigma_min is not None and args.sigma_max is not None:
        dsm_range = (float(args.sigma_min), float(args.sigma_max))

    OUT.mkdir(parents=True, exist_ok=True)

    per_tau = []
    smooth_k = int(args.smooth_k) if args.smooth_k and args.smooth_k > 0 else None
    for tau in args.tau_crit:
        per_tau.append(run_tau(
            tau_crit=tau, n_cells=args.n_cells, n_alpha=args.n_alpha,
            n_epochs=args.n_epochs, seed=args.seed,
            inject_velocity_prior=args.inject_velocity_prior,
            dsm_sigma_range=dsm_range,
            use_residual=args.use_residual, beta=args.beta,
            verbose=args.verbose,
            smooth_k=smooth_k,
        ))

    v1_row = row_v1_slicing(args.v1_json)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "ground_truth": "dynamic (strict sign of analytic lead_re per cell)",
        "per_tau": per_tau,
        "v1_slicing_on_equilibrated": v1_row,
    }
    tag = args.out_tag or ("multi_sigma" if dsm_range else "single_sigma")
    out_json = OUT / f"report_{tag}.json"
    out_json.write_text(json.dumps(report, indent=2, default=float))
    # Also update the canonical report.json pointer
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\n[equilibrated gate] wrote {out_json}")

    # Markdown table
    # Phase B gate v2 (per user):
    #   holder P and R ≥ 0.8 absolute
    #   fork P and R within 0.10 of the oracle at the same τ_crit
    oracle_by_tau = {row["tau_crit"]: row["rows"]["oracle"] for row in per_tau}

    lines = [
        "# Equilibrated pitchfork gate — four-row table",
        f"\nGenerated: {report['timestamp']}\n",
        "Gate (Phase B, v2):",
        "  * holder P and R  ≥ 0.8 (absolute)",
        "  * fork P and R    within 0.10 of the oracle on the same ground truth",
        "",
        "Ground truth: fork = analytic_lead_re > tol, holder = analytic_lead_re < -tol,",
        "where tol = 0.1 · std(analytic_lead_re) — same scheme as opspace.behaviour.",
        "",
        "| row | τ_crit | fork P | fork R | holder P | holder R | frac(Re>0) | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def _mdrow(label, tau, r):
        f = r.get("fork_pr", {})
        h = r.get("holder_pr", {})
        fp = f.get("precision", float("nan"))
        fr = f.get("recall", float("nan"))
        hp = h.get("precision", float("nan"))
        hr = h.get("recall", float("nan"))
        # Holder absolute
        def _ok(x): return isinstance(x, (int, float)) and np.isfinite(x)
        holder_ok = _ok(hp) and _ok(hr) and hp >= 0.8 and hr >= 0.8
        # Fork relative-to-oracle
        if label == "oracle":
            fork_ok = _ok(fp) and _ok(fr) and fp >= 0.8 and fr >= 0.8
        else:
            orc = oracle_by_tau.get(tau, {}).get("fork_pr", {})
            oracle_fp = orc.get("precision", float("nan"))
            oracle_fr = orc.get("recall", float("nan"))
            fork_ok = (
                _ok(fp) and _ok(fr) and _ok(oracle_fp) and _ok(oracle_fr)
                and abs(fp - oracle_fp) <= 0.10
                and abs(fr - oracle_fr) <= 0.10
            )
        verd = "PASS" if (holder_ok and fork_ok) else "FAIL"
        frac = r.get("lead_re", {}).get("frac_positive", float("nan"))
        if frac is None:
            frac = float("nan")
        def _fmt(x): return "nan" if not _ok(x) else f"{x:.2f}"
        return (f"| {label} | {tau} | {_fmt(fp)} | {_fmt(fr)} | "
                f"{_fmt(hp)} | {_fmt(hr)} | {_fmt(frac)} | **{verd}** |")

    for row in per_tau:
        tau = row["tau_crit"]
        lines.append(_mdrow("oracle", tau, row["rows"]["oracle"]))
        lines.append(_mdrow("dsm_equilibrated", tau, row["rows"]["dsm_equilibrated"]))
        if "smoothed" in row["rows"]["dsm_equilibrated"]:
            lines.append(_mdrow(
                f"dsm_equilibrated_smoothed(k={smooth_k})",
                tau,
                row["rows"]["dsm_equilibrated"]["smoothed"],
            ))
    v1_rows_per_tau = v1_row.get("per_tau", []) if v1_row.get("status") != "pending" else []
    for v in v1_rows_per_tau:
        lines.append(_mdrow("v1_slicing_equilibrated", v["tau_crit"], v))
    for row in per_tau:
        tau = row["tau_crit"]
        lines.append(_mdrow("dsm_transient", tau, row["rows"]["dsm_transient"]))
        if "smoothed" in row["rows"]["dsm_transient"]:
            lines.append(_mdrow(
                f"dsm_transient_smoothed(k={smooth_k})",
                tau,
                row["rows"]["dsm_transient"]["smoothed"],
            ))

    if v1_row.get("status") == "pending":
        lines.append(f"\n> v1_slicing_on_equilibrated: PENDING "
                     f"— {v1_row.get('note', '')}")

    # Short interpretation
    lines.extend([
        "",
        "## Reading",
        "",
        "Ground truth: fork/holder labels derived from the analytic 2D J of",
        "``f_toggle(X_true_c, α_c)``, quantized by the same 0.1·std tolerance",
        "scheme ``opspace.behaviour`` applies. The oracle becomes self-",
        "consistent and P/R ~ 1.0 by construction.",
        "",
        "* **Oracle** is the ceiling — classifier applied to analytic J.",
        "* **DSM-equilibrated** measures how well a learned drift's autodiff",
        "  Jacobian (``u = β·score`` with the residual MLP frozen zero,",
        "  ``β = σ_sde²``, multi-σ DSM) matches the oracle labels.",
        "* **v1_slicing_equilibrated** uses v1's unmodified DriftField (with",
        "  the residual MLP it cannot disable from the public API — V2_PLAN",
        "  R12), so it reproduces the uniformly-expansive-Jacobian pathology",
        "  as designed.",
        "* **DSM-transient** is the α-ramp negative control: snapshot score",
        "  on non-stationary data has no basin structure to recover.",
        "",
        "Phase C unblocks when DSM-equilibrated meets:",
        "  * holder P and R  ≥ 0.8 absolute",
        "  * fork P and R    within 0.10 of the oracle at the same τ_crit.",
    ])

    out_md = OUT / f"report_{tag}.md"
    out_md.write_text("\n".join(lines) + "\n")
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print(f"[equilibrated gate] wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
