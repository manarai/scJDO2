"""Pitchfork convergence + fixed-point audit.

Train the DSM+velocity-prior drift on the symmetric toggle-switch pitchfork at
a configurable epoch budget, then audit:

* precision/recall of the holder vs. fork label at the ≥ 0.8 target
* the sign of ∂f/∂z at analytic fixed points — symmetric FP (holder for
  α < α_crit, saddle for α > α_crit) and the two asymmetric FPs (stable
  branches, both negative eigenvalues for α > α_crit)
* a scan across pseudotime bins of the autodiff Jacobian's leading two
  eigenvalues' signs

Analytic structure of the symmetric toggle switch
    dx/dt = α / (1 + y⁴) − x
    dy/dt = α / (1 + x⁴) − y

* ``_toggle_sym_fp(α)`` returns the symmetric fixed point ``x = y = x*(α)``
  that solves ``x = α/(1 + x⁴)``. The 2D Jacobian at the symmetric FP has
  trace ``−2`` and determinant ``1 − 16 x*⁶/(1 + x*⁴)²``; both eigenvalues
  negative iff ``α < α_crit = (4/3) · 3^(-1/4)``. Above ``α_crit`` the
  determinant goes negative and one eigenvalue is positive (saddle).
* ``_toggle_asym_fp(α)`` solves the pair ``a = α/(1 + b⁴), b = α/(1 + a⁴)``
  with ``a ≠ b`` by fixed-point iteration; both branches are stable (both
  eigenvalues negative) for ``α > α_crit``.

Usage::

    python scripts/pitchfork_convergence_audit.py --n-epochs 5000 --tau-crit 0.5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import brentq

from scjdo2 import opspace
from scjdo2.data import generate_pitchfork
from scjdo2.data.synthetic_pitchfork import ALPHA_CRIT
from scjdo2.drift import DriftAdapter, DriftConfig
from scjdo2.pipeline import fit_latent_field


def _toggle_sym_fp(alpha: float) -> float:
    """Symmetric FP: solve ``x^5 + x = α``. Reliable above AND below α_crit
    (the fixed-point iteration ``x ← α/(1+x⁴)`` in the data module only
    converges for α < α_crit)."""
    if alpha <= 0:
        return 0.0
    return float(brentq(lambda x: x ** 5 + x - alpha, 0.0, max(alpha, 1e-6) + 1.0))


REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "benchmarks" / "phase_b_gate" / "convergence_audit"


# ---------------------------------------------------------------- analytic

def _toggle_asym_fp(alpha: float, tol: float = 1e-12, max_iter: int = 2000) -> tuple[float, float]:
    """Asymmetric fixed point ``(a, b)`` with ``a > b``.

    Only exists for ``alpha > alpha_crit``. We seed slightly off the diagonal
    and iterate the map ``(a, b) ← (α/(1 + b⁴), α/(1 + a⁴))``.
    """
    if alpha <= ALPHA_CRIT:
        return (float("nan"), float("nan"))
    x_s = _toggle_sym_fp(alpha)
    a = x_s + 0.1
    b = x_s - 0.1
    for _ in range(max_iter):
        a_new = alpha / (1.0 + b ** 4)
        b_new = alpha / (1.0 + a ** 4)
        if abs(a_new - a) + abs(b_new - b) < tol:
            a, b = a_new, b_new
            break
        a, b = a_new, b_new
    return (max(a, b), min(a, b))


def _analytic_J_sym(alpha: float) -> np.ndarray:
    """Jacobian of the toggle switch at ``(x_s, x_s)``."""
    x = _toggle_sym_fp(alpha)
    # d/dx [α/(1 + y⁴) − x] = −1 ; d/dy [α/(1 + y⁴) − x] = −4 α y³ / (1 + y⁴)²
    off = -4.0 * alpha * x ** 3 / (1.0 + x ** 4) ** 2
    return np.array([[-1.0, off], [off, -1.0]], dtype=np.float64)


def _analytic_J_asym(alpha: float, a: float, b: float) -> np.ndarray:
    off_ab = -4.0 * alpha * b ** 3 / (1.0 + b ** 4) ** 2
    off_ba = -4.0 * alpha * a ** 3 / (1.0 + a ** 4) ** 2
    return np.array([[-1.0, off_ab], [off_ba, -1.0]], dtype=np.float64)


# ---------------------------------------------------- find cells near a FP

def _nearest_cells(adata, target_xy: tuple[float, float], tau: float,
                   tau_tol: float = 0.04, top: int = 25) -> np.ndarray:
    """Cell indices nearest a 2D (X_true) point whose pseudotime is close to ``tau``."""
    xy = np.asarray(adata.obsm["X_true"], dtype=np.float64)
    t = adata.obs["pseudotime"].to_numpy()
    mask = np.abs(t - tau) <= tau_tol
    if not mask.any():
        return np.array([], dtype=np.int64)
    cand = np.where(mask)[0]
    d = np.linalg.norm(xy[cand] - np.asarray(target_xy, dtype=np.float64), axis=1)
    order = cand[np.argsort(d)]
    return order[:top]


def _tau_to_alpha(tau: float, alpha_min: float, alpha_max: float) -> float:
    return float(alpha_min + tau * (alpha_max - alpha_min))


def _alpha_to_tau(alpha: float, alpha_min: float, alpha_max: float) -> float:
    return float((alpha - alpha_min) / (alpha_max - alpha_min))


def _toggle_drift_true(z_true: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Analytic toggle drift evaluated per cell with per-cell α."""
    x, y = z_true[:, 0], z_true[:, 1]
    fx = alpha / (1.0 + y ** 4) - x
    fy = alpha / (1.0 + x ** 4) - y
    return np.stack([fx, fy], axis=1)


def _drift_true_space_metrics(adata) -> dict[str, Any]:
    """Project the learned drift (and its Jacobian field) back to 2D true
    space via least-squares and compare to the analytic toggle flow."""
    z_pca = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
    z_true = np.asarray(adata.obsm["X_true"], dtype=np.float64)
    v_pca = np.asarray(adata.obsm["X_drift_v"], dtype=np.float64)
    J_pca = np.asarray(adata.uns["scjdo2"]["J"], dtype=np.float64)
    t = adata.obs["pseudotime"].to_numpy()
    meta = adata.uns["pitchfork"]
    alpha_min = float(meta["alpha_min"])
    alpha_max = float(meta["alpha_max"])
    alpha = alpha_min + t * (alpha_max - alpha_min)

    # Fit A : z_pca → z_true  (least squares); use same linear map on velocities.
    A, *_ = np.linalg.lstsq(z_pca, z_true, rcond=None)  # (20, 2)
    v_true_pred = v_pca @ A
    v_true_analytic = _toggle_drift_true(z_true, alpha)
    err = v_true_pred - v_true_analytic
    err_norm = np.linalg.norm(err, axis=1)
    an_norm = np.linalg.norm(v_true_analytic, axis=1) + 1e-9

    # Project the Jacobian field similarly: J_true ≈ A.T @ J_pca @ (A.T)^{-1}
    # but A is 20×2 and not square. Instead express the local field as 2D by
    # projecting both input and output: J_true = A.T @ J_pca @ A' where
    # A' is the right-inverse of A.T (i.e. ``A @ (A.T A)^{-1}``).
    # Equivalently, use pseudo-inverses on both sides.
    Ap = np.linalg.pinv(A.T)      # (2, 20)
    # J_true (per cell) = A.T @ J_pca @ A  : (2,20) @ (20,20) @ (20,2) = (2,2)
    n = J_pca.shape[0]
    J_true_pred = np.einsum("ia,nab,bj->nij", A.T, J_pca, A)

    return {
        "A_shape": [int(A.shape[0]), int(A.shape[1])],
        "mean_rel_err_v": float(np.mean(err_norm / an_norm)),
        "median_rel_err_v": float(np.median(err_norm / an_norm)),
        "cos_sim_v_mean": float(np.mean(
            np.sum(v_true_pred * v_true_analytic, axis=1)
            / (np.linalg.norm(v_true_pred, axis=1) * an_norm + 1e-12)
        )),
        "frac_J_true_both_neg": float(np.mean([
            all(np.linalg.eigvals(J_true_pred[i]).real < 0) for i in range(n)
        ])),
        "frac_J_true_one_pos": float(np.mean([
            int(np.sum(np.linalg.eigvals(J_true_pred[i]).real > 0)) == 1 for i in range(n)
        ])),
    }


# --------------------------------------------------------- learned J audit

def _top_eigs(J: np.ndarray, k: int = 2) -> np.ndarray:
    """Return the top-``k`` eigenvalues of ``J`` by real part (desc)."""
    evals = np.linalg.eigvals(J)
    order = np.argsort(-evals.real)
    return evals[order[:k]]


def _sign_tuple(evals: np.ndarray, tol: float = 1e-3) -> str:
    def s(x: float) -> str:
        if x > tol:
            return "+"
        if x < -tol:
            return "-"
        return "0"
    return "(" + ",".join(s(float(e.real)) for e in evals) + ")"


def audit(tau_crit: float, n_epochs: int, k_bins: int, n_cells: int, seed: int,
          hidden: int, depth: int, alpha_control: float, inject_velocity_prior: bool,
          verbose: bool) -> dict[str, Any]:
    print(f"\n[audit τ_crit={tau_crit} epochs={n_epochs} prior={inject_velocity_prior}]")
    adata = generate_pitchfork(
        tau_crit=tau_crit, seed=seed,
        n_cells=n_cells, n_genes=200, n_latent=20,
        noise_sigma=0.3, T_total=10.0, dt=0.02, sigma_sde=0.20,
    )
    meta = adata.uns["pitchfork"]
    alpha_min = float(meta["alpha_min"])
    alpha_max = float(meta["alpha_max"])

    t0 = time.monotonic()
    fit_latent_field(
        adata, use_rep="X_pca", time_key="pseudotime",
        drift_config=DriftConfig(
            hidden=hidden, depth=depth, n_epochs=n_epochs, seed=seed,
            use_spectral_norm=False, batch_size=256,
            alpha_control=alpha_control,
            inject_velocity_prior=inject_velocity_prior,
        ),
        jacobian_method="autodiff",
        verbose=verbose,
    )
    dt_fit = time.monotonic() - t0
    print(f"  fit {dt_fit:.1f}s")

    opspace.spectrum(adata, n_eig=5, fisher_mask=False)
    opspace.invariants(adata)
    opspace.behaviour(adata)

    t = adata.obs["pseudotime"].to_numpy()
    beh = adata.obs["op_behaviour"].astype(str).to_numpy()
    lead_re = adata.obs["lead_re"].to_numpy()

    pred_fork = beh == "fork"
    pred_holder = beh == "holder"
    true_fork = np.abs(t - tau_crit) < 0.05
    true_holder = np.abs(t - tau_crit) >= 0.15
    def pr(pred, truth):
        tp = int(np.sum(pred & truth))
        fp = int(np.sum(pred & ~truth))
        fn = int(np.sum(~pred & truth))
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec}
    fork_pr = pr(pred_fork, true_fork)
    holder_pr = pr(pred_holder, true_holder)

    # ------------------------------------------------------ fixed-point audit
    J_cells = np.asarray(adata.uns["scjdo2"]["J"])

    def learned_eigs_near(target_xy: tuple[float, float], tau: float) -> dict[str, Any]:
        idx = _nearest_cells(adata, target_xy, tau)
        if idx.size == 0:
            return {"n": 0}
        tops = np.stack([_top_eigs(J_cells[i], k=2) for i in idx], axis=0)
        return {
            "n": int(idx.size),
            "target_xy": [float(target_xy[0]), float(target_xy[1])],
            "tau": float(tau),
            "mean_top1_re": float(np.mean(tops[:, 0].real)),
            "mean_top2_re": float(np.mean(tops[:, 1].real)),
            "sign_counts_top1": {
                "pos": int(np.sum(tops[:, 0].real > 1e-3)),
                "neg": int(np.sum(tops[:, 0].real < -1e-3)),
                "zero": int(np.sum(np.abs(tops[:, 0].real) <= 1e-3)),
            },
            "sign_counts_top2": {
                "pos": int(np.sum(tops[:, 1].real > 1e-3)),
                "neg": int(np.sum(tops[:, 1].real < -1e-3)),
                "zero": int(np.sum(np.abs(tops[:, 1].real) <= 1e-3)),
            },
        }

    fixed_points: dict[str, Any] = {}
    # Pre-bifurcation symmetric FP (α < α_crit). Only well-defined when
    # α_min < α_crit, i.e. τ_crit > 0.
    tau_pre = max(0.1, tau_crit - 0.4)
    alpha_pre = _tau_to_alpha(tau_pre, alpha_min, alpha_max)
    if alpha_pre > 0:
        x_pre = _toggle_sym_fp(alpha_pre)
        fixed_points["pre_bif_sym"] = {
            "alpha": alpha_pre,
            "analytic_J": _analytic_J_sym(alpha_pre).tolist(),
            "analytic_eigs": [complex(e).real for e in np.linalg.eigvals(_analytic_J_sym(alpha_pre))],
            "learned": learned_eigs_near((x_pre, x_pre), tau_pre),
        }

    # Post-bifurcation symmetric FP (saddle). Only exists when α_max > α_crit.
    tau_post = min(0.9, tau_crit + 0.4)
    alpha_post = _tau_to_alpha(tau_post, alpha_min, alpha_max)
    if alpha_post > ALPHA_CRIT:
        x_sad = _toggle_sym_fp(alpha_post)
        fixed_points["post_bif_saddle_sym"] = {
            "alpha": alpha_post,
            "analytic_J": _analytic_J_sym(alpha_post).tolist(),
            "analytic_eigs": [complex(e).real for e in np.linalg.eigvals(_analytic_J_sym(alpha_post))],
            "learned": learned_eigs_near((x_sad, x_sad), tau_post),
        }
        # Asymmetric stable branches
        a, b = _toggle_asym_fp(alpha_post)
        fixed_points["post_bif_stable_hi"] = {
            "alpha": alpha_post,
            "analytic_J": _analytic_J_asym(alpha_post, a, b).tolist(),
            "analytic_eigs": [complex(e).real for e in np.linalg.eigvals(_analytic_J_asym(alpha_post, a, b))],
            "learned": learned_eigs_near((a, b), tau_post),
        }
        fixed_points["post_bif_stable_lo"] = {
            "alpha": alpha_post,
            "analytic_J": _analytic_J_asym(alpha_post, b, a).tolist(),
            "analytic_eigs": [complex(e).real for e in np.linalg.eigvals(_analytic_J_asym(alpha_post, b, a))],
            "learned": learned_eigs_near((b, a), tau_post),
        }

    # ------------------------------------------------------ pseudotime scan
    bins = np.linspace(0.0, 1.0, k_bins + 1)
    scan = []
    for i in range(k_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (t >= lo) & (t < hi)
        if not mask.any():
            scan.append({"lo": float(lo), "hi": float(hi), "n": 0})
            continue
        tops = np.stack([_top_eigs(J_cells[c], k=2) for c in np.where(mask)[0]], axis=0)
        scan.append({
            "lo": float(lo), "hi": float(hi), "n": int(mask.sum()),
            "mean_top1_re": float(np.mean(tops[:, 0].real)),
            "mean_top2_re": float(np.mean(tops[:, 1].real)),
            "frac_top1_pos": float(np.mean(tops[:, 0].real > 1e-3)),
            "frac_top1_neg": float(np.mean(tops[:, 0].real < -1e-3)),
            "frac_top2_neg": float(np.mean(tops[:, 1].real < -1e-3)),
        })

    true_space = _drift_true_space_metrics(adata)

    return {
        "tau_crit": tau_crit,
        "n_epochs": n_epochs,
        "alpha_control": alpha_control,
        "inject_velocity_prior": inject_velocity_prior,
        "fit_seconds": dt_fit,
        "fork_precision_recall": fork_pr,
        "holder_precision_recall": holder_pr,
        "behaviour_counts": {b_: int(np.sum(beh == b_)) for b_ in np.unique(beh) if b_},
        "lead_re_summary": {
            "median": float(np.nanmedian(lead_re)),
            "frac_positive": float(np.nanmean(lead_re > 0)),
            "frac_negative": float(np.nanmean(lead_re < 0)),
        },
        "true_space_check": true_space,
        "fixed_points": fixed_points,
        "pseudotime_scan": scan,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tau-crit", type=float, nargs="+", default=[0.5])
    p.add_argument("--n-epochs", type=int, default=5000)
    p.add_argument("--n-cells", type=int, default=1200)
    p.add_argument("--k-bins", type=int, default=10)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--alpha-control", type=float, default=1e-3)
    p.add_argument("--inject-velocity-prior", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    for tau in args.tau_crit:
        all_results.append(audit(
            tau_crit=tau, n_epochs=args.n_epochs, k_bins=args.k_bins,
            n_cells=args.n_cells, seed=args.seed,
            hidden=args.hidden, depth=args.depth,
            alpha_control=args.alpha_control,
            inject_velocity_prior=args.inject_velocity_prior,
            verbose=args.verbose,
        ))

    tag = args.tag or f"e{args.n_epochs}_ac{args.alpha_control:g}_prior{int(args.inject_velocity_prior)}"
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "results": all_results,
    }
    out_json = OUT_DIR / f"audit_{tag}.json"
    out_json.write_text(json.dumps(report, indent=2, default=float))
    print(f"\n[audit] wrote {out_json}")

    # Short readable table
    print("\n=== summary ===")
    for r in all_results:
        f, h = r["fork_precision_recall"], r["holder_precision_recall"]
        print(f"τ={r['tau_crit']}: fork P={f['precision']:.2f} R={f['recall']:.2f}  "
              f"holder P={h['precision']:.2f} R={h['recall']:.2f}  "
              f"frac(Re>0)={r['lead_re_summary']['frac_positive']:.2f}")
        ts = r.get("true_space_check", {})
        if ts:
            print(f"   true-space v:  rel_err_median={ts['median_rel_err_v']:.3f}  "
                  f"cos_sim={ts['cos_sim_v_mean']:.3f}  "
                  f"frac(J_true both neg)={ts['frac_J_true_both_neg']:.3f}  "
                  f"frac(J_true one pos)={ts['frac_J_true_one_pos']:.3f}")
        for name, fp in r["fixed_points"].items():
            lrn = fp["learned"]
            if not lrn.get("n"):
                continue
            print(f"   {name:24s} analytic_eigs={fp['analytic_eigs']}  "
                  f"learned mean(top1,top2)=({lrn['mean_top1_re']:+.3f}, {lrn['mean_top2_re']:+.3f})  "
                  f"n={lrn['n']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
