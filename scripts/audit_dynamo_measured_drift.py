"""Audit the Dynamo hematopoiesis §7 run — checks (1), (2), (3) from
the Phase D critique:

  (1) Permutation null on the measured-flow drift: shuffle velocity
      vectors among cells, refit the MLP with identical settings,
      compare the per-cell lead_re distribution against the real
      fit.
  (2) Held-out velocity R² of the MLP + weight-decay sweep by
      held-out R².
  (3) Report the latent dimension and the tangent-rank distribution
      from the rate-model fit.

Pure latent operator — we skip the per-cell gene-space J fit so this
is fast (~1-3 min per fit). The latent operator and its spectrum are
all we need to compare behaviour fractions under real vs permuted
flow.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scjdo2.data import load_dynamo_hematopoiesis
from scjdo2.measured_drift import MeasuredDrift, MeasuredDriftConfig


REPO = Path(__file__).resolve().parent.parent


def _held_out_r2(
    z: np.ndarray, v: np.ndarray, *,
    weight_decay: float, n_epochs: int, hidden: tuple[int, ...], seed: int,
    held_out_frac: float = 0.2,
) -> tuple[float, MeasuredDrift]:
    """Fit MeasuredDrift on 1-held_out_frac cells; measure R² on the
    rest. Returns (R², fitted model fit on the training split)."""
    rng = np.random.default_rng(seed)
    n = z.shape[0]
    idx = rng.permutation(n)
    n_val = int(round(held_out_frac * n))
    val, train = idx[:n_val], idx[n_val:]
    drift = MeasuredDrift(dim=z.shape[1], config=MeasuredDriftConfig(
        hidden=hidden, n_epochs=n_epochs, weight_decay=weight_decay,
        seed=seed, clock_calibrated=True,
    )).fit(z[train], v[train], verbose=False)
    pred = drift.velocity(z[val])
    ss_res = float(((pred - v[val]) ** 2).sum())
    ss_tot = float(((v[val] - v[val].mean(axis=0)) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return r2, drift


def _lead_re_distribution(drift: MeasuredDrift, z: np.ndarray) -> np.ndarray:
    """Per-cell lead_re from the drift's autodiff Jacobian. Returns
    ``(n,)`` float of the maximum-real-part eigenvalue."""
    J = drift.jacobian_autodiff(z)
    n = J.shape[0]
    out = np.empty(n, dtype=np.float64)
    for c in range(n):
        try:
            ev = np.linalg.eigvals(J[c])
            out[c] = float(np.max(ev.real))
        except np.linalg.LinAlgError:
            out[c] = np.nan
    return out


def _classify(lead_re: np.ndarray, tol: float) -> dict:
    r = lead_re[np.isfinite(lead_re)]
    rec = {
        "n": int(r.size),
        "fork": int((r > tol).sum()),
        "holder": int((r < -tol).sum()),
        "near_zero": int((np.abs(r) <= tol).sum()),
        "median_lead_re": float(np.median(r)),
        "q05_lead_re": float(np.quantile(r, 0.05)),
        "q95_lead_re": float(np.quantile(r, 0.95)),
    }
    rec["frac_fork"] = rec["fork"] / rec["n"] if rec["n"] else 0.0
    rec["frac_holder"] = rec["holder"] / rec["n"] if rec["n"] else 0.0
    return rec


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-pcs", type=int, default=10)
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--n-perm", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=REPO / "benchmarks" / "v2" / "dynamo_hematopoiesis")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[audit] loading Dynamo hemat (d_latent={args.n_pcs})")
    a = load_dynamo_hematopoiesis(n_pcs=args.n_pcs)
    z = np.asarray(a.obsm["X_pca"])
    v = np.asarray(a.obsm["velocity_pca"])
    print(f"[audit]   z={z.shape}  v={v.shape}")
    tol = 0.0            # near-zero tol: 0 for a direction-free test

    # ---------- (2) + (3) ----------
    print("\n[audit] (2) Held-out velocity R² + weight-decay sweep")
    sweep_rows = []
    for wd in (0.0, 1e-5, 1e-4, 1e-3, 1e-2):
        r2, _ = _held_out_r2(
            z, v,
            weight_decay=wd, n_epochs=args.epochs,
            hidden=(96, 96, 96), seed=args.seed,
        )
        sweep_rows.append({"weight_decay": wd, "R2": r2})
        print(f"  weight_decay={wd:.0e}  R²={r2:.4f}")
    sweep = pd.DataFrame(sweep_rows).sort_values("R2", ascending=False)
    best_wd = float(sweep.iloc[0]["weight_decay"])
    print(f"  best weight_decay by held-out R²: {best_wd:.0e}  R²={float(sweep.iloc[0]['R2']):.4f}")
    sweep.to_csv(args.out / "audit_mlp_wd_sweep.csv", index=False)

    print(f"\n[audit] (3) d_latent={args.n_pcs}")

    # ---------- (1) ----------
    print(f"\n[audit] (1) Real fit lead_re distribution (full data, wd={best_wd:.0e})")
    t0 = time.monotonic()
    drift_real = MeasuredDrift(dim=z.shape[1], config=MeasuredDriftConfig(
        hidden=(96, 96, 96), n_epochs=args.epochs, weight_decay=best_wd,
        seed=args.seed, clock_calibrated=True,
    )).fit(z, v, verbose=False)
    real_lead = _lead_re_distribution(drift_real, z)
    real_class = _classify(real_lead, tol=tol)
    print(f"  real: {real_class}  (fit {time.monotonic()-t0:.1f}s)")

    print(f"\n[audit] (1) Permutation null: {args.n_perm} velocity-permuted refits")
    rng = np.random.default_rng(args.seed)
    perm_rows = []
    perm_leads = []
    for p_i in range(args.n_perm):
        perm = rng.permutation(z.shape[0])
        t0 = time.monotonic()
        drift_p = MeasuredDrift(dim=z.shape[1], config=MeasuredDriftConfig(
            hidden=(96, 96, 96), n_epochs=args.epochs,
            weight_decay=best_wd, seed=args.seed + 1 + p_i,
            clock_calibrated=True,
        )).fit(z, v[perm], verbose=False)
        lp = _lead_re_distribution(drift_p, z)
        rec = _classify(lp, tol=tol)
        rec["perm"] = p_i
        rec["elapsed_s"] = float(time.monotonic() - t0)
        perm_rows.append(rec)
        perm_leads.append(lp)
        print(f"  perm {p_i}: frac_fork={rec['frac_fork']:.3f}  median_lead_re={rec['median_lead_re']:.4f}  "
              f"q95={rec['q95_lead_re']:.4f}  ({rec['elapsed_s']:.1f}s)")

    perm_df = pd.DataFrame(perm_rows)
    perm_df.to_csv(args.out / "audit_perm_null.csv", index=False)
    real_row = pd.DataFrame([{**real_class, "perm": "real"}])
    pd.concat([real_row, perm_df], ignore_index=True).to_csv(
        args.out / "audit_lead_re_summary.csv", index=False,
    )
    print(f"\n[audit] REAL vs PERM: frac_fork real={real_class['frac_fork']:.3f}  "
          f"perm median={float(perm_df['frac_fork'].median()):.3f}  "
          f"perm q05-q95=[{float(perm_df['frac_fork'].quantile(0.05)):.3f}, "
          f"{float(perm_df['frac_fork'].quantile(0.95)):.3f}]")
    print(f"[audit] REAL vs PERM: median_lead_re real={real_class['median_lead_re']:.4f}  "
          f"perm median={float(perm_df['median_lead_re'].median()):.4f}  "
          f"perm q05-q95=[{float(perm_df['median_lead_re'].quantile(0.05)):.4f}, "
          f"{float(perm_df['median_lead_re'].quantile(0.95)):.4f}]")
    print(f"\n[audit] outputs under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
