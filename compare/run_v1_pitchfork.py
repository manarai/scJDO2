"""v1 reference run on scJDO2's pitchfork generator — resolves R11.

Produces v1's own ``tau_R1_lambda_max`` for τ_crit ∈ {0.1, 0.3, 0.5} across
multiple seeds. The ±0.05 Phase B target in :mod:`tests.test_phase_b_pitchfork`
should be read against the mean of these values, not against the analytic
τ_crit (see V2_PLAN.md R11).

Mirrors the readout pipeline of
``scJDO/scratchpad_run/saddle_readouts_midinterval.py`` but reads data from
:func:`scjdo2.data.generate_pitchfork` so v1 and v2 are measured on the exact
same dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from scjdo2.data import generate_pitchfork

try:  # pragma: no cover
    from scjdo.tl import fit_drift
except Exception as e:  # pragma: no cover
    print(f"[run_v1_pitchfork] v1 scjdo not importable: {e}", file=sys.stderr)
    raise


N_GRID = 200
BANDWIDTH = 0.04
N_EFF_MIN = 20.0
TAU_LO = 0.05
TAU_HI = 0.95


def _kernel_curve(scalar: np.ndarray, t: np.ndarray, grid: np.ndarray) -> np.ndarray:
    curve = np.full(grid.shape, np.nan)
    for k, tau in enumerate(grid):
        w = np.exp(-((t - tau) ** 2) / (2.0 * BANDWIDTH ** 2))
        W = float(w.sum())
        if W < 1e-9:
            continue
        n_eff = (W * W) / float((w * w).sum() + 1e-30)
        if n_eff < N_EFF_MIN:
            continue
        curve[k] = float((w * scalar).sum() / W)
    return curve


def _argmax_in(curve: np.ndarray, grid: np.ndarray) -> float:
    m = (grid >= TAU_LO) & (grid <= TAU_HI) & ~np.isnan(curve)
    if not m.any():
        return float("nan")
    return float(grid[np.where(m)[0][np.argmax(curve[m])]])


def _jacobian_lambda_max(model, Z: np.ndarray, t: np.ndarray) -> np.ndarray:
    model = model.eval()
    device = next(model.parameters()).device
    z_t = torch.as_tensor(Z, dtype=torch.float32, device=device)
    t_t = torch.as_tensor(t, dtype=torch.float32, device=device)
    J = model.jacobian(z_t, t_t).detach().cpu().numpy()
    lam = np.zeros(J.shape[0], dtype=np.float32)
    for i in range(J.shape[0]):
        lam[i] = float(np.linalg.eigvals(J[i]).real.max())
    return lam


def run_one(tau_crit: float, seed: int, n_epochs: int) -> dict:
    ad = generate_pitchfork(
        tau_crit=tau_crit, seed=seed,
        n_cells=1500, n_genes=200, n_latent=20,
        noise_sigma=0.3, T_total=10.0, dt=0.02, sigma_sde=0.20,
    )
    Z = ad.obsm["X_pca"].astype(np.float32)
    t = ad.obs["pseudotime"].to_numpy(dtype=np.float32)

    t0 = time.monotonic()
    model = fit_drift(
        ad, rep="X_pca", time_key="pseudotime", n_epochs=n_epochs,
        n_archetypes=4, n_eff_min=N_EFF_MIN, n_boot=10, grid_size=N_GRID,
        seed=seed, verbose=False,
    )
    fit_time = time.monotonic() - t0

    lam = _jacobian_lambda_max(model, Z, t)
    grid = np.linspace(0.0, 1.0, N_GRID, dtype=np.float64)
    curve = _kernel_curve(lam, t.astype(np.float64), grid)
    tau_hat = _argmax_in(curve, grid)

    return {
        "tau_crit": float(tau_crit),
        "seed": int(seed),
        "tau_R1_lambda_max": float(tau_hat),
        "fit_time_s": float(fit_time),
        "r2": float(ad.uns["scjdo"]["r2"]) if "scjdo" in ad.uns else None,
        "lambda_percentiles": np.nanpercentile(lam, [10, 50, 90]).tolist(),
        "curve_min": float(np.nanmin(curve)),
        "curve_max": float(np.nanmax(curve)),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--tau-crits", nargs="+", type=float, default=[0.1, 0.3, 0.5])
    p.add_argument("--seeds", nargs="+", type=int, default=[300, 301, 302])
    p.add_argument("--n-epochs", type=int, default=800)
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    per_run = []
    for tc in args.tau_crits:
        for s in args.seeds:
            print(f"[run_v1_pitchfork] tau_crit={tc}  seed={s}", flush=True)
            r = run_one(tc, s, args.n_epochs)
            per_run.append(r)
            print(f"  → tau_R1_lambda_max={r['tau_R1_lambda_max']:.3f}  "
                  f"fit={r['fit_time_s']:.0f}s  R²={r['r2']}", flush=True)

    # Aggregate per τ_crit.
    aggregate = {}
    for tc in args.tau_crits:
        rows = [r for r in per_run if r["tau_crit"] == tc]
        taus = [r["tau_R1_lambda_max"] for r in rows if np.isfinite(r["tau_R1_lambda_max"])]
        aggregate[str(tc)] = {
            "n_seeds": len(rows),
            "n_finite": len(taus),
            "mean_tau_hat": float(np.mean(taus)) if taus else None,
            "std_tau_hat": float(np.std(taus)) if taus else None,
            "min_tau_hat": float(np.min(taus)) if taus else None,
            "max_tau_hat": float(np.max(taus)) if taus else None,
            "err_vs_analytic": float(abs(np.mean(taus) - tc)) if taus else None,
        }

    out_json = args.out / "pitchfork_reference.json"
    with out_json.open("w") as fh:
        json.dump({"per_run": per_run, "aggregate": aggregate,
                   "n_epochs": args.n_epochs,
                   "tau_crits": args.tau_crits,
                   "seeds": args.seeds}, fh, indent=2)
    print("", flush=True)
    print(f"[run_v1_pitchfork] saved {out_json}", flush=True)
    for tc, agg in aggregate.items():
        print(f"  τ_crit={tc}  → v1 mean_tau_hat={agg['mean_tau_hat']}  "
              f"(err vs analytic={agg['err_vs_analytic']})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
