"""σ_min sweep for the Phase B gate — pick by held-out DSM loss.

Per user: σ_min ∈ {0.02, 0.05}, same σ_max=1.0. For each σ_min:
    * train the drift on 80 % of the equilibrated pitchfork
    * evaluate the DSM loss (same σ² weighting) on the held-out 20 %
    * record the mean over N_eval MC samples per cell

Winner = lower held-out DSM loss. Not chosen on the gate — the gate is
reported for both but is NOT the selection criterion.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scjdo2 import opspace
from scjdo2.data import analytic_jacobian, generate_pitchfork_equilibrated
from scjdo2.drift import DriftAdapter, DriftConfig


REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "benchmarks" / "phase_b_gate" / "equilibrated" / "sigma_sweep"


def _held_out_dsm_loss(adapter: DriftAdapter, z_test: np.ndarray, t_test: np.ndarray,
                        sigma_range: tuple[float, float], n_mc: int = 50,
                        seed: int = 0) -> float:
    """Evaluate the σ²-weighted DSM loss on held-out cells.

    L = E_σ,ε [ (σ · (s(x + σε, t) + ε/σ))² ].mean() = E[(σ·s + ε)²].mean()
    """
    device = adapter._device
    model = adapter._model
    model.eval()
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    log_lo = float(np.log(sigma_range[0]))
    log_hi = float(np.log(sigma_range[1]))
    xg = torch.as_tensor(z_test, dtype=torch.float32, device=device)
    tg = torch.as_tensor(t_test, dtype=torch.float32, device=device)
    losses = []
    with torch.no_grad():
        for _ in range(int(n_mc)):
            log_sig = torch.rand(xg.shape[0], generator=rng) * (log_hi - log_lo) + log_lo
            sigmas = torch.exp(log_sig).to(device)
            noise = torch.randn(xg.shape, generator=rng).to(device)
            x_noisy = xg + sigmas.unsqueeze(-1) * noise
            target = -noise / sigmas.unsqueeze(-1)
            s_out = model.score(x_noisy, tg)
            diff = sigmas.unsqueeze(-1) * (s_out - target)
            losses.append(float((diff * diff).mean()))
    return float(np.mean(losses))


def _pr(pred: np.ndarray, truth: np.ndarray) -> dict:
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    p = tp / (tp + fp) if (tp + fp) else float("nan")
    r = tp / (tp + fn) if (tp + fn) else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r}


def run_tau(tau_crit: float, sigma_min: float, sigma_max: float,
            n_cells: int, n_alpha: int, n_epochs: int, seed: int,
            verbose: bool) -> dict[str, Any]:
    print(f"\n=== τ_crit={tau_crit}  σ_min={sigma_min}  σ_max={sigma_max} ===")
    rng = np.random.default_rng(seed)
    adata = generate_pitchfork_equilibrated(
        tau_crit=tau_crit, seed=seed,
        n_cells=n_cells, n_alpha=n_alpha,
        n_genes=200, n_latent=20,
        noise_sigma=0.3,
        T_burn=20.0, T_sample=5.0, dt=0.02, sigma_sde=0.20,
        init_box=2.5,
    )
    perm = rng.permutation(adata.n_obs)
    n_train = int(0.8 * adata.n_obs)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    z = adata.obsm["X_pca"].astype(np.float32)
    t = adata.obs["pseudotime"].to_numpy(dtype=np.float32)

    t0 = time.monotonic()
    adapter = DriftAdapter(dim=20, config=DriftConfig(
        hidden=128, depth=3, n_epochs=n_epochs, seed=seed,
        use_spectral_norm=False, batch_size=256,
        inject_velocity_prior=False, use_residual=False,
        beta=0.04,
        dsm_sigma_range=(sigma_min, sigma_max),
    ))
    adapter.fit(z[train_idx], t=t[train_idx], verbose=verbose)
    fit_sec = time.monotonic() - t0
    print(f"  fit {fit_sec:.1f}s")

    held_out = _held_out_dsm_loss(adapter, z[test_idx], t[test_idx],
                                   sigma_range=(sigma_min, sigma_max),
                                   n_mc=50, seed=seed + 1)
    print(f"  held-out DSM loss (σ²-weighted): {held_out:.4f}")

    # Full-dataset gate: compute J via autodiff for all cells using the fitted
    # drift, then opspace.
    J = adapter.jacobian_autodiff(z, t)
    adata.uns["scjdo2"] = {"J": J, "meta": {"clock_calibrated": False,
                                            "jacobian_method": "autodiff"}}
    opspace.spectrum(adata, n_eig=5, fisher_mask=False)
    opspace.invariants(adata)
    opspace.behaviour(adata)
    beh = adata.obs["op_behaviour"].astype(str).to_numpy()
    lead_re = adata.obs["lead_re"].to_numpy()

    # Dynamic ground truth with same-tol logic
    if "alpha" in adata.obs:
        alpha = adata.obs["alpha"].to_numpy(dtype=np.float64)
    else:
        meta = adata.uns["pitchfork"]
        alpha = float(meta["alpha_min"]) + adata.obs["pseudotime"].to_numpy(np.float64) * (
            float(meta["alpha_max"]) - float(meta["alpha_min"]))
    Jan = analytic_jacobian(np.asarray(adata.obsm["X_true"], np.float64), alpha)
    an_lead = np.array([float(np.max(np.linalg.eigvals(Jan[c]).real)) for c in range(Jan.shape[0])])
    tol = float(0.1 * np.nanstd(an_lead[np.isfinite(an_lead)]))
    true_fork = an_lead > tol
    true_holder = an_lead < -tol
    pred_fork = beh == "fork"
    pred_holder = beh == "holder"

    return {
        "tau_crit": tau_crit, "sigma_min": sigma_min, "sigma_max": sigma_max,
        "n_epochs": n_epochs, "fit_seconds": fit_sec,
        "held_out_dsm_loss": held_out,
        "fork_pr": _pr(pred_fork, true_fork),
        "holder_pr": _pr(pred_holder, true_holder),
        "frac_Re_pos_learned": float(np.nanmean(lead_re > 0)),
        "ground_truth_tol": tol,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tau-crit", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    p.add_argument("--sigma-mins", type=float, nargs="+", default=[0.02, 0.05])
    p.add_argument("--sigma-max", type=float, default=1.0)
    p.add_argument("--n-cells", type=int, default=1200)
    p.add_argument("--n-alpha", type=int, default=30)
    p.add_argument("--n-epochs", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for s_min in args.sigma_mins:
        for tau in args.tau_crit:
            results.append(run_tau(
                tau_crit=tau, sigma_min=s_min, sigma_max=args.sigma_max,
                n_cells=args.n_cells, n_alpha=args.n_alpha,
                n_epochs=args.n_epochs, seed=args.seed, verbose=args.verbose,
            ))

    # Summary — mean held-out loss across τ per σ_min
    print("\n=== Summary (held-out DSM loss, averaged over τ_crit) ===")
    for s_min in args.sigma_mins:
        rows = [r for r in results if r["sigma_min"] == s_min]
        avg = float(np.mean([r["held_out_dsm_loss"] for r in rows]))
        print(f"  σ_min={s_min}: mean held-out loss = {avg:.4f}  "
              f"(per τ: {[round(r['held_out_dsm_loss'], 4) for r in rows]})")

    out_json = OUT / "report.json"
    out_json.write_text(json.dumps(
        {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
         "args": vars(args), "results": results}, indent=2, default=float))
    print(f"\n[sigma sweep] wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
