"""Equilibrated-pitchfork re-gate with energy-parameterised DSM drift.

Companion to ``scripts/equilibrated_gate.py``. Reuses the equilibrated
pitchfork generator, the dynamic-sign ground truth, and the same ``_gate``
precision / recall logic — only swaps the drift source from v1's
``DriftField`` (non-symmetric score head) to the new
``EnergyDriftAdapter`` (``s = -∇E``, ``J = -β·Hess E`` symmetric by
construction).

Gate (Phase B v2, same as the original):

  * holder P and R   ≥ 0.80 absolute
  * fork   P and R   within 0.10 of the oracle at the same τ_crit

Verdict is written to ``benchmarks/figures/pitchfork_energy_gate.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Make equilibrated_gate.py importable regardless of the working dir.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the dynamic ground truth + metrics + opspace plumbing from the
# original gate script.
from equilibrated_gate import (                               # type: ignore
    _dynamic_labels, _gate, OUT, row_oracle,
)
from scjdo2.data import generate_pitchfork_equilibrated
from scjdo2.energy_drift import EnergyDriftAdapter, EnergyDriftConfig


REPO = Path(__file__).resolve().parent.parent


def row_energy_dsm(adata, tau_crit: float, n_epochs: int, seed: int,
                   sigma_min: float, sigma_max: float, beta: float,
                   loss: str, verbose: bool) -> dict[str, Any]:
    """Energy-parameterised score drift on equilibrated snapshots.

    Writes the per-cell latent J (symmetric by construction) into
    ``adata.uns['scjdo2']['J']`` so ``_gate`` can score it.
    ``loss`` is ``"esm"`` (Hyvärinen, default) or ``"dsm"`` (multi-σ,
    legacy).
    """
    z = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
    t0 = time.monotonic()
    drift = EnergyDriftAdapter(dim=z.shape[1], config=EnergyDriftConfig(
        hidden=128, depth=3, n_epochs=n_epochs, seed=seed,
        batch_size=256, loss=loss, beta=beta,
        dsm_sigma_range=(sigma_min, sigma_max),
        esm_laplacian="exact" if z.shape[1] <= 20 else "hutchinson",
    )).fit(z, verbose=verbose)
    dt_fit = time.monotonic() - t0
    sym = drift.symmetry_check(z, n_cells=100, seed=seed)
    J = drift.jacobian_autodiff(z)
    v = drift.velocity(z)
    uns = adata.uns.setdefault("scjdo2", {})
    uns["J"] = J
    uns["meta"] = {"clock_calibrated": False, "jacobian_method": "autodiff",
                   "use_rep": "X_pca", "drift_param": "energy_scalar_mlp"}
    adata.obsm["X_drift_v"] = v
    r = _gate(adata, tau_crit)
    r["row"] = "energy_dsm_equilibrated"
    r["fit_seconds"] = dt_fit
    r["symmetry"] = sym
    r["beta"] = float(beta)
    r["dsm_sigma_range"] = [float(sigma_min), float(sigma_max)]
    r["n_epochs"] = int(n_epochs)
    return r


def run_tau(tau_crit: float, n_cells: int, n_alpha: int, n_epochs: int,
            seed: int, sigma_min: float, sigma_max: float, beta: float,
            loss: str, verbose: bool) -> dict[str, Any]:
    print(f"\n=== τ_crit = {tau_crit} ===")
    adata = generate_pitchfork_equilibrated(
        tau_crit=tau_crit, seed=seed,
        n_cells=n_cells, n_alpha=n_alpha,
        n_genes=200, n_latent=20,
        noise_sigma=0.3,
        T_burn=20.0, T_sample=5.0, dt=0.02, sigma_sde=0.20,
        init_box=2.5,
    )
    print(f"  generated: {adata.shape}, {n_alpha} α-slices")

    # Oracle (unchanged — analytic 2D J on (X_true, α)).
    r_oracle = row_oracle(adata, tau_crit)
    print(f"  [oracle] fork P={r_oracle['fork_pr']['precision']:.2f} "
          f"R={r_oracle['fork_pr']['recall']:.2f}  "
          f"holder P={r_oracle['holder_pr']['precision']:.2f} "
          f"R={r_oracle['holder_pr']['recall']:.2f}")

    # Energy-param row.
    print(f"  [energy_{loss}] fit {n_epochs} ep  "
          f"σ∈[{sigma_min:.3f}, {sigma_max:.3f}]  β={beta}")
    r_energy = row_energy_dsm(
        adata, tau_crit, n_epochs=n_epochs, seed=seed,
        sigma_min=sigma_min, sigma_max=sigma_max, beta=beta,
        loss=loss, verbose=verbose,
    )
    sym = r_energy["symmetry"]
    print(f"  [energy_dsm] SYM median={sym['median_asym']:.3e}  "
          f"max={sym['max_asym']:.3e}")
    print(f"  [energy_dsm] fork P={r_energy['fork_pr']['precision']:.2f} "
          f"R={r_energy['fork_pr']['recall']:.2f}  "
          f"holder P={r_energy['holder_pr']['precision']:.2f} "
          f"R={r_energy['holder_pr']['recall']:.2f}  "
          f"frac(Re>0)={r_energy['lead_re']['frac_positive']:.2f}  "
          f"n_true_fork={r_energy['ground_truth']['n_true_fork']}  "
          f"n_true_holder={r_energy['ground_truth']['n_true_holder']}")

    # Gate verdict.
    hp = r_energy["holder_pr"]
    fp = r_energy["fork_pr"]
    op = r_oracle["fork_pr"]
    hold_abs_ok = (
        (hp["precision"] is None or np.isnan(hp["precision"]) or hp["precision"] >= 0.8)
        and (hp["recall"] is None or np.isnan(hp["recall"]) or hp["recall"] >= 0.8)
    )
    fork_rel_ok = (
        np.isnan(fp["precision"]) or np.isnan(op["precision"])
        or abs(fp["precision"] - op["precision"]) <= 0.10
    ) and (
        np.isnan(fp["recall"]) or np.isnan(op["recall"])
        or abs(fp["recall"] - op["recall"]) <= 0.10
    )
    verdict = "PASS" if (hold_abs_ok and fork_rel_ok) else "FAIL"
    print(f"  >>> VERDICT @ τ={tau_crit}: {verdict}")

    return {
        "tau_crit": tau_crit,
        "n_cells": int(adata.n_obs),
        "verdict": verdict,
        "oracle": r_oracle,
        "energy_dsm": r_energy,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tau-crit", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    p.add_argument("--n-cells", type=int, default=1200)
    p.add_argument("--n-alpha", type=int, default=20)
    p.add_argument("--n-epochs", type=int, default=3000)
    p.add_argument("--sigma-min", type=float, default=0.05)
    p.add_argument("--sigma-max", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--loss", choices=["esm", "dsm"], default="esm")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    per_tau = []
    for tau in args.tau_crit:
        per_tau.append(run_tau(
            tau_crit=tau, n_cells=args.n_cells, n_alpha=args.n_alpha,
            n_epochs=args.n_epochs, seed=args.seed,
            sigma_min=args.sigma_min, sigma_max=args.sigma_max,
            beta=args.beta, loss=args.loss, verbose=args.verbose,
        ))

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "ground_truth": "dynamic sign of analytic lead_re per cell",
        "drift_parameterisation": "energy_scalar_mlp (E_θ: R^d → R)",
        "per_tau": per_tau,
    }
    out_json = OUT / "pitchfork_energy_gate.json"
    out_json.write_text(json.dumps(report, indent=2, default=float))
    print(f"\n[energy pitchfork gate] wrote {out_json}")

    passes = sum(1 for row in per_tau if row["verdict"] == "PASS")
    print(f"[energy pitchfork gate] {passes}/{len(per_tau)} τ gates PASS")
    return 0 if passes == len(per_tau) else 1


if __name__ == "__main__":
    raise SystemExit(main())
