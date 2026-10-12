"""Row 4 of the four-row equilibrated gate — v1 slicing on equilibrated data.

For each τ_crit in {0.1, 0.3, 0.5}:
  * generate the equilibrated pitchfork
  * train v1's DriftField via ``sjd.tl.fit_drift(..., rep="X_pca",
    windowing="kernel")`` (no ``prepare_trajectory`` needed because the
    synthetic already has ``X_pca`` and ``pseudotime``)
  * read ``uns["scjdo"]["J_tensor"]`` and ``uns["scjdo"]["t_centers"]``;
    assign each cell the nearest-``t_center`` Jacobian
  * run the SAME scjdo2 opspace spectrum / invariants / behaviour pipeline
    and the SAME gate as the other rows

Writes a JSON that ``scripts/equilibrated_gate.py --v1-json`` consumes.

Invoke with the v1 compare extra installed; see scripts/install_v1_compare.sh.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scjdo2 import opspace
from scjdo2.data import analytic_jacobian, generate_pitchfork_equilibrated


REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "benchmarks" / "phase_b_gate" / "equilibrated" / "v1_slicing.json"


def _pr(pred: np.ndarray, truth: np.ndarray) -> dict:
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    p = tp / (tp + fp) if (tp + fp) else float("nan")
    r = tp / (tp + fn) if (tp + fn) else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r}


def run_tau(tau_crit: float, n_cells: int, n_alpha: int, n_epochs: int,
            seed: int, verbose: bool) -> dict:
    import scjdo as sjd  # v1

    print(f"\n=== v1 slicing on equilibrated τ_crit={tau_crit} ===")
    adata = generate_pitchfork_equilibrated(
        tau_crit=tau_crit, seed=seed,
        n_cells=n_cells, n_alpha=n_alpha,
        n_genes=200, n_latent=20,
        noise_sigma=0.3,
        T_burn=20.0, T_sample=5.0, dt=0.02, sigma_sde=0.20,
        init_box=2.5,
    )
    # Drop uns pitchfork numpy fields that break anndata.write below (we don't
    # write to disk — but fit_drift reads/writes to uns["scjdo"], so leave as).

    t0 = time.monotonic()
    sjd.tl.fit_drift(
        adata, rep="X_pca", time_key="pseudotime",
        n_epochs=n_epochs, seed=seed,
        windowing="kernel", bandwidth="auto",
        archetype_method="snmf", n_archetypes=5, n_restarts=1,
        verbose=verbose,
    )
    dt_fit = time.monotonic() - t0

    J_tensor = np.asarray(adata.uns["scjdo"]["J_tensor"])      # (T_eval, D, D)
    t_centers = np.asarray(adata.uns["scjdo"]["t_centers"])    # (T_eval,)
    t = adata.obs["pseudotime"].to_numpy()
    n = adata.n_obs
    nearest = np.argmin(np.abs(t[:, None] - t_centers[None, :]), axis=1)
    J_per_cell = J_tensor[nearest]        # (n, D, D)

    adata.uns.setdefault("scjdo2", {})
    adata.uns["scjdo2"]["J"] = J_per_cell
    adata.uns["scjdo2"]["meta"] = {"clock_calibrated": False,
                                   "jacobian_method": "v1_slicing_nearest_center"}

    opspace.spectrum(adata, n_eig=5, fisher_mask=False)
    opspace.invariants(adata)
    opspace.behaviour(adata)

    beh = adata.obs["op_behaviour"].astype(str).to_numpy()
    lead_re = adata.obs["lead_re"].to_numpy()
    pred_fork = beh == "fork"
    pred_holder = beh == "holder"

    # Dynamic ground truth: same 0.1·std tolerance as opspace.behaviour.
    alpha = adata.obs["alpha"].to_numpy(dtype=np.float64)
    z_true = np.asarray(adata.obsm["X_true"], dtype=np.float64)
    J_analytic = analytic_jacobian(z_true, alpha)
    analytic_lead_re = np.array([
        float(np.max(np.linalg.eigvals(J_analytic[c]).real))
        for c in range(adata.n_obs)
    ])
    finite = analytic_lead_re[np.isfinite(analytic_lead_re)]
    tol = float(0.1 * np.nanstd(finite)) if finite.size else 0.1
    true_fork = analytic_lead_re > tol
    true_holder = analytic_lead_re < -tol

    return {
        "tau_crit": tau_crit,
        "fit_seconds": dt_fit,
        "T_eval": int(J_tensor.shape[0]),
        "fork_pr": _pr(pred_fork, true_fork),
        "holder_pr": _pr(pred_holder, true_holder),
        "beh_counts": {b: int(np.sum(beh == b)) for b in np.unique(beh) if b},
        "lead_re": {
            "frac_positive": float(np.nanmean(lead_re > 0)),
            "frac_negative": float(np.nanmean(lead_re < 0)),
            "median": float(np.nanmedian(lead_re)),
        },
        "ground_truth": {
            "tol": tol,
            "n_true_fork": int(true_fork.sum()),
            "n_true_holder": int(true_holder.sum()),
            "frac_true_fork": float(np.mean(true_fork)),
        },
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tau-crit", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    p.add_argument("--n-cells", type=int, default=1200)
    p.add_argument("--n-alpha", type=int, default=30)
    p.add_argument("--n-epochs", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    per_tau = []
    for tau in args.tau_crit:
        per_tau.append(run_tau(
            tau_crit=tau, n_cells=args.n_cells, n_alpha=args.n_alpha,
            n_epochs=args.n_epochs, seed=args.seed, verbose=args.verbose,
        ))

    out = {
        "status": "complete",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "per_tau": per_tau,
    }
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[v1 slicing equilibrated] wrote {OUT}")
    for r in per_tau:
        f, h = r["fork_pr"], r["holder_pr"]
        print(f"  τ={r['tau_crit']}: fork P={f['precision']:.2f} R={f['recall']:.2f}  "
              f"holder P={h['precision']:.2f} R={h['recall']:.2f}  "
              f"frac(Re>0)={r['lead_re']['frac_positive']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
