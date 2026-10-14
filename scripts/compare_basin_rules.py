"""Continuous evaluation of three basin-classifier rules on the
equilibrated pitchfork (SUMMARY §5.9 close-out).

Spec (user, 2026-10-14 final):

Instead of adding a fifth rule, evaluate the three existing ones
with CONTINUOUS metrics and pick the one with the best AUC:

  * **Spearman** of ``basin_index`` vs analytic ``lead_re`` per cell.
  * **AUC** of ``basin_index`` as a score separating:
      POSITIVE: true-holder cells (``analytic lead_re < −tol``);
      NEGATIVE: cells within ``±0.05`` of ``τ_crit`` in α
                (saddle neighbourhood).
    Higher AUC → ``basin_index`` better ranks basin above saddle.
  * **Median ``|α − τ_crit|``** of cells called *fork* by the rule,
    vs the same statistic on random cells drawn without replacement.
    Lower ratio → fork calls concentrate near the saddle.

For rule 1 (lead-eigenvalue), ``basin_index = −lead_re`` as the
natural signed surrogate.

**Gate** (locked criterion, replaces the Phase-B strict P-and-R
window): ``holder P ≥ 0.8``, ``holder R ≥ 0.8``, ``fork R ≥ 0.8``.
Fork precision in a ±0.05 saddle window is bounded by the
neighbourhood width (many cells near the saddle have analytic
lead_re near zero → "drift" ground truth → fork precision ceilings
well below 1.00). Not a gate criterion.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from equilibrated_gate import _dynamic_labels, _pr               # type: ignore

from scjdo2.data import generate_pitchfork_equilibrated
from scjdo2.energy_drift import EnergyDriftAdapter, EnergyDriftConfig
from scjdo2.opspace import classify_basin_latent, null_tol_pos


OUT = Path(__file__).resolve().parent.parent / "benchmarks" / "phase_b_gate" / "equilibrated"


# -----------------------------------------------------------------
# Rule evaluators — each returns (behaviour, basin_index) per cell
# -----------------------------------------------------------------


def rule_lead_eig(adata, J: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rule 1 (legacy opspace.behaviour): classify by lead_re sign.
    basin_index surrogate = −lead_re."""
    n = J.shape[0]
    lead = np.array([
        float(np.max(np.linalg.eigvals(Jc.astype(np.float64)).real))
        for Jc in J
    ])
    finite = lead[np.isfinite(lead)]
    tol = float(0.1 * np.std(finite)) if finite.size else 0.1
    beh = np.full(n, "", dtype=object)
    beh[lead > +tol] = "fork"
    beh[lead < -tol] = "holder"
    beh[(lead >= -tol) & (lead <= +tol)] = "drift"
    return beh, -lead


def rule_single_tol(adata, J: np.ndarray,
                    tol_neg: float) -> tuple[np.ndarray, np.ndarray]:
    """Rule 2 (§5.8): magnitude rule on Σ-tangent, single tol."""
    adata.uns.setdefault("scjdo2", {})["J_latent"] = J
    adata.uns["scjdo2"]["J"] = J
    classify_basin_latent(adata, tol=tol_neg, tol_pos=tol_neg,
                           use_rep="X_pca")
    beh = adata.obs["op_behaviour_basin"].astype(str).to_numpy()
    bi = adata.obs["op_basin_index"].to_numpy(dtype=np.float64)
    return beh, bi


def rule_two_tol(adata, J: np.ndarray, tol_neg: float,
                  tol_pos: float) -> tuple[np.ndarray, np.ndarray]:
    """Rule 3 (§5.9+): two thresholds — tol_neg from oracle,
    tol_pos from permutation null (floored at 1e-6 × max|λ|)."""
    adata.uns.setdefault("scjdo2", {})["J_latent"] = J
    adata.uns["scjdo2"]["J"] = J
    classify_basin_latent(adata, tol=tol_neg, tol_pos=tol_pos,
                           use_rep="X_pca")
    beh = adata.obs["op_behaviour_basin"].astype(str).to_numpy()
    bi = adata.obs["op_basin_index"].to_numpy(dtype=np.float64)
    return beh, bi


# -----------------------------------------------------------------
# Continuous metrics
# -----------------------------------------------------------------


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation on finite entries."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10:
        return float("nan")
    from scipy.stats import spearmanr
    rho, _ = spearmanr(x[m], y[m])
    return float(rho)


def auc_roc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC of ``scores`` separating ``labels == 1`` from
    ``labels == 0``. Higher score ↔ class 1."""
    m = np.isfinite(scores) & np.isfinite(labels)
    s = scores[m]
    y = labels[m].astype(bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def median_dist_fork(beh: np.ndarray, alpha: np.ndarray,
                     tau_crit: float, rng: np.random.Generator,
                     ) -> dict:
    fork_mask = beh == "fork"
    d_all = np.abs(alpha - tau_crit)
    if fork_mask.sum() == 0:
        return {"fork_median_dist": float("nan"),
                "random_median_dist": float("nan"),
                "ratio": float("nan"),
                "n_fork": 0}
    rand_idx = rng.choice(len(alpha), size=int(fork_mask.sum()),
                           replace=False)
    fork_med = float(np.median(d_all[fork_mask]))
    rand_med = float(np.median(d_all[rand_idx]))
    return {"fork_median_dist": fork_med,
            "random_median_dist": rand_med,
            "ratio": fork_med / rand_med if rand_med > 0 else float("nan"),
            "n_fork": int(fork_mask.sum())}


# -----------------------------------------------------------------
# Gate (locked criterion)
# -----------------------------------------------------------------


def pr_gate(beh: np.ndarray, true_fork: np.ndarray,
            true_holder: np.ndarray) -> dict:
    pred_fork = beh == "fork"
    pred_hold = beh == "holder"
    fork_pr = _pr(pred_fork, true_fork)
    hold_pr = _pr(pred_hold, true_holder)
    hold_ok = (
        (not np.isnan(hold_pr["precision"]) and hold_pr["precision"] >= 0.8)
        and (not np.isnan(hold_pr["recall"]) and hold_pr["recall"] >= 0.8)
    )
    fork_recall_ok = (
        not np.isnan(fork_pr["precision"])
        and not np.isnan(fork_pr["recall"])
        and fork_pr["recall"] >= 0.8
    )
    return {"fork_pr": fork_pr, "holder_pr": hold_pr,
            "holder_pass": bool(hold_ok),
            "fork_recall_pass": bool(fork_recall_ok),
            "gate_pass": bool(hold_ok and fork_recall_ok)}


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tau-crit", type=float, nargs="+",
                   default=[0.1, 0.3, 0.5])
    p.add_argument("--n-cells", type=int, default=1200)
    p.add_argument("--n-alpha", type=int, default=20)
    p.add_argument("--n-epochs", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--saddle-window", type=float, default=0.05)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    def _fit_fn(z_in, s):
        return EnergyDriftAdapter(dim=z_in.shape[1], config=EnergyDriftConfig(
            hidden=128, depth=3, n_epochs=max(500, args.n_epochs // 3),
            seed=int(s), batch_size=256, loss="esm", beta=1.0,
            esm_laplacian="exact" if z_in.shape[1] <= 20 else "hutchinson",
        )).fit(z_in, verbose=False)

    all_rows = []
    for tau in args.tau_crit:
        print(f"\n=== τ_crit = {tau} ===")
        adata = generate_pitchfork_equilibrated(
            tau_crit=tau, seed=args.seed,
            n_cells=args.n_cells, n_alpha=args.n_alpha,
            n_genes=200, n_latent=20, noise_sigma=0.3,
            T_burn=20.0, T_sample=5.0, dt=0.02, sigma_sde=0.20,
            init_box=2.5,
        )
        z = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
        # Saddle axis is the normalised PSEUDOTIME (not raw α, which is
        # in physical units). The equilibrated generator maps the α-ramp
        # window [α_min, α_max] linearly onto pseudotime ∈ [0, 1], with
        # the saddle at pseudotime = tau_crit.
        alpha = adata.obs["pseudotime"].to_numpy(dtype=np.float64)

        # Fit ESM once.
        t0 = time.monotonic()
        drift = EnergyDriftAdapter(dim=z.shape[1], config=EnergyDriftConfig(
            hidden=128, depth=3, n_epochs=args.n_epochs, seed=args.seed,
            batch_size=256, loss="esm", beta=1.0,
            esm_laplacian="exact" if z.shape[1] <= 20 else "hutchinson",
        )).fit(z, verbose=False)
        J = drift.jacobian_autodiff(z)
        print(f"  ESM fit + J {time.monotonic()-t0:.1f}s")

        # Compute null tol_pos (shared by rule 3).
        t0 = time.monotonic()
        null = null_tol_pos(z, _fit_fn, n_perm=10, percentile=95.0,
                             seed=args.seed)
        print(f"  null tol_pos = {null['tol_pos']:.4f}  {time.monotonic()-t0:.1f}s")

        # Dynamic ground truth.
        true_fork, true_holder, analytic_lead = _dynamic_labels(adata)
        tol_neg = float(0.1 * np.nanstd(analytic_lead))

        # Build the (positive, negative) sets for AUC:
        #   positive = holder (basin-centred cells),
        #   negative = cells in the saddle neighbourhood ±window.
        pos_mask = true_holder.astype(bool)
        neg_mask = np.abs(alpha - tau) <= args.saddle_window
        auc_labels = np.full(len(alpha), -1, dtype=int)
        auc_labels[pos_mask & ~neg_mask] = 1       # true holder, not near-saddle
        auc_labels[neg_mask & ~pos_mask] = 0       # saddle neighbourhood, not holder
        in_auc = auc_labels >= 0

        rng = np.random.default_rng(args.seed)

        rules = []
        rules.append(("lead_eig",) + rule_lead_eig(adata, J))
        rules.append(("single_tol",) + rule_single_tol(adata, J, tol_neg))
        rules.append(("two_tol_null",) + rule_two_tol(adata, J, tol_neg,
                                                       null["tol_pos"]))

        for name, beh, bi in rules:
            rho = spearman(bi, -analytic_lead)   # bi positive == basin == negative analytic lead_re
            auc = auc_roc(bi[in_auc], auc_labels[in_auc].astype(float))
            dist = median_dist_fork(beh, alpha, tau, rng)
            g = pr_gate(beh, true_fork, true_holder)
            n_hold = int((beh == "holder").sum())
            n_fork = int((beh == "fork").sum())
            n_drift = int((beh == "drift").sum())
            row = {
                "tau_crit": tau, "rule": name,
                "spearman_bi_vs_neg_analytic_lead": rho,
                "auc_holder_vs_saddle": auc,
                "fork_dist_median": dist["fork_median_dist"],
                "random_dist_median": dist["random_median_dist"],
                "fork_dist_ratio": dist["ratio"],
                "n_holder": n_hold, "n_fork": n_fork, "n_drift": n_drift,
                "n_true_holder": int(true_holder.sum()),
                "n_true_fork": int(true_fork.sum()),
                "holder_P": g["holder_pr"]["precision"],
                "holder_R": g["holder_pr"]["recall"],
                "fork_P": g["fork_pr"]["precision"],
                "fork_R": g["fork_pr"]["recall"],
                "holder_pass": g["holder_pass"],
                "fork_recall_pass": g["fork_recall_pass"],
                "gate_pass": g["gate_pass"],
            }
            all_rows.append(row)
            print(f"  [{name:14}] Spearman={rho:+.3f}  AUC={auc:.3f}  "
                  f"fork_dist={dist['fork_median_dist']:.3f}/{dist['random_median_dist']:.3f}  "
                  f"holder P/R={g['holder_pr']['precision']:.2f}/{g['holder_pr']['recall']:.2f}  "
                  f"fork P/R={g['fork_pr']['precision']:.2f}/{g['fork_pr']['recall']:.2f}  "
                  f"gate={'PASS' if g['gate_pass'] else 'FAIL'}")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "rule_comparison.csv", index=False)
    # Pick: average AUC across τ, pick max.
    avg_auc = df.groupby("rule")["auc_holder_vs_saddle"].mean().sort_values(
        ascending=False)
    best = str(avg_auc.index[0])
    all_pass = (df[df["rule"] == best]["gate_pass"].all())
    print(f"\nRule comparison (mean AUC across τ):")
    print(avg_auc.to_string())
    print(f"\n>>> Best AUC rule: {best}  gate_pass_all_tau={all_pass}")

    summary = {
        "winner": best,
        "mean_auc_by_rule": avg_auc.to_dict(),
        "all_tau_gate_pass_for_winner": bool(all_pass),
        "gate_criterion": "holder P/R >= 0.8 AND fork recall >= 0.8",
        "note": "fork precision in ±0.05 saddle window bounded by neighbourhood width; not a gate criterion",
    }
    (OUT / "rule_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(f"\n[compare] outputs → {OUT}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
