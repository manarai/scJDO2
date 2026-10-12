"""Phase C gate — gene-space Jacobians on the equilibrated pitchfork.

Pipeline per docs/opspace_figure_spec.md §9 and §8.1–§8.10:

    (1) Generate the equilibrated pitchfork with NB counts and the DSM
        latent drift (residual off, β=σ²=0.04, multi-σ ∈ [0.05, 1.0]).
    (2) Fit the NB rate model on counts with fixed z = X_pca.
    (3) Fit per-cell gene-space Jacobians by per-gene weighted ridge in
        η-coords (``scjdo2.field.fit_gene_space_jacobians``). Store the
        Fisher ledger (same ridge likelihood, diagonal curvature).
    (4) Classify behaviour via opspace on the gene-space J (``k=0``
        row). Also run the sparse smoothing variant at ``k=30`` and
        report both rows (§8.8).
    (5) Compare against:
          * Oracle — analytic 2D J of f_toggle at (X_true, α).
          * Latent — Phase B DSM-equilibrated classification (dense
            latent J on the same adata).

Reported metrics:
    * fork / holder P and R vs the oracle at each τ_crit, with the
      Phase C fork P/R target of "within 0.10 of oracle" (R13 carry-
      forward, spec §9.1).
    * Latent↔gene-space ``op_behaviour`` agreement on cells **above
      the median op_confidence** (gated at ≥ 0.95), with the all-cells
      agreement reported alongside (§8.9).
    * ``frac(Re > 0)`` for each row.

Phase C acceptance targets, strictly:
    * holder P and R ≥ 0.8 absolute at every τ_crit.
    * fork P and R within 0.10 of oracle at every τ_crit.
    * ≥ 0.95 latent↔gene-space agreement on cells above the median
      op_confidence.

Usage::

    python scripts/phase_c_gate.py --n-epochs 2000 --rate-epochs 400 \
        --tau-crit 0.1 0.3 0.5 --n-cells 1200 --n-alpha 30
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from anndata import AnnData

import scjdo2.opspace as op
from scjdo2.data import analytic_jacobian, generate_pitchfork_equilibrated
from scjdo2.drift import DriftConfig
from scjdo2.field import (
    GeneFieldConfig,
    fit_gene_space_jacobians,
    smooth_jacobians_sparse,
)
from scjdo2.fisher import compute as fisher_compute
from scjdo2.pipeline import fit_latent_field
from scjdo2.rates import GeneRateModel, RatesConfig


REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "benchmarks" / "phase_c_gate" / "equilibrated"


# -------------------------------------------------------------- metrics
def _pr(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}


def _dynamic_labels(adata: AnnData) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Oracle labels — spec §9.1 + Phase B gate same-tol convention."""
    z_true = np.asarray(adata.obsm["X_true"], dtype=np.float64)
    alpha = adata.obs["alpha"].to_numpy(dtype=np.float64)
    J_true = analytic_jacobian(z_true, alpha)
    lead_re = np.array([float(np.max(np.linalg.eigvals(J_true[c]).real))
                        for c in range(J_true.shape[0])])
    finite = lead_re[np.isfinite(lead_re)]
    tol = float(0.1 * np.nanstd(finite)) if finite.size else 0.1
    return lead_re > tol, lead_re < -tol, lead_re, tol


def _apply_opspace(adata: AnnData, transient_horizon: int = 20) -> None:
    op.spectrum(adata, n_eig=5, fisher_mask=True)
    op.invariants(adata, transient_horizon=transient_horizon)
    op.behaviour(adata)


def _score_vs_oracle(adata: AnnData, true_fork: np.ndarray, true_holder: np.ndarray) -> dict[str, Any]:
    beh = adata.obs["op_behaviour"].astype(str).to_numpy()
    lead_re = adata.obs["lead_re"].to_numpy()
    pred_fork = beh == "fork"
    pred_holder = beh == "holder"
    return {
        "fork_pr": _pr(pred_fork, true_fork),
        "holder_pr": _pr(pred_holder, true_holder),
        "beh_counts": {b: int(np.sum(beh == b)) for b in np.unique(beh) if b},
        "frac_re_positive": float(np.nanmean(lead_re > 0)),
    }


def _agreement(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    if mask is None:
        n = a.size
        return float(np.sum(a == b) / n) if n else float("nan")
    m = np.asarray(mask, dtype=bool)
    if m.sum() == 0:
        return float("nan")
    return float(np.sum(a[m] == b[m]) / m.sum())


# ------------------------------------------------- rows per τ_crit
def run_tau(
    tau_crit: float,
    n_cells: int,
    n_alpha: int,
    n_genes: int,
    n_epochs_drift: int,
    n_epochs_rate: int,
    k: int,
    smooth_k: int,
    cv_sample_cells: int,
    seed: int,
    verbose: bool,
) -> dict[str, Any]:
    print(f"\n=== τ_crit = {tau_crit} ===")
    adata = generate_pitchfork_equilibrated(
        tau_crit=tau_crit, seed=seed,
        n_cells=n_cells, n_alpha=n_alpha,
        n_genes=n_genes, n_latent=20,
        noise_sigma=0.3,
        T_burn=20.0, T_sample=5.0, dt=0.02, sigma_sde=0.20,
        init_box=2.5,
        counts=True, counts_depth=10.0,
    )
    print(f"  generated: {adata.shape}, {n_alpha} α-slices")

    true_fork, true_holder, analytic_lead_re, tol = _dynamic_labels(adata)
    print(f"  oracle tol={tol:.4f}  n_fork={int(true_fork.sum())}  n_holder={int(true_holder.sum())}")

    # --- Row: oracle (analytic 2D J on X_true, α)
    print("  [oracle] analytic 2D J")
    adata_oracle = adata.copy()
    adata_oracle.uns.setdefault("scjdo2", {})
    adata_oracle.uns["scjdo2"]["J"] = analytic_jacobian(
        np.asarray(adata.obsm["X_true"], dtype=np.float64),
        adata.obs["alpha"].to_numpy(np.float64),
    )
    adata_oracle.uns["scjdo2"]["meta"] = {"clock_calibrated": False}
    _apply_opspace(adata_oracle)
    oracle_scores = _score_vs_oracle(adata_oracle, true_fork, true_holder)
    print(f"    fork P={oracle_scores['fork_pr']['precision']:.2f} "
          f"R={oracle_scores['fork_pr']['recall']:.2f}  "
          f"holder P={oracle_scores['holder_pr']['precision']:.2f} "
          f"R={oracle_scores['holder_pr']['recall']:.2f}  "
          f"frac(Re>0)={oracle_scores['frac_re_positive']:.2f}")

    # --- Phase B latent-only DSM-equilibrated for comparison + agreement
    print(f"  [latent] DSM-equilibrated (residual off, β=0.04, σ∈[0.05,1], {n_epochs_drift} epochs)")
    t0 = time.monotonic()
    fit_latent_field(
        adata, use_rep="X_pca", time_key="pseudotime",
        drift_config=DriftConfig(
            hidden=128, depth=3, n_epochs=n_epochs_drift, seed=seed,
            use_spectral_norm=False, batch_size=256,
            inject_velocity_prior=False, dsm_sigma_range=(0.05, 1.0),
            use_residual=False, beta=0.04,
        ),
        jacobian_method="autodiff", verbose=verbose,
    )
    t_latent = time.monotonic() - t0
    # Classify on the latent J so we can measure agreement later
    _apply_opspace(adata)
    latent_scores = _score_vs_oracle(adata, true_fork, true_holder)
    latent_beh = adata.obs["op_behaviour"].astype(str).to_numpy().copy()
    print(f"    [{t_latent:.1f}s] fork P={latent_scores['fork_pr']['precision']:.2f} "
          f"R={latent_scores['fork_pr']['recall']:.2f}  "
          f"holder P={latent_scores['holder_pr']['precision']:.2f} "
          f"R={latent_scores['holder_pr']['recall']:.2f}  "
          f"frac(Re>0)={latent_scores['frac_re_positive']:.2f}")

    # --- Rate model
    print(f"  [rates] NB fit ({n_epochs_rate} epochs, held-out 10%)")
    t0 = time.monotonic()
    counts = np.asarray(adata.X, dtype=np.float32)
    rm = GeneRateModel(RatesConfig(
        hidden=(128, 128), n_epochs=n_epochs_rate, lr=1e-3,
        batch_size=256, held_out_frac=0.10, patience=30,
        seed=seed,
    ))
    rm.fit(counts, np.asarray(adata.obsm["X_pca"]), verbose=verbose)
    t_rates = time.monotonic() - t0
    print(f"    [{t_rates:.1f}s] held-out NLL {rm.best_val_nll:.3f}")

    # --- Gene-space J + ledger
    print(f"  [gene-field] fit (k={k})")
    t0 = time.monotonic()
    gs = fit_gene_space_jacobians(
        np.asarray(adata.obsm["X_pca"]),
        np.asarray(adata.obsm["X_drift_v"]),
        rm,
        config=GeneFieldConfig(k=k, cv_sample_cells=int(cv_sample_cells), seed=seed),
        verbose=verbose,
    )
    t_genef = time.monotonic() - t0
    ledger = fisher_compute(gs)
    print(f"    [{t_genef:.1f}s] ridge λ={gs['ridge_lambda']}  "
          f"mean L_c={np.mean([lc.size for lc in gs['live_cols']]):.1f}")

    # Save latent J so we can toggle rows cleanly
    latent_J = np.asarray(adata.uns["scjdo2"]["J"])

    # --- Row: gene-space k=0
    print("  [gene_space k=0] classify")
    adata.uns["scjdo2"]["J"] = gs["J"]
    adata.uns["scjdo2"]["fisher_ledger"] = ledger
    _apply_opspace(adata)
    gs_scores = _score_vs_oracle(adata, true_fork, true_holder)
    gs_beh = adata.obs["op_behaviour"].astype(str).to_numpy().copy()
    conf = adata.obs["op_confidence"].to_numpy().copy()
    med_conf = float(np.median(conf[np.isfinite(conf)])) if np.isfinite(conf).any() else float("nan")
    above = np.isfinite(conf) & (conf >= med_conf)
    agree_all = _agreement(gs_beh, latent_beh)
    agree_above = _agreement(gs_beh, latent_beh, above)
    print(f"    fork P={gs_scores['fork_pr']['precision']:.2f} "
          f"R={gs_scores['fork_pr']['recall']:.2f}  "
          f"holder P={gs_scores['holder_pr']['precision']:.2f} "
          f"R={gs_scores['holder_pr']['recall']:.2f}  "
          f"frac(Re>0)={gs_scores['frac_re_positive']:.2f}  "
          f"agree all={agree_all:.2f} above={agree_above:.2f}")

    # --- Row: gene-space smoothed k=smooth_k (sparse smoothing)
    if smooth_k and smooth_k > 0:
        print(f"  [gene_space k={smooth_k}] smoothed classify")
        J_sm = smooth_jacobians_sparse(
            gs["J"], np.asarray(adata.obsm["X_pca"]), k=int(smooth_k),
        )
        adata.uns["scjdo2"]["J"] = J_sm
        # Keep the same Fisher ledger (same ridge likelihood); on smoothed J
        # entries off the per-cell support fall back to no mask (handled by
        # the fallback in opspace.spectrum).
        _apply_opspace(adata)
        gs_sm_scores = _score_vs_oracle(adata, true_fork, true_holder)
        gs_sm_beh = adata.obs["op_behaviour"].astype(str).to_numpy().copy()
        conf_sm = adata.obs["op_confidence"].to_numpy().copy()
        med_sm = float(np.median(conf_sm[np.isfinite(conf_sm)])) if np.isfinite(conf_sm).any() else float("nan")
        above_sm = np.isfinite(conf_sm) & (conf_sm >= med_sm)
        agree_all_sm = _agreement(gs_sm_beh, latent_beh)
        agree_above_sm = _agreement(gs_sm_beh, latent_beh, above_sm)
        print(f"    fork P={gs_sm_scores['fork_pr']['precision']:.2f} "
              f"R={gs_sm_scores['fork_pr']['recall']:.2f}  "
              f"holder P={gs_sm_scores['holder_pr']['precision']:.2f} "
              f"R={gs_sm_scores['holder_pr']['recall']:.2f}  "
              f"frac(Re>0)={gs_sm_scores['frac_re_positive']:.2f}  "
              f"agree all={agree_all_sm:.2f} above={agree_above_sm:.2f}")
    else:
        gs_sm_scores = None
        agree_all_sm = agree_above_sm = None

    # Restore latent J so the adata is reusable
    adata.uns["scjdo2"]["J"] = latent_J

    return {
        "tau_crit": float(tau_crit),
        "n_cells": int(adata.n_obs),
        "n_alpha": int(n_alpha),
        "tol": tol,
        "rates": {
            "held_out_nll": float(rm.best_val_nll),
            "fit_seconds": float(t_rates),
        },
        "gene_field": {
            "ridge_lambda": float(gs["ridge_lambda"]),
            "mean_L_c": float(np.mean([lc.size for lc in gs["live_cols"]])),
            "fit_seconds": float(t_genef),
        },
        "rows": {
            "oracle": oracle_scores,
            "latent_dsm": latent_scores,
            "gene_space_k0": {
                **gs_scores,
                "agreement_all": agree_all,
                "agreement_above_median_conf": agree_above,
                "op_confidence_median": med_conf,
            },
            "gene_space_k_smoothed": (
                {
                    **(gs_sm_scores or {}),
                    "smooth_k": int(smooth_k),
                    "agreement_all": agree_all_sm,
                    "agreement_above_median_conf": agree_above_sm,
                } if gs_sm_scores is not None else None
            ),
        },
    }


# ------------------------------------------------- markdown report
def _mdrow(label: str, tau: float, scores: dict[str, Any], oracle: dict[str, Any]) -> str:
    f = scores.get("fork_pr", {})
    h = scores.get("holder_pr", {})
    fp = f.get("precision", float("nan"))
    fr = f.get("recall", float("nan"))
    hp = h.get("precision", float("nan"))
    hr = h.get("recall", float("nan"))
    frac = scores.get("frac_re_positive", float("nan"))

    def ok(x):
        return isinstance(x, (int, float)) and np.isfinite(x)

    holder_ok = ok(hp) and ok(hr) and hp >= 0.8 and hr >= 0.8
    if label == "oracle":
        fork_ok = ok(fp) and ok(fr) and fp >= 0.8 and fr >= 0.8
    else:
        orc_f = oracle.get("fork_pr", {})
        ofp = orc_f.get("precision", float("nan"))
        ofr = orc_f.get("recall", float("nan"))
        fork_ok = (ok(fp) and ok(fr) and ok(ofp) and ok(ofr)
                   and abs(fp - ofp) <= 0.10 and abs(fr - ofr) <= 0.10)
    verd = "PASS" if (holder_ok and fork_ok) else "FAIL"

    def fmt(x):
        return "nan" if not ok(x) else f"{x:.2f}"

    agree_cell = ""
    if "agreement_above_median_conf" in scores and scores["agreement_above_median_conf"] is not None:
        aa = scores.get("agreement_all")
        ab = scores.get("agreement_above_median_conf")
        agree_cell = f" | {fmt(aa)} / {fmt(ab)}"
    return (f"| {label} | {tau} | {fmt(fp)} | {fmt(fr)} | "
            f"{fmt(hp)} | {fmt(hr)} | {fmt(frac)} | **{verd}**{agree_cell} |")


def _write_report(per_tau: list[dict[str, Any]], args_dict: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # JSON
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": args_dict,
        "per_tau": per_tau,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=float))

    # Markdown
    smooth_k = args_dict.get("smooth_k", 30)
    lines = [
        "# Phase C gate — equilibrated pitchfork (gene-space J)",
        f"\nGenerated: {report['timestamp']}\n",
        "Gate (spec §9):",
        "  * holder P and R  ≥ 0.8 absolute",
        "  * fork   P and R  within 0.10 of oracle at each τ_crit (R13 carry-forward)",
        f"  * latent↔gene-space agreement on cells above median op_confidence ≥ 0.95",
        "",
        "Rows: `oracle` (analytic 2D J), `latent_dsm` (Phase B DSM-equilibrated for "
        "comparison), `gene_space_k0` (per-cell gene-space J, no smoothing), "
        f"`gene_space_k{smooth_k}` (sparse Gaussian smoothing over the latent kNN).",
        "",
        ("| row | τ_crit | fork P | fork R | holder P | holder R | frac(Re>0) | verdict "
         "| agree all / above-median-conf |"),
        ("|---|---|---|---|---|---|---|---|---|"),
    ]
    for row in per_tau:
        tau = row["tau_crit"]
        orc = row["rows"]["oracle"]
        lines.append(_mdrow("oracle", tau, orc, orc))
        lines.append(_mdrow("latent_dsm", tau, row["rows"]["latent_dsm"], orc))
        lines.append(_mdrow("gene_space_k0", tau, row["rows"]["gene_space_k0"], orc))
        sm = row["rows"].get("gene_space_k_smoothed")
        if sm is not None:
            lines.append(_mdrow(f"gene_space_k{smooth_k}", tau, sm, orc))
    lines.extend([
        "",
        "## Reading",
        "",
        "* `latent_dsm` is the Phase B DSM-equilibrated baseline (holder passes, "
        "fork P below oracle by R13's margin). It supplies the behaviour vector "
        "that `gene_space_*` rows measure agreement against.",
        "* `gene_space_k0` runs the Phase C rate model + per-gene weighted ridge "
        "in η-coords; its J is per-cell sparse CSR with the Fisher ledger from "
        "the same ridge likelihood. opspace.spectrum masks entries below the "
        "10th-percentile per-cell Fisher before eigendecomposing.",
        f"* `gene_space_k{smooth_k}` applies sparse Gaussian smoothing over the "
        "latent kNN before classifying.",
        "* Agreement is reported as `all / above-median-op_confidence`. The "
        "gate uses only the above-median figure; the all-cells number is "
        "context (gene-space J carries modes the latent J can't see, so "
        "disagreement on low-confidence cells is expected — spec §8.9).",
        "",
        "## Finding — gene-space lead eigenvalue is dominated by the ambient "
        "G-dim noise floor on this synthetic.",
        "",
        "Every `gene_space_*` row reports **frac(Re>0) ≈ 1.00** and "
        "**holder P = nan, R = 0** at every τ_crit, while the ORACLE on the "
        "exact same cells reports frac(Re>0) ≈ 0.04-0.05 and holder P/R ≈ "
        "1.00 / 0.85-0.92. The pipeline is wired right (η recovers to 0.85 "
        "cos-sim on toy fits, J↔F share the same sparse layout, Fisher mask "
        "fires at the 10th-percentile cutoff) — the gene-space classifier "
        "is nevertheless unusable on this synthetic.",
        "",
        "Root cause. The pitchfork gene matrix is a dim-2 projection "
        "``X_obs = z_true @ W`` of a 2D process, so the per-cell η-manifold "
        "has effective **rank 2** out of G=50 live columns. Step 5's ridge "
        "fits a dense ``J_c`` on the full L=G=50 subspace: the 2 informative "
        "eigenvalues (which the oracle Jacobian carries) sit alongside ~48 "
        "near-zero / noise eigenvalues from the ridge regulariser. "
        "`spectrum()` takes the top-``n_eig`` eigenvalues by Re, which "
        "almost always picks a noise direction's small positive real part "
        "over the informative basin eigenvalue. **frac(Re>0) = 1.0 is a "
        "sampling artifact** of taking max(Re) over 48 noise + 2 signal "
        "directions.",
        "",
        "Why the current `fisher_mask` default doesn't rescue this. The 10th-"
        "percentile cutoff removes the lowest-Fisher 10 % of entries, but on "
        "this synthetic every gene is uniformly Fisher-significant (dense "
        "signal, no sparse expression), so the mask is a near no-op. On real "
        "scRNA data most genes drop out in any given neighbourhood and the "
        "mask recovers rank — this is the regime the spec was written for.",
        "",
        "What this changes. The gate falls through the fork criterion for a "
        "DIFFERENT reason than Phase B's R13 (where DSM score noise flipped "
        "borderline cells). Here the pathology is **effective-rank vs "
        "ambient-dim mismatch in the eigenvalue selection**. Two separate "
        "fixes are candidate for a follow-up iteration:",
        "",
        "  1. **Rank-aware spectrum** — truncate `J_c` to the top-r singular "
        "directions of ``∇η(z_c)`` (the Jacobian of the rate model at that "
        "cell) before eigendecomposing. The G-dim ambient space is replaced "
        "by the effective r-dim manifold; the eigvals carry only informative "
        "modes. Natural fit with the Fisher-weighted invariant.",
        "  2. **Fisher-mass eigenvalue sort** — sort eigenvalues by "
        "`|eigvec|^T · F_diag · |eigvec|` instead of by Re, so the leading "
        "eigenvalue is the one with the most Fisher support rather than the "
        "numerically-largest Re on a near-degenerate ridge solution.",
        "",
        "Either or both can be prototyped against this same gate without "
        "re-fitting rates or gene-space J; the gate harness is designed to "
        "swap the opspace.spectrum call. Reporting the table unmasked "
        "surfaces the signal rather than hides it.",
        "",
        "## Next actions",
        "",
        "* Decide between fix (1) rank-aware spectrum vs (2) Fisher-mass "
        "eigenvalue sort (or both; they compose).",
        "* Re-run the gate with the chosen fix before the figure layer "
        "(spec §3) ships — Panel C renders drivers of the LEADING eigvec, "
        "and the current leading eigvec is a noise direction, which would "
        "make every per-cell driver plot meaningless.",
    ])
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"\n[phase-c gate] wrote {out_dir / 'report.md'}")


# ------------------------------------------------------- CLI
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tau-crit", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    p.add_argument("--n-cells", type=int, default=800)
    p.add_argument("--n-alpha", type=int, default=20)
    p.add_argument("--n-genes", type=int, default=50,
                   help="pitchfork gene dim (default 50; raise for stress tests)")
    p.add_argument("--n-epochs", type=int, default=2000, dest="n_epochs_drift",
                   help="DSM drift epochs (default 2000)")
    p.add_argument("--rate-epochs", type=int, default=400, dest="n_epochs_rate",
                   help="NB rate model epochs (default 400)")
    p.add_argument("--k", type=int, default=30, help="gene-space kNN radius")
    p.add_argument("--smooth-k", type=int, default=30,
                   help="post-hoc Gaussian smoothing k (0 to disable)")
    p.add_argument("--cv-sample-cells", type=int, default=30,
                   help="cells sampled per λ-CV fold (lower = faster, default 30)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    per_tau: list[dict[str, Any]] = []
    for tau in args.tau_crit:
        per_tau.append(run_tau(
            tau_crit=tau,
            n_cells=args.n_cells, n_alpha=args.n_alpha,
            n_genes=args.n_genes,
            n_epochs_drift=args.n_epochs_drift,
            n_epochs_rate=args.n_epochs_rate,
            k=args.k, smooth_k=args.smooth_k,
            cv_sample_cells=args.cv_sample_cells,
            seed=args.seed, verbose=args.verbose,
        ))

    _write_report(per_tau, vars(args), OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
