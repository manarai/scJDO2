"""Phase-B gate report — holder/fork precision+recall + permutation null.

Three sections, written to ``benchmarks/phase_b_gate/report.json`` and a human
readable ``report.md``:

(1) Pitchfork precision & recall on the strict ≥ 0.8 target.
    For each τ_crit ∈ {0.1, 0.3, 0.5}:
      true_fork(c)   = |t(c) - τ_crit| < 0.05
      true_holder(c) = |t(c) - τ_crit| ≥ 0.15                (buffered)
    pred from ``op_behaviour``. Report precision/recall for both classes.

(2) Permutation null for Re(λ_max). For a stratified sample of cells, draw
    ``n_perm`` shuffles of neighbour (z, v) pairings inside the ball, refit
    the ridge regression, record Re(λ_max). Report null median / 95-percentile
    and the overlap (two-sided P(real ≤ null)) between the observed Re(λ_max)
    and its null, averaged over the sample.

(3) Paul15 zero-holder diagnosis. Fit twice:
      (a) default: pseudotime-anchored velocity prior ON.
      (b) DSM-only: ``inject_velocity_prior=False``.
    Report behaviour counts and permutation-null overlap in both.

Usage:
    python scripts/phase_b_gate_report.py --n-perm 40
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np

from scjdo2 import opspace
from scjdo2.data import generate_pitchfork
from scjdo2.drift import DriftAdapter, DriftConfig
from scjdo2.field import FieldConfig, JacobianField
from scjdo2.pipeline import fit_latent_field


REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "benchmarks" / "phase_b_gate"
V1_PAUL15 = REPO / "benchmarks" / "v1_baseline" / "paul15" / "adata.h5ad"


# ------------------------------------------------------------------ helpers

def _precision_recall(pred_mask: np.ndarray, true_mask: np.ndarray) -> dict[str, float]:
    tp = int(np.sum(pred_mask & true_mask))
    fp = int(np.sum(pred_mask & ~true_mask))
    fn = int(np.sum(~pred_mask & true_mask))
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}


def _overlap(real: np.ndarray, null: np.ndarray) -> float:
    """P(null >= real) — two-sided overlap. 0 ↔ real is clearly separated,
    1 ↔ real is indistinguishable from null. We report the mean over cells of
    min(P(null>=real), P(null<=real)) so the number lives in [0, 0.5]."""
    real = np.asarray(real, dtype=np.float64)
    null = np.asarray(null, dtype=np.float64)
    assert null.shape[0] == real.shape[0]
    out = np.full(real.shape[0], np.nan, dtype=np.float64)
    for i in range(real.shape[0]):
        if not np.isfinite(real[i]):
            continue
        row = null[i][np.isfinite(null[i])]
        if row.size == 0:
            continue
        p_ge = float(np.mean(row >= real[i]))
        p_le = float(np.mean(row <= real[i]))
        out[i] = min(p_ge, p_le)
    return out


# ---------------------------------------------------------------- pitchfork

def pitchfork_report(tau_crit: float, n_cells: int, n_epochs: int, k: int,
                     seed: int, n_perm: int, verbose: bool) -> dict[str, Any]:
    print(f"\n[pitchfork τ_crit={tau_crit}] generate")
    adata = generate_pitchfork(
        tau_crit=tau_crit, seed=seed,
        n_cells=n_cells, n_genes=200, n_latent=20,
        noise_sigma=0.3, T_total=10.0, dt=0.02, sigma_sde=0.20,
    )
    t0 = time.monotonic()
    fit_latent_field(
        adata, use_rep="X_pca", time_key="pseudotime",
        drift_config=DriftConfig(hidden=128, depth=3, n_epochs=n_epochs,
                                 seed=seed, use_spectral_norm=False,
                                 batch_size=256),
        field_config=FieldConfig(k=k, ridge=1e-3, seed=seed),
        verbose=verbose,
    )
    dt_fit = time.monotonic() - t0
    print(f"  fit {dt_fit:.1f}s")

    opspace.spectrum(adata, n_eig=5, fisher_mask=False)
    opspace.invariants(adata)
    opspace.behaviour(adata)
    opspace.field_gradient(adata, k=15)

    t = adata.obs["pseudotime"].to_numpy()
    beh = adata.obs["op_behaviour"].astype(str).to_numpy()
    lead_re = adata.obs["lead_re"].to_numpy()

    pred_fork = beh == "fork"
    pred_holder = beh == "holder"
    true_fork = np.abs(t - tau_crit) < 0.05
    true_holder = np.abs(t - tau_crit) >= 0.15      # buffered — not fork

    fork_metrics = _precision_recall(pred_fork, true_fork)
    holder_metrics = _precision_recall(pred_holder, true_holder)

    # Behaviour counts
    beh_counts = {b: int(np.sum(beh == b)) for b in np.unique(beh) if b}

    # Permutation null — stratified sample: 15 near-τ_crit, 15 far-from-τ_crit
    rng = np.random.default_rng(seed)
    near_idx = np.where(true_fork)[0]
    far_idx = np.where(true_holder)[0]
    n_near = min(15, near_idx.size)
    n_far = min(15, far_idx.size)
    sample = np.concatenate([
        rng.choice(near_idx, size=n_near, replace=False) if n_near else np.array([], dtype=np.int64),
        rng.choice(far_idx, size=n_far, replace=False) if n_far else np.array([], dtype=np.int64),
    ])
    jf = JacobianField(FieldConfig(k=k, ridge=1e-3, seed=seed))
    z = adata.obsm["X_pca"].astype(np.float32)
    v = adata.obsm["X_drift_v"].astype(np.float32)
    jf.fit(z, v)
    null = jf.permutation_null(z, v, n_perm=n_perm, subset_idx=sample, seed=seed)
    overlap = _overlap(lead_re[sample], null)
    overlap_near = overlap[:n_near]
    overlap_far = overlap[n_near:]

    return {
        "tau_crit": tau_crit,
        "n_cells": int(adata.n_obs),
        "fit_seconds": dt_fit,
        "behaviour_counts": beh_counts,
        "fork_precision_recall": fork_metrics,
        "holder_precision_recall": holder_metrics,
        "lead_re_summary": {
            "near_tau_median": float(np.nanmedian(lead_re[near_idx])) if near_idx.size else None,
            "far_from_tau_median": float(np.nanmedian(lead_re[far_idx])) if far_idx.size else None,
            "overall_median": float(np.nanmedian(lead_re)),
            "overall_frac_positive": float(np.nanmean(lead_re > 0)),
        },
        "permutation_null": {
            "n_perm": int(n_perm),
            "n_sample_near": int(n_near),
            "n_sample_far": int(n_far),
            "null_median_near": float(np.nanmedian(null[:n_near])) if n_near else None,
            "null_median_far": float(np.nanmedian(null[n_near:])) if n_far else None,
            "overlap_mean_near": float(np.nanmean(overlap_near)) if n_near else None,
            "overlap_mean_far": float(np.nanmean(overlap_far)) if n_far else None,
            "real_median_near": float(np.nanmedian(lead_re[sample[:n_near]])) if n_near else None,
            "real_median_far": float(np.nanmedian(lead_re[sample[n_near:]])) if n_far else None,
        },
    }


# ------------------------------------------------------------------- paul15

def paul15_report(inject_velocity_prior: bool, n_epochs: int, k: int,
                  seed: int, n_perm: int, verbose: bool,
                  sample_size: int = 60) -> dict[str, Any]:
    print(f"\n[paul15 prior={inject_velocity_prior}] load")
    adata = ad.read_h5ad(V1_PAUL15)
    if "scjdo" in adata.uns:
        adata.uns.pop("scjdo")

    t0 = time.monotonic()
    fit_latent_field(
        adata, use_rep="X_pca", time_key="pseudotime",
        drift_config=DriftConfig(
            hidden=128, depth=4, n_epochs=n_epochs, seed=seed,
            use_spectral_norm=False, batch_size=256,
            inject_velocity_prior=inject_velocity_prior,
        ),
        field_config=FieldConfig(k=k, ridge=1e-3, seed=seed),
        verbose=verbose,
    )
    dt_fit = time.monotonic() - t0
    print(f"  fit {dt_fit:.1f}s")

    opspace.spectrum(adata, n_eig=5, fisher_mask=False)
    opspace.invariants(adata)
    opspace.behaviour(adata)

    beh = adata.obs["op_behaviour"].astype(str).to_numpy()
    lead_re = adata.obs["lead_re"].to_numpy()
    beh_counts = {b: int(np.sum(beh == b)) for b in np.unique(beh) if b}

    rng = np.random.default_rng(seed)
    sample = rng.choice(adata.n_obs, size=min(sample_size, adata.n_obs), replace=False)
    jf = JacobianField(FieldConfig(k=k, ridge=1e-3, seed=seed))
    z = adata.obsm["X_pca"].astype(np.float32)
    v = adata.obsm["X_drift_v"].astype(np.float32)
    jf.fit(z, v)
    null = jf.permutation_null(z, v, n_perm=n_perm, subset_idx=sample, seed=seed)
    overlap = _overlap(lead_re[sample], null)

    return {
        "inject_velocity_prior": bool(inject_velocity_prior),
        "fit_seconds": dt_fit,
        "n_cells": int(adata.n_obs),
        "behaviour_counts": beh_counts,
        "lead_re_summary": {
            "median": float(np.nanmedian(lead_re)),
            "frac_positive": float(np.nanmean(lead_re > 0)),
            "frac_negative": float(np.nanmean(lead_re < 0)),
            "q05": float(np.nanquantile(lead_re, 0.05)),
            "q95": float(np.nanquantile(lead_re, 0.95)),
        },
        "permutation_null": {
            "n_perm": int(n_perm),
            "n_sample": int(sample.size),
            "null_median": float(np.nanmedian(null)),
            "null_q95": float(np.nanquantile(null, 0.95)),
            "real_median": float(np.nanmedian(lead_re[sample])),
            "overlap_mean": float(np.nanmean(overlap)),
            "overlap_median": float(np.nanmedian(overlap)),
        },
    }


# -------------------------------------------------------------------- entry

def _passing(d: dict, threshold: float = 0.8) -> str:
    f = d["fork_precision_recall"]
    h = d["holder_precision_recall"]
    ok_f = (f["precision"] >= threshold) and (f["recall"] >= threshold)
    ok_h = (h["precision"] >= threshold) and (h["recall"] >= threshold)
    return "PASS" if (ok_f and ok_h) else "FAIL"


def write_markdown(report: dict, out: Path) -> None:
    lines = [
        "# Phase B gate report\n",
        f"Generated: {report['timestamp']}\n",
        "## Pitchfork — fork and holder precision/recall at the ≥ 0.8 target\n",
        "| τ_crit | fork P | fork R | holder P | holder R | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for row in report["pitchfork"]:
        f = row["fork_precision_recall"]
        h = row["holder_precision_recall"]
        verdict = _passing(row)
        lines.append(
            f"| {row['tau_crit']} | {f['precision']:.2f} | {f['recall']:.2f} | "
            f"{h['precision']:.2f} | {h['recall']:.2f} | **{verdict}** |"
        )
    lines.append("\n### Pitchfork — permutation null (sample 15+15)\n")
    lines.append("| τ_crit | real med (near) | null med (near) | overlap (near) | "
                 "real med (far) | null med (far) | overlap (far) |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in report["pitchfork"]:
        p = row["permutation_null"]
        lines.append(
            f"| {row['tau_crit']} | {p['real_median_near']:.3f} | {p['null_median_near']:.3f} | "
            f"{p['overlap_mean_near']:.3f} | {p['real_median_far']:.3f} | "
            f"{p['null_median_far']:.3f} | {p['overlap_mean_far']:.3f} |"
        )

    lines.append("\n## Paul15 — behaviour counts and null overlap\n")
    lines.append("| velocity prior | holder | drift | fork | damped_osc | growing_osc | "
                 "frac(Re>0) | null med | real med | overlap mean |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for row in report["paul15"]:
        c = row["behaviour_counts"]
        s = row["lead_re_summary"]
        p = row["permutation_null"]
        lines.append(
            f"| {row['inject_velocity_prior']} | {c.get('holder', 0)} | {c.get('drift', 0)} | "
            f"{c.get('fork', 0)} | {c.get('damped_oscillator', 0)} | "
            f"{c.get('growing_oscillator', 0)} | {s['frac_positive']:.3f} | "
            f"{p['null_median']:.3f} | {p['real_median']:.3f} | {p['overlap_mean']:.3f} |"
        )
    lines.append("")
    out.write_text("\n".join(lines))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-cells-pitchfork", type=int, default=1200)
    p.add_argument("--n-epochs-pitchfork", type=int, default=1000)
    p.add_argument("--k-pitchfork", type=int, default=40)
    p.add_argument("--n-epochs-paul15", type=int, default=1500)
    p.add_argument("--k-paul15", type=int, default=60)
    p.add_argument("--n-perm", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "pitchfork": [],
        "paul15": [],
    }

    for tau in (0.1, 0.3, 0.5):
        row = pitchfork_report(
            tau_crit=tau, n_cells=args.n_cells_pitchfork,
            n_epochs=args.n_epochs_pitchfork, k=args.k_pitchfork,
            seed=args.seed, n_perm=args.n_perm, verbose=args.verbose,
        )
        report["pitchfork"].append(row)
        (OUT / "report.partial.json").write_text(json.dumps(report, indent=2))

    for inject in (True, False):
        row = paul15_report(
            inject_velocity_prior=inject, n_epochs=args.n_epochs_paul15,
            k=args.k_paul15, seed=args.seed, n_perm=args.n_perm,
            verbose=args.verbose,
        )
        report["paul15"].append(row)
        (OUT / "report.partial.json").write_text(json.dumps(report, indent=2))

    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    write_markdown(report, OUT / "report.md")
    (OUT / "report.partial.json").unlink(missing_ok=True)
    print(f"\n[phase_b_gate] wrote {OUT / 'report.json'}")
    print(f"[phase_b_gate] wrote {OUT / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
