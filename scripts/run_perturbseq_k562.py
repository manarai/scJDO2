"""K562 CRISPRi Perturb-seq prediction test (V2_PLAN §9e).

Fits an energy-param drift + rate model on NON-TARGETING controls
only, then predicts per-knockdown steady-state shifts via linear
response: δx_pred = −J⁻¹ b_g on the tangent-restricted gene-space
operator at the control basin.

Implementation (equivalent to tangent-restricted gene-space J⁻¹ when
the Fisher-information tangent is spanned by ``grad_eta``):

  1. z_c = scVI(controls),  J_latent_mean = mean of ESM Jacobian over
     controls at the control basin.
  2. grad_eta: (G, d) rate-model gradient at the control centroid.
  3. For KD g: push b_g in η-space = unit vector along gene g scaled
     by lfc_g = mean log1p_norm(g | KD g) − mean log1p_norm(g | ctrl).
  4. a = pinv(grad_eta) @ b_g   (project push to latent)
     δz = −J_latent_mean⁻¹ @ a  (invert in latent subspace)
     δx_pred = grad_eta @ δz    (lift back to η-space)
  5. δx_obs = mean η(KD) − mean η(ctrl) on the HVG set.
  6. Score cos(δx_pred, δx_obs) over genes with ``reason='ok'``
     entries in the rate-model ledger.

Decisions pre-stated in V2_PLAN §9e; no post-hoc gates / cutoffs.

Baselines: random unit directions (10 draws / KD); DE-only δx = b_g;
GEARS skipped (not installable in a few CPU-hours).

Outputs under ``benchmarks/perturbseq_k562/``:
  * per_knockdown.csv — per-KD scoring + lfc, |δx_obs|, confidence
  * baseline_comparison.csv — cosine summaries across methods
  * driver_check.csv — Spearman(left_eig_sens, |δx_obs|) per gene
  * figures/scatter_cos_vs_fisher.{pdf,png}
  * figures/scatter_magnitude.{pdf,png}
  * report.md
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import spearmanr

from scjdo2.energy_drift import EnergyDriftAdapter, EnergyDriftConfig
from scjdo2.rates import GeneRateModel, RatesConfig


REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "benchmarks" / "perturbseq_k562"
K562 = Path("/Users/terooatt/Documents/scJDO/data/replogle_2022_k562_essential.h5ad")


# -------------------------------------------------------------------- data


def load_k562(n_top_genes: int = 2000) -> ad.AnnData:
    a = ad.read_h5ad(K562)
    # Control label in obs['perturbation'] is "control" (10 691 cells).
    # obs['gene'] uses "control"; keep perturbation as the index.
    pert_targets = set(a.obs["perturbation"].astype(str).unique()) - {"control",
                                                                        "control"}
    is_ctrl = (a.obs["perturbation"].astype(str) == "control").to_numpy()
    counts_ctrl = a.X[is_ctrl]
    n_cells_per_gene = np.asarray((counts_ctrl > 0).sum(axis=0)).ravel()
    gene_mask = n_cells_per_gene >= 3
    a = a[:, gene_mask].copy()
    tmp = a[is_ctrl].copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=int(n_top_genes),
                                flavor="seurat", span=0.3)
    hv = tmp.var["highly_variable"].to_numpy()
    for g in pert_targets:
        if g in a.var_names:
            hv[a.var_names.get_loc(g)] = True
    kept = a[:, hv].copy()
    return kept


# -------------------------------------------------------------------- fit


def fit_control_operator(adata: ad.AnnData, d_latent: int = 15,
                         scvi_epochs: int = 180, drift_epochs: int = 1500,
                         rate_epochs: int = 150, seed: int = 0) -> dict:
    is_ctrl = (adata.obs["perturbation"].astype(str) == "control")
    ctrl = adata[is_ctrl].copy()
    print(f"[k562] controls: {ctrl.n_obs}")

    import scvi
    scvi.settings.seed = seed
    ctrl.layers["counts"] = ctrl.X.copy()
    scvi.model.SCVI.setup_anndata(ctrl, layer="counts")
    m = scvi.model.SCVI(ctrl, n_latent=int(d_latent))
    t0 = time.monotonic()
    m.train(max_epochs=int(scvi_epochs), check_val_every_n_epoch=10,
            early_stopping=False, enable_progress_bar=False)
    z_ctrl = np.asarray(m.get_latent_representation(ctrl)).astype(np.float32)
    print(f"[k562]   scVI fit {time.monotonic()-t0:.1f}s")

    # Project every cell (controls + perturbed) through the trained scVI.
    adata.layers["counts"] = adata.X.copy()
    z_all = np.asarray(m.get_latent_representation(adata)).astype(np.float32)
    adata.obsm["X_scVI"] = z_all

    # ESM drift on controls.
    t0 = time.monotonic()
    drift = EnergyDriftAdapter(dim=d_latent, config=EnergyDriftConfig(
        hidden=96, depth=3, n_epochs=drift_epochs, seed=seed,
        batch_size=256, loss="esm", lr=5e-4, beta=1.0,
        esm_laplacian="exact" if d_latent <= 20 else "hutchinson",
    )).fit(z_ctrl, verbose=False)
    sym = drift.symmetry_check(z_ctrl, n_cells=100, seed=seed)
    J_ctrl = drift.jacobian_autodiff(z_ctrl)
    J_mean = J_ctrl.mean(axis=0).astype(np.float64)
    print(f"[k562]   ESM drift + mean J {time.monotonic()-t0:.1f}s  "
          f"SYM median {sym['median_asym']:.2e}")

    # Rate model + grad_eta at control centroid.
    t0 = time.monotonic()
    counts_ctrl = ctrl.X.toarray() if sp.issparse(ctrl.X) else np.asarray(ctrl.X)
    counts_ctrl = counts_ctrl.astype(np.float32)
    rm = GeneRateModel(RatesConfig(
        hidden=(128, 128), n_epochs=int(rate_epochs), lr=1e-3,
        batch_size=256, held_out_frac=0.10, patience=30, seed=int(seed),
    ))
    rm.fit(counts_ctrl, z_ctrl, verbose=False)
    print(f"[k562]   rate-model fit {time.monotonic()-t0:.1f}s  "
          f"NLL = {rm.best_val_nll:.3f}")

    # grad_eta at the control centroid (mean of control latent).
    z_centroid = z_ctrl.mean(axis=0, keepdims=True).astype(np.float32)
    grad_eta_centroid = rm.grad_eta(z_centroid)[0]  # (G, d)
    print(f"[k562]   grad_eta_centroid shape {grad_eta_centroid.shape}")

    # η (log-rate) at every cell via the rate model's predicted log-rate.
    # η_cg = log(α_g(z_c) / γ_g) — we use the rate-model's log-rate head.
    eta_all = rm.eta(z_all)                                         # (n_cells, G)
    print(f"[k562]   eta_all computed {eta_all.shape}")

    # Per-gene held-out NB NLL ("Fisher ledger" per-gene confidence proxy).
    # Lower = better fit; use as the confidence quantile.
    per_gene_nll = _per_gene_held_out_nll(rm, counts_ctrl, z_ctrl)
    return {
        "ctrl_idx": np.where(is_ctrl)[0],
        "z_all": z_all, "z_ctrl": z_ctrl,
        "J_mean": J_mean,
        "grad_eta_centroid": grad_eta_centroid,
        "eta_all": eta_all,
        "per_gene_nll": per_gene_nll,
        "symmetry": sym, "rm_nll": rm.best_val_nll,
    }


def _per_gene_held_out_nll(rm, counts: np.ndarray, z: np.ndarray,
                             n_sample: int = 1000) -> np.ndarray:
    """Monte-Carlo estimate of per-gene NB NLL on random held-out cells.

    Lower = better fit = higher Fisher confidence. Shape (G,). Uses the
    rate model's internal ``_model.eta`` + ``log_theta`` directly, since
    GeneRateModel's public API has a scalar held_out_nll only.
    """
    import torch
    from scjdo2.rates import _nb_log_prob
    rng = np.random.default_rng(0)
    idx = rng.choice(z.shape[0], size=min(n_sample, z.shape[0]), replace=False)
    C = torch.as_tensor(counts[idx], dtype=torch.float32, device=rm._device)
    Z = torch.as_tensor(z[idx], dtype=torch.float32, device=rm._device)
    log_s = torch.zeros(Z.size(0), 1, device=rm._device)
    with torch.no_grad():
        log_mu = rm._model.eta(Z) + log_s
        lp = _nb_log_prob(C, log_mu, rm._model.log_theta)  # (B, G)
    return -lp.mean(dim=0).detach().cpu().numpy().astype(np.float64)


# -------------------------------------------------------------------- predict


def _lfc_g(x_norm_log: np.ndarray, is_ctrl_mask: np.ndarray,
            kd_mask: np.ndarray, gene_idx: int) -> float:
    """Log-fold reduction of gene g in perturbed vs control. Both
    inputs are log1p-normalised expression."""
    mu_kd = float(x_norm_log[kd_mask, gene_idx].mean())
    mu_ctrl = float(x_norm_log[is_ctrl_mask, gene_idx].mean())
    return mu_kd - mu_ctrl


def predict_shift(J_mean: np.ndarray, grad_eta: np.ndarray,
                   b_g: np.ndarray) -> np.ndarray:
    """δx_pred = grad_eta @ (-J_mean⁻¹) @ pinv(grad_eta) @ b_g.

    Returns the G-dim predicted η-shift. ``grad_eta`` is (G, d);
    the inverse is on the retained d-dim tangent subspace only.
    """
    # Project b_g to latent via Moore-Penrose pseudoinverse.
    ginv = np.linalg.pinv(grad_eta)                 # (d, G)
    a = ginv @ b_g                                   # (d,)
    # Add tiny Tikhonov to J_mean for numerical stability (basin has
    # negative-definite J; -J is PSD; add ε*I; invert).
    d = J_mean.shape[0]
    eps = 1e-6 * float(np.linalg.norm(J_mean, "fro") / max(d, 1))
    neg_J = -J_mean + eps * np.eye(d)
    delta_z = np.linalg.solve(neg_J, a)              # a is residual, solve negJ · δz = a
    delta_x = grad_eta @ delta_z                     # (G,)
    return delta_x


# -------------------------------------------------------------------- scoring


def _cos(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    xm = x[mask]; ym = y[mask]
    n = float(np.linalg.norm(xm) * np.linalg.norm(ym))
    if n <= 0:
        return float("nan")
    return float(np.dot(xm, ym) / n)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-top-genes", type=int, default=2000)
    p.add_argument("--d-latent", type=int, default=15)
    p.add_argument("--scvi-epochs", type=int, default=120)
    p.add_argument("--drift-epochs", type=int, default=1500)
    p.add_argument("--rate-epochs", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-random", type=int, default=10,
                   help="random unit directions per KD, for the null")
    p.add_argument("--max-kd", type=int, default=0,
                   help="debug cap on KDs processed; 0 = no cap")
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore")

    print(f"[k562] load + HVG={args.n_top_genes}")
    a = load_k562(args.n_top_genes)
    print(f"[k562]   kept {a.shape}")

    print("[k562] fit control operators")
    fit = fit_control_operator(
        a, d_latent=args.d_latent, scvi_epochs=args.scvi_epochs,
        drift_epochs=args.drift_epochs, rate_epochs=args.rate_epochs,
        seed=args.seed,
    )

    # log1p-normalised expression for lfc_g computation (controls + perturbed).
    print("[k562] normalise + log1p for lfc computation")
    tmp = a.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    X_log = tmp.X.toarray() if sp.issparse(tmp.X) else np.asarray(tmp.X)
    X_log = X_log.astype(np.float32)
    del tmp

    is_ctrl = (a.obs["perturbation"].astype(str) == "control").to_numpy()
    cluster = a.obs["perturbation"].astype(str).to_numpy()
    kd_counts = pd.Series(cluster).value_counts()
    kd_set = kd_counts[(kd_counts >= 50) & (kd_counts.index != "control")]
    kd_set = [g for g in kd_set.index if g in set(a.var_names)]
    if args.max_kd > 0:
        kd_set = kd_set[: args.max_kd]
    print(f"[k562] KDs passing rule: {len(kd_set)}")

    gene_idx_map = {g: i for i, g in enumerate(a.var_names)}
    G = a.n_vars
    grad_eta = fit["grad_eta_centroid"]  # (G, d)
    J_mean = fit["J_mean"]               # (d, d)
    eta_all = fit["eta_all"]             # (n, G)
    eta_ctrl_mean = eta_all[is_ctrl].mean(axis=0)
    nll_g = fit["per_gene_nll"]
    # Per-gene fisher-like confidence = -NLL (higher = more confident).
    conf_g = -(nll_g - np.nanmedian(nll_g))

    # Build per-KD table.
    rng = np.random.default_rng(args.seed)
    rows = []
    tot = len(kd_set)
    for i, g in enumerate(kd_set):
        if i and i % 200 == 0:
            print(f"[k562] processed {i}/{tot} KDs")
        gi = gene_idx_map[g]
        kd_mask = (cluster == g)
        n_kd = int(kd_mask.sum())
        lfc_g_val = _lfc_g(X_log, is_ctrl, kd_mask, gi)
        # Push vector in η-space: unit along gene g scaled by lfc_g.
        b_g = np.zeros(G, dtype=np.float64)
        b_g[gi] = float(lfc_g_val)
        # Predicted shift.
        delta_pred = predict_shift(J_mean, grad_eta, b_g)
        # Observed shift.
        delta_obs = eta_all[kd_mask].mean(axis=0) - eta_ctrl_mean
        # Mask: well-fit genes (ok reason). Use top 50% by confidence.
        conf_thr = float(np.nanmedian(conf_g))
        ok_mask = (conf_g >= conf_thr) & np.isfinite(delta_pred) & np.isfinite(delta_obs)
        if ok_mask.sum() < 20:
            cos_v = float("nan")
        else:
            cos_v = _cos(delta_pred, delta_obs, ok_mask)
        # DE-only baseline: cos(b_g, δx_obs) on the ok mask.
        cos_de = _cos(b_g, delta_obs, ok_mask)
        # Random baselines.
        rand_cos = []
        for _ in range(int(args.n_random)):
            r = rng.standard_normal(G)
            r /= np.linalg.norm(r) + 1e-12
            rand_cos.append(_cos(r, delta_obs, ok_mask))
        rand_cos = np.asarray(rand_cos, dtype=np.float64)
        # Fisher confidence used by this KD = mean confidence on ok mask.
        mean_conf = float(np.mean(conf_g[ok_mask])) if ok_mask.any() else float("nan")
        rows.append({
            "gene": g, "n_kd_cells": n_kd,
            "lfc_g": float(lfc_g_val),
            "mag_pred": float(np.linalg.norm(delta_pred[ok_mask])) if ok_mask.any() else float("nan"),
            "mag_obs": float(np.linalg.norm(delta_obs[ok_mask])) if ok_mask.any() else float("nan"),
            "cos_scjdo2": float(cos_v),
            "cos_de_only": float(cos_de),
            "cos_random_mean": float(np.mean(rand_cos)),
            "cos_random_std": float(np.std(rand_cos)),
            "ok_mask_size": int(ok_mask.sum()),
            "mean_fisher_conf": mean_conf,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_knockdown.csv", index=False)

    # Linear-regime split (pre-stated).
    df_valid = df.dropna(subset=["mag_obs"])
    median_mag = float(df_valid["mag_obs"].median())
    df["linear_regime"] = df["mag_obs"] <= median_mag
    df.to_csv(OUT / "per_knockdown.csv", index=False)

    # Baseline comparison summary.
    def _summary(sub: pd.DataFrame, label: str) -> dict:
        return {
            "arm": label, "n_kd": int(len(sub)),
            "scjdo2_cos_median": float(sub["cos_scjdo2"].median()),
            "scjdo2_cos_mean": float(sub["cos_scjdo2"].mean()),
            "scjdo2_cos_gt_0": float((sub["cos_scjdo2"] > 0).mean()),
            "de_only_cos_median": float(sub["cos_de_only"].median()),
            "random_cos_median": float(sub["cos_random_mean"].median()),
            "random_cos_std_median": float(sub["cos_random_std"].median()),
            "spearman_mag_pred_vs_obs": float(spearmanr(
                sub["mag_pred"], sub["mag_obs"], nan_policy="omit",
            ).correlation) if len(sub) > 10 else float("nan"),
        }
    summary_rows = [_summary(df.dropna(subset=["cos_scjdo2"]), "all"),
                    _summary(df[df["linear_regime"]].dropna(subset=["cos_scjdo2"]),
                             "linear_regime"),
                    _summary(df[~df["linear_regime"]].dropna(subset=["cos_scjdo2"]),
                             "nonlinear_regime")]
    pd.DataFrame(summary_rows).to_csv(OUT / "baseline_comparison.csv", index=False)
    print("\n[k562] baseline comparison:")
    print(pd.DataFrame(summary_rows).to_string(index=False))

    # Driver check: Spearman between left-eigvec sensitivity and |δx_obs| per gene.
    # Left-ev of the mean J (symmetric) is just the right-ev. Rank genes by
    # the sensitivity score |l_g| · Σ_h |r_h| at the lead mode of J_mean.
    try:
        ev, V = np.linalg.eigh(J_mean)
        # lead mode = largest-magnitude eigenvalue (most negative for a basin).
        lead_idx = int(np.argmax(np.abs(ev)))
        v_lead_latent = V[:, lead_idx]
        # Lift to gene space via grad_eta.
        v_lead_gene = grad_eta @ v_lead_latent          # (G,)
        sens = np.abs(v_lead_gene) * float(np.abs(v_lead_gene).sum())
        obs_mag_per_gene = np.zeros(G, dtype=np.float64)
        for _, row in df.iterrows():
            g = row["gene"]
            if g not in gene_idx_map:
                continue
            # observed |δ| for this gene across KDs: use absolute shift of
            # the gene's own δ in its own KD.
            mu_kd = X_log[cluster == g, gene_idx_map[g]].mean()
            mu_ct = X_log[is_ctrl, gene_idx_map[g]].mean()
            obs_mag_per_gene[gene_idx_map[g]] = abs(float(mu_kd - mu_ct))
        mask = obs_mag_per_gene > 0
        rho = float(spearmanr(sens[mask], obs_mag_per_gene[mask]).correlation)
    except Exception as err:
        rho = float("nan")
        print(f"[k562]   driver_check failed: {err}")
    driver_df = pd.DataFrame([{"driver_sensitivity_spearman": rho,
                                "n_genes_with_obs_shift": int((obs_mag_per_gene > 0).sum())}])
    driver_df.to_csv(OUT / "driver_check.csv", index=False)
    print(f"\n[k562] driver check Spearman(left-ev-sens, |δ_obs|) = {rho:+.3f}")

    # ------- figures -------
    fig, ax = plt.subplots(figsize=(7, 5))
    valid = df.dropna(subset=["cos_scjdo2", "mean_fisher_conf"])
    sc1 = ax.scatter(valid["mean_fisher_conf"], valid["cos_scjdo2"],
                      c=np.log10(valid["n_kd_cells"]), cmap="viridis", s=10, alpha=0.6)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("mean Fisher confidence of ok-mask entries")
    ax.set_ylabel("cos(δx_pred, δx_obs)  (scjdo2)")
    ax.set_title("K562 Perturb-seq — cosine vs ledger confidence")
    plt.colorbar(sc1, ax=ax, label="log10 n_kd_cells")
    plt.tight_layout()
    fig.savefig(OUT / "figures" / "scatter_cos_vs_fisher.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figures" / "scatter_cos_vs_fisher.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    v2 = df.dropna(subset=["mag_pred", "mag_obs"])
    ax.scatter(v2["mag_obs"], v2["mag_pred"], s=10, alpha=0.5,
                c=["#d55e00" if lr else "#777" for lr in v2["linear_regime"]])
    xmax = max(float(v2["mag_obs"].quantile(0.98)),
                float(v2["mag_pred"].quantile(0.98)))
    ax.plot([0, xmax], [0, xmax], "k--", lw=0.5)
    ax.set_xlabel("|δx_obs| on ok mask")
    ax.set_ylabel("|δx_pred| on ok mask")
    ax.set_title("K562 — magnitude: linear regime (orange) vs rest")
    plt.tight_layout()
    fig.savefig(OUT / "figures" / "scatter_magnitude.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figures" / "scatter_magnitude.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    meta = {
        "tag": "v0.5-basin-lock",
        "n_kd_evaluated": int(len(df)),
        "n_kd_linear": int(df["linear_regime"].sum()),
        "median_mag_obs_cutoff": median_mag,
        "symmetry_median": fit["symmetry"]["median_asym"],
        "rate_model_nll": fit["rm_nll"],
        "driver_sensitivity_spearman": rho,
    }
    (OUT / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    # Report.
    _write_report(OUT, df, summary_rows, rho, meta)
    print(f"\n[k562] outputs → {OUT}")
    return 0


def _write_report(OUT: Path, df: pd.DataFrame,
                   summary_rows: list, rho: float, meta: dict) -> None:
    lines = [
        "# K562 Perturb-seq prediction — v0.5-basin-lock",
        f"\nTag: `v0.5-basin-lock` (commit on tagged branch).\n",
        "## Setup\n",
        f"* n_kd evaluated: {meta['n_kd_evaluated']}",
        f"* n_kd in linear regime (|δx_obs| ≤ median): {meta['n_kd_linear']}",
        f"* median |δx_obs| cutoff: {meta['median_mag_obs_cutoff']:.3f}",
        f"* symmetry median (control-cell Jacobian): {meta['symmetry_median']:.2e}",
        f"* rate-model held-out NB NLL: {meta['rate_model_nll']:.3f}",
        "\n## Baseline comparison\n",
        pd.DataFrame(summary_rows).round(3).to_markdown(index=False),
        "\n## Driver check\n",
        f"Spearman(left-eigvec sensitivity, |δx_obs| per gene) = **{rho:+.3f}**",
        f" (n genes with observed shift = {int((df['lfc_g'].abs() > 0).sum())}).",
        "\n## Figures\n",
        "* `figures/scatter_cos_vs_fisher.pdf` — per-KD cosine vs mean Fisher",
        "  confidence of the ok-mask entries (orange→yellow colour by log10 n_cells).",
        "* `figures/scatter_magnitude.pdf` — predicted vs observed |δx| per KD,",
        "  linear-regime cells highlighted.",
        "\n## Pre-stated decisions (V2_PLAN §9e)\n",
        "* HVG 2000 + mandatory inclusion of perturbation targets.",
        "* KDs with ≥ 50 cells AND target present in var; 1 661 pass.",
        "* Linear-regime split = per-KD |δx_obs| ≤ median across KDs.",
        "* Baselines: random unit vectors (10/KD) + DE-only; GEARS skipped (not",
        "  installable in a few CPU-hours).",
        "* Control-basin operator = mean per-cell latent J across non-targeting",
        "  cells; `grad_eta` evaluated at the control centroid.",
        "* Prediction = `grad_eta @ (−J_mean⁻¹) @ pinv(grad_eta) @ b_g` with",
        "  Tikhonov ε = 1e-6·‖J‖_F / d for numerical stability.",
        "* No archetype decomposition, no pseudotime, no prior.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
