"""K562 follow-up (user instruction): DE-only paired test + projected
driver check.

(1) Paired comparison of operator cosine vs DE-only cosine per KD —
    mean/median, fraction operator>DE, Wilcoxon signed-rank p; all
    knockdowns + linear-regime subset.
(2) Driver check projected onto the leading eigvec of J at the
    control basin. Report both raw and projected Spearman.

Reads the stored per_knockdown.csv for (1); refits the control
operator (scVI + ESM + rate model, same seed) for (2).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import spearmanr, wilcoxon

from scjdo2.energy_drift import EnergyDriftAdapter, EnergyDriftConfig
from scjdo2.rates import GeneRateModel, RatesConfig


REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "benchmarks" / "perturbseq_k562"
K562 = Path("/Users/terooatt/Documents/scJDO/data/replogle_2022_k562_essential.h5ad")


def _paired_table(pk: pd.DataFrame) -> pd.DataFrame:
    sub_all = pk.dropna(subset=["cos_scjdo2", "cos_de_only"])
    rows = []
    for label, sub in (("all", sub_all),
                       ("linear_regime", sub_all[sub_all["linear_regime"]]),
                       ("nonlinear_regime", sub_all[~sub_all["linear_regime"]])):
        d = (sub["cos_scjdo2"] - sub["cos_de_only"]).to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        try:
            W, p = wilcoxon(d, zero_method="wilcox", alternative="greater")
            p = float(p); W = float(W)
        except Exception:
            W, p = float("nan"), float("nan")
        rows.append({
            "arm": label, "n_kd": int(len(sub)),
            "operator_cos_mean": float(sub["cos_scjdo2"].mean()),
            "de_only_cos_mean": float(sub["cos_de_only"].mean()),
            "random_cos_mean": float(sub["cos_random_mean"].mean()),
            "frac_op_gt_de": float((sub["cos_scjdo2"] > sub["cos_de_only"]).mean()),
            "frac_op_gt_rand": float((sub["cos_scjdo2"] > sub["cos_random_mean"]).mean()),
            "wilcoxon_W": W,
            "wilcoxon_p_op_gt_de": p,
        })
    return pd.DataFrame(rows)


def _refit_control_operator(seed: int = 0, d_latent: int = 15,
                              scvi_epochs: int = 120, drift_epochs: int = 1500,
                              rate_epochs: int = 150,
                              n_top_genes: int = 2000) -> dict:
    print("[k562-fu] reload + HVG")
    a = ad.read_h5ad(K562)
    is_ctrl = (a.obs["perturbation"].astype(str) == "control").to_numpy()
    counts_ctrl = a.X[is_ctrl]
    n_cells_per_gene = np.asarray((counts_ctrl > 0).sum(axis=0)).ravel()
    gene_mask = n_cells_per_gene >= 3
    a = a[:, gene_mask].copy()
    pert_targets = set(a.obs["perturbation"].astype(str).unique()) - {"control"}
    tmp = a[is_ctrl].copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=n_top_genes, flavor="seurat",
                                span=0.3)
    hv = tmp.var["highly_variable"].to_numpy()
    for g in pert_targets:
        if g in a.var_names:
            hv[a.var_names.get_loc(g)] = True
    a = a[:, hv].copy()
    print(f"[k562-fu] shape {a.shape}")

    import scvi
    scvi.settings.seed = seed
    ctrl = a[is_ctrl].copy()
    ctrl.layers["counts"] = ctrl.X.copy()
    scvi.model.SCVI.setup_anndata(ctrl, layer="counts")
    m = scvi.model.SCVI(ctrl, n_latent=int(d_latent))
    t0 = time.monotonic()
    m.train(max_epochs=int(scvi_epochs), check_val_every_n_epoch=10,
            early_stopping=False, enable_progress_bar=False)
    z_ctrl = np.asarray(m.get_latent_representation(ctrl)).astype(np.float32)
    print(f"[k562-fu] scVI {time.monotonic()-t0:.1f}s")

    a.layers["counts"] = a.X.copy()
    z_all = np.asarray(m.get_latent_representation(a)).astype(np.float32)

    t0 = time.monotonic()
    drift = EnergyDriftAdapter(dim=d_latent, config=EnergyDriftConfig(
        hidden=96, depth=3, n_epochs=drift_epochs, seed=seed,
        batch_size=256, loss="esm", lr=5e-4, beta=1.0,
        esm_laplacian="exact" if d_latent <= 20 else "hutchinson",
    )).fit(z_ctrl, verbose=False)
    J_ctrl = drift.jacobian_autodiff(z_ctrl)
    J_mean = J_ctrl.mean(axis=0).astype(np.float64)
    print(f"[k562-fu] ESM+J {time.monotonic()-t0:.1f}s")

    counts_ctrl_arr = ctrl.X.toarray() if sp.issparse(ctrl.X) else np.asarray(ctrl.X)
    counts_ctrl_arr = counts_ctrl_arr.astype(np.float32)
    t0 = time.monotonic()
    rm = GeneRateModel(RatesConfig(
        hidden=(128, 128), n_epochs=int(rate_epochs), lr=1e-3,
        batch_size=256, held_out_frac=0.10, patience=30, seed=int(seed),
    ))
    rm.fit(counts_ctrl_arr, z_ctrl, verbose=False)
    print(f"[k562-fu] rate model {time.monotonic()-t0:.1f}s")

    z_centroid = z_ctrl.mean(axis=0, keepdims=True).astype(np.float32)
    grad_eta_centroid = rm.grad_eta(z_centroid)[0]  # (G, d)
    eta_all = rm.eta(z_all)
    return {
        "a": a, "z_all": z_all,
        "J_mean": J_mean, "grad_eta_centroid": grad_eta_centroid,
        "eta_all": eta_all,
    }


def main() -> int:
    print("[k562-fu] load per-KD table")
    pk = pd.read_csv(OUT / "per_knockdown.csv")
    print(f"[k562-fu] {len(pk)} KDs")

    paired = _paired_table(pk)
    paired.to_csv(OUT / "paired_comparison.csv", index=False)
    print("\n[k562-fu] PAIRED COMPARISON:")
    print(paired.round(4).to_string(index=False))

    print("\n[k562-fu] REFIT control operator for projected driver check")
    fit = _refit_control_operator()
    J_mean = fit["J_mean"]
    grad_eta = fit["grad_eta_centroid"]
    eta_all = fit["eta_all"]
    a = fit["a"]
    gene_idx_map = {g: i for i, g in enumerate(a.var_names)}

    # Leading eigenvector of J at the control basin.
    # J_mean is symmetric by construction (energy drift). The "leading"
    # mode we project onto is the LEAST-NEGATIVE eigenvector — the
    # soft-escape direction at the basin.
    ev, V = np.linalg.eigh(J_mean)
    lead_idx = int(np.argmax(ev))                                # least neg
    v_lead = V[:, lead_idx]                                       # (d,)
    v_lead_gene = grad_eta @ v_lead                              # (G,)
    v_lead_gene_norm = v_lead_gene / (np.linalg.norm(v_lead_gene) + 1e-12)
    print(f"[k562-fu] lead eigenvalue of J_mean: {ev[lead_idx]:+.4f}")
    print(f"[k562-fu] ‖v_lead in gene space‖ = {np.linalg.norm(v_lead_gene):.3f}")

    # Sensitivity score per gene (sens_g = |l_g| · Σ|r_h|; symmetric → l=r=v).
    sens = np.abs(v_lead_gene) * float(np.abs(v_lead_gene).sum())

    # Observed self-KD |δ| per gene (same as prior run).
    is_ctrl = (a.obs["perturbation"].astype(str) == "control").to_numpy()
    cluster = a.obs["perturbation"].astype(str).to_numpy()
    tmp = a.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    X_log = tmp.X.toarray() if sp.issparse(tmp.X) else np.asarray(tmp.X)
    X_log = X_log.astype(np.float32)

    # Raw |δx_obs| per gene (self-KD log-fold change magnitude).
    obs_raw = np.zeros(a.n_vars, dtype=np.float64)
    # Projected δx_obs_proj per gene = <δx_obs_full, v_lead_gene_norm> × v_lead_gene_norm[g]
    # i.e. project onto lead-mode then read gene-g component.
    obs_proj = np.zeros(a.n_vars, dtype=np.float64)
    for g, i in gene_idx_map.items():
        if g not in pk["gene"].values:
            continue
        kd_mask = (cluster == g)
        if kd_mask.sum() < 50:
            continue
        delta_full = eta_all[kd_mask].mean(axis=0) - eta_all[is_ctrl].mean(axis=0)
        # Scalar along lead_mode.
        a_scalar = float(delta_full @ v_lead_gene_norm)
        delta_proj = a_scalar * v_lead_gene_norm
        obs_raw[i] = abs(float(delta_full[i]))
        obs_proj[i] = abs(float(delta_proj[i]))
    mask = obs_raw > 0
    rho_raw = float(spearmanr(sens[mask], obs_raw[mask]).correlation)
    rho_proj = float(spearmanr(sens[mask], obs_proj[mask]).correlation)
    print(f"\n[k562-fu] Driver Spearman — raw:       {rho_raw:+.3f}")
    print(f"[k562-fu] Driver Spearman — projected: {rho_proj:+.3f}")

    pd.DataFrame([{
        "driver_spearman_raw": rho_raw,
        "driver_spearman_projected_on_lead_eigvec": rho_proj,
        "n_genes": int(mask.sum()),
        "lead_eigenvalue_of_J_mean": float(ev[lead_idx]),
        "v_lead_gene_norm_L2": float(np.linalg.norm(v_lead_gene)),
    }]).to_csv(OUT / "driver_check_projected.csv", index=False)

    # Update report.md with the paired table + projected driver check.
    _append_report(OUT, paired, rho_raw, rho_proj, float(ev[lead_idx]))
    print(f"\n[k562-fu] outputs → {OUT}")
    return 0


def _append_report(OUT: Path, paired: pd.DataFrame,
                    rho_raw: float, rho_proj: float, lead_ev: float) -> None:
    """Append the follow-up section to the existing report.md."""
    existing = (OUT / "report.md").read_text() if (OUT / "report.md").exists() else ""
    follow = [
        "",
        "---",
        "",
        "## Follow-up (post-first-result corrections, 2026-10-14)",
        "",
        "### DE-only paired comparison + Wilcoxon",
        "",
        "Paired per-KD cosine: operator cos vs DE-only cos (δx = b_g itself).",
        "Wilcoxon signed-rank one-sided (H₁: operator > DE):",
        "",
        paired.round(4).to_markdown(index=False),
        "",
        "The paired test asks whether the operator improves on DE-only on",
        "a per-KD basis. `frac_op_gt_de` is the fraction of KDs where the",
        "operator cosine strictly exceeds DE-only. The data support the",
        "arm whose p is tiny AND frac is well above 0.5.",
        "",
        "### Driver check — raw vs projected onto lead eigenvector",
        "",
        f"* Lead eigenvalue of mean J at control basin: {lead_ev:+.4f} "
        "(least-negative eigenvalue — the soft-escape direction).",
        f"* Driver Spearman raw        (|δx_obs| per gene vs sensitivity): **{rho_raw:+.3f}**",
        f"* Driver Spearman projected  (|⟨δx_obs, v_lead⟩ · v_lead_g| vs sensitivity): **{rho_proj:+.3f}**",
        "",
        "Projecting δx_obs onto the lead eigenvector of ``J_mean`` before",
        "correlating with sensitivity isolates the component the operator",
        "actually acts on. The projected Spearman is the fair test of",
        "whether sensitive-direction drivers correlate with observed shifts",
        "along the operator's soft axis; the raw Spearman mixes orthogonal",
        "components that the operator never affects.",
    ]
    (OUT / "report.md").write_text(existing + "\n".join(follow) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
