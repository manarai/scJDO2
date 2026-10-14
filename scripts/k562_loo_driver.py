"""K562 — leave-one-out driver check + random-eigvec null.

For each perturbed gene g with observed self-KD shift δx_obs:
  1. Zero out gene g's own coordinate in δx_obs BEFORE projecting onto
     the lead eigvec of J_mean at the control basin.
  2. Compute |δ_proj_LOO[g]| as the g-th coordinate of the projection.
  3. Partial Spearman(sens[g], |δ_proj_LOO[g]| | log_mean_ctrl_expr[g])
     — the LOO driver claim.
  4. As a null, project onto a random unit vector in gene space (10
     draws); report distribution of partial Spearmans.

Decision rule (pre-stated, applied below):
  LOO driver check HOLDS iff
      partial_spearman_LOO ≥ 0.30
    AND
      partial_spearman_LOO is above the 95th percentile of the random-
      eigvec null distribution.

Tag v0.6-perturb iff LOO driver check holds (and the generic-response
control, already decided in V2_PLAN §9f, remains failing — so the tag
will NOT be issued regardless, but we execute the check and record).

Reads/refits what k562_controls.py did; writes a new CSV and extends
RESULT.md.
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
from scipy.stats import spearmanr, rankdata

from scjdo2.energy_drift import EnergyDriftAdapter, EnergyDriftConfig
from scjdo2.rates import GeneRateModel, RatesConfig


REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "benchmarks" / "perturbseq_k562"
K562 = Path("/Users/terooatt/Documents/scJDO/data/replogle_2022_k562_essential.h5ad")


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    xr, yr, zr = rankdata(x), rankdata(y), rankdata(z)
    def _resid(v, u):
        v = v - v.mean(); u = u - u.mean()
        beta = float(np.dot(v, u) / (np.dot(u, u) + 1e-12))
        return v - beta * u
    xr_res = _resid(xr, zr)
    yr_res = _resid(yr, zr)
    den = float(np.sqrt((xr_res ** 2).sum() * (yr_res ** 2).sum()) + 1e-12)
    return float(np.dot(xr_res, yr_res) / den)


def _refit() -> dict:
    print("[k562-loo] load + HVG")
    a = ad.read_h5ad(K562)
    is_ctrl = (a.obs["perturbation"].astype(str) == "control").to_numpy()
    counts_ctrl = a.X[is_ctrl]
    n_cells_per_gene = np.asarray((counts_ctrl > 0).sum(axis=0)).ravel()
    gene_mask = n_cells_per_gene >= 3
    a = a[:, gene_mask].copy()
    pert_targets = set(a.obs["perturbation"].astype(str).unique()) - {"control"}
    tmp = a[is_ctrl].copy()
    sc.pp.normalize_total(tmp, target_sum=1e4); sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=2000, flavor="seurat", span=0.3)
    hv = tmp.var["highly_variable"].to_numpy()
    for g in pert_targets:
        if g in a.var_names:
            hv[a.var_names.get_loc(g)] = True
    a = a[:, hv].copy()
    print(f"[k562-loo] shape {a.shape}")

    import scvi
    scvi.settings.seed = 0
    ctrl = a[is_ctrl].copy()
    ctrl.layers["counts"] = ctrl.X.copy()
    scvi.model.SCVI.setup_anndata(ctrl, layer="counts")
    m = scvi.model.SCVI(ctrl, n_latent=15)
    m.train(max_epochs=120, enable_progress_bar=False, early_stopping=False)
    z_ctrl = np.asarray(m.get_latent_representation(ctrl)).astype(np.float32)
    a.layers["counts"] = a.X.copy()
    z_all = np.asarray(m.get_latent_representation(a)).astype(np.float32)

    drift = EnergyDriftAdapter(dim=15, config=EnergyDriftConfig(
        hidden=96, depth=3, n_epochs=1500, seed=0, batch_size=256,
        loss="esm", lr=5e-4, beta=1.0, esm_laplacian="exact",
    )).fit(z_ctrl, verbose=False)
    J_mean = drift.jacobian_autodiff(z_ctrl).mean(axis=0).astype(np.float64)

    counts_ctrl_arr = ctrl.X.toarray() if sp.issparse(ctrl.X) else np.asarray(ctrl.X)
    counts_ctrl_arr = counts_ctrl_arr.astype(np.float32)
    rm = GeneRateModel(RatesConfig(
        hidden=(128, 128), n_epochs=150, lr=1e-3, batch_size=256,
        held_out_frac=0.10, patience=30, seed=0,
    ))
    rm.fit(counts_ctrl_arr, z_ctrl, verbose=False)
    z_centroid = z_ctrl.mean(axis=0, keepdims=True).astype(np.float32)
    grad_eta = rm.grad_eta(z_centroid)[0]
    eta_all = rm.eta(z_all)

    tmp = a[is_ctrl].copy()
    sc.pp.normalize_total(tmp, target_sum=1e4); sc.pp.log1p(tmp)
    X = tmp.X.toarray() if sp.issparse(tmp.X) else np.asarray(tmp.X)
    mean_log_ctrl = np.asarray(X.mean(axis=0)).ravel()

    return {"a": a, "is_ctrl": is_ctrl, "z_all": z_all,
            "J_mean": J_mean, "grad_eta": grad_eta,
            "eta_all": eta_all, "mean_log_ctrl": mean_log_ctrl}


def main() -> int:
    print("[k562-loo] refit control operators")
    fit = _refit()
    a = fit["a"]
    J_mean = fit["J_mean"]; grad_eta = fit["grad_eta"]
    eta_all = fit["eta_all"]; is_ctrl = fit["is_ctrl"]
    mean_log_ctrl = fit["mean_log_ctrl"]
    G = a.n_vars
    gene_idx = {g: i for i, g in enumerate(a.var_names)}
    cluster = a.obs["perturbation"].astype(str).to_numpy()

    pk = pd.read_csv(OUT / "per_knockdown.csv")
    kd_list = [g for g in pk["gene"] if g in gene_idx]
    print(f"[k562-loo] n_kd = {len(kd_list)}")

    # Lead eigvec of J_mean (symmetric, PSD-negative) — least-negative
    # = soft escape direction.
    ev, V = np.linalg.eigh(J_mean)
    lead_idx = int(np.argmax(ev))
    v_lead = V[:, lead_idx]
    v_lead_gene = grad_eta @ v_lead
    v_lead_norm = v_lead_gene / (np.linalg.norm(v_lead_gene) + 1e-12)
    sens = np.abs(v_lead_gene) * float(np.abs(v_lead_gene).sum())
    print(f"[k562-loo] lead eigenvalue of J_mean: {ev[lead_idx]:+.4f}")

    eta_ctrl_mean = eta_all[is_ctrl].mean(axis=0)

    # Observed δx_obs full and LOO-projected per gene g.
    obs_proj_full = np.zeros(G)
    obs_proj_loo = np.zeros(G)
    obs_raw = np.zeros(G)
    for g in kd_list:
        gi = gene_idx[g]
        kd_mask = (cluster == g)
        delta = eta_all[kd_mask].mean(axis=0) - eta_ctrl_mean  # (G,)
        # Full projection scalar.
        a_full = float(delta @ v_lead_norm)
        obs_proj_full[gi] = abs(a_full * v_lead_norm[gi])
        # LOO: zero gene g's own coordinate before projecting.
        delta_loo = delta.copy(); delta_loo[gi] = 0.0
        a_loo = float(delta_loo @ v_lead_norm)
        obs_proj_loo[gi] = abs(a_loo * v_lead_norm[gi])
        obs_raw[gi] = abs(float(delta[gi]))

    valid = obs_raw > 0
    sp_full = float(spearmanr(sens[valid], obs_proj_full[valid]).correlation)
    sp_loo = float(spearmanr(sens[valid], obs_proj_loo[valid]).correlation)
    expr = mean_log_ctrl[valid]
    partial_full = _partial_spearman(sens[valid], obs_proj_full[valid], expr)
    partial_loo = _partial_spearman(sens[valid], obs_proj_loo[valid], expr)

    # Random-eigvec null: project onto random unit vectors in gene space.
    print("[k562-loo] random-eigvec null (n_null=20)")
    rng = np.random.default_rng(0)
    nulls = []
    for _ in range(20):
        u = rng.standard_normal(G)
        u /= np.linalg.norm(u) + 1e-12
        obs_null = np.zeros(G)
        sens_null = np.abs(u) * float(np.abs(u).sum())  # same functional form
        for g in kd_list:
            gi = gene_idx[g]
            kd_mask = (cluster == g)
            delta = eta_all[kd_mask].mean(axis=0) - eta_ctrl_mean
            delta[gi] = 0.0                                   # LOO
            a_u = float(delta @ u)
            obs_null[gi] = abs(a_u * u[gi])
        partial_null = _partial_spearman(sens_null[valid], obs_null[valid], expr)
        nulls.append(partial_null)
    nulls = np.asarray(nulls)
    null_q95 = float(np.percentile(nulls, 95))
    null_mean = float(nulls.mean())
    null_std = float(nulls.std())
    print(f"[k562-loo] partial Spearman:")
    print(f"           full projection : {partial_full:+.3f}")
    print(f"           LOO projection  : {partial_loo:+.3f}")
    print(f"           random null     : mean {null_mean:+.3f}  "
          f"std {null_std:.3f}  95pct {null_q95:+.3f}")

    # Decision.
    hold_abs = bool(partial_loo >= 0.30)
    hold_null = bool(partial_loo > null_q95)
    loo_holds = bool(hold_abs and hold_null)
    print(f"\n[k562-loo] LOO driver — hold_abs={hold_abs}  "
          f"hold_null={hold_null}  OVERALL={loo_holds}")

    pd.DataFrame([{
        "partial_spearman_full": partial_full,
        "partial_spearman_LOO": partial_loo,
        "null_partial_mean": null_mean,
        "null_partial_std": null_std,
        "null_partial_q95": null_q95,
        "LOO_hold_abs_gt_0_30": hold_abs,
        "LOO_hold_above_null_q95": hold_null,
        "LOO_holds": loo_holds,
        "lead_eigenvalue_of_J_mean": float(ev[lead_idx]),
    }]).to_csv(OUT / "driver_check_LOO.csv", index=False)
    (OUT / "driver_check_LOO_nulls.csv").write_text(
        "partial_null\n" + "\n".join(f"{v:.6f}" for v in nulls) + "\n",
    )

    # Decide tag.
    gate_prev = json.loads((OUT / "controls_gate.json").read_text())
    gate = {
        **gate_prev,
        "LOO_partial_spearman": partial_loo,
        "LOO_hold_abs_gt_0_30": hold_abs,
        "LOO_hold_above_null_q95": hold_null,
        "LOO_holds": loo_holds,
        "tag_v06_perturb": bool(loo_holds),
    }
    (OUT / "controls_gate.json").write_text(json.dumps(gate, indent=2, default=str))
    print(f"\n[k562-loo] tag_v06_perturb = {gate['tag_v06_perturb']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
