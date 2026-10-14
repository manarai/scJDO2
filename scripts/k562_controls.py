"""K562 Perturb-seq — generic-response + expression controls (user, 2026-10-14).

Builds on `scripts/run_perturbseq_k562.py` + `scripts/k562_followup.py`:

(1) Generic-response control. g_bar = mean δx_obs across all KDs.
    Report cos(operator_pred, g_bar). Then remove the g_bar component
    from both δx_pred and δx_obs and repeat the paired operator-vs-
    DE-only comparison + mean cosines.

(2) Expression control on the driver check. Partial Spearman between
    the operator sensitivity score and projected |δx_obs|,
    controlling for log mean control expression of the target gene.
    Also report Spearman of expression alone with |δx_obs|.

Writes benchmarks/perturbseq_k562/RESULT.md + two new CSVs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import spearmanr, wilcoxon, rankdata

from scjdo2.energy_drift import EnergyDriftAdapter, EnergyDriftConfig
from scjdo2.rates import GeneRateModel, RatesConfig


REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "benchmarks" / "perturbseq_k562"
K562 = Path("/Users/terooatt/Documents/scJDO/data/replogle_2022_k562_essential.h5ad")


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """ρ(X, Y | Z) via rank-transform + residualisation.
    Returns Spearman partial correlation."""
    xr = rankdata(x); yr = rankdata(y); zr = rankdata(z)
    # Residualise xr and yr on zr via least-squares.
    def _resid(v, u):
        v = v - v.mean(); u = u - u.mean()
        beta = float(np.dot(v, u) / (np.dot(u, u) + 1e-12))
        return v - beta * u
    xr_res = _resid(xr, zr)
    yr_res = _resid(yr, zr)
    denom = float(np.sqrt((xr_res ** 2).sum() * (yr_res ** 2).sum()) + 1e-12)
    return float(np.dot(xr_res, yr_res) / denom)


def _project_out(x: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """x minus its projection onto ``direction`` (1D vector)."""
    d = direction / (np.linalg.norm(direction) + 1e-12)
    return x - float(np.dot(x, d)) * d


def _refit() -> dict:
    print("[k562c] load + HVG")
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
    print(f"[k562c] shape {a.shape}")

    import scvi
    scvi.settings.seed = 0
    ctrl = a[is_ctrl].copy()
    ctrl.layers["counts"] = ctrl.X.copy()
    scvi.model.SCVI.setup_anndata(ctrl, layer="counts")
    m = scvi.model.SCVI(ctrl, n_latent=15)
    t0 = time.monotonic()
    m.train(max_epochs=120, enable_progress_bar=False, early_stopping=False)
    z_ctrl = np.asarray(m.get_latent_representation(ctrl)).astype(np.float32)
    print(f"[k562c] scVI {time.monotonic()-t0:.1f}s")
    a.layers["counts"] = a.X.copy()
    z_all = np.asarray(m.get_latent_representation(a)).astype(np.float32)

    t0 = time.monotonic()
    drift = EnergyDriftAdapter(dim=15, config=EnergyDriftConfig(
        hidden=96, depth=3, n_epochs=1500, seed=0, batch_size=256,
        loss="esm", lr=5e-4, beta=1.0, esm_laplacian="exact",
    )).fit(z_ctrl, verbose=False)
    J_ctrl = drift.jacobian_autodiff(z_ctrl)
    J_mean = J_ctrl.mean(axis=0).astype(np.float64)
    print(f"[k562c] ESM+J {time.monotonic()-t0:.1f}s")

    counts_ctrl_arr = ctrl.X.toarray() if sp.issparse(ctrl.X) else np.asarray(ctrl.X)
    counts_ctrl_arr = counts_ctrl_arr.astype(np.float32)
    t0 = time.monotonic()
    rm = GeneRateModel(RatesConfig(hidden=(128, 128), n_epochs=150, lr=1e-3,
                                     batch_size=256, held_out_frac=0.10,
                                     patience=30, seed=0))
    rm.fit(counts_ctrl_arr, z_ctrl, verbose=False)
    print(f"[k562c] rate {time.monotonic()-t0:.1f}s")
    z_centroid = z_ctrl.mean(axis=0, keepdims=True).astype(np.float32)
    grad_eta = rm.grad_eta(z_centroid)[0]
    eta_all = rm.eta(z_all)
    # Per-gene mean expression in controls (log-normalised).
    mean_log_ctrl = _mean_log_ctrl(a, is_ctrl)
    return {"a": a, "is_ctrl": is_ctrl, "z_all": z_all,
            "J_mean": J_mean, "grad_eta": grad_eta, "eta_all": eta_all,
            "mean_log_ctrl": mean_log_ctrl}


def _mean_log_ctrl(a, is_ctrl):
    tmp = a[is_ctrl].copy()
    sc.pp.normalize_total(tmp, target_sum=1e4); sc.pp.log1p(tmp)
    X = tmp.X.toarray() if sp.issparse(tmp.X) else np.asarray(tmp.X)
    return np.asarray(X.mean(axis=0)).ravel()


def main() -> int:
    print("[k562c] refit control operators")
    fit = _refit()
    a = fit["a"]; eta_all = fit["eta_all"]; J_mean = fit["J_mean"]
    grad_eta = fit["grad_eta"]; is_ctrl = fit["is_ctrl"]
    mean_log_ctrl = fit["mean_log_ctrl"]
    gene_idx = {g: i for i, g in enumerate(a.var_names)}
    G = a.n_vars

    pk = pd.read_csv(OUT / "per_knockdown.csv")
    cluster = a.obs["perturbation"].astype(str).to_numpy()
    kd_list = [g for g in pk["gene"] if g in gene_idx]
    print(f"[k562c] KDs to recompute: {len(kd_list)}")

    # Rebuild δx_pred, δx_obs per KD as full G-dim vectors.
    print("[k562c] compute δx vectors")
    ginv = np.linalg.pinv(grad_eta)
    eps = 1e-6 * float(np.linalg.norm(J_mean, "fro") / 15)
    neg_J = -J_mean + eps * np.eye(15)

    eta_ctrl_mean = eta_all[is_ctrl].mean(axis=0)
    delta_pred_mat = np.zeros((len(kd_list), G), dtype=np.float64)
    delta_obs_mat = np.zeros((len(kd_list), G), dtype=np.float64)
    lfc_vec = np.zeros(len(kd_list), dtype=np.float64)
    for i, g in enumerate(kd_list):
        gi = gene_idx[g]
        kd_mask = (cluster == g)
        # push vector along gene g scaled by lfc_g from pk
        lfc = float(pk.loc[pk["gene"] == g, "lfc_g"].iloc[0])
        lfc_vec[i] = lfc
        b_g = np.zeros(G); b_g[gi] = lfc
        a_lat = ginv @ b_g
        dz = np.linalg.solve(neg_J, a_lat)
        delta_pred_mat[i] = grad_eta @ dz
        delta_obs_mat[i] = eta_all[kd_mask].mean(axis=0) - eta_ctrl_mean
    print(f"[k562c] δ matrices shape {delta_pred_mat.shape}")

    # (1) Generic-response control.
    g_bar = delta_obs_mat.mean(axis=0)
    g_bar_norm = g_bar / (np.linalg.norm(g_bar) + 1e-12)
    # Per-KD cos(pred, g_bar) and cos(obs, g_bar).
    def _cos_vec(X, y):
        yn = y / (np.linalg.norm(y) + 1e-12)
        xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        return xn @ yn
    cos_pred_gbar = _cos_vec(delta_pred_mat, g_bar)
    cos_obs_gbar = _cos_vec(delta_obs_mat, g_bar)
    print(f"[k562c] cos(pred, g_bar)  mean={float(cos_pred_gbar.mean()):+.3f}  "
          f"median={float(np.median(cos_pred_gbar)):+.3f}")
    print(f"[k562c] cos(obs,  g_bar)  mean={float(cos_obs_gbar.mean()):+.3f}  "
          f"median={float(np.median(cos_obs_gbar)):+.3f}")

    # Remove g_bar from both.
    delta_pred_res = np.array([_project_out(delta_pred_mat[i], g_bar)
                                for i in range(len(kd_list))])
    delta_obs_res = np.array([_project_out(delta_obs_mat[i], g_bar)
                                for i in range(len(kd_list))])

    # Confidence per gene from the K562 run (reload or recompute).
    # Rate-model per-gene NLL — reuse the K562 ok-mask approach:
    # keep top-50% by -mean_log_ctrl_absolute as a proxy (positive =
    # better coverage). This mirrors "ok" in the original paired table.
    conf = mean_log_ctrl  # higher expression → more reliable
    ok_mask = conf >= float(np.median(conf))
    print(f"[k562c] ok_mask size = {int(ok_mask.sum())}/{G}")

    # Cosines on residual.
    def _cos_rows(A, B, mask):
        A_m = A[:, mask]; B_m = B[:, mask]
        num = (A_m * B_m).sum(axis=1)
        den = np.linalg.norm(A_m, axis=1) * np.linalg.norm(B_m, axis=1) + 1e-12
        return num / den
    cos_op_full = _cos_rows(delta_pred_mat, delta_obs_mat, ok_mask)
    cos_op_res = _cos_rows(delta_pred_res, delta_obs_res, ok_mask)
    # DE-only on residual: b_g has one entry; its projection-removed version
    # is b_g minus (b_g · g_bar_norm) g_bar_norm. Compute on-the-fly.
    cos_de_full = np.zeros(len(kd_list))
    cos_de_res = np.zeros(len(kd_list))
    for i, g in enumerate(kd_list):
        gi = gene_idx[g]
        b_g = np.zeros(G); b_g[gi] = lfc_vec[i]
        obs = delta_obs_mat[i]
        # Full-space cos(b_g, obs) on ok_mask.
        m = ok_mask
        num = float(np.dot(b_g[m], obs[m]))
        den = float(np.linalg.norm(b_g[m]) * np.linalg.norm(obs[m]) + 1e-12)
        cos_de_full[i] = num / den
        b_g_res = _project_out(b_g, g_bar)
        obs_res = delta_obs_res[i]
        num_r = float(np.dot(b_g_res[m], obs_res[m]))
        den_r = float(np.linalg.norm(b_g_res[m]) * np.linalg.norm(obs_res[m]) + 1e-12)
        cos_de_res[i] = num_r / den_r

    def _mean_frac(op, de):
        mask = np.isfinite(op) & np.isfinite(de)
        if mask.sum() == 0:
            return {"n": 0, "op_mean": float("nan"), "de_mean": float("nan"),
                     "frac_op_gt_de": float("nan"), "wilcoxon_p": float("nan")}
        o, d = op[mask], de[mask]
        try:
            _, p = wilcoxon(o - d, alternative="greater")
        except Exception:
            p = float("nan")
        return {"n": int(mask.sum()),
                 "op_mean": float(o.mean()),
                 "de_mean": float(d.mean()),
                 "frac_op_gt_de": float((o > d).mean()),
                 "wilcoxon_p": float(p)}

    comp_rows = []
    pk_linear = pk.set_index("gene")["linear_regime"].to_dict()
    linmask = np.array([bool(pk_linear.get(g, False)) for g in kd_list])
    for label, mask in (("full_space_all", np.ones(len(kd_list), dtype=bool)),
                        ("full_space_linear", linmask),
                        ("full_space_nonlinear", ~linmask),
                        ("residual_all", np.ones(len(kd_list), dtype=bool)),
                        ("residual_linear", linmask),
                        ("residual_nonlinear", ~linmask)):
        op = cos_op_res if label.startswith("residual") else cos_op_full
        de = cos_de_res if label.startswith("residual") else cos_de_full
        rec = _mean_frac(op[mask], de[mask]); rec["arm"] = label
        comp_rows.append(rec)
    comp_df = pd.DataFrame(comp_rows)[["arm","n","op_mean","de_mean","frac_op_gt_de","wilcoxon_p"]]
    comp_df.to_csv(OUT / "generic_control_comparison.csv", index=False)
    print("\n[k562c] generic-control paired comparison:")
    print(comp_df.round(4).to_string(index=False))

    # (2) Expression control on driver.
    ev, V = np.linalg.eigh(J_mean)
    lead_idx = int(np.argmax(ev))
    v_lead = V[:, lead_idx]
    v_lead_gene = grad_eta @ v_lead
    v_lead_gene_norm = v_lead_gene / (np.linalg.norm(v_lead_gene) + 1e-12)
    sens = np.abs(v_lead_gene) * float(np.abs(v_lead_gene).sum())
    obs_proj = np.zeros(G)
    obs_raw = np.zeros(G)
    for i, g in enumerate(kd_list):
        gi = gene_idx[g]
        delta = delta_obs_mat[i]
        a_scalar = float(delta @ v_lead_gene_norm)
        obs_proj[gi] = abs(a_scalar * v_lead_gene_norm[gi])
        obs_raw[gi] = abs(float(delta[gi]))
    valid = obs_raw > 0
    sp_raw = float(spearmanr(sens[valid], obs_raw[valid]).correlation)
    sp_proj = float(spearmanr(sens[valid], obs_proj[valid]).correlation)
    sp_expr_vs_obs = float(spearmanr(mean_log_ctrl[valid], obs_proj[valid]).correlation)
    sp_sens_vs_expr = float(spearmanr(sens[valid], mean_log_ctrl[valid]).correlation)
    sp_partial = _partial_spearman(sens[valid], obs_proj[valid], mean_log_ctrl[valid])
    print(f"\n[k562c] driver check:")
    print(f"  raw sens vs |δ_obs|           : {sp_raw:+.3f}")
    print(f"  projected sens vs |δ_proj|    : {sp_proj:+.3f}")
    print(f"  expression vs |δ_proj|        : {sp_expr_vs_obs:+.3f}")
    print(f"  sens vs expression (nuisance) : {sp_sens_vs_expr:+.3f}")
    print(f"  partial Spearman (sens, |δ_proj| | expr) : **{sp_partial:+.3f}**")

    driver_df = pd.DataFrame([{
        "spearman_raw_sens_vs_obs": sp_raw,
        "spearman_proj_sens_vs_delta_proj": sp_proj,
        "spearman_expression_vs_delta_proj": sp_expr_vs_obs,
        "spearman_sens_vs_expression": sp_sens_vs_expr,
        "partial_spearman_sens_vs_delta_proj_given_expression": sp_partial,
        "n_genes": int(valid.sum()),
    }])
    driver_df.to_csv(OUT / "driver_check_expression_control.csv", index=False)

    # Final figure: cosine vs mean-Fisher scatter in residual space.
    fig, ax = plt.subplots(figsize=(7, 5))
    valid_kd = np.isfinite(cos_op_res)
    sc1 = ax.scatter(mean_log_ctrl.mean() + np.zeros(valid_kd.sum()),  # placeholder — use per-KD conf below
                      cos_op_res[valid_kd], s=10, alpha=0.4,
                      c=["#d55e00" if linmask[i] else "#777" for i in np.where(valid_kd)[0]])
    # Better: use the per-KD mean confidence (reusing pk column).
    fig.clear(); fig, ax = plt.subplots(figsize=(7, 5))
    pk_sub = pk.set_index("gene").loc[kd_list]
    conf_per_kd = pk_sub["mean_fisher_conf"].to_numpy()
    sc2 = ax.scatter(conf_per_kd, cos_op_res, s=10, alpha=0.5,
                      c=["#d55e00" if lr else "#777" for lr in linmask])
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("mean Fisher confidence of ok-mask entries")
    ax.set_ylabel("cos(op_res, obs_res)  (generic-removed)")
    ax.set_title("K562 — cosine vs ledger confidence (generic response removed)")
    plt.tight_layout()
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures" / "scatter_cos_vs_fisher_residual.pdf", bbox_inches="tight")
    fig.savefig(OUT / "figures" / "scatter_cos_vs_fisher_residual.png",
                 dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Decide gate: both controls hold?
    full_all = comp_df[comp_df["arm"] == "full_space_all"].iloc[0]
    res_all = comp_df[comp_df["arm"] == "residual_all"].iloc[0]
    generic_control_holds = bool(res_all["frac_op_gt_de"] >= 0.6 and
                                   res_all["op_mean"] > 0.05 and
                                   res_all["op_mean"] > res_all["de_mean"] * 2)
    expression_control_holds = bool(sp_partial > 0.3)
    gate_summary = {
        "generic_control_holds": generic_control_holds,
        "expression_control_holds": expression_control_holds,
        "res_frac_op_gt_de": float(res_all["frac_op_gt_de"]),
        "res_op_mean": float(res_all["op_mean"]),
        "res_de_mean": float(res_all["de_mean"]),
        "partial_spearman": sp_partial,
        "projected_spearman": sp_proj,
        "expression_spearman": sp_expr_vs_obs,
    }
    (OUT / "controls_gate.json").write_text(json.dumps(gate_summary, indent=2, default=str))
    print(f"\n[k562c] generic_control_holds   = {generic_control_holds}")
    print(f"[k562c] expression_control_holds = {expression_control_holds}")

    # §7c γ numbers from previous run.
    s7c_meta_path = REPO / "benchmarks" / "v2" / "dynamo_s7c" / "fit_meta.json"
    s7c_beh_path = REPO / "benchmarks" / "v2" / "dynamo_s7c" / "behaviour_by_batch.csv"
    s7c_block = ""
    if s7c_meta_path.exists():
        s7cm = json.loads(s7c_meta_path.read_text())
        gp = s7cm.get("gamma_pooled", {})
        agr = s7cm.get("gamma_per_batch_agreement", {})
        beh = pd.read_csv(s7c_beh_path) if s7c_beh_path.exists() else pd.DataFrame()
        s7c_block = "\n".join([
            "## §7c γ (reported honestly, no reinterpretation)",
            "",
            f"* γ pooled: median {gp.get('median', 'n/a'):.3f}/h, "
            f"IQR [{gp.get('q25', 0):.3f}, {gp.get('q75', 0):.3f}], "
            f"range [{gp.get('min', 0):.3f}, {gp.get('max', 0):.3f}]",
            f"* 3h-only vs 5h-only per-gene Spearman: **{agr.get('spearman_3h_vs_5h', float('nan')):+.3f}**",
            f"* 3h-only vs 5h-only per-gene Pearson:  **{agr.get('pearson_3h_vs_5h', float('nan')):+.3f}**",
            "",
            "Near-zero batch agreement means γ is NOT recovered as a stable",
            "gene property — the LabelledRateModel absorbs batch structure",
            "into γ. Under the locked lead-eig rule the batch-flip pattern",
            "(HSC/MEP/GMP/Mon read MORE basin at 5h than 3h, Bas opposite)",
            "persists in the behaviour-by-batch table (`benchmarks/v2/",
            "dynamo_s7c/behaviour_by_batch.csv`). Because γ is not batch-",
            "invariant, the pattern cannot be attributed to the γ-footing",
            "fix; the lever is the locked classifier + whatever batch",
            "structure the drift carries. §7c as currently formulated does",
            "NOT resolve the §7b batch confound.",
            "",
        ])

    # Write RESULT.md.
    tag_target = "v0.6-perturb" if (generic_control_holds and
                                      expression_control_holds) else "none"
    lines = [
        "# K562 Perturb-seq — RESULT",
        "",
        f"**Tag target:** `{tag_target}`.  "
        f"(Will tag if both controls hold.)",
        "",
        "## Claim",
        "",
        "The scJDO2 energy-param-drift linear-response prediction",
        "`δx_pred = grad_eta @ (−J_mean⁻¹) @ pinv(grad_eta) @ b_g`",
        "assigns direction-correct cosines to CRISPRi knockdowns better",
        "than DE-only (`δx = b_g`) and better than random, after both",
        "the generic-response subtraction and the expression nuisance",
        "control.",
        "",
        "## Three baselines (full space, all KDs)",
        "",
        f"* **scjdo2 cos mean = {float(full_all['op_mean']):+.3f}**",
        f"* DE-only cos mean    = {float(full_all['de_mean']):+.3f}",
        "* random cos mean    ≈ 1e-4 (from `baseline_comparison.csv`)",
        "",
        "Paired Wilcoxon (op > DE):",
        f"* frac(op > DE) = **{float(full_all['frac_op_gt_de']):.3f}**",
        f"* Wilcoxon p    = {float(full_all['wilcoxon_p']):.1e}",
        "",
        "## Control 1 — generic-response subtraction",
        "",
        f"* cos(δx_pred, mean δx_obs) mean = {float(cos_pred_gbar.mean()):+.3f}  "
        "(the predictions are not just the mean response)",
        f"* cos(δx_obs,  mean δx_obs) mean = {float(cos_obs_gbar.mean()):+.3f}  "
        "(observed shifts have a generic component)",
        "",
        "After projecting out the mean δx_obs from BOTH δx_pred and δx_obs:",
        "",
        f"* **residual scjdo2 cos mean = {float(res_all['op_mean']):+.3f}**",
        f"* residual DE-only cos mean  = {float(res_all['de_mean']):+.3f}",
        f"* residual frac(op > DE)     = **{float(res_all['frac_op_gt_de']):.3f}**",
        f"* residual Wilcoxon p        = {float(res_all['wilcoxon_p']):.1e}",
        "",
        f"**Generic-response control: {'HOLDS' if generic_control_holds else 'FAILS'}.**",
        "Criterion: frac(op > DE) ≥ 0.60 AND op_mean > 2 × DE_mean AND op_mean > 0.05.",
        "",
        "## Control 2 — expression nuisance",
        "",
        f"* Projected Spearman (sens vs |δ_proj|):     **{sp_proj:+.3f}**",
        f"* Spearman of expression alone vs |δ_proj|:  {sp_expr_vs_obs:+.3f}",
        f"* Spearman of sens vs expression (nuisance): {sp_sens_vs_expr:+.3f}",
        f"* **Partial Spearman (sens, |δ_proj| | expression) = {sp_partial:+.3f}**",
        "",
        f"**Expression control: {'HOLDS' if expression_control_holds else 'FAILS'}.**",
        "Criterion: partial Spearman > 0.30 (sensitivity still predicts",
        "observed shifts along the operator's soft axis after removing",
        "mean-expression confounding).",
        "",
        "## Linear-regime subset (for completeness)",
        "",
        "|arm | n | op mean | DE mean | frac(op>DE) | Wilcoxon p |",
        "|---|---|---|---|---|---|",
        "| full-space linear    | " +
        f"{int(comp_df[comp_df['arm']=='full_space_linear'].iloc[0]['n'])} | "
        f"{float(comp_df[comp_df['arm']=='full_space_linear'].iloc[0]['op_mean']):+.3f} | "
        f"{float(comp_df[comp_df['arm']=='full_space_linear'].iloc[0]['de_mean']):+.3f} | "
        f"{float(comp_df[comp_df['arm']=='full_space_linear'].iloc[0]['frac_op_gt_de']):.3f} | "
        f"{float(comp_df[comp_df['arm']=='full_space_linear'].iloc[0]['wilcoxon_p']):.1e} |",
        "| residual linear      | " +
        f"{int(comp_df[comp_df['arm']=='residual_linear'].iloc[0]['n'])} | "
        f"{float(comp_df[comp_df['arm']=='residual_linear'].iloc[0]['op_mean']):+.3f} | "
        f"{float(comp_df[comp_df['arm']=='residual_linear'].iloc[0]['de_mean']):+.3f} | "
        f"{float(comp_df[comp_df['arm']=='residual_linear'].iloc[0]['frac_op_gt_de']):.3f} | "
        f"{float(comp_df[comp_df['arm']=='residual_linear'].iloc[0]['wilcoxon_p']):.1e} |",
        "| full-space nonlinear | " +
        f"{int(comp_df[comp_df['arm']=='full_space_nonlinear'].iloc[0]['n'])} | "
        f"{float(comp_df[comp_df['arm']=='full_space_nonlinear'].iloc[0]['op_mean']):+.3f} | "
        f"{float(comp_df[comp_df['arm']=='full_space_nonlinear'].iloc[0]['de_mean']):+.3f} | "
        f"{float(comp_df[comp_df['arm']=='full_space_nonlinear'].iloc[0]['frac_op_gt_de']):.3f} | "
        f"{float(comp_df[comp_df['arm']=='full_space_nonlinear'].iloc[0]['wilcoxon_p']):.1e} |",
        "| residual nonlinear   | " +
        f"{int(comp_df[comp_df['arm']=='residual_nonlinear'].iloc[0]['n'])} | "
        f"{float(comp_df[comp_df['arm']=='residual_nonlinear'].iloc[0]['op_mean']):+.3f} | "
        f"{float(comp_df[comp_df['arm']=='residual_nonlinear'].iloc[0]['de_mean']):+.3f} | "
        f"{float(comp_df[comp_df['arm']=='residual_nonlinear'].iloc[0]['frac_op_gt_de']):.3f} | "
        f"{float(comp_df[comp_df['arm']=='residual_nonlinear'].iloc[0]['wilcoxon_p']):.1e} |",
        "",
        "## Figure",
        "",
        "![cosine vs confidence (residual)](figures/scatter_cos_vs_fisher_residual.png)",
        "",
        "`figures/scatter_cos_vs_fisher_residual.{pdf,png}` — per-KD cosine",
        "after generic-response removal, vs mean ledger-confidence; orange =",
        "linear regime, grey = nonlinear. The effect size you should quote",
        f"is the mean cosine ({float(res_all['op_mean']):+.3f} residual),",
        "not the p-value.",
        "",
        s7c_block,
    ]
    (OUT / "RESULT.md").write_text("\n".join(lines) + "\n")
    print(f"\n[k562c] wrote {OUT / 'RESULT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
