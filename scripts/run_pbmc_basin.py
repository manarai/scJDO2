"""PBMC positive-control for the basin arm (SUMMARY §5).

The four haematopoiesis datasets in §1-§4 are progenitor compartments
with no mature resting cells — zero holders there is consistent with
biology, not a positive test of the estimator. This script runs the
scJDO2 pipeline on a homeostatic PBMC dataset where resting mature
cells (CD4 T, CD8 T, B, classical mono) are the dominant populations
and should sit at or near fixed points of their lineages.

Spec (user, 2026-10):

* **Data.** 10x PBMC 3k via ``scanpy.datasets.pbmc3k`` + louvain labels
  from ``pbmc3k_processed`` (2638 cells, 8 types). PBMC 10k v3 was the
  preferred target but requires external download; pbmc3k already
  contains the resting mature types, so it is the recorded fallback.
* **Embedding.** scVI. ``d_latent`` selected by held-out NB marginal
  log-likelihood (sweep over {5, 10, 15}).
* **Drift.** DSM score-only (``inject_velocity_prior=False``,
  ``use_residual=False``). The resulting J is the Hessian of log p —
  symmetric — so only holder / drift / fork are reachable; oscillator
  / sensitive classes are out of reach of snapshot data by
  construction, which this run acknowledges rather than treats as a
  bug.
* **Gene set.** Full expressed set (``sc.pp.filter_genes(min_cells=3)``);
  cycle genes (Tirosh S + G2M) kept in the embedding and the gene-
  space anchor fit.
* **Anchor fit.** SNR-ranked live columns (R16 default); cycle genes
  retained.
* **Known-rotation projection.** The cycle-score subspace in the
  latent frame (``U_cycle`` = QR of latent regression of S / G2M
  scores) is projected out of each ``J_c``:
  ``J_c ← (I - UU^T) J_c (I - UU^T)``. This matches §7a variant C.
* **Permutation null.** Shuffle latent positions, refit DSM, record
  the resulting lead_re distribution; compare to the real run.
* **Outputs.** Behaviour by cell type; median lead_re + frac(Re>0);
  tangent rank distribution; for each holder-dominated type, the
  dominant archetype's eigenvalue range (relative units) and the
  top-10 genes along the stiffest and softest modes (with Fisher).
* **Gate.** If the four major resting types (CD4 T, CD8 T, B,
  CD14+ Monocytes) are predominantly holders → tag the repo and
  continue to §7c. Else stop and report the diagnostic.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from scjdo2 import opspace
from scjdo2.data import (
    CYCLE_GENES,
    NK_PROLIF_TYPES,
    RESTING_TYPES,
    load_pbmc3k,
    present_cycle_genes,
)
from scjdo2.drift import DriftAdapter, DriftConfig
from scjdo2.pipeline import fit_rate_and_gene_field


REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "benchmarks" / "v2" / "pbmc_basin"


# -----------------------------------------------------------------
# scVI embedding + d_latent sweep by held-out NB marginal likelihood
# -----------------------------------------------------------------


def _fit_scvi(adata, n_latent: int, n_epochs: int, seed: int = 0,
              batch_key: str | None = None):
    """Return (latent, marginal_ll). Tiny wrapper over scvi.model.SCVI."""
    import scvi
    scvi.settings.seed = seed
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass
    layers_backup = adata.layers.get("counts")
    adata.layers["counts"] = adata.X.copy() if layers_backup is None else layers_backup
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=batch_key)
    m = scvi.model.SCVI(adata, n_latent=int(n_latent))
    m.train(max_epochs=int(n_epochs), check_val_every_n_epoch=10,
            early_stopping=False, enable_progress_bar=False)
    z = np.asarray(m.get_latent_representation(adata)).astype(np.float32)
    try:
        mll = float(m.get_marginal_ll(adata, n_mc_samples=50))
    except Exception as err:
        print(f"[pbmc]   marginal_ll fail for d={n_latent}: {err}")
        mll = float("nan")
    return z, mll, m


def _sweep_d_latent(adata, d_options: list[int], sweep_epochs: int,
                    seed: int = 0) -> tuple[int, pd.DataFrame]:
    """Short-training sweep across ``d_options``; return the best
    d_latent by marginal log-likelihood (highest = best fit).

    scvi 1.2's ``get_marginal_ll`` returns a (per-cell averaged) log
    likelihood: negative numbers, less-negative = better fit.
    """
    rows = []
    for d in d_options:
        print(f"[pbmc]   sweep d_latent={d}  epochs={sweep_epochs}")
        try:
            _, mll, _ = _fit_scvi(adata.copy(), n_latent=d, n_epochs=sweep_epochs,
                                   seed=seed)
        except Exception as err:
            print(f"[pbmc]     fit failed: {err}")
            mll = float("nan")
        rows.append({"d_latent": d, "marginal_ll": mll})
    df = pd.DataFrame(rows)
    print("[pbmc] sweep result (higher marginal LL = better held-out fit):")
    print(df.to_string(index=False))
    valid = df.dropna()
    best = int(valid.loc[valid["marginal_ll"].idxmax(), "d_latent"]) if len(valid) else int(d_options[len(d_options) // 2])
    return best, df


# -----------------------------------------------------------------
# Known-rotation projection on the latent cycle subspace
# -----------------------------------------------------------------


def _score(X_log: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros(X_log.shape[0], dtype=np.float32)
    bg = X_log.mean(axis=1)
    sig = X_log[:, mask].mean(axis=1)
    return (sig - bg).astype(np.float32)


def _latent_cycle_basis(z: np.ndarray, s_score: np.ndarray,
                        g2m_score: np.ndarray) -> np.ndarray:
    def _unit(w): return w / (np.linalg.norm(w) + 1e-12)
    def _reg(y): return _unit(np.linalg.lstsq(z, y, rcond=None)[0])
    u_S = _reg(s_score)
    u_G2M = _reg(g2m_score)
    U = np.column_stack([u_S, u_G2M]).astype(np.float64)
    Q, _ = np.linalg.qr(U)
    return Q


def _project_out_latent(J: np.ndarray, U: np.ndarray) -> np.ndarray:
    I = np.eye(J.shape[1], dtype=np.float64)
    P = I - U @ U.T
    return np.einsum("ij,cjk,kl->cil", P, J.astype(np.float64), P).astype(np.float32)


# -----------------------------------------------------------------
# Permutation null: shuffle z-positions, refit DSM, measure lead_re
# -----------------------------------------------------------------


def _permutation_null(z: np.ndarray, t: np.ndarray, n_perm: int,
                      drift_cfg: DriftConfig, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_perm):
        perm = rng.permutation(z.shape[0])
        z_perm = z[perm]
        drift = DriftAdapter(dim=z.shape[1], config=drift_cfg).fit(
            z_perm, t=t, verbose=False,
        )
        J = drift.jacobian_autodiff(z_perm)
        lead = _lead_re(J)
        rows.append({
            "perm": p,
            "median_lead_re": float(np.median(lead)),
            "frac_pos": float((lead > 0).mean()),
            "frac_neg": float((lead < 0).mean()),
            "min_lead_re": float(np.min(lead)),
            "max_lead_re": float(np.max(lead)),
        })
    return pd.DataFrame(rows)


def _lead_re(J: np.ndarray) -> np.ndarray:
    out = np.empty(J.shape[0], dtype=np.float64)
    for c in range(J.shape[0]):
        try:
            ev = np.linalg.eigvals(J[c].astype(np.float64))
            out[c] = float(np.max(ev.real))
        except np.linalg.LinAlgError:
            out[c] = np.nan
    return out


# -----------------------------------------------------------------
# Per-type archetype: stiffest / softest modes + Fisher
# -----------------------------------------------------------------


def _mode_table(info: dict, gene_names: np.ndarray, mode_col: int,
                side: str, top: int = 10) -> pd.DataFrame:
    """Rank genes by |lifted eigvec[:, mode_col]| on either right or
    left side. ``info`` is a single archetype dict from
    ``uns['scjdo2']['archetypes'][k]``."""
    key = "eigvec_full" if side == "right" else "eigvec_left_full"
    V = np.asarray(info[key])
    if mode_col >= V.shape[1]:
        return pd.DataFrame()
    w = V[:, mode_col]
    mag = np.abs(w)
    fisher = np.asarray(info["fisher"])
    reasons = np.asarray(info["reason"])
    order = np.argsort(-mag)
    rows = []
    for i in order[:top]:
        rows.append({
            "rank": int(np.argwhere(order == i)[0, 0]) + 1,
            "gene": str(gene_names[i]),
            "mag": float(mag[i]),
            "weight_re": float(w.real[i]),
            "fisher": float(fisher[i]),
            "reason": str(reasons[i]),
        })
    return pd.DataFrame(rows)


def _per_type_report(adata, resting_types) -> dict:
    beh = adata.obs["op_behaviour"].astype(str).to_numpy()
    cluster = adata.obs["cluster"].astype(str).to_numpy()
    lead_re = adata.obs["lead_re"].to_numpy(dtype=np.float64)
    rows = []
    for cl in sorted(set(cluster)):
        m = cluster == cl
        c = pd.Series(beh[m]).value_counts()
        rec = {
            "cluster": cl,
            "n": int(m.sum()),
            "is_resting": cl in resting_types,
        }
        for k in ("fork", "holder", "drift", "damped_oscillator",
                  "growing_oscillator"):
            rec[k] = int(c.get(k, 0))
        rec["frac_fork"] = rec["fork"] / rec["n"] if rec["n"] else 0.0
        rec["frac_holder"] = rec["holder"] / rec["n"] if rec["n"] else 0.0
        lr = lead_re[m]
        lr = lr[np.isfinite(lr)]
        rec["median_lead_re"] = float(np.median(lr)) if lr.size else float("nan")
        rec["frac_pos_lead_re"] = float((lr > 0).mean()) if lr.size else float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)


def _dominant_archetype_for_cluster(adata, cl: str) -> int | None:
    W = adata.obsm.get("archetype_weights")
    if W is None:
        return None
    cluster = adata.obs["cluster"].astype(str).to_numpy()
    m = cluster == cl
    if not m.any():
        return None
    k = int(np.argmax(W[m].sum(axis=0)))
    return k


# -----------------------------------------------------------------
# Figures: operator-space + ledger
# -----------------------------------------------------------------


def _figures(adata, out: Path, decision_label: str = "PBMC3k"):
    uns = adata.uns.get("scjdo2", {})
    beh = adata.obs["op_behaviour"].astype(str).to_numpy()
    lead_re = adata.obs["lead_re"].to_numpy(dtype=np.float64)
    tsne = adata.obsm.get("X_tsne")
    if tsne is None:
        tsne = adata.obsm.get("X_umap")
    tsne = np.asarray(tsne) if tsne is not None else None

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    palette = {"fork": "#d55e00", "holder": "#0072b2",
               "growing_oscillator": "#009e73",
               "damped_oscillator": "#56b4e9",
               "drift": "#999999"}
    if tsne is not None and tsne.shape[1] >= 2:
        for b in sorted(set(beh)):
            m = beh == b
            axes[0, 0].scatter(tsne[m, 0], tsne[m, 1], s=4, alpha=0.6,
                               label=f"{b} ({m.sum()})",
                               c=palette.get(b, "#333"))
        axes[0, 0].legend(loc="upper right", fontsize=7, markerscale=1.5)
        axes[0, 0].set_title(f"{decision_label} — behaviour on embedding")
        axes[0, 0].set_xlabel("dim-1"); axes[0, 0].set_ylabel("dim-2")

    # Lead_re per cluster (boxplot).
    clusters = adata.obs["cluster"].astype(str).to_numpy()
    cl_names = sorted(set(clusters))
    data = [lead_re[clusters == c] for c in cl_names]
    axes[0, 1].boxplot(data, labels=cl_names, showfliers=False)
    axes[0, 1].axhline(0.0, color="k", lw=0.5, ls="--")
    axes[0, 1].set_title("lead_re per cluster")
    axes[0, 1].tick_params(axis="x", labelrotation=45)

    # Tangent rank histogram.
    tr = uns.get("op_tangent_rank")
    if tr is not None:
        tr = np.asarray(tr)
        axes[1, 0].hist(tr, bins=np.arange(0, tr.max() + 2) - 0.5,
                        color="#444")
        axes[1, 0].set_title(f"tangent rank (median {int(np.median(tr))})")
        axes[1, 0].set_xlabel("rank"); axes[1, 0].set_ylabel("cells")

    # Per-cell Fisher total mass (ledger figure).
    led = uns.get("fisher_ledger") or {}
    try:
        F_list = led.get("fisher") if hasattr(led, "get") else getattr(led, "fisher", None)
        if F_list is not None:
            totals = np.array([float(F.diagonal().sum()) for F in F_list],
                              dtype=np.float64)
            totals = totals[np.isfinite(totals) & (totals > 0)]
            if totals.size:
                axes[1, 1].hist(np.log10(totals), bins=50, color="#246")
                axes[1, 1].set_title("per-cell Σ Fisher diag (log10)")
                axes[1, 1].set_xlabel("log10 Σ diag F_c"); axes[1, 1].set_ylabel("cells")
    except Exception as err:
        axes[1, 1].text(0.5, 0.5, f"ledger err: {err}", ha="center",
                        transform=axes[1, 1].transAxes)

    plt.tight_layout()
    for ext in (".pdf", ".png"):
        plt.savefig(out / f"operator_space_ledger{ext}", dpi=140)
    plt.close(fig)


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--d-latent-sweep", type=str, default="5,10,15",
                   help="comma-separated d_latent options for the scVI sweep")
    p.add_argument("--sweep-epochs", type=int, default=80)
    p.add_argument("--scvi-epochs", type=int, default=200)
    p.add_argument("--drift-epochs", type=int, default=1500)
    p.add_argument("--rate-epochs", type=int, default=150)
    p.add_argument("--n-anchors", type=int, default=200)
    p.add_argument("--anchor-max-live", type=int, default=200)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--n-perm", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-cells", type=int, default=3,
                   help="min_cells filter for the FULL gene set")
    p.add_argument("--n-top-genes", type=int, default=0,
                   help="if > 0, restrict to the top HVG (cycle genes + "
                        "resting markers always kept); 0 keeps the full "
                        "post-min_cells gene set")
    p.add_argument("--skip-sweep", action="store_true",
                   help="skip the d_latent sweep and use d=10")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print("[pbmc] load pbmc3k + louvain labels")
    a = load_pbmc3k()
    print(f"[pbmc]   shape={a.shape}  clusters={sorted(set(a.obs['cluster']))}")
    for cl in sorted(set(a.obs["cluster"])):
        print(f"[pbmc]     {cl}: {(a.obs['cluster'] == cl).sum()}")

    # Full gene set after min_cells filter.
    sc.pp.filter_genes(a, min_cells=int(args.min_cells))
    print(f"[pbmc]   full gene set after min_cells≥{args.min_cells}: {a.n_vars} genes")

    # Optional HVG restriction with cycle + marker override. 13k genes
    # on a 36 GB machine hits ~26 GB of CSR in Phase C — this is the
    # practical cap (noted in SUMMARY §5). Default: no restriction.
    marker_keep = ("CD3D", "CD4", "CD8A", "CD8B", "MS4A1", "CD79A",
                   "NKG7", "GNLY", "CD14", "FCGR3A", "CST3", "LYZ",
                   "FCER1A", "PPBP")
    if args.n_top_genes and args.n_top_genes < a.n_vars:
        tmp = a.copy()
        sc.pp.normalize_total(tmp, target_sum=1e4)
        sc.pp.log1p(tmp)
        sc.pp.highly_variable_genes(tmp, n_top_genes=int(args.n_top_genes),
                                    flavor="seurat")
        hv = tmp.var["highly_variable"].to_numpy()
        for g in list(CYCLE_GENES) + list(marker_keep):
            if g in a.var_names:
                hv[a.var_names.get_loc(g)] = True
        a = a[:, hv].copy()
        print(f"[pbmc]   HVG restriction → {a.n_vars} genes "
              f"(cycle + resting markers forced in)")

    # Cycle genes present.
    cyc_present = present_cycle_genes(a)
    print(f"[pbmc]   cycle genes present: {len(cyc_present)} / {len(CYCLE_GENES)}")

    # Cell-cycle scores (log1p-normalised background subtraction) for the
    # known-rotation projection.
    tmp = a.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    X_log = np.asarray(tmp.X.toarray() if hasattr(tmp.X, "toarray") else tmp.X,
                       dtype=np.float32)
    s_mask = np.array([g in CYCLE_GENES[:43] for g in a.var_names], dtype=bool)   # S
    g2m_mask = np.array([g in CYCLE_GENES[43:] for g in a.var_names], dtype=bool)  # G2M
    S_score = _score(X_log, s_mask)
    G2M_score = _score(X_log, g2m_mask)
    a.obs["_cycle_S"] = S_score
    a.obs["_cycle_G2M"] = G2M_score

    # scVI d_latent sweep + full fit.
    d_options = [int(x) for x in args.d_latent_sweep.split(",")]
    if args.skip_sweep:
        best_d = 10
        sweep_df = pd.DataFrame([{"d_latent": 10, "marginal_ll": float("nan"),
                                   "note": "sweep skipped"}])
    else:
        print(f"\n[pbmc] scVI d_latent sweep: {d_options}")
        best_d, sweep_df = _sweep_d_latent(a, d_options, args.sweep_epochs,
                                           seed=args.seed)
    sweep_df.to_csv(OUT / "scvi_d_latent_sweep.csv", index=False)
    print(f"[pbmc]   chosen d_latent = {best_d}")

    print(f"\n[pbmc] scVI final fit (d_latent={best_d}, epochs={args.scvi_epochs})")
    t0 = time.monotonic()
    z, mll, _m = _fit_scvi(a, n_latent=best_d, n_epochs=args.scvi_epochs,
                           seed=args.seed)
    print(f"[pbmc]   scVI {time.monotonic()-t0:.1f}s  marginal_ll={mll:.1f}")
    a.obsm["X_scVI"] = z
    # Pseudotime placeholder — DSM score path uses it only as a per-cell
    # time; since the drift is time-free (we condition on t=0 only), use
    # the first latent coord sorted. Keeping constant t would also work.
    a.obs["pseudotime"] = np.zeros(a.n_obs, dtype=np.float32)

    # Latent cycle basis for the known-rotation projection.
    U_cycle = _latent_cycle_basis(z, S_score, G2M_score)
    print(f"[pbmc]   latent cycle basis: shape {U_cycle.shape}  "
          f"cosine(u_S, u_G2M) ~ orthonormalised")

    # DSM score-only drift (no prior, no residual).
    print(f"\n[pbmc] DSM score drift (no prior, no residual) — {args.drift_epochs} ep")
    t0 = time.monotonic()
    drift = DriftAdapter(dim=z.shape[1], config=DriftConfig(
        hidden=96, depth=3, n_epochs=args.drift_epochs,
        seed=args.seed, use_spectral_norm=False, batch_size=256,
        inject_velocity_prior=False, use_residual=False,
    )).fit(z, verbose=False)
    v = drift.velocity(z)
    J = drift.jacobian_autodiff(z)
    print(f"[pbmc]   drift + J {time.monotonic()-t0:.1f}s")
    print(f"[pbmc]   J symmetry check (median |J - J.T|_F / |J|_F): "
          f"{np.median([np.linalg.norm(Jc - Jc.T, 'fro') / (np.linalg.norm(Jc, 'fro') + 1e-9) for Jc in J]):.3e}")

    # Known-rotation projection.
    J_proj = _project_out_latent(J, U_cycle)
    print(f"[pbmc]   rotation-projected J ready  "
          f"(cycle U rank {U_cycle.shape[1]})")

    # Push into AnnData where the pipeline / opspace will find it.
    a.obsm["X_drift_v"] = v
    a.uns.setdefault("scjdo2", {})["J"] = J_proj
    a.uns["scjdo2"]["meta"] = {
        "clock_calibrated": False, "use_rep": "X_scVI",
        "time_key": "pseudotime", "jacobian_method": "autodiff",
        "known_rotation_projection": True,
    }
    a.uns["scjdo2"]["field_reason_summary"] = {"ok": 1.0, "autodiff": True}

    # Permutation null.
    print(f"\n[pbmc] permutation null (n={args.n_perm})")
    t0 = time.monotonic()
    perm_cfg = DriftConfig(
        hidden=96, depth=3, n_epochs=max(300, args.drift_epochs // 3),
        seed=args.seed + 101, use_spectral_norm=False, batch_size=256,
        inject_velocity_prior=False, use_residual=False,
    )
    perm_df = _permutation_null(z, np.zeros(a.n_obs, dtype=np.float32),
                                 args.n_perm, perm_cfg, seed=args.seed + 101)
    perm_df.to_csv(OUT / "permutation_null.csv", index=False)
    print(f"[pbmc]   perm null {time.monotonic()-t0:.1f}s")
    print(perm_df.to_string(index=False))

    # Phase C.
    print(f"\n[pbmc] Phase C (anchor max_live={args.anchor_max_live}, "
          f"n_anchors={args.n_anchors})")
    t0 = time.monotonic()
    fit_rate_and_gene_field(
        a, use_rep="X_scVI", n_epochs_rate=args.rate_epochs,
        k=args.k, cv_sample_cells=20, seed=args.seed,
        anchor_fit=True, n_anchors=args.n_anchors, anchor_top_k=10,
        anchor_max_live=args.anchor_max_live,
    )
    print(f"[pbmc]   Phase C {time.monotonic()-t0:.1f}s")

    print("\n[pbmc] opspace.compute")
    opspace.compute(a)

    # Per-type behaviour + lead_re.
    per_type = _per_type_report(a, RESTING_TYPES)
    per_type.to_csv(OUT / "behaviour_by_cluster.csv", index=False)
    print("\n[pbmc] BEHAVIOUR BY CLUSTER:")
    print(per_type.to_string(index=False))

    # Real vs null lead_re summary.
    real_lead_re = a.obs["lead_re"].to_numpy(dtype=np.float64)
    rr = real_lead_re[np.isfinite(real_lead_re)]
    perm_med = float(np.median(perm_df["median_lead_re"]))
    perm_frac_pos = float(np.median(perm_df["frac_pos"]))
    comp_rows = [{
        "arm": "real",
        "median_lead_re": float(np.median(rr)),
        "frac_pos": float((rr > 0).mean()),
        "frac_neg": float((rr < 0).mean()),
    }, {
        "arm": "perm_null_median",
        "median_lead_re": perm_med,
        "frac_pos": perm_frac_pos,
        "frac_neg": float(np.median(perm_df["frac_neg"])),
    }]
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(OUT / "real_vs_perm.csv", index=False)
    print("\n[pbmc] REAL vs PERMUTATION:")
    print(comp.to_string(index=False))

    # Per-holder-type dominant archetype + stiff/soft genes.
    arch = a.uns["scjdo2"].get("archetypes") or {}
    gene_names = np.asarray(a.var_names)
    stiff_soft_rows = []
    for cl in sorted(set(a.obs["cluster"])):
        k = _dominant_archetype_for_cluster(a, cl)
        if k is None or k not in arch:
            continue
        info = arch[k]
        evals = np.asarray(info["eigenvalues"])
        lead_mode = 0
        stiff_mode = int(np.argmin(evals.real))
        soft_mode = int(np.argmax(evals.real))
        if evals.size and (evals.real < 0).any():
            # Softest-decay mode: the least-negative Re among the
            # negative-Re eigenvalues — the escape direction out of a
            # basin. Falls back to the overall max-Re mode if none are
            # negative.
            neg = np.where(evals.real < 0)[0]
            soft_mode = int(neg[np.argmax(evals.real[neg])])
        for side in ("right", "left"):
            top_stiff = _mode_table(info, gene_names, stiff_mode, side)
            top_soft = _mode_table(info, gene_names, soft_mode, side)
            for row in top_stiff.to_dict("records"):
                row.update({"cluster": cl, "archetype": int(k),
                            "side": side, "mode": "stiffest",
                            "eig_re": float(evals.real[stiff_mode]),
                            "eig_im": float(evals.imag[stiff_mode])})
                stiff_soft_rows.append(row)
            for row in top_soft.to_dict("records"):
                row.update({"cluster": cl, "archetype": int(k),
                            "side": side, "mode": "softest",
                            "eig_re": float(evals.real[soft_mode]),
                            "eig_im": float(evals.imag[soft_mode])})
                stiff_soft_rows.append(row)
    stiff_soft = pd.DataFrame(stiff_soft_rows)
    stiff_soft.to_csv(OUT / "stiff_soft_by_cluster.csv", index=False)
    if not stiff_soft.empty:
        print("\n[pbmc] STIFFEST / SOFTEST per cluster (head):")
        print(stiff_soft.head(40).to_string(index=False))

    # Archetype × cluster soft-weight shares.
    W = a.obsm.get("archetype_weights")
    if W is not None:
        clusters = a.obs["cluster"].astype(str).to_numpy()
        cl_names = sorted(set(clusters))
        acw_rows = []
        for k in range(W.shape[1]):
            w = W[:, k]
            tot = float(w.sum()) or 1.0
            rec = {"archetype": int(k),
                   "behaviour": arch.get(k, {}).get("behaviour", ""),
                   "lead_re": float(arch.get(k, {}).get("eigenvalues", np.zeros(1))[0].real)
                              if arch.get(k) else 0.0,
                   "n_cells_hard": int((W.argmax(axis=1) == k).sum())}
            for cl in cl_names:
                m = clusters == cl
                rec[cl] = float(w[m].sum() / tot)
            acw_rows.append(rec)
        acw = pd.DataFrame(acw_rows)
        acw.to_csv(OUT / "archetype_by_cluster.csv", index=False)
        print("\n[pbmc] ARCHETYPE × CLUSTER:")
        print(acw.to_string(index=False))

    # Figures.
    _figures(a, OUT)

    # Gate decision.
    resting_rows = per_type[per_type["is_resting"]]
    holder_resting = int(resting_rows["holder"].sum())
    n_resting = int(resting_rows["n"].sum())
    gate = {
        "resting_types": list(RESTING_TYPES),
        "n_resting_cells": n_resting,
        "n_resting_holders": holder_resting,
        "frac_holder_resting": holder_resting / n_resting if n_resting else 0.0,
        "verdict": "pass" if (holder_resting / max(1, n_resting)) >= 0.5 else "fail",
    }
    (OUT / "gate.json").write_text(json.dumps(gate, indent=2))
    print("\n[pbmc] GATE:")
    print(json.dumps(gate, indent=2))

    print(f"\n[pbmc] outputs → {OUT}")
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    raise SystemExit(main())
