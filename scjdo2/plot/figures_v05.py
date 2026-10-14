"""Six v0.5-basin-lock figure functions for symmetric-operator results.

All functions return a ``matplotlib.figure.Figure`` and, if ``out`` is
given, save ``out.pdf`` + ``out.png`` at 200 dpi. Each figure has a
6pt footer printing the tag + classification rule, and every
eigenvalue axis is labelled ``(relative)`` unless the AnnData's
``uns['scjdo2']['meta']['clock_calibrated']`` is True.

Functions (snapshot defaults, symmetric J under
``opspace.classify_basin_latent(rule='lead_eig')``):

* :func:`spectrum_ladder`     — per-cluster centroid spectrum + oracle
* :func:`soft_axis_map`       — embedding + per-cluster soft-axis arrow
* :func:`ledger_map`          — embedding by provenance class
* :func:`oracle_agreement`    — learned vs analytic lead_re on pitchfork
* :func:`perturb_paired`      — K562 operator vs DE-only + decomposition
* :func:`basin_index_by_type` — per-type basin_index distribution

The measured-flow figures stay available in ``scjdo2.plot.figures``
behind ``measured_flow=True`` on ``figure_operator_space``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib.figure import Figure

from .style import BEHAVIOUR_COLOURS, DIVERGING_CMAP, style


TAG = "v0.5-basin-lock"
RULE_DEFAULT = "lead_eig"


def _footer(fig: Figure, adata: Optional[AnnData] = None,
            rule: str = RULE_DEFAULT, extra: str = "") -> None:
    """6pt footer with tag + classification rule, bottom-right."""
    tag = TAG
    if adata is not None:
        meta = adata.uns.get("v05_meta") or {}
        tag = str(meta.get("tag", tag))
    msg = f"tag {tag}  ·  rule classify_basin_latent(rule={rule!r})"
    if extra:
        msg = msg + "  ·  " + extra
    fig.text(0.99, 0.005, msg, ha="right", va="bottom", fontsize=6, color="0.4")


def _axis_tag_relative(adata: AnnData) -> str:
    """Return `` (relative)`` unless the drift is clock-calibrated."""
    meta = (adata.uns.get("scjdo2", {}) or {}).get("meta", {}) or {}
    return "" if bool(meta.get("clock_calibrated", False)) else "  (relative)"


def _save(fig: Figure, out: Optional[str | Path]) -> None:
    if out is None:
        return
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")


# =====================================================================
# 1. spectrum_ladder
# =====================================================================


def spectrum_ladder(adata: AnnData,
                     by: str = "cluster",
                     oracle: bool = True,
                     use_rep: Optional[str] = None,
                     n_cells_centroid: int = 30,
                     out: Optional[str | Path] = None) -> Figure:
    """Per-cluster centroid-operator eigenvalue ladder + Gaussian oracle.

    Vertical ladders: for each cluster (``adata.obs[by]``) the eigenvalues
    of the per-cell latent ``J`` averaged over the ``n_cells_centroid``
    cells nearest the cluster centroid in ``obs[use_rep]``. Hollow
    markers = eigenvalues of ``−Σ⁻¹`` (Gaussian oracle on the cluster
    latent). Soft directions (``|λ| < oracle_tol``) are shaded. Annotates
    ``n_soft`` and ``learned/oracle`` neg-direction curvature ratio.
    """
    if by not in adata.obs:
        raise KeyError(f"obs[{by!r}] not found")
    use_rep = use_rep or (adata.uns.get("scjdo2", {}).get("meta", {}) or {}).get(
        "use_rep", "X_scVI" if "X_scVI" in adata.obsm else "X_pca",
    )
    if use_rep not in adata.obsm:
        raise KeyError(f"obsm[{use_rep!r}] not found")
    z = np.asarray(adata.obsm[use_rep], dtype=np.float64)
    J = np.asarray(adata.uns["scjdo2"]["J_latent"]
                    if "J_latent" in adata.uns.get("scjdo2", {})
                    else adata.uns["scjdo2"]["J"])
    cluster = adata.obs[by].astype(str).to_numpy()
    clusters = sorted(set(cluster))

    # Oracle tol = 0.1 × median |λ(−Σ⁻¹)| across clusters.
    oracle_leads = []
    oracle_specs = {}
    for cl in clusters:
        m = cluster == cl
        if m.sum() < 10:
            continue
        Sigma = np.cov(z[m], rowvar=False, bias=False) + 1e-4 * np.eye(z.shape[1])
        ev_or = np.linalg.eigvalsh(-np.linalg.inv(Sigma))
        oracle_specs[cl] = ev_or
        oracle_leads.append(abs(float(ev_or.max())))
    tol = 0.1 * float(np.median(oracle_leads)) if oracle_leads else 0.1

    rows_annot = []
    with style():
        fig, ax = plt.subplots(figsize=(1.0 + 0.6 * len(clusters), 4.5))
        xs = np.arange(len(clusters))
        for i, cl in enumerate(clusters):
            m = cluster == cl
            if m.sum() < 10:
                continue
            mu = z[m].mean(axis=0)
            d_mu = np.linalg.norm(z - mu, axis=1)
            idx = np.argpartition(d_mu, min(n_cells_centroid, len(z) - 1))[
                : n_cells_centroid
            ]
            J_mu = J[idx].mean(axis=0)
            ev = np.linalg.eigvalsh(0.5 * (J_mu + J_mu.T))
            # Soft shading in [-tol, +tol].
            ax.axvspan(i - 0.3, i + 0.3, ymin=0, ymax=1, color="none")
            # Learned ladder.
            ax.scatter(np.full(ev.size, i), ev, marker="_", s=90,
                        color=BEHAVIOUR_COLOURS["fork"] if ev.max() > tol
                        else BEHAVIOUR_COLOURS["holder"],
                        linewidths=1.3, zorder=3)
            # Oracle (hollow markers).
            if oracle and cl in oracle_specs:
                ax.scatter(np.full(oracle_specs[cl].size, i + 0.17),
                            oracle_specs[cl], marker="o",
                            facecolors="none", edgecolors="black",
                            linewidths=0.8, s=20, zorder=4)
            # Curvature ratio on negative directions.
            neg_learned = ev[ev < -tol]
            ratio = float("nan")
            if cl in oracle_specs:
                neg_or = oracle_specs[cl][oracle_specs[cl] < 0]
                if neg_or.size and neg_learned.size:
                    ratio = float(np.median(neg_learned) / np.median(neg_or))
            n_soft = int((np.abs(ev) < tol).sum())
            rows_annot.append({"cluster": cl, "n_soft": n_soft,
                                "ratio_neg": ratio})

        # Tol shaded band.
        ax.axhspan(-tol, +tol, color="0.9", alpha=0.4, zorder=0,
                    label=f"soft band |λ| < {tol:.3f}")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(clusters, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel(f"λ (centroid){_axis_tag_relative(adata)}", fontsize=8)
        ax.set_title("spectrum ladder — centroid operator (hollow = Gaussian oracle −Σ⁻¹)",
                      fontsize=8)
        ax.legend(fontsize=6, loc="upper right")
        # Annotate n_soft + ratio in a small table below.
        annot = "  ".join(
            f"{r['cluster'][:6]}: n_soft={r['n_soft']}  r={r['ratio_neg']:+.2f}"
            for r in rows_annot
        )
        fig.text(0.02, 0.015, annot, fontsize=5, color="0.3")
        _footer(fig, adata)
        _save(fig, out)
    return fig


# =====================================================================
# 2. soft_axis_map
# =====================================================================


def soft_axis_map(adata: AnnData,
                   by: str = "cluster",
                   use_rep: Optional[str] = None,
                   n_cells_centroid: int = 30,
                   out: Optional[str | Path] = None) -> Figure:
    """Embedding scatter coloured by ``basin_index`` with per-cluster
    soft-axis arrow (most-positive eigenvector of the centroid
    operator, projected into the 2-D embedding). Length ∝ eigenvalue
    magnitude; annotation = cosine with the direction to the nearest
    other cluster centroid, named.
    """
    use_rep = use_rep or ("X_umap" if "X_umap" in adata.obsm
                           else "X_tsne" if "X_tsne" in adata.obsm
                           else "X_scVI" if "X_scVI" in adata.obsm
                           else "X_pca")
    if "op_basin_index" not in adata.obs:
        raise RuntimeError("op_basin_index missing — run classify_basin_latent")
    emb = np.asarray(adata.obsm[use_rep])[:, :2]
    bi = adata.obs["op_basin_index"].to_numpy()
    cluster = adata.obs[by].astype(str).to_numpy()
    clusters = sorted(set(cluster))

    # Compute per-cluster centroid in latent + 2D embedding.
    latent_key = (adata.uns.get("scjdo2", {}).get("meta", {}) or {}).get(
        "use_rep", "X_scVI" if "X_scVI" in adata.obsm else "X_pca",
    )
    z_lat = np.asarray(adata.obsm[latent_key], dtype=np.float64)
    J = np.asarray(adata.uns["scjdo2"].get("J_latent",
                     adata.uns["scjdo2"]["J"]))

    cl_lat_centroids = {}
    cl_emb_centroids = {}
    for cl in clusters:
        m = cluster == cl
        if m.sum() < 5:
            continue
        cl_lat_centroids[cl] = z_lat[m].mean(axis=0)
        cl_emb_centroids[cl] = emb[m].mean(axis=0)

    def _nearest(cl):
        best = None; best_d = np.inf
        for other, mu2 in cl_lat_centroids.items():
            if other == cl:
                continue
            d = float(np.linalg.norm(cl_lat_centroids[cl] - mu2))
            if d < best_d:
                best, best_d = other, d
        return best, best_d

    # Build Panel A scatter.
    with style():
        fig, ax = plt.subplots(figsize=(7, 6))
        import matplotlib.colors as mcolors
        finite = bi[np.isfinite(bi)]
        vmax = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
        norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
        sc1 = ax.scatter(emb[:, 0], emb[:, 1], c=bi, cmap=DIVERGING_CMAP,
                          norm=norm, s=5, alpha=0.5, edgecolors="none")
        plt.colorbar(sc1, ax=ax, label=f"basin_index{_axis_tag_relative(adata)}",
                      shrink=0.55, pad=0.02)

        # For each cluster: arrow along most-positive eigvec, length ∝ λ.
        for cl, mu_lat in cl_lat_centroids.items():
            m = cluster == cl
            d_mu = np.linalg.norm(z_lat - mu_lat, axis=1)
            idx = np.argpartition(d_mu, min(n_cells_centroid, len(z_lat) - 1))[
                : n_cells_centroid
            ]
            J_mu = J[idx].mean(axis=0)
            ev, V = np.linalg.eigh(0.5 * (J_mu + J_mu.T))
            lead = int(np.argmax(ev))
            v_pos = V[:, lead]                                      # (d_lat,)
            # Project v_pos into the 2-D embedding via local projection:
            # linear fit of emb[kNN] on z_lat[kNN].
            z_c = z_lat[idx]
            e_c = emb[idx]
            # emb ≈ z_c @ W + b   →  W ∈ R^{d_lat × 2}
            try:
                W, *_ = np.linalg.lstsq(z_c - z_c.mean(0),
                                          e_c - e_c.mean(0), rcond=None)
            except np.linalg.LinAlgError:
                continue
            arrow_2d = v_pos @ W                                     # (2,)
            # Nearest cluster centroid in latent.
            nb, d_nb = _nearest(cl)
            cos_val = float("nan"); nb_name = "—"
            if nb is not None:
                u = cl_lat_centroids[nb] - mu_lat
                u = u / (np.linalg.norm(u) + 1e-12)
                cos_val = float(np.dot(v_pos, u))
                nb_name = nb
            # Length ∝ positive eigenvalue (clip for display).
            scale = 0.6 * (float(ev[lead]) / max(abs(float(ev.max())), 1e-6))
            # In embedding units: scale by local embedding std.
            emb_std = float(np.std(emb)) + 1e-6
            arrow_disp = arrow_2d / (np.linalg.norm(arrow_2d) + 1e-12) * scale * emb_std
            mu_e = cl_emb_centroids[cl]
            ax.annotate("", xy=(mu_e[0] + arrow_disp[0],
                                 mu_e[1] + arrow_disp[1]),
                         xytext=mu_e,
                         arrowprops=dict(arrowstyle="->",
                                          color="black", linewidth=1.2))
            ax.text(mu_e[0] + arrow_disp[0] * 1.08,
                     mu_e[1] + arrow_disp[1] * 1.08,
                     f"{cl}\n→{nb_name}  |cos|={abs(cos_val):.2f}",
                     fontsize=5, ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.1",
                                facecolor="white", edgecolor="none",
                                alpha=0.7))
        ax.set_xlabel(f"{use_rep}-1")
        ax.set_ylabel(f"{use_rep}-2")
        ax.set_title("soft-axis map — arrow = most-positive eigenvector of centroid J",
                      fontsize=9)
        _footer(fig, adata)
        _save(fig, out)
    return fig


# =====================================================================
# 3. ledger_map
# =====================================================================


def ledger_map(adata: AnnData,
                use_rep: Optional[str] = None,
                out: Optional[str | Path] = None) -> Figure:
    """Embedding coloured by provenance class per cell.

    Classes (snapshot default):
      * basin        — op_behaviour_basin == "holder" (basin, measured)
      * saddle       — op_behaviour_basin == "fork"   (density ridge)
      * drift        — op_behaviour_basin == "drift"  (|lead_re| < tol)
      * not_determinable — op_behaviour_basin missing / NaN

    Legend reports per-class counts.
    """
    use_rep = use_rep or ("X_umap" if "X_umap" in adata.obsm
                           else "X_tsne" if "X_tsne" in adata.obsm
                           else "X_scVI" if "X_scVI" in adata.obsm
                           else "X_pca")
    emb = np.asarray(adata.obsm[use_rep])[:, :2]
    beh = adata.obs["op_behaviour_basin"].astype(str).to_numpy() \
        if "op_behaviour_basin" in adata.obs \
        else np.full(adata.n_obs, "not_determinable", dtype=object)
    class_map = {"holder": "basin (measured)",
                  "fork":   "saddle region",
                  "drift":  "drift (near-flat)",
                  "":       "not_determinable"}
    labels = np.array([class_map.get(b, b) for b in beh], dtype=object)
    palette = {
        "basin (measured)": BEHAVIOUR_COLOURS["holder"],
        "saddle region":    BEHAVIOUR_COLOURS["fork"],
        "drift (near-flat)": BEHAVIOUR_COLOURS["drift"],
        "not_determinable": "#444444",
    }
    with style():
        fig, ax = plt.subplots(figsize=(7, 6))
        for cat, col in palette.items():
            m = labels == cat
            if not m.any():
                continue
            ax.scatter(emb[m, 0], emb[m, 1], s=5, alpha=0.5, color=col,
                        label=f"{cat} ({int(m.sum())})",
                        edgecolors="none")
        ax.set_xlabel(f"{use_rep}-1"); ax.set_ylabel(f"{use_rep}-2")
        ax.set_title("ledger_map — provenance class per cell", fontsize=9)
        ax.legend(fontsize=7, loc="upper right", markerscale=2.0)
        _footer(fig, adata)
        _save(fig, out)
    return fig


# =====================================================================
# 4. oracle_agreement — pitchfork specific
# =====================================================================


def oracle_agreement(adata: AnnData,
                      tau_crit: float = 0.3,
                      window: float = 0.05,
                      out: Optional[str | Path] = None) -> Figure:
    """Learned vs analytic lead_re per cell on the equilibrated
    pitchfork; the ``±window`` saddle band marked, holder region
    highlighted; AUC + Spearman in the corner.
    """
    # Analytic lead_re per cell.
    try:
        from scjdo2.data import analytic_jacobian
    except Exception as err:
        raise RuntimeError(
            "oracle_agreement needs scjdo2.data.analytic_jacobian",
        ) from err
    if "X_true" not in adata.obsm:
        raise RuntimeError("obsm['X_true'] missing — equilibrated pitchfork only")
    z_true = np.asarray(adata.obsm["X_true"], dtype=np.float64)
    alpha = adata.obs["alpha"].to_numpy(dtype=np.float64) \
        if "alpha" in adata.obs else None
    if alpha is None:
        meta = adata.uns.get("pitchfork", {})
        a_min, a_max = float(meta["alpha_min"]), float(meta["alpha_max"])
        alpha = a_min + adata.obs["pseudotime"].to_numpy(float) * (a_max - a_min)
    J_true = analytic_jacobian(z_true, alpha)
    lead_true = np.array([float(np.max(np.linalg.eigvals(J_true[c]).real))
                           for c in range(J_true.shape[0])])
    # Learned.
    if "lead_re" in adata.obs:
        lead_learn = adata.obs["lead_re"].to_numpy()
    else:
        lead_learn = -adata.obs["op_basin_index"].to_numpy()

    pt = adata.obs["pseudotime"].to_numpy(float)
    in_window = np.abs(pt - tau_crit) <= window
    pos_mask = lead_true < -0.1 * np.nanstd(lead_true)  # true holders
    valid = np.isfinite(lead_true) & np.isfinite(lead_learn)
    from scipy.stats import spearmanr
    rho = float(spearmanr(lead_true[valid], lead_learn[valid]).correlation)

    try:
        from sklearn.metrics import roc_auc_score
        y = pos_mask[valid & ~in_window].astype(int)
        y_in = in_window[valid & ~in_window]
        # Positive = true-holder outside window; negative = saddle window cells.
        y_full = np.concatenate([pos_mask[valid & ~in_window],
                                   np.zeros(int((valid & in_window).sum()), dtype=bool)])
        s_full = np.concatenate([-lead_learn[valid & ~in_window],
                                   -lead_learn[valid & in_window]])
        auc = float(roc_auc_score(y_full.astype(int), s_full)) \
            if y_full.sum() and y_full.sum() < y_full.size else float("nan")
    except Exception:
        auc = float("nan")

    with style():
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(lead_true[valid], lead_learn[valid], s=5, alpha=0.4,
                    color="0.3")
        # Highlight saddle window.
        in_val = valid & in_window
        ax.scatter(lead_true[in_val], lead_learn[in_val], s=12, alpha=0.7,
                    color=BEHAVIOUR_COLOURS["fork"],
                    label=f"|pt − τ_crit| ≤ {window}")
        # Holder region shade.
        y_min, y_max = np.nanpercentile(lead_learn[valid], [2, 98])
        x_min, x_max = np.nanpercentile(lead_true[valid], [2, 98])
        ax.axvspan(x_min, 0, color=BEHAVIOUR_COLOURS["holder"], alpha=0.08)
        ax.axhline(0, color="k", lw=0.4, ls="--")
        ax.axvline(0, color="k", lw=0.4, ls="--")
        ax.set_xlabel(f"analytic lead_re{_axis_tag_relative(adata)}")
        ax.set_ylabel(f"learned lead_re{_axis_tag_relative(adata)}")
        ax.set_title("oracle agreement — learned vs analytic (equilibrated pitchfork)",
                      fontsize=9)
        ax.text(0.03, 0.97, f"Spearman = {rho:+.3f}\nAUC (holder vs saddle) = {auc:+.3f}",
                 transform=ax.transAxes, va="top", ha="left", fontsize=7,
                 bbox=dict(boxstyle="round,pad=0.3",
                           facecolor="white", edgecolor="0.5"))
        ax.legend(fontsize=7, loc="lower right")
        _footer(fig, adata)
        _save(fig, out)
    return fig


# =====================================================================
# 5. perturb_paired — K562
# =====================================================================


def perturb_paired(result: dict | pd.DataFrame | Path | str,
                    out: Optional[str | Path] = None) -> Figure:
    """Two-panel K562 figure.

    ``result`` can be a per-knockdown DataFrame OR a path to the CSV OR
    a dict pointing at both the per-KD and generic-control tables.

    Left panel: operator cos vs DE-only cos as paired points per KD +
    y=x diagonal + win fraction annotation.
    Right panel: generic-response decomposition — mean cosine for
    full / generic-component / residual, across operator, DE-only
    and random.
    """
    if isinstance(result, (str, Path)):
        pk = pd.read_csv(result)
        gen = None
    elif isinstance(result, pd.DataFrame):
        pk = result
        gen = None
    else:
        pk = result.get("per_kd")
        gen = result.get("generic_table")
        if isinstance(pk, (str, Path)):
            pk = pd.read_csv(pk)
        if isinstance(gen, (str, Path)):
            gen = pd.read_csv(gen)

    valid = pk.dropna(subset=["cos_scjdo2", "cos_de_only"])
    win_frac = float((valid["cos_scjdo2"] > valid["cos_de_only"]).mean())

    with style():
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.5))
        # Left: paired scatter.
        ax_l.scatter(valid["cos_de_only"], valid["cos_scjdo2"], s=6, alpha=0.4,
                      color="0.3")
        lim = max(float(valid[["cos_de_only", "cos_scjdo2"]].quantile(0.99).max()),
                   abs(float(valid[["cos_de_only", "cos_scjdo2"]].quantile(0.01).min())))
        ax_l.plot([-lim, lim], [-lim, lim], "k--", lw=0.5)
        ax_l.axhline(0, color="k", lw=0.3); ax_l.axvline(0, color="k", lw=0.3)
        ax_l.set_xlim(-lim, lim); ax_l.set_ylim(-lim, lim)
        ax_l.set_xlabel("DE-only  cos(b_g, δx_obs)")
        ax_l.set_ylabel("operator  cos(δx_pred, δx_obs)")
        ax_l.set_title(f"K562 per-KD paired  —  frac(op > DE) = {win_frac:.2%}  "
                        f"(n = {len(valid)})", fontsize=9)
        ax_l.set_aspect("equal", adjustable="box")

        # Right: generic decomposition bars.
        bars = []
        if gen is not None and not gen.empty:
            # Expect rows for arm in {full_space_all, residual_all}.
            try:
                full = gen[gen["arm"] == "full_space_all"].iloc[0]
                res = gen[gen["arm"] == "residual_all"].iloc[0]
                bars.append(("full", "operator", float(full["op_mean"])))
                bars.append(("full", "DE-only",  float(full["de_mean"])))
                bars.append(("residual", "operator", float(res["op_mean"])))
                bars.append(("residual", "DE-only",  float(res["de_mean"])))
            except (IndexError, KeyError):
                pass
        if bars:
            labels_x = ["full\noperator", "full\nDE-only",
                        "residual\noperator", "residual\nDE-only"]
            vals = [b[2] for b in bars]
            colors = ["#d55e00", "#0072b2", "#d55e00", "#0072b2"]
            ax_r.bar(range(len(bars)), vals, color=colors)
            ax_r.set_xticks(range(len(bars)))
            ax_r.set_xticklabels(labels_x, fontsize=7)
            ax_r.axhline(0, color="k", lw=0.5)
            ax_r.set_ylabel("mean cosine")
            ax_r.set_title("generic-response decomposition  "
                            "(full → residual = KD-specific)",
                            fontsize=9)
            for i, v in enumerate(vals):
                ax_r.text(i, v, f"{v:+.3f}", ha="center",
                           va="bottom" if v >= 0 else "top", fontsize=7)
        else:
            ax_r.text(0.5, 0.5,
                      "generic_control_comparison.csv not provided",
                      ha="center", va="center", transform=ax_r.transAxes,
                      fontsize=8, color="0.5")
            ax_r.set_axis_off()
        _footer(fig, None,
                 extra="K562 Perturb-seq  ·  controls per V2_PLAN §9f-g")
        _save(fig, out)
    return fig


# =====================================================================
# 6. basin_index_by_type
# =====================================================================


def basin_index_by_type(adata: AnnData,
                         by: str = "cluster",
                         resting_types: Optional[Sequence[str]] = None,
                         out: Optional[str | Path] = None) -> Figure:
    """Ordered per-type ``basin_index`` distribution (strip) with the
    locked-rule basin fraction per type and the dataset resting-mean
    (47.2% on PBMC) line.
    """
    if "op_basin_index" not in adata.obs:
        raise RuntimeError("op_basin_index missing")
    bi = adata.obs["op_basin_index"].to_numpy()
    beh = adata.obs["op_behaviour_basin"].astype(str).to_numpy() \
        if "op_behaviour_basin" in adata.obs \
        else np.full(adata.n_obs, "", dtype=object)
    cluster = adata.obs[by].astype(str).to_numpy()
    clusters = sorted(set(cluster))
    rows = []
    for cl in clusters:
        m = cluster == cl
        n = int(m.sum())
        hold = int((beh[m] == "holder").sum())
        bi_cl = bi[m][np.isfinite(bi[m])]
        rows.append({"cluster": cl, "n": n, "frac_holder": hold / n if n else 0.0,
                      "bi_median": float(np.median(bi_cl)) if bi_cl.size else 0.0,
                      "data": bi_cl,
                      "is_resting": bool(resting_types and cl in set(resting_types))})
    rows = sorted(rows, key=lambda r: r["bi_median"], reverse=True)
    order = [r["cluster"] for r in rows]
    # Dataset-level resting basin fraction.
    if resting_types:
        resting_set = set(resting_types)
        rst = [r for r in rows if r["cluster"] in resting_set]
        n_rest = sum(r["n"] for r in rst)
        hold_rest = sum(int(r["frac_holder"] * r["n"]) for r in rst)
        line_frac = hold_rest / n_rest if n_rest else 0.0
    else:
        hold_all = sum(int(r["frac_holder"] * r["n"]) for r in rows)
        n_all = sum(r["n"] for r in rows)
        line_frac = hold_all / n_all if n_all else 0.0

    with style():
        fig, (ax_v, ax_b) = plt.subplots(2, 1, figsize=(1.0 + 0.65 * len(order), 6.0),
                                           sharex=True,
                                           gridspec_kw=dict(height_ratios=[2.5, 1]))
        # Strip plot with jitter.
        rng = np.random.default_rng(0)
        for i, r in enumerate(rows):
            x = rng.uniform(-0.3, 0.3, size=r["data"].size) + i
            col = BEHAVIOUR_COLOURS["holder"] if r["is_resting"] else "0.5"
            ax_v.scatter(x, r["data"], s=3, alpha=0.3, color=col)
            ax_v.plot([i - 0.35, i + 0.35],
                       [r["bi_median"], r["bi_median"]],
                       color="black", lw=1.2)
        ax_v.axhline(0, color="k", lw=0.4, ls="--")
        ax_v.set_ylabel(f"basin_index  = −lead_re{_axis_tag_relative(adata)}")
        ax_v.set_title("basin_index per cluster (median = black bar)", fontsize=9)
        # Per-cluster basin fraction bar.
        fracs = [r["frac_holder"] for r in rows]
        colors = [BEHAVIOUR_COLOURS["holder"] if r["is_resting"] else "0.5"
                   for r in rows]
        ax_b.bar(range(len(rows)), fracs, color=colors)
        ax_b.axhline(line_frac, color="black", lw=1.0, ls="--",
                      label=f"resting basin fraction = {line_frac:.3f}"
                            if resting_types else
                            f"dataset basin fraction = {line_frac:.3f}")
        ax_b.set_xticks(range(len(order)))
        ax_b.set_xticklabels(order, rotation=35, ha="right", fontsize=7)
        ax_b.set_ylabel("frac holder")
        ax_b.set_ylim(0, 1)
        ax_b.legend(fontsize=7, loc="upper right")
        _footer(fig, adata)
        _save(fig, out)
    return fig
