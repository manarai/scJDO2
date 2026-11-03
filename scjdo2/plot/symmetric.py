"""v0.5-basin-lock figures for symmetric-operator results.

All functions return a ``matplotlib.figure.Figure`` and, if ``out`` is
given, save ``out.pdf`` + ``out.png`` at 200 dpi. Each figure has a
6pt footer printing the tag + classification rule, and every
eigenvalue axis is labelled ``(relative)`` unless the AnnData's
``uns['scjdo2']['meta']['clock_calibrated']`` is True.

Functions (snapshot defaults, symmetric J under
``opspace.classify_basin_latent(rule='lead_eig')``):

* :func:`figure_spectrum_ladder`  — per-cluster centroid spectrum + oracle
* :func:`figure_soft_axis_map`    — embedding + per-cluster soft-axis arrow
* :func:`figure_ledger_map`       — embedding by provenance class
* :func:`figure_oracle_agreement` — learned vs analytic lead_re on pitchfork
* :func:`figure_basin_index`      — per-type basin_index distribution

The K562 figure moved to :mod:`scjdo2.plot.perturbation` and the aging
figures to :mod:`scjdo2.plot.aging`. The measured-flow figures are in
:mod:`scjdo2.plot.operator_space` behind ``measured_flow=True`` on
``figure_operator_space``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from matplotlib.figure import Figure

from ._save import save_figure
from .style import BEHAVIOUR_COLOURS, DIVERGING_CMAP, add_footer, style


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
    add_footer(fig, msg)


def _axis_tag_relative(adata: AnnData) -> str:
    """Return `` (relative)`` unless the drift is clock-calibrated."""
    meta = (adata.uns.get("scjdo2", {}) or {}).get("meta", {}) or {}
    return "" if bool(meta.get("clock_calibrated", False)) else "  (relative)"




# =====================================================================
# 1. spectrum_ladder
# =====================================================================


def figure_spectrum_ladder(adata: AnnData,
                            by: str = "cluster",
                            oracle: bool = True,
                            use_rep: Optional[str] = None,
                            n_cells_centroid: int = 30,
                            out: Optional[str | Path] = None,
                            groups: Optional[Sequence[str]] = None,
                            group_key: str = "age_group") -> Figure:
    """Per-cluster centroid-operator eigenvalue ladder + Gaussian oracle.

    When ``groups`` is given (e.g. ``("young", "old")``), draw two
    ladders side-by-side per cluster, one per group label in
    ``adata.obs[group_key]``. Oracle markers are per-group.

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
    spectra: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    with style():
        fig, ax = plt.subplots(figsize=(1.0 + 0.6 * len(clusters), 4.5))
        xs = np.arange(len(clusters))
        if groups is None:
            group_list = [None]
            offsets = [0.0]
            colors_group = {None: None}
        else:
            group_list = list(groups)
            # Spread two groups ±0.18 around the cluster x.
            if len(group_list) == 1:
                offsets = [0.0]
            else:
                offsets = np.linspace(-0.18, 0.18, len(group_list)).tolist()
            group_cmap = [BEHAVIOUR_COLOURS["holder"], BEHAVIOUR_COLOURS["fork"],
                           BEHAVIOUR_COLOURS["drift"]][: len(group_list)]
            colors_group = dict(zip(group_list, group_cmap))

        grp_arr = adata.obs[group_key].astype(str).to_numpy() \
            if (groups is not None and group_key in adata.obs) \
            else None
        for gi, g in enumerate(group_list):
            dx = offsets[gi]
            for i, cl in enumerate(clusters):
                m = cluster == cl
                if g is not None and grp_arr is not None:
                    m = m & (grp_arr == g)
                if m.sum() < 10:
                    continue
                mu = z[m].mean(axis=0)
                pool = np.flatnonzero(m)
                d_mu = np.linalg.norm(z[pool] - mu, axis=1)
                take = min(int(n_cells_centroid), len(pool))
                idx = pool[np.argpartition(d_mu, take - 1)[:take]]
                J_mu = J[idx].mean(axis=0)
                ev, V = np.linalg.eigh(0.5 * (J_mu + J_mu.T))
                if g is not None:
                    spectra[(str(cl), str(g))] = (ev, V)
                col = (colors_group.get(g)
                        or (BEHAVIOUR_COLOURS["fork"] if ev.max() > tol
                            else BEHAVIOUR_COLOURS["holder"]))
                ax.scatter(np.full(ev.size, i + dx), ev,
                            marker="_", s=90, color=col,
                            linewidths=1.3, zorder=3,
                            label=g if (gi >= 0 and i == 0 and g is not None)
                                    else None)
                # Oracle hollow markers for this group's spectrum.
                if oracle:
                    # Rebuild oracle per group (so Σ changes with the
                    # cluster × group subset).
                    # −Σ⁻¹ is only defined when the cluster has enough cells
                    # to estimate a full-rank covariance in z's dimension;
                    # below that the inverse explodes (e.g. 14 cells in 50-D).
                    if m.sum() >= max(10, 2 * z.shape[1]):
                        Sigma_g = (np.cov(z[m], rowvar=False, bias=False)
                                    + 1e-4 * np.eye(z.shape[1]))
                        ev_or_g = np.linalg.eigvalsh(-np.linalg.inv(Sigma_g))
                        ax.scatter(np.full(ev_or_g.size, i + dx + 0.07),
                                    ev_or_g, marker="o",
                                    facecolors="none", edgecolors="black",
                                    linewidths=0.8, s=18, zorder=4)
                    else:
                        ev_or_g = np.array([])
                else:
                    ev_or_g = np.array([])
                neg_learned = ev[ev < -tol]
                ratio = float("nan")
                if ev_or_g.size:
                    neg_or = ev_or_g[ev_or_g < 0]
                    if neg_or.size and neg_learned.size:
                        ratio = float(np.median(neg_learned) / np.median(neg_or))
                n_soft = int((np.abs(ev) < tol).sum())
                rows_annot.append({"cluster": cl, "group": g, "x": i + dx,
                                    "n_soft": n_soft, "ratio_neg": ratio,
                                    "oracle": bool(ev_or_g.size)})

        # Connect matched directions across age groups by maximum absolute
        # eigendirection cosine.  The thin lines encode matching only; the
        # vertical coordinates remain the stored centroid eigenvalues.
        if groups is not None and len(group_list) >= 2:
            g0, g1 = str(group_list[0]), str(group_list[1])
            for i, cl in enumerate(clusters):
                left = spectra.get((str(cl), g0))
                right = spectra.get((str(cl), g1))
                if left is None or right is None:
                    continue
                ev0, V0 = left
                ev1, V1 = right
                pairs: list[tuple[int, int]] = []
                used0: set[int] = set()
                used1: set[int] = set()
                for flat in np.argsort(np.abs(V0.T @ V1).ravel())[::-1]:
                    r, c = np.unravel_index(int(flat), (V0.shape[1], V1.shape[1]))
                    if r not in used0 and c not in used1:
                        pairs.append((int(r), int(c)))
                        used0.add(int(r)); used1.add(int(c))
                for r, c in pairs:
                    ax.plot([i + offsets[0], i + offsets[1]], [ev0[r], ev1[c]],
                            color="0.55", lw=0.35, alpha=0.7, zorder=1)
        # Tol shaded band.
        ax.axhspan(-tol, +tol, color="0.9", alpha=0.4, zorder=0,
                    label=f"soft band |λ| < {tol:.3f}")
        ax.axhline(0, color="k", lw=0.5)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - 0.10 * (hi - lo), hi)       # room for the n_soft / r labels
        ax.set_xticks(xs)
        ax.set_xticklabels(clusters, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel(f"λ (centroid){_axis_tag_relative(adata)}", fontsize=8)
        ax.set_title("spectrum ladder — centroid operator (hollow = Gaussian oracle −Σ⁻¹)",
                      fontsize=8)
        ax.legend(fontsize=6, loc="upper right")
        # n_soft and learned/oracle ratio under each ladder (inside the axes,
        # so they never collide with the rotated tick labels).
        import matplotlib.transforms as mtransforms
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        for r in rows_annot:
            r_txt = (f"r={r['ratio_neg']:.2f}" if np.isfinite(r["ratio_neg"])
                     else ("oracle n/a" if oracle and not r["oracle"] else "r=—"))
            ax.text(r["x"], 0.02, f"n_soft={r['n_soft']}\n{r_txt}", transform=trans,
                    ha="center", va="bottom", fontsize=5, color="0.3")
        _footer(fig, adata)
        save_figure(fig, out, png_dpi=200, tight=True)
    return fig


# =====================================================================
# 2. soft_axis_map
# =====================================================================


def figure_soft_axis_map(adata: AnnData,
                         by: str = "cluster",
                         use_rep: Optional[str] = None,
                         n_cells_centroid: int = 30,
                         out: Optional[str | Path] = None,
                         groups: Optional[Sequence[str]] = None,
                         group_key: str = "age_group",
                         displacement_arrow: bool = False) -> Figure:
    """Embedding with centroid soft-axis arrows.

    When ``groups`` supplies young and old labels and
    ``displacement_arrow=True``, the black arrow is the young centroid's
    leading eigendirection and the rust arrow is the stored old-minus-young
    centroid displacement.  No direction is fit or re-estimated here.
    """
    use_rep = use_rep or ("X_umap" if "X_umap" in adata.obsm
                           else "X_tsne" if "X_tsne" in adata.obsm
                           else "X_scVI" if "X_scVI" in adata.obsm
                           else "X_pca")
    if "op_basin_index" not in adata.obs:
        raise RuntimeError("op_basin_index missing — run classify_basin_latent")
    emb = np.asarray(adata.obsm[use_rep], dtype=np.float64)[:, :2]
    bi = adata.obs["op_basin_index"].to_numpy()
    cluster = adata.obs[by].astype(str).to_numpy()
    clusters = sorted(set(cluster))
    latent_key = (adata.uns.get("scjdo2", {}).get("meta", {}) or {}).get(
        "use_rep", "X_scVI" if "X_scVI" in adata.obsm else "X_pca",
    )
    z_lat = np.asarray(adata.obsm[latent_key], dtype=np.float64)
    J = np.asarray(adata.uns["scjdo2"].get("J_latent", adata.uns["scjdo2"]["J"]))
    age = (adata.obs[group_key].astype(str).to_numpy()
           if groups is not None and group_key in adata.obs else None)
    young_label = str(groups[0]) if groups else None
    old_label = str(groups[-1]) if groups and len(groups) > 1 else None
    cent_lat: dict[str, np.ndarray] = {}
    cent_emb: dict[str, np.ndarray] = {}
    young_masks: dict[str, np.ndarray] = {}
    old_cent_emb: dict[str, np.ndarray] = {}
    for cl in clusters:
        base = cluster == cl
        ref = base & (age == young_label) if age is not None and young_label else base
        if ref.sum() < 5:
            ref = base
        if ref.sum() < 5:
            continue
        cent_lat[cl] = z_lat[ref].mean(axis=0)
        cent_emb[cl] = emb[ref].mean(axis=0)
        young_masks[cl] = ref
        if age is not None and old_label:
            old = base & (age == old_label)
            if old.sum() >= 5:
                old_cent_emb[cl] = emb[old].mean(axis=0)

    def nearest(cl: str) -> str | None:
        others = [(float(np.linalg.norm(cent_lat[cl] - other)), name)
                  for name, other in cent_lat.items() if name != cl]
        return min(others)[1] if others else None

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
        for cl, mu_lat in cent_lat.items():
            mask = young_masks[cl]
            pool = np.flatnonzero(mask)
            d_mu = np.linalg.norm(z_lat[pool] - mu_lat, axis=1)
            take = min(int(n_cells_centroid), len(pool))
            idx = pool[np.argpartition(d_mu, take - 1)[:take]]
            J_mu = J[idx].mean(axis=0)
            ev, V = np.linalg.eigh(0.5 * (J_mu + J_mu.T))
            v_soft = V[:, int(np.argmax(ev))]
            try:
                W, *_ = np.linalg.lstsq(z_lat[idx] - z_lat[idx].mean(0),
                                         emb[idx] - emb[idx].mean(0), rcond=None)
            except np.linalg.LinAlgError:
                continue
            arrow = v_soft @ W
            emb_std = float(np.std(emb)) + 1e-6
            scale = 0.6 * (float(ev.max()) / max(abs(float(ev.max())), 1e-6))
            arrow = arrow / (np.linalg.norm(arrow) + 1e-12) * scale * emb_std
            mu = cent_emb[cl]
            nb = nearest(cl)
            ax.annotate("", xy=(mu[0] + arrow[0], mu[1] + arrow[1]), xytext=mu,
                        arrowprops=dict(arrowstyle="->", color="black", linewidth=1.2))
            if displacement_arrow and cl in old_cent_emb:
                delta = old_cent_emb[cl] - mu
                ax.annotate("", xy=(mu[0] + delta[0], mu[1] + delta[1]), xytext=mu,
                            arrowprops=dict(arrowstyle="->", color="#D85A30",
                                            linewidth=1.2, linestyle="--"))
            nb_name = nb if nb is not None else "—"
            ax.text(mu[0] + arrow[0] * 1.08, mu[1] + arrow[1] * 1.08,
                    f"{cl}\n→{nb_name}", fontsize=5, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                              edgecolor="none", alpha=0.7))
        if displacement_arrow:
            from matplotlib.lines import Line2D
            ax.legend(handles=[Line2D([], [], color="black", lw=1.2, label="u_young"),
                               Line2D([], [], color="#D85A30", lw=1.2, ls="--",
                                      label="old − young centroid")],
                      fontsize=6, loc="upper right")
        ax.set_xlabel(f"{use_rep}-1"); ax.set_ylabel(f"{use_rep}-2")
        ax.set_title("soft-axis map — black: soft axis per type (→ nearest type)"
                     + ("; rust: old − young centroid" if displacement_arrow else ""),
                     fontsize=9)
        _footer(fig, adata)
        save_figure(fig, out, png_dpi=200, tight=True)
    return fig

# =====================================================================
# 3. ledger_map
# =====================================================================


def figure_ledger_map(adata: AnnData,
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
        save_figure(fig, out, png_dpi=200, tight=True)
    return fig


# =====================================================================
# 4. oracle_agreement — pitchfork specific
# =====================================================================


def figure_oracle_agreement(adata: AnnData,
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
        save_figure(fig, out, png_dpi=200, tight=True)
    return fig


# =====================================================================
# 5. perturb_paired — K562
# =====================================================================


def figure_basin_index(adata: AnnData,
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
        save_figure(fig, out, png_dpi=200, tight=True)
    return fig
