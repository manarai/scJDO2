"""Driver / responder influence network, one row per archetype.

Reads :func:`scjdo2.influence.archetype_influence` output. Each row has
three panels:

(a) **Driver / responder map** — in-influence (x) vs out-influence (y),
    log scale, measurable genes only. Dotted line: influential cut
    (``out + in``); genes below it are passengers. Dashed diagonals: the
    polarity cuts — above the upper one = driver, below the lower one =
    responder, between = hub.
(b) **Influence over time** — each leading driver's share of the total
    out-influence (solid; top two hubs dashed). Shares remove the common
    e^{λt} growth so rank changes over time are visible. ``t*`` marked.
(c) **Influence network at t*** — nodes laid out in columns by role
    (driver | hub | responder / other). Arrow ``j → i`` is ``M[i, j](t*)``:
    colour = sign, width = magnitude. With a reference network, solid
    = also in the reference, dashed = dynamics only, faint grey dotted =
    reference only (between drawn nodes). Node size = out-influence;
    a white node with a dashed outline is ``unmeasured`` (zero Fisher or
    off-tangent): its influence is not measured, not zero.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from ..influence import ROLE_COLOURS, annotate_reference, archetype_influence, influence_edges
from .style import OPSPACE_FIG_WIDTH_IN, relative_label, style

_POS_EDGE = "#B2182B"
_NEG_EDGE = "#2166AC"
_COLUMN = {"driver": 0.0, "hub": 1.0}


def _map_panel(ax, tbl: pd.DataFrame, thr: dict, n_label: int = 3) -> None:
    m = tbl[tbl["role"] != "unmeasured"]
    m = m[(m["out_influence"] > 0) & (m["in_influence"] > 0)]
    for role in ("passenger", "hub", "responder", "driver"):
        s = m[m["role"] == role]
        ax.scatter(s["in_influence"], s["out_influence"], s=5 if role == "passenger" else 14,
                   c=ROLE_COLOURS[role], edgecolors="0.25" if role != "passenger" else "none",
                   linewidths=0.3, alpha=0.45 if role == "passenger" else 0.9,
                   zorder=2 if role == "passenger" else 3)
    if not len(m):
        ax.text(0.5, 0.5, "no gene with both scores > 0", ha="center", va="center",
                fontsize=6.5, transform=ax.transAxes)
        return
    ax.set_xscale("log")
    ax.set_yscale("log")
    if len(m):
        lo = 0.7 * float(min(m["in_influence"].min(), m["out_influence"].min()))
        hi = 1.4 * float(max(m["in_influence"].max(), m["out_influence"].max()))
        x = np.geomspace(lo, hi, 50)
        # Polarity boundaries: out = in · 2^p  (diagonals in log-log).
        for key in ("polarity_lo", "polarity_hi"):
            p = thr.get(key, np.nan)
            if np.isfinite(p):
                ax.plot(x, x * 2.0 ** p, color="0.45", lw=0.5, ls="--", zorder=1)
        # Influential cut: out + in = total.
        tot = thr.get("total", np.nan)
        if np.isfinite(tot) and tot > 0:
            xi = x[x < tot]
            ax.plot(xi, tot - xi, color="0.45", lw=0.5, ls=":", zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    labelled = pd.concat([
        m[m["role"] == "driver"].nlargest(n_label, "polarity"),
        m[m["role"] == "responder"].nsmallest(n_label, "polarity"),
        m[m["role"] == "hub"].nlargest(1, "out_influence"),
    ])
    for i, (_, r) in enumerate(labelled.iterrows()):
        dx = -3 if r["role"] == "driver" else 3
        ax.annotate(r["gene"], (r["in_influence"], r["out_influence"]), fontsize=5.5,
                    xytext=(dx, 2 + 4 * (i % 2)), textcoords="offset points",
                    ha="right" if dx < 0 else "left", zorder=4)
    n_un = int((tbl["role"] == "unmeasured").sum())
    ax.text(0.98, 0.02, f"{n_un} unmeasured\n(not shown)", transform=ax.transAxes,
            fontsize=5.5, va="bottom", ha="right", color="0.35")
    ax.set_xlabel("in-influence (moved by others)")
    ax.set_ylabel("out-influence (moves others)")


def _curve_panel(ax, info: dict, genes: np.ndarray, adata: AnnData, max_genes: int = 6) -> None:
    """Share of total out-influence over time for the leading drivers.

    Shares remove the common e^{λt} growth, so what is left is how the
    ranking among genes changes with time.
    """
    cur = info["influence_curve"]
    t = np.asarray(cur["t"])
    out = np.asarray(cur["out"])
    share = out / np.clip(out.sum(axis=1, keepdims=True), 1e-300, None)
    tbl = info["influence"]
    idx = {g: i for i, g in enumerate(genes)}
    drivers = [idx[g] for g in tbl[tbl["role"] == "driver"]["gene"].head(max_genes)]
    hubs = [idx[g] for g in tbl[tbl["role"] == "hub"]["gene"].head(2)]
    cmap = plt.get_cmap("tab10")
    for c, j in enumerate(drivers + hubs):
        ax.plot(t, 100 * share[:, j], lw=0.9, color=cmap(c % 10),
                ls="-" if j in drivers else "--", label=str(genes[j]))
    t_star = float(info["influence_t"])
    ax.axvline(t_star, color="0.3", lw=0.6, ls=":")
    ax.text(t_star, 1.0, " t*", transform=ax.get_xaxis_transform(), fontsize=6, va="top")
    ax.set_xscale("log")
    ax.set_xlabel(relative_label(adata, "time t"))
    ax.set_ylabel("share of out-influence (%)")
    ax.legend(fontsize=5, frameon=False, ncol=2, loc="best", handlelength=1.4)


def _network_panel(ax, tbl: pd.DataFrame, edges: pd.DataFrame,
                   ref_network: Optional[pd.DataFrame]) -> None:
    ax.set_axis_off()
    if edges.empty:
        ax.text(0.5, 0.5, "no edges above threshold", ha="center", va="center",
                fontsize=7, transform=ax.transAxes)
        return
    role = dict(zip(tbl["gene"], tbl["role"]))
    out_inf = dict(zip(tbl["gene"], tbl["out_influence"]))
    nodes = list(dict.fromkeys(list(edges["source"]) + list(edges["target"])))
    cols: dict[float, list[str]] = {0.0: [], 1.0: [], 2.0: []}
    for g in nodes:
        cols[_COLUMN.get(role.get(g, "passenger"), 2.0)].append(g)
    pos: dict[str, tuple[float, float]] = {}
    for x, gs in cols.items():
        gs.sort(key=lambda g: -out_inf.get(g, 0.0))
        n = len(gs)
        for i, g in enumerate(gs):
            pos[g] = (x, 1.0 - (i + 0.5) / max(n, 1))

    vmax = float(edges["abs_weight"].max()) or 1.0
    has_ref = ref_network is not None and not ref_network.empty
    e = annotate_reference(edges, ref_network)
    for _, r in e.iterrows():
        (x0, y0), (x1, y1) = pos[r["source"]], pos[r["target"]]
        rad = 0.45 if abs(x0 - x1) < 1e-9 else 0.12
        ls = "-" if (r["in_reference"] or not has_ref) else (0, (3, 2))
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>" if r["sign"] >= 0 else "-[", mutation_scale=6,
            lw=0.4 + 2.2 * r["abs_weight"] / vmax, ls=ls,
            color=_POS_EDGE if r["sign"] >= 0 else _NEG_EDGE,
            alpha=0.85, shrinkA=6, shrinkB=6, zorder=2,
        ))
    if has_ref:
        drawn = set(zip(e["source"].str.upper(), e["target"].str.upper()))
        up = {g.upper(): g for g in nodes}
        ref = ref_network[ref_network["source"].astype(str).str.upper().isin(up)
                          & ref_network["target"].astype(str).str.upper().isin(up)]
        for s, t in zip(ref["source"].astype(str).str.upper(), ref["target"].astype(str).str.upper()):
            if (s, t) in drawn or s == t:
                continue
            (x0, y0), (x1, y1) = pos[up[s]], pos[up[t]]
            ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), connectionstyle="arc3,rad=-0.2",
                                         arrowstyle="-|>", mutation_scale=5, lw=0.4, ls=":",
                                         color="0.7", shrinkA=6, shrinkB=6, zorder=1))

    omax = max([out_inf.get(g, 0.0) for g in nodes] + [1e-12])
    for g in nodes:
        x, y = pos[g]
        rl = role.get(g, "passenger")
        size = 25 + 160 * np.sqrt(out_inf.get(g, 0.0) / omax)
        ax.scatter([x], [y], s=size, c=ROLE_COLOURS[rl], edgecolors="0.2",
                   linewidths=0.6, linestyle="--" if rl == "unmeasured" else "-", zorder=3)
        ax.text(x + 0.09, y, g, fontsize=6, va="center", zorder=4)
    for x, lab in ((0.0, "drivers"), (1.0, "hubs"), (2.0, "responders / other")):
        ax.text(x, 1.04, lab, ha="center", fontsize=6.5, color="0.3")
    ax.set_xlim(-0.3, 2.6)
    ax.set_ylim(-0.05, 1.1)


def _legend_handles(has_ref: bool) -> list:
    h = [Line2D([], [], ls="", marker="o", ms=5, mfc=ROLE_COLOURS[r], mec="0.2", label=r)
         for r in ("driver", "hub", "responder", "passenger", "unmeasured")]
    h += [Line2D([], [], color=_POS_EDGE, lw=1.2, label="activating (+)"),
          Line2D([], [], color=_NEG_EDGE, lw=1.2, label="repressing (−)")]
    if has_ref:
        h += [Line2D([], [], color="0.2", lw=1.0, label="dynamics + reference"),
              Line2D([], [], color="0.2", lw=1.0, ls=(0, (3, 2)), label="dynamics only"),
              Line2D([], [], color="0.7", lw=0.8, ls=":", label="reference only")]
    return h


def figure_influence_network(
    adata: AnnData,
    archetype_ids: Optional[Sequence[int]] = None,
    t: float | str = "auto",
    ref_network: Optional[pd.DataFrame] = None,
    n_sources: int = 5,
    n_targets: int = 4,
    quantile: float = 0.9,
    save: Optional[str | Path] = None,
) -> Figure:
    """Driver / responder map, influence-over-time and network per archetype.

    Runs :func:`scjdo2.influence.archetype_influence` if an archetype
    has no ``influence`` table yet (or if ``t`` is not ``"auto"``).
    ``ref_network``: optional DataFrame with ``source`` / ``target``
    (e.g. v1 ``scjdo.tl.load_network``) for the edge-support styling.
    """
    arch = adata.uns["scjdo2"]["archetypes"]
    ids = sorted(arch) if archetype_ids is None else [int(a) for a in archetype_ids]
    need = [a for a in ids if "influence" not in arch[a]] if t == "auto" else ids
    if need:
        archetype_influence(adata, t=t, quantile=quantile, archetype_ids=need)
    genes = np.asarray(adata.var_names)
    has_ref = ref_network is not None and not ref_network.empty

    with style():
        fig, axes = plt.subplots(len(ids), 3, squeeze=False,
                                 figsize=(OPSPACE_FIG_WIDTH_IN, 2.35 * len(ids) + 0.5),
                                 gridspec_kw={"width_ratios": [1.0, 1.0, 1.35]})
        for row, ak in enumerate(ids):
            info = arch[ak]
            tbl = info["influence"]
            edges = influence_edges(adata, ak, n_sources=n_sources, n_targets=n_targets)
            a0, a1, a2 = axes[row]
            _map_panel(a0, tbl, info["influence_thresholds"])
            _curve_panel(a1, info, genes, adata)
            _network_panel(a2, tbl, edges, ref_network)
            lead = complex(np.asarray(info["eigenvalues"])[0]) if len(info["eigenvalues"]) else complex("nan")
            a0.set_title(
                f"archetype {ak} · {info.get('behaviour', '?')} · n={info.get('n_cells', '?')}",
                loc="left", fontsize=7.5,
            )
            a2.set_title(f"t* = {info["influence_t"]:.3g} ({info['influence_t_rule']}) · "
                         f"Re λ₁ = {lead.real:.3g}", fontsize=6.5, loc="left")
        fig.legend(handles=_legend_handles(has_ref), loc="lower center",
                   ncol=6 if not has_ref else 5, fontsize=6, frameon=False,
                   bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.06 if has_ref else 0.04, 1, 1))
        if save is not None:
            fig.savefig(save, dpi=300, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Expression cascade along pseudotime (timing QC)
# ---------------------------------------------------------------------------

_CASCADE_ROLES = ("driver", "hub", "responder")


def _pick_genes(tbl: pd.DataFrame, n_per_role: int) -> pd.DataFrame:
    rows = []
    for rl in _CASCADE_ROLES:
        s = tbl[(tbl["role"] == rl) & tbl["onset"].notna()].sort_values("onset")
        rows.append(s.head(n_per_role))
    return pd.concat(rows)


def figure_expression_cascade(
    adata: AnnData,
    archetype_ids: Optional[Sequence[int]] = None,
    time_key: str = "pseudotime",
    n_per_role: int = 10,
    n_bins: int = 25,
    save: Optional[str | Path] = None,
) -> tuple[Figure, pd.DataFrame]:
    """Timing QC: do drivers change before responders along pseudotime?

    Per archetype (rows):

    (a) heatmap of min-max scaled, smoothed expression over the
        archetype's member cells; rows grouped driver | hub | responder
        (colour strip at left) and sorted by onset (black tick);
    (b) median scaled curve per role with IQR band — falling genes are
        flipped (1 − x) so every curve reads low → high; dotted
        verticals = median onset per role;
    (c) onset per gene by role, with the role-test p (drivers earlier)
        and the edge test (fraction of driver→target edges where the
        source changes first, vs random gene pairs).

    Returns ``(figure, summary)`` — one summary row per archetype.
    """
    from ..timing import expression_timing

    arch = adata.uns["scjdo2"]["archetypes"]
    ids = sorted(arch) if archetype_ids is None else [int(a) for a in archetype_ids]
    rows = []
    with style():
        fig, axes = plt.subplots(len(ids), 3, squeeze=False,
                                 figsize=(OPSPACE_FIG_WIDTH_IN, 2.3 * len(ids) + 0.4),
                                 gridspec_kw={"width_ratios": [1.5, 1.0, 0.8]})
        for row, ak in enumerate(ids):
            res = expression_timing(adata, ak, time_key=time_key, n_bins=n_bins)
            tbl, cen, sc = res["genes"], res["centres"], res["scaled"]
            gidx = {g: i for i, g in enumerate(tbl["gene"])}
            pick = _pick_genes(tbl, n_per_role)
            a0, a1, a2 = axes[row]

            # (a) heatmap
            if len(pick):
                H = np.stack([sc[:, gidx[g]] for g in pick["gene"]])
                a0.imshow(H, aspect="auto", cmap="magma", vmin=0, vmax=1,
                          extent=(cen[0], cen[-1], len(pick), 0), interpolation="nearest")
                for i, (_, r) in enumerate(pick.iterrows()):
                    a0.add_patch(plt.Rectangle((cen[0] - 0.03 * (cen[-1] - cen[0]), i),
                                               0.025 * (cen[-1] - cen[0]), 1,
                                               color=ROLE_COLOURS[r["role"]], clip_on=False))
                    a0.plot([r["onset"]] * 2, [i + 0.15, i + 0.85], color="white", lw=1.2)
                    a0.plot([r["onset"]] * 2, [i + 0.15, i + 0.85], color="black", lw=0.6)
                a0.set_yticks(np.arange(len(pick)) + 0.5)
                a0.set_yticklabels([f"{g}{'↓' if d < 0 else ''}" for g, d in
                                    zip(pick["gene"], pick["direction"])], fontsize=5)
                a0.set_xlim(cen[0] - 0.035 * (cen[-1] - cen[0]), cen[-1])
            else:
                a0.text(0.5, 0.5, "no genes with an onset", transform=a0.transAxes,
                        ha="center", va="center", fontsize=7)
            a0.set_xlabel(time_key)
            a0.set_title(f"archetype {ak} · {arch[ak].get('behaviour', '?')} · "
                         f"{res['n_cells']} cells", loc="left", fontsize=7.5)

            # (b) role medians (falling genes flipped so all rise)
            for rl in _CASCADE_ROLES:
                s = tbl[(tbl["role"] == rl) & tbl["onset"].notna()]
                if s.empty:
                    continue
                C = np.stack([sc[:, gidx[g]] if d > 0 else 1 - sc[:, gidx[g]]
                              for g, d in zip(s["gene"], s["direction"])], axis=1)
                med = np.median(C, axis=1)
                q1, q3 = np.percentile(C, [25, 75], axis=1)
                a1.fill_between(cen, q1, q3, color=ROLE_COLOURS[rl], alpha=0.2, lw=0)
                a1.plot(cen, med, color=ROLE_COLOURS[rl], lw=1.2, label=f"{rl} (n={len(s)})")
                a1.axvline(float(s["onset"].median()), color=ROLE_COLOURS[rl], lw=0.7, ls=":")
            a1.set_ylim(-0.02, 1.02)
            a1.set_xlabel(time_key)
            a1.set_ylabel("scaled expression\n(falling genes flipped)")
            a1.legend(fontsize=5.5, frameon=False, loc="upper left")

            # (c) onset by role
            for x, rl in enumerate(_CASCADE_ROLES):
                v = tbl.loc[(tbl["role"] == rl) & tbl["onset"].notna(), "onset"].to_numpy()
                if v.size == 0:
                    continue
                jit = (np.random.default_rng(x).random(v.size) - 0.5) * 0.35
                a2.scatter(x + jit, v, s=9, c=ROLE_COLOURS[rl], edgecolors="0.25",
                           linewidths=0.3, zorder=3)
                a2.plot([x - 0.3, x + 0.3], [np.median(v)] * 2, color="0.1", lw=1.0, zorder=4)
            a2.set_xticks(range(len(_CASCADE_ROLES)))
            a2.set_xticklabels(_CASCADE_ROLES, fontsize=6)
            a2.set_ylabel(f"onset ({time_key})")
            rt, et = res["role_test"], res["edge_test"]
            p_txt = (f"drivers earlier: p = {rt['p_driver_earlier']:.2g}"
                     if np.isfinite(rt["p_driver_earlier"]) else
                     f"role test n/a (drivers {rt['n_driver']}, responders {rt['n_responder']})")
            e_txt = (f"edges source-first {et['frac_source_first']:.0%} "
                     f"(null {et['null_frac']:.0%}, p = {et['p_perm']:.2g}, n={et['n_edges']})"
                     if np.isfinite(et["frac_source_first"]) else "edge test n/a")
            a2.set_title(p_txt + "\n" + e_txt, fontsize=5.5, loc="left")
            rows.append({"archetype": ak, "behaviour": arch[ak].get("behaviour"),
                         "n_cells": res["n_cells"], **rt, **{f"edge_{k}": v for k, v in et.items()}})
        fig.tight_layout()
        if save is not None:
            fig.savefig(save, dpi=300, bbox_inches="tight")
    return fig, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reference-network recovery (TRRUST etc.)
# ---------------------------------------------------------------------------


def figure_reference_auroc(check, save: Optional[str | Path] = None) -> Figure:
    """AUROC of reference targets under the Jacobian vs co-expression.

    ``check`` is the dict from :func:`scjdo2.drivers.reference_check`
    (or its ``summary`` DataFrame alone).

    (a) per archetype: mean AUROC across reference TFs for the Jacobian
        (``|M[i, TF](t*)|``) and for ``|Spearman|`` co-expression; the
        grey tick is the 95th percentile of the random-target null for
        the Jacobian; ``*`` marks p < 0.05, ``**`` survives Bonferroni.
    (b) per TF (mean over archetypes): Jacobian vs co-expression AUROC;
        points above the diagonal are recovered better by the Jacobian.
        Marker size = number of reference targets in the gene set.
    """
    summ = check["summary"] if isinstance(check, dict) else check
    per_tf = check.get("per_tf") if isinstance(check, dict) else None
    with style():
        ncol = 2 if per_tf is not None and len(per_tf) else 1
        fig, axes = plt.subplots(1, ncol, squeeze=False,
                                 figsize=(OPSPACE_FIG_WIDTH_IN, 2.6),
                                 gridspec_kw={"width_ratios": [1.6, 1.0][:ncol]})
        a0 = axes[0, 0]
        x = np.arange(len(summ))
        w = 0.38
        a0.bar(x - w / 2, summ["mean_auroc_jacobian"], w, color="#E69F00",
               edgecolor="0.25", lw=0.4, label="Jacobian |M(t*)|")
        a0.bar(x + w / 2, summ["mean_auroc_coexpr"], w, color="#B4B2A9",
               edgecolor="0.25", lw=0.4, label="co-expression |ρ|")
        if "null_q95_jacobian" in summ:
            for xi, q in zip(x, summ["null_q95_jacobian"]):
                a0.plot([xi - w, xi], [q, q], color="0.3", lw=0.8)
        a0.axhline(0.5, color="0.5", lw=0.5, ls="--")
        bonf = 0.05 / max(len(summ), 1)
        for xi, auc, p in zip(x, summ["mean_auroc_jacobian"], summ["p_jacobian"]):
            mark = "**" if p < bonf else ("*" if p < 0.05 else "")
            if mark:
                a0.text(xi - w / 2, auc + 0.01, mark, ha="center", fontsize=7)
        a0.set_xticks(x)
        a0.set_xticklabels([str(a) for a in summ["archetype"]])
        a0.set_xlabel("archetype")
        a0.set_ylabel("mean AUROC of reference targets")
        lo = float(min(summ[["mean_auroc_jacobian", "mean_auroc_coexpr"]].min().min(), 0.45))
        a0.set_ylim(lo - 0.03, 1.0)
        n_e = int(summ["n_edges"].iloc[0]) if len(summ) else 0
        n_t = int(summ["n_tfs"].iloc[0]) if len(summ) else 0
        a0.set_title(f"reference recovery · {n_e} edges, {n_t} TFs in gene set "
                     "(grey tick: null 95%)", loc="left", fontsize=7)
        a0.legend(fontsize=6, frameon=False, loc="upper right")
        if ncol == 2:
            a1 = axes[0, 1]
            m = per_tf.groupby("tf")[["n_targets", "auroc_jacobian", "auroc_coexpr"]].mean()
            a1.scatter(m["auroc_coexpr"], m["auroc_jacobian"], s=12 + 10 * m["n_targets"],
                       c="#E69F00", edgecolors="0.25", linewidths=0.4, zorder=3)
            for tf, r in m.iterrows():
                a1.annotate(tf, (r["auroc_coexpr"], r["auroc_jacobian"]), fontsize=5.5,
                            xytext=(2, 2), textcoords="offset points")
            a1.plot([0, 1], [0, 1], color="0.5", lw=0.5, ls="--")
            a1.set_xlim(0, 1)
            a1.set_ylim(0, 1)
            a1.set_xlabel("AUROC, co-expression")
            a1.set_ylabel("AUROC, Jacobian")
            a1.set_title("per TF (mean over archetypes)", loc="left", fontsize=7)
        fig.tight_layout()
        if save is not None:
            fig.savefig(save, dpi=300, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Branch-aware timing
# ---------------------------------------------------------------------------


def figure_branch_timing(result: dict, save: Optional[str | Path] = None) -> Figure:
    """Do driver → target edges change source-first along each branch?

    ``result`` is the dict from :func:`scjdo2.drivers.branch_timing`.

    (a) archetype × branch matrix: fraction of edges whose source changes
        before its target (0.5 = chance; orange = source first, blue =
        target first). Text: fraction, edge count, ``*`` if p < 0.05.
        Last column: all branches pooled for that archetype.
    (b) onset lag ``onset(target) − onset(source)`` per branch, all
        archetypes together; > 0 means the driver changes first.
    """
    summ = result["summary"]
    pooled = result.get("pooled")
    lags = result.get("lags")
    branches = list(dict.fromkeys(summ["branch"]))
    arch_ids = sorted(summ["archetype"].unique())
    with style():
        fig, (a0, a1) = plt.subplots(1, 2, figsize=(OPSPACE_FIG_WIDTH_IN, 0.32 * len(arch_ids) + 1.4),
                                     gridspec_kw={"width_ratios": [1.3, 1.0]})
        cols = branches + (["pooled"] if pooled is not None else [])
        F = np.full((len(arch_ids), len(cols)), np.nan)
        txt = [["" for _ in cols] for _ in arch_ids]
        for i, ak in enumerate(arch_ids):
            for j, br in enumerate(branches):
                r = summ[(summ["archetype"] == ak) & (summ["branch"] == br)]
                if r.empty or not np.isfinite(r["frac_source_first"].iloc[0]):
                    txt[i][j] = "–"
                    continue
                r = r.iloc[0]
                F[i, j] = r["frac_source_first"]
                star = "*" if r["p_perm"] < 0.05 else ""
                txt[i][j] = f"{r['frac_source_first']:.2f}{star}\nn={int(r['n_edges'])}"
            if pooled is not None:
                r = pooled[pooled["archetype"] == ak]
                if len(r) and np.isfinite(r["frac_source_first"].iloc[0]):
                    r = r.iloc[0]
                    F[i, -1] = r["frac_source_first"]
                    star = "*" if r["p_perm"] < 0.05 else ""
                    txt[i][-1] = f"{r['frac_source_first']:.2f}{star}\nn={int(r['n_pairs'])}"
        im = a0.imshow(F, cmap="PuOr_r", vmin=0, vmax=1, aspect="auto")
        for i in range(len(arch_ids)):
            for j in range(len(cols)):
                dark = np.isfinite(F[i, j]) and abs(F[i, j] - 0.5) > 0.35
                a0.text(j, i, txt[i][j], ha="center", va="center", fontsize=5.5,
                        color="white" if dark else "black")
        a0.set_xticks(range(len(cols)))
        a0.set_xticklabels(cols, fontsize=6.5)
        a0.set_yticks(range(len(arch_ids)))
        a0.set_yticklabels([f"arch {a}" for a in arch_ids], fontsize=6.5)
        if pooled is not None:
            a0.axvline(len(branches) - 0.5, color="0.2", lw=0.8)
        cb = fig.colorbar(im, ax=a0, fraction=0.04, pad=0.02)
        cb.set_label("fraction source-first", fontsize=6)
        ov = result.get("overall")
        if ov is not None and len(ov):
            o = ov.iloc[0]
            a0.set_title(f"source-first {o['frac_source_first']:.2f} overall "
                         f"(null {o['null_frac']:.2f}; {int(o['n_pairs'])} edge×branch pairs; "
                         f"cells = {o['cells']})", loc="left", fontsize=6.5)

        if lags is not None and len(lags):
            for j, br in enumerate(branches):
                v = lags.loc[lags["branch"] == br, "lag"].dropna().to_numpy()
                if v.size == 0:
                    continue
                jit = (np.random.default_rng(j).random(v.size) - 0.5) * 0.4
                a1.scatter(j + jit, v, s=6, c="#E69F00", edgecolors="none", alpha=0.6, zorder=2)
                a1.boxplot(v, positions=[j], widths=0.5, showfliers=False,
                           medianprops={"color": "0.1", "lw": 1.0},
                           boxprops={"lw": 0.6}, whiskerprops={"lw": 0.6}, capprops={"lw": 0.6})
            a1.axhline(0, color="0.4", lw=0.6, ls="--")
            a1.set_xticks(range(len(branches)))
            a1.set_xticklabels(branches, fontsize=6.5)
            a1.set_ylabel("onset(target) − onset(source)\n(> 0: driver first)")
        fig.tight_layout()
        if save is not None:
            fig.savefig(save, dpi=300, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# One archetype: network genes across cell states (optional QC)
# ---------------------------------------------------------------------------


def figure_network_expression(
    adata: AnnData,
    archetype_id: int,
    cluster_key: str,
    states: Optional[Sequence[str]] = None,
    time_key: str = "pseudotime",
    result: Optional[dict] = None,
    save: Optional[str | Path] = None,
) -> Figure:
    """Does each driver switch on before its responders, state by state?

    One archetype. Genes are the sources (drivers, hubs) and targets of
    its influence network; columns are cell states in trajectory order.

    Left: dot heatmap, one block per source (source row first, bold,
    then its targets). Colour = mean expression scaled 0–1 per gene
    across states; dot size = fraction of cells expressing; ▼ = on-state
    (where the scaled mean crosses 0.5). ``↓`` marks genes that fall.
    The left strip gives each gene's role in this archetype.

    Right: one panel per source — the source (thick) and its targets
    (thin) across states, falling genes flipped so all read low → high.
    Title: how many targets switch on after the source.
    """
    from matplotlib.gridspec import GridSpec

    from ..timing import network_state_expression

    res = result if result is not None else network_state_expression(
        adata, archetype_id, cluster_key, states=states, time_key=time_key)
    sts, sc, fr = res["states"], res["scaled"], res["frac"]
    g = res["genes"].set_index("gene")
    e = res["edges"]
    sources = list(dict.fromkeys(e["source"]))
    blocks = [(s, list(e.loc[e["source"] == s, "target"])) for s in sources]
    rows = []
    for s, tg in blocks:
        rows.append((s, True))
        rows += [(t, False) for t in tg]
    n_rows, S = len(rows), len(sts)
    info = adata.uns["scjdo2"]["archetypes"][int(archetype_id)]

    with style():
        fig = plt.figure(figsize=(OPSPACE_FIG_WIDTH_IN, max(3.0, 0.17 * n_rows + 1.2)))
        gs = GridSpec(max(len(blocks), 1), 2, figure=fig, width_ratios=[1.25, 1.0],
                      wspace=0.35, hspace=0.75)
        a0 = fig.add_subplot(gs[:, 0])
        for i, (gene, is_src) in enumerate(rows):
            y = n_rows - 1 - i
            v = sc.loc[gene].to_numpy(dtype=float)
            f = fr.loc[gene].to_numpy(dtype=float)
            a0.scatter(np.arange(S), np.full(S, y), c=v, cmap="magma", vmin=0, vmax=1,
                       s=8 + 70 * np.nan_to_num(f), edgecolors="0.3", linewidths=0.2, zorder=3)
            on = g.loc[gene, "on_state"]
            if np.isfinite(on):
                a0.scatter([on], [y + 0.38], marker="v", s=12, c="black", zorder=4)
            rl = g.loc[gene, "role"]
            a0.add_patch(plt.Rectangle((-0.9, y - 0.4), 0.25, 0.8,
                                       color=ROLE_COLOURS.get(rl, "#FFFFFF"), ec="0.4",
                                       lw=0.2, clip_on=False))
            if is_src and i > 0:
                a0.axhline(y + 0.5, color="0.6", lw=0.5)
        labels = [f"{gene}{' ↓' if g.loc[gene, 'direction'] < 0 else ''}" for gene, _ in rows]
        a0.set_yticks(np.arange(n_rows)[::-1])
        a0.set_yticklabels(labels, fontsize=5.5)
        for tick, (_, is_src) in zip(a0.get_yticklabels(), rows):
            if is_src:
                tick.set_fontweight("bold")
        a0.set_xticks(np.arange(S))
        a0.set_xticklabels(sts, rotation=60, ha="right", fontsize=6)
        a0.set_xlim(-1.0, S - 0.5)
        a0.set_ylim(-0.7, n_rows - 0.3)
        a0.set_title(f"archetype {archetype_id} · {info.get('behaviour', '?')} · "
                     "network genes across states", loc="left", fontsize=7.5)
        for sp in ("top", "right"):
            a0.spines[sp].set_visible(False)

        summ = res["summary"].set_index("source") if len(res["summary"]) else None
        for k, (s, tg) in enumerate(blocks):
            ax = fig.add_subplot(gs[k, 1])
            flip = lambda gene: (1 - sc.loc[gene].to_numpy(dtype=float)
                                 if g.loc[gene, "direction"] < 0 else sc.loc[gene].to_numpy(dtype=float))
            for t in tg:
                ax.plot(np.arange(S), flip(t), color=ROLE_COLOURS.get(g.loc[t, "role"], "0.6")
                        if g.loc[t, "role"] not in ("passenger", "unmeasured") else "0.6",
                        lw=0.7, alpha=0.8)
            ax.plot(np.arange(S), flip(s), color="0.1", lw=1.8)
            on = g.loc[s, "on_state"]
            if np.isfinite(on):
                ax.axvline(on, color="0.1", lw=0.6, ls=":")
            n_t = int(summ.loc[s, "n_targets"]) if summ is not None and s in summ.index else 0
            n_f = int(summ.loc[s, "n_driver_first"]) if summ is not None and s in summ.index else 0
            ax.set_title(f"{s} ({g.loc[s, 'role']}): first in {n_f}/{n_t} targets",
                         fontsize=6, loc="left")
            ax.set_ylim(-0.05, 1.05)
            ax.set_yticks([0, 1])
            ax.set_xticks(np.arange(S))
            ax.set_xticklabels(sts if k == len(blocks) - 1 else [], rotation=60,
                               ha="right", fontsize=5.5)
        if save is not None:
            fig.savefig(save, dpi=300, bbox_inches="tight")
    return fig
