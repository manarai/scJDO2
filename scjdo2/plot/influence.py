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
