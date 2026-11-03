"""Timing QC figures: pseudotime cascade, branch timing, network genes
across cell states. They draw results stored by :mod:`scjdo2.drivers`
(``timing``, ``branch_timing``, ``network_expression``)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib.figure import Figure

from ._save import save_figure
from .style import OPSPACE_FIG_WIDTH_IN, ROLE_COLOURS, style


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
    n_per_role: int = 10,
    out: Optional[str | Path] = None,
) -> Figure:
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

    Needs :func:`scjdo2.drivers.timing` (stored per archetype as
    ``timing``); the summary table comes from that call, not from here.
    ``out``: path stem; writes PDF + PNG.
    """
    arch = adata.uns.get("scjdo2", {}).get("archetypes") or {}
    ids = sorted(arch) if archetype_ids is None else [int(a) for a in archetype_ids]
    missing = [a for a in ids if "timing" not in arch[a]]
    if not arch or missing:
        raise RuntimeError(
            f"no timing results for archetypes {missing or 'any'}; "
            "run scjdo2.drivers.timing(adata) first"
        )
    with style():
        fig, axes = plt.subplots(len(ids), 3, squeeze=False,
                                 figsize=(OPSPACE_FIG_WIDTH_IN, 2.3 * len(ids) + 0.4),
                                 gridspec_kw={"width_ratios": [1.5, 1.0, 0.8]})
        for row, ak in enumerate(ids):
            res = arch[ak]["timing"]
            time_key = res["time_key"]
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
        fig.tight_layout()
        save_figure(fig, out, tight=True)
    return fig


# ---------------------------------------------------------------------------
# Branch-aware timing
# ---------------------------------------------------------------------------


def figure_branch_timing(result: dict, out: Optional[str | Path] = None) -> Figure:
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
        save_figure(fig, out, tight=True)
    return fig


# ---------------------------------------------------------------------------
# One archetype: network genes across cell states (optional QC)
# ---------------------------------------------------------------------------


def figure_network_expression(
    adata: AnnData,
    archetype_id: int,
    result: Optional[dict] = None,
    out: Optional[str | Path] = None,
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

    Draws ``result`` if given, else the archetype's stored
    ``network_expression`` (from :func:`scjdo2.drivers.network_expression`).
    ``out``: path stem; writes PDF + PNG.
    """
    from matplotlib.gridspec import GridSpec

    info = adata.uns["scjdo2"]["archetypes"][int(archetype_id)]
    res = result if result is not None else info.get("network_expression")
    if res is None:
        raise RuntimeError(
            f"no network_expression result for archetype {archetype_id}; run "
            "scjdo2.drivers.network_expression(adata, archetype_id, cluster_key) first"
        )
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
        save_figure(fig, out, tight=True)
    return fig
