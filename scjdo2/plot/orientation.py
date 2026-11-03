"""Orientation-layer figure (docs/ORIENTATION_SPEC.md §12, §15, §16).

:func:`figure_orientation` summarises a stored :func:`scjdo2.orient.orient`
result in four panels:

* **A, B** — the 2-D embedding coloured by how each cell's flow (A) and
  developmental (B) signs were obtained: ``measured``, ``imputed`` or
  ``none``;
* **C** — per group: fraction of cells oriented in each output, the
  conflict rate within each output and the flow-vs-developmental
  disagreement;
* **D** — the sources used, what decided the signs, and the reliability /
  guard outcomes recorded in ``uns['scjdo2']['orient_meta']``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from anndata import AnnData
from matplotlib.figure import Figure

from .style import style

CLASS_COLOURS = {"measured": "#1b7837", "imputed": "#e08214", "none": "#d9d9d9"}


def _xy(adata: AnnData, basis: Optional[str]) -> tuple[np.ndarray, str]:
    if basis is None:
        basis = "X_umap" if "X_umap" in adata.obsm else adata.uns.get("scjdo2", {}).get(
            "meta", {}).get("use_rep", "X_pca")
    if basis not in adata.obsm:
        raise KeyError(f"obsm[{basis!r}] missing")
    return np.asarray(adata.obsm[basis])[:, :2], basis


def figure_orientation(adata: AnnData, groupby: Optional[str] = None,
                       basis: Optional[str] = None,
                       out: Optional[str | Path] = None) -> Figure:
    """Four-panel summary of the orientation layer's outputs.

    Needs the obs columns written by :func:`scjdo2.orient.orient`
    (``flow_orientation_class``, ``dev_orientation_class``, ...). ``groupby``
    defaults to ``"cluster"`` or ``"day"`` when present, else one group.
    """
    import matplotlib.pyplot as plt

    meta = adata.uns.get("scjdo2", {}).get("orient_meta")
    if meta is None or "flow_orientation_class" not in adata.obs:
        raise RuntimeError("no orientation result stored; run scjdo2.orient.orient first")
    xy, basis = _xy(adata, basis)
    if groupby is None:
        groupby = next((g for g in ("cluster", "cell_type", "day") if g in adata.obs), None)
    groups = (adata.obs[groupby].astype(str).to_numpy() if groupby
              else np.full(adata.n_obs, "all"))
    order = list(dict.fromkeys(sorted(groups)))

    with style():
        fig, ax = plt.subplots(2, 2, figsize=(9.0, 7.2))
        for a, g, title in ((ax[0, 0], "flow", "A  flow: how cells move now"),
                            (ax[0, 1], "dev", "B  developmental: where they go")):
            cls = adata.obs[f"{g}_orientation_class"].astype(str).to_numpy()
            for c in ("none", "imputed", "measured"):
                m = cls == c
                a.scatter(xy[m, 0], xy[m, 1], s=4, c=CLASS_COLOURS[c], lw=0,
                          label=f"{c} ({m.mean():.0%})", rasterized=True)
            srcs = meta["groups"].get(g, [])
            a.set_title(f"{title}\nsources: {', '.join(srcs) or 'none'}", fontsize=8, loc="left")
            a.set_xticks([]); a.set_yticks([]); a.set_xlabel(f"{basis} 1"); a.set_ylabel(f"{basis} 2")
            a.legend(frameon=False, fontsize=7, markerscale=2, loc="best")

        a = ax[1, 0]
        cols = [("flow", "flow_orientation_class", "oriented"), ("dev", "dev_orientation_class", "oriented"),
                ("flow", "flow_conflict_flag", "conflict"), ("dev", "dev_conflict_flag", "conflict"),
                ("cells", "flow_vs_dev_disagree", "with flow ≠ dev on an axis")]
        width = 0.8 / len(cols)
        x = np.arange(len(order))
        for i, (g, col, kind) in enumerate(cols):
            v = adata.obs[col].to_numpy()
            frac = (v != "none") if kind == "oriented" else v.astype(bool)
            vals = [float(np.mean(frac[groups == o])) for o in order]
            a.bar(x + (i - len(cols) / 2 + 0.5) * width, vals, width, label=f"{g} {kind}")
        a.set_xticks(x, order, rotation=45, ha="right", fontsize=7)
        a.set_ylim(0, 1); a.set_ylabel("fraction of cells")
        a.set_title(f"C  by {groupby or 'all cells'}", fontsize=8, loc="left")
        a.legend(frameon=False, fontsize=6, ncol=2)

        a = ax[1, 1]; a.axis("off")
        lines = [f"flow coverage {meta['flow_coverage']:.0%}   dev coverage {meta['dev_coverage']:.0%}",
                 f"flow vs dev disagreement {meta['flow_vs_dev_disagreement_rate']:.0%} "
                 "(of pairs both sign)", ""]
        for g, rel in meta.get("reliability", {}).items():
            for s, r in rel.items():
                if r.get("status") == "not_checked":
                    lines.append(f"{s}: not checked ({r.get('reason', '')})")
                else:
                    lines.append(f"{s}: {r['status']} (agreement {r['agreement']:.2f} vs "
                                 f"{', '.join(r['against'])}; threshold {r['threshold']:.2f})")
        pt = meta.get("per_source", {}).get("pseudotime")
        if pt:
            rho = pt.get("spearman_with_day")
            lines.append(f"pseudotime: start cell at percentile {100 * pt['root_percentile']:.1f}"
                         + (f"; Spearman with day {rho:.2f}" if rho is not None else ""))
        lines += ["", f"J unchanged: {meta['hash_unchanged']}",
                  "Signs are never blended; the two outputs are never merged."]
        a.text(0, 1, "D  ledger\n\n" + "\n".join(lines), va="top", fontsize=7, family="monospace")
        fig.tight_layout()
        if out is not None:
            out = Path(out)
            fig.savefig(out.with_suffix(".pdf"))
            fig.savefig(out.with_suffix(".png"), dpi=200)
    return fig
