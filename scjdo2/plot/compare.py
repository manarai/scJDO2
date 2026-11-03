"""v1 vs scJDO2 side-by-side comparison figure."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from matplotlib.figure import Figure

from ._save import save_figure
from .operator_space import _panel_a
from .style import OPSPACE_FIG_HEIGHT_IN, OPSPACE_FIG_WIDTH_IN, style


def figure_v1_vs_v2(
    adata_v2: AnnData,
    v1_results: str | Path | dict,
    out: Optional[str | Path] = None,
    use_rep: Optional[str] = None,
) -> Figure:
    """Side-by-side comparison of v1 slicing archetypes vs. scJDO2 Panel A.

    ``v1_results`` is either a path to a v1 baseline directory (produced by
    ``compare/run_v1.py``, containing ``adata.h5ad`` with
    ``uns["scjdo"]["activations"]`` and ``uns["scjdo"]["t_centers"]``) or
    a pre-loaded dict with keys ``activations``, ``t_centers``.
    """
    import anndata as ad

    if isinstance(v1_results, (str, Path)):
        v1_adata = ad.read_h5ad(Path(v1_results) / "adata.h5ad")
        v1 = {
            "activations": np.asarray(v1_adata.uns["scjdo"]["activations"]),
            "t_centers": np.asarray(v1_adata.uns["scjdo"]["t_centers"]),
        }
    else:
        v1 = dict(v1_results)

    acts = np.asarray(v1["activations"])          # (T, K)
    tc = np.asarray(v1["t_centers"])              # (T,)
    K = acts.shape[1]

    with style():
        fig = plt.figure(figsize=(OPSPACE_FIG_WIDTH_IN, OPSPACE_FIG_HEIGHT_IN))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.1], wspace=0.3)
        ax_v1 = fig.add_subplot(gs[0, 0])
        ax_v2 = fig.add_subplot(gs[0, 1])

        for k in range(K):
            ax_v1.plot(tc, acts[:, k], linewidth=1.0, label=f"archetype {k}")
        ax_v1.set_xlabel("pseudotime")
        ax_v1.set_ylabel("archetype activation")
        ax_v1.set_title("v1 slicing", loc="left", fontsize=9)
        ax_v1.legend(loc="upper right", fontsize=6)

        _panel_a(ax_v2, adata_v2, use_rep=use_rep, lineages=None, label_fork=True)
        ax_v2.set_title("scJDO2 operator field", loc="left", fontsize=9)

        save_figure(fig, out)
    return fig
