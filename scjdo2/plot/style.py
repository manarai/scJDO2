"""Journal-ready matplotlib style for scJDO2 figures.

Shared colour key for operator-space behaviour classes. Marker shape is a
required second channel so the figure survives greyscale printing.

See ``docs/scJDO2_figure_layer_build_instructions.md`` §2 for the full
style contract this module implements: behaviour colours/markers,
``sensitive_ring``, diverging/cyclic colormaps, ``style()`` context,
``confidence_alpha`` (rank-based rescale), ``relative_label`` (axis
suffix that respects ``uns["scjdo2"]["clock_calibrated"]``).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Colour-blind-safe palette shared across all operator-space figures.
BEHAVIOUR_COLOURS: dict[str, str] = {
    "holder": "#378ADD",
    "drift": "#888780",
    "fork": "#D85A30",
    "damped_oscillator": "#7F77DD",
    "growing_oscillator": "#7F77DD",
}

# Marker shape for each behaviour. 'o' = filled circle; 'o+' = ringed circle;
# 'o+dot' = ringed circle with dot.  We implement these by scatter calls that
# combine basic markers; the renderer helpers in :mod:`scjdo2.plot.figures`
# consume this mapping directly.
BEHAVIOUR_MARKERS: dict[str, str] = {
    "holder": "o",
    "drift": "o",
    "fork": "o",
    "damped_oscillator": "o+",
    "growing_oscillator": "o+dot",
}

DIVERGING_CMAP = "RdBu_r"       # blue–white–red; the standard choice
PHASE_CMAP = "twilight"         # cyclic, perceptually uniform

# §2 aliases (shorter names from the build instructions).
DIVERGING = DIVERGING_CMAP
CYCLIC = PHASE_CMAP

# Figure sizing — matches journal 180 mm wide full-column.
OPSPACE_FIG_WIDTH_IN: float = 180.0 / 25.4
OPSPACE_FIG_HEIGHT_IN: float = (180.0 / 25.4) * 0.65

_STYLE_RC = {
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "axes.linewidth": 0.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,          # true-type embed for editable PDFs
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.dpi": 150,
}


@contextmanager
def style() -> Iterator[None]:
    """Apply the journal style for the duration of the ``with`` block.

    >>> with scjdo2.plot.style():
    ...     fig = scjdo2.plot.figure_operator_space(adata)
    """
    with mpl.rc_context(_STYLE_RC):
        yield


def behaviour_legend_handles():
    """Return (handles, labels) for a single shared legend showing the
    behaviour → colour × marker key, including the dashed sensitive ring."""
    import matplotlib.lines as mlines

    handles = []
    labels = []
    for beh, colour in BEHAVIOUR_COLOURS.items():
        marker = BEHAVIOUR_MARKERS.get(beh, "o")
        if marker == "o+dot":
            h = mlines.Line2D([], [], marker="o", linestyle="",
                              markerfacecolor=colour, markeredgecolor=colour,
                              markersize=7, markeredgewidth=1.5)
        elif marker == "o+":
            h = mlines.Line2D([], [], marker="o", linestyle="",
                              markerfacecolor="none", markeredgecolor=colour,
                              markersize=7, markeredgewidth=1.5)
        else:
            h = mlines.Line2D([], [], marker="o", linestyle="",
                              markerfacecolor=colour, markeredgecolor=colour,
                              markersize=6)
        handles.append(h)
        labels.append(beh.replace("_", " "))
    # Sensitive ring: dashed outer circle, same colour semantics (grey proxy).
    sens = mlines.Line2D([], [], marker="o", linestyle="",
                         markerfacecolor="none", markeredgecolor="#404040",
                         markersize=10, markeredgewidth=0.75)
    sens.set_dashes((3, 2))
    handles.append(sens)
    labels.append("sensitive (henrici ≥ p90)")
    return handles, labels


# ----------------------------------------------------------------------
# §2 helpers used by figures.py
# ----------------------------------------------------------------------


def sensitive_ring(
    ax: "plt.Axes",
    x: float | Sequence[float],
    y: float | Sequence[float],
    colour: str,
    r: float,
) -> list:
    """Draw a dashed outer ring on each (x, y): radius 2r, 0.75 pt, dash (3, 2).

    Returns the list of artist handles so tests can count them. ``r`` is
    the base marker radius (in display units); the ring sits at 2× that,
    matching the §0 "sensitive-holder mark" rule.
    """
    import matplotlib.patches as mpatches

    xs = np.atleast_1d(np.asarray(x, dtype=float))
    ys = np.atleast_1d(np.asarray(y, dtype=float))
    handles: list = []
    for xi, yi in zip(xs, ys):
        circ = mpatches.Circle(
            (xi, yi), radius=2.0 * r, fill=False,
            edgecolor=colour, linewidth=0.75,
            linestyle=(0, (3, 2)),
        )
        ax.add_patch(circ)
        handles.append(circ)
    return handles


def confidence_alpha(
    values: Sequence[float] | np.ndarray,
    lo: float = 0.15,
    hi: float = 1.0,
) -> np.ndarray:
    """Rank-based rescale of ``values`` into ``[lo, hi]``.

    Confidence columns are heavy-tailed (per-cell Fisher medians); using
    the rank instead of the raw magnitude keeps the alpha channel evenly
    distributed across the figure without losing the ordering. NaN
    entries map to ``lo`` so missing-confidence cells are not loud.
    """
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, lo, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return out
    ranks = np.empty(arr.shape, dtype=float)
    order = np.argsort(arr[finite], kind="mergesort")
    rk = np.empty(order.shape, dtype=float)
    rk[order] = np.linspace(0.0, 1.0, order.size) if order.size > 1 else np.array([1.0])
    ranks_full = np.zeros_like(arr, dtype=float)
    idx = np.where(finite)[0]
    ranks_full[idx] = rk
    out[finite] = lo + (hi - lo) * ranks_full[finite]
    return out


def relative_label(adata: Any, base: str) -> str:
    """Return ``base`` or ``f"{base} (relative)"`` depending on calibration.

    The spec requires every eigenvalue axis to carry "(relative)" unless
    the clock-calibration flag is ``True``. Both current conventions are
    honoured: ``uns["scjdo2"]["clock_calibrated"]`` (the top-level flag
    introduced by the §0 addendum) and the legacy
    ``uns["scjdo2"]["meta"]["clock_calibrated"]`` nested form. ``adata``
    can also be a plain dict (handy for tests).
    """
    try:
        uns = getattr(adata, "uns", adata)
        scj = uns["scjdo2"] if isinstance(uns, dict) else uns.get("scjdo2", {})
        meta = scj.get("meta", {}) if isinstance(scj, dict) else {}
        calibrated = bool(scj.get("clock_calibrated", False) or meta.get("clock_calibrated", False))
    except (KeyError, AttributeError, TypeError):
        calibrated = False
    return base if calibrated else f"{base} (relative)"
