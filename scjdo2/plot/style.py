"""Journal-ready matplotlib style for scJDO2 figures.

Shared colour key for operator-space behaviour classes. Marker shape is a
required second channel so the figure survives greyscale printing.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import matplotlib as mpl
import matplotlib.pyplot as plt

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
    behaviour → colour × marker key."""
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
    return handles, labels
