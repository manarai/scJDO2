"""One save routine for every scjdo2 figure.

Convention: figure functions take ``out`` — a path *without* extension
(any extension given is replaced) — and write one file per format,
PDF + PNG by default. ``None`` writes nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from matplotlib.figure import Figure


def save_figure(
    fig: Figure,
    out: Optional[str | Path],
    formats: Sequence[str] = ("pdf", "png"),
    png_dpi: int = 300,
    tight: bool = False,
) -> list[Path]:
    """Write ``fig`` to ``out.<fmt>`` for each format; returns the paths.

    ``png_dpi`` applies to raster formats; vector formats keep the
    figure's own dpi. ``tight=True`` crops to the drawn content
    (``bbox_inches="tight"``). Creates the parent directory.
    """
    if out is None:
        return []
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kw = {"bbox_inches": "tight"} if tight else {}
    paths = []
    for fmt in formats:
        p = out.with_suffix(f".{fmt}")
        if fmt in ("png", "jpg", "jpeg", "tif", "tiff"):
            fig.savefig(p, dpi=png_dpi, **kw)
        else:
            fig.savefig(p, **kw)
        paths.append(p)
    return paths
