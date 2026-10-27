"""Shared style, provenance and output helpers for the v57 figure notebooks.

Every figure notebook does:

    from _style import *
    prov = Provenance("Fig3")
    df = pd.read_csv(prov.use("benchmarks/...csv", panel="a", task="S14"))
    ...
    prov.value("auc_ref_min", x)          # number used in the legend
    save(fig, "Fig3", kind="main")
    prov.legend("...markdown...")
    prov.write()

Paths given to ``Provenance.use`` are relative to the repository root unless
absolute. sha256 is computed at read time and written to
``manuscript/figures/provenance.json`` (one entry per figure).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIGDIR = REPO / "manuscript" / "figures"
PROV_PATH = FIGDIR / "provenance.json"
VALUES_DIR = FIGDIR / "values"
LEGEND_DIR = FIGDIR / "legends"

# Final S15 results, merged into the repository (orientation branch).
S15_DEFAULT = REPO / "benchmarks" / "synthetic_multimodal" / "s15"
S15_DIR = Path(os.environ.get("SCJDO2_S15_DIR", str(S15_DEFAULT)))

# ---------------------------------------------------------------- palette
# Okabe-Ito (colour-blind safe). Operator colours are fixed across figures.
OKABE = {
    "black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermillion": "#D55E00", "purple": "#CC79A7", "grey": "#7F7F7F",
}
C_FLOW = OKABE["blue"]          # flow operator (OT time course unless stated)
C_SCORE = OKABE["vermillion"]   # score operator
C_BASE = OKABE["grey"]          # Leiden / baselines / nulls
C_LABEL = OKABE["green"]        # labelling-based flow J
C_SCVELO = OKABE["purple"]      # scVelo-based flow J
C_LINEAGE = OKABE["orange"]     # lineage-based flow J
C_SELECTED = OKABE["orange"]    # S14 tuned ("selected") configuration
C_REF = C_FLOW                  # S14 a-priori reference configuration
C_REAL = OKABE["black"]
C_GEN = {"v1": OKABE["skyblue"], "v2": OKABE["orange"], "v3": OKABE["green"],
         "dyngen": OKABE["purple"]}
C_PENDING = "#BBBBBB"

# ---------------------------------------------------------------- output spec
# Journal sizes: single column 89 mm, double column 183 mm, max height 240 mm.
MM = 1 / 25.4
SINGLE_COL = 89 * MM    # 3.504 in
ONEHALF_COL = 120 * MM  # 4.724 in
DOUBLE_COL = 183 * MM   # 7.205 in
MAX_HEIGHT = 240 * MM
DPI = 600               # PNG and TIFF
MIN_FONT = 6.0          # pt, enforced at save time
MIN_LW = 0.5            # pt, enforced at save time (zero widths are left at zero)
FORMATS = ("pdf", "png", "tiff")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "legend.title_fontsize": 6.5,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "patch.linewidth": 0.5,
    "hatch.linewidth": 0.5,
    "axes.grid": False,
    "savefig.dpi": DPI,
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,     # embed TrueType
    "ps.fonttype": 42,
    "figure.dpi": 100,
})

PANEL_FONTSIZE = 9


def panel_letter(ax, letter, x=None, y=None):
    """Register a bold panel letter for ``ax``.

    The letter is placed at save time, top-left and outside the axes' full
    extent (tick labels, axis label and title included), so it never overlaps
    the panel's own title. ``x``/``y`` are accepted for backwards compatibility
    and ignored.
    """
    fig = ax.figure
    if not hasattr(fig, "_panel_letters"):
        fig._panel_letters = []
    fig._panel_letters.append((ax, letter, x, y))


def _place_letters(fig):
    from matplotlib.transforms import Bbox
    if not getattr(fig, "_panel_letters", None):
        return
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    spots = []
    for ax, letter, _x, _y in fig._panel_letters:
        # panel extent = axes + y tick labels / y label + titles (legends and x labels ignored)
        ab = ax.get_window_extent(r)
        x0, top = ab.x0, ab.y1
        ya = ax.yaxis.get_tightbbox(r)
        if ya is not None and ya.width > 0:
            x0 = min(x0, ya.x0)
        for t in (ax.title, ax._left_title, ax._right_title):
            if t.get_text():
                tb = t.get_window_extent(r)
                top = max(top, tb.y1)
                x0 = min(x0, tb.x0)
        spots.append([inv.transform((x0, top)), inv.transform((0, ab.y1))[1], letter])
    # letters of panels whose axes share a top edge (same row) are aligned at the highest top
    for s_ in spots:
        row = [o for o in spots if abs(o[1] - s_[1]) < 0.02]
        s_.append(max(o[0][1] for o in row))
    for (xy, _ay, letter, y1) in spots:
        fig.text(xy[0] - 0.004, y1, letter, fontsize=PANEL_FONTSIZE, fontweight="bold",
                 ha="right", va="top")
    fig._panel_letters = []


def enforce_minimums(fig, min_font=MIN_FONT, min_lw=MIN_LW):
    """Clamp every text to >= min_font pt and every non-zero line width to >= min_lw pt."""
    import matplotlib.text as mtext
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    import matplotlib.collections as mcoll
    for t in fig.findobj(mtext.Text):
        if t.get_text() and t.get_fontsize() < min_font:
            t.set_fontsize(min_font)
    for ln in fig.findobj(mlines.Line2D):
        lw = ln.get_linewidth()
        if 0 < lw < min_lw:
            ln.set_linewidth(min_lw)
        mew = ln.get_markeredgewidth()
        if 0 < mew < min_lw and ln.get_marker() not in (None, "None", "", " "):
            ln.set_markeredgewidth(min_lw)
    for pa in fig.findobj(mpatches.Patch):
        lw = pa.get_linewidth()
        if 0 < lw < min_lw and pa.get_edgecolor()[3] > 0:
            pa.set_linewidth(min_lw)
    for c in fig.findobj(mcoll.Collection):
        lws = np.atleast_1d(c.get_linewidths())
        if lws.size and np.any((lws > 0) & (lws < min_lw)):
            c.set_linewidths(np.where((lws > 0) & (lws < min_lw), min_lw, lws))
    for ax in fig.axes:
        for sp in ax.spines.values():
            if 0 < sp.get_linewidth() < min_lw:
                sp.set_linewidth(min_lw)


def placeholder(ax, text, fontsize=6.5):
    """A clearly labelled placeholder panel (no data drawn)."""
    ax.set_facecolor("#F4F4F4")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linestyle((0, (3, 2)))
        s.set_color("#888888")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="#555555")


def stamp(fig, text, color="#B00000", y=1.04):
    fig.text(0.995, y, text, ha="right", va="top", fontsize=7,
             color=color, fontweight="bold")


SIZES = {}   # name -> (width_mm, height_mm, png_px) of the last save


def save(fig, name, kind="main", pad=0.03):
    """Write <name>.pdf (vector, TrueType), .png (600 dpi) and .tiff (600 dpi, LZW, RGB)."""
    import io
    from PIL import Image
    out = FIGDIR / kind
    out.mkdir(parents=True, exist_ok=True)
    enforce_minimums(fig)
    _place_letters(fig)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=pad,
                metadata={"CreationDate": None, "ModDate": None})
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", pad_inches=pad,
                metadata={"Software": None})
    (out / f"{name}.png").write_bytes(buf.getvalue())
    im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    im.save(out / f"{name}.tiff", compression="tiff_lzw", dpi=(DPI, DPI))
    w_mm, h_mm = im.size[0] / DPI * 25.4, im.size[1] / DPI * 25.4
    SIZES[name] = (round(w_mm, 1), round(h_mm, 1), im.size)
    flag = ""
    if w_mm > 183.5 or h_mm > 240.5:
        flag = "  WARNING: exceeds 183 x 240 mm"
    print(f"[save] {kind}/{name}: {w_mm:.1f} x {h_mm:.1f} mm, PNG {im.size[0]} x {im.size[1]} px @ {DPI} dpi{flag}")
    return out / f"{name}.pdf", out / f"{name}.png"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path):
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else REPO / p


class Provenance:
    def __init__(self, figure, notebook=None, kind="main", interim=False):
        self.figure = figure
        self.notebook = notebook
        self.kind = kind
        self.interim = interim
        self.sources = {}   # rel path -> {sha256, panels:set, tasks:set}
        self.values = {}
        self.panels = {}    # panel -> {description, status}
        self._legend = None

    def use(self, path, panel, task, note=None):
        p = resolve(path)
        if not p.exists():
            raise FileNotFoundError(p)
        rel = _rel(p)
        e = self.sources.setdefault(rel, {"sha256": sha256(p), "panels": set(),
                                          "tasks": set(), "notes": set()})
        e["panels"].add(str(panel))
        e["tasks"].add(task)
        if note:
            e["notes"].add(note)
        return p

    def panel(self, panel, description, status="data"):
        self.panels[str(panel)] = {"description": description, "status": status}

    def value(self, key, value):
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        self.values[key] = value
        return value

    def legend(self, markdown):
        self._legend = markdown

    def write(self):
        allp = json.loads(PROV_PATH.read_text()) if PROV_PATH.exists() else {}
        allp[self.figure] = {
            "notebook": self.notebook,
            "kind": self.kind,
            "interim": self.interim,
            "outputs": [f"manuscript/figures/{self.kind}/{self.figure}.{ext}" for ext in FORMATS]
            if self.kind in ("main", "supplementary") else [],
            "panels": self.panels,
            "sources": {k: {"sha256": v["sha256"], "panels": sorted(v["panels"]),
                            "tasks": sorted(v["tasks"]), "notes": sorted(v["notes"])}
                        for k, v in sorted(self.sources.items())},
        }
        PROV_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROV_PATH.write_text(json.dumps(dict(sorted(allp.items())), indent=1) + "\n")
        VALUES_DIR.mkdir(parents=True, exist_ok=True)
        (VALUES_DIR / f"{self.figure}.json").write_text(
            json.dumps(self.values, indent=1, default=float) + "\n")
        if self._legend is not None:
            LEGEND_DIR.mkdir(parents=True, exist_ok=True)
            (LEGEND_DIR / f"{self.figure}.md").write_text(self._legend.strip() + "\n")


def f2(x, nd=2):
    """Format a number to nd decimals (half away from zero, as quoted in text)."""
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(1).scaleb(-nd)
    out = str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))
    if out.startswith("-") and Decimal(out) == 0:
        out = out[1:]   # no negative zero
    return out


def rng_str(lo, hi, nd=2, sep="–"):
    a, b = f2(lo, nd), f2(hi, nd)
    if a == b:
        return a.replace("-", "−")
    if a.startswith("-") or b.startswith("-"):
        return f"{a.replace('-', '−')} to {b.replace('-', '−')}"
    return f"{a}{sep}{b}"


def repel_labels(ax, xs, ys, labels, fontsize=6, color="black", pad_pt=1.5, step_pt=3.0, max_r_pt=40, line_color="#888888"):
    """Place point labels without overlapping each other or the points (greedy, deterministic).

    Candidate positions are tried on rings of increasing radius around each point
    (right, left, above, below, diagonals); the first that collides with no placed label,
    no other point and stays inside the axes is used. A thin leader line is drawn when
    the label sits further than ~6 pt from its point.
    """
    from matplotlib.transforms import Bbox
    fig = ax.figure
    ax.set_xlim(ax.get_xlim()); ax.set_ylim(ax.get_ylim())   # freeze limits
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pt = fig.dpi / 72.0
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    axbb = ax.get_window_extent(r)
    rad = 3.0 * pt   # marker half-size used as an obstacle
    obstacles = [Bbox.from_extents(px - rad, py - rad, px + rad, py + rad) for px, py in pts]
    placed = []
    order = np.argsort(-pts[:, 1])  # top to bottom
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]
    texts = []
    for i in order:
        t = ax.text(0, 0, labels[i], fontsize=fontsize, color=color, transform=None)
        bb0 = t.get_window_extent(r)
        w, h = bb0.width, bb0.height
        px, py = pts[i]
        best = None
        radius = 2.0
        while best is None and radius <= max_r_pt:
            for dx, dy in dirs:
                ox, oy = dx * radius * pt, dy * radius * pt
                x0 = px + ox + (0 if dx > 0 else (-w if dx < 0 else -w / 2))
                y0 = py + oy + (0 if dy > 0 else (-h if dy < 0 else -h / 2))
                bb = Bbox.from_extents(x0 - pad_pt, y0 - pad_pt, x0 + w + pad_pt, y0 + h + pad_pt)
                if not (axbb.x0 <= bb.x0 and bb.x1 <= axbb.x1 + 30 * pt and axbb.y0 <= bb.y0 and bb.y1 <= axbb.y1):
                    continue
                if any(bb.overlaps(o) for j, o in enumerate(obstacles) if j != i) or any(bb.overlaps(p) for p in placed):
                    continue
                if bb.overlaps(obstacles[i]):
                    continue
                best = (bb, x0, y0, radius)
                break
            radius += step_pt
        if best is None:   # fall back: right of the point
            x0, y0 = px + 2 * pt, py - h / 2
            best = (Bbox.from_extents(x0, y0, x0 + w, y0 + h), x0, y0, 2.0)
        bb, x0, y0, radius = best
        placed.append(bb)
        xd, yd = ax.transData.inverted().transform((x0, y0))
        t.remove()
        t = ax.text(xd, yd, labels[i], fontsize=fontsize, color=color, ha="left", va="bottom", zorder=4)
        if radius > 6:
            cx, cy = ax.transData.inverted().transform((min(max(px, bb.x0), bb.x1), min(max(py, bb.y0), bb.y1)))
            ax.plot([xs[i], cx], [ys[i], cy], color=line_color, lw=0.5, zorder=2)
        texts.append(t)
    return texts
