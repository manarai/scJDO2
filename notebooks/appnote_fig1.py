"""Application Note, Figure 1: the scjdo2.flow pipeline and its evaluation.

Reads committed result files only (S14 held-out test, S14 lineage addendum, S20 scaling)
and writes manuscript/figures/appnote/Fig1.{pdf,png,tiff} (600 dpi).

    python notebooks/appnote_fig1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _style import *  # noqa: E402,F401,F403
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

B = REPO / "benchmarks"
test = pd.read_csv(B / "synthetic_multimodal/s14/test.csv")
add = json.loads((B / "synthetic_multimodal/s14/addendum_lineage.json").read_text())
scal = pd.read_csv(B / "scaling/scaling.csv")

fig = plt.figure(figsize=(DOUBLE_COL * 0.97, 3.7))
gs = fig.add_gridspec(2, 4, height_ratios=[0.62, 1.0], hspace=0.55, wspace=0.75,
                      width_ratios=[1.15, 1.2, 0.75, 1.0], left=0.08, right=0.99, top=0.98, bottom=0.17)

# a: pipeline -------------------------------------------------------------------
ax = fig.add_subplot(gs[0, :]); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 30)
steps = [("Time course\nPCA of log-normalised\ncounts, sampling times", "#F7F7F7"),
         ("Entropic OT between\nconsecutive times\n(dense or mini-batch)", "#D6E6F4"),
         ("Barycentric velocity\n(barycentre − z) / Δt", "#D6E6F4"),
         ("Smooth drift f(z)\n(MLP fitted to velocity)", "#D6E6F4"),
         ("Flow operator\n$J_{flow}=\\partial f/\\partial z$ per cell\nλ₁ and descriptors", "#D6E6F4")]
w, gap, y0, h = 17.2, 2.9, 12, 15
for i, (t, fc) in enumerate(steps):
    x = 0.5 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc=fc,
                                ec=C_FLOW if i else "#333333", lw=0.8))
    ax.text(x + w / 2, y0 + h / 2, t, ha="center", va="center", fontsize=6.2, linespacing=1.25)
    if i:
        ax.add_patch(FancyArrowPatch((x - gap + 0.3, y0 + h / 2), (x - 0.3, y0 + h / 2),
                                     arrowstyle="-|>", mutation_scale=8, lw=0.9, color="#333333"))
ax.text(50, 3.5, 'import scjdo2 as sj;   sj.flow(adata, time_key="day")   →   '
        'adata.obs["flow_lambda1"],  adata.uns["scjdo2"]["J"]',
        ha="center", va="center", fontsize=6.4, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="#FAFAFA", ec="#BBBBBB", lw=0.6))
panel_letter(ax, "a", x=-0.035, y=1.0)

# b: held-out fork AUC ------------------------------------------------------------
ax = fig.add_subplot(gs[1, 0])
ref = test[(test.kind == "synthetic") & (test.configuration_id == "reference") & test.auc_lam1.notna()]
presets = ["v3_transition", "v3_default", "v3_batch", "v3_stress"]
markers = {10: "o", 15: "s", 20: "^"}
for i, p in enumerate(presets):
    r = ref[ref.preset == p]
    for k, m in markers.items():
        v = r[r.k == k].auc_lam1.values
        ax.scatter(i + (k - 15) / 40 + np.linspace(-0.04, 0.04, len(v)), v, marker=m, s=10,
                   color=C_FLOW, lw=0, label=f"k = {k}" if i == 0 else None)
ax.axhline(0.5, color=C_BASE, lw=0.6, ls=":")
ax.set_xticks(range(4)); ax.set_xticklabels([p.replace("v3_", "") for p in presets], rotation=30, ha="right",
                                            rotation_mode="anchor")
ax.set_ylim(0.4, 1.02); ax.set_ylabel("fork AUC of λ₁\n(branch vs committed)")
ax.set_title("held-out synthetic data\n(a-priori settings)", fontsize=6.5)
ax.legend(fontsize=5.5, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.42), handletextpad=0.1)
panel_letter(ax, "b")

# c: LARRY lineage ---------------------------------------------------------------
ax = fig.add_subplot(gs[1, 1])
items = [("drift", add["drift_pooled"]["spearman"], add["drift_pooled"]["null_max"], C_FLOW),
         ("drift,\ncentred", add["drift_centred"]["spearman"], add["drift_centred"]["null_max"], C_FLOW),
         ("mean\nshift", add["mean_shift_pooled"]["spearman"], None, C_BASE),
         ("λ₁ vs\n|displ.|", add["lambda1_vs_actual_magnitude"], None, C_BASE)]
for i, (lab, val, nmax, c) in enumerate(items):
    ax.bar(i, val, width=0.65, color=c)
    if nmax is not None:
        ax.plot([i - 0.38, i + 0.38], [nmax, nmax], color="black", lw=0.8)
ax.axhline(0, color="black", lw=0.5)
ax.set_xticks(range(len(items))); ax.set_xticklabels([x[0] for x in items], fontsize=5.6)
ax.set_ylabel("Spearman with clonal displacement"); ax.set_ylim(-0.1, 0.65)
ax.set_title(f"LARRY lineage, {add['n_cell_barcode_pairs']} pairs\n(line: max of 1,000 nulls)", fontsize=6.5)
panel_letter(ax, "c")

# d: held-out day -------------------------------------------------------------------
ax = fig.add_subplot(gs[1, 2])
he = add["heldout_energy"]
names = [("flow", "flow", C_FLOW), ("mean_displacement", "mean\nshift", C_BASE), ("no_movement", "no\nmove", C_BASE)]
for i, (k, lab, c) in enumerate(names):
    y = he["point"][k]; lo, hi = he["ci95"][k]
    ax.bar(i, y, width=0.65, color=c)
    ax.errorbar(i, y, yerr=[[y - lo], [hi - y]], color="black", lw=0.6, capsize=1.5)
ax.set_xticks(range(3)); ax.set_xticklabels([n[1].replace("\n", " ") for n in names], fontsize=5.4, rotation=35,
                                            ha="right", rotation_mode="anchor")
ax.set_ylabel("energy distance to\nheld-out day 6"); ax.set_ylim(0, 1.0)
ax.set_title("LARRY held-out day\n(95 % bootstrap CI)", fontsize=6.5)
panel_letter(ax, "d")

# e: memory scaling -----------------------------------------------------------------
ax = fig.add_subplot(gs[1, 3])
ot = scal[scal.stage == "ot"]
for solver, c, lab in [("dense", C_BASE, "dense OT"), ("minibatch", C_FLOW, "mini-batch OT")]:
    r = ot[(ot.solver == solver) & (ot.n_per_day != "full")].copy()
    r["n"] = r.n_per_day.astype(int) * 3
    r = r.sort_values("n")
    ax.plot(r.n, r.max_rss_bytes / 1e9, marker="o", ms=2.5, lw=0.9, color=c, label=lab)
full = ot[(ot.solver == "minibatch") & (ot.n_per_day == "full")].iloc[0]
fit = scal[(scal.solver == "minibatch") & (scal.n_per_day == "full") & (scal.stage != "ot")].iloc[0]
ax.scatter([130887], [full.max_rss_bytes / 1e9], color=C_FLOW, marker="*", s=36, zorder=3)
ax.annotate(f"all 130,887 cells:\n{(full.wall_seconds_external + fit.wall_seconds_external) / 60:.0f} min in total",
            (130887, full.max_rss_bytes / 1e9), xytext=(0.04, 0.97), textcoords="axes fraction", fontsize=5.4,
            ha="left", va="top", arrowprops=dict(arrowstyle="-", lw=0.5, color="#555555"))
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(4e3, 2.5e5); ax.set_ylim(0.4, 30)
ax.set_xlabel("cells (3 days)"); ax.set_ylabel("peak memory, OT (GB)")
ax.set_title("scaling (LARRY)", fontsize=6.5)
ax.legend(fontsize=5.2, frameon=False, loc="lower right", handlelength=1.2)
panel_letter(ax, "e")

save(fig, "Fig1", kind="appnote")
