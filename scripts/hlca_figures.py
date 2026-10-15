#!/usr/bin/env python3
"""H05 stored-results renderer for the HLCA aging-basin analysis.

This script performs no model fitting and no H04-test rerun.  It validates frozen
upstream checksums, reads committed tables (and the primary drift H5AD only for
the two figures that need stored per-cell metadata/coordinates), writes the H05
figures plus manifest and descriptive mouse–human comparison, then appends the
specified §§8–10 record to the existing H04 RESULT.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from scjdo2.compare import centroid_operator

ROOT = Path(__file__).resolve().parents[1]
HLCA = ROOT / "benchmarks" / "aging_hlca"
FIG = ROOT / "benchmarks" / "figures" / "aging_hlca"
PRIMARY = HLCA / "data" / "h03_primary_drift.h5ad"
TESTS = HLCA / "tests_primary.csv"
SENSITIVITY = HLCA / "tests_sensitivity.csv"
OPERATORS = HLCA / "donor_operators.csv"
ELIGIBILITY = HLCA / "eligibility.csv"
MOUSE_A03 = ROOT / "benchmarks" / "aging" / "compare_pooled.csv"
MOUSE_X04 = ROOT / "benchmarks" / "aging" / "x04" / "summary.csv"
MOUSE_HUMAN = HLCA / "mouse_human.csv"
RESULT = HLCA / "RESULT.md"
MANIFEST = FIG / "MANIFEST.csv"

DPI = 200
FOOTER = "PREREG_aging_hlca · hlca/H04 frozen tests · scArches young-only reference · d_latent=15 · units relative"
DESCRIPTIVE_FOOTER = "descriptive comparison; no statistic (PREREG_aging_hlca §8)"
PAIRS = (("fibroblast of lung", "Fibroblasts"), ("natural killer cell", "Innate lymphoid cell NK"))
PRIMARY_TYPES = ("AT2", "Basal", "Dendritic cells", "EC arterial", "EC capillary", "EC venous", "Fibroblasts", "Innate lymphoid cell NK", "Macrophages", "Monocytes", "Multiciliated lineage", "Secretory", "T cell lineage")
ANALYSES = ("primary", "a_within_study", "b_banovich", "c_parenchyma", "d_never_smokers", "e_secondary_space", "f_two_group")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_report_inputs(report: str, paths: tuple[str, ...]) -> None:
    document = json.loads((ROOT / report).read_text())
    artifacts = {entry["path"]: entry for entry in document["artifacts"]}
    for rel in paths:
        item = artifacts.get(rel)
        if item is None:
            raise RuntimeError(f"{report} does not inventory required H05 input {rel}")
        path = ROOT / rel
        if not path.exists() or path.stat().st_size != int(item["bytes"]) or sha256(path) != item["sha256"]:
            raise RuntimeError(f"checksum mismatch for {rel} against {report}")


def verify_inputs() -> None:
    verify_report_inputs("coord/hlca/reports/H03.json", (
        "benchmarks/aging_hlca/data/h03_primary_drift.h5ad",
        "benchmarks/aging_hlca/eligibility.csv",
        "benchmarks/aging_hlca/qc/hlca_scvi_metrics.csv",
        "benchmarks/aging_hlca/qc/scvi_history_reference.csv",
        "benchmarks/aging_hlca/qc/scvi_history_query.csv",
        "benchmarks/aging_hlca/qc/hlca_drift_metrics.csv",
    ))
    verify_report_inputs("coord/hlca/reports/H04.json", (
        "benchmarks/aging_hlca/tests_primary.csv",
        "benchmarks/aging_hlca/tests_sensitivity.csv",
        "benchmarks/aging_hlca/donor_operators.csv",
    ))
    verify_report_inputs("coord/aging/reports/A03.json", ("benchmarks/aging/compare_pooled.csv",))
    verify_report_inputs("coord/aging/reports/X04.json", ("benchmarks/aging/x04/summary.csv",))


def add_footer(fig: plt.Figure, extra: str | None = None) -> None:
    if extra:
        fig.text(0.5, 0.032, extra, ha="center", va="bottom", fontsize=6.2, color="0.27")
    fig.text(0.5, 0.012, FOOTER, ha="center", va="bottom", fontsize=5.7, color="0.38")


def save(fig: plt.Figure, name: str, source_files: tuple[str, ...], manifest: list[dict[str, str]], extra_footer: str | None = None) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    add_footer(fig, extra_footer)
    fig.tight_layout(rect=(0, 0.06 if extra_footer else 0.04, 1, 0.97))
    for suffix in (".pdf", ".png"):
        path = FIG / f"{name}{suffix}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        manifest.append({"path": str(path.relative_to(ROOT)), "figure": name, "source_files": ";".join(source_files)})
    plt.close(fig)


def truth(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def call_text(row: pd.Series) -> str:
    iso = "I+" if truth(row["iso_pass"]) else "I−"
    ant = "A+" if truth(row["ant_pass"]) else "A−"
    stiff = {"loss": "S loss", "gain": "S gain"}.get(str(row["stiff_class"]), "S—")
    return f"{iso}\n{ant}\n{stiff}"


def plot_tests_overview(primary: pd.DataFrame) -> plt.Figure:
    labels = primary["ann_level_3"].astype(str).tolist()
    x = np.arange(len(primary))
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    for ax, observed, null, title, ylabel, passed in (
        (axes[0], "iso_S", "iso_q95", "Isotropy: S compared with the stored 95% permutation threshold", "S", primary["iso_pass"].map(truth)),
        (axes[1], "ant_stat", "ant_q95", "Anticipation: alignment statistic compared with the stored 95% threshold", "|cos|", primary["ant_pass"].map(truth)),
    ):
        colors = np.where(passed.to_numpy(), "#2A9D8F", "#9AA5B1")
        ax.bar(x - 0.18, primary[observed], width=0.36, color=colors, label="observed")
        ax.bar(x + 0.18, primary[null], width=0.36, color="#E9C46A", label="permutation q95")
        for index, flag in enumerate(passed):
            if flag:
                ax.text(index - .18, float(primary.iloc[index][observed]), "★", ha="center", va="bottom", fontsize=9, color="#0B5D52")
        ax.set_ylabel(ylabel); ax.set_title(title, loc="left", fontsize=10); ax.grid(axis="y", color="0.9", linewidth=.7); ax.legend(fontsize=7, ncol=2)
    slopes = primary["stiff_slope"].to_numpy(float)
    colors = np.where(primary["stiff_class"].eq("loss"), "#D85A30", np.where(primary["stiff_class"].eq("gain"), "#8E6BBE", "#9AA5B1"))
    axes[2].bar(x, slopes, color=colors)
    axes[2].axhline(0, color="0.25", linewidth=.8)
    axes[2].set_title("Stiffness: stored slope; H04 did not retain a permutation-q95 band", loc="left", fontsize=10)
    axes[2].set_ylabel("slope of log median |λ| / age"); axes[2].grid(axis="y", color="0.9", linewidth=.7)
    axes[2].text(.995, .96, "orange = stiffness loss call", ha="right", va="top", transform=axes[2].transAxes, fontsize=7, color="0.35")
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    fig.suptitle("H04 primary donor-level tests (13 frozen-eligible types)", x=.01, ha="left", fontsize=13, fontweight="bold")
    return fig


def plot_sensitivity_heatmap(primary: pd.DataFrame, sensitivity: pd.DataFrame) -> plt.Figure:
    frame = pd.concat([primary.assign(analysis="primary"), sensitivity], ignore_index=True)
    codes = np.zeros((len(PRIMARY_TYPES), len(ANALYSES)), dtype=int)
    text = np.full(codes.shape, "NA", dtype=object)
    for i, cell_type in enumerate(PRIMARY_TYPES):
        for j, analysis in enumerate(ANALYSES):
            rows = frame[(frame["analysis"].eq(analysis)) & (frame["ann_level_3"].eq(cell_type))]
            if rows.empty:
                continue
            row = rows.iloc[0]
            has_iso, has_ant = truth(row["iso_pass"]), truth(row["ant_pass"])
            stiff = str(row["stiff_class"])
            code = 1 + int(has_iso) + 2 * int(has_ant) + (4 if stiff == "loss" else 5 if stiff == "gain" else 0)
            codes[i, j] = min(code, 7)
            text[i, j] = call_text(row)
    cmap = ListedColormap(["#D4D8DD", "#F4F5F6", "#94C9E8", "#8CD4C8", "#E9B28E", "#B39DDB", "#B0709F", "#6B8E23"])
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(codes, cmap=cmap, vmin=0, vmax=7, aspect="auto")
    for i in range(codes.shape[0]):
        for j in range(codes.shape[1]):
            ax.text(j, i, text[i, j], ha="center", va="center", fontsize=6.6, color="#17212B")
    ax.set_xticks(range(len(ANALYSES)), ["primary", "a", "b", "c", "d", "e", "f"], fontsize=9)
    ax.set_yticks(range(len(PRIMARY_TYPES)), PRIMARY_TYPES, fontsize=8)
    ax.set_title("Sensitivity calls by type\ncell text: isotropy / anticipation / stiffness; grey = ineligible", loc="left", fontsize=11)
    ax.set_xlabel("primary and §7 sensitivity analyses")
    fig.text(.01, .955, "I+ / I− = isotropy pass/fail · A+ / A− = anticipation pass/fail · S loss / S gain / S— = stiffness call", ha="left", fontsize=7, color="0.32")
    return fig


def plot_stiffness_vs_age(operators: pd.DataFrame) -> plt.Figure:
    frame = operators[operators["analysis"].eq("primary")].copy()
    studies = sorted(frame["study_or_dataset"].astype(str).unique())
    palette = {study: plt.cm.tab20(index % 20) for index, study in enumerate(studies)}
    fig, axes = plt.subplots(4, 4, figsize=(16, 13), squeeze=False)
    for ax, cell_type in zip(axes.ravel(), PRIMARY_TYPES, strict=False):
        rows = frame[frame["ann_level_3"].eq(cell_type)]
        for study, group in rows.groupby("study_or_dataset", sort=True):
            ax.scatter(group["age"], group["log_median_abs_lambda"], s=16, alpha=.82, color=palette[str(study)], label=str(study))
        slope, intercept = np.polyfit(rows["age"], rows["log_median_abs_lambda"], 1)
        grid = np.linspace(rows["age"].min(), rows["age"].max(), 100)
        ax.plot(grid, intercept + slope * grid, color="#222222", linewidth=1.0)
        suffix = " — lead (H04 review)" if cell_type in {"T cell lineage", "Monocytes"} else ""
        ax.set_title(cell_type + suffix, fontsize=8.5, loc="left")
        ax.grid(color="0.92", linewidth=.6); ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(PRIMARY_TYPES):]:
        ax.axis("off")
    axes[3, 0].set_xlabel("age"); axes[3, 1].set_xlabel("age"); axes[3, 2].set_xlabel("age")
    axes[0, 0].set_ylabel("log median |λ|"); axes[1, 0].set_ylabel("log median |λ|"); axes[2, 0].set_ylabel("log median |λ|")
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=5, color=palette[s], label=s) for s in studies]
    axes[3, 1].legend(handles=handles, fontsize=5.3, ncol=2, loc="center", title="study_or_dataset", title_fontsize=6)
    fig.suptitle("Primary donor operators: stiffness against age", x=.01, ha="left", fontsize=13, fontweight="bold")
    return fig


def parse_centroid(value: str) -> np.ndarray:
    vector = np.array([float(item) for item in str(value).split(";")], dtype=float)
    if vector.shape != (15,):
        raise RuntimeError(f"expected 15-D primary donor centroid, got {vector.shape}")
    return vector


def plot_ec_capillary_anticipation(operators: pd.DataFrame) -> plt.Figure:
    adata = ad.read_h5ad(PRIMARY)
    z = np.asarray(adata.obsm["X_scVI_primary"], dtype=float)
    j = np.asarray(adata.uns["scjdo2"]["J_latent"], dtype=float)
    obs = adata.obs
    ctype = obs["ann_level_3"].astype(str).to_numpy()
    age = obs["age"].to_numpy(float)
    young = (ctype == "EC capillary") & (age < 45)
    reference = centroid_operator(z[young], j[young], n_cells_centroid=30)
    values, vectors = np.linalg.eigh(0.5 * (reference + reference.T))
    soft_axis = vectors[:, int(np.argmax(values))]
    frame = operators[(operators["analysis"].eq("primary")) & (operators["ann_level_3"].eq("EC capillary"))].copy()
    frame["projection"] = [float(parse_centroid(item) @ soft_axis) for item in frame["centroid"]]
    studies = sorted(frame["study_or_dataset"].astype(str).unique())
    palette = {study: plt.cm.tab20(index % 20) for index, study in enumerate(studies)}
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for study, group in frame.groupby("study_or_dataset", sort=True):
        ax.scatter(group["age"], group["projection"], label=str(study), color=palette[str(study)], s=34, alpha=.85)
    slope, intercept = np.polyfit(frame["age"], frame["projection"], 1)
    grid = np.linspace(frame["age"].min(), frame["age"].max(), 100)
    ax.plot(grid, intercept + slope * grid, color="#222", linewidth=1.2, label="OLS")
    ax.axhline(0, color="0.7", linewidth=.7)
    ax.set_title("EC capillary: donor centroid projection on the young reference soft axis", loc="left", fontsize=11)
    ax.set_xlabel("age"); ax.set_ylabel("centroid projection (relative units)"); ax.grid(color="0.9", linewidth=.7)
    ax.legend(fontsize=6.6, ncol=2, frameon=False, title="study_or_dataset", title_fontsize=7)
    return fig


def shared_seed_call(minimum: float, maximum: float) -> str:
    if float(minimum) == 1.0 and float(maximum) == 1.0:
        return "pass"
    if float(minimum) == 0.0 and float(maximum) == 0.0:
        return "fail"
    return "mixed"


def mouse_human_table(primary: pd.DataFrame, sensitivity: pd.DataFrame) -> pd.DataFrame:
    a03 = pd.read_csv(MOUSE_A03)
    x04 = pd.read_csv(MOUSE_X04)
    rows: list[dict] = []
    for mouse_type, human_type in PAIRS:
        mouse = a03[(a03["tissue"].eq("lung")) & (a03["cell_type"].eq(mouse_type))]
        mapped = x04[(x04["space"].eq("shared")) & (x04["tissue"].eq("lung")) & (x04["cell_type"].eq(mouse_type))]
        human_primary = primary[primary["ann_level_3"].eq(human_type)]
        human_two = sensitivity[(sensitivity["analysis"].eq("f_two_group")) & (sensitivity["ann_level_3"].eq(human_type))]
        if not (len(mouse) == len(mapped) == len(human_primary) == len(human_two) == 1):
            raise RuntimeError(f"missing exactly-one descriptive comparison row for {mouse_type}/{human_type}")
        mouse_row, map_row, hp, h2 = mouse.iloc[0], mapped.iloc[0], human_primary.iloc[0], human_two.iloc[0]
        rows.append({
            "mouse_tissue": "lung", "mouse_cell_type": mouse_type, "human_ann_level_3": human_type,
            "mouse_a03_stiffness_ratio": float(mouse_row["median_abs_lambda_old"] / mouse_row["median_abs_lambda_young"]),
            "mouse_x04_shared_stiffness_min": float(map_row["stiff_ratio_min"]),
            "mouse_x04_shared_stiffness_max": float(map_row["stiff_ratio_max"]),
            "human_two_group_stiffness_ratio": float(np.exp(h2["stiff_slope"])),
            "mouse_a03_iso_pass": bool(mouse_row["iso_pass"]), "mouse_a03_ant_pass": bool(mouse_row["ant_pass"]),
            "mouse_x04_shared_iso_call": shared_seed_call(map_row["iso_pass_min"], map_row["iso_pass_max"]),
            "mouse_x04_shared_ant_call": shared_seed_call(map_row["ant_pass_min"], map_row["ant_pass_max"]),
            "human_primary_iso_pass": bool(hp["iso_pass"]), "human_primary_ant_pass": bool(hp["ant_pass"]),
            "human_two_group_iso_pass": bool(h2["iso_pass"]), "human_two_group_ant_pass": bool(h2["ant_pass"]),
        })
    result = pd.DataFrame(rows)
    if len(result) != 2:
        raise RuntimeError("PREREG §8 requires exactly two mouse–human rows")
    return result


def plot_mouse_human(table: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
    for ax, row in zip(axes, table.itertuples(index=False), strict=True):
        mouse = float(row.mouse_a03_stiffness_ratio)
        lo, hi = float(row.mouse_x04_shared_stiffness_min), float(row.mouse_x04_shared_stiffness_max)
        human = float(row.human_two_group_stiffness_ratio)
        ax.errorbar([0], [mouse], yerr=[[mouse - lo], [hi - mouse]], fmt="o", color="#378ADD", capsize=5, label="mouse: A03; X04 shared range")
        ax.scatter([1], [human], marker="s", s=55, color="#D85A30", label="human: H04 §7f exp(slope)")
        ax.axhline(1, color="0.45", linewidth=.8, linestyle="--")
        ax.set_xticks([0, 1], ["mouse lung", "human HLCA"])
        ax.set_ylim(min(.85, lo, human) * .95, max(1.2, hi, human) * 1.06)
        ax.set_title(f"{row.mouse_cell_type}\n↔ {row.human_ann_level_3}", fontsize=10)
        ax.set_ylabel("old / young stiffness ratio")
        mouse_calls = f"A03 / X04 shared\nisotropy: {'pass' if row.mouse_a03_iso_pass else 'fail'} / {row.mouse_x04_shared_iso_call}\nanticipation: {'pass' if row.mouse_a03_ant_pass else 'fail'} / {row.mouse_x04_shared_ant_call}"
        human_calls = f"H04 primary / §7f\nisotropy: {'pass' if row.human_primary_iso_pass else 'fail'} / {'pass' if row.human_two_group_iso_pass else 'fail'}\nanticipation: {'pass' if row.human_primary_ant_pass else 'fail'} / {'pass' if row.human_two_group_ant_pass else 'fail'}"
        ax.text(.03, .035, mouse_calls, transform=ax.transAxes, ha="left", va="bottom", fontsize=7)
        ax.text(.97, .035, human_calls, transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
        ax.grid(axis="y", color="0.9")
    axes[0].legend(fontsize=7, loc="upper left", frameon=False)
    fig.suptitle("Pre-registered mouse–human comparison", x=.01, ha="left", fontsize=13, fontweight="bold")
    return fig


def plot_scvi_history() -> plt.Figure:
    ref = pd.read_csv(HLCA / "qc" / "scvi_history_reference.csv")
    query = pd.read_csv(HLCA / "qc" / "scvi_history_query.csv")
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(ref["epoch"], ref["elbo_validation"], color="#378ADD", label="young reference")
    ax.plot(query["epoch"], query["elbo_validation"], color="#D85A30", label="older query")
    ax.set_xlabel("epoch"); ax.set_ylabel("validation ELBO"); ax.set_title("scVI validation history", loc="left"); ax.grid(color="0.9"); ax.legend(frameon=False)
    return fig


def plot_query_vs_ref_elbo() -> plt.Figure:
    metrics = pd.read_csv(HLCA / "qc" / "hlca_scvi_metrics.csv")
    values = metrics["best_validation_elbo"].to_numpy(float)
    labels = metrics["fit"].astype(str).str.replace("_", " ").tolist()
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    bars = ax.bar(labels, values, color=["#378ADD", "#D85A30"])
    for bar, value in zip(bars, values, strict=True): ax.text(bar.get_x()+bar.get_width()/2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("best validation ELBO"); ax.set_title("Final validation ELBO", loc="left"); ax.grid(axis="y", color="0.9")
    return fig


def plot_symmetry() -> plt.Figure:
    metrics = pd.read_csv(HLCA / "qc" / "hlca_drift_metrics.csv")
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    bars = ax.bar(metrics["fit"], metrics["sym_max"], color=["#378ADD", "#D85A30"])
    ax.axhline(1e-4, color="#D85A30", linestyle="--", linewidth=1, label="H03 gate 1e-4")
    ax.set_yscale("log"); ax.set_ylabel("maximum |J − Jᵀ|"); ax.set_title("Stored energy-drift symmetry", loc="left"); ax.legend(fontsize=7, frameon=False); ax.grid(axis="y", color="0.9")
    for bar, value in zip(bars, metrics["sym_max"], strict=True): ax.text(bar.get_x()+bar.get_width()/2, value*1.5, f"{value:.2e}", ha="center", fontsize=8)
    return fig


def plot_eligibility_counts() -> plt.Figure:
    table = pd.read_csv(ELIGIBILITY)
    counts = table[table["eligible"]].groupby("analysis", sort=False).size().reindex(["primary", "banovich", "parenchyma", "never_smokers"])
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    bars = ax.bar(counts.index.str.replace("_", " "), counts.values, color="#2A9D8F")
    for bar, value in zip(bars, counts.values, strict=True): ax.text(bar.get_x()+bar.get_width()/2, value+.15, str(value), ha="center", fontsize=9)
    ax.set_ylabel("eligible types"); ax.set_ylim(0, max(counts.values)+2); ax.set_title("H03/H04 eligibility counts", loc="left"); ax.grid(axis="y", color="0.9")
    return fig


def plot_donor_composition() -> plt.Figure:
    adata = ad.read_h5ad(PRIMARY, backed="r")
    try:
        donor = adata.obs[["donor_id", "age", "study_or_dataset"]].copy()
    finally:
        adata.file.close()
    donor["donor_id"] = donor["donor_id"].astype(str)
    donor = donor.drop_duplicates("donor_id")
    donor["age_bin"] = pd.cut(donor["age"], bins=[-np.inf, 45, 60, np.inf], labels=["<45", "45–59", "≥60"], right=False)
    table = pd.crosstab(donor["study_or_dataset"].astype(str), donor["age_bin"]).reindex(columns=["<45", "45–59", "≥60"], fill_value=0)
    table = table.loc[table.sum(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(10.2, 6.1))
    colors = ["#378ADD", "#E9C46A", "#D85A30"]
    left = np.zeros(len(table))
    for label, color in zip(table.columns, colors, strict=True):
        ax.barh(table.index, table[label], left=left, label=str(label), color=color)
        left += table[label].to_numpy()
    ax.invert_yaxis(); ax.set_xlabel("unique donors"); ax.set_title("HLCA donor composition by age bin and study", loc="left"); ax.legend(title="age", fontsize=8, frameon=False); ax.grid(axis="x", color="0.9")
    return fig


def append_result(table: pd.DataFrame, manifest: pd.DataFrame) -> None:
    existing = RESULT.read_text()
    marker = "\n## §8 Mouse–human comparison"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n"
    outcome = "**isotropic noise increase; no aging claim beyond stiffness** — no majority stiffness change"
    if outcome not in existing:
        raise RuntimeError("H04 outcome line is absent or differs; refusing to change RESULT.md")
    display = table.copy()
    for col in ["mouse_a03_stiffness_ratio", "mouse_x04_shared_stiffness_min", "mouse_x04_shared_stiffness_max", "human_two_group_stiffness_ratio"]:
        display[col] = display[col].map(lambda v: f"{float(v):.4f}")
    cols = ["mouse_cell_type", "human_ann_level_3", "mouse_a03_stiffness_ratio", "mouse_x04_shared_stiffness_min", "mouse_x04_shared_stiffness_max", "human_two_group_stiffness_ratio", "mouse_a03_iso_pass", "mouse_a03_ant_pass", "human_two_group_iso_pass", "human_two_group_ant_pass"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"]*len(cols)) + "|"]
    for row in display[cols].itertuples(index=False, name=None): lines.append("| " + " | ".join(map(str, row)) + " |")
    paths = "\n".join(f"- `{row.path}` — {row.figure}" for row in manifest.itertuples(index=False))
    limits = """* The frozen outcome reports no direction-specific (0/13) and no
  majority anticipatory (1/13) change with age in human lung operators,
  and no majority change in stiffness (1/13 softening).
* The single stiffness hit (T cell lineage) does not hold within
  studies (p = 0.19), in parenchyma (p = 0.08) or in the all-ages HLCA
  space (p = 0.73), and is not significant after correction for 13
  types. The single anticipation hit (EC capillary, p = 0.047) is
  borderline in every analysis.
* 17 donors are aged ≥ 60 and cluster in two studies; age is partly
  confounded with study, assay and sampling site. Cells of older donors
  fit the young-only reference less well (validation ELBO ≈ 655 vs
  ≈ 550). One scVI seed. Effects smaller than about a two-fold change
  of one direction over the age range are not excluded.
* The mouse–human comparison covers two matched cell types and is
  descriptive."""
    existing += "## §8 Mouse–human comparison\n"
    existing += "Descriptive comparison only; no statistic was computed. Mouse stiffness ratio is the stored A03 old/young median-curvature ratio; its X04 range is the stored shared-space seed range. Human ratio is `exp(stiff_slope)` from H04 §7(f).\n\n" + "\n".join(lines) + "\n"
    existing += "\n## §9 Figures\n" + paths + "\n"
    existing += "\n## §10 Limits\n" + limits + "\n"
    RESULT.write_text(existing)


def run() -> dict[str, object]:
    verify_inputs()
    primary = pd.read_csv(TESTS)
    sensitivity = pd.read_csv(SENSITIVITY)
    operators = pd.read_csv(OPERATORS)
    if tuple(primary["ann_level_3"].astype(str)) != PRIMARY_TYPES or len(primary) != 13:
        raise RuntimeError("H04 primary table is not the frozen 13-type table")
    if tuple(sorted(sensitivity["analysis"].unique())) != tuple(sorted(ANALYSES[1:])):
        raise RuntimeError("H04 sensitivity analyses are incomplete")
    manifest: list[dict[str, str]] = []
    save(plot_tests_overview(primary), "tests_overview", ("benchmarks/aging_hlca/tests_primary.csv",), manifest)
    save(plot_sensitivity_heatmap(primary, sensitivity), "sensitivity_heatmap", ("benchmarks/aging_hlca/tests_primary.csv", "benchmarks/aging_hlca/tests_sensitivity.csv"), manifest)
    save(plot_stiffness_vs_age(operators), "stiffness_vs_age", ("benchmarks/aging_hlca/donor_operators.csv",), manifest)
    save(plot_ec_capillary_anticipation(operators), "ec_capillary_anticipation", ("benchmarks/aging_hlca/donor_operators.csv", "benchmarks/aging_hlca/data/h03_primary_drift.h5ad"), manifest)
    comparison = mouse_human_table(primary, sensitivity)
    comparison.to_csv(MOUSE_HUMAN, index=False)
    save(plot_mouse_human(comparison), "mouse_human", ("benchmarks/aging/compare_pooled.csv", "benchmarks/aging/x04/summary.csv", "benchmarks/aging_hlca/tests_primary.csv", "benchmarks/aging_hlca/tests_sensitivity.csv"), manifest, DESCRIPTIVE_FOOTER)
    save(plot_scvi_history(), "qc_scvi_history", ("benchmarks/aging_hlca/qc/scvi_history_reference.csv", "benchmarks/aging_hlca/qc/scvi_history_query.csv"), manifest)
    save(plot_query_vs_ref_elbo(), "qc_query_vs_ref_elbo", ("benchmarks/aging_hlca/qc/hlca_scvi_metrics.csv",), manifest)
    save(plot_symmetry(), "qc_symmetry", ("benchmarks/aging_hlca/qc/hlca_drift_metrics.csv",), manifest)
    save(plot_eligibility_counts(), "qc_eligibility_counts", ("benchmarks/aging_hlca/eligibility.csv",), manifest)
    save(plot_donor_composition(), "qc_donor_composition", ("benchmarks/aging_hlca/data/h03_primary_drift.h5ad",), manifest)
    manifest_frame = pd.DataFrame(manifest, columns=["path", "figure", "source_files"])
    manifest_frame.to_csv(MANIFEST, index=False)
    append_result(comparison, manifest_frame)
    payload = {"n_figures": int(manifest_frame["figure"].nunique()), "n_skipped": 0, "mouse_human_rows": int(len(comparison)), "manifest_rows": int(len(manifest_frame))}
    print(json.dumps(payload, sort_keys=True), flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
