#!/usr/bin/env python3
"""X04: exploratory shared-vs-joint per-type quantities, fixed reading rule, and figures."""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scjdo2.compare import centroid_operator, compare_types
from x04_common import AGING, OUT, verify_reported

TISSUES = ("lung", "marrow")
N_CENTROID = 30
N_NULL = 1000
MIN_CELLS = 200
EPS = 1e-12
FOOTER = "EXPLORATORY · aging/X04 · not part of the frozen outcome"
BASINS = {("lung", "fibroblast of lung"), ("lung", "natural killer cell"), ("marrow", "granulocyte")}


def source_path(space: str, tissue: str, seed: int) -> tuple[Path, str]:
    if space == "shared" and seed == 0:
        return AGING / f"{tissue}_shared.h5ad", "X_scVI_shared"
    return OUT / f"{tissue}_{space}_s{seed}.h5ad", f"X_scVI_{space}"


def fit_metadata(adata: ad.AnnData) -> dict:
    return dict(adata.uns.get("x04_drift", adata.uns.get("a02_drift", {})))


def type_rows(tissue: str, space: str, seed: int) -> tuple[list[dict], dict]:
    path, representation = source_path(space, tissue, seed)
    if not path.exists():
        raise FileNotFoundError(path)
    adata = ad.read_h5ad(path)
    z = np.asarray(adata.obsm[representation], dtype=np.float64)
    J = np.asarray(adata.uns["scjdo2"].get("J_latent", adata.uns["scjdo2"]["J"]), dtype=np.float64)
    tests = compare_types(adata, by="cell_type", groups=("young", "old"), group_key="age_group",
                          use_rep=representation, n_cells_centroid=N_CENTROID, min_cells=MIN_CELLS,
                          n_null=N_NULL, seed=0).set_index("cell_type")
    cell_type = adata.obs["cell_type"].astype(str).to_numpy()
    age = adata.obs["age_group"].astype(str).to_numpy()
    holder = adata.obs["op_behaviour_basin"].astype(str).eq("holder").to_numpy()
    rows: list[dict] = []
    for ct in sorted(tests.index):
        young = (cell_type == ct) & (age == "young")
        old = (cell_type == ct) & (age == "old")
        Jy = 0.5 * (centroid_operator(z[young], J[young], N_CENTROID) + centroid_operator(z[young], J[young], N_CENTROID).T)
        Jo = 0.5 * (centroid_operator(z[old], J[old], N_CENTROID) + centroid_operator(z[old], J[old], N_CENTROID).T)
        ey, eo = np.linalg.eigvalsh(Jy), np.linalg.eigvalsh(Jo)
        med_y, med_o = float(np.median(np.abs(ey))), float(np.median(np.abs(eo)))
        aniso_y = med_y / (abs(float(ey.max())) + EPS)
        aniso_o = med_o / (abs(float(eo.max())) + EPS)
        cov_y = np.cov(z[young], rowvar=False); cov_o = np.cov(z[old], rowvar=False)
        trace_y, trace_o = float(np.trace(cov_y)), float(np.trace(cov_o))
        test = tests.loc[ct]
        rows.append({
            "space": space, "seed": int(seed), "tissue": tissue, "cell_type": ct,
            "n_young": int(young.sum()), "n_old": int(old.sum()),
            "stiff_ratio": med_o / (med_y + EPS), "aniso_ratio": aniso_o / (aniso_y + EPS),
            "spread_ratio": trace_o / (trace_y + EPS),
            "offset_z": float(np.linalg.norm(z[old].mean(0) - z[young].mean(0)) / np.sqrt(trace_y + EPS)),
            "basin_fraction_young": float(holder[young].mean()), "basin_fraction_old": float(holder[old].mean()),
            "S": float(test.S), "iso_null_q95": float(test.iso_null_q95), "iso_pass": bool(test.iso_pass),
            "cos_abs_u_young_d": float(test.cos_abs_u_young_d), "ant_null_q95": float(test.ant_null_q95), "ant_pass": bool(test.ant_pass),
        })
    meta = fit_metadata(adata)
    return rows, {"tissue": tissue, "space": space, "seed": seed, "path": str(path.relative_to(AGING.parent.parent)),
                  "sym_max": float(meta["symmetry"]["max_asym"]), "drift": meta}


def summarise(per_type: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = ["stiff_ratio", "aniso_ratio", "spread_ratio", "offset_z", "basin_fraction_young", "basin_fraction_old",
               "S", "iso_null_q95", "cos_abs_u_young_d", "ant_null_q95", "iso_pass", "ant_pass"]
    grouped = per_type.groupby(["space", "tissue", "cell_type"], sort=True)[numeric].agg(["mean", "min", "max"])
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    summary = grouped.reset_index()
    flags: list[dict] = []
    for (tissue, cell_type), group in summary.groupby(["tissue", "cell_type"], sort=True):
        shared = group.loc[group.space.eq("shared")].iloc[0]
        joint = group.loc[group.space.eq("joint")].iloc[0]
        shared_range = float(shared.stiff_ratio_max - shared.stiff_ratio_min)
        joint_range = float(joint.stiff_ratio_max - joint.stiff_ratio_min)
        no_overlap = bool(joint.stiff_ratio_max < shared.stiff_ratio_min or shared.stiff_ratio_max < joint.stiff_ratio_min)
        hidden_soft = bool(joint.stiff_ratio_mean <= 0.90 and shared.stiff_ratio_mean >= 0.95 and no_overlap)
        shared_offset_range = float(shared.offset_z_max - shared.offset_z_min)
        joint_offset_range = float(joint.offset_z_max - joint.offset_z_min)
        hidden_displacement = bool(joint.offset_z_mean - shared.offset_z_mean > max(shared_offset_range, joint_offset_range))
        flags.append({"tissue": tissue, "cell_type": cell_type, "basin_type": (tissue, cell_type) in BASINS,
                      "hidden_softening": hidden_soft, "hidden_displacement": hidden_displacement,
                      "shows_hidden_change": bool(hidden_soft or hidden_displacement),
                      "shared_stiff_range": shared_range, "joint_stiff_range": joint_range,
                      "shared_offset_range": shared_offset_range, "joint_offset_range": joint_offset_range})
    flag_frame = pd.DataFrame(flags)
    summary = summary.merge(flag_frame, on=["tissue", "cell_type"], how="left").sort_values(["space", "tissue", "cell_type"])
    return summary, flag_frame


def make_figure(summary: pd.DataFrame, metric: str, ylabel: str, path: Path, lines: list[tuple[float, str, str]]) -> None:
    types = (summary[["tissue", "cell_type"]].drop_duplicates().sort_values(["tissue", "cell_type"]))
    labels = [f"{r.tissue}\n{r.cell_type}" for r in types.itertuples(index=False)]
    x = np.arange(len(types), dtype=float)
    fig, ax = plt.subplots(figsize=(max(10, 1.25 * len(types)), 5.2))
    for space, color, shift in (("shared", "#378ADD", -0.16), ("joint", "#D85A30", 0.16)):
        data = summary[summary.space.eq(space)].set_index(["tissue", "cell_type"])
        ordered = [data.loc[(r.tissue, r.cell_type)] for r in types.itertuples(index=False)]
        mean = np.array([r[f"{metric}_mean"] for r in ordered], float)
        low = mean - np.array([r[f"{metric}_min"] for r in ordered], float)
        high = np.array([r[f"{metric}_max"] for r in ordered], float) - mean
        ax.errorbar(x + shift, mean, yerr=np.vstack([low, high]), fmt="o", color=color, capsize=3, label=space)
    for y, style, label in lines:
        ax.axhline(y, color="0.35", lw=0.8, ls=style, label=label)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=8)
    ax.set_ylabel(ylabel); ax.set_title(f"X04 exploratory: {ylabel}, seed mean and min–max")
    ax.legend(fontsize=8, ncol=4); ax.grid(axis="y", color="0.9", lw=0.6)
    fig.text(0.5, 0.008, FOOTER, ha="center", va="bottom", fontsize=6, color="0.35")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame[columns].itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> int:
    verify_reported("A02", ["benchmarks/aging/lung_shared.h5ad", "benchmarks/aging/marrow_shared.h5ad"])
    rows: list[dict] = []; fits: list[dict] = []
    for tissue in TISSUES:
        for space in ("shared", "joint"):
            for seed in (0, 1, 2):
                current, meta = type_rows(tissue, space, seed)
                rows.extend(current); fits.append(meta)
    per_type = pd.DataFrame(rows).sort_values(["space", "seed", "tissue", "cell_type"])
    OUT.mkdir(parents=True, exist_ok=True)
    per_type.to_csv(OUT / "per_type.csv", index=False)
    summary, flags = summarise(per_type)
    summary.to_csv(OUT / "summary.csv", index=False)
    n_soft = int(flags.hidden_softening.sum()); n_disp = int(flags.hidden_displacement.sum())
    basin_flagged = flags[flags.basin_type & flags.shows_hidden_change][["tissue", "cell_type"]]
    verdict = ("mapping hides change" if int((flags.basin_type & flags.shows_hidden_change).sum()) >= 2 or int(flags.shows_hidden_change.sum()) >= 4
               else "no evidence that mapping hides change")
    max_sym = float(max(entry["sym_max"] for entry in fits))
    shared_audits = []
    for tissue in TISSUES:
        for seed in (1, 2):
            a = ad.read_h5ad(OUT / f"{tissue}_shared_s{seed}.h5ad")
            shared_audits.append(float(a.uns["x04_scvi"]["reference_encoder_weights_max_abs_change"]))
    metrics = {
        "x04_verdict": verdict, "n_hidden_softening": n_soft, "n_hidden_displacement": n_disp,
        "basin_types_flagged": basin_flagged.to_dict("records"),
        "joint_stiff_ratio_by_type": {f"{r.tissue}|{r.cell_type}": float(r.stiff_ratio_mean) for r in summary[summary.space.eq("joint")].itertuples(index=False)},
        "shared_stiff_ratio_by_type": {f"{r.tissue}|{r.cell_type}": float(r.stiff_ratio_mean) for r in summary[summary.space.eq("shared")].itertuples(index=False)},
        "seed_range_max_stiff_ratio": float((summary.stiff_ratio_max - summary.stiff_ratio_min).max()),
        "shared_seed_ref_weights_max_abs_change": float(max(shared_audits)), "sym_max_all_fits": max_sym,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    figures = AGING.parent / "figures" / "aging" / "x04"
    make_figure(summary, "stiff_ratio", "old / young median |λ|", figures / "stiff_ratio", [(0.90, "--", "joint hidden-softening threshold"), (0.95, ":", "shared threshold")])
    make_figure(summary, "offset_z", "offset z", figures / "offset_z", [])
    rule = '''* the type **shows hidden softening** if joint `stiff_ratio` ≤ 0.90 **and** shared `stiff_ratio` ≥ 0.95 **and** the joint and shared seed ranges do not overlap;
* **shows hidden displacement** if joint `offset_z` exceeds the shared `offset_z` by more than the larger of the two seed ranges;
* verdict for the arm: **"mapping hides change"** if ≥ 2 of the 3 basin types (lung fibroblast, lung NK, marrow granulocyte) or ≥ 4 of all 8 types show hidden softening or hidden displacement; otherwise **"no evidence that mapping hides change"**.'''
    display = summary[summary.space.eq("joint")].merge(summary[summary.space.eq("shared")][["tissue", "cell_type", "stiff_ratio_mean", "offset_z_mean"]], on=["tissue", "cell_type"], suffixes=("_joint", "_shared"))
    result = "# EXPLORATORY — aging/X04 mapping-shrinkage check\n\n"
    result += "## Question\nDoes a joint young+old scVI space expose age change that young-reference scArches mapping suppresses? This exploratory analysis does not modify the frozen A03 outcome.\n\n"
    result += "## What ran\nThree CPU joint scVI seeds (0–2), shared-mapping seed replicates (1–2; A02 seed 0 reused), and the locked 1,500-epoch seed-0 exact-ESM drift fit for every new latent. Per-type quantities used 30-cell centroid operators and `compare_types(..., n_null=1000, seed=0)`.\n\n"
    result += "## Reading rule (fixed before the run)\n" + rule + "\n\n"
    result += "## Per-type seed means and flags\n" + markdown_table(display, ["tissue", "cell_type", "stiff_ratio_mean_joint", "stiff_ratio_mean_shared", "offset_z_mean_joint", "offset_z_mean_shared", "hidden_softening", "hidden_displacement", "shows_hidden_change"]) + "\n\n"
    result += f"## Mechanical verdict\n**{verdict}**\n\nNo interpretation beyond the fixed reading rule is made here.\n"
    (OUT / "RESULT_X04.md").write_text(result)
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
