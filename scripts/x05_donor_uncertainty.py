#!/usr/bin/env python3
"""X05: exploratory donor-level uncertainty from stored TMS latent operators.

No model is fitted. Revision 1 equalizes per-mouse cell counts before exact
mouse-label permutations; leave-one-old-mouse-out calls remain A03-exact.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import zlib
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

from scjdo2.compare import centroid_operator, compare_types

ROOT = Path(__file__).resolve().parents[1]
AGING = ROOT / "benchmarks" / "aging"
OUT = AGING / "x05"
FIG = ROOT / "benchmarks" / "figures" / "aging" / "x05"
TISSUES = ("lung", "marrow")
SEEDS = (0, 1, 2)
N_CENTROID = 30
MIN_CELLS = 200
N_NULL = 1000
N_EQUAL_DRAWS = 200
DRAW_SEED = 0
EPS = 1e-12
FOOTER = "EXPLORATORY · aging/X05 · not part of the frozen outcome"
GCP = ("marrow", "granulocytopoietic cell")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_inputs() -> dict[str, dict]:
    """Fail closed unless all task inputs match their accepted reports."""
    expected = {
        "A02": [f"benchmarks/aging/{tissue}_shared.h5ad" for tissue in TISSUES],
        "X04": [
            f"benchmarks/aging/x04/{tissue}_shared_s{seed}.h5ad"
            for tissue in TISSUES for seed in (1, 2)
        ],
    }
    verified: dict[str, dict] = {}
    for report_name, paths in expected.items():
        report = json.loads((ROOT / "coord" / "aging" / "reports" / f"{report_name}.json").read_text())
        entries = {entry["path"]: entry for entry in report["artifacts"]}
        for relpath in paths:
            path = ROOT / relpath
            actual = _sha256(path)
            expected_entry = entries[relpath]
            if path.stat().st_size != int(expected_entry["bytes"]) or actual != expected_entry["sha256"]:
                raise RuntimeError(f"checksum mismatch for {relpath}")
            verified[relpath] = {"bytes": path.stat().st_size, "sha256": actual}
    return verified


def input_path(tissue: str, seed: int) -> tuple[Path, str]:
    if seed == 0:
        return AGING / f"{tissue}_shared.h5ad", "X_scVI_shared"
    return AGING / "x04" / f"{tissue}_shared_s{seed}.h5ad", "X_scVI_shared"


def jacobians(adata: ad.AnnData) -> np.ndarray:
    scj = adata.uns["scjdo2"]
    return np.asarray(scj["J_latent"] if "J_latent" in scj else scj["J"], dtype=np.float64)


def symmetric_operator(z: np.ndarray, j_cells: np.ndarray) -> np.ndarray:
    operator = centroid_operator(z, j_cells, n_cells_centroid=N_CENTROID)
    return 0.5 * (operator + operator.T)


def operator_metrics(operator: np.ndarray) -> tuple[float, float, float]:
    eigenvalues = np.linalg.eigvalsh(operator)
    lead = float(eigenvalues.max())
    median_abs = float(np.median(np.abs(eigenvalues)))
    return median_abs, lead, float(median_abs / (abs(lead) + EPS))


def donor_ages(obs: pd.DataFrame) -> dict[str, str]:
    pairs = obs.loc[:, ["donor_id", "age_group"]].astype(str).drop_duplicates()
    if int(pairs.groupby("donor_id")["age_group"].nunique().max()) != 1:
        raise RuntimeError("a donor has more than one age_group")
    return dict(zip(pairs["donor_id"], pairs["age_group"], strict=True))


def eligible_types(adata: ad.AnnData, use_rep: str) -> pd.DataFrame:
    return compare_types(
        adata, by="cell_type", groups=("young", "old"), group_key="age_group",
        use_rep=use_rep, n_cells_centroid=N_CENTROID, min_cells=MIN_CELLS,
        n_null=N_NULL, seed=0,
    ).sort_values("cell_type", kind="stable").reset_index(drop=True)


def stored_a03_calls(tissue: str) -> pd.DataFrame:
    table = pd.read_csv(AGING / f"{tissue}_compare_shared.csv", usecols=["cell_type", "iso_pass", "ant_pass"])
    table["iso_pass"] = table["iso_pass"].astype(bool)
    table["ant_pass"] = table["ant_pass"].astype(bool)
    return table.set_index("cell_type")


def validate_baseline_calls(tissue: str, recomputed: pd.DataFrame) -> pd.DataFrame:
    stored = stored_a03_calls(tissue)
    for row in recomputed.itertuples(index=False):
        if str(row.cell_type) not in stored.index:
            raise RuntimeError(f"A03 call missing {tissue}/{row.cell_type}")
        if bool(row.iso_pass) != bool(stored.loc[str(row.cell_type), "iso_pass"]) or bool(row.ant_pass) != bool(stored.loc[str(row.cell_type), "ant_pass"]):
            raise RuntimeError(f"A03 call mismatch for {tissue}/{row.cell_type}")
    return stored


def draw_seed(cell_type: str, donor_id: str) -> int:
    """Revision-specified seed 0 plus a stable type×mouse CRC32 offset."""
    return (DRAW_SEED + zlib.crc32(f"{cell_type}|{donor_id}".encode("utf-8"))) % (2**32)


def per_mouse_table(tissue: str, seed: int, adata: ad.AnnData, use_rep: str, types: list[str]) -> pd.DataFrame:
    """Equal-count and full-count mouse operators, retaining below-floor rows."""
    z = np.asarray(adata.obsm[use_rep], dtype=np.float64)
    j = jacobians(adata)
    obs = adata.obs
    cell_type = obs["cell_type"].astype(str).to_numpy()
    donor = obs["donor_id"].astype(str).to_numpy()
    age = obs["age_group"].astype(str).to_numpy()
    ages = donor_ages(obs)
    donors = sorted(ages)
    rows: list[dict] = []
    for ct in types:
        counts = {mouse: int(((cell_type == ct) & (donor == mouse)).sum()) for mouse in donors}
        passing_counts = [count for count in counts.values() if count >= N_CENTROID]
        if not passing_counts:
            raise RuntimeError(f"no mouse passes the 30-cell floor for {tissue}/{ct}")
        n_common = min(passing_counts)
        young_mask = (cell_type == ct) & (age == "young")
        young_z = z[young_mask]
        young_center = young_z.mean(axis=0)
        young_cov_trace = float(np.trace(np.cov(young_z, rowvar=False)))
        for mouse in donors:
            mask = (cell_type == ct) & (donor == mouse)
            n_cells = counts[mouse]
            record = {
                "tissue": tissue, "seed": int(seed), "cell_type": ct, "donor_id": mouse,
                "age_group": ages[mouse], "n_cells": n_cells, "n_common": int(n_common),
                "n_equal_draws": int(N_EQUAL_DRAWS), "included_n_ge_30": bool(n_cells >= N_CENTROID),
                "pooled_young_cov_trace": young_cov_trace,
                "median_abs_lambda": float("nan"), "median_abs_lambda_q025": float("nan"),
                "median_abs_lambda_q975": float("nan"), "median_abs_lambda_fullcount": float("nan"),
                "lead_eig": float("nan"), "anisotropy": float("nan"),
                "lead_eig_fullcount": float("nan"), "anisotropy_fullcount": float("nan"),
                "offset_z": float("nan"),
                "exclusion_reason": "" if n_cells >= N_CENTROID else "fewer_than_30_cells",
            }
            if n_cells >= N_CENTROID:
                full_median, full_lead, full_anisotropy = operator_metrics(symmetric_operator(z[mask], j[mask]))
                indices = np.flatnonzero(mask)
                rng = np.random.default_rng(draw_seed(ct, mouse))
                medians = np.empty(N_EQUAL_DRAWS, dtype=float)
                leads = np.empty(N_EQUAL_DRAWS, dtype=float)
                anisotropies = np.empty(N_EQUAL_DRAWS, dtype=float)
                for draw in range(N_EQUAL_DRAWS):
                    selected = rng.choice(indices, size=n_common, replace=False)
                    medians[draw], leads[draw], anisotropies[draw] = operator_metrics(symmetric_operator(z[selected], j[selected]))
                record.update({
                    "median_abs_lambda": float(np.median(medians)),
                    "median_abs_lambda_q025": float(np.quantile(medians, 0.025)),
                    "median_abs_lambda_q975": float(np.quantile(medians, 0.975)),
                    "median_abs_lambda_fullcount": full_median,
                    "lead_eig": float(np.median(leads)), "anisotropy": float(np.median(anisotropies)),
                    "lead_eig_fullcount": full_lead, "anisotropy_fullcount": full_anisotropy,
                    "offset_z": float(np.linalg.norm(z[mask].mean(axis=0) - young_center) / np.sqrt(young_cov_trace + EPS)),
                })
            rows.append(record)
    return pd.DataFrame(rows)


def exact_mouse_permutations(per_mouse: pd.DataFrame, value_column: str, value_source: str) -> pd.DataFrame:
    """Exact one-sided mouse-label permutation tests for one operator value."""
    rows: list[dict] = []
    for (tissue, seed, cell_type), group in per_mouse.groupby(["tissue", "seed", "cell_type"], sort=True):
        valid = group[group["included_n_ge_30"]].copy().sort_values("donor_id", kind="stable")
        young = valid[valid["age_group"].eq("young")]
        old = valid[valid["age_group"].eq("old")]
        base = {
            "tissue": tissue, "seed": int(seed), "cell_type": cell_type, "value_source": value_source,
            "n_common": int(valid["n_common"].iloc[0]) if len(valid) else float("nan"),
            "n_young_mice": int(len(young)), "n_old_mice": int(len(old)), "n_mice": int(len(valid)),
            "T_log_median_abs_lambda": float("nan"), "n_relabellings": float("nan"),
            "p_stiffer": float("nan"), "p_softer": float("nan"), "status": "ok",
        }
        if len(young) == 0 or len(old) == 0:
            base["status"] = "insufficient_mice_after_30_cell_floor"
            rows.append(base)
            continue
        values = np.log(valid[value_column].to_numpy(dtype=float))
        labels = valid["age_group"].to_numpy(dtype=str)
        n_young = int((labels == "young").sum())
        observed = float(values[labels == "old"].mean() - values[labels == "young"].mean())
        null = []
        all_indices = np.arange(len(values))
        for selected_young in itertools.combinations(range(len(values)), n_young):
            selected_young = np.asarray(selected_young, dtype=int)
            selected_old = np.setdiff1d(all_indices, selected_young, assume_unique=True)
            null.append(float(values[selected_old].mean() - values[selected_young].mean()))
        null_values = np.asarray(null, dtype=float)
        base.update({
            "T_log_median_abs_lambda": observed, "n_relabellings": int(len(null_values)),
            "p_stiffer": float(np.mean(null_values >= observed - 1e-12)),
            "p_softer": float(np.mean(null_values <= observed + 1e-12)),
        })
        rows.append(base)
    return pd.DataFrame(rows)


def loo_table(tissue: str, adata: ad.AnnData, use_rep: str, types: list[str], a03_calls: pd.DataFrame) -> pd.DataFrame:
    """A03-exact comparisons after dropping each old donor, without refitting."""
    obs = adata.obs
    age = obs["age_group"].astype(str).to_numpy()
    donor = obs["donor_id"].astype(str).to_numpy()
    cell_type = obs["cell_type"].astype(str).to_numpy()
    z = np.asarray(adata.obsm[use_rep], dtype=np.float64)
    j = jacobians(adata)
    rows: list[dict] = []
    for dropped in sorted(set(donor[age == "old"])):
        retained = donor != dropped
        reduced = adata[retained].copy()
        reduced_scj = dict(reduced.uns["scjdo2"])
        reduced_scj["J"] = j[retained].copy()
        reduced_scj["J_latent"] = j[retained].copy()
        reduced.uns["scjdo2"] = reduced_scj
        calls = compare_types(
            reduced, by="cell_type", groups=("young", "old"), group_key="age_group",
            use_rep=use_rep, n_cells_centroid=N_CENTROID, min_cells=MIN_CELLS,
            n_null=N_NULL, seed=0,
        ).set_index("cell_type")
        for ct in types:
            young = retained & (cell_type == ct) & (age == "young")
            old = retained & (cell_type == ct) & (age == "old")
            record = {
                "tissue": tissue, "dropped_mouse": dropped, "cell_type": ct,
                "n_young": int(young.sum()), "n_old": int(old.sum()),
                "a03_iso_pass": bool(a03_calls.loc[ct, "iso_pass"]),
                "a03_ant_pass": bool(a03_calls.loc[ct, "ant_pass"]),
                "eligible_after_drop": False, "iso_pass": float("nan"), "ant_pass": float("nan"),
                "S": float("nan"), "iso_null_q95": float("nan"), "stiff_ratio": float("nan"),
            }
            if ct in calls.index:
                test = calls.loc[ct]
                median_young, _, _ = operator_metrics(symmetric_operator(z[young], j[young]))
                median_old, _, _ = operator_metrics(symmetric_operator(z[old], j[old]))
                record.update({
                    "eligible_after_drop": True, "iso_pass": bool(test.iso_pass), "ant_pass": bool(test.ant_pass),
                    "S": float(test.S), "iso_null_q95": float(test.iso_null_q95),
                    "stiff_ratio": float(median_old / (median_young + EPS)),
                })
            rows.append(record)
    return pd.DataFrame(rows)


def _permutation_row(permutations: pd.DataFrame, tissue: str, cell_type: str, seed: int, source: str) -> pd.Series:
    return permutations[
        (permutations["tissue"].eq(tissue)) & (permutations["cell_type"].eq(cell_type)) &
        (permutations["seed"].eq(seed)) & (permutations["value_source"].eq(source))
    ].iloc[0]


def _verdict(gcp: pd.Series, prefix: str) -> str:
    p0, p1, p2 = (float(gcp[f"p_stiffer_{prefix}seed{seed}"]) for seed in SEEDS)
    return (
        "granulocytopoietic stiffening exceeds mouse-to-mouse variation"
        if p0 <= 0.05 and (p1 <= 0.05 or p2 <= 0.05) and float(gcp["loo_min_stiff_ratio"]) >= 1.5
        else "not distinguishable from mouse-to-mouse variation"
    )


def stability_summary(loo: pd.DataFrame, per_seed0: pd.DataFrame, permutations: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    for (tissue, cell_type), group in loo.groupby(["tissue", "cell_type"], sort=True):
        eligible_loo = group[group["eligible_after_drop"]].copy()
        source = per_seed0[(per_seed0["tissue"].eq(tissue)) & (per_seed0["cell_type"].eq(cell_type))]
        excluded = source[~source["included_n_ge_30"]]["donor_id"].tolist()
        row = {
            "tissue": tissue, "cell_type": cell_type,
            "n_common": int(source["n_common"].iloc[0]),
            "a03_iso_pass": bool(group["a03_iso_pass"].iloc[0]), "a03_ant_pass": bool(group["a03_ant_pass"].iloc[0]),
            "n_loo_drops": int(len(group)), "n_loo_eligible": int(len(eligible_loo)),
            "iso_call_stability": float(np.mean(eligible_loo["iso_pass"].astype(bool).to_numpy() == bool(group["a03_iso_pass"].iloc[0]))) if len(eligible_loo) else float("nan"),
            "ant_call_stability": float(np.mean(eligible_loo["ant_pass"].astype(bool).to_numpy() == bool(group["a03_ant_pass"].iloc[0]))) if len(eligible_loo) else float("nan"),
            "loo_min_stiff_ratio": float(eligible_loo["stiff_ratio"].min()) if len(eligible_loo) else float("nan"),
            "mice_excluded_by_30_cell_floor": ";".join(excluded),
        }
        for source_name, prefix in (("equal_count", ""), ("full_count", "fullcount_")):
            for seed in SEEDS:
                q = _permutation_row(permutations, tissue, cell_type, seed, source_name)
                row[f"T_{prefix}seed{seed}"] = float(q["T_log_median_abs_lambda"]) if pd.notna(q["T_log_median_abs_lambda"]) else float("nan")
                row[f"p_stiffer_{prefix}seed{seed}"] = float(q["p_stiffer"]) if pd.notna(q["p_stiffer"]) else float("nan")
                row[f"p_softer_{prefix}seed{seed}"] = float(q["p_softer"]) if pd.notna(q["p_softer"]) else float("nan")
                row[f"n_relabellings_{prefix}seed{seed}"] = int(q["n_relabellings"]) if pd.notna(q["n_relabellings"]) else float("nan")
                row[f"permutation_status_{prefix}seed{seed}"] = str(q["status"])
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["tissue", "cell_type"], kind="stable")
    gcp = summary[(summary["tissue"].eq(GCP[0])) & (summary["cell_type"].eq(GCP[1]))].iloc[0]
    verdict = _verdict(gcp, "")
    fullcount_verdict = _verdict(gcp, "fullcount_")
    summary["x05_verdict"] = verdict
    summary["x05_verdict_fullcount"] = fullcount_verdict
    return summary, {"x05_verdict": verdict, "x05_verdict_fullcount": fullcount_verdict, "gcp": gcp.to_dict()}


def plot_per_mouse(per_mouse: pd.DataFrame, permutations: pd.DataFrame) -> None:
    types = per_mouse[["tissue", "cell_type"]].drop_duplicates().sort_values(["tissue", "cell_type"])
    fig, axes = plt.subplots(4, 2, figsize=(12, 15), squeeze=False)
    for index, (ax, item) in enumerate(zip(axes.ravel(), types.itertuples(index=False), strict=True)):
        data = per_mouse[(per_mouse["tissue"].eq(item.tissue)) & (per_mouse["cell_type"].eq(item.cell_type)) & per_mouse["included_n_ge_30"]]
        for pos, age, color in ((0, "young", "#378ADD"), (1, "old", "#D85A30")):
            current = data[data["age_group"].eq(age)]
            values = current["median_abs_lambda"].to_numpy(dtype=float)
            low = values - current["median_abs_lambda_q025"].to_numpy(dtype=float)
            high = current["median_abs_lambda_q975"].to_numpy(dtype=float) - values
            jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) else np.array([])
            ax.errorbar(np.full(len(values), pos) + jitter, values, yerr=np.vstack([low, high]), fmt="o", color=color, ms=5, capsize=2, lw=0.8, zorder=3)
            ax.scatter(np.full(len(values), pos) + jitter, current["median_abs_lambda_fullcount"], color=color, marker="x", s=28, lw=0.9, alpha=0.65, zorder=4)
        equal = _permutation_row(permutations, item.tissue, item.cell_type, 0, "equal_count")
        full = _permutation_row(permutations, item.tissue, item.cell_type, 0, "full_count")
        eq_label = "NA" if pd.isna(equal.p_stiffer) else f"{float(equal.p_stiffer):.3g}"
        full_label = "NA" if pd.isna(full.p_stiffer) else f"{float(full.p_stiffer):.3g}"
        ax.text(0.5, 0.96, f"p equal={eq_label}; full={full_label}", transform=ax.transAxes, ha="center", va="top", fontsize=7.5)
        ax.set_yscale("log"); ax.set_xticks([0, 1]); ax.set_xticklabels(["young", "old"])
        ax.set_title(f"{item.tissue}: {item.cell_type} (n_common={int(data['n_common'].iloc[0])})", fontsize=8.5)
        ax.set_ylabel("median |λ| per mouse"); ax.grid(axis="y", color="0.9", lw=0.6)
        if index == 0:
            ax.legend(handles=[Line2D([0], [0], marker="o", color="0.25", label="equal-count median ± 2.5–97.5%", linestyle="None"), Line2D([0], [0], marker="x", color="0.25", label="full-count provenance", linestyle="None")], fontsize=6.5, loc="lower left")
    fig.suptitle("X05 exploratory: equal-count per-mouse centroid-operator stiffness (seed 0)", y=0.995)
    fig.text(0.5, 0.005, FOOTER, ha="center", va="bottom", fontsize=7, color="0.35")
    fig.tight_layout(rect=(0, 0.025, 1, 0.98)); FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "per_mouse_stiffness.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "per_mouse_stiffness.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_loo_heatmap(loo: pd.DataFrame) -> None:
    code = {"F/F": 0, "T/F": 1, "F/T": 2, "T/T": 3, "NA": 4}
    cmap = ListedColormap(["#D8D8D8", "#8AB6E9", "#EAB08A", "#A98BC8", "#FFFFFF"])
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), squeeze=False)
    for ax, tissue in zip(axes.ravel(), TISSUES, strict=True):
        data = loo[loo["tissue"].eq(tissue)]
        types = sorted(data["cell_type"].unique()); donors = sorted(data["dropped_mouse"].unique())
        matrix = np.full((len(types), len(donors)), code["NA"], dtype=int); text = np.full(matrix.shape, "NA", dtype=object)
        for i, ct in enumerate(types):
            for j, mouse in enumerate(donors):
                row = data[(data["cell_type"].eq(ct)) & (data["dropped_mouse"].eq(mouse))].iloc[0]
                label = "NA" if not bool(row.eligible_after_drop) else f"{'T' if bool(row.iso_pass) else 'F'}/{'T' if bool(row.ant_pass) else 'F'}"
                matrix[i, j] = code[label]; text[i, j] = label
        ax.imshow(matrix, cmap=cmap, vmin=0, vmax=4, aspect="auto")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, text[i, j], ha="center", va="center", fontsize=7)
        ax.set_yticks(range(len(types))); ax.set_yticklabels(types, fontsize=8)
        ax.set_xticks(range(len(donors))); ax.set_xticklabels(donors, rotation=35, ha="right", fontsize=8)
        ax.set_title(f"{tissue}: leave-one-old-mouse-out calls (iso/ant)")
    fig.suptitle("X05 exploratory: leave-one-old-mouse-out A03 calls", y=0.995)
    fig.text(0.5, 0.005, FOOTER + " · T=True, F=False, NA=<200 cells", ha="center", va="bottom", fontsize=7, color="0.35")
    fig.tight_layout(rect=(0, 0.04, 1, 0.98)); FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "loo_call_stability.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / "loo_call_stability.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    output = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame[columns].itertuples(index=False):
        values = [f"{value:.4g}" if isinstance(value, (float, np.floating)) else str(value) for value in row]
        output.append("| " + " | ".join(values) + " |")
    return "\n".join(output)


def write_result(summary: pd.DataFrame, metadata: dict) -> None:
    rule = '''* Granulocytopoietic stiffening **exceeds mouse-to-mouse variation** if all three hold: `p_stiffer ≤ 0.05` in seed 0; `p_stiffer ≤ 0.05` in at least one of seeds 1, 2; and the leave-one-out `stiff_ratio` stays ≥ 1.5 for every dropped old mouse. Otherwise **"not distinguishable from mouse-to-mouse variation"**.
* For every type, report the fraction of leave-one-out drops in which `iso_pass` and `ant_pass` equal the A03 call ("call stability").'''
    columns = ["tissue", "cell_type", "n_common", "p_stiffer_seed0", "p_stiffer_seed1", "p_stiffer_seed2", "p_stiffer_fullcount_seed0", "loo_min_stiff_ratio", "iso_call_stability", "ant_call_stability"]
    text = "# EXPLORATORY — aging/X05 per-mouse uncertainty\n\n"
    text += "## Question\nDoes granulocytopoietic stiffness exceed mouse-to-mouse variation, and how stable are A03 calls when one old mouse is omitted? This exploratory task does not alter the frozen outcome.\n\n"
    text += "## What ran\nStored A02 seed-0 and X04 shared seeds 1–2 embeddings/Jacobians only. Revision 1 uses 200 deterministic equal-count subsamples per donor/type, where `n_common` is the smallest count among mice meeting the 30-cell floor; full-count values remain provenance. No model was fitted.\n\n"
    text += "## Reading rule (fixed before the run)\n" + rule + "\n\n"
    text += "## Per-type results\n" + markdown_table(summary, columns) + "\n\n"
    text += f"## Mechanical verdict (equal-count)\n**{metadata['x05_verdict']}**\n\n"
    text += f"## Full-count provenance verdict\n**{metadata['x05_verdict_fullcount']}**\n\n"
    text += "The equal-count correction was added after review because the original design compared unequal cell counts. No interpretation beyond the fixed rule is made here.\n"
    (OUT / "RESULT_X05.md").write_text(text)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "input_checksums.json").write_text(json.dumps(verify_inputs(), indent=2) + "\n")
    per_seed: dict[tuple[str, int], pd.DataFrame] = {}
    primary_parts: list[pd.DataFrame] = []; loo_parts: list[pd.DataFrame] = []
    for tissue in TISSUES:
        path, rep = input_path(tissue, 0)
        primary = ad.read_h5ad(path)
        baseline = eligible_types(primary, rep)
        a03_calls = validate_baseline_calls(tissue, baseline)
        types = baseline["cell_type"].astype(str).tolist()
        per_seed[(tissue, 0)] = per_mouse_table(tissue, 0, primary, rep, types)
        primary_parts.append(per_seed[(tissue, 0)])
        loo_parts.append(loo_table(tissue, primary, rep, types, a03_calls))
        for seed in (1, 2):
            replica_path, replica_rep = input_path(tissue, seed)
            replica = ad.read_h5ad(replica_path)
            per_seed[(tissue, seed)] = per_mouse_table(tissue, seed, replica, replica_rep, types)
    permutation_parts = []
    for tissue in TISSUES:
        for seed in SEEDS:
            table = per_seed[(tissue, seed)]
            permutation_parts.extend([
                exact_mouse_permutations(table, "median_abs_lambda", "equal_count"),
                exact_mouse_permutations(table, "median_abs_lambda_fullcount", "full_count"),
            ])
    per_mouse = pd.concat(primary_parts, ignore_index=True).sort_values(["tissue", "cell_type", "age_group", "donor_id"], kind="stable")
    permutations = pd.concat(permutation_parts, ignore_index=True).sort_values(["value_source", "tissue", "cell_type", "seed"], kind="stable")
    loo = pd.concat(loo_parts, ignore_index=True).sort_values(["tissue", "cell_type", "dropped_mouse"], kind="stable")
    summary, metadata = stability_summary(loo, per_mouse, permutations)
    per_mouse.to_csv(OUT / "per_mouse.csv", index=False)
    permutations.to_csv(OUT / "mouse_permutation.csv", index=False)
    loo.to_csv(OUT / "loo_old.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    plot_per_mouse(per_mouse, permutations); plot_loo_heatmap(loo); write_result(summary, metadata)
    gcp = metadata["gcp"]
    metrics = {
        "x05_verdict": metadata["x05_verdict"], "x05_verdict_fullcount": metadata["x05_verdict_fullcount"],
        "gcp_n_common": int(gcp["n_common"]),
        "gcp_T_seed0": float(gcp["T_seed0"]), "gcp_p_stiffer_seed0": float(gcp["p_stiffer_seed0"]),
        "gcp_p_stiffer_seed1": float(gcp["p_stiffer_seed1"]), "gcp_p_stiffer_seed2": float(gcp["p_stiffer_seed2"]),
        "gcp_loo_min_stiff_ratio": float(gcp["loo_min_stiff_ratio"]),
        "n_relabellings_marrow": {r.cell_type: int(r.n_relabellings_seed0) if pd.notna(r.n_relabellings_seed0) else None for r in summary[summary.tissue.eq("marrow")].itertuples(index=False)},
        "n_relabellings_lung": {r.cell_type: int(r.n_relabellings_seed0) if pd.notna(r.n_relabellings_seed0) else None for r in summary[summary.tissue.eq("lung")].itertuples(index=False)},
        "mice_excluded_by_type": {f"{r.tissue}|{r.cell_type}": ([] if not r.mice_excluded_by_30_cell_floor else r.mice_excluded_by_30_cell_floor.split(";")) for r in summary.itertuples(index=False)},
        "call_stability_by_type": {f"{r.tissue}|{r.cell_type}": {"iso": float(r.iso_call_stability), "ant": float(r.ant_call_stability), "n_eligible": int(r.n_loo_eligible)} for r in summary.itertuples(index=False)},
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
