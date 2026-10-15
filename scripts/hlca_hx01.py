#!/usr/bin/env python3
"""EXPLORATORY HX01: subtype composition and study dependence follow-up.

No model is fitted. The script uses the accepted H01/H03/H04 artifacts only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import zlib
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as student_t

from scjdo2.compare import _perm_ages, donor_operator, stiffness_trend

ROOT = Path(__file__).resolve().parents[1]
HLCA = ROOT / "benchmarks" / "aging_hlca"
OUT = HLCA / "hx01"
FIG = ROOT / "benchmarks" / "figures" / "aging_hlca" / "hx01"
PRIMARY = HLCA / "data" / "h03_primary_drift.h5ad"
FULL = HLCA / "data" / "hlca_core_full_metadata.h5ad"
DONOR_OPS = HLCA / "donor_operators.csv"
TESTS_PRIMARY = HLCA / "tests_primary.csv"
H01_REPORT = ROOT / "coord" / "hlca" / "reports" / "H01.json"
H03_REPORT = ROOT / "coord" / "hlca" / "reports" / "H03.json"
H04_REPORT = ROOT / "coord" / "hlca" / "reports" / "H04.json"

COMPOSITION = OUT / "composition.csv"
COMPOSITION_SUMMARY = OUT / "composition_summary.csv"
SUBTYPE = OUT / "subtype_stiffness.csv"
SUBTYPE_DONORS = OUT / "subtype_donor_metrics.csv"
LOSO = OUT / "leave_one_study_out.csv"
FE = OUT / "study_fixed_effects.csv"
RESULT = OUT / "RESULT_HX01.md"
CHECKSUMS = OUT / "input_checksums.json"
METRICS = OUT / "metrics.json"

SEED = 0
N_COMMON = 25
N_DRAWS = 100
N_CENTROID = 25
N_PERM = 1_000
FOOTER = "EXPLORATORY · hlca/HX01"
FOCAL = {
    "T cell lineage": {
        "fraction_name": "CD8 / (CD4 + CD8)",
        "numerator": "CD8 T cells",
        "denominator": ("CD4 T cells", "CD8 T cells"),
        "subtypes": ("CD4 T cells", "CD8 T cells"),
    },
    "Monocytes": {
        "fraction_name": "Non-classical / (Classical + Non-classical)",
        "numerator": "Non-classical monocytes",
        "denominator": ("Classical monocytes", "Non-classical monocytes"),
        "subtypes": ("Classical monocytes", "Non-classical monocytes"),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reported_artifact(report_path: Path, rel: str) -> dict[str, object]:
    report = json.loads(report_path.read_text())
    item = next((x for x in report["artifacts"] if x["path"] == rel), None)
    if item is None:
        raise RuntimeError(f"{report_path} does not inventory {rel}")
    return item


def verify_inputs() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for report_path, rel in (
        (H01_REPORT, "benchmarks/aging_hlca/data/hlca_core_full_metadata.h5ad"),
        (H03_REPORT, "benchmarks/aging_hlca/data/h03_primary_drift.h5ad"),
        (H04_REPORT, "benchmarks/aging_hlca/donor_operators.csv"),
    ):
        item = reported_artifact(report_path, rel)
        path = ROOT / rel
        actual = sha256(path)
        if path.stat().st_size != int(item["bytes"]) or actual != item["sha256"]:
            raise RuntimeError(f"input checksum mismatch: {rel}")
        records.append({"path": rel, "bytes": path.stat().st_size, "sha256": actual, "source_report": str(report_path.relative_to(ROOT))})
    return records


def load_inputs() -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = ad.read_h5ad(PRIMARY)
    z = np.asarray(primary.obsm.get("X_scVI_primary"), dtype=np.float64)
    jacobian = np.asarray(primary.uns.get("scjdo2", {}).get("J_latent"), dtype=np.float64)
    if z.shape != (primary.n_obs, 15) or jacobian.shape != (primary.n_obs, 15, 15):
        raise RuntimeError(f"unexpected H03 primary latent/J dimensions: {z.shape}, {jacobian.shape}")
    required_primary = {"donor_id", "ann_level_3", "age", "study_or_dataset"}
    missing = sorted(required_primary - set(primary.obs.columns))
    if missing:
        raise RuntimeError(f"H03 primary drift data missing {missing}")
    obs = primary.obs.loc[:, sorted(required_primary)].copy()
    obs.index = obs.index.astype(str)
    obs["donor_id"] = obs["donor_id"].astype(str)
    obs["ann_level_3"] = obs["ann_level_3"].astype(str)
    obs["study_or_dataset"] = obs["study_or_dataset"].astype(str)
    obs["age"] = pd.to_numeric(obs["age"], errors="raise").astype(float)
    if obs.index.duplicated().any():
        raise RuntimeError("H03 primary cell IDs are duplicated")

    full = ad.read_h5ad(FULL, backed="r")
    required_full = {"donor_id", "ann_level_4"}
    missing = sorted(required_full - set(full.obs.columns))
    if missing:
        raise RuntimeError(f"full metadata missing {missing}")
    full_obs = full.obs.loc[:, ["donor_id", "ann_level_4"]].copy()
    full_obs.index = full.obs_names.astype(str)
    full_obs["donor_id"] = full_obs["donor_id"].astype(str)
    full_obs["ann_level_4"] = full_obs["ann_level_4"].astype(str)
    full.file.close()
    if full_obs.index.duplicated().any():
        raise RuntimeError("full metadata cell IDs are duplicated")

    donor_ops = pd.read_csv(DONOR_OPS)
    expected_ops = {"analysis", "ann_level_3", "donor_id", "age", "study_or_dataset", "log_median_abs_lambda"}
    missing = sorted(expected_ops - set(donor_ops.columns))
    if missing:
        raise RuntimeError(f"H04 donor operators missing {missing}")
    donor_ops["donor_id"] = donor_ops["donor_id"].astype(str)
    donor_ops["ann_level_3"] = donor_ops["ann_level_3"].astype(str)
    donor_ops["study_or_dataset"] = donor_ops["study_or_dataset"].astype(str)
    donor_ops["age"] = pd.to_numeric(donor_ops["age"], errors="raise").astype(float)
    donor_ops["log_median_abs_lambda"] = pd.to_numeric(donor_ops["log_median_abs_lambda"], errors="raise").astype(float)
    h04 = pd.read_csv(TESTS_PRIMARY).set_index("ann_level_3")
    for parent in FOCAL:
        rows = donor_ops[(donor_ops["analysis"] == "primary") & (donor_ops["ann_level_3"] == parent)]
        if rows.empty or rows["donor_id"].duplicated().any() or parent not in h04.index:
            raise RuntimeError(f"invalid H04 primary donor rows for {parent}")
    return z, jacobian, obs, full_obs, donor_ops, h04


def add_ann_level_4(obs: pd.DataFrame, full_obs: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    focus = obs[obs["ann_level_3"].isin(FOCAL)].copy()
    matched = focus.index.isin(full_obs.index)
    unmatched = int((~matched).sum())
    focus["ann_level_4"] = full_obs.reindex(focus.index)["ann_level_4"].to_numpy()
    return focus, unmatched


def safe_spearman(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> tuple[int, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if int(keep.sum()) < 3 or np.unique(x[keep]).size < 2 or np.unique(y[keep]).size < 2:
        return int(keep.sum()), float("nan"), float("nan")
    result = spearmanr(x[keep], y[keep])
    return int(keep.sum()), float(result.statistic), float(result.pvalue)


def composition_rows(full_obs: pd.DataFrame, donor_ops: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for parent, spec in FOCAL.items():
        operators = donor_ops[(donor_ops["analysis"] == "primary") & (donor_ops["ann_level_3"] == parent)].copy().sort_values("donor_id", kind="stable")
        donors = set(operators["donor_id"])
        meta = full_obs[full_obs["donor_id"].isin(donors)]
        labels = set(spec["denominator"])
        counts = meta[meta["ann_level_4"].isin(labels)].groupby(["donor_id", "ann_level_4"], sort=True).size().unstack(fill_value=0)
        for _, item in operators.iterrows():
            donor = str(item["donor_id"])
            cd = counts.loc[donor] if donor in counts.index else pd.Series(dtype=float)
            denominator = int(sum(int(cd.get(label, 0)) for label in spec["denominator"]))
            numerator = int(cd.get(spec["numerator"], 0))
            fraction = float(numerator / denominator) if denominator else float("nan")
            rows.append({
                "ann_level_3": parent, "fraction_name": spec["fraction_name"], "donor_id": donor,
                "age": float(item["age"]), "study_or_dataset": str(item["study_or_dataset"]),
                "numerator_label": spec["numerator"], "numerator_count": numerator,
                "denominator_count": denominator, "fraction": fraction,
                "log_median_abs_lambda": float(item["log_median_abs_lambda"]),
            })
        frame = pd.DataFrame([x for x in rows if x["ann_level_3"] == parent])
        n_age, rho_age, p_age = safe_spearman(frame["fraction"], frame["age"])
        n_stiff, rho_stiff, p_stiff = safe_spearman(frame["fraction"], frame["log_median_abs_lambda"])
        summaries.append({"ann_level_3": parent, "fraction_name": spec["fraction_name"], "n_h04_donors": int(len(frame)), "n_valid_fraction": int(frame["fraction"].notna().sum()), "rho_age": rho_age, "p_age": p_age, "rho_h04_log_stiffness": rho_stiff, "p_h04_log_stiffness": p_stiff, "n_spearman_age": n_age, "n_spearman_h04_log_stiffness": n_stiff})
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def log_stiffness(op: np.ndarray) -> float:
    values = np.linalg.eigvalsh(0.5 * (op + op.T))
    return float(np.log(np.median(np.abs(values))))


def subtype_stiffness(z: np.ndarray, jacobian: np.ndarray, focus: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    donor_rows: list[dict[str, object]] = []
    positions = pd.Series(np.arange(len(focus), dtype=int), index=focus.index)
    # focus rows preserve the H03 order only after extracting their original positions.
    index_to_position = {cell_id: int(pos) for pos, cell_id in enumerate(focus.index)}
    h03_position = {cell_id: idx for idx, cell_id in enumerate(pd.Index(focus.index))}
    # Map focus rows to their original H03 positions through an explicit column set by caller.
    if "_h03_position" not in focus:
        raise RuntimeError("focus metadata lacks H03 position")
    for parent, spec in FOCAL.items():
        for subtype in spec["subtypes"]:
            subset = focus[(focus["ann_level_3"] == parent) & (focus["ann_level_4"] == subtype)]
            operators: list[np.ndarray] = []
            ages: list[float] = []
            for donor in sorted(set(subset["donor_id"])):
                one = subset[subset["donor_id"] == donor]
                if len(one) < N_COMMON:
                    continue
                idx = one["_h03_position"].to_numpy(dtype=int)
                operator, _ = donor_operator(
                    z[idx], jacobian[idx], n_common=N_COMMON, n_draws=N_DRAWS,
                    n_cells_centroid=N_CENTROID, seed=zlib.crc32(f"{subtype}|{donor}".encode()),
                )
                age = float(one["age"].iloc[0])
                study = str(one["study_or_dataset"].iloc[0])
                metric = log_stiffness(operator)
                operators.append(operator); ages.append(age)
                donor_rows.append({"ann_level_3": parent, "ann_level_4": subtype, "donor_id": donor, "age": age, "study_or_dataset": study, "n_subsample_cells": int(len(one)), "log_median_abs_lambda": metric})
            if len(operators) < 4:
                raise RuntimeError(f"{subtype} retained fewer than four donors after the 25-cell floor")
            result = stiffness_trend(operators, ages, n_perm=N_PERM, seed=SEED)
            if not np.isfinite([result.statistic, result.p_value, result.null_q95]).all():
                raise RuntimeError(f"non-finite subtype stiffness result for {subtype}")
            rows.append({"ann_level_3": parent, "ann_level_4": subtype, "n_donors": len(operators), "stiff_slope": float(result.statistic), "stiff_p": float(result.p_value), "passes": bool(result.passes), "null_q95_abs_slope": float(result.null_q95), "n_common": N_COMMON, "n_draws": N_DRAWS, "n_perm": N_PERM})
    return pd.DataFrame(rows), pd.DataFrame(donor_rows)


def slope(y: np.ndarray, ages: np.ndarray) -> float:
    age_centered = ages - ages.mean()
    return float(age_centered @ (y - y.mean()) / (age_centered @ age_centered))


def stiffness_from_logs(log_values: np.ndarray, ages: np.ndarray) -> tuple[float, float, float]:
    observed = slope(log_values, ages)
    null = np.array([slope(log_values, item) for item in _perm_ages(ages, N_PERM, SEED)])
    p_value = float((1 + np.sum(np.abs(null) >= abs(observed))) / (1 + len(null)))
    return observed, p_value, float(np.percentile(np.abs(null), 95))


def slope_ci(log_values: np.ndarray, ages: np.ndarray) -> tuple[float, float]:
    x = np.column_stack([np.ones(len(ages)), ages])
    beta, _, rank, _ = np.linalg.lstsq(x, log_values, rcond=None)
    if rank != 2 or len(ages) <= 2:
        return float("nan"), float("nan")
    residual = log_values - x @ beta
    variance = float(residual @ residual / (len(ages) - 2))
    se = math.sqrt(variance / float(np.sum((ages - ages.mean()) ** 2)))
    width = float(student_t.ppf(.975, len(ages) - 2) * se)
    return float(beta[1] - width), float(beta[1] + width)


def fixed_effect_slope(frame: pd.DataFrame) -> float:
    studies = sorted(frame["study_or_dataset"].astype(str).unique())
    dummies = [(frame["study_or_dataset"].astype(str).to_numpy() == study).astype(float) for study in studies[1:]]
    x = np.column_stack([np.ones(len(frame)), frame["age"].to_numpy(float), *dummies])
    beta, _, rank, _ = np.linalg.lstsq(x, frame["log_median_abs_lambda"].to_numpy(float), rcond=None)
    if rank != x.shape[1]:
        raise RuntimeError("rank-deficient study fixed-effects model")
    return float(beta[1])


def study_dependence(donor_ops: pd.DataFrame, h04: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    loso: list[dict[str, object]] = []
    fixed: list[dict[str, object]] = []
    for parent in FOCAL:
        frame = donor_ops[(donor_ops["analysis"] == "primary") & (donor_ops["ann_level_3"] == parent)].copy().sort_values("donor_id", kind="stable")
        values = frame["log_median_abs_lambda"].to_numpy(float); ages = frame["age"].to_numpy(float)
        observed, p_value, _ = stiffness_from_logs(values, ages)
        reference = h04.loc[parent]
        if not np.isclose(observed, float(reference["stiff_slope"]), rtol=0, atol=1e-12) or not np.isclose(p_value, float(reference["stiff_p"]), rtol=0, atol=1e-12):
            raise RuntimeError(f"H04 primary stiffness reproduction failed for {parent}: {(observed, p_value)}")
        for study in sorted(frame["study_or_dataset"].unique()):
            keep = frame[frame["study_or_dataset"] != study]
            estimate, leave_p, q95 = stiffness_from_logs(keep["log_median_abs_lambda"].to_numpy(float), keep["age"].to_numpy(float))
            low, high = slope_ci(keep["log_median_abs_lambda"].to_numpy(float), keep["age"].to_numpy(float))
            loso.append({"ann_level_3": parent, "left_out_study": study, "n_donors": int(len(keep)), "stiff_slope": estimate, "stiff_p": leave_p, "null_q95_abs_slope": q95, "ci95_low": low, "ci95_high": high})
        fixed.append({"ann_level_3": parent, "n_donors": int(len(frame)), "n_studies": int(frame["study_or_dataset"].nunique()), "h04_stiff_slope": float(reference["stiff_slope"]), "h04_stiff_p": float(reference["stiff_p"]), "fe_age_slope": fixed_effect_slope(frame)})
    return pd.DataFrame(loso), pd.DataFrame(fixed)


def verdicts(comp: pd.DataFrame, subtype: pd.DataFrame, loso: pd.DataFrame, fixed: pd.DataFrame) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for parent, spec in FOCAL.items():
        h04_slope = float(fixed.loc[fixed["ann_level_3"] == parent, "h04_stiff_slope"].iloc[0])
        fe_slope = float(fixed.loc[fixed["ann_level_3"] == parent, "fe_age_slope"].iloc[0])
        subtype_slopes = subtype.loc[subtype["ann_level_3"] == parent, "stiff_slope"].to_numpy(float)
        p_age = float(comp.loc[comp["ann_level_3"] == parent, "p_age"].iloc[0])
        max_loso_p = float(loso.loc[loso["ann_level_3"] == parent, "stiff_p"].max())
        composition_explained = bool(np.all(np.abs(subtype_slopes) < abs(h04_slope) / 2) and p_age <= .05)
        study_driven = bool(max_loso_p > .05 or abs(fe_slope) < abs(h04_slope) / 2)
        if composition_explained and study_driven:
            verdict = "composition-explained and study-driven"
        elif composition_explained:
            verdict = "composition-explained"
        elif study_driven:
            verdict = "study-driven"
        else:
            verdict = "unexplained"
        output[parent] = {"verdict": verdict, "composition_explained": composition_explained, "study_driven": study_driven, "max_loso_p": max_loso_p, "fe_slope": fe_slope, "h04_slope": h04_slope, "composition_p_age": p_age}
    return output


def plot_composition(comp: pd.DataFrame, summary: pd.DataFrame) -> list[Path]:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), sharex=False)
    for ax, parent in zip(axes, FOCAL, strict=True):
        frame = comp[comp["ann_level_3"] == parent].dropna(subset=["fraction"])
        studies = sorted(frame["study_or_dataset"].unique())
        for index, study in enumerate(studies):
            one = frame[frame["study_or_dataset"] == study]
            ax.scatter(one["age"], one["fraction"], s=26, alpha=.82, color=plt.cm.tab20(index % 20), label=study)
        if len(frame) >= 2:
            beta = np.polyfit(frame["age"], frame["fraction"], 1)
            x = np.linspace(float(frame.age.min()), float(frame.age.max()), 100)
            ax.plot(x, np.polyval(beta, x), color="#1a1a1a", linewidth=1.3, label="OLS guide")
        item = summary[summary["ann_level_3"] == parent].iloc[0]
        ax.set_title(f"{parent}\nSpearman ρ={item.rho_age:.3g}, p={item.p_age:.4g}", loc="left", weight="bold")
        ax.set_xlabel("donor age (years)"); ax.set_ylabel(str(item.fraction_name)); ax.set_ylim(-.04, 1.04); ax.grid(color=".92")
        ax.legend(fontsize=5.8, frameon=False, loc="best", ncol=2)
    fig.text(.5, .01, FOOTER, ha="center", fontsize=7, color=".35"); fig.tight_layout(rect=(0, .04, 1, .98))
    paths=[]
    for suffix in ("pdf", "png"):
        path=FIG/f"composition_vs_age.{suffix}"; fig.savefig(path,dpi=200,bbox_inches="tight"); paths.append(path)
    plt.close(fig); return paths


def plot_subtypes(donors: pd.DataFrame, subtype: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.0))
    for ax, row in zip(axes.ravel(), subtype.itertuples(index=False), strict=True):
        frame = donors[(donors["ann_level_3"] == row.ann_level_3) & (donors["ann_level_4"] == row.ann_level_4)]
        studies = sorted(frame["study_or_dataset"].unique())
        for index, study in enumerate(studies):
            one=frame[frame["study_or_dataset"]==study]
            ax.scatter(one["age"],one["log_median_abs_lambda"],s=25,alpha=.82,color=plt.cm.tab20(index%20),label=study)
        beta=np.polyfit(frame["age"],frame["log_median_abs_lambda"],1); x=np.linspace(float(frame.age.min()),float(frame.age.max()),100)
        ax.plot(x,np.polyval(beta,x),color="#1a1a1a",linewidth=1.25,label="OLS slope")
        ax.set_title(f"{row.ann_level_4} (n={row.n_donors})\nslope={row.stiff_slope:.3g}; p={row.stiff_p:.4g}",loc="left",weight="bold")
        ax.set_xlabel("donor age (years)"); ax.set_ylabel("log median |λ|"); ax.grid(color=".92"); ax.legend(fontsize=5.5,frameon=False,loc="best",ncol=2)
    fig.text(.5,.01,FOOTER,ha="center",fontsize=7,color=".35"); fig.tight_layout(rect=(0,.04,1,.98))
    paths=[]
    for suffix in ("pdf","png"):
        path=FIG/f"subtype_stiffness_vs_age.{suffix}"; fig.savefig(path,dpi=200,bbox_inches="tight"); paths.append(path)
    plt.close(fig); return paths


def plot_loso(loso: pd.DataFrame, fixed: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.2), sharex=False)
    for ax, parent in zip(axes, FOCAL, strict=True):
        frame=loso[loso["ann_level_3"]==parent].sort_values("stiff_slope",kind="stable").reset_index(drop=True)
        y=np.arange(len(frame)); lower=frame.stiff_slope-frame.ci95_low; upper=frame.ci95_high-frame.stiff_slope
        ax.errorbar(frame.stiff_slope,y,xerr=np.vstack([lower,upper]),fmt='o',color='#3f6c8e',ecolor='#3f6c8e',capsize=2.5)
        reference=float(fixed.loc[fixed["ann_level_3"]==parent,"h04_stiff_slope"].iloc[0])
        ax.axvline(reference,color='#c94c4c',linestyle='--',linewidth=1.2,label='H04 full-data slope')
        ax.axvline(0,color='.55',linewidth=.8)
        ax.set_yticks(y,frame.left_out_study,fontsize=6.2); ax.set_xlabel("leave-one-study-out stiffness slope (95% OLS CI)")
        ax.set_title(parent,loc='left',weight='bold'); ax.grid(axis='x',color='.92'); ax.legend(fontsize=6.5,frameon=False,loc='best')
    fig.text(.5,.01,FOOTER,ha="center",fontsize=7,color=".35"); fig.tight_layout(rect=(0,.04,1,.98))
    paths=[]
    for suffix in ("pdf","png"):
        path=FIG/f"leave_one_study_out.{suffix}"; fig.savefig(path,dpi=200,bbox_inches="tight"); paths.append(path)
    plt.close(fig); return paths


def md_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines=["| "+" | ".join(columns)+" |","|"+"|".join(["---"]*len(columns))+"|"]
    for values in frame.loc[:,columns].itertuples(index=False,name=None):
        cells=[]
        for value in values:
            if isinstance(value,(float,np.floating)):
                cells.append(f"{value:.6g}")
            elif isinstance(value,(bool,np.bool_)):
                cells.append(str(bool(value)).lower())
            else:
                cells.append(str(value).replace('|','\\|'))
        lines.append("| "+" | ".join(cells)+" |")
    return "\n".join(lines)


def write_result(comp_summary: pd.DataFrame, subtype: pd.DataFrame, loso: pd.DataFrame, fixed: pd.DataFrame, verdict: dict[str, dict[str, object]]) -> None:
    rule="""* a hint is **composition-explained** if the subtype slopes (step 3) are both |slope| < half the H04 slope **and** the composition fraction correlates with age at p ≤ 0.05;
* a hint is **study-driven** if leaving out any single study moves p above 0.05 **or** the study-fixed-effects slope is < half the H04 slope in magnitude;
* otherwise **unexplained**."""
    verdict_table=pd.DataFrame([{"ann_level_3": parent, **values} for parent,values in verdict.items()])
    text="EXPLORATORY\n\n"
    text+="## What ran\nH03 primary drift cells were joined to H01 full-metadata `ann_level_4` labels by cell ID. Composition uses all full-metadata cells of each H04 contributing donor. Subtype stiffness uses only the H03 drift subsample at the specified 25-cell floor, 100 deterministic equal-count draws, and 1,000 seed-zero age permutations. Leave-one-study-out analyses reuse H04 donor log-stiffness values without rebuilding operators.\n\n"
    text+="## Reading rule (verbatim)\n"+rule+"\n\n"
    text+="## Composition correlations\n"+md_table(comp_summary,["ann_level_3","fraction_name","n_h04_donors","n_valid_fraction","rho_age","p_age","rho_h04_log_stiffness","p_h04_log_stiffness"])+"\n\n"
    text+="## Subtype stiffness\n"+md_table(subtype,["ann_level_3","ann_level_4","n_donors","stiff_slope","stiff_p","passes","n_common","n_draws","n_perm"])+"\n\n"
    text+="## Study dependence\n"+md_table(fixed,["ann_level_3","n_donors","n_studies","h04_stiff_slope","h04_stiff_p","fe_age_slope"])+"\n\n"
    text+="### Leave-one-study-out\n"+md_table(loso,["ann_level_3","left_out_study","n_donors","stiff_slope","stiff_p","ci95_low","ci95_high"])+"\n\n"
    text+="## Mechanical verdicts\n"+md_table(verdict_table,["ann_level_3","verdict","composition_explained","study_driven","composition_p_age","max_loso_p","h04_slope","fe_slope"])+"\n"
    RESULT.write_text(text)


def run(dry_run: bool) -> dict[str, object]:
    checksums=verify_inputs()
    z,jacobian,obs,full_obs,donor_ops,h04=load_inputs()
    focus,unmatched=add_ann_level_4(obs,full_obs)
    focus["_h03_position"]=np.flatnonzero(obs["ann_level_3"].isin(FOCAL))
    if unmatched:
        raise RuntimeError(f"{unmatched} focus H03 cells are absent from full metadata")
    labels=set(focus["ann_level_4"].astype(str))
    required_labels={label for spec in FOCAL.values() for label in spec["subtypes"]}
    missing=sorted(required_labels-labels)
    if missing:
        raise RuntimeError(f"required ann_level_4 subtype labels missing from H03 focus cells: {missing}")
    plan={"seed":SEED,"n_focus_cells":int(len(focus)),"n_unmatched_cells":unmatched,"n_common":N_COMMON,"n_draws":N_DRAWS,"n_perm":N_PERM,"subtypes":[label for spec in FOCAL.values() for label in spec["subtypes"]],"h04_primary_donors":{parent:int(((donor_ops.analysis=='primary')&(donor_ops.ann_level_3==parent)).sum()) for parent in FOCAL}}
    if dry_run:
        print(json.dumps({**plan,"dry_run":True},indent=2,sort_keys=True)); return plan
    OUT.mkdir(parents=True,exist_ok=True); CHECKSUMS.write_text(json.dumps(checksums,indent=2)+"\n")
    composition,comp_summary=composition_rows(full_obs,donor_ops)
    subtype,subtype_donors=subtype_stiffness(z,jacobian,focus)
    loso,fixed=study_dependence(donor_ops,h04)
    decision=verdicts(comp_summary,subtype,loso,fixed)
    composition.to_csv(COMPOSITION,index=False); comp_summary.to_csv(COMPOSITION_SUMMARY,index=False); subtype.to_csv(SUBTYPE,index=False); subtype_donors.to_csv(SUBTYPE_DONORS,index=False); loso.to_csv(LOSO,index=False); fixed.to_csv(FE,index=False)
    figure_paths=plot_composition(composition,comp_summary)+plot_subtypes(subtype_donors,subtype)+plot_loso(loso,fixed)
    write_result(comp_summary,subtype,loso,fixed,decision)
    output={**plan,"n_unmatched_cells":unmatched,"verdict_t_cell":decision['T cell lineage']['verdict'],"verdict_monocytes":decision['Monocytes']['verdict'],"fe_slope_t_cell":decision['T cell lineage']['fe_slope'],"fe_slope_monocytes":decision['Monocytes']['fe_slope'],"max_loso_p_t_cell":decision['T cell lineage']['max_loso_p'],"max_loso_p_monocytes":decision['Monocytes']['max_loso_p'],"composition":comp_summary.set_index('ann_level_3').to_dict(orient='index'),"subtype_stiffness":subtype.set_index('ann_level_4').to_dict(orient='index'),"n_figures":len(figure_paths)//2,"figure_paths":[str(path.relative_to(ROOT)) for path in figure_paths]}
    METRICS.write_text(json.dumps(output,indent=2,default=str)+"\n")
    print(json.dumps(output,indent=2,default=str,sort_keys=True)); return output


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=SEED)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.seed!=SEED:
        raise ValueError('HX01 is frozen at seed 0')
    run(args.dry_run)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
