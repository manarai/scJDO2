#!/usr/bin/env python3
"""H04: frozen donor-level HLCA aging-basin tests (PREREG §§4–§7).

No model is fitted.  This script consumes only the H03 per-cell drift outputs,
constructs deterministic equal-count donor operators once per requested analysis
and type, runs exactly 1,000 age permutations with seed zero, writes all H04
outputs, and mechanically derives the fixed §6 outcome header.
"""
from __future__ import annotations

import argparse
import json
import math
import zlib
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from scjdo2.compare import (
    anticipation_trend,
    centroid_operator,
    donor_operator,
    isotropy_trend,
    stiffness_trend,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "aging_hlca"
PRIMARY_INPUT = OUT / "data" / "h03_primary_drift.h5ad"
SECONDARY_INPUT = OUT / "data" / "h03_secondary_drift.h5ad"
ELIGIBILITY = OUT / "eligibility.csv"
DONOR_OPERATORS = OUT / "donor_operators.csv"
PRIMARY_TABLE = OUT / "tests_primary.csv"
SENSITIVITY_TABLE = OUT / "tests_sensitivity.csv"
RESULT = OUT / "RESULT.md"

SEED = 0
N_COMMON = 50
N_DRAWS = 100
N_CENTROID = 30
N_PERM = 1_000
MIN_DONORS = 20
MIN_LT45 = 5
MIN_GE60 = 5
EPS = 1e-12

PRIMARY_TYPES = (
    "AT2", "Basal", "Dendritic cells", "EC arterial", "EC capillary",
    "EC venous", "Fibroblasts", "Innate lymphoid cell NK", "Macrophages",
    "Monocytes", "Multiciliated lineage", "Secretory", "T cell lineage",
)
ANALYSIS_ORDER = (
    "primary", "a_within_study", "b_banovich", "c_parenchyma",
    "d_never_smokers", "e_secondary_space", "f_two_group",
)
TEST_COLUMNS = (
    "ann_level_3", "n_donors", "n_lt45", "n_ge60", "iso_S", "iso_q95",
    "iso_p", "iso_pass", "median_beta", "ant_stat", "ant_q95", "ant_p",
    "ant_pass", "stiff_slope", "stiff_p", "stiff_class",
)


def representation_and_jacobian(adata: ad.AnnData, expected_dim: int) -> tuple[np.ndarray, np.ndarray, str]:
    candidates = [key for key in adata.obsm.keys() if key != "X_drift_v"]
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one source representation, found {candidates}")
    key = candidates[0]
    z = np.asarray(adata.obsm[key])
    scj = adata.uns.get("scjdo2", {})
    j = np.asarray(scj.get("J_latent", scj.get("J")))
    if z.shape != (adata.n_obs, expected_dim) or j.shape != (adata.n_obs, expected_dim, expected_dim):
        raise RuntimeError(f"unexpected representation/Jacobian dimensions: z={z.shape}, J={j.shape}")
    if not np.isfinite(z).all() or not np.isfinite(j).all():
        raise RuntimeError("H03 representation or Jacobian contains a non-finite value")
    return z, j, key


def h03_eligibility() -> dict[str, tuple[str, ...]]:
    table = pd.read_csv(ELIGIBILITY)
    required = {"analysis", "ann_level_3", "n_donors", "n_lt45", "n_ge60", "eligible"}
    if set(table.columns) != required:
        raise RuntimeError(f"unexpected H03 eligibility columns: {table.columns.tolist()}")
    expected = {"primary", "banovich", "parenchyma", "never_smokers"}
    if set(table["analysis"]) != expected:
        raise RuntimeError("unexpected H03 eligibility analyses")
    output: dict[str, tuple[str, ...]] = {}
    for analysis, group in table.groupby("analysis", sort=True):
        output[str(analysis)] = tuple(group.loc[group["eligible"], "ann_level_3"].astype(str).tolist())
    if output["primary"] != PRIMARY_TYPES:
        raise RuntimeError(f"H03 primary eligibility no longer matches the frozen 13-type list: {output['primary']}")
    return output


def validate_obs(adata: ad.AnnData) -> pd.DataFrame:
    needed = {
        "donor_id", "ann_level_3", "age", "study_or_dataset",
        "anatomical_region_ccf_score", "smoking_status",
    }
    missing = sorted(needed - set(adata.obs.columns))
    if missing:
        raise RuntimeError(f"H03 drift obs missing columns: {missing}")
    obs = adata.obs.loc[:, sorted(needed)].copy()
    obs.index = obs.index.astype(str)
    obs["donor_id"] = obs["donor_id"].astype(str)
    obs["ann_level_3"] = obs["ann_level_3"].astype(str)
    obs["study_or_dataset"] = obs["study_or_dataset"].astype(str)
    obs["smoking_status"] = obs["smoking_status"].astype(str)
    obs["age"] = pd.to_numeric(obs["age"], errors="raise")
    obs["anatomical_region_ccf_score"] = pd.to_numeric(obs["anatomical_region_ccf_score"], errors="coerce")
    if obs.index.duplicated().any() or obs["donor_id"].eq("nan").any():
        raise RuntimeError("drift sample has duplicate cells or a missing donor")
    donor_meta = obs.groupby("donor_id", sort=True).agg(
        age=("age", "nunique"), study_or_dataset=("study_or_dataset", "nunique")
    )
    if not donor_meta.eq(1).all().all():
        raise RuntimeError("age or study_or_dataset varies within a donor")
    return obs


def eligibility_for_mask(obs: pd.DataFrame, mask: np.ndarray) -> tuple[tuple[str, ...], dict[str, dict[str, int]]]:
    """Reapply the frozen equal-count donor eligibility on a concrete cell mask."""
    rows = obs.loc[np.asarray(mask, dtype=bool)]
    details: dict[str, dict[str, int]] = {}
    types: list[str] = []
    for cell_type, group in rows.groupby("ann_level_3", sort=True):
        donors = group.groupby("donor_id", sort=True).agg(age=("age", "first"), n_cells=("age", "size"))
        contributors = donors.loc[donors["n_cells"] >= N_COMMON]
        record = {
            "n_donors": int(len(contributors)),
            "n_lt45": int((contributors["age"] < 45).sum()),
            "n_ge60": int((contributors["age"] >= 60).sum()),
        }
        details[str(cell_type)] = record
        if (record["n_donors"] >= MIN_DONORS and record["n_lt45"] >= MIN_LT45 and record["n_ge60"] >= MIN_GE60):
            types.append(str(cell_type))
    return tuple(types), details


def metrics_for_operator(op: np.ndarray) -> tuple[float, float, float]:
    values = np.linalg.eigvalsh(0.5 * (op + op.T))
    lead = float(values.max())
    median_abs = float(np.median(np.abs(values)))
    return float(math.log(median_abs)), lead, float(median_abs / (abs(lead) + EPS))


def semicolon(values: np.ndarray) -> str:
    return ";".join(format(float(value), ".17g") for value in np.asarray(values).ravel())


def donor_records(
    *, analysis: str, obs: pd.DataFrame, z: np.ndarray, j: np.ndarray,
    cell_mask: np.ndarray, types: tuple[str, ...], type_details: dict[str, dict[str, int]],
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """Build frozen donor operators and subset-specific pooled-young references."""
    all_type = obs["ann_level_3"].to_numpy(dtype=str)
    all_donor = obs["donor_id"].to_numpy(dtype=str)
    all_age = obs["age"].to_numpy(dtype=float)
    all_study = obs["study_or_dataset"].to_numpy(dtype=str)
    mask = np.asarray(cell_mask, dtype=bool)
    rows: list[dict] = []
    grouped: dict[str, dict[str, object]] = {}
    for cell_type in types:
        type_mask = mask & (all_type == cell_type)
        young = type_mask & (all_age < 45)
        if int(young.sum()) < N_CENTROID:
            raise RuntimeError(f"{analysis}/{cell_type} has fewer than {N_CENTROID} pooled young cells")
        reference = centroid_operator(z[young], j[young], n_cells_centroid=N_CENTROID)
        contributors: list[dict] = []
        for donor in sorted(set(all_donor[type_mask])):
            idx = np.flatnonzero(type_mask & (all_donor == donor))
            if len(idx) < N_COMMON:
                continue
            donor_age = float(all_age[idx[0]])
            donor_study = str(all_study[idx[0]])
            seeded = zlib.crc32(f"{cell_type}|{donor}".encode())
            operator, centroid = donor_operator(
                z[idx], j[idx], n_common=N_COMMON, n_draws=N_DRAWS,
                n_cells_centroid=N_CENTROID, seed=seeded,
            )
            log_median, lead, anisotropy = metrics_for_operator(operator)
            rows.append({
                "analysis": analysis, "ann_level_3": cell_type, "donor_id": donor,
                "age": donor_age, "study_or_dataset": donor_study,
                "log_median_abs_lambda": log_median, "lead_eig": lead,
                "anisotropy": anisotropy, "centroid": semicolon(centroid),
            })
            contributors.append({
                "donor_id": donor, "age": donor_age, "study_or_dataset": donor_study,
                "operator": operator, "centroid": centroid,
            })
        observed = {
            "n_donors": len(contributors),
            "n_lt45": sum(item["age"] < 45 for item in contributors),
            "n_ge60": sum(item["age"] >= 60 for item in contributors),
        }
        if cell_type in type_details and observed != type_details[cell_type]:
            raise RuntimeError(f"contributor-count disagreement for {analysis}/{cell_type}: {observed} vs {type_details[cell_type]}")
        grouped[cell_type] = {"reference": reference, "contributors": contributors}
    return pd.DataFrame(rows), grouped


def classify_stiffness(slope: float, p_value: float) -> str:
    if slope < 0 and p_value <= 0.05:
        return "loss"
    if slope > 0 and p_value <= 0.05:
        return "gain"
    return "none"


def test_records(
    *, grouped: dict[str, dict[str, object]], types: tuple[str, ...],
    strata: bool = False, binary_ages: bool = False,
) -> pd.DataFrame:
    rows: list[dict] = []
    for cell_type in types:
        item = grouped[cell_type]
        contributors = item["contributors"]
        operators = [x["operator"] for x in contributors]
        centroids = [x["centroid"] for x in contributors]
        ages = np.asarray([x["age"] for x in contributors], dtype=float)
        permutation_ages = (ages >= 60).astype(float) if binary_ages else ages
        stratum = [x["study_or_dataset"] for x in contributors] if strata else None
        iso = isotropy_trend(operators, permutation_ages, item["reference"], n_perm=N_PERM, seed=SEED, strata=stratum)
        ant = anticipation_trend(centroids, permutation_ages, item["reference"], n_perm=N_PERM, seed=SEED, strata=stratum)
        stiff = stiffness_trend(operators, permutation_ages, n_perm=N_PERM, seed=SEED, strata=stratum)
        if not np.isfinite([iso.statistic, iso.null_q95, iso.p_value, ant.statistic, ant.null_q95, ant.p_value, stiff.statistic, stiff.p_value]).all():
            raise RuntimeError(f"non-finite frozen test metric for {cell_type}")
        beta = np.asarray(iso.detail.get("beta"), dtype=float)
        if beta.size == 0 or not np.isfinite(beta).all():
            raise RuntimeError(f"non-finite isotropy beta for {cell_type}")
        rows.append({
            "ann_level_3": cell_type,
            "n_donors": len(contributors),
            "n_lt45": int((ages < 45).sum()),
            "n_ge60": int((ages >= 60).sum()),
            "iso_S": float(iso.statistic), "iso_q95": float(iso.null_q95), "iso_p": float(iso.p_value), "iso_pass": bool(iso.passes),
            "median_beta": float(np.median(beta)),
            "ant_stat": float(ant.statistic), "ant_q95": float(ant.null_q95), "ant_p": float(ant.p_value), "ant_pass": bool(ant.passes),
            "stiff_slope": float(stiff.statistic), "stiff_p": float(stiff.p_value),
            "stiff_class": classify_stiffness(float(stiff.statistic), float(stiff.p_value)),
        })
    return pd.DataFrame(rows, columns=TEST_COLUMNS)


def derive_outcome(primary: pd.DataFrame) -> tuple[str, str, str]:
    n = len(primary)
    if n != len(PRIMARY_TYPES):
        raise RuntimeError(f"primary outcome requires 13 types, received {n}")
    iso_majority = int(primary["iso_pass"].sum()) / n > 0.5
    ant_majority = int(primary["ant_pass"].sum()) / n > 0.5
    headers = {
        (True, True): "directional, anticipatory loss",
        (True, False): "specific loosening without anticipation",
        (False, False): "isotropic noise increase; no aging claim beyond stiffness",
        (False, True): "anticipation without specific loosening (corner case; report as such)",
    }
    loss_majority = int(primary["stiff_class"].eq("loss").sum()) / n > 0.5
    gain_majority = int(primary["stiff_class"].eq("gain").sum()) / n > 0.5
    qualifier = (
        "— with stiffness loss in the majority of types" if loss_majority else
        "— with stiffness gain in the majority of types" if gain_majority else
        "— no majority stiffness change"
    )
    return headers[(iso_majority, ant_majority)], qualifier, f"**{headers[(iso_majority, ant_majority)]}** {qualifier}"


def markdown_table(frame: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for values in frame.loc[:, list(columns)].itertuples(index=False, name=None):
        formatted: list[str] = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                formatted.append(f"{float(value):.6g}")
            elif isinstance(value, (bool, np.bool_)):
                formatted.append("true" if value else "false")
            else:
                formatted.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def write_result(
    *, primary: pd.DataFrame, sensitivity: pd.DataFrame,
    h03_eligible: dict[str, tuple[str, ...]], reapplied: dict[str, tuple[str, ...]],
    outcome: str,
) -> None:
    primary_set = set(PRIMARY_TYPES)
    sensitivity_summary: list[dict] = []
    for analysis in ANALYSIS_ORDER[1:]:
        rows = sensitivity.loc[sensitivity["analysis"].eq(analysis)]
        eligible = tuple(rows["ann_level_3"].astype(str).tolist())
        lost = sorted(primary_set - set(eligible))
        sensitivity_summary.append({
            "analysis": analysis,
            "n_types": len(eligible),
            "eligible_types": "; ".join(eligible),
            "primary_types_not_tested": "; ".join(lost) if lost else "none",
        })
    text = "# HLCA aging basins — H04 frozen tests\n\n"
    text += "## §4 Eligible types and operators\n"
    text += "As executed: `n_common=50`, 100 deterministic equal-count donor draws, 30-cell centroid operators, and seed `crc32(type|donor)` were used. The primary eligibility set is: " + "; ".join(PRIMARY_TYPES) + ".\n\n"
    text += "| Analysis | Eligible types | n eligible |\n|---|---|---:|\n"
    text += f"| primary | {'; '.join(PRIMARY_TYPES)} | {len(PRIMARY_TYPES)} |\n"
    for key, label in (("a_within_study", "a_within_study"), ("b_banovich", "b_banovich"), ("c_parenchyma", "c_parenchyma"), ("d_never_smokers", "d_never_smokers")):
        text += f"| {label} | {'; '.join(reapplied[key])} | {len(reapplied[key])} |\n"
    text += f"| e_secondary_space | {'; '.join(PRIMARY_TYPES)} | {len(PRIMARY_TYPES)} |\n"
    text += f"| f_two_group | {'; '.join(PRIMARY_TYPES)} | {len(PRIMARY_TYPES)} |\n\n"
    text += "`donor_operators.csv` records every contributing donor/type/operator under each analysis.\n\n"
    text += "## §5 Outcome\n"
    text += outcome + "\n\n"
    text += "The header and qualifier above are the fixed §6 rule applied to `tests_primary.csv`; no sensitivity result changes them.\n\n"
    text += "## §6 Per-type table\n"
    text += markdown_table(primary, TEST_COLUMNS) + "\n\n"
    text += "## §7 Sensitivity table\n"
    text += "### Eligibility after subset restriction\n"
    text += markdown_table(pd.DataFrame(sensitivity_summary), ("analysis", "n_types", "primary_types_not_tested")) + "\n\n"
    text += "### Tests as executed\n"
    text += markdown_table(sensitivity, ("analysis",) + TEST_COLUMNS) + "\n"
    RESULT.write_text(text)


def run() -> dict[str, object]:
    eligibility = h03_eligibility()
    primary_adata = ad.read_h5ad(PRIMARY_INPUT)
    secondary_adata = ad.read_h5ad(SECONDARY_INPUT)
    primary_z, primary_j, primary_rep = representation_and_jacobian(primary_adata, 15)
    secondary_z, secondary_j, secondary_rep = representation_and_jacobian(secondary_adata, 30)
    primary_obs = validate_obs(primary_adata)
    secondary_obs = validate_obs(secondary_adata)
    if not primary_obs.index.equals(secondary_obs.index) or not primary_obs.equals(secondary_obs):
        raise RuntimeError("primary and secondary H03 drift metadata disagree")
    obs = primary_obs
    full = np.ones(len(obs), dtype=bool)
    masks = {
        "primary": full,
        "a_within_study": full,
        "b_banovich": obs["study_or_dataset"].eq("Banovich_Kropski_2020").to_numpy(),
        "c_parenchyma": obs["anatomical_region_ccf_score"].eq(0.97).to_numpy(),
        "d_never_smokers": obs["smoking_status"].eq("never").to_numpy(),
        "e_secondary_space": full,
        "f_two_group": (obs["age"].lt(45) | obs["age"].ge(60)).to_numpy(),
    }
    reapplied: dict[str, tuple[str, ...]] = {}
    for analysis, expected_h03 in (("primary", "primary"), ("a_within_study", "primary"), ("b_banovich", "banovich"), ("c_parenchyma", "parenchyma"), ("d_never_smokers", "never_smokers")):
        found, _ = eligibility_for_mask(obs, masks[analysis])
        if found != eligibility[expected_h03]:
            raise RuntimeError(f"H04 re-applied eligibility differs from H03 for {analysis}: {found} vs {eligibility[expected_h03]}")
        reapplied[analysis] = found
    reapplied["e_secondary_space"] = PRIMARY_TYPES
    reapplied["f_two_group"] = PRIMARY_TYPES

    all_operator_rows: list[pd.DataFrame] = []
    tests: dict[str, pd.DataFrame] = {}
    for analysis in ANALYSIS_ORDER:
        types = reapplied[analysis]
        mask = masks[analysis]
        if analysis == "e_secondary_space":
            z, j = secondary_z, secondary_j
        else:
            z, j = primary_z, primary_j
        _, details = eligibility_for_mask(obs, mask)
        donors, grouped = donor_records(
            analysis=analysis, obs=obs, z=z, j=j, cell_mask=mask,
            types=types, type_details=details,
        )
        if analysis in {"primary", "a_within_study", "b_banovich", "c_parenchyma", "d_never_smokers", "e_secondary_space"} and donors.empty:
            raise RuntimeError(f"no donor operator rows for {analysis}")
        all_operator_rows.append(donors)
        tests[analysis] = test_records(
            grouped=grouped, types=types,
            strata=(analysis == "a_within_study"), binary_ages=(analysis == "f_two_group"),
        )

    primary = tests["primary"].copy()
    if tuple(primary["ann_level_3"]) != PRIMARY_TYPES or (primary["n_donors"] < MIN_DONORS).any():
        raise RuntimeError("primary tests do not retain the frozen 13 types with ≥20 contributing donors")
    outcome_header, qualifier, outcome = derive_outcome(primary)
    sensitivity = pd.concat(
        [tests[name].assign(analysis=name) for name in ANALYSIS_ORDER[1:]], ignore_index=True
    ).loc[:, ("analysis",) + TEST_COLUMNS]
    operators = pd.concat(all_operator_rows, ignore_index=True).sort_values(
        ["analysis", "ann_level_3", "donor_id"], kind="stable"
    )
    # Reorder rows deterministically by frozen analysis order, then type/donor.
    operators["analysis"] = pd.Categorical(operators["analysis"], categories=ANALYSIS_ORDER, ordered=True)
    operators = operators.sort_values(["analysis", "ann_level_3", "donor_id"], kind="stable").reset_index(drop=True)
    operators["analysis"] = operators["analysis"].astype(str)
    sensitivity["analysis"] = pd.Categorical(sensitivity["analysis"], categories=ANALYSIS_ORDER, ordered=True)
    sensitivity = sensitivity.sort_values(["analysis", "ann_level_3"], kind="stable").reset_index(drop=True)
    sensitivity["analysis"] = sensitivity["analysis"].astype(str)
    DONOR_OPERATORS.parent.mkdir(parents=True, exist_ok=True)
    operators.to_csv(DONOR_OPERATORS, index=False)
    primary.to_csv(PRIMARY_TABLE, index=False)
    sensitivity.to_csv(SENSITIVITY_TABLE, index=False)
    write_result(primary=primary, sensitivity=sensitivity, h03_eligible=eligibility, reapplied=reapplied, outcome=outcome)
    output = {
        "n_eligible": int(len(primary)),
        "iso_pass_count": int(primary["iso_pass"].sum()),
        "ant_pass_count": int(primary["ant_pass"].sum()),
        "stiff_loss_count": int(primary["stiff_class"].eq("loss").sum()),
        "stiff_gain_count": int(primary["stiff_class"].eq("gain").sum()),
        "outcome_header": outcome_header,
        "stiffness_qualifier": qualifier,
        "sensitivity_iso_pass_by_analysis": {name: int(tests[name]["iso_pass"].sum()) for name in ANALYSIS_ORDER[1:]},
        "sensitivity_ant_pass_by_analysis": {name: int(tests[name]["ant_pass"].sum()) for name in ANALYSIS_ORDER[1:]},
        "sensitivity_stiff_loss_by_analysis": {name: int(tests[name]["stiff_class"].eq("loss").sum()) for name in ANALYSIS_ORDER[1:]},
        "n_sensitivity_types_by_analysis": {name: int(len(tests[name])) for name in ANALYSIS_ORDER[1:]},
        "reapplied_eligible_types": {name: list(value) for name, value in reapplied.items()},
        "representations": {"primary": primary_rep, "secondary": secondary_rep},
        "n_donor_operator_rows": int(len(operators)),
        "seed": SEED,
    }
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.seed != SEED:
        raise ValueError("H04 is frozen at seed 0")
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
