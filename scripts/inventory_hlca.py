#!/usr/bin/env python3
"""Build the H01 HLCA donor, age, cell-type, and confound inventory.

This script operates only on metadata in the raw-count AnnData written by
fetch_hlca.py. It does not fit a model or compute an aging statistic.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "benchmarks" / "aging_hlca" / "data" / "hlca_core_full_metadata.h5ad"
OUT = ROOT / "benchmarks" / "aging_hlca" / "inventory"
AGE_BINS = ["<30", "30–44", "45–59", "≥60"]


def choose_column(frame: pd.DataFrame, candidates: Iterable[str], *, required: bool = False) -> str | None:
    """Choose a metadata column using exact then case-insensitive candidate names."""
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    lower = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    if required:
        raise RuntimeError(f"required metadata field missing; tried {list(candidates)}")
    return None


def parse_age(value: object) -> float:
    """Map numeric/range age metadata deterministically to years.

    A numeric field is returned directly. A two-number range becomes its
    arithmetic midpoint. A one-number threshold (for example, '>65') becomes
    the stated boundary. Other values are missing rather than imputed.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    if isinstance(value, (int, float, np.number)):
        value = float(value)
        return value if np.isfinite(value) else float("nan")
    numbers = [float(token) for token in re.findall(r"(?<![A-Za-z])(\d+(?:\.\d+)?)", str(value))]
    if len(numbers) == 1:
        return numbers[0]
    if len(numbers) >= 2:
        return float(np.mean(numbers[:2]))
    return float("nan")


def age_bin(age: float) -> str | None:
    if not np.isfinite(age):
        return None
    if age < 30:
        return "<30"
    if age < 45:
        return "30–44"
    if age < 60:
        return "45–59"
    return "≥60"


def first_nonmissing(values: pd.Series) -> object:
    nonmissing = values.dropna()
    if nonmissing.empty:
        return pd.NA
    return nonmissing.iloc[0]


def joined_values(values: pd.Series) -> str | pd._libs.missing.NAType:
    cleaned = sorted({str(value) for value in values.dropna() if str(value).strip() and str(value) != "nan"})
    return "; ".join(cleaned) if cleaned else pd.NA


def candidate_age_fields(obs: pd.DataFrame) -> list[str]:
    preferred = ["age_or_mean_of_age_range", "age", "age_years", "donor_age"]
    found = [field for field in preferred if field in obs.columns]
    discovered = [str(column) for column in obs.columns if "age" in str(column).lower() and str(column) not in found]
    return found + discovered


def choose_age_field(obs: pd.DataFrame) -> tuple[str, pd.Series, list[dict[str, object]]]:
    candidates = candidate_age_fields(obs)
    if not candidates:
        raise RuntimeError("HLCA core has no candidate age field")
    audit: list[dict[str, object]] = []
    selected: tuple[str, pd.Series] | None = None
    for field in candidates:
        converted = obs[field].map(parse_age)
        usable = int(converted.notna().sum())
        audit.append({"field": field, "n_parseable_cells": usable, "n_unique_parseable": int(converted.nunique(dropna=True))})
        if usable and selected is None:
            selected = (field, converted)
    if selected is None:
        raise RuntimeError(f"candidate age fields have no parseable values: {audit}")
    return selected[0], selected[1], audit


def examples(values: pd.Series) -> str:
    seen = []
    for value in values.dropna().astype(str):
        if value not in seen:
            seen.append(value)
        if len(seen) == 5:
            break
    return " | ".join(seen)


def main() -> int:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"run scripts/fetch_hlca.py first: {DATA_PATH}")
    OUT.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(DATA_PATH, backed="r")
    obs = adata.obs.copy()
    if obs.empty:
        raise RuntimeError("HLCA core has no observation metadata")

    donor_field = choose_column(obs, ["donor_id", "donor", "subject_id", "individual_id", "donor_id_or_uuid"], required=True)
    age_field, age_years, age_audit = choose_age_field(obs)
    study_field = choose_column(obs, ["dataset", "study", "dataset_id", "study_id"], required=True)
    sex_field = choose_column(obs, ["sex", "sex_ontology_term_id"])
    smoking_field = choose_column(obs, ["smoking_status", "smoking", "smoker_status"])
    bmi_field = choose_column(obs, ["BMI", "bmi", "body_mass_index"])
    sampling_field = choose_column(obs, ["anatomical_region_ccf_score", "tissue", "anatomical_region", "sampling_site"])
    assay_field = choose_column(obs, ["assay", "assay_ontology_term_id", "assay_type"])
    annotation_fields = [field for field in ("ann_level_3", "ann_level_4", "cell_type") if field in obs.columns]
    if not annotation_fields:
        raise RuntimeError("HLCA core lacks ann_level_3, ann_level_4, and cell_type")

    obs_columns = pd.DataFrame(
        {
            "name": [str(column) for column in obs.columns],
            "dtype": [str(obs[column].dtype) for column in obs.columns],
            "n_unique": [int(obs[column].nunique(dropna=True)) for column in obs.columns],
            "n_missing": [int(obs[column].isna().sum()) for column in obs.columns],
            "example_values": [examples(obs[column]) for column in obs.columns],
        }
    )
    roles = {age_field: "selected_age_field", donor_field: "donor_id", study_field: "study_or_dataset"}
    for field in annotation_fields:
        roles[field] = "annotation_level"
    obs_columns["selected_role"] = obs_columns["name"].map(roles).fillna("")
    obs_columns.to_csv(OUT / "obs_columns.csv", index=False)

    cell_meta = pd.DataFrame(index=obs.index)
    cell_meta["_donor"] = obs[donor_field].astype("string")
    cell_meta["_age_years"] = age_years
    cell_meta["_study"] = obs[study_field].astype("string")
    for label, field in {
        "sex": sex_field,
        "smoking_status": smoking_field,
        "BMI": bmi_field,
        "sampling_site": sampling_field,
        "assay": assay_field,
    }.items():
        cell_meta[label] = obs[field] if field is not None else pd.NA
    cell_meta = cell_meta[cell_meta["_donor"].notna() & (cell_meta["_donor"].astype(str) != "")].copy()
    if cell_meta.empty:
        raise RuntimeError(f"donor field {donor_field!r} contains no usable donor IDs")

    grouped = cell_meta.groupby("_donor", observed=True, sort=True)
    donors = pd.DataFrame(index=grouped.size().index)
    donors.index.name = "donor_id"
    donors["age_years"] = grouped["_age_years"].median()
    donors["sex"] = grouped["sex"].agg(joined_values)
    donors["smoking_status"] = grouped["smoking_status"].agg(joined_values)
    donors["BMI"] = grouped["BMI"].agg(first_nonmissing)
    donors["study_or_dataset"] = grouped["_study"].agg(joined_values)
    donors["sampling_sites"] = grouped["sampling_site"].agg(joined_values)
    donors["assay"] = grouped["assay"].agg(joined_values)
    donors["n_cells"] = grouped.size()
    donors = donors.reset_index()
    donors["BMI"] = donors["BMI"].map(parse_age)
    donors["age_bin"] = donors["age_years"].map(age_bin)
    conversion_rule = "numeric values used directly; two-endpoint ranges converted to midpoint; one-bound thresholds converted to stated boundary; unparseable values missing"
    donors["age_conversion"] = conversion_rule
    donors = donors[
        ["donor_id", "age_years", "age_bin", "age_conversion", "sex", "smoking_status", "BMI", "study_or_dataset", "sampling_sites", "assay", "n_cells"]
    ].sort_values(["study_or_dataset", "donor_id"], kind="stable")
    donors.to_csv(OUT / "donors.csv", index=False)
    donors_with_age = donors[donors["age_years"].notna()].copy()
    if donors_with_age.empty:
        raise RuntimeError(f"selected age field {age_field!r} has no donor-level ages")

    age_rows: list[dict[str, object]] = []
    for study, frame in donors_with_age.groupby("study_or_dataset", observed=True, sort=True):
        counts = frame["age_bin"].value_counts()
        age_rows.append(
            {
                "study_or_dataset": study,
                "n_donors": int(len(frame)),
                "n_donors_lt30": int(counts.get("<30", 0)),
                "n_donors_30_44": int(counts.get("30–44", 0)),
                "n_donors_45_59": int(counts.get("45–59", 0)),
                "n_donors_ge60": int(counts.get("≥60", 0)),
                "age_min": float(frame["age_years"].min()),
                "age_median": float(frame["age_years"].median()),
                "age_max": float(frame["age_years"].max()),
            }
        )
    age_by_study = pd.DataFrame(age_rows).sort_values("study_or_dataset", kind="stable")
    age_by_study.to_csv(OUT / "age_by_study.csv", index=False)

    donor_lookup = donors.set_index("donor_id")[["age_years", "age_bin", "study_or_dataset"]]
    cell_meta = cell_meta.join(donor_lookup, on="_donor", rsuffix="_donor")
    cell_meta = cell_meta.rename(columns={"age_years": "_donor_age", "age_bin": "_age_bin", "study_or_dataset": "_donor_study"})
    for field in annotation_fields:
        cell_meta[field] = obs.loc[cell_meta.index, field].astype("string")

    celltype_rows: list[pd.DataFrame] = []
    for field in annotation_fields:
        frame = cell_meta[cell_meta["_age_bin"].notna() & cell_meta[field].notna()].copy()
        summary = (
            frame.groupby([field, "_age_bin"], observed=True)
            .agg(n_cells=("_donor", "size"), n_donors=("_donor", "nunique"))
            .reset_index()
            .rename(columns={field: "cell_type", "_age_bin": "age_bin"})
        )
        summary.insert(0, "annotation_level", field)
        celltype_rows.append(summary)
    celltype_by_agebin = pd.concat(celltype_rows, ignore_index=True).sort_values(
        ["annotation_level", "cell_type", "age_bin"], kind="stable"
    )
    celltype_by_agebin.to_csv(OUT / "celltype_by_agebin.csv", index=False)

    confound_rows: list[dict[str, object]] = []
    for covariate in ("smoking_status", "sex", "sampling_sites", "assay"):
        for (bin_name, level), count in donors_with_age.assign(_level=donors_with_age[covariate].astype("string").fillna("MISSING").astype(str)).groupby(
            ["age_bin", "_level"], observed=True
        ).size().items():
            confound_rows.append(
                {
                    "analysis": "age_bin_crosstab",
                    "covariate": covariate,
                    "age_bin": bin_name,
                    "level": level,
                    "n_donors": int(count),
                    "spearman_rho": np.nan,
                    "spearman_p": np.nan,
                    "n_complete": np.nan,
                }
            )
    numeric_covariates = {"BMI": donors_with_age["BMI"]}
    for covariate, values in numeric_covariates.items():
        usable = pd.DataFrame({"age": donors_with_age["age_years"], "value": pd.to_numeric(values, errors="coerce")}).dropna()
        if len(usable) >= 3 and usable["value"].nunique() >= 2:
            rho, p_value = spearmanr(usable["age"], usable["value"])
        else:
            rho, p_value = np.nan, np.nan
        confound_rows.append(
            {
                "analysis": "spearman_age_numeric_covariate",
                "covariate": covariate,
                "age_bin": "",
                "level": "",
                "n_donors": np.nan,
                "spearman_rho": rho,
                "spearman_p": p_value,
                "n_complete": int(len(usable)),
            }
        )
    pd.DataFrame(confound_rows).sort_values(["analysis", "covariate", "age_bin", "level"], kind="stable").to_csv(OUT / "confounds.csv", index=False)

    contrast_donors = donors_with_age.assign(_side=np.where(donors_with_age["age_years"] < 45, "<45", np.where(donors_with_age["age_years"] >= 60, "≥60", pd.NA)))
    contrast_donors = contrast_donors[contrast_donors["_side"].notna()].copy()
    study_side = contrast_donors.groupby(["study_or_dataset", "_side"], observed=True).size().unstack(fill_value=0)
    eligible_studies = study_side[(study_side.get("<45", 0) > 0) & (study_side.get("≥60", 0) > 0)].copy()
    contrast_rows: list[dict[str, object]] = []
    pooled_good: set[tuple[str, str]] = set()
    cell_meta["_side"] = np.where(cell_meta["_donor_age"] < 45, "<45", np.where(cell_meta["_donor_age"] >= 60, "≥60", pd.NA))
    for field in annotation_fields:
        current = cell_meta[cell_meta["_side"].notna() & cell_meta[field].notna()].copy()
        pooled = current.groupby([field, "_side"], observed=True).agg(n_cells=("_donor", "size")).unstack(fill_value=0)
        for cell_type, values in pooled.iterrows():
            young_n = int(values.get(("n_cells", "<45"), 0))
            old_n = int(values.get(("n_cells", "≥60"), 0))
            if young_n >= 200 and old_n >= 200:
                pooled_good.add((field, str(cell_type)))
        for study, sides in eligible_studies.iterrows():
            subset = current[current["_donor_study"].astype(str) == str(study)]
            counts = subset.groupby([field, "_side"], observed=True).agg(n_cells=("_donor", "size")).unstack(fill_value=0)
            for cell_type, values in counts.iterrows():
                n_lt45 = int(values.get(("n_cells", "<45"), 0))
                n_ge60 = int(values.get(("n_cells", "≥60"), 0))
                contrast_rows.append(
                    {
                        "study_or_dataset": study,
                        "annotation_level": field,
                        "cell_type": cell_type,
                        "n_donors_lt45": int(sides.get("<45", 0)),
                        "n_donors_ge60": int(sides.get("≥60", 0)),
                        "n_cells_lt45": n_lt45,
                        "n_cells_ge60": n_ge60,
                        "meets_200_cells_both_sides": bool(n_lt45 >= 200 and n_ge60 >= 200),
                    }
                )
    within = pd.DataFrame(contrast_rows, columns=[
        "study_or_dataset", "annotation_level", "cell_type", "n_donors_lt45", "n_donors_ge60", "n_cells_lt45", "n_cells_ge60", "meets_200_cells_both_sides"
    ])
    within.to_csv(OUT / "within_study_contrast.csv", index=False)

    decisions = {
        "task": "hlca/H01",
        "source_h5ad": str(DATA_PATH.relative_to(ROOT)),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "donor_field": donor_field,
        "age_field": age_field,
        "age_field_rationale": "Selected the first preferred HLCA age field with parseable donor-level values; numeric values retain their stated years and ranges use deterministic midpoints.",
        "age_field_candidates": age_audit,
        "age_conversion": conversion_rule,
        "study_field": study_field,
        "selected_metadata_fields": {
            "sex": sex_field,
            "smoking_status": smoking_field,
            "BMI": bmi_field,
            "sampling_site": sampling_field,
            "assay": assay_field,
        },
        "annotation_levels": annotation_fields,
        "n_donors": int(len(donors)),
        "n_donors_with_age": int(len(donors_with_age)),
        "n_donors_lt45": int((donors_with_age["age_years"] < 45).sum()),
        "n_donors_ge60": int((donors_with_age["age_years"] >= 60).sum()),
        "n_studies_with_within_study_contrast": int(len(eligible_studies)),
        "n_celltypes_ge200_both_sides_pooled": int(len(pooled_good)),
    }
    (OUT / "inventory_decisions.json").write_text(json.dumps(decisions, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decisions, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
