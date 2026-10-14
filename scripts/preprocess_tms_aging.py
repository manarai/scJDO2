"""Preprocess TMS aging data under PREREG_aging_basins.md §2b.

For a selected tissue this script excludes 4-week cells, applies the donor/type
bookkeeping floor, balances the larger age group per cell type with deterministic
donor-balanced sampling, retains the frozen >=200-per-age eligible types, writes
QC audits, and stores a 2,000-HVG AnnData object.
"""
from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

YOUNG_LABELS = {"3 month-old stage", "3 months"}
OLD_LABELS = {"18 month-old stage", "20 month-old stage and over"}
EXCLUDED_LABEL = "4 weeks"
AGE_GROUPS = ("young", "old")
DONOR_TYPE_FLOOR = 20
MIN_ELIGIBLE_CELLS = 200
N_HVGS = 2000


def _seed_for(seed: int, tissue: str, cell_type: str) -> int:
    """Give each tissue/type a stable, reproducible sampling stream."""
    return (seed + zlib.crc32(f"{tissue}|{cell_type}".encode())) % (2**32)


def _sample_evenly_by_donor(frame: pd.DataFrame, target: int, rng: np.random.Generator) -> pd.Index:
    """Sample exactly ``target`` rows while water-filling across donor capacities.

    At each draw round, donors with the smallest current contribution and spare
    cells are considered first.  Ties are randomized with the task seed, so
    unsaturated donors differ by at most one contributed cell whenever their
    available counts permit it.
    """
    by_donor = {
        str(donor): group.index.to_numpy(copy=True)
        for donor, group in frame.groupby("donor_id", sort=True, observed=True)
    }
    available = {donor: len(indices) for donor, indices in by_donor.items()}
    if target > sum(available.values()):
        raise ValueError(f"target={target} exceeds available cells={sum(available.values())}")

    allocation = {donor: 0 for donor in by_donor}
    remaining = target
    while remaining:
        active = [donor for donor in sorted(by_donor) if allocation[donor] < available[donor]]
        if not active:
            raise RuntimeError("donor-balanced sampler ran out of cells")
        least = min(allocation[donor] for donor in active)
        tied = [donor for donor in active if allocation[donor] == least]
        for donor in rng.permutation(np.asarray(tied, dtype=object)):
            donor = str(donor)
            if allocation[donor] >= available[donor]:
                continue
            allocation[donor] += 1
            remaining -= 1
            if remaining == 0:
                break

    selected = [
        rng.choice(by_donor[donor], size=n_cells, replace=False)
        for donor, n_cells in allocation.items()
        if n_cells
    ]
    return pd.Index(np.concatenate(selected))


def _age_group(stage: pd.Series) -> pd.Series:
    stage = stage.astype(str)
    group = pd.Series(pd.NA, index=stage.index, dtype="string")
    group.loc[stage.isin(YOUNG_LABELS)] = "young"
    group.loc[stage.isin(OLD_LABELS)] = "old"
    return group


def _qc_counts(frame: pd.DataFrame, column: str, tissue: str) -> pd.DataFrame:
    columns = ["age_group", "donor_id", "cell_type"]
    out = frame.groupby(columns, observed=True).size().rename(column).reset_index()
    out.insert(0, "tissue", tissue)
    return out.sort_values(columns, kind="stable").reset_index(drop=True)


def preprocess(tissue: str, seed: int, root: Path) -> dict[str, int]:
    data_dir = root / "benchmarks" / "aging" / "data"
    qc_dir = root / "benchmarks" / "aging" / "qc"
    source_path = data_dir / f"{tissue}.h5ad"
    processed_path = data_dir / f"{tissue}_processed.h5ad"
    cell_filter_path = root / "benchmarks" / "aging" / f"{tissue}_cell_filter.csv"
    qc_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(source_path)
    required_obs = {"donor_id", "cell_type", "development_stage"}
    missing = required_obs.difference(adata.obs.columns)
    if missing:
        raise ValueError(f"{tissue}: missing required obs columns: {sorted(missing)}")

    audit = adata.obs[["donor_id", "cell_type", "development_stage"]].copy()
    audit.index.name = "cell_id"
    audit["donor_id"] = audit["donor_id"].astype(str)
    audit["cell_type"] = audit["cell_type"].astype(str)
    audit["development_stage"] = audit["development_stage"].astype(str)
    audit["age_group"] = _age_group(audit["development_stage"])
    audit["filter_reason"] = "age_included"
    audit.loc[audit["development_stage"].eq(EXCLUDED_LABEL), "filter_reason"] = "age_label_excluded_4_weeks"
    audit.loc[audit["age_group"].isna() & ~audit["development_stage"].eq(EXCLUDED_LABEL), "filter_reason"] = "age_label_unrecognized"

    unexpected = audit.loc[audit["filter_reason"].eq("age_label_unrecognized"), "development_stage"]
    unexpected_fraction = len(unexpected) / len(audit)
    if unexpected_fraction >= 0.05:
        counts = unexpected.value_counts().to_dict()
        raise ValueError(f"{tissue}: unrecognized age labels are {unexpected_fraction:.2%} of cells: {counts}")

    before_active = audit.loc[audit["filter_reason"].eq("age_included")].copy()
    before_qc = _qc_counts(before_active, "n_cells_before", tissue)

    donor_type_sizes = before_active.groupby(["age_group", "donor_id", "cell_type"], observed=True).size()
    small_sets = donor_type_sizes.loc[donor_type_sizes < DONOR_TYPE_FLOOR]
    if not small_sets.empty:
        row_keys = pd.MultiIndex.from_frame(before_active[["age_group", "donor_id", "cell_type"]])
        audit.loc[before_active.index[row_keys.isin(small_sets.index)], "filter_reason"] = "donor_type_floor_lt_20"

    type_rows: list[dict[str, object]] = []
    larger_by_type: dict[str, str] = {}
    cell_types = sorted(before_active["cell_type"].unique())
    for cell_type in cell_types:
        current = audit.loc[
            audit["filter_reason"].eq("age_included") & audit["cell_type"].eq(cell_type)
        ]
        counts = current.groupby("age_group", observed=True).size()
        n_young = int(counts.get("young", 0))
        n_old = int(counts.get("old", 0))
        if not n_young or not n_old:
            audit.loc[current.index, "filter_reason"] = "type_missing_age_group"
            type_rows.append({"cell_type": cell_type, "n_young": 0, "n_old": 0, "eligible": False})
            continue

        target = min(n_young, n_old)
        larger = "young" if n_young > n_old else "old" if n_old > n_young else ""
        if larger:
            larger_by_type[cell_type] = larger
            larger_rows = current.loc[current["age_group"].eq(larger)]
            selected = _sample_evenly_by_donor(
                larger_rows, target, np.random.default_rng(_seed_for(seed, tissue, cell_type))
            )
            excluded = larger_rows.index.difference(selected)
            audit.loc[excluded, "filter_reason"] = "donor_balanced_downsample_excluded"

        balanced = audit.loc[
            audit["filter_reason"].eq("age_included") & audit["cell_type"].eq(cell_type)
        ]
        balanced_counts = balanced.groupby("age_group", observed=True).size()
        n_young = int(balanced_counts.get("young", 0))
        n_old = int(balanced_counts.get("old", 0))
        eligible = n_young >= MIN_ELIGIBLE_CELLS and n_old >= MIN_ELIGIBLE_CELLS
        if not eligible:
            audit.loc[balanced.index, "filter_reason"] = "type_below_200_per_age"
        type_rows.append(
            {"cell_type": cell_type, "n_young": n_young, "n_old": n_old, "eligible": eligible}
        )

    eligible_types = pd.DataFrame(type_rows).sort_values("cell_type", kind="stable").reset_index(drop=True)
    eligible_map = eligible_types.set_index("cell_type")["eligible"]
    audit["eligible"] = audit["cell_type"].map(eligible_map).fillna(False).astype(bool)
    audit["downsampled_age_group"] = audit["cell_type"].map(larger_by_type).fillna("")

    after_active = audit.loc[
        audit["filter_reason"].isin(["age_included", "type_below_200_per_age"])
    ].copy()
    after_qc = _qc_counts(after_active, "n_cells_after", tissue)
    after_qc["n_cells_contributed"] = after_qc["n_cells_after"]
    after_qc["eligible"] = after_qc["cell_type"].map(eligible_map).fillna(False).astype(bool)
    after_qc["downsampled_age_group"] = after_qc["cell_type"].map(larger_by_type).fillna("")

    before_qc.to_csv(qc_dir / f"{tissue}_donor_type_counts_before.csv", index=False)
    after_qc.to_csv(qc_dir / f"{tissue}_donor_type_counts_after.csv", index=False)
    eligible_types.to_csv(qc_dir / f"{tissue}_eligible_types.csv", index=False)
    audit.reset_index().to_csv(cell_filter_path, index=False)

    included = audit.index[audit["filter_reason"].eq("age_included")]
    processed = adata[included].copy()
    processed.obs["age_group"] = audit.loc[included, "age_group"].astype(str).to_numpy()
    processed.layers["counts"] = processed.X.copy()
    processed.raw = processed
    if processed.n_obs:
        sc.pp.highly_variable_genes(
            processed,
            flavor="seurat_v3",
            n_top_genes=N_HVGS,
            batch_key="donor_id",
        )
        processed = processed[:, processed.var["highly_variable"]].copy()
    processed.write_h5ad(processed_path)

    result = {
        "raw_cells": int(adata.n_obs),
        "cells_after": int(processed.n_obs),
        "eligible_types": int(eligible_types["eligible"].sum()),
        "unrecognized_age_cells": int((audit["filter_reason"] == "age_label_unrecognized").sum()),
        "excluded_4_week_cells": int((audit["filter_reason"] == "age_label_excluded_4_weeks").sum()),
    }
    print(f"[{tissue}] {result}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tissue", required=True, choices=("lung", "marrow"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    preprocess(args.tissue, args.seed, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
