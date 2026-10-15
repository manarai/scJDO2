#!/usr/bin/env python3
"""H03 §4: fixed donor-level eligibility from the deterministic drift sample."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "aging_hlca"
INPUT = OUT / "drift_subsample_cells.csv.gz"
OUTPUT = OUT / "eligibility.csv"
N_COMMON = 50
MIN_DONORS = 20
MIN_LT45 = 5
MIN_GE60 = 5


def subset_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    region = pd.to_numeric(frame["anatomical_region_ccf_score"], errors="coerce")
    return {
        "primary": np.ones(len(frame), dtype=bool),
        "banovich": frame["study_or_dataset"].astype(str).eq("Banovich_Kropski_2020").to_numpy(),
        "parenchyma": region.eq(0.97).to_numpy(),
        "never_smokers": frame["smoking_status"].astype(str).eq("never").to_numpy(),
    }


def eligibility_rows(frame: pd.DataFrame, analysis: str) -> list[dict]:
    rows: list[dict] = []
    for cell_type, group in frame.groupby("ann_level_3", dropna=False, sort=True):
        by_donor = group.groupby("donor_id", sort=True).agg(age=("age", "first"), n_cells=("cell_id", "size"))
        contributing = by_donor.loc[by_donor["n_cells"] >= N_COMMON]
        n_donors = int(len(contributing))
        n_lt45 = int((contributing["age"] < 45).sum())
        n_ge60 = int((contributing["age"] >= 60).sum())
        rows.append({
            "analysis": analysis,
            "ann_level_3": str(cell_type),
            "n_donors": n_donors,
            "n_lt45": n_lt45,
            "n_ge60": n_ge60,
            "eligible": bool(n_donors >= MIN_DONORS and n_lt45 >= MIN_LT45 and n_ge60 >= MIN_GE60),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    frame = pd.read_csv(INPUT, compression="gzip")
    required = {"cell_id", "donor_id", "ann_level_3", "age", "study_or_dataset", "anatomical_region_ccf_score", "smoking_status"}
    if set(frame.columns) != required:
        raise RuntimeError(f"unexpected drift-sample columns: {frame.columns.tolist()}")
    if frame["cell_id"].astype(str).duplicated().any():
        raise RuntimeError("drift sample contains duplicate cell IDs")
    frame["age"] = pd.to_numeric(frame["age"], errors="raise")
    all_rows: list[dict] = []
    for analysis, mask in subset_masks(frame).items():
        all_rows.extend(eligibility_rows(frame.loc[mask].copy(), analysis))
    result = pd.DataFrame(all_rows, columns=["analysis", "ann_level_3", "n_donors", "n_lt45", "n_ge60", "eligible"])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    counts = result.loc[result["eligible"]].groupby("analysis")["ann_level_3"].agg(list).to_dict()
    print(json.dumps({
        "output": str(OUTPUT), "n_common": N_COMMON,
        "eligible_counts": {key: len(value) for key, value in counts.items()},
        "eligible_types": counts,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
