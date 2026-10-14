"""Fetch TMS Lung 10x + Marrow 10x via cellxgene_census 2024-07-01.

Pulled fresh from the census and cached as .h5ad in
``benchmarks/aging/data/{lung,marrow}.h5ad``. Honours PREREG §1
dataset_ids.

Writes raw count matrices + ``obs['age_group']`` ∈ {young, old} +
``obs['cell_type']`` + donor info where available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

CENSUS_VERSION = "2024-07-01"

DATASETS = {
    "lung":   "0fb7916e-7a68-4a4c-a441-3ab3989f29a7",
    "marrow": "0bd1a1de-3aee-40e0-b2ec-86c7a30c7149",
}

YOUNG_MONTHS = {"3 month-old stage", "3 months"}
OLD_MONTHS = {
    "18 month-old stage", "21 month-old stage", "24 month-old stage",
    "18 months", "21 months", "24 months",
    "19 month-old stage", "22 month-old stage", "23 month-old stage",
    "20 month-old stage and over",
}


def fetch(tissue: str, out_dir: Path) -> Path:
    import cellxgene_census

    dataset_id = DATASETS[tissue]
    out_path = out_dir / f"{tissue}.h5ad"
    if out_path.exists():
        print(f"[{tissue}] cached: {out_path}")
        return out_path

    print(f"[{tissue}] opening census {CENSUS_VERSION}, dataset_id={dataset_id}")
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        adata = cellxgene_census.get_anndata(
            census,
            organism="Mus musculus",
            obs_value_filter=f"dataset_id == '{dataset_id}'",
        )
    print(f"[{tissue}] fetched: n_obs={adata.n_obs}, n_vars={adata.n_vars}")

    # Build age_group column.
    dev = adata.obs["development_stage"].astype(str)
    age_group = pd.Series(index=adata.obs.index, dtype=object)
    for v in dev.unique():
        if v in YOUNG_MONTHS:
            age_group.loc[dev == v] = "young"
        elif v in OLD_MONTHS:
            age_group.loc[dev == v] = "old"
    adata.obs["age_group"] = age_group
    adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)
    print(f"[{tissue}] age_group counts: {age_group.value_counts(dropna=False).to_dict()}")

    adata.write_h5ad(out_path)
    print(f"[{tissue}] wrote {out_path}")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent
                            / "benchmarks" / "aging" / "data")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for tissue in ("lung", "marrow"):
        fetch(tissue, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
