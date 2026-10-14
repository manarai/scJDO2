#!/usr/bin/env python3
"""A03: shared-space aging tests, standalone summaries, and aging modes.

The confirmatory §4b isotropy and §5 anticipation tests execute exactly once on
A02's scArches shared embeddings. Standalone fits are only summarized within
age, because their coordinate systems are unrelated by design.
"""
from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from scjdo2.compare import aging_modes, centroid_operator, compare_types, match_directions


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "aging"
TISSUES = ("lung", "marrow")
SEED = 0
N_CELLS_CENTROID = 30
N_NULL = 1000
MIN_CELLS = 200
EPS = 1e-12


def _jacobians(adata: ad.AnnData) -> np.ndarray:
    scj = adata.uns["scjdo2"]
    return np.asarray(scj["J_latent"] if "J_latent" in scj else scj["J"], dtype=np.float64)


def _spectrum_string(values: np.ndarray) -> str:
    return ";".join(f"{float(value):.17g}" for value in np.sort(values))


def _type_age_metrics(adata: ad.AnnData, use_rep: str) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    """Centroid-operator quantities for each cell type and age label."""
    z = np.asarray(adata.obsm[use_rep], dtype=np.float64)
    J = _jacobians(adata)
    cell_type = adata.obs["cell_type"].astype(str).to_numpy()
    age_group = adata.obs["age_group"].astype(str).to_numpy()
    behaviour = adata.obs["op_behaviour_basin"].astype(str).to_numpy()
    basin_index = adata.obs["basin_index"].to_numpy(dtype=float)
    tolerance = float(adata.uns["a02_drift"]["lead_eig_tolerance"])
    rows: list[dict] = []
    operators: dict[tuple[str, str], np.ndarray] = {}
    for ct in sorted(set(cell_type)):
        for age in ("young", "old"):
            mask = (cell_type == ct) & (age_group == age)
            n = int(mask.sum())
            if not n:
                continue
            J_centroid = centroid_operator(z[mask], J[mask], N_CELLS_CENTROID)
            J_centroid = 0.5 * (J_centroid + J_centroid.T)
            eig = np.linalg.eigvalsh(J_centroid)
            lead = float(eig.max())
            median_abs = float(np.median(np.abs(eig)))
            rows.append(
                {
                    "cell_type": ct,
                    "age_group": age,
                    "n": n,
                    "lead_eig": lead,
                    "median_abs_lambda": median_abs,
                    "anisotropy": float(median_abs / (abs(lead) + EPS)),
                    "n_soft": int((np.abs(eig) < tolerance).sum()),
                    "basin_fraction": float(np.mean(behaviour[mask] == "holder")),
                    "basin_index": float(np.median(basin_index[mask])),
                    "spectrum": _spectrum_string(eig),
                    "spectrum_values": eig,
                    "centroid": z[mask].mean(axis=0),
                }
            )
            operators[(ct, age)] = J_centroid
    return pd.DataFrame(rows), operators


def _split_half_displacement_noise(z_young: np.ndarray, n_old: int, seed: int) -> float:
    """The §5 young split-half centroid-distance 95th percentile."""
    half = max(2, min(int(n_old), len(z_young) // 2))
    rng = np.random.default_rng(seed)
    null = np.empty(N_NULL, dtype=np.float64)
    for i in range(N_NULL):
        index = rng.permutation(len(z_young))
        null[i] = float(np.linalg.norm(z_young[index[:half]].mean(axis=0) - z_young[index[half:2 * half]].mean(axis=0)))
    return float(np.percentile(null, 95))


def _shared_compare(tissue: str, shared: ad.AnnData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run pre-registered tests in the shared coordinate system only."""
    results = compare_types(
        shared,
        by="cell_type",
        groups=("young", "old"),
        group_key="age_group",
        use_rep="X_scVI_shared",
        n_cells_centroid=N_CELLS_CENTROID,
        min_cells=MIN_CELLS,
        n_null=N_NULL,
        seed=SEED,
    )
    details, operators = _type_age_metrics(shared, "X_scVI_shared")
    by_age = {
        age: details[details["age_group"].eq(age)].set_index("cell_type")
        for age in ("young", "old")
    }
    z = np.asarray(shared.obsm["X_scVI_shared"], dtype=np.float64)
    types = shared.obs["cell_type"].astype(str).to_numpy()
    groups = shared.obs["age_group"].astype(str).to_numpy()
    records: list[dict] = []
    for row in results.to_dict("records"):
        ct = str(row["cell_type"])
        young = by_age["young"].loc[ct]
        old = by_age["old"].loc[ct]
        matched = match_directions(operators[(ct, "young")], operators[(ct, "old")])
        lead_match = int(np.argmax(matched.eig_a))
        lead_ratio = float(matched.eig_b[lead_match] / matched.eig_a[lead_match])
        m_young = (types == ct) & (groups == "young")
        m_old = (types == ct) & (groups == "old")
        ct_seed = SEED + zlib.crc32(ct.encode("utf-8")) % 2**31
        record = {
            "tissue": tissue,
            **row,
            "substate_alignment": float("nan"),
            "lambda_ratio_soft_axis": lead_ratio,
            "lead_eig_young": float(young["lead_eig"]),
            "lead_eig_old": float(old["lead_eig"]),
            "median_abs_lambda_young": float(young["median_abs_lambda"]),
            "median_abs_lambda_old": float(old["median_abs_lambda"]),
            "anisotropy_young": float(young["anisotropy"]),
            "anisotropy_old": float(old["anisotropy"]),
            "n_soft_young": int(young["n_soft"]),
            "n_soft_old": int(old["n_soft"]),
            "basin_fraction_young": float(young["basin_fraction"]),
            "basin_fraction_old": float(old["basin_fraction"]),
            "basin_index_young": float(young["basin_index"]),
            "basin_index_old": float(old["basin_index"]),
            "spectrum_young": str(young["spectrum"]),
            "spectrum_old": str(old["spectrum"]),
            "spectrum_values_young": young["spectrum_values"],
            "spectrum_values_old": old["spectrum_values"],
            "displacement": float(np.linalg.norm(z[m_old].mean(axis=0) - z[m_young].mean(axis=0))),
            "split_half_noise": _split_half_displacement_noise(z[m_young], int(m_old.sum()), ct_seed),
        }
        records.append(record)
    return pd.DataFrame(records), details


def _standalone_compare(
    tissue: str,
    standalone: dict[str, ad.AnnData],
    shared_details: pd.DataFrame,
    artifacts: dict[str, bool],
) -> pd.DataFrame:
    """Write axis-free standalone versus shared type-age summaries."""
    shared = shared_details.set_index(["cell_type", "age_group"])
    rows: list[dict] = []
    for age in ("young", "old"):
        use_rep = f"X_scVI_{age}"
        detail, _ = _type_age_metrics(standalone[age], use_rep)
        for record in detail.to_dict("records"):
            ct = str(record.pop("cell_type"))
            record.pop("spectrum_values")
            record.pop("centroid")
            shared_record = shared.loc[(ct, age)]
            out = {
                "tissue": tissue,
                "cell_type": ct,
                "age_group": age,
                **record,
                "correction_artifact": bool(artifacts[ct]),
            }
            for column in ("n", "lead_eig", "median_abs_lambda", "anisotropy", "n_soft", "basin_fraction", "basin_index", "spectrum"):
                out[f"shared_{column}"] = shared_record[column]
            rows.append(out)
    columns = [
        "tissue", "cell_type", "age_group", "n", "lead_eig",
        "median_abs_lambda", "anisotropy", "n_soft", "basin_fraction",
        "basin_index", "spectrum", "correction_artifact", "shared_n",
        "shared_lead_eig", "shared_median_abs_lambda", "shared_anisotropy",
        "shared_n_soft", "shared_basin_fraction", "shared_basin_index",
        "shared_spectrum",
    ]
    return pd.DataFrame(rows)[columns].sort_values(["cell_type", "age_group"], kind="stable")


def run_tissue(tissue: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    shared = ad.read_h5ad(OUT / f"{tissue}_shared.h5ad")
    standalone = {
        age: ad.read_h5ad(OUT / f"{tissue}_standalone_{age}.h5ad")
        for age in ("young", "old")
    }
    shared_table, shared_details = _shared_compare(tissue, shared)
    standalone_details = {
        age: _type_age_metrics(standalone[age], f"X_scVI_{age}")[0].set_index("cell_type")
        for age in ("young", "old")
    }
    artifacts: dict[str, bool] = {}
    for ct in shared_table["cell_type"]:
        mismatch = False
        for age in ("young", "old"):
            shared_fraction = float(shared_details.set_index(["cell_type", "age_group"]).loc[(ct, age), "basin_fraction"])
            standalone_fraction = float(standalone_details[age].loc[ct, "basin_fraction"])
            mismatch |= (shared_fraction >= 0.5) != (standalone_fraction >= 0.5)
        artifacts[ct] = bool(mismatch)
    shared_table["correction_artifact"] = shared_table["cell_type"].map(artifacts).astype(bool)
    standalone_table = _standalone_compare(tissue, standalone, shared_details, artifacts)
    # Object arrays support mode clustering but must never enter CSV output.
    shared_csv = shared_table.drop(columns=["spectrum_values_young", "spectrum_values_old"])
    shared_csv.to_csv(OUT / f"{tissue}_compare_shared.csv", index=False)
    standalone_table.to_csv(OUT / f"{tissue}_compare_standalone.csv", index=False)
    return shared_table, standalone_table


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tissue", choices=(*TISSUES, "all"), default="all")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.seed != SEED:
        raise ValueError("A03 is frozen at seed 0")
    tissues = TISSUES if args.tissue == "all" else (args.tissue,)
    pooled_parts: list[pd.DataFrame] = []
    for tissue in tissues:
        shared_table, _ = run_tissue(tissue)
        pooled_parts.append(shared_table)
        print(f"[{tissue}] eligible={len(shared_table)} correction_artifacts={int(shared_table['correction_artifact'].sum())}", flush=True)
    if args.tissue == "all":
        pooled = pd.concat(pooled_parts, ignore_index=True)
        pooled_csv = pooled.drop(columns=["spectrum_values_young", "spectrum_values_old"])
        pooled_csv.to_csv(OUT / "compare_pooled.csv", index=False)
        mode_input = pooled.drop(columns=["spectrum_young", "spectrum_old"]).rename(columns={
            "spectrum_values_young": "spectrum_young",
            "spectrum_values_old": "spectrum_old",
        })
        modes = aging_modes(mode_input, seed=SEED)
        modes = pd.concat([pooled[["tissue", "cell_type"]].reset_index(drop=True), modes.reset_index(drop=True)], axis=1)
        modes.to_csv(OUT / "aging_modes.csv", index=False)
        print(f"[pooled] eligible={len(pooled)} correction_artifacts={int(pooled['correction_artifact'].sum())} silhouette_k={int(modes['silhouette_k'].iloc[0])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
