#!/usr/bin/env python3
"""H03 §1–2: bounded preprocessing, frozen Seurat-v3 HVGs, and drift sampling.

AnnData 0.11's backed CSR adapter cannot be indexed by Scanpy's Seurat-v3
implementation on the installed Python runtime.  This script executes the same
per-batch Seurat-v3 formula (Scanpy 1.12.4; same skmisc loess and clipped-sum
helper) in streaming sparse-row chunks.  It never materializes the 1.14-billion
nonzero raw-count matrix, and therefore preserves the frozen rule without
exceeding H03's memory ceiling.
"""
from __future__ import annotations

import argparse
import json
import resource
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from skmisc.loess import loess
from scanpy.preprocessing._highly_variable_genes import _sum_and_sum_squares_clipped

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks" / "aging_hlca" / "data" / "hlca_core_full_metadata.h5ad"
OUT = ROOT / "benchmarks" / "aging_hlca"
QC = OUT / "qc"
SUBSAMPLE = OUT / "drift_subsample_cells.csv.gz"
HVG_CSV = QC / "hvg_seurat_v3.csv"
SUMMARY_CSV = QC / "preprocess_summary.csv"
SEED = 0
N_HVGS = 2_000
SPAN = 1.0
MAX_PER_DONOR_TYPE = 60
ROW_CHUNK = 512


def max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.
    return value if Path("/System").exists() else value * 1024


def contiguous_runs(rows: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open consecutive source-row runs in a sorted index vector."""
    if rows.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(rows) != 1) + 1
    starts = np.r_[0, cuts]
    stops = np.r_[cuts, len(rows)]
    return [(int(rows[start]), int(rows[stop - 1]) + 1) for start, stop in zip(starts, stops, strict=True)]


def iter_csr_rows(
    handle: h5py.File,
    rows: np.ndarray,
    n_vars: int,
    *,
    row_chunk: int = ROW_CHUNK,
):
    """Yield exact raw-count CSR chunks for sorted selected H5AD observation rows."""
    group = handle["raw"]["X"]
    indptr = group["indptr"]
    data = group["data"]
    indices = group["indices"]
    for run_start, run_stop in contiguous_runs(rows):
        for start in range(run_start, run_stop, row_chunk):
            stop = min(start + row_chunk, run_stop)
            p0 = int(indptr[start])
            p1 = int(indptr[stop])
            local_indptr = np.asarray(indptr[start : stop + 1], dtype=np.int64) - p0
            yield sparse.csr_matrix(
                (
                    np.asarray(data[p0:p1]),
                    np.asarray(indices[p0:p1], dtype=np.int64),
                    local_indptr,
                ),
                shape=(stop - start, n_vars),
            )


def load_rows_csr(handle: h5py.File, rows: np.ndarray, n_vars: int) -> sparse.csr_matrix:
    """Load one study as CSR, with a bounded peak below the full matrix size.

    Scanpy's public Seurat-v3 implementation does not support backed CSR in the
    installed version.  Loading one study at a time keeps the exact Scanpy
    numeric kernel while avoiding a full 584,944-cell materialization.
    """
    if rows.size == 0:
        return sparse.csr_matrix((0, n_vars), dtype=np.float32)
    group = handle["raw"]["X"]
    source_indptr = np.asarray(group["indptr"], dtype=np.int64)
    counts = source_indptr[rows + 1] - source_indptr[rows]
    total_nnz = int(counts.sum())
    data_out = np.empty(total_nnz, dtype=group["data"].dtype)
    indices_out = np.empty(total_nnz, dtype=group["indices"].dtype)
    indptr_out = np.empty(len(rows) + 1, dtype=np.int64)
    indptr_out[0] = 0
    row_cursor = 0
    nnz_cursor = 0
    for run_start, run_stop in contiguous_runs(rows):
        for start in range(run_start, run_stop, ROW_CHUNK):
            stop = min(start + ROW_CHUNK, run_stop)
            p0 = int(source_indptr[start])
            p1 = int(source_indptr[stop])
            width = p1 - p0
            data_out[nnz_cursor : nnz_cursor + width] = group["data"][p0:p1]
            indices_out[nnz_cursor : nnz_cursor + width] = group["indices"][p0:p1]
            local_counts = np.diff(source_indptr[start : stop + 1])
            indptr_out[row_cursor + 1 : row_cursor + 1 + len(local_counts)] = nnz_cursor + np.cumsum(local_counts)
            row_cursor += len(local_counts)
            nnz_cursor += width
    if row_cursor != len(rows) or nnz_cursor != total_nnz:
        raise RuntimeError("failed to assemble the requested exact raw-count rows")
    return sparse.csr_matrix((data_out, indices_out, indptr_out), shape=(len(rows), n_vars))


def mean_var_stream(handle: h5py.File, rows: np.ndarray, n_vars: int) -> tuple[np.ndarray, np.ndarray]:
    """Unbiased per-gene moments matching the Seurat-v3 input moments."""
    total = np.zeros(n_vars, dtype=np.float64)
    total_sq = np.zeros(n_vars, dtype=np.float64)
    n_rows = 0
    for matrix in iter_csr_rows(handle, rows, n_vars):
        total += np.asarray(matrix.sum(axis=0), dtype=np.float64).ravel()
        total_sq += np.asarray(matrix.multiply(matrix).sum(axis=0), dtype=np.float64).ravel()
        n_rows += matrix.shape[0]
    if n_rows < 2:
        raise RuntimeError("a study batch has fewer than two cells for Seurat-v3 HVG selection")
    mean = total / n_rows
    variance = (total_sq - n_rows * np.square(mean)) / (n_rows - 1)
    # Small negative cancellation artifacts are not biological variance.
    variance = np.maximum(variance, 0.0)
    return mean, variance


def clipped_sums_stream(
    handle: h5py.File,
    rows: np.ndarray,
    n_vars: int,
    clip_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream Scanpy's exact sparse clipped-sum helper over source chunks."""
    sq_sum = np.zeros(n_vars, dtype=np.float64)
    sum_ = np.zeros(n_vars, dtype=np.float64)
    for matrix in iter_csr_rows(handle, rows, n_vars):
        chunk_sq, chunk_sum = _sum_and_sum_squares_clipped(
            matrix.indices,
            matrix.data,
            n_cols=n_vars,
            clip_val=clip_val,
            nnz=matrix.nnz,
        )
        sq_sum += chunk_sq
        sum_ += chunk_sum
    return sq_sum, sum_


def seurat_v3_streamed(handle: h5py.File, obs: pd.DataFrame, var_names: pd.Index) -> pd.DataFrame:
    """Apply Scanpy 1.12.4 Seurat-v3 exactly, one study matrix at a time."""
    n_vars = len(var_names)
    batch = obs["study_or_dataset"].astype(str).to_numpy()
    source_rows = obs["_source_row"].to_numpy(dtype=np.int64)
    rows_by_batch = {
        name: np.sort(source_rows[batch == name])
        for name in sorted(pd.unique(batch).tolist())
    }
    normalized: list[np.ndarray] = []
    means: list[np.ndarray] = []
    variances: list[np.ndarray] = []
    for batch_name, rows in rows_by_batch.items():
        matrix = load_rows_csr(handle, rows, n_vars)
        # This is Scanpy's own aggregate kernel used by highly_variable_genes;
        # using it retains bitwise-equivalent per-batch moments without holding
        # the other 13 study matrices in memory.
        batch_adata = ad.AnnData(
            X=matrix,
            obs=pd.DataFrame({"_hvg_batch": ["one"] * matrix.shape[0]}),
            var=pd.DataFrame(index=var_names),
        )
        aggregate = sc.get.aggregate(batch_adata, by="_hvg_batch", func=["mean", "var"])
        mean = np.asarray(aggregate.layers["mean"][0], dtype=np.float64).ravel()
        variance = np.asarray(aggregate.layers["var"][0], dtype=np.float64).ravel()
        del aggregate, batch_adata
        estimat_var = np.zeros(n_vars, dtype=np.float64)
        not_const = variance > 0
        if not_const.any():
            model = loess(np.log10(mean[not_const]), np.log10(variance[not_const]), span=SPAN, degree=2)
            model.fit()
            estimat_var[not_const] = model.outputs.fitted_values
        reg_std = np.sqrt(np.power(10.0, estimat_var))
        clip_val = reg_std * np.sqrt(matrix.shape[0]) + mean
        squared_sum, count_sum = _sum_and_sum_squares_clipped(
            matrix.indices,
            matrix.data,
            n_cols=n_vars,
            clip_val=clip_val,
            nnz=matrix.nnz,
        )
        norm_var = (1.0 / ((matrix.shape[0] - 1) * np.square(reg_std))) * (
            matrix.shape[0] * np.square(mean) + squared_sum - 2.0 * count_sum * mean
        )
        normalized.append(norm_var)
        means.append(mean)
        variances.append(variance)
        print(f"HVG batch {batch_name}: {matrix.shape[0]} cells, {matrix.nnz} nonzeros", flush=True)
        del matrix
    norm = np.vstack(normalized)
    ranks = np.argsort(np.argsort(-norm, axis=1), axis=1).astype(np.float32)
    n_batch_hvg = np.sum(ranks < N_HVGS, axis=0)
    ranks[ranks >= N_HVGS] = np.nan
    median_rank = np.ma.median(np.ma.masked_invalid(ranks), axis=0).filled(np.nan)
    table = pd.DataFrame(
        {
            "gene_id": var_names.astype(str),
            "highly_variable_rank": median_rank,
            "highly_variable_nbatches": n_batch_hvg,
            "variances_norm": np.mean(norm, axis=0),
            "mean_across_batches": np.mean(np.vstack(means), axis=0),
            "variance_across_batches": np.mean(np.vstack(variances), axis=0),
        }
    )
    ordered = table.sort_values(
        ["highly_variable_rank", "highly_variable_nbatches"],
        ascending=[True, False],
        na_position="last",
        kind="stable",
    ).index.to_numpy()
    table["highly_variable"] = False
    table.loc[ordered[:N_HVGS], "highly_variable"] = True
    if int(table["highly_variable"].sum()) != N_HVGS:
        raise RuntimeError("frozen Seurat-v3 selection did not return exactly 2,000 HVGs")
    return table


def drift_subsample(obs: pd.DataFrame) -> pd.DataFrame:
    """Draw the frozen ≤60 cell donor×ann_level_3 drift sample."""
    work = obs.loc[obs["ann_level_3"].astype(str).ne("Rare")].copy()
    rows: list[pd.DataFrame] = []
    for (donor, cell_type), group in work.groupby(["donor_id", "ann_level_3"], observed=True, sort=True):
        group = group.sort_index(kind="stable")
        n_take = min(MAX_PER_DONOR_TYPE, len(group))
        rng = np.random.default_rng(zlib.crc32(f"{donor}|{cell_type}".encode()))
        positions = np.sort(rng.choice(len(group), size=n_take, replace=False))
        rows.append(group.iloc[positions])
    result = pd.concat(rows, axis=0).sort_index(kind="stable")
    result = result.rename_axis("cell_id").reset_index()
    columns = [
        "cell_id", "donor_id", "ann_level_3", "age", "study_or_dataset",
        "anatomical_region_ccf_score", "smoking_status",
    ]
    return result[columns]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.seed != SEED:
        raise ValueError("H03 preprocessing is frozen at seed 0")
    if not SOURCE.exists():
        raise FileNotFoundError(f"missing H01 canonical source: {SOURCE}")
    started = time.monotonic()
    source = ad.read_h5ad(SOURCE, backed="r")
    try:
        obs = source.obs.copy()
        raw_var = source.raw.var.copy()
        age = pd.to_numeric(obs["age_or_mean_of_age_range"], errors="coerce")
        keep = age.notna().to_numpy()
        if int(keep.sum()) == 0:
            raise RuntimeError("no cells have numeric HLCA ages")
        obs = obs.loc[keep].copy()
        obs["age"] = age.loc[keep].to_numpy(dtype=float)
        # H01's source field is called dataset; this alias is the frozen batch key.
        obs["study_or_dataset"] = obs["dataset"].astype(str).to_numpy()
        source_rows = np.flatnonzero(keep)
        obs["_source_row"] = source_rows
        if obs["donor_id"].astype(str).nunique() != 106:
            raise RuntimeError("H03 must retain exactly the 106 donors with numeric age")
        print(f"H03 preprocessing: {len(obs)} cells, 106 donors, {raw_var.shape[0]} raw genes", flush=True)
        with h5py.File(SOURCE, "r") as handle:
            hvg = seurat_v3_streamed(handle, obs, raw_var.index)
        drift = drift_subsample(obs)
    finally:
        source.file.close()
    QC.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    hvg.to_csv(HVG_CSV, index=False)
    drift.to_csv(SUBSAMPLE, index=False, compression="gzip")
    rare_cells = int(obs["ann_level_3"].astype(str).eq("Rare").sum())
    summary = pd.DataFrame(
        [
            {
                "stage": "numeric_age_filter",
                "n_cells": int(len(obs)),
                "n_donors": int(obs["donor_id"].astype(str).nunique()),
                "n_ann_level_3": int(obs["ann_level_3"].astype(str).nunique()),
                "n_hvgs": np.nan,
                "details": "excluded exactly the donor without a numeric age; retained Rare for HVG and scVI",
            },
            {
                "stage": "seurat_v3_hvg",
                "n_cells": int(len(obs)),
                "n_donors": int(obs["donor_id"].astype(str).nunique()),
                "n_ann_level_3": int(obs["ann_level_3"].astype(str).nunique()),
                "n_hvgs": int(hvg["highly_variable"].sum()),
                "details": "flavor=seurat_v3; batch_key=study_or_dataset; span=1.0; n_top_genes=2000",
            },
            {
                "stage": "drift_subsample",
                "n_cells": int(len(drift)),
                "n_donors": int(drift["donor_id"].astype(str).nunique()),
                "n_ann_level_3": int(drift["ann_level_3"].astype(str).nunique()),
                "n_hvgs": np.nan,
                "details": f"at most 60 cells per donor×type; crc32(donor|type); Rare excluded from drift ({rare_cells} training cells retained)",
            },
        ]
    )
    summary.to_csv(SUMMARY_CSV, index=False)
    payload = {
        "hvg_csv": str(HVG_CSV),
        "drift_cells_csv": str(SUBSAMPLE),
        "n_cells_numeric_age": int(len(obs)),
        "n_donors_numeric_age": int(obs["donor_id"].astype(str).nunique()),
        "n_hvgs": int(hvg["highly_variable"].sum()),
        "n_drift_cells": int(len(drift)),
        "peak_rss_bytes": max_rss_bytes(),
        "wall_seconds": time.monotonic() - started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
