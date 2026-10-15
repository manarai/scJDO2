#!/usr/bin/env python3
"""H03 Revision 1: full in-memory Scanpy equivalence for frozen HVGs."""
from __future__ import annotations

import gc
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks" / "aging_hlca" / "data" / "hlca_core_full_metadata.h5ad"
CUSTOM_HVG = ROOT / "benchmarks" / "aging_hlca" / "qc" / "hvg_seurat_v3.csv"
OUT = ROOT / "benchmarks" / "aging_hlca" / "qc" / "hvg_equivalence.csv"
N_HVGS = 2_000


def max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if Path("/System").exists() else value * 1024


def load_raw_counts_in_memory() -> tuple[sparse.csr_matrix, pd.DataFrame, pd.DataFrame]:
    """Load only raw counts plus metadata, avoiding a duplicate X matrix.

    The task requires the full HVG input in memory. The source H5AD stores X and
    raw.X separately, each with the same 1.14B nonzero dimensions. Reading with
    ``anndata.read_h5ad(..., backed=False)`` would materialize both, which risks
    the 28-GB ceiling.  Here raw.X itself is loaded into an in-memory CSR array
    directly; its column indices are safely stored as int32 (27,402 columns),
    exactly preserving count values and coordinates while keeping the equivalent
    Scanpy input well below the ceiling.
    """
    source = ad.read_h5ad(SOURCE, backed="r")
    try:
        obs = source.obs.copy()
        var = source.raw.var.copy()
    finally:
        source.file.close()
    with h5py.File(SOURCE, "r") as h5:
        raw = h5["raw"]["X"]
        data = np.empty(raw["data"].shape, dtype=np.float32)
        indices = np.empty(raw["indices"].shape, dtype=np.int32)
        indptr = np.empty(raw["indptr"].shape, dtype=np.int64)
        raw["data"].read_direct(data)
        raw["indices"].read_direct(indices)
        raw["indptr"].read_direct(indptr)
    matrix = sparse.csr_matrix((data, indices, indptr), shape=(len(obs), len(var)), copy=False)
    return matrix, obs, var


def main() -> int:
    started = time.monotonic()
    if not SOURCE.exists() or not CUSTOM_HVG.exists():
        raise FileNotFoundError("H03 source or custom frozen HVG file is missing")
    custom = pd.read_csv(CUSTOM_HVG).set_index("gene_id")["highly_variable"].astype(bool)
    matrix, obs, var = load_raw_counts_in_memory()
    age = pd.to_numeric(obs["age_or_mean_of_age_range"], errors="coerce")
    keep = age.notna().to_numpy()
    obs["study_or_dataset"] = obs["dataset"].astype(str).to_numpy()
    full = ad.AnnData(X=matrix, obs=obs, var=var)
    # Release the all-donor source matrix as soon as the required 106-donor
    # in-memory input has been formed.
    adata = full[keep].copy()
    del full, matrix, obs, var
    gc.collect()
    if adata.n_obs != 581_517 or adata.obs["donor_id"].astype(str).nunique() != 106:
        raise RuntimeError("full in-memory HVG input does not match H03's 106-donor cohort")
    print(f"full Scanpy HVG input: {adata.n_obs} cells × {adata.n_vars} genes", flush=True)
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        n_top_genes=N_HVGS,
        batch_key="study_or_dataset",
        span=1.0,
    )
    scanpy = adata.var["highly_variable"].astype(bool)
    if not custom.index.equals(scanpy.index):
        raise RuntimeError("custom and Scanpy HVG identifiers are not aligned")
    equivalence = pd.DataFrame({
        "gene": scanpy.index.astype(str),
        "in_custom": custom.to_numpy(dtype=bool),
        "in_scanpy": scanpy.to_numpy(dtype=bool),
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    equivalence.to_csv(OUT, index=False)
    overlap = int((equivalence["in_custom"] & equivalence["in_scanpy"]).sum())
    payload = {
        "output": str(OUT),
        "n_cells": int(adata.n_obs),
        "n_donors": int(adata.obs["donor_id"].astype(str).nunique()),
        "n_custom": int(equivalence["in_custom"].sum()),
        "n_scanpy": int(equivalence["in_scanpy"].sum()),
        "hvg_overlap_full": overlap,
        "identical": bool(overlap == N_HVGS),
        "peak_rss_bytes": max_rss_bytes(),
        "wall_seconds": float(time.monotonic() - started),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    if overlap != N_HVGS:
        raise RuntimeError(f"full-data HVG mismatch: overlap {overlap}/{N_HVGS}; use Scanpy's genes downstream")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
