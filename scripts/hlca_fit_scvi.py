#!/usr/bin/env python3
"""H03 §3 primary space: young SCVI reference and frozen old scArches query."""
from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scvi
import torch
from scipy import sparse

from scripts.hlca_preprocess import SOURCE, load_rows_csr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "aging_hlca"
QC = OUT / "qc"
DATA = OUT / "data"
HVG_CSV = QC / "hvg_seurat_v3.csv"
DRIFT_CELLS = OUT / "drift_subsample_cells.csv.gz"
METRICS = QC / "hlca_scvi_metrics.csv"
HISTORY = QC / "hlca_scvi_history.csv"
DRIFT_LATENT = DATA / "h03_primary_drift_latent.h5ad"
SEED = 0
MAX_EPOCHS = 400
PATIENCE = 30
N_LATENT = 15
N_LAYERS = 2


def max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if Path("/System").exists() else value * 1024


def training_kwargs() -> dict:
    return {
        "max_epochs": MAX_EPOCHS,
        "accelerator": "cpu",
        "devices": 1,
        "early_stopping": True,
        "early_stopping_patience": PATIENCE,
        "check_val_every_n_epoch": 1,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }


def load_hvg_adata(age_kind: str) -> ad.AnnData:
    """Construct one age-space count matrix from raw H01 rows and frozen HVGs."""
    if age_kind not in {"young", "old"}:
        raise ValueError(age_kind)
    hvg = pd.read_csv(HVG_CSV)
    genes = hvg.loc[hvg["highly_variable"], "gene_id"].astype(str).tolist()
    if len(genes) != N_LATENT * 0 + 2_000 or len(set(genes)) != 2_000:
        raise RuntimeError("H03 requires exactly 2,000 unique frozen HVGs")
    source = ad.read_h5ad(SOURCE, backed="r")
    try:
        obs = source.obs.copy()
        age = pd.to_numeric(obs["age_or_mean_of_age_range"], errors="coerce")
        finite = age.notna().to_numpy()
        mask = finite & ((age.to_numpy() < 45) if age_kind == "young" else (age.to_numpy() >= 45))
        rows = np.flatnonzero(mask)
        if rows.size == 0:
            raise RuntimeError(f"no {age_kind} cells")
        work_obs = obs.iloc[rows].copy()
        work_obs["age"] = age.iloc[rows].to_numpy(dtype=float)
        work_obs["study_or_dataset"] = work_obs["dataset"].astype(str).to_numpy()
        raw_var_names = source.raw.var_names.astype(str)
        positions = raw_var_names.get_indexer(genes)
        if (positions < 0).any():
            raise RuntimeError("frozen HVG list is not contained in the H01 raw matrix")
        # Keep the source gene order for the sparse column slice; reindex to the
        # deterministic Seurat-v3 list afterwards so reference and query agree.
        source_order = np.sort(positions)
        source_to_output = np.searchsorted(source_order, positions)
        with h5py.File(SOURCE, "r") as handle:
            full = load_rows_csr(handle, rows, source.n_vars)
        hvg_counts_source_order = full[:, source_order].tocsr()
        del full
        counts = hvg_counts_source_order[:, source_to_output].tocsr()
        del hvg_counts_source_order
        var = source.raw.var.iloc[positions].copy()
        var.index = pd.Index(genes, dtype=str)
    finally:
        source.file.close()
    adata = ad.AnnData(X=counts, obs=work_obs, var=var)
    # Raw counts are intentionally retained as the model layer; X is its alias
    # to avoid a second multi-gigabyte sparse allocation.
    adata.layers["counts"] = adata.X
    return adata


def make_model(adata: ad.AnnData) -> scvi.model.SCVI:
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="study_or_dataset")
    return scvi.model.SCVI(adata, n_latent=N_LATENT, n_layers=N_LAYERS, gene_likelihood="nb")


def train(model: scvi.model.SCVI, *, plan_kwargs: dict | None = None) -> tuple[dict[str, pd.DataFrame], float]:
    kwargs = training_kwargs()
    if plan_kwargs is not None:
        kwargs["plan_kwargs"] = plan_kwargs
    started = time.monotonic()
    model.train(**kwargs)
    elapsed = time.monotonic() - started
    history = {str(name): value.copy() for name, value in model.history.items()}
    if "elbo_train" not in history or "elbo_validation" not in history:
        raise RuntimeError("H03 requires both train and validation ELBO histories")
    return history, elapsed


def history_rows(history: dict[str, pd.DataFrame], fit: str) -> list[dict]:
    rows: list[dict] = []
    for metric in ("elbo_train", "elbo_validation"):
        frame = history[metric]
        values = np.asarray(frame.iloc[:, 0], dtype=float)
        for epoch, value in enumerate(values, start=1):
            rows.append({"fit": fit, "metric": metric, "epoch": epoch, "value": float(value)})
    return rows


def audit_frozen_latent_encoder(reference: scvi.model.SCVI, query: scvi.model.SCVI) -> tuple[float, int, int]:
    """Assert every reference z_encoder parameter/buffer is unchanged exactly."""
    reference_state = reference.module.state_dict()
    query_state = query.module.state_dict()
    parameter_names = {name for name, _ in reference.module.named_parameters()}
    buffer_names = {name for name, _ in reference.module.named_buffers()}
    selected = [name for name in reference_state if name.startswith("z_encoder")]
    if not selected:
        raise RuntimeError("could not find z_encoder state for required frozen audit")
    maxima: list[float] = []
    parameter_count = 0
    buffer_count = 0
    for name in selected:
        if name not in query_state:
            raise RuntimeError(f"query state lacks latent-encoder key {name!r}")
        ref = reference_state[name].detach().cpu()
        mapped = query_state[name].detach().cpu()
        if mapped.ndim != ref.ndim or any(mapped.shape[i] < ref.shape[i] for i in range(ref.ndim)):
            raise RuntimeError(f"incompatible query shape for latent-encoder key {name!r}")
        slices = tuple(slice(0, size) for size in ref.shape)
        delta = torch.max(torch.abs(mapped[slices].to(dtype=ref.dtype) - ref)).item()
        maxima.append(float(delta))
        if name in parameter_names:
            parameter_count += int(ref.numel())
        elif name in buffer_names:
            buffer_count += int(ref.numel())
        else:
            raise RuntimeError(f"audit key {name!r} is neither parameter nor buffer")
    max_change = float(max(maxima))
    if max_change != 0.0:
        raise RuntimeError(f"scArches frozen latent-encoder audit failed: max_abs_change={max_change}")
    return max_change, parameter_count, buffer_count


def drift_latents(
    reference: scvi.model.SCVI,
    young: ad.AnnData,
    query: scvi.model.SCVI,
    old: ad.AnnData,
) -> ad.AnnData:
    requested = pd.read_csv(DRIFT_CELLS, compression="gzip")
    wanted = pd.Index(requested["cell_id"].astype(str))
    pieces: list[ad.AnnData] = []
    vectors: list[np.ndarray] = []
    for model, data, label in ((reference, young, "young"), (query, old, "old")):
        mask = data.obs_names.astype(str).isin(wanted)
        subset = data[mask].copy()
        z = np.asarray(model.get_latent_representation(subset), dtype=np.float32)
        if z.shape != (subset.n_obs, N_LATENT):
            raise RuntimeError(f"unexpected {label} latent shape {z.shape}")
        pieces.append(subset)
        vectors.append(z)
    combined = ad.concat(pieces, join="outer", merge="same", index_unique=None)
    latent = np.vstack(vectors)
    if combined.obs_names.duplicated().any() or combined.n_obs != len(wanted):
        raise RuntimeError("drift latent cells do not form the frozen unique sample")
    order = combined.obs_names.get_indexer(wanted)
    if (order < 0).any():
        raise RuntimeError("one or more frozen drift cells are missing from primary mapping")
    combined = combined[order].copy()
    latent = latent[order]
    # Do not retain the count matrix in this local output; it contains only the
    # per-cell shared-space input required for the two H03 drift fits.
    result = ad.AnnData(X=np.zeros((combined.n_obs, 0), dtype=np.float32), obs=combined.obs.copy())
    result.obsm["X_scVI_primary"] = latent
    result.uns["h03_primary"] = {"representation": "X_scVI_primary", "n_latent": N_LATENT, "seed": SEED}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.seed != SEED:
        raise ValueError("H03 primary mapping is frozen at seed 0")
    if not HVG_CSV.exists() or not DRIFT_CELLS.exists():
        raise FileNotFoundError("run scripts/hlca_preprocess.py first")
    started = time.monotonic()
    scvi.settings.seed = SEED
    young = load_hvg_adata("young")
    reference = make_model(young)
    ref_history, ref_wall = train(reference)
    scvi.settings.seed = SEED
    old = load_hvg_adata("old")
    query = scvi.model.SCVI.load_query_data(
        old,
        reference,
        accelerator="cpu",
        device="auto",
        unfrozen=False,
        freeze_expression=True,
        freeze_decoder_first_layer=True,
        freeze_batchnorm_encoder=True,
        freeze_batchnorm_decoder=False,
    )
    query_history, query_wall = train(query, plan_kwargs={"weight_decay": 0.0})
    max_change, parameter_count, buffer_count = audit_frozen_latent_encoder(reference, query)
    latent = drift_latents(reference, young, query, old)
    DATA.mkdir(parents=True, exist_ok=True)
    latent.write_h5ad(DRIFT_LATENT)
    records = []
    for fit, data, history, wall in (("young_reference", young, ref_history, ref_wall), ("old_query", old, query_history, query_wall)):
        validation = np.asarray(history["elbo_validation"].iloc[:, 0], dtype=float)
        records.append(
            {
                "fit": fit,
                "n_cells": int(data.n_obs),
                "n_donors": int(data.obs["donor_id"].astype(str).nunique()),
                "epochs_run": int(len(validation)),
                "best_validation_elbo": float(np.min(validation)),
                "wall_seconds": float(wall),
                "n_latent": N_LATENT,
                "n_layers": N_LAYERS,
                "gene_likelihood": "nb",
                "accelerator": "cpu",
                "frozen_latent_encoder_max_abs_change": max_change if fit == "old_query" else 0.0,
                "frozen_latent_encoder_parameter_count": parameter_count if fit == "old_query" else 0,
                "frozen_latent_encoder_buffer_count": buffer_count if fit == "old_query" else 0,
            }
        )
    QC.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(METRICS, index=False)
    pd.DataFrame(history_rows(ref_history, "young_reference") + history_rows(query_history, "old_query")).to_csv(HISTORY, index=False)
    print(json.dumps({
        "metrics": str(METRICS), "history": str(HISTORY), "drift_latent": str(DRIFT_LATENT),
        "n_drift_cells": int(latent.n_obs), "frozen_latent_encoder_max_abs_change": max_change,
        "peak_rss_bytes": max_rss_bytes(), "wall_seconds": time.monotonic() - started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
