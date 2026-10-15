#!/usr/bin/env python3
"""H03 §3: fit exact energy drift in the primary and published HLCA spaces."""
from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from scjdo2.energy_drift import EnergyDriftAdapter, EnergyDriftConfig
from scjdo2.opspace import classify_basin_latent

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "aging_hlca"
DATA = OUT / "data"
QC = OUT / "qc"
PRIMARY_LATENT = DATA / "h03_primary_drift_latent.h5ad"
PUBLISHED_EMBEDDING = DATA / "HLCA_full_v1.1_emb.h5ad"
DRIFT_CELLS = OUT / "drift_subsample_cells.csv.gz"
PRIMARY_OUTPUT = DATA / "h03_primary_drift.h5ad"
SECONDARY_OUTPUT = DATA / "h03_secondary_drift.h5ad"
METRICS_OUTPUT = QC / "hlca_drift_metrics.csv"
BASIN_OUTPUT = QC / "hlca_basin_fractions.csv"
SEED = 0
EPOCHS = 1_500
BETA = 1.0
LR = 5e-4
SYMMETRY_LIMIT = 1e-4


def max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if Path("/System").exists() else value * 1024


def lead_real(J: np.ndarray) -> np.ndarray:
    values = np.empty(J.shape[0], dtype=np.float64)
    for i, jacobian in enumerate(np.asarray(J, dtype=np.float64)):
        values[i] = float(np.max(np.linalg.eigvals(jacobian).real))
    return values


def full_esm_loss(drift: EnergyDriftAdapter, z: np.ndarray, J: np.ndarray) -> float:
    score = np.asarray(drift.score(z), dtype=np.float64)
    trace_j = np.trace(np.asarray(J, dtype=np.float64), axis1=1, axis2=2)
    return float(0.5 * np.mean(np.sum(score * score, axis=1)) + np.mean(trace_j))


def fit_energy_drift(z: np.ndarray) -> EnergyDriftAdapter:
    """The frozen CPU exact-score-matching energy fit from PREREG §3."""
    return EnergyDriftAdapter(
        dim=z.shape[1],
        config=EnergyDriftConfig(
            hidden=96,
            depth=3,
            n_epochs=EPOCHS,
            seed=SEED,
            batch_size=256,
            loss="esm",
            lr=LR,
            beta=BETA,
            esm_laplacian="exact",
            device="cpu",
        ),
    ).fit(z, verbose=True)


def ordered_sample() -> pd.DataFrame:
    sample = pd.read_csv(DRIFT_CELLS, compression="gzip")
    if sample["cell_id"].astype(str).duplicated().any():
        raise RuntimeError("H03 drift sample has duplicated cell IDs")
    sample["cell_id"] = sample["cell_id"].astype(str)
    sample = sample.set_index("cell_id", drop=True)
    if len(sample) != 56_904:
        raise RuntimeError(f"unexpected H03 drift-sample size: {len(sample)}")
    return sample


def reorder_to_sample(adata: ad.AnnData, sample: pd.DataFrame) -> ad.AnnData:
    positions = pd.Index(adata.obs_names.astype(str)).get_indexer(sample.index)
    if (positions < 0).any() or len(np.unique(positions)) != len(positions):
        raise RuntimeError("mapped primary latent does not contain the exact frozen drift sample")
    result = adata[positions].copy()
    result.obs = sample.copy()
    result.obs_names = sample.index.copy()
    return result


def load_primary(sample: pd.DataFrame) -> tuple[ad.AnnData, str]:
    if not PRIMARY_LATENT.exists():
        raise FileNotFoundError(f"run hlca_fit_scvi.py first: {PRIMARY_LATENT}")
    primary = reorder_to_sample(ad.read_h5ad(PRIMARY_LATENT), sample)
    representation = "X_scVI_primary"
    z = np.asarray(primary.obsm.get(representation), dtype=np.float32)
    if z.shape != (len(sample), 15) or not np.isfinite(z).all():
        raise RuntimeError(f"primary latent is not the required 56,904×15 finite matrix: {z.shape}")
    return primary, representation


def load_published_embedding(sample: pd.DataFrame) -> tuple[ad.AnnData, str]:
    if not PUBLISHED_EMBEDDING.exists():
        raise FileNotFoundError(f"missing published HLCA embedding: {PUBLISHED_EMBEDDING}")
    source = ad.read_h5ad(PUBLISHED_EMBEDDING, backed="r")
    try:
        source_ids = pd.Index(source.obs_names.astype(str))
        positions = source_ids.get_indexer(sample.index)
        if (positions < 0).any() or len(np.unique(positions)) != len(positions):
            raise RuntimeError("published HLCA embedding cannot be matched one-to-one by cell ID")
        order = np.argsort(positions, kind="stable")
        sorted_positions = positions[order]
        values_sorted = np.asarray(source.X[sorted_positions], dtype=np.float32)
        values = np.empty_like(values_sorted)
        values[order] = values_sorted
    finally:
        source.file.close()
    if values.shape[0] != len(sample) or values.ndim != 2 or not np.isfinite(values).all():
        raise RuntimeError(f"published embedding has invalid matched shape {values.shape}")
    secondary = ad.AnnData(
        X=np.zeros((len(sample), 0), dtype=np.float32), obs=sample.copy()
    )
    secondary.obsm["X_hlca_published"] = values
    secondary.uns["h03_secondary"] = {
        "representation": "X_hlca_published",
        "source": str(PUBLISHED_EMBEDDING),
        "n_latent": int(values.shape[1]),
        "seed": SEED,
    }
    return secondary, "X_hlca_published"


def save_model(drift: EnergyDriftAdapter, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": drift._model.state_dict(), "config": vars(drift.config), "dim": int(drift.dim)},
        path,
    )


def basin_fractions(adata: ad.AnnData, fit: str) -> list[dict]:
    work = pd.DataFrame(
        {
            "ann_level_3": adata.obs["ann_level_3"].astype(str).to_numpy(),
            "behaviour": adata.obs["op_behaviour_basin"].astype(str).to_numpy(),
        }
    )
    rows: list[dict] = []
    for (cell_type, behaviour), group in work.groupby(["ann_level_3", "behaviour"], sort=True):
        rows.append({"fit": fit, "ann_level_3": str(cell_type), "behaviour": str(behaviour), "n_cells": int(len(group)), "fraction": float(len(group) / len(work))})
    return rows


def run_fit(adata: ad.AnnData, *, fit: str, representation: str, output: Path) -> tuple[dict, list[dict]]:
    z = np.asarray(adata.obsm[representation], dtype=np.float32)
    started = time.monotonic()
    print(f"[{fit}] energy fit: {z.shape[0]} cells × {z.shape[1]} dimensions", flush=True)
    drift = fit_energy_drift(z)
    fit_seconds = time.monotonic() - started
    print(f"[{fit}] calculating per-cell Jacobians", flush=True)
    J = drift.jacobian_autodiff(z)
    symmetry = drift.symmetry_check(z, n_cells=100, seed=SEED)
    if float(symmetry["max_asym"]) > SYMMETRY_LIMIT:
        raise RuntimeError(
            f"{fit}: symmetry gate failed: sym_max={symmetry['max_asym']} > {SYMMETRY_LIMIT}"
        )
    lead = lead_real(J)
    tolerance = float(0.1 * np.std(lead))
    adata.uns["scjdo2"] = {
        "J": J,
        "J_latent": J,
        "meta": {
            "clock_calibrated": False,
            "use_rep": representation,
            "jacobian_method": "autodiff",
            "energy_drift": True,
        },
    }
    adata.obsm["X_drift_v"] = np.asarray(drift.velocity(z), dtype=np.float32)
    classify_basin_latent(adata, tol=tolerance, use_rep=representation, rule="lead_eig")
    adata.obs["basin_index"] = adata.obs["op_basin_index"].to_numpy(dtype=np.float32)
    loss = full_esm_loss(drift, z, J)
    adata.uns["h03_drift"] = {
        "fit": fit,
        "representation": representation,
        "seed": SEED,
        "device": "cpu",
        "n_epochs": EPOCHS,
        "beta": BETA,
        "lr": LR,
        "loss": "esm",
        "esm_laplacian": "exact",
        "final_loss_full_data": loss,
        "lead_eig_tolerance": tolerance,
        "symmetry": symmetry,
        "wall_seconds": float(fit_seconds),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output)
    save_model(drift, DATA / f"h03_{fit}_energy.pt")
    summary = {
        "fit": fit,
        "representation": representation,
        "n_cells": int(adata.n_obs),
        "n_latent": int(z.shape[1]),
        "n_epochs": EPOCHS,
        "seed": SEED,
        "final_loss": loss,
        "lead_eig_tolerance": tolerance,
        "sym_median": float(symmetry["median_asym"]),
        "sym_max": float(symmetry["max_asym"]),
        "wall_seconds": float(fit_seconds),
        "peak_rss_bytes": max_rss_bytes(),
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary, basin_fractions(adata, fit)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.seed != SEED:
        raise ValueError("H03 drift is frozen at seed 0")
    sample = ordered_sample()
    primary, primary_representation = load_primary(sample)
    secondary, secondary_representation = load_published_embedding(sample)
    summaries, fractions = [], []
    for adata, fit, representation, output in (
        (primary, "primary", primary_representation, PRIMARY_OUTPUT),
        (secondary, "secondary", secondary_representation, SECONDARY_OUTPUT),
    ):
        summary, rows = run_fit(adata, fit=fit, representation=representation, output=output)
        summaries.append(summary)
        fractions.extend(rows)
    QC.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(METRICS_OUTPUT, index=False)
    pd.DataFrame(fractions).to_csv(BASIN_OUTPUT, index=False)
    print(json.dumps({
        "metrics": str(METRICS_OUTPUT), "basin_fractions": str(BASIN_OUTPUT),
        "primary_output": str(PRIMARY_OUTPUT), "secondary_output": str(SECONDARY_OUTPUT),
        "peak_rss_bytes": max_rss_bytes(), "finished_utc": datetime.now(timezone.utc).isoformat(),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
