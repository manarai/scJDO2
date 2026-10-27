#!/usr/bin/env python3
"""S11 operator fits in the fixed real-data common embeddings.

Use the project `.venv` for measured/local/score operators.  Invoke the
`dynamo` mode only under `.venv-dynamo`; it never touches the real H5AD files.
Every command has one dataset/source/operator unit and writes a checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks/real_flow"
ARRAYS = OUT / "arrays"
VELOCITIES = ARRAYS / "velocities"
OPERATORS = ARRAYS / "operators"
EMBED = ARRAYS / "embeddings"
CHECKPOINTS = OUT / "checkpoints"
SCNT_PROCESSED = Path.home() / "Documents/scJDO2/benchmarks/data/dynamo_hematopoiesis/data/hematopoiesis.h5ad"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def checkpoint(unit: str, started: float, payload: dict[str, Any]) -> None:
    write_json(CHECKPOINTS / f"{unit}.json", {
        "task": "orient/S11", "unit": unit,
        "started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "finished_utc": utc(), "wall_seconds": round(time.time() - started, 3), **payload,
    })


def velocity_path(dataset: str, source: str) -> Path:
    return VELOCITIES / f"{dataset}_{source}.npz"


def operator_path(dataset: str, source: str, operator: str, seed: int | None = None) -> Path:
    suffix = f"_s{seed}" if seed is not None else ""
    return OPERATORS / f"{dataset}_{source}_{operator}{suffix}.npz"


def load_source(dataset: str, source: str) -> dict[str, np.ndarray]:
    path = velocity_path(dataset, source)
    if not path.exists():
        raise FileNotFoundError(f"missing completed S11 velocity source: {path}")
    data = {key: value for key, value in np.load(path, allow_pickle=False).items()}
    z, v = data["z"], data["velocity"]
    if z.shape != v.shape or z.ndim != 2 or z.shape[1] != 30:
        raise RuntimeError(f"invalid S11 source shape in {path}: z={z.shape}, velocity={v.shape}")
    if not np.isfinite(z).all() or not np.isfinite(v).all():
        raise RuntimeError(f"non-finite completed source in {path}")
    return data


def save_operator(dataset: str, source: str, operator: str, jacobian: np.ndarray,
                  source_data: dict[str, np.ndarray], details: dict[str, Any], started: float,
                  seed: int | None = None) -> None:
    jacobian = np.asarray(jacobian, dtype=np.float64)
    n, d = source_data["z"].shape
    if jacobian.shape != (n, d, d):
        raise RuntimeError(f"operator shape {jacobian.shape} != {(n, d, d)}")
    if not np.isfinite(jacobian).all():
        raise RuntimeError("operator contains non-finite Jacobian entries")
    OPERATORS.mkdir(parents=True, exist_ok=True)
    path = operator_path(dataset, source, operator, seed)
    np.savez_compressed(path, jacobian=jacobian.astype(np.float32), z=source_data["z"].astype(np.float32),
                        cell_index=source_data["cell_index"].astype(np.int64),
                        cell_id=source_data["cell_id"].astype(str), cell_type=source_data["cell_type"].astype(str))
    checkpoint(f"operator_{dataset}_{source}_{operator}" + (f"_s{seed}" if seed is not None else ""), started, {
        "dataset": dataset, "source": source, "operator": operator, "seed": seed,
        "n_cells": int(n), "dimension": int(d), "operator_path": str(path.relative_to(ROOT)), "details": details,
    })


def run_measured(dataset: str, source: str, seed: int) -> None:
    started = time.time()
    from scjdo2.measured_drift import MeasuredDrift
    data = load_source(dataset, source)
    fitted = MeasuredDrift(30, seed=seed).fit(data["z"], data["velocity"], verbose=True)
    jacobian = fitted.jacobian_autodiff(data["z"])
    save_operator(dataset, source, "measured", jacobian, data,
                  {"seed": seed, "config": "MeasuredDrift default (n_epochs=1500, batch=256, hidden=96x3)"}, started, seed)


def run_local(dataset: str, source: str) -> None:
    started = time.time()
    from scjdo2.orient import OrientConfig
    from scjdo2.orient.base import local_jacobian, neighbours
    data = load_source(dataset, source)
    cfg = OrientConfig()
    idx = neighbours(data["z"], k=30)
    jacobian = local_jacobian(data["z"], data["velocity"], idx, ridge=cfg.ridge)
    save_operator(dataset, source, "local", jacobian, data,
                  {"k_neighbours": 30, "ridge": cfg.ridge, "implementation": "scjdo2.orient.base.local_jacobian"}, started)


def score_source_data(dataset: str) -> dict[str, np.ndarray]:
    if dataset == "scnt_hema":
        return load_source(dataset, "labelling")
    path = EMBED / "larry_x_pca30.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    emb = {key: value for key, value in np.load(path, allow_pickle=False).items()}
    index = emb["sample_index"].astype(np.int64)
    return {
        "z": emb["z_all"][index].astype(np.float32), "cell_index": index,
        "cell_id": emb["cell_id"][index].astype(str), "cell_type": emb["cell_type"][index].astype(str),
    }


def run_score(dataset: str) -> None:
    started = time.time()
    from scjdo2.energy_drift import EnergyDriftAdapter, EnergyDriftConfig
    data = score_source_data(dataset)
    cfg = EnergyDriftConfig(hidden=96, depth=3, n_epochs=1500, seed=0, batch_size=256,
                            loss="esm", lr=5e-4, beta=1.0, esm_laplacian="exact")
    fitted = EnergyDriftAdapter(30, cfg).fit(data["z"], verbose=True)
    symmetry = fitted.symmetry_check(data["z"], seed=0)
    if symmetry["max_asym"] > 1e-4:
        raise RuntimeError(f"S02 energy symmetry check fails: {symmetry['max_asym']}")
    jacobian = fitted.jacobian_autodiff(data["z"])
    # save_operator needs an unused velocity array only for the shape check; score is unsupervised.
    data["velocity"] = fitted.velocity(data["z"]).astype(np.float32)
    save_operator(dataset, "score", "score", jacobian, data,
                  {"config": {"hidden": 96, "depth": 3, "n_epochs": 1500, "seed": 0,
                              "batch_size": 256, "loss": "esm", "lr": 5e-4, "beta": 1.0,
                              "esm_laplacian": "exact"}, "symmetry": symmetry}, started)


def run_dynamo(dataset: str, source: str) -> None:
    """Run only in the verified `.venv-dynamo`; use Dynamo's analytical API."""
    started = time.time()
    import anndata as ad
    import dynamo
    data = load_source(dataset, source)
    adata = ad.AnnData(X=np.zeros((data["z"].shape[0], 1), dtype=np.float32))
    adata.obs_names = data["cell_id"].astype(str)
    adata.obsm["X_pca"] = data["z"].astype(np.float32)
    adata.obsm["velocity_pca"] = data["velocity"].astype(np.float32)
    dynamo.vf.VectorField(adata, basis="pca", normalize=False, grid_velocity=False, cores=1)
    dynamo.vf.jacobian(adata, basis="pca", method="analytical", store_in_adata=True)
    raw = np.asarray(adata.uns["jacobian_pca"]["jacobian"], dtype=np.float64)
    if raw.ndim != 3:
        raise RuntimeError(f"unexpected Dynamo analytic Jacobian shape {raw.shape}")
    jacobian = np.transpose(raw, (2, 0, 1))
    save_operator(dataset, source, "dynamo", jacobian, data,
                  {"dynamo_version": getattr(dynamo, "__version__", "1.5.3"), "method": "SparseVFC + vf.jacobian(method=analytical)",
                   "basis": "X_pca30", "vecfld_iteration": int(adata.uns["VecFld_pca"].get("iteration", -1))}, started)



def run_stored_dynamo() -> None:
    """Read-only invariant summary of the processed file's own PCA vector field."""
    started = time.time()
    import anndata as ad
    import dynamo
    if not SCNT_PROCESSED.exists():
        raise FileNotFoundError(SCNT_PROCESSED)
    adata = ad.read_h5ad(SCNT_PROCESSED)
    if "VecFld_pca" not in adata.uns or "X_pca" not in adata.obsm:
        raise RuntimeError("processed scNT file lacks stored VecFld_pca or X_pca")
    dynamo.vf.jacobian(adata, basis="pca", method="analytical", store_in_adata=True)
    jac = np.asarray(adata.uns["jacobian_pca"]["jacobian"], dtype=np.float64)
    payload = {
        "task": "orient/S11", "kind": "report_only_stored_dynamo_baseline",
        "dynamo_version": getattr(dynamo, "__version__", "1.5.3"), "basis": "processed_file_X_pca",
        "n_cells": int(adata.n_obs), "embedding_dimension": int(adata.obsm["X_pca"].shape[1]),
        "jacobian_shape_d_d_n": list(jac.shape), "jacobian_finite": bool(np.isfinite(jac).all()),
        "jacobian_median_abs": float(np.median(np.abs(jac))),
        "note": "Different PCA basis from S11 common embedding; report-only invariants, never cross-basis compared.",
    }
    write_json(OUT / "dynamo_stored_invariants.json", payload)
    checkpoint("operator_scnt_hema_stored_dynamo", started, payload)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("scnt_hema", "larry"), required=True)
    parser.add_argument("--operator", choices=("measured", "local", "dynamo", "score", "stored_dynamo"), required=True)
    parser.add_argument("--source", choices=("labelling", "scvelo", "dynamo_gene", "ot", "lineage"))
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    args = parser.parse_args()
    source_sets = {"scnt_hema": {"labelling", "scvelo", "dynamo_gene"}, "larry": {"ot", "lineage"}}
    if args.operator == "score":
        if args.source is not None or args.seed is not None:
            parser.error("score is per dataset and accepts neither --source nor --seed")
        run_score(args.dataset)
        return
    if args.operator == "stored_dynamo":
        if args.dataset != "scnt_hema" or args.source is not None or args.seed is not None:
            parser.error("stored_dynamo requires --dataset scnt_hema and no --source/--seed")
        run_stored_dynamo()
        return
    if args.source not in source_sets[args.dataset]:
        parser.error("a source belonging to the selected dataset is required")
    if args.operator == "measured":
        if args.seed is None:
            parser.error("measured requires --seed 0, 1, or 2")
        run_measured(args.dataset, args.source, args.seed)
    else:
        if args.seed is not None:
            parser.error(f"{args.operator} does not accept --seed")
        (run_local if args.operator == "local" else run_dynamo)(args.dataset, args.source)


if __name__ == "__main__":
    main()
