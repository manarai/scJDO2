#!/usr/bin/env python3
"""S03 fixed scVelo and Palantir runs on copied S01 datasets.

This runner never writes an S01 input.  Each output is an S01 copy plus only
scVelo latent velocities and the requested Palantir pseudotime/root keys.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import palantir
import scanpy as sc
import scvelo as scv

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks/synthetic_multimodal/MANIFEST.csv"
USER_DIR = ROOT / "benchmarks/synthetic_multimodal/data/user"
RESULT_DIR = ROOT / "benchmarks/synthetic_multimodal/s03"
USER_MANIFEST = RESULT_DIR / "user_file_manifest.json"
PRESETS = ("default", "stress", "batch")
DYNAMIC_OBSM = (
    "velocity_pca",
    "velocity_fa",
)
STOCHASTIC_OBSM = (
    "velocity_pca_stochastic",
    "velocity_fa_stochastic",
)
ADDED_OBSM = DYNAMIC_OBSM + STOCHASTIC_OBSM
ADDED_OBS = ("palantir_pseudotime",)
ADDED_UNS = ("palantir_start_cell",)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows() -> list[dict[str, str]]:
    with MANIFEST.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row for row in rows
        if row["preset"] in PRESETS and int(row["seed"]) in (0, 1, 2)
    ]
    if len(selected) != 9:
        raise RuntimeError(f"expected 9 S01 inputs, found {len(selected)}")
    return selected


def selected_rows(preset: str, seed: int | None) -> list[dict[str, str]]:
    rows = read_rows()
    return [
        row for row in rows
        if (preset == "all" or row["preset"] == preset)
        and (seed is None or int(row["seed"]) == seed)
    ]


def output_path(row: dict[str, str]) -> Path:
    return USER_DIR / f"{row['preset']}_s{int(row['seed'])}_user.h5ad"


def _array_equal(left: Any, right: Any) -> bool:
    return np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True)


def validate_output(source: ad.AnnData, result: ad.AnnData) -> None:
    """Validate that the user file retained S01 exactly apart from allowed keys."""
    if not source.obs_names.equals(result.obs_names):
        raise RuntimeError("output cell order differs from its S01 input")
    for key in ("X_pca", "X_fa"):
        if key not in source.obsm or key not in result.obsm:
            raise KeyError(f"missing required embedding {key!r}")
        if not _array_equal(source.obsm[key], result.obsm[key]):
            raise RuntimeError(f"{key} changed during the user run")
    obsm_extra = set(result.obsm) - set(source.obsm)
    obs_extra = set(result.obs.columns) - set(source.obs.columns)
    uns_extra = set(result.uns) - set(source.uns)
    if not set(DYNAMIC_OBSM).issubset(obsm_extra) or not obsm_extra.issubset(set(ADDED_OBSM)):
        raise RuntimeError(f"unexpected output obsm keys: {sorted(obsm_extra)}")
    stochastic_extra = set(STOCHASTIC_OBSM) & obsm_extra
    if stochastic_extra not in (set(), set(STOCHASTIC_OBSM)):
        raise RuntimeError("stochastic output must provide both PCA and FA keys or neither")
    if obs_extra != set(ADDED_OBS):
        raise RuntimeError(f"unexpected output obs keys: {sorted(obs_extra)}")
    if uns_extra != set(ADDED_UNS):
        raise RuntimeError(f"unexpected output uns keys: {sorted(uns_extra)}")
    for key in sorted(obsm_extra):
        values = np.asarray(result.obsm[key])
        if values.shape != np.asarray(source.obsm[key.replace("velocity_", "X_").replace("_stochastic", "")]).shape:
            raise RuntimeError(f"unexpected shape for obsm[{key!r}]: {values.shape}")
        if not np.isfinite(values).all():
            raise RuntimeError(f"non-finite values in obsm[{key!r}]")
    pseudo = np.asarray(result.obs["palantir_pseudotime"], dtype=float)
    if not np.isfinite(pseudo).all():
        raise RuntimeError("Palantir pseudotime contains non-finite values")
    root = str(result.uns["palantir_start_cell"])
    if root not in result.obs_names:
        raise RuntimeError(f"Palantir start cell {root!r} is absent from output")
    # The task fixes this expert-selected root as Palantir input and requires
    # recording it. Its fitted pseudotime rank is descriptive, not a gate.


def existing_records() -> dict[str, dict[str, Any]]:
    if not USER_MANIFEST.exists():
        return {}
    payload = json.loads(USER_MANIFEST.read_text())
    return {entry["unit"]: entry for entry in payload.get("files", [])}


def write_records(records: dict[str, dict[str, Any]]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "versions": {
            "scvelo": version("scvelo"),
            "palantir": version("palantir"),
            "scanpy": version("scanpy"),
            "anndata": version("anndata"),
        },
        "palantir_api": {
            "diffusion_maps": "palantir.utils.run_diffusion_maps(AnnData, n_components=10, knn=30, seed=0, pca_key='X_pca')",
            "multiscale": "palantir.utils.determine_multiscale_space(AnnData)",
            "pseudotime": "palantir.core.run_palantir(AnnData, early_cell=..., num_waypoints=500, knn=30, seed=0)",
        },
        "files": [records[key] for key in sorted(records)],
    }
    USER_MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_one(row: dict[str, str], force: bool = False) -> dict[str, Any]:
    preset, seed = row["preset"], int(row["seed"])
    unit = f"{preset}_s{seed}"
    source_path = ROOT / row["path"]
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    input_hash = sha256(source_path)
    if input_hash != row["sha256"]:
        raise RuntimeError(f"S01 checksum mismatch for {source_path}")
    destination = output_path(row)
    records = existing_records()
    prior = records.get(unit)
    if destination.exists() and prior is not None and not force:
        if prior.get("input_sha256") != input_hash:
            raise RuntimeError(f"refusing stale user output with changed input: {destination}")
        if sha256(destination) != prior.get("sha256"):
            raise RuntimeError(f"refusing altered user output: {destination}")
        source = ad.read_h5ad(source_path)
        result = ad.read_h5ad(destination)
        validate_output(source, result)
        print(f"[S03] reusing {unit}", flush=True)
        return prior
    if destination.exists() and not force:
        raise RuntimeError(f"user output exists without a matching manifest record: {destination}")

    started = time.time()
    source = ad.read_h5ad(source_path)
    original_pca = np.asarray(source.obsm["X_pca"]).copy()
    original_fa = np.asarray(source.obsm["X_fa"]).copy()
    original_obs_names = source.obs_names.copy()
    root = str(source.uns["suggested_start_cell"])
    if root not in source.obs_names:
        raise RuntimeError(f"suggested_start_cell {root!r} missing from {unit}")

    # All scVelo/Palantir mutations occur on this disposable copy.
    work = source.copy()
    work.X = work.layers["spliced"].copy()
    scv.pp.filter_genes(work, min_shared_counts=20)
    if not work.obs_names.equals(original_obs_names):
        raise RuntimeError("scVelo gene filtering changed cell order")
    scv.pp.normalize_per_cell(work)
    # scVelo 0.3.x removed its alias; the binding S03 revision authorizes
    # Scanpy's identical default transform of X only.
    sc.pp.log1p(work)
    sc.pp.neighbors(work, use_rep="X_pca", n_neighbors=30, random_state=0)
    connectivity_before = work.obsp["connectivities"].copy()
    # n_neighbors=None reads the Scanpy graph above; it never enters scVelo's
    # automatic-neighbor branch and therefore cannot recompute PCA/neighbors.
    scv.pp.moments(work, n_neighbors=None, n_pcs=None, use_rep=None)
    if (work.obsp["connectivities"] != connectivity_before).nnz:
        raise RuntimeError("scVelo moments altered the fixed Scanpy neighbor graph")
    if not _array_equal(original_pca, work.obsm["X_pca"]):
        raise RuntimeError("X_pca changed before scVelo velocity fitting")

    scv.tl.recover_dynamics(work)
    scv.tl.velocity(work, mode="dynamical")
    scv.tl.velocity_graph(work)
    scv.tl.velocity_embedding(work, basis="pca")
    scv.tl.velocity_embedding(work, basis="fa")
    velocity_pca = np.asarray(work.obsm["velocity_pca"]).copy()
    velocity_fa = np.asarray(work.obsm["velocity_fa"]).copy()

    # Revision 2: stochastic is a report-only sensitivity. Try it once under
    # the fixed settings; its scvelo 0.3.4 failure must not block the primary.
    stochastic: dict[str, Any]
    try:
        scv.tl.velocity(work, mode="stochastic")
        scv.tl.velocity_graph(work)
        scv.tl.velocity_embedding(work, basis="pca")
        scv.tl.velocity_embedding(work, basis="fa")
        velocity_pca_stochastic = np.asarray(work.obsm["velocity_pca"]).copy()
        velocity_fa_stochastic = np.asarray(work.obsm["velocity_fa"]).copy()
        stochastic = {"status": "complete", "error": None}
    except Exception as exc:  # Binding Revision 2: record and continue.
        velocity_pca_stochastic = None
        velocity_fa_stochastic = None
        stochastic = {"status": "error", "error_type": type(exc).__name__, "error": str(exc)}
        print(f"[S03] stochastic sensitivity failed for {unit}: {type(exc).__name__}: {exc}", flush=True)

    # AnnData-native Palantir API, with exactly the fixed S03 values.
    palantir.utils.run_diffusion_maps(work, n_components=10, knn=30, seed=0, pca_key="X_pca")
    palantir.utils.determine_multiscale_space(work)
    palantir.core.run_palantir(
        work,
        early_cell=root,
        num_waypoints=500,
        knn=30,
        seed=0,
    )
    pseudotime = np.asarray(work.obs["palantir_pseudotime"], dtype=float).copy()

    # Construct the requested S01-plus-added-keys file; scVelo/Palantir internal
    # intermediates are deliberately not copied into the user-facing output.
    result = source.copy()
    result.obsm["velocity_pca"] = velocity_pca
    result.obsm["velocity_fa"] = velocity_fa
    if stochastic["status"] == "complete":
        result.obsm["velocity_pca_stochastic"] = velocity_pca_stochastic
        result.obsm["velocity_fa_stochastic"] = velocity_fa_stochastic
    result.obs["palantir_pseudotime"] = pseudotime
    result.uns["palantir_start_cell"] = root
    validate_output(source, result)
    if not _array_equal(original_pca, result.obsm["X_pca"]) or not _array_equal(original_fa, result.obsm["X_fa"]):
        raise RuntimeError("S01 embeddings changed in user output")

    USER_DIR.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(destination)
    # Re-read from disk to validate H5AD serialization and added-key scope.
    validate_output(source, ad.read_h5ad(destination))
    finished = time.time()
    record = {
        "unit": unit,
        "preset": preset,
        "seed": seed,
        "input_path": str(source_path.relative_to(ROOT)),
        "input_sha256": input_hash,
        "path": str(destination.relative_to(ROOT)),
        "sha256": sha256(destination),
        "bytes": destination.stat().st_size,
        "keys_added": {
            "obsm": sorted(set(result.obsm) - set(source.obsm)),
            "obs": list(ADDED_OBS),
            "uns": list(ADDED_UNS),
        },
        "stochastic": stochastic,
        "versions": {"scvelo": version("scvelo"), "palantir": version("palantir"), "scanpy": version("scanpy")},
        "started_utc": datetime.fromtimestamp(started, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "finished_utc": datetime.fromtimestamp(finished, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "wall_seconds": round(finished - started, 3),
        "n_cells": int(result.n_obs),
        "n_genes_input": int(source.n_vars),
        "n_genes_scvelo": int(work.n_vars),
        "palantir_start_cell": root,
        "palantir_start_pseudotime": float(result.obs.loc[root, "palantir_pseudotime"]),
    }
    print(f"[S03] complete {unit}: {record['wall_seconds']:.1f}s", flush=True)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("all",) + PRESETS, default="all")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--force", action="store_true", help="replace a prior user output")
    args = parser.parse_args()
    records = existing_records()
    for row in selected_rows(args.preset, args.seed):
        record = run_one(row, force=args.force)
        records[record["unit"]] = record
        write_records(records)
    print(json.dumps({"user_files": len(records), "manifest": str(USER_MANIFEST.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
