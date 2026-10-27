#!/usr/bin/env python3
"""Run fixed S03 scVelo plus S06 day-1/day-4 Palantir inputs on v2 data.

All fitting inputs are observed counts, observed days, cell IDs, and stored PCA.
The minimal Palantir inputs deliberately expose only PCA coordinates and IDs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import palantir
import pandas as pd
import scanpy as sc
import scvelo as scv

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks/synthetic_multimodal/MANIFEST_v2.csv"
DATA_DIR = ROOT / "benchmarks/synthetic_multimodal/data/s06_user"
RESULT_DIR = ROOT / "benchmarks/synthetic_multimodal/s06"
RUN_MANIFEST = RESULT_DIR / "user_run_manifest.json"
PRESETS = ("v2_default", "v2_stress", "v2_batch", "v2_stationary_cycle")
NON_STATIONARY = ("v2_default", "v2_stress", "v2_batch")
N_ROOTS = 5
DYNAMIC = ("velocity_pca", "velocity_fa")
STOCHASTIC = ("velocity_pca_stochastic", "velocity_fa_stochastic")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat().replace("+00:00", "Z")


def rows() -> list[dict[str, str]]:
    with MANIFEST.open(newline="") as handle:
        out = list(csv.DictReader(handle))
    expected = {(preset, seed) for preset in PRESETS for seed in range(3)}
    actual = {(r["preset"], int(r["seed"])) for r in out}
    if actual != expected:
        raise RuntimeError(f"v2 manifest units differ: {sorted(actual ^ expected)}")
    for r in out:
        p = ROOT / r["path"]
        if not p.is_file() or sha256(p) != r["sha256"]:
            raise RuntimeError(f"v2 input checksum mismatch: {p}")
    return sorted(out, key=lambda r: (PRESETS.index(r["preset"]), int(r["seed"])))


def scvelo_path(preset: str, seed: int) -> Path:
    return DATA_DIR / f"{preset}_s{seed}_scvelo.h5ad"


def palantir_path(preset: str, seed: int, root_group: str, mode: str, root_index: int) -> Path:
    return DATA_DIR / f"{preset}_s{seed}_{root_group}_{mode}_r{root_index}.h5ad"


def validate_scvelo(source: ad.AnnData, result: ad.AnnData) -> None:
    if not source.obs_names.equals(result.obs_names):
        raise RuntimeError("scVelo output changed cell order")
    for key in ("X_pca", "X_fa"):
        if not np.array_equal(np.asarray(source.obsm[key]), np.asarray(result.obsm[key])):
            raise RuntimeError(f"scVelo output changed {key}")
    added = set(result.obsm) - set(source.obsm)
    if not set(DYNAMIC).issubset(added) or not added.issubset(set(DYNAMIC + STOCHASTIC)):
        raise RuntimeError(f"unexpected scVelo keys: {sorted(added)}")
    stochastic = set(STOCHASTIC) & added
    if stochastic not in (set(), set(STOCHASTIC)):
        raise RuntimeError("stochastic velocity must add both embeddings or neither")
    for key in added:
        method = key.removeprefix("velocity_").removesuffix("_stochastic")
        value = np.asarray(result.obsm[key])
        if value.shape != np.asarray(source.obsm[f"X_{method}"]).shape or not np.isfinite(value).all():
            raise RuntimeError(f"invalid {key}")


def validate_palantir(result: ad.AnnData, expected_names: pd.Index, given: str, mode: str, group: str) -> None:
    if not result.obs_names.equals(expected_names) or result.n_vars != 0:
        raise RuntimeError("Palantir output changed cells or retains a feature matrix")
    if list(result.obs.columns) != ["palantir_pseudotime"] or set(result.obsm) or set(result.layers) or set(result.obsp):
        raise RuntimeError("Palantir output retains unexpected arrays")
    expected_uns = {"palantir_start_cell", "palantir_given_cell", "palantir_mode", "palantir_root_group"}
    if set(result.uns) != expected_uns:
        raise RuntimeError(f"unexpected Palantir uns keys: {sorted(result.uns)}")
    pseudo = result.obs["palantir_pseudotime"].to_numpy(dtype=float)
    if not np.isfinite(pseudo).all():
        raise RuntimeError("nonfinite Palantir pseudotime")
    actual = str(result.uns["palantir_start_cell"])
    if actual not in result.obs_names or str(result.uns["palantir_given_cell"]) != given:
        raise RuntimeError("incorrect Palantir root metadata")
    if str(result.uns["palantir_mode"]) != mode or str(result.uns["palantir_root_group"]) != group:
        raise RuntimeError("incorrect Palantir mode/group")
    if actual != str(result.obs_names[int(np.argmin(pseudo))]):
        raise RuntimeError("stored actual start differs from pseudotime minimum")


def minimal_palantir_input(source: ad.AnnData) -> ad.AnnData:
    pca = np.asarray(source.obsm["X_pca"], dtype=np.float64)
    if pca.ndim != 2 or not np.isfinite(pca).all():
        raise RuntimeError("invalid stored X_pca")
    return ad.AnnData(X=np.empty((source.n_obs, 0), dtype=np.float32),
                      obs=pd.DataFrame(index=source.obs_names.copy()),
                      var=pd.DataFrame(index=pd.Index([], dtype=str)),
                      obsm={"X_pca": pca.copy()})


def root_sets(source: ad.AnnData, seed: int) -> dict[str, list[str]]:
    days = source.obs["day"].to_numpy(dtype=int)
    out: dict[str, list[str]] = {}
    for label, day, rng_seed in (("day1", 1, 1000 + seed), ("day4", 4, 2000 + seed)):
        names = source.obs_names[days == day].to_numpy(dtype=str)
        if len(names) < N_ROOTS:
            raise RuntimeError(f"{label} has only {len(names)} cells")
        out[label] = list(np.random.default_rng(rng_seed).choice(names, N_ROOTS, replace=False))
    return out


def run_scvelo(source: ad.AnnData) -> tuple[ad.AnnData, dict[str, Any], int]:
    """Exact fixed S03 scVelo block, on a disposable source copy."""
    original_pca = np.asarray(source.obsm["X_pca"]).copy()
    original_fa = np.asarray(source.obsm["X_fa"]).copy()
    work = source.copy()
    work.X = work.layers["spliced"].copy()
    scv.pp.filter_genes(work, min_shared_counts=20)
    if not work.obs_names.equals(source.obs_names):
        raise RuntimeError("scVelo gene filtering changed cell order")
    scv.pp.normalize_per_cell(work)
    # Binding S03 Revision 1: Scanpy's default log1p of X only.
    sc.pp.log1p(work)
    sc.pp.neighbors(work, use_rep="X_pca", n_neighbors=30, random_state=0)
    connectivity = work.obsp["connectivities"].copy()
    scv.pp.moments(work, n_neighbors=None, n_pcs=None, use_rep=None)
    if (work.obsp["connectivities"] != connectivity).nnz:
        raise RuntimeError("scVelo moments altered the fixed Scanpy neighbour graph")
    if not np.array_equal(original_pca, work.obsm["X_pca"]):
        raise RuntimeError("scVelo changed X_pca before velocity fitting")
    scv.tl.recover_dynamics(work)
    scv.tl.velocity(work, mode="dynamical")
    scv.tl.velocity_graph(work)
    scv.tl.velocity_embedding(work, basis="pca")
    scv.tl.velocity_embedding(work, basis="fa")
    result = source.copy()
    result.obsm["velocity_pca"] = np.asarray(work.obsm["velocity_pca"]).copy()
    result.obsm["velocity_fa"] = np.asarray(work.obsm["velocity_fa"]).copy()
    try:
        scv.tl.velocity(work, mode="stochastic")
        scv.tl.velocity_graph(work)
        scv.tl.velocity_embedding(work, basis="pca")
        scv.tl.velocity_embedding(work, basis="fa")
        result.obsm["velocity_pca_stochastic"] = np.asarray(work.obsm["velocity_pca"]).copy()
        result.obsm["velocity_fa_stochastic"] = np.asarray(work.obsm["velocity_fa"]).copy()
        stochastic = {"status": "complete", "error": None}
    except Exception as exc:
        stochastic = {"status": "error", "error_type": type(exc).__name__, "error": str(exc)}
    if not np.array_equal(original_fa, result.obsm["X_fa"]):
        raise RuntimeError("scVelo changed X_fa")
    validate_scvelo(source, result)
    return result, stochastic, int(work.n_vars)


def run_palantir(source: ad.AnnData, given: str, group: str, mode: str) -> tuple[ad.AnnData, dict[str, Any]]:
    started = time.time()
    work = minimal_palantir_input(source)
    palantir.utils.run_diffusion_maps(work, n_components=10, knn=30, seed=0, pca_key="X_pca")
    palantir.utils.determine_multiscale_space(work)
    palantir.core.run_palantir(work, early_cell=given, num_waypoints=500, knn=30, seed=0,
                               use_early_cell_as_start=(mode == "strict"))
    pseudotime = work.obs["palantir_pseudotime"].to_numpy(dtype=float)
    actual = str(work.obs_names[int(np.argmin(pseudotime))])
    output = ad.AnnData(X=np.empty((source.n_obs, 0), dtype=np.float32),
                        obs=pd.DataFrame({"palantir_pseudotime": pseudotime}, index=source.obs_names.copy()),
                        var=pd.DataFrame(index=pd.Index([], dtype=str)),
                        uns={"palantir_start_cell": actual, "palantir_given_cell": given,
                             "palantir_mode": mode, "palantir_root_group": group})
    validate_palantir(output, source.obs_names, given, mode, group)
    details = {"given_cell": given, "actual_start_cell": actual, "actual_start_day": int(source.obs.loc[actual, "day"]),
               "given_day": int(source.obs.loc[given, "day"]), "given_differs_from_actual": bool(given != actual),
               "wall_seconds": round(time.time() - started, 3)}
    return output, details


def read_manifest() -> dict[str, Any]:
    if not RUN_MANIFEST.exists():
        return {"scvelo_files": [], "palantir_runs": [], "skipped": []}
    return json.loads(RUN_MANIFEST.read_text())


def write_manifest(payload: dict[str, Any]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    payload.update({"task": "orient/S06", "versions": {name: version(name) for name in ("scvelo", "palantir", "scanpy", "numpy", "anndata")},
                    "scvelo_settings": "exact S03 dynamic primary; stochastic optional",
                    "palantir_settings": {"n_components": 10, "knn": 30, "num_waypoints": 500, "seed": 0},
                    "root_draws": {"day1": "np.random.default_rng(1000 + seed).choice(day1_names, 5, replace=False)",
                                   "day4": "np.random.default_rng(2000 + seed).choice(day4_names, 5, replace=False)"}})
    for key in ("scvelo_files", "palantir_runs", "skipped"):
        payload[key] = sorted(payload.get(key, []), key=lambda r: str(r.get("unit", "")))
    RUN_MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_dataset(row: dict[str, str], force: bool) -> None:
    preset, seed = row["preset"], int(row["seed"])
    unit = f"{preset}_s{seed}"
    source = ad.read_h5ad(ROOT / row["path"])
    payload = read_manifest()
    if preset == "v2_stationary_cycle":
        payload["skipped"] = [r for r in payload.get("skipped", []) if r.get("unit") != unit]
        payload["skipped"].append({"unit": unit, "reason": "stationary_cycle is flags-only; no four-day Palantir/scVelo scoring run"})
        write_manifest(payload)
        print(f"[S06 user] skipped {unit}: flags only", flush=True)
        return
    scv_path = scvelo_path(preset, seed)
    existing_scv = {r["unit"]: r for r in payload.get("scvelo_files", [])}.get(unit)
    if scv_path.exists() and existing_scv is not None and not force:
        if existing_scv["input_sha256"] != row["sha256"] or sha256(scv_path) != existing_scv["sha256"]:
            raise RuntimeError(f"altered/stale scVelo output: {scv_path}")
        validate_scvelo(source, ad.read_h5ad(scv_path))
        scv_record = existing_scv
    else:
        started = time.time()
        result, stochastic, n_genes = run_scvelo(source)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        result.write_h5ad(scv_path)
        validate_scvelo(source, ad.read_h5ad(scv_path))
        scv_record = {"unit": unit, "path": str(scv_path.relative_to(ROOT)), "sha256": sha256(scv_path), "bytes": scv_path.stat().st_size,
                      "input_path": row["path"], "input_sha256": row["sha256"], "n_cells": int(source.n_obs), "n_genes_scvelo": n_genes,
                      "stochastic": stochastic, "started_utc": utc(started), "finished_utc": utc(time.time()), "wall_seconds": round(time.time()-started,3),
                      "regenerate": f"python scripts/s06_user_runs.py --preset {preset} --seed {seed}"}
        payload["scvelo_files"] = [r for r in payload.get("scvelo_files", []) if r["unit"] != unit] + [scv_record]
        write_manifest(payload)
    existing_pt = {r["unit"]: r for r in payload.get("palantir_runs", [])}
    roots = root_sets(source, seed)
    for group in ("day1", "day4"):
        for mode in ("default", "strict"):
            for root_index, given in enumerate(roots[group]):
                run_unit = f"{unit}_{group}_{mode}_r{root_index}"
                destination = palantir_path(preset, seed, group, mode, root_index)
                prior = existing_pt.get(run_unit)
                if destination.exists() and prior is not None and not force:
                    if prior["input_sha256"] != row["sha256"] or prior["given_cell"] != given or sha256(destination) != prior["sha256"]:
                        raise RuntimeError(f"altered/stale Palantir output: {destination}")
                    validate_palantir(ad.read_h5ad(destination), source.obs_names, given, mode, group)
                    continue
                output, details = run_palantir(source, given, group, mode)
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                output.write_h5ad(destination)
                validate_palantir(ad.read_h5ad(destination), source.obs_names, given, mode, group)
                record = {"unit": run_unit, "preset": preset, "seed": seed, "root_group": group, "mode": mode, "root_index": root_index,
                          "input_path": row["path"], "input_sha256": row["sha256"], "path": str(destination.relative_to(ROOT)), "sha256": sha256(destination),
                          "bytes": destination.stat().st_size, **details,
                          "regenerate": f"python scripts/s06_user_runs.py --preset {preset} --seed {seed}"}
                payload["palantir_runs"] = [r for r in payload.get("palantir_runs", []) if r["unit"] != run_unit] + [record]
                existing_pt[run_unit] = record
                write_manifest(payload)
                print(f"[S06 user] complete {run_unit}: {details['wall_seconds']:.1f}s", flush=True)
    print(f"[S06 user] complete {unit}: scvelo={scv_record['wall_seconds']:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=PRESETS)
    parser.add_argument("--seed", type=int, choices=range(3))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if (args.preset is None) != (args.seed is None): parser.error("--preset and --seed are paired")
    selected = [r for r in rows() if args.preset is None or (r["preset"] == args.preset and int(r["seed"]) == args.seed)]
    for row in selected: run_dataset(row, args.force)


if __name__ == "__main__":
    main()
