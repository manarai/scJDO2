#!/usr/bin/env python3
"""Fit the S06 v2 observed-data energy and labelling units.

This wrapper imports S02's fixed observed-count helper functions but deliberately
stops before S02's truth-scored orientation calls.  It writes one PCA and one FA
fitted H5AD for a requested manifest dataset.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np

import orient_end_to_end as s02
from scjdo2.energy_drift import EnergyDriftAdapter, EnergyDriftConfig
from scjdo2.measured_drift import MeasuredDrift

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks/synthetic_multimodal/MANIFEST_v2.csv"
DATA_DIR = ROOT / "benchmarks/synthetic_multimodal/data/s06"
RESULT_DIR = ROOT / "benchmarks/synthetic_multimodal/s06"
FIT_MANIFEST = RESULT_DIR / "fit_manifest.json"
PRESETS = ("v2_default", "v2_stress", "v2_batch", "v2_stationary_cycle")
METHODS = ("pca", "fa")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def utc(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def rows() -> list[dict[str, str]]:
    with MANIFEST.open(newline="") as handle:
        out = list(csv.DictReader(handle))
    expected = {(preset, seed) for preset in PRESETS for seed in range(3)}
    actual = {(row["preset"], int(row["seed"])) for row in out}
    if actual != expected:
        raise RuntimeError(f"unexpected v2 manifest units: {sorted(actual ^ expected)}")
    for row in out:
        path = ROOT / row["path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            raise RuntimeError(f"input checksum mismatch: {path}")
    return sorted(out, key=lambda row: (PRESETS.index(row["preset"]), int(row["seed"])))


def output_path(preset: str, seed: int, method: str) -> Path:
    return DATA_DIR / f"{preset}_s{seed}_{method}.h5ad"


def metrics_path(preset: str, seed: int, method: str) -> Path:
    return DATA_DIR / f"{preset}_s{seed}_{method}.metrics.json"


def read_fit_manifest() -> dict[str, dict[str, Any]]:
    if not FIT_MANIFEST.exists():
        return {}
    return {str(row["unit"]): row for row in json.loads(FIT_MANIFEST.read_text()).get("files", [])}


def write_fit_manifest(records: dict[str, dict[str, Any]]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "orient/S06",
        "stage": "observed_data_fit",
        "methods": list(METHODS),
        "s02_equivalent": {
            "energy": {"hidden": 96, "depth": 3, "n_epochs": 1500, "seed": 0,
                       "batch_size": 256, "loss": "esm", "lr": 5e-4, "beta": 1.0,
                       "esm_laplacian": "exact"},
            "labelled_and_chain_rule": "imported from scripts/orient_end_to_end.py",
        },
        "files": [records[key] for key in sorted(records)],
    }
    FIT_MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def validate_result(source: ad.AnnData, result: ad.AnnData, method: str) -> None:
    if not source.obs_names.equals(result.obs_names):
        raise RuntimeError("fitted unit changed source cell order")
    if not np.array_equal(np.asarray(source.obsm[f"X_{method}"]), np.asarray(result.obsm[f"X_{method}"])):
        raise RuntimeError(f"fitted unit changed X_{method}")
    required_obsm = {"X_drift_v", "velocity_labelling_raw", "velocity_labelling"}
    if not required_obsm.issubset(set(result.obsm)):
        raise RuntimeError("fitted unit missing observed fitted velocities")
    if "energy" not in result.obs or "scjdo2" not in result.uns or "J" not in result.uns["scjdo2"]:
        raise RuntimeError("fitted unit missing energy/J output")
    n, d = result.n_obs, np.asarray(result.obsm[f"X_{method}"]).shape[1]
    if np.asarray(result.uns["scjdo2"]["J"]).shape != (n, d, d):
        raise RuntimeError("invalid stored J shape")
    for key in required_obsm:
        value = np.asarray(result.obsm[key])
        if value.shape != (n, d) or not np.isfinite(value).all():
            raise RuntimeError(f"invalid {key} output")


def fit_one(row: dict[str, str], method: str, force: bool) -> dict[str, Any]:
    preset, seed = row["preset"], int(row["seed"])
    unit = f"{preset}_s{seed}_{method}"
    destination = output_path(preset, seed, method)
    metrics = metrics_path(preset, seed, method)
    if destination.exists() and metrics.exists() and not force:
        cached = json.loads(metrics.read_text())
        if cached.get("input_sha256") == row["sha256"] and cached.get("method") == method:
            source = ad.read_h5ad(ROOT / row["path"])
            result = ad.read_h5ad(destination)
            validate_result(source, result, method)
            if sha256(destination) != cached.get("sha256"):
                raise RuntimeError(f"cached H5AD checksum mismatch: {destination}")
            print(f"[S06 fit] reusing {unit}", flush=True)
            return cached
    started = time.time()
    source_path = ROOT / row["path"]
    source = ad.read_h5ad(source_path)
    data = source.copy()
    z = np.asarray(data.obsm[f"X_{method}"], dtype=np.float32)
    if z.ndim != 2 or z.shape[0] != data.n_obs or not np.isfinite(z).all():
        raise ValueError(f"invalid X_{method} for {unit}")
    cfg = EnergyDriftConfig(hidden=96, depth=3, n_epochs=1500, seed=0, batch_size=256,
                            loss="esm", lr=5e-4, beta=1.0, esm_laplacian="exact")
    print(f"[S06 fit] energy {unit}", flush=True)
    energy = EnergyDriftAdapter(z.shape[1], cfg).fit(z, verbose=True)
    symmetry = energy.symmetry_check(z, seed=0)
    if float(symmetry["max_asym"]) > 1e-4:
        raise RuntimeError(f"symmetry check {symmetry['max_asym']:.6g} exceeds frozen 1e-4 gate")
    data.uns["scjdo2"] = {"J": energy.jacobian_autodiff(z)}
    data.obsm["X_drift_v"] = energy.velocity(z).astype(np.float32)
    data.obs["energy"] = energy.energy(z).astype(np.float32)
    # Imported exact S02 functions: only observed layers/readouts are supplied.
    raw_gene, labelled_details, _labelled = s02.labelled_velocity(data, z)
    raw_latent = s02.chain_rule_count_velocity(data, method, raw_gene)
    data.obsm["velocity_labelling_raw"] = raw_latent
    measured = MeasuredDrift(z.shape[1], seed=0).fit(z, raw_latent, verbose=True)
    data.obsm["velocity_labelling"] = measured.velocity(z).astype(np.float32)
    validate_result(source, data, method)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data.write_h5ad(destination, compression="gzip")
    validate_result(source, ad.read_h5ad(destination), method)
    finished = time.time()
    payload = {
        "unit": unit, "preset": preset, "seed": seed, "method": method,
        "input_path": row["path"], "input_sha256": row["sha256"],
        "path": str(destination.relative_to(ROOT)), "sha256": sha256(destination),
        "bytes": destination.stat().st_size, "n_cells": int(data.n_obs),
        "symmetry_median": float(symmetry["median_asym"]), "symmetry_max": float(symmetry["max_asym"]),
        "labelled_fit_nll": float(labelled_details["held_out_nll"]),
        "labelled_fit_epochs": int(labelled_details["epochs_run"]),
        "labelled_gamma_min": float(labelled_details["gamma_min"]),
        "labelled_gamma_max": float(labelled_details["gamma_max"]),
        "started_utc": utc(started), "finished_utc": utc(finished), "wall_seconds": round(finished - started, 3),
        "regenerate": f"python scripts/s06_fit_units.py --preset {preset} --seed {seed}",
    }
    metrics.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[S06 fit] complete {unit}: {payload['wall_seconds']:.1f}s", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=PRESETS)
    parser.add_argument("--seed", type=int, choices=range(3))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if (args.preset is None) != (args.seed is None):
        parser.error("--preset and --seed must be supplied together")
    selected = [row for row in rows() if args.preset is None or (row["preset"] == args.preset and int(row["seed"]) == args.seed)]
    records = read_fit_manifest()
    for row in selected:
        for method in METHODS:
            item = fit_one(row, method, args.force)
            records[item["unit"]] = item
            write_fit_manifest(records)


if __name__ == "__main__":
    main()
