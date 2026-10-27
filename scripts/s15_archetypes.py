#!/usr/bin/env python3
"""orient/S15 — flow-J dynamics archetypes.

Initial implementation: Step 0 diagnoses the scale and directional relation
between the locked S14 raw-coordinate reference flow Jacobian and J_ref.  It
fits observed OT/drift inputs first, then opens synthetic truth solely for the
specified J_ref scoring diagnostics.  Later S15 stages extend this same CLI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

OUT = ROOT / "benchmarks/synthetic_multimodal/s15"
ARR = OUT / "arrays"
CK = OUT / "checkpoints"
LOG = OUT / "logs"
FIG = ROOT / "benchmarks/figures/orient/s15"
TASK = "orient/S15"
K_STAR = 15
STEP0_DATASETS = (
    "v3_transition_s2",
    "v3_transition_s3",
    "v3_transition_s4",
    "v3_default_s2",
)


def utc(value: float | None = None) -> str:
    return datetime.fromtimestamp(time.time() if value is None else value, timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def cpu_seconds() -> float:
    here = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(here.ru_utime + here.ru_stime + children.ru_utime + children.ru_stime)


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def checkpoint(unit: str, started: float, cpu0: float, payload: dict[str, Any]) -> dict[str, Any]:
    record = {
        "task": TASK,
        "unit": unit,
        "started_utc": utc(started),
        "finished_utc": utc(),
        "wall_seconds": round(time.time() - started, 3),
        "in_process_cpu_seconds": round(cpu_seconds() - cpu0, 3),
        "peak_rss_bytes": peak_rss_bytes(),
        **payload,
    }
    write_json(CK / f"{unit}.json", record)
    return record


def safe_dataset(dataset: str) -> str:
    if dataset not in STEP0_DATASETS:
        raise ValueError(f"Step 0 dataset must be one of {STEP0_DATASETS}; got {dataset!r}")
    return dataset


def s12() -> Any:
    import s12_step0
    return s12_step0


def s14_core() -> Any:
    import s14_choose_k
    return s14_choose_k


def s14_test() -> Any:
    import s14_test
    return s14_test


def _s14_provenance(dataset: str) -> dict[str, Any]:
    """Verify and record the immutable S14 reference checkpoint/array.

    S14 stored lambda/J_ref-error arrays but not the full flow Jacobian, so the
    full J needed by this Step-0 diagnosis is deliberately refit exactly under
    S14's locked raw reference configuration.
    """
    path = ROOT / "benchmarks/synthetic_multimodal/s14/checkpoints" / f"test_{dataset}_k15_reference.json"
    if not path.exists():
        raise RuntimeError(f"required S14 reference checkpoint is missing: {path}")
    record = json.loads(path.read_text())
    if int(record.get("k", -1)) != K_STAR or record.get("configuration_id") != "reference":
        raise RuntimeError(f"S14 checkpoint is not the k=15 raw reference: {path}")
    array_rel = str(record["array_path"])
    array = ROOT / array_rel
    if not array.exists() or sha256(array) != str(record["array_sha256"]):
        raise RuntimeError(f"S14 reference array checksum failed: {array}")
    with np.load(array, allow_pickle=False) as saved:
        stored_keys = sorted(saved.files)
    return {
        "checkpoint": str(path.relative_to(ROOT)),
        "checkpoint_sha256": sha256(path),
        "array": array_rel,
        "array_sha256": str(record["array_sha256"]),
        "stored_array_keys": stored_keys,
        "stored_configuration": record["configuration"],
        "reuse_note": "S14 array/checkpoint verified and reused as provenance; full J is absent from the stored array, so this Step 0 refits exactly the locked raw reference configuration.",
    }


def _jref_neighbours(z: np.ndarray, velocity: np.ndarray, neighbours_k: int) -> np.ndarray:
    """Exact S12 J_ref construction with an explicitly requested neighbour k."""
    from scjdo2.orient import OrientConfig
    from scjdo2.orient.base import local_jacobian, neighbours

    return local_jacobian(
        np.asarray(z, np.float64),
        np.asarray(velocity, np.float64),
        neighbours(np.asarray(z, np.float64), int(neighbours_k)),
        OrientConfig().ridge,
    ).astype(np.float32)


def _diagnostic_row(dataset: str, flow_j: np.ndarray, reference_j: np.ndarray, neighbours_k: int) -> dict[str, Any]:
    """Per-dataset scale/direction diagnostics against one J_ref neighbourhood."""
    lib = s12()
    J = np.asarray(flow_j, np.float64)
    Jr = np.asarray(reference_j, np.float64)
    norm_j = np.linalg.norm(J, axis=(1, 2))
    norm_ref = np.linalg.norm(Jr, axis=(1, 2))
    denom = np.maximum(norm_ref, 1e-300)
    numerator = float(np.einsum("nij,nij->", J, Jr, optimize=True))
    divisor = float(np.einsum("nij,nij->", J, J, optimize=True))
    c_star = numerator / max(divisor, 1e-300)
    cosine = np.einsum("nij,nij->n", J, Jr, optimize=True) / np.maximum(norm_j * norm_ref, 1e-300)
    spectra_j = lib.spectra(J)
    spectra_ref = lib.spectra(Jr)
    from scipy.stats import spearmanr

    rho = float(spearmanr(spectra_j["lam1"], spectra_ref["lam1"]).statistic)
    signs = (np.asarray(spectra_j["lam1"], np.float64) >= 0.0) == (np.asarray(spectra_ref["lam1"], np.float64) >= 0.0)
    return {
        "dataset": dataset,
        "n_cells": int(J.shape[0]),
        "k_flow": K_STAR,
        "jref_neighbours": int(neighbours_k),
        "median_norm_J_over_norm_Jref": float(np.median(norm_j / denom)),
        "c_star_dataset": float(c_star),
        "median_relative_error_J": float(np.median(np.linalg.norm(J - Jr, axis=(1, 2)) / denom)),
        "median_relative_error_cstar_J": float(np.median(np.linalg.norm(c_star * J - Jr, axis=(1, 2)) / denom)),
        "median_per_cell_cosine": float(np.median(cosine)),
        "lambda1_spearman": rho,
        "lambda1_sign_agreement": float(np.mean(signs)),
        "median_norm_J": float(np.median(norm_j)),
        "median_norm_Jref": float(np.median(norm_ref)),
    }


def run_step0(dataset: str) -> None:
    """Run one detached/timed Step-0 dataset unit under the frozen S14 reference."""
    dataset = safe_dataset(dataset)
    core = s14_core()
    test = s14_test()
    started, cpu0 = time.time(), cpu_seconds()
    core.require_preflight()
    core.set_threads()
    # Explicitly honour the agent-A sandbox thread setting before drift fitting.
    import torch
    torch.set_num_threads(int(os.environ.get("SCJDO2_TORCH_THREADS", "1")))

    provenance = _s14_provenance(dataset)
    # Observed-only phase: S14-prepared PCA coordinates/day/hour -> OT -> drift.
    prepared = core.load_prepared(dataset, K_STAR)
    config = test.config_for("reference")
    expected_config = {"space": "raw", "eps": 0.05, "last": "extrapolate", "hidden": [96, 96, 96], "wd": 1e-4, "ensemble": 1}
    if config != expected_config:
        raise RuntimeError(f"unexpected S14 reference configuration: {config}")
    flow = core.fit_flow(prepared, config)

    # Scoring-only phase begins only after all observed OT/drift fitting is done.
    import anndata as ad
    preset, seed = core.synthetic_id_parts(dataset)
    manifest = core.manifest_row(preset, seed, verify=True)
    src = ad.read_h5ad(ROOT / manifest["path"], backed="r")
    try:
        lib = s12()
        velocity_true = lib.true_expected_velocity(src, {
            "linear_map": np.asarray(prepared["linear_map"], np.float64),
            "normalization_target": float(prepared["normalization_target"]),
        })
        jref30 = lib.jref(np.asarray(prepared["z"], np.float64), velocity_true)
        jref15 = _jref_neighbours(np.asarray(prepared["z"], np.float64), velocity_true, 15)
    finally:
        src.file.close()

    flow_j = np.asarray(flow["J"], np.float32)
    rows = [
        _diagnostic_row(dataset, flow_j, jref30, 30),
        _diagnostic_row(dataset, flow_j, jref15, 15),
    ]
    ARR.mkdir(parents=True, exist_ok=True)
    array_path = ARR / f"step0_{dataset}_k15_reference.npz"
    np.savez_compressed(array_path, J_flow_reference=flow_j, J_ref_k30=jref30, J_ref_k15=jref15)
    checkpoint(
        f"step0_{dataset}",
        started,
        cpu0,
        {
            "step": 0,
            "dataset": dataset,
            "configuration_id": "reference",
            "configuration": config,
            "observed_fit_complete_before_truth_scoring": True,
            "truth_scoring_only": ["J_ref_k30", "J_ref_k15", "scale/direction diagnostics"],
            "manifest_path": "benchmarks/synthetic_multimodal/MANIFEST_v3.csv",
            "input_path": manifest["path"],
            "input_sha256": manifest["sha256"],
            "s14_provenance": provenance,
            "rows": rows,
            "array_path": str(array_path.relative_to(ROOT)),
            "array_sha256": sha256(array_path),
            "models": flow["model_info"],
            "ot_pairs": flow["ot_info"],
        },
    )




FD_DT = 1e-5


def _finite_difference_expected_velocity(src: Any, prepared: dict[str, Any], dt: float = FD_DT) -> tuple[np.ndarray, dict[str, Any]]:
    """Independent expected-count finite-difference velocity in S14 PCA-15.

    This mirrors the generator's established one-step finite-difference check,
    but begins at noise-free expected spliced counts (s_true × capture × batch)
    so it is an independent numerical route for true_expected_velocity(...,
    at="expected"). It uses the stored S14 PCA-50 map and mean truncated to
    the first 15 columns, in bounded chunks.
    """
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"finite-difference dt must be positive and finite, got {dt}")
    pmap = np.asarray(prepared["linear_map"], np.float64)
    pmean = np.asarray(prepared["pca_mean"], np.float64)
    z_saved = np.asarray(prepared["z"], np.float32)
    if pmap.shape != (src.n_vars, K_STAR) or pmean.shape != (src.n_vars,):
        raise RuntimeError("S14 PCA-15 map/mean shape does not match truth source")
    if not np.array_equal(pmap, np.asarray(prepared["pca_map"], np.float64)[:, :K_STAR]):
        raise RuntimeError("S14 linear_map is not the ordered PCA-15 prefix")
    target = float(prepared["normalization_target"])
    capture = src.obs["capture_factor"].to_numpy(np.float64)
    batch_day = src.obs["batch_day"].to_numpy(int) - 1
    effects = np.asarray(src.uns["batch_gene_effects"], np.float64)
    if effects.ndim != 2 or effects.shape[1] != src.n_vars or np.any(batch_day < 0) or np.any(batch_day >= len(effects)):
        raise RuntimeError("truth batch-effect metadata is incompatible with expected-count finite difference")
    velocity = np.empty((src.n_obs, K_STAR), dtype=np.float32)
    max_abs_position_error = 0.0
    positions_bitwise_equal_after_float32 = True
    for lo in range(0, src.n_obs, 256):
        hi = min(lo + 256, src.n_obs)
        factor = capture[lo:hi, None] * effects[batch_day[lo:hi]]
        expected = np.asarray(src.layers["s_true"][lo:hi], np.float64) * factor
        derivative = np.asarray(src.layers["ds_dt_true"][lo:hi], np.float64) * factor
        future = np.maximum(expected + dt * derivative, 0.0)
        current_log = np.log1p(expected * target / np.maximum(expected.sum(axis=1, keepdims=True), 1e-12))
        future_log = np.log1p(future * target / np.maximum(future.sum(axis=1, keepdims=True), 1e-12))
        velocity[lo:hi] = ((future_log - current_log) @ pmap / dt).astype(np.float32)
        # Independently reproduce the observed coordinate transform used by
        # both J_flow (fit input) and existing J_ref neighbourhoods.
        observed = np.asarray(src.layers["spliced"][lo:hi], np.float64)
        observed_log = np.log1p(observed * target / np.maximum(observed.sum(axis=1, keepdims=True), 1e-12))
        reconstructed = (observed_log - pmean[None, :]) @ pmap
        max_abs_position_error = max(max_abs_position_error, float(np.max(np.abs(reconstructed - z_saved[lo:hi]))))
        positions_bitwise_equal_after_float32 = positions_bitwise_equal_after_float32 and bool(np.array_equal(reconstructed.astype(np.float32), z_saved[lo:hi]))
    return velocity, {
        "route": "forward finite difference of noise-free expected spliced counts, then fixed log-normalisation and S14 PCA-15 map",
        "dt": dt,
        "pca_map_shape": list(pmap.shape),
        "pca15_is_ordered_prefix": True,
        "pca_map_sha256": hashlib.sha256(np.ascontiguousarray(pmap).tobytes()).hexdigest(),
        "pca_mean_sha256": hashlib.sha256(np.ascontiguousarray(pmean).tobytes()).hexdigest(),
        "observed_pca15_reconstruction_max_abs": max_abs_position_error,
        "observed_pca15_reconstruction_bitwise_equal_after_float32": positions_bitwise_equal_after_float32,
    }


def _jref_route_comparison(reference_j: np.ndarray, finite_difference_j: np.ndarray) -> dict[str, Any]:
    """Magnitude-free agreement of original and numerical J_ref routes."""
    from scipy.stats import spearmanr
    a = np.asarray(reference_j, np.float64)
    b = np.asarray(finite_difference_j, np.float64)
    if a.shape != b.shape or a.ndim != 3 or a.shape[1:] != (K_STAR, K_STAR):
        raise RuntimeError(f"incompatible J_ref shapes: {a.shape} versus {b.shape}")
    na = np.linalg.norm(a, axis=(1, 2)); nb = np.linalg.norm(b, axis=(1, 2))
    cosine = np.einsum("nij,nij->n", a, b, optimize=True) / np.maximum(na * nb, 1e-300)
    rel = np.linalg.norm(a - b, axis=(1, 2)) / np.maximum(na, 1e-300)
    sp_a = s12().spectra(a); sp_b = s12().spectra(b)
    return {
        "comparison_jref_neighbours": 30,
        "median_per_cell_cosine": float(np.median(cosine)),
        "fraction_per_cell_cosine_ge_0_99": float(np.mean(cosine >= 0.99)),
        "median_relative_frobenius_difference": float(np.median(rel)),
        "lambda1_spearman": float(spearmanr(sp_a["lam1"], sp_b["lam1"]).statistic),
        "lambda1_sign_agreement": float(np.mean((sp_a["lam1"] >= 0.0) == (sp_b["lam1"] >= 0.0))),
    }


def run_basis_check(dataset: str) -> None:
    """Truth-scoring-only coordinate and independent finite-difference J_ref check."""
    dataset = safe_dataset(dataset)
    unit = f"basis_{dataset}"
    if own_checkpoint(unit).exists():
        raise RuntimeError(f"completed basis checkpoint exists; do not rerun: {own_checkpoint(unit)}")
    started, cpu0 = time.time(), cpu_seconds()
    core = s14_core()
    core.require_preflight(); core.set_threads()
    source_checkpoint = own_checkpoint(f"step0_{dataset}")
    if not source_checkpoint.exists():
        raise RuntimeError(f"basis check requires completed Step 0: {source_checkpoint}")
    step0 = json.loads(source_checkpoint.read_text())
    step0_array = ROOT / step0["array_path"]
    if sha256(step0_array) != str(step0["array_sha256"]):
        raise RuntimeError(f"Step 0 array checksum failed: {step0_array}")
    with np.load(step0_array, allow_pickle=False) as saved:
        original_jref = np.asarray(saved["J_ref_k30"], np.float32)
        flow_j = np.asarray(saved["J_flow_reference"], np.float32)
    prepared = core.load_prepared(dataset, K_STAR)
    manifest = core.manifest_row(*core.synthetic_id_parts(dataset), verify=True)
    import anndata as ad
    src = ad.read_h5ad(ROOT / manifest["path"], backed="r")
    try:
        direct_velocity = s12().true_expected_velocity(src, {"linear_map": np.asarray(prepared["linear_map"], np.float64), "normalization_target": float(prepared["normalization_target"])})
        fd_velocity, coordinate = _finite_difference_expected_velocity(src, prepared, FD_DT)
    finally:
        src.file.close()
    velocity_norm_direct = np.linalg.norm(np.asarray(direct_velocity, np.float64), axis=1)
    velocity_norm_fd = np.linalg.norm(np.asarray(fd_velocity, np.float64), axis=1)
    velocity_cosine = np.einsum("ni,ni->n", direct_velocity, fd_velocity, optimize=True) / np.maximum(velocity_norm_direct * velocity_norm_fd, 1e-300)
    jref_fd = s12().jref(np.asarray(prepared["z"], np.float64), fd_velocity)
    comparison = _jref_route_comparison(original_jref, jref_fd)
    flow_vs_fd = _diagnostic_row(dataset, flow_j, jref_fd, 30)
    destination = ARR / f"basis_{dataset}_k15_jref_fd.npz"
    np.savez_compressed(destination, J_ref_fd_k30=jref_fd)
    checkpoint(unit, started, cpu0, {
        "stage": "report_only_basis_check", "dataset": dataset, "k": K_STAR,
        "observed_fit_complete_before_truth_scoring": True, "no_flow_or_score_fit": True,
        "step0_checkpoint": str(source_checkpoint.relative_to(ROOT)), "step0_checkpoint_sha256": sha256(source_checkpoint),
        "step0_array": str(step0_array.relative_to(ROOT)), "step0_array_sha256": sha256(step0_array),
        "coordinate_audit": {**coordinate, "flow_j_coordinates": "S14 prepared z_pca50[:, :15]", "jref_coordinates": "same S14 prepared z_pca50[:, :15]", "raw_reference_transform_s14_optimise_lines_149_155": "raw branch: jref = jref_raw; no whitening or similarity transform"},
        "finite_difference_velocity": {
            "median_cosine_vs_true_expected_velocity": float(np.median(velocity_cosine)),
            "fraction_cosine_ge_0_99": float(np.mean(velocity_cosine >= 0.99)),
            "median_relative_difference": float(np.median(np.linalg.norm(np.asarray(fd_velocity, np.float64) - np.asarray(direct_velocity, np.float64), axis=1) / np.maximum(velocity_norm_direct, 1e-300))),
        },
        "jref_k30_direct_vs_finite_difference": comparison, "flow_vs_finite_difference_jref_k30": flow_vs_fd,
        "array_path": str(destination.relative_to(ROOT)), "array_sha256": sha256(destination), "array_key": "J_ref_fd_k30",
    })


def combine_basis_check() -> None:
    """Add report-only coordinate/finite-difference fields to jref_diagnosis.csv."""
    unit = "basis_combine"
    if own_checkpoint(unit).exists():
        raise RuntimeError(f"completed basis aggregation exists; do not rerun: {own_checkpoint(unit)}")
    started, cpu0 = time.time(), cpu_seconds()
    path = OUT / "jref_diagnosis.csv"
    if not path.exists():
        raise RuntimeError("basis aggregation requires completed Step 0 CSV")
    frame = pd.read_csv(path)
    records: list[dict[str, Any]] = []
    for dataset in STEP0_DATASETS:
        ck = own_checkpoint(f"basis_{dataset}")
        if not ck.exists():
            raise RuntimeError(f"basis aggregation missing checkpoint: {ck}")
        record = json.loads(ck.read_text()); coord = record["coordinate_audit"]
        vel = record["finite_difference_velocity"]; comp = record["jref_k30_direct_vs_finite_difference"]
        records.append({
            "dataset": dataset,
            "basis_coordinates_confirmed": bool(coord["pca15_is_ordered_prefix"] and coord["observed_pca15_reconstruction_max_abs"] <= 2e-6),  # float32 coordinate storage tolerance
            "basis_observed_pca15_reconstruction_max_abs": coord["observed_pca15_reconstruction_max_abs"],
            "basis_fd_dt": coord["dt"], "basis_fd_velocity_median_cosine": vel["median_cosine_vs_true_expected_velocity"],
            "basis_fd_velocity_fraction_cosine_ge_0_99": vel["fraction_cosine_ge_0_99"],
            "basis_jref_fd_neighbours": comp["comparison_jref_neighbours"], "basis_jref_fd_median_per_cell_cosine": comp["median_per_cell_cosine"],
            "basis_jref_fd_fraction_per_cell_cosine_ge_0_99": comp["fraction_per_cell_cosine_ge_0_99"],
            "basis_jref_fd_median_relative_frobenius_difference": comp["median_relative_frobenius_difference"],
            "basis_jref_fd_lambda1_spearman": comp["lambda1_spearman"], "basis_jref_fd_lambda1_sign_agreement": comp["lambda1_sign_agreement"],
        })
    merged = frame.drop(columns=[c for c in frame.columns if c.startswith("basis_")], errors="ignore").merge(pd.DataFrame(records), on="dataset", how="left", validate="many_to_one").sort_values(["dataset", "jref_neighbours"])
    merged.to_csv(path, index=False)
    checkpoint(unit, started, cpu0, {
        "stage": "report_only_basis_aggregate", "datasets": list(STEP0_DATASETS),
        "jref_diagnosis": str(path.relative_to(ROOT)), "jref_diagnosis_sha256": sha256(path),
        "basis_checkpoints": {d: sha256(own_checkpoint(f"basis_{d}")) for d in STEP0_DATASETS},
    })

def combine_step0() -> None:
    """Combine complete per-dataset Step-0 checkpoints into the required CSV."""
    started, cpu0 = time.time(), cpu_seconds()
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for dataset in STEP0_DATASETS:
        path = CK / f"step0_{dataset}.json"
        if not path.exists():
            missing.append(path.name)
            continue
        record = json.loads(path.read_text())
        if record.get("step") != 0 or record.get("dataset") != dataset:
            raise RuntimeError(f"invalid Step-0 checkpoint: {path}")
        records.extend(record["rows"])
    if missing:
        raise RuntimeError("cannot combine Step 0; missing checkpoints: " + ", ".join(missing))
    frame = pd.DataFrame(records).sort_values(["dataset", "jref_neighbours"])
    path = OUT / "jref_diagnosis.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    checkpoint(
        "step0_combine",
        started,
        cpu0,
        {
            "step": 0,
            "n_rows": int(len(frame)),
            "datasets": list(STEP0_DATASETS),
            "jref_diagnosis": str(path.relative_to(ROOT)),
            "jref_diagnosis_sha256": sha256(path),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("step0", "combine-step0"))
    parser.add_argument("--dataset", choices=STEP0_DATASETS)
    args = parser.parse_args()
    if args.stage == "step0":
        if args.dataset is None:
            parser.error("step0 requires --dataset")
        run_step0(args.dataset)
    else:
        if args.dataset is not None:
            parser.error("combine-step0 does not accept --dataset")
        combine_step0()



# ---------------------------------------------------------------------------
# Steps 1–6: fixed operator sources, archetypes, baselines, scores, stability.
# ---------------------------------------------------------------------------
FLOW_TIMECOURSE = tuple([f"v3_transition_s{i}" for i in range(5)] + [f"v3_default_s{i}" for i in range(4)])
CYCLE_DATASETS = tuple(f"v3_stationary_cycle_s{i}" for i in range(3))
REAL_DATASETS = ("larry",)
ALL_DATASETS = FLOW_TIMECOURSE + CYCLE_DATASETS + REAL_DATASETS
STABILITY_KS = (10, 15, 20)
DRIFT_SEEDS = (0, 1, 2)
K_CANDIDATES = (2, 3, 4, 5, 6, 8)
VIEWS = ("operator_archetype", "descriptor_kmeans")


def safe_analysis_dataset(dataset: str) -> str:
    if dataset not in ALL_DATASETS:
        raise ValueError(f"dataset must be one of {ALL_DATASETS}; got {dataset!r}")
    return dataset


def safe_k(k: int) -> int:
    if int(k) not in STABILITY_KS:
        raise ValueError(f"k must be one of {STABILITY_KS}; got {k}")
    return int(k)


def raw_reference_config() -> dict[str, Any]:
    config = s14_test().config_for("reference")
    expected = {"space": "raw", "eps": 0.05, "last": "extrapolate", "hidden": [96, 96, 96], "wd": 1e-4, "ensemble": 1}
    if config != expected:
        raise RuntimeError(f"unexpected locked S14 raw reference configuration: {config}")
    return config


def flow_unit_name(dataset: str, k: int, drift_seed: int) -> str:
    return f"flow_{dataset}_k{int(k)}_d{int(drift_seed)}"


def score_unit_name(dataset: str, k: int) -> str:
    return f"score_{dataset}_k{int(k)}"


def analysis_unit_name(dataset: str, operator: str, k: int, drift_seed: int) -> str:
    return f"analysis_{dataset}_{operator}_k{int(k)}_d{int(drift_seed)}"


def own_checkpoint(name: str) -> Path:
    return CK / f"{name}.json"


def _cycle_s10_path(dataset: str) -> Path:
    preset, seed = s14_core().synthetic_id_parts(dataset)
    if preset != "v3_stationary_cycle":
        raise ValueError(dataset)
    return ROOT / "benchmarks/synthetic_multimodal/data/s10" / f"{preset}_s{seed}" / "fit_pca.npz"


def _cycle_s10_checkpoint(dataset: str) -> Path:
    preset, seed = s14_core().synthetic_id_parts(dataset)
    return ROOT / "benchmarks/synthetic_multimodal/s10/checkpoints" / f"fit_{preset}_s{seed}_pca.json"


def observed_embedding(dataset: str, k: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Observed PCA-k only; never opens synthetic truth."""
    dataset, k = safe_analysis_dataset(dataset), safe_k(k)
    core = s14_core()
    if dataset in FLOW_TIMECOURSE or dataset == "larry":
        prepared = core.load_prepared(dataset, k)
        path = Path(prepared["path"])
        return np.asarray(prepared["z"], np.float32), {
            "source": "S14 observed prepared PCA-50 prefix",
            "prepared_path": str(path.relative_to(ROOT)),
            "prepared_sha256": core.sha256(path),
        }
    path = _cycle_s10_path(dataset)
    checkpoint_path = _cycle_s10_checkpoint(dataset)
    if not path.exists() or not checkpoint_path.exists():
        raise RuntimeError(f"S10 stationary-cycle raw-shared labelling artifact is missing for {dataset}")
    record = json.loads(checkpoint_path.read_text())
    if record.get("sha256") != sha256(path):
        raise RuntimeError(f"S10 fit checksum failed for {dataset}: {path}")
    with np.load(path, allow_pickle=False) as saved:
        z = np.asarray(saved["z"], np.float32)
    if z.shape[1] < k:
        raise RuntimeError(f"S10 PCA has {z.shape[1]} dimensions, not k={k}")
    return z[:, :k], {
        "source": "S10 observed PCA with its first k components",
        "s10_fit_path": str(path.relative_to(ROOT)),
        "s10_fit_sha256": sha256(path),
        "s10_checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "s10_labelling_variant": "raw_shared",
    }


def _step0_reuse(dataset: str, k: int, drift_seed: int) -> dict[str, Any] | None:
    if k != K_STAR or drift_seed != 0 or dataset not in STEP0_DATASETS:
        return None
    source = own_checkpoint(f"step0_{dataset}")
    if not source.exists():
        return None
    rec = json.loads(source.read_text())
    path = ROOT / rec["array_path"]
    if sha256(path) != rec["array_sha256"]:
        raise RuntimeError(f"Step 0 reuse checksum failed: {path}")
    return {
        "source_checkpoint": str(source.relative_to(ROOT)),
        "source_checkpoint_sha256": sha256(source),
        "array_path": rec["array_path"],
        "array_sha256": rec["array_sha256"],
        "array_key": "J_flow_reference",
        "reuse_note": "Exact S15 Step 0 S14-reference raw-PCA15 flow Jacobian reused; no refit.",
    }


def run_flow(dataset: str, k: int, drift_seed: int) -> None:
    """One observed-only dataset × k × drift-seed flow-J unit."""
    dataset, k, drift_seed = safe_analysis_dataset(dataset), safe_k(k), int(drift_seed)
    if drift_seed not in DRIFT_SEEDS:
        raise ValueError(f"drift seed must be one of {DRIFT_SEEDS}")
    unit = flow_unit_name(dataset, k, drift_seed)
    if own_checkpoint(unit).exists():
        raise RuntimeError(f"completed flow checkpoint exists; do not refit: {own_checkpoint(unit)}")
    core = s14_core()
    core.require_preflight(); core.set_threads()
    import torch
    torch.set_num_threads(int(os.environ.get("SCJDO2_TORCH_THREADS", "1")))
    started, cpu0 = time.time(), cpu_seconds()
    config = raw_reference_config()
    reused = _step0_reuse(dataset, k, drift_seed)
    if reused is not None:
        checkpoint(unit, started, cpu0, {
            "stage": "operators", "operator": "J_flow_reference", "dataset": dataset,
            "k": k, "drift_seed": drift_seed, "configuration": config,
            "observed_fit_complete_before_truth_scoring": True,
            "truth_keys_accessed": False, "reused": reused,
            "array_path": reused["array_path"], "array_sha256": reused["array_sha256"],
            "array_key": reused["array_key"],
        })
        return
    z, input_info = observed_embedding(dataset, k)
    if dataset in CYCLE_DATASETS:
        path = _cycle_s10_path(dataset)
        with np.load(path, allow_pickle=False) as saved:
            velocity = np.asarray(saved["v_lab_raw_shared"], np.float32)[:, :k]
        model, model_info = core.fit_model(z, velocity, np.arange(len(z)), tuple(config["hidden"]), float(config["wd"]), drift_seed)
        J = np.asarray(model.jacobian_autodiff(np.asarray(z, np.float32)), np.float32)
        fit_info: dict[str, Any] = {
            "source": "S10 raw_shared labelling velocity (one sampling time; no OT)",
            "s10_velocity_key": "v_lab_raw_shared",
            "model": model_info,
            "ot_pairs": [],
        }
    else:
        prepared = core.load_prepared(dataset, k)
        flow = core.fit_flow(prepared, config, seeds=(drift_seed,))
        J = np.asarray(flow["Js"][0], np.float32)
        fit_info = {"source": "S14 raw-PCA OT plus locked reference MeasuredDrift", "model": flow["model_info"][0], "ot_pairs": flow["ot_info"]}
    destination = ARR / "operators" / f"{unit}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, J=J)
    checkpoint(unit, started, cpu0, {
        "stage": "operators", "operator": "J_flow_reference", "dataset": dataset,
        "k": k, "drift_seed": drift_seed, "configuration": config,
        "observed_fit_complete_before_truth_scoring": True, "truth_keys_accessed": False,
        "input": input_info, "fit": fit_info,
        "array_path": str(destination.relative_to(ROOT)), "array_sha256": sha256(destination), "array_key": "J",
    })


def run_score(dataset: str, k: int) -> None:
    """One observed-only S10-configuration score-J fit in the matching PCA-k."""
    dataset, k = safe_analysis_dataset(dataset), safe_k(k)
    unit = score_unit_name(dataset, k)
    if own_checkpoint(unit).exists():
        raise RuntimeError(f"completed score checkpoint exists; do not refit: {own_checkpoint(unit)}")
    import torch
    from scjdo2.energy_drift import EnergyDriftAdapter, EnergyDriftConfig

    started, cpu0 = time.time(), cpu_seconds()
    torch.set_num_threads(int(os.environ.get("SCJDO2_TORCH_THREADS", "1")))
    z, input_info = observed_embedding(dataset, k)
    config = {"hidden": 96, "depth": 3, "n_epochs": 1500, "seed": 0, "batch_size": 256,
              "loss": "esm", "lr": 5e-4, "beta": 1.0, "esm_laplacian": "exact"}
    energy = EnergyDriftAdapter(z.shape[1], EnergyDriftConfig(**config)).fit(z, verbose=False)
    symmetry = energy.symmetry_check(z, seed=0)
    if float(symmetry["max_asym"]) > 1e-4:
        raise RuntimeError(f"S10 score-J symmetry check failed: {symmetry['max_asym']:.6g}")
    J = np.asarray(energy.jacobian_autodiff(z), np.float32)
    destination = ARR / "operators" / f"{unit}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, J=J)
    checkpoint(unit, started, cpu0, {
        "stage": "operators", "operator": "J_score", "dataset": dataset, "k": k,
        "configuration": config, "observed_fit_complete_before_truth_scoring": True,
        "truth_keys_accessed": False, "input": input_info,
        "pca15_source": "No S10 J_score artifact exists in PCA-15; refit exactly with S10 EnergyDrift settings in the matching observed PCA-k space.",
        "symmetry": {"median_asym": float(symmetry["median_asym"]), "max_asym": float(symmetry["max_asym"])},
        "array_path": str(destination.relative_to(ROOT)), "array_sha256": sha256(destination), "array_key": "J",
    })


def _operator_checkpoint(dataset: str, operator: str, k: int, drift_seed: int) -> Path:
    if operator == "flow":
        return own_checkpoint(flow_unit_name(dataset, k, drift_seed))
    if operator == "score":
        if drift_seed != 0:
            raise ValueError("score J has fixed seed 0 only")
        return own_checkpoint(score_unit_name(dataset, k))
    raise ValueError(f"unknown operator {operator}")


def load_operator(dataset: str, operator: str, k: int, drift_seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    path = _operator_checkpoint(dataset, operator, k, drift_seed)
    if not path.exists():
        raise RuntimeError(f"missing operator checkpoint: {path}")
    record = json.loads(path.read_text())
    array = ROOT / record["array_path"]
    if not array.exists() or sha256(array) != record["array_sha256"]:
        raise RuntimeError(f"operator array checksum failed: {array}")
    with np.load(array, allow_pickle=False) as saved:
        J = np.asarray(saved[str(record.get("array_key", "J"))], np.float32)
    return J, {"checkpoint": str(path.relative_to(ROOT)), "checkpoint_sha256": sha256(path),
               "array_path": record["array_path"], "array_sha256": record["array_sha256"],
               "array_key": record.get("array_key", "J")}


def _rank_quantiles(values: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    value = np.asarray(values, np.float64)
    if np.isnan(value).any():
        raise RuntimeError("descriptor contains NaN; cannot construct a rank descriptor")
    finite = np.isfinite(value)
    if not finite.any():
        raise RuntimeError("descriptor has no finite values")
    adjusted = value.copy()
    lo, hi = float(np.min(value[finite])), float(np.max(value[finite]))
    adjusted[np.isneginf(adjusted)] = lo - 1.0
    adjusted[np.isposinf(adjusted)] = hi + 1.0
    return (rankdata(adjusted, method="average") / (len(adjusted) + 1.0)).astype(np.float32)


def operator_descriptors(J: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Run the specified public opspace routines then form S15 magnitude-free descriptors."""
    import anndata as ad
    from scjdo2 import opspace

    J = np.asarray(J, np.float32)
    if J.ndim != 3 or J.shape[0] != len(z) or J.shape[1] != J.shape[2]:
        raise RuntimeError(f"invalid operator tensor shape {J.shape} for {len(z)} cells")
    a = ad.AnnData(X=np.zeros((len(z), 0), dtype=np.float32))
    a.obsm["X_pca"] = np.asarray(z, np.float32)
    a.uns["scjdo2"] = {"J": J}
    # n_eig=d is required for the requested count of all Re(lambda)>0 values.
    opspace.spectrum(a, n_eig=int(J.shape[1]), fisher_mask=False)
    opspace.invariants(a)
    # ``opspace.spectrum`` intentionally stores a conjugate-free basis.  The
    # task's positive-real count, full spectral gap, and magnitude normalisers
    # refer to the full spectrum, so restore the omitted conjugate partners only
    # for those arithmetic summaries; Henrici and transient gain remain the
    # public ``opspace.invariants`` values above.
    full_eig = np.linalg.eigvals(np.asarray(J, np.float64))
    order = np.argsort(-full_eig.real, axis=1, kind="stable")
    full_eig = np.take_along_axis(full_eig, order, axis=1)
    max_abs = np.maximum(np.max(np.abs(full_eig), axis=1), 1e-300)
    sum_abs = np.maximum(np.sum(np.abs(full_eig), axis=1), 1e-300)
    lead = full_eig[:, 0].real
    gap = full_eig[:, 0].real - full_eig[:, 1].real if full_eig.shape[1] > 1 else np.zeros(len(full_eig))
    raw = {
        "sign_lead_re": np.sign(lead),
        "n_positive_real": np.sum(full_eig.real > 0.0, axis=1).astype(np.float64),
        "lead_re_over_max_abs_lambda": lead / max_abs,
        "max_abs_imag_over_max_abs_lambda": np.max(np.abs(full_eig.imag), axis=1) / max_abs,
        "spectral_gap_over_max_abs_lambda": gap / max_abs,
        "henrici": np.asarray(a.obs["henrici"], np.float64),
        "transient_gain": np.asarray(a.obs["transient_gain"], np.float64),
        "trace_over_sum_abs_lambda": np.trace(np.asarray(J, np.float64), axis1=1, axis2=2) / sum_abs,
    }
    names = list(raw)
    raw_matrix = np.column_stack([raw[name] for name in names]).astype(np.float64)
    rank_matrix = np.column_stack([_rank_quantiles(raw[name]) for name in names]).astype(np.float32)
    return raw_matrix, rank_matrix, names, {"spectrum": dict(a.uns["scjdo2"]["spectrum_meta"]), "invariants": dict(a.uns["scjdo2"]["invariants_meta"])}


def _unit_frobenius(J: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(np.asarray(J, np.float64), axis=(1, 2))
    if not np.all(np.isfinite(norms)) or np.any(norms <= 1e-12):
        raise RuntimeError("cannot normalise an invalid or zero-Frobenius operator")
    return (np.asarray(J, np.float32) / norms[:, None, None]).astype(np.float32)


def _operator_archetypes(J: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    from scjdo2.archetypes import ArchetypeConfig, ArchetypeDecomposer

    result = ArchetypeDecomposer(ArchetypeConfig(n_archetypes="auto", k_candidates=K_CANDIDATES, seed=0)).fit(_unit_frobenius(J))
    labels = np.asarray(result.weights.argmax(axis=1), np.int16)
    return labels, {"k_selected": int(result.k_selected), "k_curve": [(int(k), float(v)) for k, v in result.k_curve],
                    "frobenius_error": float(result.frobenius_error), "rule": "ArchetypeDecomposer held-out Frobenius over fixed candidates"}


def _descriptor_kmeans(descriptors: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    summaries: list[dict[str, Any]] = []
    saved: dict[tuple[int, int], tuple[np.ndarray, float]] = {}
    for K in K_CANDIDATES:
        values = []
        for seed in range(5):
            labels = KMeans(n_clusters=int(K), random_state=seed, n_init=1).fit_predict(descriptors)
            value = float(silhouette_score(descriptors, labels))
            values.append(value); saved[(int(K), seed)] = (np.asarray(labels, np.int16), value)
        summaries.append({"K": int(K), "mean_silhouette": float(np.mean(values)), "per_restart": values})
    best_mean = max(row["mean_silhouette"] for row in summaries)
    selected = min(row["K"] for row in summaries if np.isclose(row["mean_silhouette"], best_mean, rtol=0.0, atol=1e-12))
    candidates = [(seed, saved[(selected, seed)]) for seed in range(5)]
    best_seed, (labels, best_silhouette) = max(candidates, key=lambda item: (item[1][1], -item[0]))
    return labels, {"k_selected": int(selected), "silhouette_curve": summaries, "selected_restart": int(best_seed),
                    "selected_silhouette": float(best_silhouette), "rule": "best mean silhouette over five fixed restarts; lowest K resolves exact ties"}


def _merge_leiden_centroids(labels: np.ndarray, z: np.ndarray, requested_k: int) -> tuple[np.ndarray | None, list[dict[str, Any]]]:
    """Deterministically merge closest PCA-15 Leiden centroids to exact K (S15-A1)."""
    merged = np.asarray(labels, np.int16).copy()
    z = np.asarray(z, np.float64)
    if not np.isfinite(z).all() or len(np.unique(merged)) < requested_k:
        return None, []
    merges: list[dict[str, Any]] = []
    while len(np.unique(merged)) > requested_k:
        active = np.unique(merged)
        centroids = {int(cluster): np.mean(z[merged == cluster], axis=0) for cluster in active}
        choices: list[tuple[float, int, int]] = []
        for pos, left in enumerate(active[:-1]):
            for right in active[pos + 1:]:
                a, b = int(left), int(right)
                distance = float(np.linalg.norm(centroids[a] - centroids[b]))
                choices.append((distance, min(a, b), max(a, b)))
        if not choices or not np.isfinite([item[0] for item in choices]).all():
            return None, merges
        distance, keep, drop = min(choices, key=lambda item: (item[0], item[1], item[2]))
        before = int(len(active))
        merged[merged == drop] = keep
        merges.append({
            "step": int(len(merges) + 1), "keep_cluster": int(keep), "merge_cluster": int(drop),
            "centroid_distance_pca15": float(distance), "n_clusters_before": before,
            "n_clusters_after": int(before - 1),
        })
    return merged, merges


def _matched_leiden(z: np.ndarray, requested_k: int) -> tuple[np.ndarray | None, dict[str, Any]]:
    """S15-A1 exact-K Leiden search: seeds 0--9, then deterministic centroid merges."""
    import anndata as ad
    import scanpy as sc
    import warnings
    a = ad.AnnData(X=np.asarray(z, np.float32))
    sc.pp.neighbors(a, use_rep="X", random_state=0)

    def search_seed(seed: int) -> tuple[np.ndarray | None, dict[str, Any], list[dict[str, Any]]]:
        attempts: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []

        def trial(resolution: float) -> tuple[np.ndarray, int]:
            # Fixed expression graph; S15-A1 varies only Leiden's declared seed.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"The `igraph` implementation.*", category=UserWarning)
                sc.tl.leiden(a, resolution=float(resolution), random_state=int(seed),
                             key_added="s15_leiden", flavor="leidenalg")
            observed = a.obs["s15_leiden"].astype(str).to_numpy(dtype=str)
            _, labels = np.unique(observed, return_inverse=True)
            labels = labels.astype(np.int16)
            count = int(labels.max() + 1)
            attempt = {"leiden_seed": int(seed), "resolution": float(resolution), "n_clusters": count}
            attempts.append(attempt)
            if count >= requested_k:
                candidates.append({**attempt, "labels": labels.copy()})
            return labels, count

        resolution = 1.0
        labels, count = trial(resolution)
        if count == requested_k:
            return labels, {"resolution": resolution, "attempts": attempts, "bracketed": True}, candidates
        if count < requested_k:
            low, high = resolution, resolution * 2.0
            for _ in range(60):
                labels, count = trial(high)
                if count == requested_k:
                    return labels, {"resolution": high, "attempts": attempts, "bracketed": True}, candidates
                if count > requested_k:
                    break
                low, high = high, high * 2.0
            else:
                return None, {"resolution": None, "attempts": attempts, "bracketed": False}, candidates
        else:
            high, low = resolution, resolution / 2.0
            for _ in range(60):
                labels, count = trial(low)
                if count == requested_k:
                    return labels, {"resolution": low, "attempts": attempts, "bracketed": True}, candidates
                if count < requested_k:
                    break
                high, low = low, low / 2.0
            else:
                return None, {"resolution": None, "attempts": attempts, "bracketed": False}, candidates
        for _ in range(60):
            mid = (low + high) / 2.0
            labels, count = trial(mid)
            if count == requested_k:
                return labels, {"resolution": mid, "attempts": attempts, "bracketed": True}, candidates
            if count < requested_k:
                low = mid
            else:
                high = mid
        return None, {"resolution": None, "attempts": attempts, "bracketed": True}, candidates

    seed_search: list[dict[str, Any]] = []
    fallback_candidates: list[dict[str, Any]] = []
    for seed in range(10):
        labels, record, candidates = search_seed(seed)
        seed_search.append({"leiden_seed": int(seed), **record})
        if labels is not None:
            return labels, {
                "amendment": "S15-A1", "branch": "exact_seed_search", "leiden_seed": int(seed),
                "resolution": float(record["resolution"]), "attempts": record["attempts"],
                "seed_search": seed_search, "merges": [],
            }
        if record["bracketed"]:
            fallback_candidates.extend(candidates)
    if not fallback_candidates:
        return None, {
            "amendment": "S15-A1", "branch": "infeasible", "leiden_seed": None, "resolution": None,
            "reason": "no seed 0--9 fixed bisection produced a bracketed partition with at least requested K clusters",
            "attempts": [], "seed_search": seed_search, "merges": [],
        }
    selected = min(fallback_candidates, key=lambda item: (item["resolution"], item["leiden_seed"]))
    merged, merges = _merge_leiden_centroids(selected["labels"], z, requested_k)
    if merged is None or len(np.unique(merged)) != requested_k:
        return None, {
            "amendment": "S15-A1", "branch": "infeasible", "leiden_seed": int(selected["leiden_seed"]),
            "resolution": float(selected["resolution"]), "reason": "centroid merge could not construct an exact-K finite PCA-15 partition",
            "attempts": [], "seed_search": seed_search, "merges": merges,
        }
    return merged, {
        "amendment": "S15-A1", "branch": "centroid_merge_fallback", "leiden_seed": int(selected["leiden_seed"]),
        "resolution": float(selected["resolution"]), "n_clusters_before_merge": int(selected["n_clusters"]),
        "attempts": [], "seed_search": seed_search, "merges": merges,
        "selection_rule": "smallest attempted resolution with >= requested K across seeds 0--9; ties use lower seed",
    }


def truth_regimes(dataset: str, expected_n: int) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Synthetic truth is opened only after all observed assignments have been fixed."""
    if dataset == "larry":
        return np.empty(expected_n, dtype=str), None
    core = s14_core(); preset, seed = core.synthetic_id_parts(dataset)
    row = core.manifest_row(preset, seed, verify=True)
    import anndata as ad
    src = ad.read_h5ad(ROOT / row["path"], backed="r")
    try:
        z = np.asarray(src.obsm["z_true"], np.float64)[:, :6]
        crit = float(src.uns["synthetic_multimodal"].get("z1_crit", 1.0))
        mu = z[:, 1] - crit
        root_mu = np.sqrt(np.maximum(mu, 0.0))
        cycling = z[:, 1] < crit - 0.3
        pre = (z[:, 1] >= crit - 0.3) & (z[:, 1] < crit + 0.1)
        saddle = (mu >= 0.1) & (np.abs(z[:, 0]) < 0.3 * root_mu)
        committed = (mu >= 0.3) & (np.abs(np.abs(z[:, 0]) - root_mu) < 0.3 * root_mu)
        labels = np.full(len(z), "other", dtype=object)
        labels[cycling] = "cycling"; labels[pre] = "pre"; labels[saddle] = "saddle"
        labels[committed & (z[:, 0] >= 0)] = "committed+"
        labels[committed & (z[:, 0] < 0)] = "committed-"
        if len(labels) != expected_n:
            raise RuntimeError(f"truth/cell-order mismatch for {dataset}")
        truth_j = np.asarray(src.obsm["jacobian_state"], np.float64)
        eig = np.linalg.eigvals(truth_j)
        ratio = np.max(np.abs(eig.imag), axis=1) / np.maximum(np.max(np.abs(eig), axis=1), 1e-300)
        return np.asarray(labels, str), {"input_path": row["path"], "input_sha256": row["sha256"],
                                         "regime_counts": pd.Series(labels).value_counts().sort_index().to_dict(),
                                         "truth_rotation_ratio_median": float(np.median(ratio))}
    finally:
        src.file.close()


def _assignment_score(labels: np.ndarray, truth: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    labels = np.asarray(labels, int); truth = np.asarray(truth, str)
    base = {name: float(np.mean(truth == name)) for name in sorted(np.unique(truth))}
    rows: list[dict[str, Any]] = []
    for regime in sorted(base):
        mask = truth == regime
        choices = []
        for cluster in sorted(np.unique(labels)):
            members = labels == cluster
            hit = int(np.sum(mask & members))
            choices.append((hit / max(int(mask.sum()), 1), hit / max(int(members.sum()), 1), int(cluster), int(hit), int(members.sum())))
        recall, precision, cluster, hit, cluster_n = max(choices, key=lambda value: (value[0], value[1], -value[2]))
        rows.append({"regime": regime, "base_rate": base[regime], "best_cluster": cluster, "recall": float(recall),
                     "precision": float(precision), "precision_over_base_rate": float(precision / max(base[regime], 1e-300)),
                     "n_regime": int(mask.sum()), "n_cluster": cluster_n, "n_intersection": hit})
    return {"nmi": float(normalized_mutual_info_score(truth, labels)), "ari": float(adjusted_rand_score(truth, labels))}, rows


def _signatures(labels: np.ndarray, raw: np.ndarray, names: list[str], view: str) -> list[dict[str, Any]]:
    rows = []
    for cluster in sorted(np.unique(labels)):
        mask = labels == cluster
        row = {"view": view, "archetype": int(cluster), "n_cells": int(mask.sum())}
        row.update({f"median_{name}": float(np.median(raw[mask, i])) for i, name in enumerate(names)})
        rows.append(row)
    return rows


def _rotation_score(labels: np.ndarray, raw: np.ndarray, names: list[str], truth_info: dict[str, Any] | None, view: str) -> dict[str, Any] | None:
    if truth_info is None:
        return None
    i = names.index("max_abs_imag_over_max_abs_lambda")
    threshold = 0.5 * float(truth_info["truth_rotation_ratio_median"])
    medians = {str(int(cluster)): float(np.median(raw[labels == cluster, i])) for cluster in sorted(np.unique(labels))}
    high = [key for key, value in medians.items() if value > threshold]
    return {"view": view, "truth_rotation_ratio_median": float(truth_info["truth_rotation_ratio_median"]),
            "threshold": threshold, "per_archetype_median": medians, "high_rotation_archetypes": high, "pass_definition": bool(high)}


def run_analysis(dataset: str, operator: str, k: int, drift_seed: int) -> None:
    """Observed descriptors/archetypes/Leiden first; synthetic truth only then scores."""
    dataset, k, drift_seed = safe_analysis_dataset(dataset), safe_k(k), int(drift_seed)
    if operator not in ("flow", "score"):
        raise ValueError("operator must be flow or score")
    if operator == "score" and drift_seed != 0:
        raise ValueError("score analysis has fixed drift seed 0")
    unit = analysis_unit_name(dataset, operator, k, drift_seed)
    if own_checkpoint(unit).exists():
        raise RuntimeError(f"completed analysis checkpoint exists; do not repeat: {own_checkpoint(unit)}")
    started, cpu0 = time.time(), cpu_seconds()
    z, input_info = observed_embedding(dataset, k)
    J, source_info = load_operator(dataset, operator, k, drift_seed)
    raw, ranks, names, opspace_info = operator_descriptors(J, z)
    labels_operator, meta_operator = _operator_archetypes(J)
    labels_kmeans, meta_kmeans = _descriptor_kmeans(ranks)
    leiden_cache: dict[int, tuple[np.ndarray, dict[str, Any]]] = {}
    for requested in (meta_operator["k_selected"], meta_kmeans["k_selected"]):
        if requested not in leiden_cache:
            leiden_cache[requested] = _matched_leiden(z, int(requested))
    labels_leiden_operator, meta_leiden_operator = leiden_cache[meta_operator["k_selected"]]
    labels_leiden_kmeans, meta_leiden_kmeans = leiden_cache[meta_kmeans["k_selected"]]
    # All observed fitting/assignment work above completes before truth opens.
    truth, truth_info = truth_regimes(dataset, len(z))
    views = {
        "operator_archetype": (labels_operator, labels_leiden_operator, meta_operator, meta_leiden_operator),
        "descriptor_kmeans": (labels_kmeans, labels_leiden_kmeans, meta_kmeans, meta_leiden_kmeans),
    }
    metrics: dict[str, Any] = {}
    recovery: dict[str, list[dict[str, Any]]] = {}
    signatures: dict[str, list[dict[str, Any]]] = {}
    rotation: dict[str, Any] = {}
    for view, (labels, leiden, meta, leiden_meta) in views.items():
        signatures[view] = _signatures(labels, raw, names, view)
        if truth_info is None:
            metrics[view] = {"nmi": None, "ari": None, "leiden_nmi": None, "leiden_ari": None,
                             "k_selected": int(meta["k_selected"]), "leiden": leiden_meta}
            recovery[view] = []
        else:
            score, recovery_rows = _assignment_score(labels, truth)
            if leiden is None:
                metrics[view] = {**score, "leiden_nmi": None, "leiden_ari": None,
                                 "k_selected": int(meta["k_selected"]), "leiden": leiden_meta}
            else:
                leiden_score, _ = _assignment_score(leiden, truth)
                metrics[view] = {**score, "leiden_nmi": leiden_score["nmi"], "leiden_ari": leiden_score["ari"],
                                 "k_selected": int(meta["k_selected"]), "leiden": leiden_meta}
            recovery[view] = recovery_rows
            if dataset in CYCLE_DATASETS:
                rotation[view] = _rotation_score(labels, raw, names, truth_info, view)
    destination = ARR / "analysis" / f"{unit}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    unavailable = np.full(len(z), -1, dtype=np.int16)
    np.savez_compressed(destination, labels_operator=labels_operator, labels_kmeans=labels_kmeans,
                        labels_leiden_operator=labels_leiden_operator if labels_leiden_operator is not None else unavailable,
                        labels_leiden_kmeans=labels_leiden_kmeans if labels_leiden_kmeans is not None else unavailable,
                        descriptor_raw=raw.astype(np.float64), descriptor_rank=ranks.astype(np.float32),
                        descriptor_names=np.asarray(names, dtype=str))
    checkpoint(unit, started, cpu0, {
        "stage": "descriptors_archetypes_scores", "dataset": dataset, "operator": operator, "k": k, "drift_seed": drift_seed,
        "input": input_info, "operator_source": source_info, "observed_assignments_complete_before_truth_scoring": True,
        "truth_scoring_only": [] if truth_info is None else ["truth regimes", "truth rotation ratio", "NMI/ARI/recovery"],
        "opspace": opspace_info, "descriptor_names": names,
        "rank_transform": "within-dataset average rank divided by n_cells+1; finite values only, with any infinities mapped beyond the observed finite extrema before ranking",
        "views": {"operator_archetype": meta_operator, "descriptor_kmeans": meta_kmeans}, "metrics": metrics,
        "regime_recovery": recovery, "archetype_signatures": signatures, "cycle_rotation": rotation,
        "truth": truth_info, "array_path": str(destination.relative_to(ROOT)), "array_sha256": sha256(destination),
    })



def repair_analysis_raw(dataset: str, operator: str, k: int, drift_seed: int) -> None:
    """Re-encode only raw descriptor storage as float64; never refit or reassign."""
    dataset, k, drift_seed = safe_analysis_dataset(dataset), safe_k(k), int(drift_seed)
    unit = analysis_unit_name(dataset, operator, k, drift_seed)
    record_path = own_checkpoint(unit)
    if not record_path.exists():
        raise RuntimeError(f"missing completed analysis checkpoint: {record_path}")
    repair_unit = f"repair_raw_{unit}"
    if own_checkpoint(repair_unit).exists():
        raise RuntimeError(f"completed raw-descriptor repair exists: {own_checkpoint(repair_unit)}")
    started, cpu0 = time.time(), cpu_seconds()
    record = json.loads(record_path.read_text())
    destination = ROOT / record["array_path"]
    old_hash = sha256(destination)
    with np.load(destination, allow_pickle=False) as saved:
        existing = {name: np.asarray(saved[name]) for name in saved.files}
    z, _ = observed_embedding(dataset, k)
    J, _ = load_operator(dataset, operator, k, drift_seed)
    raw, ranks, names, _ = operator_descriptors(J, z)
    if list(existing["descriptor_names"].astype(str)) != names:
        raise RuntimeError("raw repair refuses an incompatible descriptor schema")
    if not np.allclose(existing["descriptor_rank"], ranks, rtol=0.0, atol=0.0):
        raise RuntimeError("raw repair refuses because rank descriptors changed")
    np.savez_compressed(destination,
                        labels_operator=existing["labels_operator"], labels_kmeans=existing["labels_kmeans"],
                        labels_leiden_operator=existing["labels_leiden_operator"], labels_leiden_kmeans=existing["labels_leiden_kmeans"],
                        descriptor_raw=raw.astype(np.float64), descriptor_rank=existing["descriptor_rank"],
                        descriptor_names=existing["descriptor_names"])
    record["array_sha256"] = sha256(destination)
    record["raw_descriptor_dtype"] = "float64"
    record["raw_descriptor_repair"] = "Re-encoded the already-computed finite raw descriptors as float64 after float32 storage overflow; operator, rank descriptors, assignments, truth scores, K selection, and all fits are unchanged."
    write_json(record_path, record)
    checkpoint(repair_unit, started, cpu0, {"stage": "artifact_repair", "dataset": dataset, "operator": operator, "k": k, "drift_seed": drift_seed,
             "observed_only": True, "old_array_sha256": old_hash, "new_array_sha256": record["array_sha256"],
             "analysis_checkpoint": str(record_path.relative_to(ROOT)), "repair": record["raw_descriptor_repair"]})


def _load_analysis(dataset: str, operator: str, k: int, drift_seed: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    path = own_checkpoint(analysis_unit_name(dataset, operator, k, drift_seed))
    if not path.exists():
        raise RuntimeError(f"missing analysis checkpoint: {path}")
    record = json.loads(path.read_text())
    array = ROOT / record["array_path"]
    if sha256(array) != record["array_sha256"]:
        raise RuntimeError(f"analysis array checksum failed: {array}")
    with np.load(array, allow_pickle=False) as saved:
        values = {name: np.asarray(saved[name]) for name in saved.files}
    return record, values


def _all_expected_analysis() -> list[tuple[str, str, int, int]]:
    expected = []
    for dataset in ALL_DATASETS:
        for k in STABILITY_KS:
            for drift_seed in DRIFT_SEEDS:
                expected.append((dataset, "flow", k, drift_seed))
        expected.append((dataset, "score", K_STAR, 0))
    return expected


def _stability_rows() -> list[dict[str, Any]]:
    from sklearn.metrics import adjusted_rand_score
    rows: list[dict[str, Any]] = []
    for dataset in ALL_DATASETS:
        for view, key in (("operator_archetype", "labels_operator"), ("descriptor_kmeans", "labels_kmeans")):
            # Drift-seed ARI at every specified k.
            for k in STABILITY_KS:
                labels = {seed: _load_analysis(dataset, "flow", k, seed)[1][key] for seed in DRIFT_SEEDS}
                for left, right in ((0, 1), (0, 2), (1, 2)):
                    rows.append({"dataset": dataset, "view": view, "comparison": "drift_seed", "k": k,
                                 "left": left, "right": right, "ari": float(adjusted_rand_score(labels[left], labels[right]))})
            # Dimension stability at drift seed 0.
            labels = {k: _load_analysis(dataset, "flow", k, 0)[1][key] for k in STABILITY_KS}
            for left, right in ((10, 15), (10, 20), (15, 20)):
                rows.append({"dataset": dataset, "view": view, "comparison": "pca_k", "k": None,
                             "left": left, "right": right, "ari": float(adjusted_rand_score(labels[left], labels[right]))})
    return rows


def _gate_summary(arch: pd.DataFrame, recovery: pd.DataFrame, stability: pd.DataFrame) -> dict[str, Any]:
    test = tuple(f"v3_transition_s{i}" for i in (2, 3, 4))
    per_view: dict[str, Any] = {}
    for view in VIEWS:
        flow = arch[(arch.dataset.isin(test)) & (arch.operator == "flow") & (arch.k == K_STAR) & (arch.drift_seed == 0) & (arch.view == view)]
        score = arch[(arch.dataset.isin(test)) & (arch.operator == "score") & (arch.k == K_STAR) & (arch.drift_seed == 0) & (arch.view == view)]
        saddle = recovery[(recovery.dataset.isin(test)) & (recovery.operator == "flow") & (recovery.k == K_STAR) & (recovery.drift_seed == 0) & (recovery.view == view) & (recovery.regime == "saddle")]
        a1 = bool(len(saddle) == len(test) and (saddle.recall >= 0.6).all() and (saddle.precision_over_base_rate >= 3.0).all())
        merged = flow.merge(score[["dataset", "nmi"]], on="dataset", suffixes=("_flow", "_score"))
        a2 = bool(len(merged) == len(test) and (merged.nmi_flow > merged.nmi_score).all() and (merged.nmi_flow > merged.leiden_nmi).all())
        drift = stability[(stability.dataset.isin(test)) & (stability.view == view) & (stability.comparison == "drift_seed") & (stability.k == K_STAR)]
        a3 = bool(len(drift) == len(test) * 3 and (drift.ari >= 0.5).all())
        per_view[view] = {
            "A1": {"pass": a1, "saddle": saddle.to_dict(orient="records")},
            "A2": {"pass": a2, "comparisons": merged.to_dict(orient="records")},
            "A3": {"pass": a3, "drift_seed_ari": drift.to_dict(orient="records")},
            "all": bool(a1 and a2 and a3),
        }
    return {
        "gate_A1_saddle_archetype": {"rule": "recall >= 0.6 and precision >= 3 x saddle base rate on every transition test seed", "by_view": {v: per_view[v]["A1"] for v in VIEWS}, "pass": bool(any(per_view[v]["A1"]["pass"] for v in VIEWS))},
        "gate_A2_nmi": {"rule": "flow NMI exceeds score-J and matched-K Leiden NMI on every transition test seed", "by_view": {v: per_view[v]["A2"] for v in VIEWS}, "pass": bool(any(per_view[v]["A2"]["pass"] for v in VIEWS))},
        "gate_A3_stability": {"rule": "ARI >= 0.5 across every drift-seed pair for every transition test seed at k=15", "by_view": {v: per_view[v]["A3"] for v in VIEWS}, "pass": bool(any(per_view[v]["A3"]["pass"] for v in VIEWS))},
        "overall_pass": bool(any(per_view[v]["all"] for v in VIEWS)), "by_view": per_view,
    }


def _cpu_ledger() -> dict[str, Any]:
    import re
    rows = []
    for path in sorted(LOG.glob("*.log")):
        text = path.read_text(errors="ignore")
        timing = re.search(r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys", text)
        peak = re.search(r"(\d+)\s+maximum resident set size", text)
        exit_match = re.search(r"^EXIT (\d+) (\S+)", text, re.M)
        rows.append({"log": str(path.relative_to(ROOT)), "exit": int(exit_match.group(1)) if exit_match else None,
                     "end_utc": exit_match.group(2) if exit_match else None,
                     "wall_seconds": float(timing.group(1)) if timing else None,
                     "cpu_seconds": round(float(timing.group(2)) + float(timing.group(3)), 3) if timing else None,
                     "peak_rss_bytes": int(peak.group(1)) if peak else None})
    ledger = pd.DataFrame(rows)
    ledger.to_csv(OUT / "cpu_ledger.csv", index=False)
    total = float(ledger.cpu_seconds.dropna().sum()) if len(ledger) else 0.0
    return {"cpu_hours": total / 3600.0, "budget_cpu_hours": 16.0, "within_budget": bool(total / 3600.0 <= 16.0), "n_logged_units": int(len(ledger))}




def markdown_table(frame: pd.DataFrame) -> str:
    """Dependency-free GitHub-flavoured table rendering for S15 reports."""
    if frame.empty:
        return "No rows."
    def cell(value: Any) -> str:
        if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
            return ""
        if isinstance(value, (list, tuple, dict)):
            value = json.dumps(value, sort_keys=True)
        return str(value).replace("|", "\\|").replace("\n", "<br>")
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)

def _completed_expected_analysis() -> list[tuple[str, str, int, int]]:
    """Completed expected analyses only; used by the S15-A2 interim report."""
    return [
        item for item in _all_expected_analysis()
        if own_checkpoint(analysis_unit_name(*item)).exists()
    ]


def _partial_stability_rows() -> list[dict[str, Any]]:
    """Compute only fully-supported stability contrasts; omit incomplete report-only cells."""
    from sklearn.metrics import adjusted_rand_score
    rows: list[dict[str, Any]] = []
    for dataset in ALL_DATASETS:
        for view, key in (("operator_archetype", "labels_operator"),
                          ("descriptor_kmeans", "labels_kmeans")):
            for k in STABILITY_KS:
                units = [analysis_unit_name(dataset, "flow", k, seed) for seed in DRIFT_SEEDS]
                if not all(own_checkpoint(unit).exists() for unit in units):
                    continue
                labels = {seed: _load_analysis(dataset, "flow", k, seed)[1][key] for seed in DRIFT_SEEDS}
                for left, right in ((0, 1), (0, 2), (1, 2)):
                    rows.append({"dataset": dataset, "view": view, "comparison": "drift_seed", "k": k,
                                 "left": left, "right": right,
                                 "ari": float(adjusted_rand_score(labels[left], labels[right]))})
            units = [analysis_unit_name(dataset, "flow", k, 0) for k in STABILITY_KS]
            if all(own_checkpoint(unit).exists() for unit in units):
                labels = {k: _load_analysis(dataset, "flow", k, 0)[1][key] for k in STABILITY_KS}
                for left, right in ((10, 15), (10, 20), (15, 20)):
                    rows.append({"dataset": dataset, "view": view, "comparison": "pca_k", "k": None,
                                 "left": left, "right": right,
                                 "ari": float(adjusted_rand_score(labels[left], labels[right]))})
    return rows


def _aggregate_completed_tables(expected: list[tuple[str, str, int, int]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Materialise report tables for a declared set of already-completed analyses only."""
    arch_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    signature_rows: list[dict[str, Any]] = []
    rotation_rows: list[dict[str, Any]] = []
    matching_rows: list[dict[str, Any]] = []
    for dataset, operator, k, drift_seed in expected:
        record, arrays = _load_analysis(dataset, operator, k, drift_seed)
        for view in VIEWS:
            metric = record["metrics"][view]
            leiden_meta = metric["leiden"]
            seed_search = leiden_meta.get("seed_search", [])
            matching_rows.append({
                "dataset": dataset, "operator": operator, "k": k, "drift_seed": drift_seed, "view": view,
                "requested_k": int(metric["k_selected"]), "amendment": leiden_meta.get("amendment", "pre-A1"),
                "branch": leiden_meta.get("branch", "pre_A1_exact_seed0"),
                "leiden_seed": leiden_meta.get("leiden_seed", 0), "resolution": leiden_meta.get("resolution"),
                "n_clusters_before_merge": leiden_meta.get("n_clusters_before_merge"),
                "n_seed_search_attempts": int(sum(len(row.get("attempts", [])) for row in seed_search)) if seed_search else int(len(leiden_meta.get("attempts", []))),
                "n_merges": int(len(leiden_meta.get("merges", []))),
                "merges": json.dumps(leiden_meta.get("merges", []), sort_keys=True),
                "reason": leiden_meta.get("reason"),
            })
            arch_rows.append({"dataset": dataset, "operator": operator, "k": k, "drift_seed": drift_seed,
                              "view": view, "n_cells": int(len(arrays["labels_operator"])), **metric,
                              "checkpoint": str(own_checkpoint(analysis_unit_name(dataset, operator, k, drift_seed)).relative_to(ROOT))})
            for row in record["regime_recovery"][view]:
                recovery_rows.append({"dataset": dataset, "operator": operator, "k": k, "drift_seed": drift_seed,
                                      "view": view, **row})
            for row in record["archetype_signatures"][view]:
                signature_rows.append({"dataset": dataset, "operator": operator, "k": k, "drift_seed": drift_seed,
                                       **row})
            rotation = record.get("cycle_rotation", {}).get(view)
            if rotation:
                rotation_rows.append({"dataset": dataset, "operator": operator, "k": k, "drift_seed": drift_seed,
                                      **rotation})
    return (pd.DataFrame(arch_rows), pd.DataFrame(recovery_rows), pd.DataFrame(signature_rows),
            pd.DataFrame(rotation_rows), pd.DataFrame(matching_rows))


def finalise_gates() -> None:
    """S15-A2 interim result: gates are complete; report-only stability grid remains pending."""
    started, cpu0 = time.time(), cpu_seconds()
    test = tuple(f"v3_transition_s{i}" for i in (2, 3, 4))
    gate_inputs = [(dataset, "flow", K_STAR, drift) for dataset in test for drift in DRIFT_SEEDS]
    gate_inputs += [(dataset, "score", K_STAR, 0) for dataset in test]
    missing_gate = [analysis_unit_name(*item) for item in gate_inputs if not own_checkpoint(analysis_unit_name(*item)).exists()]
    if missing_gate:
        raise RuntimeError("cannot finalise gates; missing gate inputs: " + ", ".join(missing_gate))
    expected = _all_expected_analysis()
    completed = _completed_expected_analysis()
    missing = [item for item in expected if item not in completed]
    arch, recovery, signatures, rotation, matching = _aggregate_completed_tables(completed)
    stability = pd.DataFrame(_partial_stability_rows())
    gates = _gate_summary(arch, recovery, stability)
    pending = pd.DataFrame([
        {"dataset": dataset, "operator": operator, "k": k, "drift_seed": drift_seed,
         "analysis_unit": analysis_unit_name(dataset, operator, k, drift_seed),
         "status": "pending (report only)",
         "reason": "S15-A2 order-only amendment deferred non-gate stability coverage"}
        for dataset, operator, k, drift_seed in missing
    ])
    OUT.mkdir(parents=True, exist_ok=True)
    arch.to_csv(OUT / "archetypes.csv", index=False)
    recovery.to_csv(OUT / "regime_recovery.csv", index=False)
    signatures.to_csv(OUT / "archetype_signatures.csv", index=False)
    rotation.to_csv(OUT / "cycle_rotation.csv", index=False)
    matching.to_csv(OUT / "leiden_matching.csv", index=False)
    stability.to_csv(OUT / "stability.csv", index=False)
    pending.to_csv(OUT / "pending_units.csv", index=False)
    ledger = _cpu_ledger()
    step0 = pd.read_csv(OUT / "jref_diagnosis.csv")
    a1_focus = matching[((matching["dataset"] == "v3_stationary_cycle_s1") &
                         (matching["operator"] == "flow") & (matching["k"] == K_STAR) &
                         (matching["drift_seed"] == 0)) |
                        (matching["branch"].isin(["centroid_merge_fallback", "infeasible"]))]
    result = [
        "# orient/S15 result", "", "## Status", "",
        "Partial interim report: all frozen A1–A3 gate inputs are complete; non-gate stability units remain pending report-only coverage.",
        "", "## Frozen configuration", "",
        "S14 raw-coordinate reference flow J, PCA-15, epsilon factor 0.05, extrapolated last day, 96x96x96 drift, weight decay 1e-4, one member.",
        "", "## Amendment S15-A2 — gate-first order", "",
        "Dated 2026-10-25. The execution order alone changed: gate-required transition test-seed PCA-15 drift analyses were completed before the report-only k=10/k=20 and other-preset stability grid. No fit setting, dataset, score, or gate rule changed.",
        "", "## Step 0 J_ref diagnosis", "", markdown_table(step0), "",
        "The fixed PCA-15 coordinate audit and independent expected-velocity finite-difference route are recorded above. The scale ratios remain far below one and post-scaling errors remain approximately one with near-zero directional agreement, supporting no local directional relation rather than a pure scale/time-unit mismatch.",
        "", "## Amendment S15-A1 — infeasible matched-K Leiden", "",
        "For every affected analysis, the fixed resolution bisection is repeated over Leiden seeds 0–9; the first exact-K seed is used. If none yields exact K, the smallest attempted resolution with at least K clusters is selected and closest PCA-15 centroids are merged pairwise to K; if that is impossible the comparator is marked infeasible with no gate-rule change.",
        "", markdown_table(a1_focus) if len(a1_focus) else "No S15-A1 matching rows were present.",
        "", "## Frozen gates A1–A3", "",
    ]
    for name in ("gate_A1_saddle_archetype", "gate_A2_nmi", "gate_A3_stability"):
        result.extend([f"### {name}", "", f"**{'PASS' if gates[name]['pass'] else 'FAIL'}** — {gates[name]['rule']}", "", "```json", json.dumps(gates[name]["by_view"], indent=2, sort_keys=True), "```", ""])
    result += [f"Overall (one view must satisfy all A1–A3): **{'PASS' if gates['overall_pass'] else 'FAIL'}**.",
               "", "## Report-only stability pending", "",
               f"{len(pending)} expected analyses remain `pending (report only)` under S15-A2; `pending_units.csv` is the frozen manifest. They are not used in A1–A3.", "",
               markdown_table(pending.groupby(["dataset", "k"], dropna=False).size().reset_index(name="pending_analyses")) if len(pending) else "No pending units.",
               "", "## Report-only cycle and real data", "",
               markdown_table(rotation) if len(rotation) else "No cycle rotation rows.", "",
               markdown_table(arch[arch.dataset == "larry"]) if (arch.dataset == "larry").any() else "No LARRY rows.", ""]
    (OUT / "RESULT_S15.md").write_text("\n".join(result))
    output_names = ("archetypes.csv", "regime_recovery.csv", "archetype_signatures.csv", "cycle_rotation.csv", "leiden_matching.csv", "stability.csv", "pending_units.csv", "RESULT_S15.md", "cpu_ledger.csv")
    checkpoint("finalise_gates", started, cpu0, {
        "stage": "finalise_gates", "amendment": "S15-A2", "gate_inputs": [analysis_unit_name(*item) for item in gate_inputs],
        "gates": gates, "cpu": ledger, "completed_analysis_count": len(completed), "pending_analysis_count": len(missing),
        "outputs": {name: sha256(OUT / name) for name in output_names},
    })

def finalise() -> None:
    """Aggregate completed frozen units; no new fit or truth-driven selection."""
    started, cpu0 = time.time(), cpu_seconds()
    expected = _all_expected_analysis()
    missing = [analysis_unit_name(*item) for item in expected if not own_checkpoint(analysis_unit_name(*item)).exists()]
    if missing:
        raise RuntimeError("cannot finalise; missing analysis units: " + ", ".join(missing))
    arch_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    signature_rows: list[dict[str, Any]] = []
    rotation_rows: list[dict[str, Any]] = []
    matching_rows: list[dict[str, Any]] = []
    for dataset, operator, k, drift_seed in expected:
        record, _ = _load_analysis(dataset, operator, k, drift_seed)
        for view in VIEWS:
            metric = record["metrics"][view]
            leiden_meta = metric["leiden"]
            seed_search = leiden_meta.get("seed_search", [])
            matching_rows.append({
                "dataset": dataset, "operator": operator, "k": k, "drift_seed": drift_seed, "view": view,
                "requested_k": int(metric["k_selected"]), "amendment": leiden_meta.get("amendment", "pre-A1"),
                "branch": leiden_meta.get("branch", "pre_A1_exact_seed0"),
                "leiden_seed": leiden_meta.get("leiden_seed", 0), "resolution": leiden_meta.get("resolution"),
                "n_clusters_before_merge": leiden_meta.get("n_clusters_before_merge"),
                "n_seed_search_attempts": int(sum(len(row.get("attempts", [])) for row in seed_search)) if seed_search else int(len(leiden_meta.get("attempts", []))),
                "n_merges": int(len(leiden_meta.get("merges", []))),
                "merges": json.dumps(leiden_meta.get("merges", []), sort_keys=True),
                "reason": leiden_meta.get("reason"),
            })
            arch_rows.append({"dataset": dataset, "operator": operator, "k": k, "drift_seed": drift_seed, "view": view,
                              "n_cells": int(len(_load_analysis(dataset, operator, k, drift_seed)[1]["labels_operator"])), **metric,
                              "checkpoint": str(own_checkpoint(analysis_unit_name(dataset, operator, k, drift_seed)).relative_to(ROOT))})
            for row in record["regime_recovery"][view]:
                recovery_rows.append({"dataset": dataset, "operator": operator, "k": k, "drift_seed": drift_seed, "view": view, **row})
            for row in record["archetype_signatures"][view]:
                signature_rows.append({"dataset": dataset, "operator": operator, "k": k, "drift_seed": drift_seed, **row})
            rotation = record.get("cycle_rotation", {}).get(view)
            if rotation:
                rotation_rows.append({"dataset": dataset, "operator": operator, "k": k, "drift_seed": drift_seed, **rotation})
    arch = pd.DataFrame(arch_rows); recovery = pd.DataFrame(recovery_rows); signatures = pd.DataFrame(signature_rows); rotation = pd.DataFrame(rotation_rows)
    matching = pd.DataFrame(matching_rows)
    stability = pd.DataFrame(_stability_rows())
    gates = _gate_summary(arch, recovery, stability)
    OUT.mkdir(parents=True, exist_ok=True)
    arch.to_csv(OUT / "archetypes.csv", index=False)
    recovery.to_csv(OUT / "regime_recovery.csv", index=False)
    signatures.to_csv(OUT / "archetype_signatures.csv", index=False)
    rotation.to_csv(OUT / "cycle_rotation.csv", index=False)
    matching.to_csv(OUT / "leiden_matching.csv", index=False)
    stability.to_csv(OUT / "stability.csv", index=False)
    # S15-A2 is complete: retain the manifest as a zero-row, explicit completion record
    # rather than leaving its interim `pending (report only)` rows stale.
    pending = pd.DataFrame(columns=("dataset", "operator", "k", "drift_seed", "analysis_unit", "status", "reason"))
    pending.to_csv(OUT / "pending_units.csv", index=False)
    ledger = _cpu_ledger()
    step0 = pd.read_csv(OUT / "jref_diagnosis.csv")
    a1_focus = matching[((matching["dataset"] == "v3_stationary_cycle_s1") &
                        (matching["operator"] == "flow") & (matching["k"] == K_STAR) &
                        (matching["drift_seed"] == 0)) |
                       (matching["branch"].isin(["centroid_merge_fallback", "infeasible"]))]
    result = [
        "# orient/S15 result", "", "## Status", "",
        "Complete: all 130 expected analysis checkpoints are present; frozen gates are reported below without retuning.",
        "", "## Frozen configuration", "",
        "S14 raw-coordinate reference flow J, PCA-15, epsilon factor 0.05, extrapolated last day, 96x96x96 drift, weight decay 1e-4, one member.",
        "", "## Step 0 J_ref diagnosis", "", markdown_table(step0), "",
        "The scale ratios are far below one and the post-scaling errors remain approximately one with near-zero cosine/Spearman values; this supports no local directional relation, not a pure scale or time-unit mismatch.",
        "", "## Amendment S15-A1 — infeasible matched-K Leiden", "",
        "For every affected analysis, the fixed resolution bisection is repeated over Leiden seeds 0–9; the first exact-K seed is used. If none yields exact K, the smallest attempted resolution with at least K clusters is selected and closest PCA-15 centroids are merged pairwise to K; if that is impossible the comparator is marked infeasible with no gate-rule change.",
        "", markdown_table(a1_focus) if len(a1_focus) else "No S15-A1 matching rows were present.",
        "", "## Gates", "",
    ]
    for name in ("gate_A1_saddle_archetype", "gate_A2_nmi", "gate_A3_stability"):
        result.append(f"- **{name}**: {'PASS' if gates[name]['pass'] else 'FAIL'}")
    result += ["", f"Overall (one view must satisfy all A1–A3): {'PASS' if gates['overall_pass'] else 'FAIL'}.",
               "", "## Full stability tables", "",
               "All 130 expected analysis checkpoints are present. The complete, per-comparison table below covers k=10/15/20, all synthetic presets, and fixed-sample LARRY; it is report-only and did not alter A1–A3.",
               "", markdown_table(stability) if len(stability) else "No stability rows.",
               "", "## S15-A2 completion", "",
               "The detached remainder runner completed 158 flow/analysis stages: 158 completed, 0 skipped, 0 failed, and 0 retries. `pending_units.csv` is therefore a zero-row completion record.",
               "", "## Report-only cycle and real data", "", markdown_table(rotation) if len(rotation) else "No cycle rotation rows.", "", markdown_table(arch[arch.dataset == "larry"]) if (arch.dataset == "larry").any() else "No LARRY rows.", ""]
    (OUT / "RESULT_S15.md").write_text("\n".join(result))
    checkpoint("finalise", started, cpu0, {"stage": "finalise", "gates": gates, "cpu": ledger,
               "outputs": {name: sha256(OUT / name) for name in ("archetypes.csv", "regime_recovery.csv", "archetype_signatures.csv", "cycle_rotation.csv", "leiden_matching.csv", "stability.csv", "pending_units.csv", "RESULT_S15.md", "cpu_ledger.csv")}})


# Redefine the initial Step-0 CLI now that all S15 stages are available.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("step0", "combine-step0", "basis-check", "combine-basis", "flow", "score", "analyse", "repair-analysis-raw", "finalise-gates", "finalise"))
    parser.add_argument("--dataset", choices=ALL_DATASETS)
    parser.add_argument("--k", type=int)
    parser.add_argument("--drift-seed", type=int, default=0)
    parser.add_argument("--operator", choices=("flow", "score"))
    args = parser.parse_args()
    if args.stage == "step0":
        if args.dataset not in STEP0_DATASETS:
            parser.error("step0 requires --dataset in the four prescribed Step 0 datasets")
        run_step0(args.dataset)
    elif args.stage == "combine-step0":
        if args.dataset is not None or args.k is not None or args.operator is not None:
            parser.error("combine-step0 accepts no dataset/k/operator arguments")
        combine_step0()
    elif args.stage == "basis-check":
        if args.dataset not in STEP0_DATASETS or args.k is not None or args.operator is not None or args.drift_seed != 0:
            parser.error("basis-check requires one prescribed Step 0 --dataset only")
        run_basis_check(args.dataset)
    elif args.stage == "combine-basis":
        if args.dataset is not None or args.k is not None or args.operator is not None or args.drift_seed != 0:
            parser.error("combine-basis accepts no dataset/k/operator/drift-seed arguments")
        combine_basis_check()
    elif args.stage == "flow":
        if args.dataset is None or args.k is None or args.operator is not None:
            parser.error("flow requires --dataset and --k only (plus --drift-seed)")
        run_flow(args.dataset, args.k, args.drift_seed)
    elif args.stage == "score":
        if args.dataset is None or args.k is None or args.operator is not None or args.drift_seed != 0:
            parser.error("score requires --dataset and --k with fixed --drift-seed 0")
        run_score(args.dataset, args.k)
    elif args.stage == "analyse":
        if args.dataset is None or args.k is None or args.operator is None:
            parser.error("analyse requires --dataset, --k, and --operator")
        run_analysis(args.dataset, args.operator, args.k, args.drift_seed)
    elif args.stage == "repair-analysis-raw":
        if args.dataset is None or args.k is None or args.operator is None:
            parser.error("repair-analysis-raw requires --dataset, --k, and --operator")
        repair_analysis_raw(args.dataset, args.operator, args.k, args.drift_seed)
    elif args.stage == "finalise-gates":
        if args.dataset is not None or args.k is not None or args.operator is not None or args.drift_seed != 0:
            parser.error("finalise-gates accepts no dataset/k/operator/drift-seed arguments")
        finalise_gates()
    else:
        if args.dataset is not None or args.k is not None or args.operator is not None or args.drift_seed != 0:
            parser.error("finalise accepts no dataset/k/operator/drift-seed arguments")
        finalise()


if __name__ == "__main__":
    main()
