"""Shared, locked helpers for exploratory aging/X04 scripts."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import torch

from fit_tms_drift import _fit_energy_drift, _full_esm_loss, _lead_re, _save_drift
from scjdo2.opspace import classify_basin_latent

ROOT = Path(__file__).resolve().parents[1]
AGING = ROOT / "benchmarks" / "aging"
OUT = AGING / "x04"
MODELS = OUT / "models"
DRIFT_SEED = 0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_reported(report_name: str, relpaths: list[str]) -> dict[str, dict]:
    """Fail closed unless on-disk inputs match the accepted upstream report."""
    report = json.loads((ROOT / "coord" / "aging" / "reports" / f"{report_name}.json").read_text())
    entries = {entry["path"]: entry for entry in report["artifacts"]}
    verified: dict[str, dict] = {}
    for relpath in relpaths:
        entry = entries[relpath]
        path = ROOT / relpath
        actual = sha256(path)
        if path.stat().st_size != int(entry["bytes"]) or actual != entry["sha256"]:
            raise RuntimeError(f"upstream checksum mismatch: {relpath}")
        verified[relpath] = {"bytes": path.stat().st_size, "sha256": actual}
    return verified


def _leading_delta(reference: torch.Tensor, query: torch.Tensor, name: str) -> tuple[float, int]:
    if query.ndim != reference.ndim or any(query.shape[i] < reference.shape[i] for i in range(reference.ndim)):
        raise RuntimeError(f"query tensor {name!r} has incompatible shape {tuple(query.shape)} vs {tuple(reference.shape)}")
    sl = tuple(slice(0, n) for n in reference.shape)
    delta = torch.max(torch.abs(query.detach().cpu().to(torch.float64)[sl] - reference.detach().cpu().to(torch.float64)))
    return float(delta.item()), int(reference.numel())


def frozen_reference_audit(reference, query) -> dict[str, float | int]:
    """Audit frozen reference latent-encoder parameters and BatchNorm buffers."""
    ref_params = dict(reference.module.named_parameters())
    query_params = dict(query.module.named_parameters())
    param_deltas: list[float] = []
    param_count = 0
    for name, ref in ref_params.items():
        if "encoder" not in name:
            continue
        if name not in query_params:
            raise RuntimeError(f"query model is missing reference encoder parameter {name!r}")
        delta, count = _leading_delta(ref, query_params[name], name)
        param_deltas.append(delta); param_count += count
    if not param_deltas:
        raise RuntimeError("no reference encoder parameters audited")

    ref_buffers = dict(reference.module.named_buffers())
    query_buffers = dict(query.module.named_buffers())
    buffer_deltas: list[float] = []
    buffer_count = 0
    for name, ref in ref_buffers.items():
        if "encoder" not in name or not any(token in name for token in ("running_mean", "running_var", "num_batches_tracked")):
            continue
        if name not in query_buffers:
            raise RuntimeError(f"query model is missing reference BatchNorm buffer {name!r}")
        delta, count = _leading_delta(ref, query_buffers[name], name)
        buffer_deltas.append(delta); buffer_count += count
    if not buffer_deltas:
        raise RuntimeError("no reference encoder BatchNorm buffers audited")
    return {
        "reference_encoder_weights_max_abs_change": float(max(param_deltas)),
        "reference_encoder_parameters_checked": int(param_count),
        "reference_encoder_batchnorm_buffers_max_abs_change": float(max(buffer_deltas)),
        "reference_encoder_batchnorm_buffers_checked": int(buffer_count),
    }


def fit_drift(adata: ad.AnnData, *, representation: str, fit_label: str, output: Path, model_output: Path) -> dict:
    """Run the A02-locked CPU ESM fit, basin classification, and persistence."""
    z = np.asarray(adata.obsm[representation], dtype=np.float32)
    if z.ndim != 2 or z.shape[0] != adata.n_obs or not np.isfinite(z).all():
        raise RuntimeError(f"invalid representation {representation!r} for {fit_label}")
    started = time.monotonic()
    drift = _fit_energy_drift(z)  # imports A02's exact locked configuration (seed 0)
    wall_seconds = time.monotonic() - started
    jacobian = drift.jacobian_autodiff(z)
    symmetry = drift.symmetry_check(z, n_cells=100, seed=DRIFT_SEED)
    lead_re = _lead_re(jacobian)
    tolerance = float(0.1 * np.std(lead_re))
    scj = adata.uns.setdefault("scjdo2", {})
    scj["J"] = jacobian
    scj["J_latent"] = jacobian
    scj["meta"] = {"clock_calibrated": False, "use_rep": representation,
                   "jacobian_method": "autodiff", "energy_drift": True}
    adata.obsm["X_drift_v"] = np.asarray(drift.velocity(z), dtype=np.float32)
    classify_basin_latent(adata, tol=tolerance, use_rep=representation, rule="lead_eig")
    adata.obs["basin_index"] = adata.obs["op_basin_index"].to_numpy(dtype=np.float32)
    final_loss = _full_esm_loss(drift, z, jacobian)
    adata.uns["x04_drift"] = {
        "fit": fit_label, "representation": representation, "seed": DRIFT_SEED,
        "device": "cpu", "n_epochs": 1500, "beta": 1.0, "lr": 5e-4,
        "loss": "esm", "esm_laplacian": "exact", "final_loss_full_data": final_loss,
        "lead_eig_tolerance": tolerance, "symmetry": symmetry, "wall_seconds": float(wall_seconds),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output)
    _save_drift(drift, model_output)
    return {"fit": fit_label, "final_loss": final_loss, "sym_median": float(symmetry["median_asym"]),
            "sym_max": float(symmetry["max_asym"]), "wall_seconds": float(wall_seconds),
            "lead_eig_tolerance": tolerance}
