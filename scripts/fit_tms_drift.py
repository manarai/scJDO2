#!/usr/bin/env python3
"""A02 §4: fit frozen energy-drift models and classify TMS latent basins."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from scjdo2.energy_drift import EnergyDriftAdapter, EnergyDriftConfig
from scjdo2.opspace import classify_basin_latent


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "benchmarks" / "aging"
QC_DIR = OUT_DIR / "qc"
MODEL_DIR = OUT_DIR / "models"
TISSUES = ("lung", "marrow")
DRIFT_EPOCHS = 1500
SEED = 0
BETA = 1.0
LR = 5e-4


def _fit_energy_drift(z: np.ndarray) -> EnergyDriftAdapter:
    """Instantiate the locked exact-score-matching energy model on CPU."""
    return EnergyDriftAdapter(
        dim=z.shape[1],
        config=EnergyDriftConfig(
            hidden=96,
            depth=3,
            n_epochs=DRIFT_EPOCHS,
            seed=SEED,
            batch_size=256,
            loss="esm",
            lr=LR,
            beta=BETA,
            esm_laplacian="exact",
            device="cpu",
        ),
    ).fit(z, verbose=False)


def _lead_re(J: np.ndarray) -> np.ndarray:
    """Return the leading real eigenvalue used by the preregistered rule."""
    out = np.empty(J.shape[0], dtype=np.float64)
    for i, jacobian in enumerate(J):
        out[i] = float(np.max(np.linalg.eigvals(jacobian.astype(np.float64)).real))
    return out


def _full_esm_loss(drift: EnergyDriftAdapter, z: np.ndarray, J: np.ndarray) -> float:
    """Evaluate the locked ESM objective on all cells after the final update."""
    score = np.asarray(drift.score(z), dtype=np.float64)
    trace_j = np.trace(np.asarray(J, dtype=np.float64), axis1=1, axis2=2)
    # J = -Hess(E) because beta=1, so 1/2||grad E||² - tr Hess(E)
    # equals 1/2||score||² + tr(J).
    return float(0.5 * np.mean(np.sum(score * score, axis=1)) + np.mean(trace_j))


def _save_drift(drift: EnergyDriftAdapter, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": drift._model.state_dict(),  # model is fitted above
            "config": vars(drift.config),
            "dim": int(drift.dim),
        },
        path,
    )


def _basin_rows(adata: ad.AnnData, fit: str) -> list[dict]:
    if "cell_type" not in adata.obs or "age_group" not in adata.obs:
        raise RuntimeError("drift input lacks cell_type or age_group")
    behaviour = adata.obs["op_behaviour_basin"].astype(str)
    work = pd.DataFrame(
        {
            "cell_type": adata.obs["cell_type"].astype(str).to_numpy(),
            "age_group": adata.obs["age_group"].astype(str).to_numpy(),
            "is_holder": behaviour.eq("holder").to_numpy(),
        }
    )
    rows: list[dict] = []
    for (cell_type, age_group), group in work.groupby(["cell_type", "age_group"], sort=True):
        rows.append(
            {
                "fit": fit,
                "cell_type": str(cell_type),
                "age_group": str(age_group),
                "n": int(len(group)),
                "basin_fraction": float(group["is_holder"].mean()),
            }
        )
    return rows


def _run_fit(adata: ad.AnnData, fit: str, representation: str, output: Path) -> tuple[dict, list[dict]]:
    z = np.asarray(adata.obsm[representation], dtype=np.float32)
    if z.ndim != 2 or z.shape[0] != adata.n_obs or not np.isfinite(z).all():
        raise RuntimeError(f"invalid latent representation {representation!r} for {fit}")
    started = time.monotonic()
    drift = _fit_energy_drift(z)
    wall_seconds = time.monotonic() - started
    J = drift.jacobian_autodiff(z)
    symmetry = drift.symmetry_check(z, n_cells=100, seed=SEED)
    lead_re = _lead_re(J)
    tolerance = float(0.1 * np.std(lead_re))

    scj = adata.uns.setdefault("scjdo2", {})
    scj["J"] = J
    scj["J_latent"] = J
    scj["meta"] = {
        "clock_calibrated": False,
        "use_rep": representation,
        "jacobian_method": "autodiff",
        "energy_drift": True,
    }
    adata.obsm["X_drift_v"] = np.asarray(drift.velocity(z), dtype=np.float32)
    # PREREG: lead-eigenvalue rule with tol = 0.1 * std(lead_re).
    classify_basin_latent(adata, tol=tolerance, use_rep=representation, rule="lead_eig")
    adata.obs["basin_index"] = adata.obs["op_basin_index"].to_numpy(dtype=np.float32)
    final_loss = _full_esm_loss(drift, z, J)
    adata.uns["a02_drift"] = {
        "fit": fit,
        "representation": representation,
        "seed": SEED,
        "device": "cpu",
        "n_epochs": DRIFT_EPOCHS,
        "beta": BETA,
        "lr": LR,
        "loss": "esm",
        "esm_laplacian": "exact",
        "final_loss_full_data": final_loss,
        "lead_eig_tolerance": tolerance,
        "symmetry": symmetry,
        "wall_seconds": float(wall_seconds),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output)
    _save_drift(drift, MODEL_DIR / f"{output.stem}_drift.pt")
    summary = {
        "fit": fit,
        "final_loss": final_loss,
        "sym_median": float(symmetry["median_asym"]),
        "sym_max": float(symmetry["max_asym"]),
    }
    return summary, _basin_rows(adata, fit)


def run_tissue(tissue: str) -> None:
    shared_path = OUT_DIR / f"{tissue}_shared.h5ad"
    if not shared_path.exists():
        raise FileNotFoundError(f"run fit_tms_scvi.py first: {shared_path}")
    source = ad.read_h5ad(shared_path)
    young_mask = source.obs["age_group"].astype(str).eq("young").to_numpy()
    old_mask = source.obs["age_group"].astype(str).eq("old").to_numpy()
    fits = (
        ("shared", source.copy(), "X_scVI_shared", OUT_DIR / f"{tissue}_shared.h5ad"),
        ("standalone_young", source[young_mask].copy(), "X_scVI_young", OUT_DIR / f"{tissue}_standalone_young.h5ad"),
        ("standalone_old", source[old_mask].copy(), "X_scVI_old", OUT_DIR / f"{tissue}_standalone_old.h5ad"),
    )
    drift_rows: list[dict] = []
    basin_rows: list[dict] = []
    for fit, work, representation, output in fits:
        print(f"[{tissue}] fitting {fit} on {work.n_obs} cells", flush=True)
        summary, fractions = _run_fit(work, fit, representation, output)
        drift_rows.append(summary)
        basin_rows.extend(fractions)
        print(json.dumps({"tissue": tissue, **summary}, sort_keys=True), flush=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(drift_rows, columns=["fit", "final_loss", "sym_median", "sym_max"]).to_csv(
        QC_DIR / f"{tissue}_drift_metrics.csv", index=False
    )
    pd.DataFrame(basin_rows, columns=["fit", "cell_type", "age_group", "n", "basin_fraction"]).to_csv(
        QC_DIR / f"{tissue}_basin_fraction.csv", index=False
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tissue", choices=TISSUES, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.seed != SEED:
        raise ValueError("A02 is frozen at seed 0")
    run_tissue(args.tissue)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
