#!/usr/bin/env python3
"""X04: locked joint young+old scVI spaces and energy-drift fits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scvi

from fit_tms_scvi import N_LAYERS, N_LATENT, _make_scvi, _snapshot_model, _train
from x04_common import AGING, MODELS, OUT, fit_drift, verify_reported

TISSUES = ("lung", "marrow")


def run_tissue(tissue: str, seed: int) -> dict:
    verified = verify_reported("A01", [f"benchmarks/aging/data/{tissue}_processed.h5ad"])
    source = ad.read_h5ad(AGING / "data" / f"{tissue}_processed.h5ad")
    if "counts" not in source.layers or set(source.obs["age_group"].astype(str)) != {"young", "old"}:
        raise RuntimeError(f"{tissue} does not provide both age groups and counts")
    # Age is intentionally not a covariate: setup supplies only donor batch labels.
    work = source.copy()
    scvi.settings.seed = int(seed)
    model = _make_scvi(work)
    epochs, elbo, scvi_wall = _train(model)
    _snapshot_model(model, MODELS / f"{tissue}_joint_s{seed}.pt")
    latent = np.asarray(model.get_latent_representation(work), dtype=np.float32)
    work.obsm["X_scVI_joint"] = latent
    work.uns["x04_scvi"] = {
        "space": "joint", "seed": int(seed), "n_latent": N_LATENT, "n_layers": N_LAYERS,
        "gene_likelihood": "nb", "batch_key": "donor_id", "age_covariate": False,
        "max_epochs": 400, "early_stopping_patience": 30, "epochs_run": epochs,
        "best_validation_elbo": elbo, "scvi_wall_seconds": scvi_wall, "device": "cpu",
        "verified_inputs": verified,
    }
    output = OUT / f"{tissue}_joint_s{seed}.h5ad"
    drift = fit_drift(work, representation="X_scVI_joint", fit_label=f"joint_s{seed}", output=output,
                      model_output=MODELS / f"{tissue}_joint_s{seed}_drift.pt")
    summary = {"tissue": tissue, "space": "joint", "seed": seed, "output": str(output),
               "epochs_run": epochs, "best_validation_elbo": elbo, **drift}
    qc = OUT / "qc"; qc.mkdir(parents=True, exist_ok=True)
    (qc / f"{tissue}_joint_s{seed}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tissue", choices=(*TISSUES, "all"), required=True)
    parser.add_argument("--seed", choices=(0, 1, 2), type=int, required=True)
    args = parser.parse_args()
    for tissue in TISSUES if args.tissue == "all" else (args.tissue,):
        run_tissue(tissue, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
