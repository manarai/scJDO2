#!/usr/bin/env python3
"""X04: A02-equivalent young-reference scArches mapping seeds 1 and 2."""
from __future__ import annotations

import argparse
import json

import anndata as ad
import numpy as np
import scvi

from fit_tms_scvi import N_LAYERS, N_LATENT, _make_scvi, _snapshot_model, _train
from x04_common import AGING, MODELS, OUT, fit_drift, frozen_reference_audit, verify_reported

TISSUES = ("lung", "marrow")


def run_tissue(tissue: str, seed: int) -> dict:
    if seed not in (1, 2):
        raise ValueError("X04 shared mapping fits only seeds 1 and 2; A02 seed 0 is reused")
    verified = verify_reported("A01", [f"benchmarks/aging/data/{tissue}_processed.h5ad"])
    source = ad.read_h5ad(AGING / "data" / f"{tissue}_processed.h5ad")
    young_mask = source.obs["age_group"].astype(str).eq("young").to_numpy()
    old_mask = source.obs["age_group"].astype(str).eq("old").to_numpy()
    young, old = source[young_mask].copy(), source[old_mask].copy()
    scvi.settings.seed = int(seed)
    reference = _make_scvi(young)
    ref_epochs, ref_elbo, ref_wall = _train(reference)
    _snapshot_model(reference, MODELS / f"{tissue}_shared_s{seed}_reference.pt")
    scvi.settings.seed = int(seed)
    query = scvi.model.SCVI.load_query_data(
        old, reference, accelerator="cpu", device="cpu", unfrozen=False,
        freeze_expression=True, freeze_decoder_first_layer=True,
        freeze_batchnorm_encoder=True, freeze_batchnorm_decoder=False,
    )
    query_epochs, query_elbo, query_wall = _train(query, plan_kwargs={"weight_decay": 0.0})
    audit = frozen_reference_audit(reference, query)
    if audit["reference_encoder_weights_max_abs_change"] != 0.0 or audit["reference_encoder_batchnorm_buffers_max_abs_change"] != 0.0:
        raise RuntimeError(f"frozen-reference audit failed: {audit}")
    _snapshot_model(query, MODELS / f"{tissue}_shared_s{seed}_query.pt")
    shared = source.copy()
    latent = np.empty((shared.n_obs, N_LATENT), dtype=np.float32)
    latent[young_mask] = np.asarray(reference.get_latent_representation(young), dtype=np.float32)
    latent[old_mask] = np.asarray(query.get_latent_representation(old), dtype=np.float32)
    shared.obsm["X_scVI_shared"] = latent
    shared.uns["x04_scvi"] = {
        "space": "shared", "seed": int(seed), "n_latent": N_LATENT, "n_layers": N_LAYERS,
        "gene_likelihood": "nb", "batch_key": "donor_id", "mapping_path": "scvi.model.SCVI.load_query_data",
        "max_epochs": 400, "early_stopping_patience": 30, "device": "cpu", "verified_inputs": verified,
        "reference_young": {"epochs_run": ref_epochs, "best_validation_elbo": ref_elbo, "wall_seconds": ref_wall},
        "query_shared": {"epochs_run": query_epochs, "best_validation_elbo": query_elbo, "wall_seconds": query_wall},
        **audit,
    }
    output = OUT / f"{tissue}_shared_s{seed}.h5ad"
    drift = fit_drift(shared, representation="X_scVI_shared", fit_label=f"shared_s{seed}", output=output,
                      model_output=MODELS / f"{tissue}_shared_s{seed}_drift.pt")
    summary = {"tissue": tissue, "space": "shared", "seed": seed, "output": str(output), **audit, **drift}
    qc = OUT / "qc"; qc.mkdir(parents=True, exist_ok=True)
    (qc / f"{tissue}_shared_s{seed}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tissue", choices=(*TISSUES, "all"), required=True)
    parser.add_argument("--seed", choices=(1, 2), type=int, required=True)
    args = parser.parse_args()
    for tissue in TISSUES if args.tissue == "all" else (args.tissue,):
        run_tissue(tissue, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
