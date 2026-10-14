#!/usr/bin/env python3
"""A02 §3: train young references and map old TMS cells with scArches.

The task contract fixes CPU training, seed 0, 400 maximum epochs, a 15-D
negative-binomial SCVI model, and the scArches query-mapping procedure.  The
script stores only summaries in git; H5AD/model artifacts stay local.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
import scvi


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "benchmarks" / "aging" / "data"
OUT_DIR = ROOT / "benchmarks" / "aging"
QC_DIR = OUT_DIR / "qc"
MODEL_DIR = OUT_DIR / "models"
TISSUES = ("lung", "marrow")
MAX_EPOCHS = 400
EARLY_STOPPING_PATIENCE = 30
N_LATENT = 15
N_LAYERS = 2


def _train(model: scvi.model.SCVI, *, plan_kwargs: dict | None = None) -> tuple[int, float, float]:
    """Train a model under the frozen CPU/early-stopping configuration."""
    started = time.monotonic()
    kwargs: dict = {
        "max_epochs": MAX_EPOCHS,
        "accelerator": "cpu",
        "devices": 1,
        "early_stopping": True,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "check_val_every_n_epoch": 1,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }
    if plan_kwargs is not None:
        kwargs["plan_kwargs"] = plan_kwargs
    model.train(**kwargs)
    wall_seconds = time.monotonic() - started

    history = model.history
    if not history:
        raise RuntimeError("SCVI training returned no history")
    history_key = "elbo_validation" if "elbo_validation" in history else "elbo_train"
    values = np.asarray(history[history_key].iloc[:, 0], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise RuntimeError(f"SCVI history {history_key!r} has no finite values")
    return int(values.size), float(np.min(values)), float(wall_seconds)


def _snapshot_model(model: scvi.model.SCVI, path: Path) -> None:
    """Persist a local model state as an ignored .pt artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.module.state_dict().items()},
            "init_params": model.init_params_,
            "registry": model.registry_,
        },
        path,
    )


def _reference_weights_max_abs_change(
    reference_model: scvi.model.SCVI,
    query_model: scvi.model.SCVI,
) -> tuple[float, int]:
    """Audit frozen reference encoder/decoder parameters after query training.

    `load_query_data` appends new categorical-batch rows to some tensors.  For
    every reference encoder/decoder parameter, compare only the leading slice
    occupied by the original reference tensor.  Thus new rows are excluded but
    every original weight that scArches freezes is checked exactly. The
    scArches procedure intentionally leaves the library-size encoder and
    decoder BatchNorm affine parameters trainable; they are not reference
    expression weights and are excluded from this audit.
    """
    reference = dict(reference_model.module.named_parameters())
    query = dict(query_model.module.named_parameters())
    maxima: list[float] = []
    checked = 0
    for name, ref in reference.items():
        if "encoder" not in name and "decoder" not in name:
            continue
        # scvi-tools _set_params_online_update explicitly leaves l_encoder and
        # decoder BatchNorm affine terms trainable. These are not frozen
        # reference expression weights under the documented scArches setup.
        if "l_encoder" in name:
            continue
        if "decoder" in name and ".fc_layers." in name and (name.endswith(".1.weight") or name.endswith(".1.bias")):
            continue
        if name not in query:
            raise RuntimeError(f"query model is missing reference parameter {name!r}")
        q = query[name]
        if q.ndim != ref.ndim or any(q.shape[d] < ref.shape[d] for d in range(ref.ndim)):
            raise RuntimeError(
                f"query parameter {name!r} has shape {tuple(q.shape)}, "
                f"incompatible with reference shape {tuple(ref.shape)}"
            )
        reference_slice = tuple(slice(0, size) for size in ref.shape)
        delta = torch.max(torch.abs(q.detach().cpu()[reference_slice] - ref.detach().cpu()))
        maxima.append(float(delta.item()))
        checked += int(ref.numel())
    if not maxima:
        raise RuntimeError("no reference encoder/decoder parameters were audited")
    return float(max(maxima)), checked


def _make_scvi(adata: ad.AnnData) -> scvi.model.SCVI:
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="donor_id")
    return scvi.model.SCVI(
        adata,
        n_latent=N_LATENT,
        n_layers=N_LAYERS,
        gene_likelihood="nb",
    )


def run_tissue(tissue: str, seed: int) -> None:
    input_path = DATA_DIR / f"{tissue}_processed.h5ad"
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    source = ad.read_h5ad(input_path)
    if "counts" not in source.layers:
        raise RuntimeError(f"{input_path} lacks the required counts layer")
    if "age_group" not in source.obs or "donor_id" not in source.obs:
        raise RuntimeError(f"{input_path} lacks age_group or donor_id")

    young_mask = source.obs["age_group"].astype(str).eq("young").to_numpy()
    old_mask = source.obs["age_group"].astype(str).eq("old").to_numpy()
    if not young_mask.any() or not old_mask.any():
        raise RuntimeError(f"{tissue} lacks one of the required age groups")

    # Use independent copies because setup_anndata stores model-specific state.
    young = source[young_mask].copy()
    old_for_query = source[old_mask].copy()
    old_standalone = source[old_mask].copy()

    scvi.settings.seed = seed
    reference = _make_scvi(young)
    ref_epochs, ref_elbo, ref_wall = _train(reference)
    _snapshot_model(reference, MODEL_DIR / f"{tissue}_ref.pt")

    # scArches surgery copies the reference, extends only query categories, and
    # installs online-update hooks that freeze the reference expression weights.
    scvi.settings.seed = seed
    query = scvi.model.SCVI.load_query_data(
        old_for_query,
        reference,
        accelerator="cpu",
        device="auto",
        unfrozen=False,
        freeze_expression=True,
        freeze_decoder_first_layer=True,
        freeze_batchnorm_encoder=True,
        freeze_batchnorm_decoder=False,
    )
    query_epochs, query_elbo, query_wall = _train(query, plan_kwargs={"weight_decay": 0.0})
    ref_change, ref_parameters_checked = _reference_weights_max_abs_change(reference, query)
    _snapshot_model(query, MODEL_DIR / f"{tissue}_query_shared.pt")

    scvi.settings.seed = seed
    standalone_old = _make_scvi(old_standalone)
    old_epochs, old_elbo, old_wall = _train(standalone_old)
    _snapshot_model(standalone_old, MODEL_DIR / f"{tissue}_old.pt")

    # The shared AnnData retains all cells and all three task-specified views.
    shared = source.copy()
    x_young = np.full((shared.n_obs, N_LATENT), np.nan, dtype=np.float32)
    x_old = np.full((shared.n_obs, N_LATENT), np.nan, dtype=np.float32)
    x_shared = np.empty((shared.n_obs, N_LATENT), dtype=np.float32)
    ref_z = np.asarray(reference.get_latent_representation(young), dtype=np.float32)
    query_z = np.asarray(query.get_latent_representation(old_for_query), dtype=np.float32)
    old_z = np.asarray(standalone_old.get_latent_representation(old_standalone), dtype=np.float32)
    x_young[young_mask] = ref_z
    x_old[old_mask] = old_z
    x_shared[young_mask] = ref_z
    x_shared[old_mask] = query_z
    shared.obsm["X_scVI_young"] = x_young
    shared.obsm["X_scVI_shared"] = x_shared
    shared.obsm["X_scVI_old"] = x_old
    shared.uns["a02_scvi"] = {
        "seed": int(seed),
        "accelerator": "cpu",
        "max_epochs": int(MAX_EPOCHS),
        "early_stopping_patience": int(EARLY_STOPPING_PATIENCE),
        "n_latent": int(N_LATENT),
        "n_layers": int(N_LAYERS),
        "gene_likelihood": "nb",
        "mapping_path": "scvi.model.SCVI.load_query_data",
        "reference_weights_max_abs_change": float(ref_change),
        "reference_parameters_checked": int(ref_parameters_checked),
        "wall_seconds": {
            "reference_young": float(ref_wall),
            "query_shared": float(query_wall),
            "standalone_old": float(old_wall),
        },
        "scvi_version": str(scvi.__version__),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shared.write_h5ad(OUT_DIR / f"{tissue}_shared.h5ad")

    rows = [
        {"fit": "reference_young", "epochs_run": ref_epochs, "best_val_elbo": ref_elbo, "mapping_path": "young_reference"},
        {"fit": "query_shared", "epochs_run": query_epochs, "best_val_elbo": query_elbo, "mapping_path": "scvi.model.SCVI.load_query_data"},
        {"fit": "standalone_old", "epochs_run": old_epochs, "best_val_elbo": old_elbo, "mapping_path": "standalone_old"},
    ]
    QC_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["fit", "epochs_run", "best_val_elbo", "mapping_path"]).to_csv(
        QC_DIR / f"{tissue}_scvi_metrics.csv", index=False
    )
    print(json.dumps({
        "tissue": tissue,
        "output": str(OUT_DIR / f"{tissue}_shared.h5ad"),
        "reference_weights_max_abs_change": ref_change,
        "reference_parameters_checked": ref_parameters_checked,
        "fits": rows,
    }, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tissue", choices=TISSUES, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.seed != 0:
        raise ValueError("A02 is frozen at seed 0")
    run_tissue(args.tissue, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
