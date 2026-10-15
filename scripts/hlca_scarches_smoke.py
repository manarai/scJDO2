#!/usr/bin/env python3
"""Run the H01 5,000-cell HLCA scArches plumbing smoke test.

The script maps cells from one HLCA core study back onto the published HLCA
reference with the documented SCANVI surgery API. It intentionally discards the
query embedding: this is an environment and frozen-weight audit, not an analysis.
"""
from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import scarches as sca
import scvi
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks" / "aging_hlca" / "data" / "hlca_core_raw_counts.h5ad"
MODEL = ROOT / "benchmarks" / "aging_hlca" / "data" / "HLCA_reference_model"
OUT = ROOT / "benchmarks" / "aging_hlca" / "inventory" / "scarches_smoke.json"
N_CELLS = 5_000
EPOCHS = 20
SEED = 0


def select_study(obs):
    for candidate in ("dataset", "study", "dataset_id", "study_id"):
        if candidate in obs.columns:
            usable = obs[candidate].dropna().astype(str)
            if not usable.empty:
                return candidate, str(usable.value_counts().index[0])
    raise RuntimeError("HLCA core has no usable study/dataset field for smoke query selection")


def max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def frozen_snapshot(model) -> dict[str, torch.Tensor]:
    """Capture only parameters that scArches marked frozen after surgery setup."""
    snapshot = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.module.named_parameters()
        if not parameter.requires_grad
    }
    if not snapshot:
        raise RuntimeError("scArches surgery did not expose any frozen reference parameters to audit")
    return snapshot


def max_frozen_change(model, before: dict[str, torch.Tensor]) -> tuple[float, int]:
    after = dict(model.module.named_parameters())
    maxima: list[float] = []
    checked = 0
    for name, initial in before.items():
        if name not in after:
            raise RuntimeError(f"frozen parameter vanished after surgery training: {name}")
        current = after[name].detach().cpu()
        if current.shape != initial.shape:
            raise RuntimeError(f"frozen parameter shape changed: {name}: {tuple(initial.shape)} -> {tuple(current.shape)}")
        maxima.append(float(torch.max(torch.abs(current - initial)).item()))
        checked += int(initial.numel())
    return float(max(maxima)), checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-cells", type=int, default=N_CELLS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    if (args.seed, args.n_cells, args.epochs) != (SEED, N_CELLS, EPOCHS):
        raise ValueError("H01 fixes seed=0, n_cells=5000, and epochs=20")
    if not DATA.exists() or not (MODEL / "model.pt").exists():
        raise FileNotFoundError("run scripts/fetch_hlca.py before the smoke test")

    scvi.settings.seed = SEED
    scvi.settings.dl_num_workers = 0
    torch.set_num_threads(1)
    source = ad.read_h5ad(DATA, backed="r")
    study_field, study = select_study(source.obs)
    candidates = np.flatnonzero(source.obs[study_field].astype(str).to_numpy() == study)
    if len(candidates) < N_CELLS:
        raise RuntimeError(f"largest HLCA study {study!r} has only {len(candidates)} cells; requires {N_CELLS}")
    rng = np.random.default_rng(SEED)
    selected = np.sort(rng.choice(candidates, size=N_CELLS, replace=False))
    query = source[selected].to_memory()
    source.file.close()
    query.obs["dataset"] = "H01_smoke_query"
    query.obs["scanvi_label"] = "unlabeled"

    started = time.perf_counter()
    prepared = sca.models.SCANVI.prepare_query_anndata(query, str(MODEL), inplace=False)
    surgery = sca.models.SCANVI.load_query_data(
        prepared,
        str(MODEL),
        accelerator="cpu",
        device="auto",
        unfrozen=False,
        freeze_dropout=True,
        freeze_expression=True,
        freeze_decoder_first_layer=True,
        freeze_batchnorm_encoder=True,
        freeze_batchnorm_decoder=False,
        freeze_classifier=True,
    )
    frozen_before = frozen_snapshot(surgery)
    surgery.train(
        max_epochs=EPOCHS,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        plan_kwargs={"weight_decay": 0.0},
    )
    wall_seconds = float(time.perf_counter() - started)
    max_change, parameters_checked = max_frozen_change(surgery, frozen_before)
    metrics = {
        "task": "hlca/H01",
        "success": True,
        "seed": SEED,
        "n_cells": N_CELLS,
        "query_study_field": study_field,
        "query_study": study,
        "epochs": EPOCHS,
        "api_path": "sca.models.SCANVI.prepare_query_anndata + sca.models.SCANVI.load_query_data",
        "wall_seconds": wall_seconds,
        "peak_rss_bytes": max_rss_bytes(),
        "seconds_per_10000_cells": wall_seconds * 10_000 / N_CELLS,
        "frozen_reference_weights_max_abs_change": max_change,
        "frozen_reference_parameters_checked": parameters_checked,
        "scarches_version": str(sca.__version__),
        "scvi_tools_version": str(scvi.__version__),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
