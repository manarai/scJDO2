#!/usr/bin/env python3
"""H03 §3 primary space: young SCVI reference and frozen old scArches query.

``--dry-run`` validates only the execution plumbing: two deterministic 2,000-cell
fits at two epochs.  It never writes H03 result artifacts and writes all of its
logs, audit trails, and model histories under the gitignored data/logs/dryrun/.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scvi
import torch
from lightning.pytorch.callbacks import Callback
from scipy import sparse

from hlca_preprocess import SOURCE, load_rows_csr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "aging_hlca"
QC = OUT / "qc"
DATA = OUT / "data"
HVG_CSV = QC / "hvg_seurat_v3.csv"
DRIFT_CELLS = OUT / "drift_subsample_cells.csv.gz"
METRICS = QC / "hlca_scvi_metrics.csv"
HISTORY_REFERENCE = QC / "scvi_history_reference.csv"
HISTORY_QUERY = QC / "scvi_history_query.csv"
LOG_DIR = DATA / "logs"
DRYRUN_DIR = LOG_DIR / "dryrun"
DRIFT_LATENT = DATA / "h03_primary_drift_latent.h5ad"
SEED = 0
MAX_EPOCHS = 400
PATIENCE = 30
N_LATENT = 15
N_LAYERS = 2
DRYRUN_CELLS = 2_000
DRYRUN_EPOCHS = 2


def max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if Path("/System").exists() else value * 1024


def training_kwargs(max_epochs: int) -> dict:
    return {
        "max_epochs": int(max_epochs),
        "accelerator": "cpu",
        "devices": 1,
        "early_stopping": True,
        "early_stopping_patience": PATIENCE,
        "check_val_every_n_epoch": 1,
        "enable_progress_bar": False,
        "enable_model_summary": False,
    }


def load_hvg_adata(age_kind: str) -> ad.AnnData:
    """Construct one age-space count matrix from raw H01 rows and frozen HVGs."""
    if age_kind not in {"young", "old"}:
        raise ValueError(age_kind)
    hvg = pd.read_csv(HVG_CSV)
    genes = hvg.loc[hvg["highly_variable"], "gene_id"].astype(str).tolist()
    if len(genes) != 2_000 or len(set(genes)) != 2_000:
        raise RuntimeError("H03 requires exactly 2,000 unique frozen HVGs")
    source = ad.read_h5ad(SOURCE, backed="r")
    try:
        obs = source.obs.copy()
        age = pd.to_numeric(obs["age_or_mean_of_age_range"], errors="coerce")
        finite = age.notna().to_numpy()
        mask = finite & ((age.to_numpy() < 45) if age_kind == "young" else (age.to_numpy() >= 45))
        rows = np.flatnonzero(mask)
        if rows.size == 0:
            raise RuntimeError(f"no {age_kind} cells")
        work_obs = obs.iloc[rows].copy()
        work_obs["age"] = age.iloc[rows].to_numpy(dtype=float)
        work_obs["study_or_dataset"] = work_obs["dataset"].astype(str).to_numpy()
        raw_var_names = source.raw.var_names.astype(str)
        positions = raw_var_names.get_indexer(genes)
        if (positions < 0).any():
            raise RuntimeError("frozen HVG list is not contained in the H01 raw matrix")
        source_order = np.sort(positions)
        source_to_output = np.searchsorted(source_order, positions)
        with h5py.File(SOURCE, "r") as handle:
            full = load_rows_csr(handle, rows, source.n_vars)
        hvg_counts_source_order = full[:, source_order].tocsr()
        del full
        counts = hvg_counts_source_order[:, source_to_output].tocsr()
        del hvg_counts_source_order
        var = source.raw.var.iloc[positions].copy()
        var.index = pd.Index(genes, dtype=str)
    finally:
        source.file.close()
    adata = ad.AnnData(X=counts, obs=work_obs, var=var)
    # Raw counts stay in the model layer; X is its alias to avoid a second sparse allocation.
    adata.layers["counts"] = adata.X
    return adata


def subsample_for_dry_run(adata: ad.AnnData, age_kind: str) -> ad.AnnData:
    if adata.n_obs < DRYRUN_CELLS:
        raise RuntimeError(f"dry run {age_kind} set has only {adata.n_obs} cells")
    rng = np.random.default_rng(SEED)
    index = np.sort(rng.choice(adata.n_obs, size=DRYRUN_CELLS, replace=False))
    return adata[index].copy()


def make_model(adata: ad.AnnData) -> scvi.model.SCVI:
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="study_or_dataset")
    return scvi.model.SCVI(adata, n_latent=N_LATENT, n_layers=N_LAYERS, gene_likelihood="nb")


class ELBOAuditTrail(Callback):
    """Non-authoritative per-hook metric stream for crash diagnosis only.

    Lightning presents train and validation metrics at distinct hook times.  The
    trail therefore records whichever scalar values exist; model.history saved
    after a successful fit is the only authoritative training history.
    """

    def __init__(self, output: Path, fit_name: str) -> None:
        super().__init__()
        self.output = output
        self.fit_name = fit_name
        self.exception_count = 0

    def _exception(self, hook: str, exc: Exception) -> None:
        self.exception_count += 1
        print(
            f"[{self.fit_name}] audit hook {hook} ignored {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )

    @staticmethod
    def _scalar(trainer, name: str) -> float:
        for metrics in (trainer.callback_metrics, trainer.logged_metrics, trainer.progress_bar_metrics):
            value = metrics.get(name)
            if value is None:
                continue
            if hasattr(value, "detach"):
                value = value.detach().cpu().item()
            value = float(value)
            if np.isfinite(value):
                return value
        return float("nan")

    def on_fit_start(self, trainer, pl_module) -> None:
        try:
            self.output.parent.mkdir(parents=True, exist_ok=True)
            with self.output.open("w", newline="") as handle:
                csv.writer(handle).writerow(["epoch", "hook", "elbo_train", "elbo_validation"])
                handle.flush()
                os.fsync(handle.fileno())
        except Exception as exc:  # The audit path must never stop a frozen fit.
            self._exception("on_fit_start", exc)

    def _append(self, trainer, hook: str) -> None:
        try:
            if trainer.sanity_checking:
                return
            epoch = int(trainer.current_epoch) + 1
            train = self._scalar(trainer, "elbo_train")
            validation = self._scalar(trainer, "elbo_validation")
            with self.output.open("a", newline="") as handle:
                csv.writer(handle).writerow([epoch, hook, repr(train), repr(validation)])
                handle.flush()
                os.fsync(handle.fileno())
            print(
                f"[{self.fit_name}] {hook} epoch={epoch} "
                f"elbo_train={train:.8g} elbo_validation={validation:.8g}",
                flush=True,
            )
        except Exception as exc:  # The audit path must never stop a frozen fit.
            self._exception(hook, exc)
            return

    def on_validation_end(self, trainer, pl_module) -> None:
        self._append(trainer, "on_validation_end")

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self._append(trainer, "on_train_epoch_end")


def history_frame(history: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if "elbo_train" not in history or "elbo_validation" not in history:
        raise RuntimeError("H03 requires both train and validation ELBO histories")
    frames: list[pd.DataFrame] = []
    for name, frame in history.items():
        if not isinstance(frame, pd.DataFrame) or frame.shape[1] != 1:
            raise RuntimeError(f"unexpected SCVI history table for {name!r}")
        values = frame.iloc[:, 0].rename(str(name))
        frames.append(values)
    result = pd.concat(frames, axis=1)
    result.index = pd.RangeIndex(start=1, stop=len(result) + 1, name="epoch")
    return result


def require_complete_history(history: pd.DataFrame, *, expected_epochs: int | None = None) -> None:
    required = ["elbo_train", "elbo_validation"]
    if any(name not in history for name in required):
        raise RuntimeError(f"model.history missing required columns: {required}")
    if expected_epochs is not None and len(history) != expected_epochs:
        raise RuntimeError(f"expected {expected_epochs} completed epochs, got {len(history)}")
    for name in required:
        values = history[name].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"model.history contains non-finite {name}")


def train(
    model: scvi.model.SCVI,
    *,
    fit_name: str,
    audit_path: Path,
    history_path: Path,
    max_epochs: int,
    plan_kwargs: dict | None = None,
) -> tuple[pd.DataFrame, float, int]:
    """Run one fit and save its authoritative, post-fit SCVI history."""
    kwargs = training_kwargs(max_epochs)
    if plan_kwargs is not None:
        kwargs["plan_kwargs"] = plan_kwargs
    audit = ELBOAuditTrail(audit_path, fit_name)
    kwargs["callbacks"] = [audit]
    started = time.monotonic()
    model.train(**kwargs)
    elapsed = time.monotonic() - started
    history = history_frame({str(name): value.copy() for name, value in model.history.items()})
    require_complete_history(history, expected_epochs=max_epochs if max_epochs == DRYRUN_EPOCHS else None)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(history_path, index_label="epoch")
    return history, elapsed, audit.exception_count


def audit_frozen_latent_encoder(reference: scvi.model.SCVI, query: scvi.model.SCVI) -> tuple[float, int, int]:
    """Assert every reference z_encoder parameter and BatchNorm buffer is unchanged."""
    reference_state = reference.module.state_dict()
    query_state = query.module.state_dict()
    parameter_names = {name for name, _ in reference.module.named_parameters()}
    buffer_names = {name for name, _ in reference.module.named_buffers()}
    selected = [name for name in reference_state if name.startswith("z_encoder")]
    if not selected:
        raise RuntimeError("could not find z_encoder state for required frozen audit")
    maxima: list[float] = []
    parameter_count = 0
    buffer_count = 0
    for name in selected:
        if name not in query_state:
            raise RuntimeError(f"query state lacks latent-encoder key {name!r}")
        ref = reference_state[name].detach().cpu()
        mapped = query_state[name].detach().cpu()
        if mapped.ndim != ref.ndim or any(mapped.shape[i] < ref.shape[i] for i in range(ref.ndim)):
            raise RuntimeError(f"incompatible query shape for latent-encoder key {name!r}")
        slices = tuple(slice(0, size) for size in ref.shape)
        delta = torch.max(torch.abs(mapped[slices].to(dtype=ref.dtype) - ref)).item()
        maxima.append(float(delta))
        if name in parameter_names:
            parameter_count += int(ref.numel())
        elif name in buffer_names:
            buffer_count += int(ref.numel())
        else:
            raise RuntimeError(f"audit key {name!r} is neither parameter nor buffer")
    max_change = float(max(maxima))
    if max_change != 0.0:
        raise RuntimeError(f"scArches frozen latent-encoder audit failed: max_abs_change={max_change}")
    return max_change, parameter_count, buffer_count


def drift_latents(reference: scvi.model.SCVI, young: ad.AnnData, query: scvi.model.SCVI, old: ad.AnnData) -> ad.AnnData:
    requested = pd.read_csv(DRIFT_CELLS, compression="gzip")
    wanted = pd.Index(requested["cell_id"].astype(str))
    pieces: list[ad.AnnData] = []
    vectors: list[np.ndarray] = []
    for model, data, label in ((reference, young, "young"), (query, old, "old")):
        mask = data.obs_names.astype(str).isin(wanted)
        subset = data[mask].copy()
        z = np.asarray(model.get_latent_representation(subset), dtype=np.float32)
        if z.shape != (subset.n_obs, N_LATENT):
            raise RuntimeError(f"unexpected {label} latent shape {z.shape}")
        pieces.append(subset)
        vectors.append(z)
    combined = ad.concat(pieces, join="outer", merge="same", index_unique=None)
    latent = np.vstack(vectors)
    if combined.obs_names.duplicated().any() or combined.n_obs != len(wanted):
        raise RuntimeError("drift latent cells do not form the frozen unique sample")
    order = combined.obs_names.get_indexer(wanted)
    if (order < 0).any():
        raise RuntimeError("one or more frozen drift cells are missing from primary mapping")
    combined = combined[order].copy()
    latent = latent[order]
    result = ad.AnnData(X=np.zeros((combined.n_obs, 0), dtype=np.float32), obs=combined.obs.copy())
    result.obsm["X_scVI_primary"] = latent
    result.uns["h03_primary"] = {"representation": "X_scVI_primary", "n_latent": N_LATENT, "seed": SEED}
    return result


def run(*, dry_run: bool) -> dict:
    if not HVG_CSV.exists() or not DRIFT_CELLS.exists():
        raise FileNotFoundError("run scripts/hlca_preprocess.py first")
    started = time.monotonic()
    if dry_run:
        run_dir = DRYRUN_DIR
        max_epochs = DRYRUN_EPOCHS
        ref_history_path = run_dir / "scvi_history_reference.csv"
        query_history_path = run_dir / "scvi_history_query.csv"
    else:
        run_dir = LOG_DIR
        max_epochs = MAX_EPOCHS
        ref_history_path = HISTORY_REFERENCE
        query_history_path = HISTORY_QUERY
    run_dir.mkdir(parents=True, exist_ok=True)
    scvi.settings.seed = SEED
    young = load_hvg_adata("young")
    if dry_run:
        young = subsample_for_dry_run(young, "young")
    reference = make_model(young)
    ref_history, ref_wall, ref_exceptions = train(
        reference,
        fit_name="reference",
        audit_path=run_dir / "reference_audit.csv",
        history_path=ref_history_path,
        max_epochs=max_epochs,
    )
    scvi.settings.seed = SEED
    old = load_hvg_adata("old")
    if dry_run:
        old = subsample_for_dry_run(old, "old")
    query = scvi.model.SCVI.load_query_data(
        old,
        reference,
        accelerator="cpu",
        device="auto",
        unfrozen=False,
        freeze_expression=True,
        freeze_decoder_first_layer=True,
        freeze_batchnorm_encoder=True,
        freeze_batchnorm_decoder=False,
    )
    query_history, query_wall, query_exceptions = train(
        query,
        fit_name="query",
        audit_path=run_dir / "query_audit.csv",
        history_path=query_history_path,
        max_epochs=max_epochs,
        plan_kwargs={"weight_decay": 0.0},
    )
    max_change, parameter_count, buffer_count = audit_frozen_latent_encoder(reference, query)
    output = {
        "dry_run": bool(dry_run),
        "reference_history": str(ref_history_path),
        "query_history": str(query_history_path),
        "reference_audit": str(run_dir / "reference_audit.csv"),
        "query_audit": str(run_dir / "query_audit.csv"),
        "reference_epochs": int(len(ref_history)),
        "query_epochs": int(len(query_history)),
        "logging_hook_exceptions": int(ref_exceptions + query_exceptions),
        "frozen_latent_encoder_max_abs_change": max_change,
        "frozen_latent_encoder_parameter_count": parameter_count,
        "frozen_latent_encoder_buffer_count": buffer_count,
        "reference_wall_seconds": float(ref_wall),
        "query_wall_seconds": float(query_wall),
        "peak_rss_bytes": max_rss_bytes(),
        "wall_seconds": time.monotonic() - started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    if dry_run:
        (run_dir / "dryrun_summary.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        return output
    latent = drift_latents(reference, young, query, old)
    DATA.mkdir(parents=True, exist_ok=True)
    latent.write_h5ad(DRIFT_LATENT)
    records = []
    for fit, data, history, wall in (("young_reference", young, ref_history, ref_wall), ("old_query", old, query_history, query_wall)):
        validation = history["elbo_validation"].to_numpy(dtype=float)
        records.append(
            {
                "fit": fit,
                "n_cells": int(data.n_obs),
                "n_donors": int(data.obs["donor_id"].astype(str).nunique()),
                "epochs_run": int(len(validation)),
                "best_validation_elbo": float(np.min(validation)),
                "wall_seconds": float(wall),
                "n_latent": N_LATENT,
                "n_layers": N_LAYERS,
                "gene_likelihood": "nb",
                "accelerator": "cpu",
                "frozen_latent_encoder_max_abs_change": max_change if fit == "old_query" else 0.0,
                "frozen_latent_encoder_parameter_count": parameter_count if fit == "old_query" else 0,
                "frozen_latent_encoder_buffer_count": buffer_count if fit == "old_query" else 0,
                "logging_hook_exceptions": int(ref_exceptions + query_exceptions),
            }
        )
    QC.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(METRICS, index=False)
    output.update({"metrics": str(METRICS), "drift_latent": str(DRIFT_LATENT), "n_drift_cells": int(latent.n_obs)})
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.seed != SEED:
        raise ValueError("H03 primary mapping is frozen at seed 0")
    print(json.dumps(run(dry_run=bool(args.dry_run)), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
