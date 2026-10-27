#!/usr/bin/env python3
"""S11 read-only real-data embeddings and independent velocity sources.

Every command reads the declared H5AD inputs in place and writes only under
``benchmarks/real_flow``.  Each successful prepare/source action emits a JSON
checkpoint.  No truth keys or orientation functions are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import scvelo as scv
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
DATA = Path.home() / "Documents/scJDO2/benchmarks/data"
SCNT_RAW = DATA / "dynamo_hematopoiesis/data/hematopoiesis_raw.h5ad"
SCNT_PROCESSED = DATA / "dynamo_hematopoiesis/data/hematopoiesis.h5ad"
LARRY_RAW = DATA / "larry/larry_inVitro_raw.h5ad"
OUT = ROOT / "benchmarks/real_flow"
ARRAYS = OUT / "arrays"
EMBED = ARRAYS / "embeddings"
VELOCITIES = ARRAYS / "velocities"
SCVELO_FITS = ARRAYS / "scvelo_fits"
CHECKPOINTS = OUT / "checkpoints"
MAP = OUT / "scnt_hema_gene_map.csv"

EXPECTED = {
    SCNT_RAW: "2e0e274452fd730a30552b18fea26d9dcf665fff33ffc9de2007ec5c9eedb68d",
    LARRY_RAW: "3e13ff16d0f3d1338932860abe15b1609749bc7f70d4b2596058ae3d7763a80e",
}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_input(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    observed = sha256(path)
    expected = EXPECTED.get(path)
    if expected is not None and observed != expected:
        raise RuntimeError(f"S08 checksum mismatch for {path}: {observed} != {expected}")
    return observed


def dense(x: Any, dtype: np.dtype = np.float64) -> np.ndarray:
    if sp.issparse(x):
        return x.toarray().astype(dtype, copy=False)
    return np.asarray(x, dtype=dtype)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def checkpoint(unit: str, started: float, payload: dict[str, Any]) -> None:
    payload = {
        "task": "orient/S11",
        "unit": unit,
        "started_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "finished_utc": utc(),
        "wall_seconds": round(time.time() - started, 3),
        **payload,
    }
    write_json(CHECKPOINTS / f"{unit}.json", payload)


def map_frame() -> pd.DataFrame:
    if not MAP.exists():
        raise FileNotFoundError(f"required committed S11 map missing: {MAP}")
    frame = pd.read_csv(MAP)
    need = {"symbol", "ensembl", "n_candidates", "value_ratio_cv"}
    if set(frame.columns) != need:
        raise RuntimeError(f"unexpected map columns: {list(frame.columns)}")
    if len(frame) != 1956 or not (frame["n_candidates"].to_numpy() == 1).all():
        raise RuntimeError("S11 map is not the required 1956/1956 unique map")
    if frame["symbol"].duplicated().any() or frame["ensembl"].duplicated().any():
        raise RuntimeError("S11 map identifiers are not unique")
    return frame


def _scnt_indices(raw: ad.AnnData, processed: ad.AnnData) -> tuple[pd.DataFrame, np.ndarray]:
    frame = map_frame()
    if not np.array_equal(frame["symbol"].to_numpy(str), processed.var_names.to_numpy(str)):
        raise RuntimeError("committed map is not in processed var_names order")
    positions = raw.var_names.get_indexer(frame["ensembl"].to_numpy(str))
    if (positions < 0).any():
        raise RuntimeError("committed map Ensembl ID absent from raw file")
    if not raw.obs_names.equals(processed.obs_names):
        raise RuntimeError("raw/processed cell order differs")
    return frame, positions.astype(int)


def _fit_embedding(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    library = counts.sum(axis=1, dtype=np.float64)
    if (library <= 0).any():
        raise RuntimeError("non-positive library encountered in common embedding")
    target = float(np.median(library))
    transformed = np.log1p(counts * (target / library)[:, None])
    pca = PCA(n_components=30, random_state=0)
    z = pca.fit_transform(transformed)
    return z.astype(np.float32), pca.components_.T.astype(np.float64), pca.mean_.astype(np.float64), target


def _chain_rule_total(
    counts: np.ndarray,
    gene_velocity: np.ndarray,
    linear_map: np.ndarray,
    target: float,
) -> np.ndarray:
    """Observed-count chain rule matching S02's helper, for S11 total RNA."""
    counts = np.asarray(counts, dtype=np.float64)
    gene_velocity = np.asarray(gene_velocity, dtype=np.float64)
    if counts.shape != gene_velocity.shape:
        raise ValueError(f"count/velocity mismatch: {counts.shape} vs {gene_velocity.shape}")
    library = counts.sum(axis=1)
    denom = np.maximum(library[:, None], 1e-12)
    scaled = counts * target / denom
    d_library = gene_velocity.sum(axis=1)
    d_scaled = target * (gene_velocity / denom - counts * d_library[:, None] / np.maximum(denom**2, 1e-12))
    d_log = d_scaled / (1.0 + scaled)
    return (d_log @ linear_map).astype(np.float32)


def scnt_embedding_path() -> Path:
    return EMBED / "scnt_hema_x_pca30.npz"


def larry_embedding_path() -> Path:
    return EMBED / "larry_x_pca30.npz"


def prepare_scnt() -> None:
    started = time.time()
    raw_hash, processed_hash = check_input(SCNT_RAW), sha256(SCNT_PROCESSED)
    raw, processed = ad.read_h5ad(SCNT_RAW), ad.read_h5ad(SCNT_PROCESSED)
    frame, raw_idx = _scnt_indices(raw, processed)
    counts = dense(raw.layers["total"][:, raw_idx], np.float64)
    z, linear_map, mean, target = _fit_embedding(counts)
    EMBED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        scnt_embedding_path(),
        z=z, linear_map=linear_map, pca_mean=mean, normalization_target=np.float64(target),
        raw_indices=raw_idx, cell_id=raw.obs_names.to_numpy(str), cell_type=raw.obs["cell_type"].to_numpy(str),
        pulse_hours=raw.obs["time"].to_numpy(np.float32),
    )
    checkpoint("prepare_scnt_hema", started, {
        "input_sha256": {"raw": raw_hash, "processed": processed_hash},
        "map_sha256": sha256(MAP), "n_cells": int(raw.n_obs), "n_genes": int(len(frame)),
        "normalization_target": target, "embedding": str(scnt_embedding_path().relative_to(ROOT)),
    })


def _transform_larry_rows(x: Any, library: np.ndarray, target: float) -> np.ndarray:
    arr = dense(x, np.float64)
    return np.log1p(arr * (target / np.maximum(library, 1e-12))[:, None])


def prepare_larry() -> None:
    started = time.time()
    input_hash = check_input(LARRY_RAW)
    data = ad.read_h5ad(LARRY_RAW)
    days = data.obs["Time point"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(0)
    sample_parts: list[np.ndarray] = []
    for day in (2.0, 4.0, 6.0):
        candidates = np.flatnonzero(days == day)
        if len(candidates) < 4000:
            raise RuntimeError(f"LARRY day {day:g} has only {len(candidates)} cells")
        sample_parts.append(rng.choice(candidates, size=4000, replace=False))
    sample_idx = np.concatenate(sample_parts).astype(int)
    # `seurat_v3` is task-specified and operates on the in-memory count matrix only.
    sc.pp.highly_variable_genes(data, flavor="seurat_v3", n_top_genes=2000, subset=False)
    hvg = data.var["highly_variable"].to_numpy(dtype=bool)
    if int(hvg.sum()) != 2000:
        raise RuntimeError(f"expected 2,000 Seurat-v3 HVGs, found {int(hvg.sum())}")
    library = np.asarray(data.X.sum(axis=1)).ravel().astype(np.float64)
    if (library <= 0).any():
        raise RuntimeError("non-positive LARRY library encountered")
    target = float(np.median(library))
    x_hvg = data.X[:, hvg]
    sample_x = _transform_larry_rows(x_hvg[sample_idx], library[sample_idx], target)
    pca = PCA(n_components=30, random_state=0)
    pca.fit(sample_x)
    z_all = np.empty((data.n_obs, 30), dtype=np.float32)
    chunk = 4096
    for lo in range(0, data.n_obs, chunk):
        hi = min(lo + chunk, data.n_obs)
        z_all[lo:hi] = pca.transform(_transform_larry_rows(x_hvg[lo:hi], library[lo:hi], target)).astype(np.float32)
    EMBED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        larry_embedding_path(), z_all=z_all, sample_index=sample_idx,
        day=days.astype(np.float32), cell_id=data.obs_names.to_numpy(str),
        cell_type=data.obs["Cell type annotation"].to_numpy(str),
        hvg_index=np.flatnonzero(hvg).astype(np.int32), linear_map=pca.components_.T.astype(np.float64),
        pca_mean=pca.mean_.astype(np.float64), normalization_target=np.float64(target),
    )
    checkpoint("prepare_larry", started, {
        "input_sha256": input_hash, "n_cells": int(data.n_obs), "n_hvg": int(hvg.sum()),
        "sample_per_time": 4000, "sample_seed": 0, "normalization_target": target,
        "embedding": str(larry_embedding_path().relative_to(ROOT)),
    })


def load_scnt_embedding() -> dict[str, np.ndarray]:
    path = scnt_embedding_path()
    if not path.exists():
        raise RuntimeError("run --dataset scnt_hema --action prepare first")
    return {key: value for key, value in np.load(path, allow_pickle=False).items()}


def load_larry_embedding() -> dict[str, np.ndarray]:
    path = larry_embedding_path()
    if not path.exists():
        raise RuntimeError("run --dataset larry --action prepare first")
    return {key: value for key, value in np.load(path, allow_pickle=False).items()}


def save_velocity(dataset: str, source: str, z: np.ndarray, velocity: np.ndarray, index: np.ndarray, cell_id: np.ndarray, cell_type: np.ndarray, details: dict[str, Any], started: float) -> None:
    if z.shape != velocity.shape or z.shape[0] != len(index):
        raise RuntimeError(f"invalid velocity output shapes z={z.shape}, v={velocity.shape}, index={len(index)}")
    if not np.isfinite(z).all() or not np.isfinite(velocity).all():
        raise RuntimeError("non-finite embedding or velocity")
    VELOCITIES.mkdir(parents=True, exist_ok=True)
    path = VELOCITIES / f"{dataset}_{source}.npz"
    np.savez_compressed(path, z=z.astype(np.float32), velocity=velocity.astype(np.float32),
                        cell_index=index.astype(np.int64), cell_id=cell_id.astype(str), cell_type=cell_type.astype(str))
    checkpoint(f"velocity_{dataset}_{source}", started, {
        "dataset": dataset, "source": source, "n_cells": int(z.shape[0]), "dimension": int(z.shape[1]),
        "velocity": str(path.relative_to(ROOT)), "details": details,
    })


def _scnt_common() -> tuple[ad.AnnData, ad.AnnData, np.ndarray, dict[str, np.ndarray], np.ndarray]:
    raw, processed = ad.read_h5ad(SCNT_RAW), ad.read_h5ad(SCNT_PROCESSED)
    _frame, raw_idx = _scnt_indices(raw, processed)
    emb = load_scnt_embedding()
    counts = dense(raw.layers["total"][:, raw_idx], np.float64)
    return raw, processed, raw_idx, emb, counts


def velocity_scnt_labelling() -> None:
    started = time.time()
    raw, _processed, raw_idx, emb, counts = _scnt_common()
    sys.path.insert(0, str(ROOT / "scripts"))
    from orient_end_to_end import labelled_velocity  # frozen S02 helper
    # In-memory AnnData prevents any alteration of the source H5AD and supplies S02's pulse key.
    total = dense(raw.layers["total"][:, raw_idx], np.float32)
    work = ad.AnnData(X=total, obs=pd.DataFrame({"pulse_hours": raw.obs["time"].to_numpy(np.float32)}, index=raw.obs_names.copy()))
    work.layers["total"] = total
    work.layers["new"] = dense(raw.layers["new"][:, raw_idx], np.float32)
    gene_v, details, _model = labelled_velocity(work, emb["z"])
    latent_v = _chain_rule_total(counts, gene_v, emb["linear_map"], float(emb["normalization_target"]))
    save_velocity("scnt_hema", "labelling", emb["z"], latent_v, np.arange(raw.n_obs), raw.obs_names.to_numpy(str), raw.obs["cell_type"].to_numpy(str), details, started)


def scvelo_fit_path() -> Path:
    return SCVELO_FITS / "scnt_hema_scvelo_fit.npz"


def fit_scnt_scvelo() -> None:
    """The one authorized exact S03 refit; persist before any chain-rule mapping."""
    started = time.time()
    raw, _processed, raw_idx, emb, _counts = _scnt_common()
    work = raw.copy()
    work.X = work.layers["spliced"].copy()
    work.obsm["X_pca"] = np.asarray(emb["z"], dtype=np.float32)
    scv.pp.filter_genes(work, min_shared_counts=20)
    if not work.obs_names.equals(raw.obs_names):
        raise RuntimeError("scVelo filtering altered cell order")
    scv.pp.normalize_per_cell(work)
    sc.pp.log1p(work)
    sc.pp.neighbors(work, use_rep="X_pca", n_neighbors=30, random_state=0)
    before = work.obsp["connectivities"].copy()
    scv.pp.moments(work, n_neighbors=None, n_pcs=None, use_rep=None)
    if (work.obsp["connectivities"] != before).nnz:
        raise RuntimeError("scVelo moments altered fixed X_pca30 neighbour graph")
    scv.tl.recover_dynamics(work)
    scv.tl.velocity(work, mode="dynamical")
    scv.tl.velocity_graph(work)
    scv.tl.velocity_embedding(work, basis="pca")
    if "velocity_genes" not in work.var:
        raise RuntimeError("scVelo did not provide required var['velocity_genes']")
    retained = {str(name): i for i, name in enumerate(work.var_names)}
    selected_names = raw.var_names[raw_idx].to_numpy(str)
    selected_velocity = np.full((raw.n_obs, len(raw_idx)), np.nan, dtype=np.float32)
    selected_velocity_genes = np.zeros(len(raw_idx), dtype=bool)
    present = [(j, retained[name]) for j, name in enumerate(selected_names) if name in retained]
    if present:
        j, k = zip(*present)
        selected_velocity[:, np.asarray(j)] = dense(work.layers["velocity"][:, np.asarray(k)], np.float32)
        selected_velocity_genes[np.asarray(j)] = work.var["velocity_genes"].to_numpy(dtype=bool)[np.asarray(k)]
    SCVELO_FITS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(scvelo_fit_path(), velocity=selected_velocity, velocity_genes=selected_velocity_genes,
                        raw_index=raw_idx.astype(np.int32), cell_id=raw.obs_names.to_numpy(str),
                        map_sha256=np.asarray(sha256(MAP)), embedding_sha256=np.asarray(sha256(scnt_embedding_path())))
    checkpoint("fit_scnt_hema_scvelo", started, {
        "dataset": "scnt_hema", "source": "scvelo", "fit_only": True,
        "fit_artifact": str(scvelo_fit_path().relative_to(ROOT)), "n_retained_genes": int(work.n_vars),
        "n_pca_genes_present": int(len(present)), "n_velocity_genes": int(selected_velocity_genes.sum()),
        "settings": "S03 fixed neighbours=30, moments reuse, recover_dynamics, velocity(dynamical)",
    })


def velocity_scnt_scvelo() -> None:
    """Apply the Revision-3 finite-gene rule to the persisted exact fit; never refit."""
    started = time.time()
    raw, _processed, raw_idx, emb, counts = _scnt_common()
    path = scvelo_fit_path()
    if not path.exists():
        raise RuntimeError("missing persisted scVelo fit; run the explicitly authorized --action fit_scvelo unit")
    fit = {key: value for key, value in np.load(path, allow_pickle=False).items()}
    if not np.array_equal(fit["raw_index"].astype(int), raw_idx) or not np.array_equal(fit["cell_id"].astype(str), raw.obs_names.to_numpy(str)):
        raise RuntimeError("persisted scVelo fit does not match current mapped raw input")
    if str(fit["map_sha256"]) != sha256(MAP) or str(fit["embedding_sha256"]) != sha256(scnt_embedding_path()):
        raise RuntimeError("persisted scVelo fit map or embedding hash mismatch")
    raw_velocity = np.asarray(fit["velocity"], dtype=np.float64)
    declared = np.asarray(fit["velocity_genes"], dtype=bool)
    usable = declared & np.isfinite(raw_velocity).all(axis=0)
    gene_v = np.zeros_like(raw_velocity)
    gene_v[:, usable] = raw_velocity[:, usable]
    latent_v = _chain_rule_total(counts, gene_v, emb["linear_map"], float(emb["normalization_target"]))
    keep = np.isfinite(latent_v).all(axis=1)
    details = {
        "mode": "dynamical", "fit_reused": True,
        "finite_gene_rule": "velocity_genes true and finite in every cell; all other genes zero",
        "n_pca_genes": int(len(usable)), "n_genes_used": int(usable.sum()),
        "n_genes_zeroed": int((~usable).sum()), "n_cells_dropped": int((~keep).sum()),
    }
    if not keep.any():
        raise RuntimeError("Revision-3 finite-gene rule left no finite scVelo source cells")
    save_velocity("scnt_hema", "scvelo", emb["z"][keep], latent_v[keep], np.arange(raw.n_obs)[keep], raw.obs_names.to_numpy(str)[keep], raw.obs["cell_type"].to_numpy(str)[keep], details, started)


def velocity_scnt_dynamo_gene() -> None:
    started = time.time()
    raw, processed, _raw_idx, emb, counts = _scnt_common()
    raw_values = dense(processed.layers["velocity_alpha_minus_gamma_s"], np.float64)
    usable = np.isfinite(raw_values).all(axis=0)
    values = np.zeros_like(raw_values)
    values[:, usable] = raw_values[:, usable]
    latent_v = _chain_rule_total(counts, values, emb["linear_map"], float(emb["normalization_target"]))
    if not np.isfinite(latent_v).all():
        raise RuntimeError("Dynamo finite-gene rule left non-finite mapped velocity; S11 permits cell dropping only for scVelo")
    details = {
        "layer": "velocity_alpha_minus_gamma_s", "input": str(SCNT_PROCESSED),
        "stored_vecfld_pca_present": bool("VecFld_pca" in processed.uns),
        "finite_gene_rule": "finite in every cell; all other genes zero",
        "n_pca_genes": int(len(usable)), "n_genes_used": int(usable.sum()),
        "n_genes_zeroed": int((~usable).sum()), "n_cells_dropped": 0,
    }
    save_velocity("scnt_hema", "dynamo_gene", emb["z"], latent_v, np.arange(raw.n_obs), raw.obs_names.to_numpy(str), raw.obs["cell_type"].to_numpy(str), details, started)


def velocity_larry_ot() -> None:
    started = time.time()
    import ot
    emb = load_larry_embedding()
    z_all, day, sampled = emb["z_all"], emb["day"], emb["sample_index"]
    z_parts: list[np.ndarray] = []; v_parts: list[np.ndarray] = []; i_parts: list[np.ndarray] = []; details: dict[str, Any] = {"library": "POT", "version": getattr(ot, "__version__", "unknown"), "balanced": True, "pairs": []}
    for earlier in (2.0, 4.0):
        left = sampled[day[sampled] == earlier]
        right = sampled[day[sampled] == earlier + 2.0]
        x, y = z_all[left].astype(np.float64), z_all[right].astype(np.float64)
        cost = ot.dist(x, y, metric="sqeuclidean")
        epsilon = float(0.05 * np.median(cost))
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise RuntimeError(f"invalid OT epsilon {epsilon} for day {earlier:g}")
        a = np.full(len(left), 1.0 / len(left)); b = np.full(len(right), 1.0 / len(right))
        plan = ot.sinkhorn(a, b, cost, reg=epsilon)
        bary = (plan @ y) / np.maximum(plan.sum(axis=1, keepdims=True), 1e-12)
        z_parts.append(x); v_parts.append((bary - x) / 48.0); i_parts.append(left)
        details["pairs"].append({"from_day": earlier, "to_day": earlier + 2.0, "n_source": int(len(left)), "n_target": int(len(right)), "epsilon": epsilon})
    index = np.concatenate(i_parts)
    save_velocity("larry", "ot", np.vstack(z_parts), np.vstack(v_parts), index, emb["cell_id"][index], emb["cell_type"][index], details, started)


def velocity_larry_lineage() -> None:
    started = time.time()
    emb = load_larry_embedding()
    data = ad.read_h5ad(LARRY_RAW)
    if not np.array_equal(data.obs_names.to_numpy(str), emb["cell_id"].astype(str)):
        raise RuntimeError("LARRY embedding cell order differs from raw input")
    clone = data.obsm["clone_matrix"].tocsr()
    csc = clone.tocsc()
    z_all, day, sampled = emb["z_all"], emb["day"], emb["sample_index"]
    z_parts: list[np.ndarray] = []; v_parts: list[np.ndarray] = []; i_parts: list[np.ndarray] = []
    retained_by_day: dict[str, int] = {}
    for earlier in (2.0, 4.0):
        next_day = earlier + 2.0
        later_mask = day == next_day
        mean_by_clone: dict[int, np.ndarray] = {}
        for col in range(csc.shape[1]):
            rows = csc.indices[csc.indptr[col]:csc.indptr[col + 1]]
            later = rows[later_mask[rows]]
            if len(later):
                mean_by_clone[col] = z_all[later].mean(axis=0)
        selected = sampled[day[sampled] == earlier]
        keep: list[int] = []; velocity: list[np.ndarray] = []
        for cell in selected:
            cols = clone.indices[clone.indptr[cell]:clone.indptr[cell + 1]]
            futures = [mean_by_clone[int(col)] for col in cols if int(col) in mean_by_clone]
            if futures:
                keep.append(int(cell)); velocity.append((np.mean(futures, axis=0) - z_all[cell]) / 48.0)
        if keep:
            keep_arr = np.asarray(keep, dtype=int)
            z_parts.append(z_all[keep_arr]); v_parts.append(np.vstack(velocity)); i_parts.append(keep_arr)
        retained_by_day[str(int(earlier))] = len(keep)
    if not i_parts:
        raise RuntimeError("no LARRY sampled cells had a clone on the next time point")
    index = np.concatenate(i_parts)
    save_velocity("larry", "lineage", np.vstack(z_parts), np.vstack(v_parts), index, emb["cell_id"][index], emb["cell_type"][index], {"next_timepoint_only": True, "all_cells_used_for_clone_means": True, "retained_by_day": retained_by_day}, started)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("scnt_hema", "larry"), required=True)
    parser.add_argument("--action", choices=("prepare", "fit_scvelo", "source"), required=True)
    parser.add_argument("--source", choices=("labelling", "scvelo", "dynamo_gene", "ot", "lineage"))
    args = parser.parse_args()
    if args.action == "prepare":
        if args.source is not None:
            parser.error("--source is invalid with --action prepare")
        (prepare_scnt if args.dataset == "scnt_hema" else prepare_larry)()
        return
    if args.action == "fit_scvelo":
        if args.dataset != "scnt_hema" or args.source is not None:
            parser.error("fit_scvelo requires --dataset scnt_hema and no --source")
        fit_scnt_scvelo()
        return
    if args.source is None:
        parser.error("--source is required with --action source")
    valid = {"scnt_hema": {"labelling", "scvelo", "dynamo_gene"}, "larry": {"ot", "lineage"}}
    if args.source not in valid[args.dataset]:
        parser.error(f"source {args.source!r} does not belong to {args.dataset}")
    runners = {
        "labelling": velocity_scnt_labelling, "scvelo": velocity_scnt_scvelo,
        "dynamo_gene": velocity_scnt_dynamo_gene, "ot": velocity_larry_ot,
        "lineage": velocity_larry_lineage,
    }
    runners[args.source]()


if __name__ == "__main__":
    main()
