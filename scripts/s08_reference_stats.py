#!/usr/bin/env python3
"""orient/S08: common real/synthetic reference-statistics pipeline.

Each statistic function consumes the same ``Dataset`` representation regardless
of whether the source is real or synthetic.  This entry point is deliberately
one dataset × group, so every expensive call has an independent checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks/real_reference"
DATA = OUT / "data"
CHECKPOINTS = OUT / "checkpoints"
MANIFEST_V1 = ROOT / "benchmarks/synthetic_multimodal/MANIFEST.csv"
MANIFEST_V2 = ROOT / "benchmarks/synthetic_multimodal/MANIFEST_v2.csv"
REAL_PATHS = {
    "scnt_hema": Path.home() / "Documents/scJDO2/benchmarks/data/dynamo_hematopoiesis/data/hematopoiesis_raw.h5ad",
    "larry": Path.home() / "Documents/scJDO2/benchmarks/data/larry/larry_inVitro_raw.h5ad",
    "setty": Path.home() / "Documents/scJDO/data/BoneMarrow/human_cd34_bone_marrow.h5ad",
    "scnt_neuron": Path.home() / "Documents/scJDO/reproducibility/gates_r26/gate2/data/neuron_labeling.h5ad",
    "hema_dynamo_derived": Path.home() / "Documents/scJDO2/benchmarks/data/dynamo_hematopoiesis/data/hematopoiesis.h5ad",
}
REAL_KEYS = ("scnt_hema", "larry", "setty", "scnt_neuron", "pancreas", "dentate")
GROUPS = ("integrity", "counts_trajectory", "splicing", "labelling", "lineage")
RAW_LAYERS = {
    "scnt_hema": ("new", "total", "spliced", "unspliced"),
    "larry": ("X",),
    "setty": ("spliced", "unspliced"),
    "pancreas": ("spliced", "unspliced"),
    "dentate": ("spliced", "unspliced"),
}


@dataclass(frozen=True)
class Dataset:
    key: str
    path: Path
    kind: str  # real | synthetic
    source: str
    seed: int | None = None
    preset: str | None = None


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def registry() -> dict[str, Dataset]:
    items: dict[str, Dataset] = {
        "scnt_hema": Dataset("scnt_hema", REAL_PATHS["scnt_hema"], "real", "local read-only"),
        "larry": Dataset("larry", REAL_PATHS["larry"], "real", "local read-only"),
        "setty": Dataset("setty", REAL_PATHS["setty"], "real", "local read-only"),
        "scnt_neuron": Dataset("scnt_neuron", REAL_PATHS["scnt_neuron"], "real", "local read-only"),
        "pancreas": Dataset("pancreas", DATA / "pancreas.h5ad", "real", "scvelo.datasets.pancreas"),
        "dentate": Dataset("dentate", DATA / "dentate.h5ad", "real", "scvelo.datasets.dentategyrus"),
    }
    for row in manifest_rows(MANIFEST_V1):
        if row["preset"] != "default":
            continue
        key = f"v1_default_s{row['seed']}"
        items[key] = Dataset(key, ROOT / row["path"], "synthetic", "MANIFEST.csv", int(row["seed"]), "v1_default")
    for row in manifest_rows(MANIFEST_V2):
        if row["preset"] not in {"v2_default", "v2_stress", "v2_batch"}:
            continue
        key = f"{row['preset']}_s{row['seed']}"
        items[key] = Dataset(key, ROOT / row["path"], "synthetic", "MANIFEST_v2.csv", int(row["seed"]), row["preset"])
    expected = {f"v1_default_s{i}" for i in range(3)} | {f"{p}_s{i}" for p in ("v2_default", "v2_stress", "v2_batch") for i in range(3)} | set(REAL_KEYS)
    if set(items) != expected:
        raise RuntimeError(f"unexpected S08 registry difference: {set(items) ^ expected}")
    return items


def matrix_values(matrix: Any) -> np.ndarray:
    if sp.issparse(matrix):
        return np.asarray(matrix.data)
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix).ravel()


def is_integer_matrix(matrix: Any) -> tuple[bool, str, int]:
    values = matrix_values(matrix)
    if values.size == 0:
        return True, str(getattr(matrix, "dtype", "unknown")), 0
    # Integer-valued float data are valid raw counts; no tolerance-based rounding is applied.
    return bool(np.equal(values, np.floor(values)).all() and np.isfinite(values).all()), str(values.dtype), int(values.size)


def open_data(dataset: Dataset, *, backed: bool = False) -> ad.AnnData:
    if not dataset.path.is_file():
        raise FileNotFoundError(f"S08 required dataset missing: {dataset.path}")
    return ad.read_h5ad(dataset.path, backed="r" if backed else None)


def layer(data: ad.AnnData, key: str) -> Any:
    return data.X if key == "X" else data.layers[key]


def raw_layers(dataset: Dataset, data: ad.AnnData) -> tuple[str, ...]:
    if dataset.key.startswith(("v1_", "v2_")):
        return ("X", "new", "total", "spliced", "unspliced")
    return RAW_LAYERS.get(dataset.key, ())


def require_schema(dataset: Dataset, data: ad.AnnData) -> None:
    key = dataset.key
    needed: dict[str, tuple[str, ...]] = {
        "scnt_hema": ("new", "total", "spliced", "unspliced"),
        "larry": ("clone_matrix",),
        "setty": ("spliced", "unspliced", "palantir_pseudotime", "clusters"),
        "scnt_neuron": ("new", "total"),
        "pancreas": ("spliced", "unspliced", "clusters"),
        "dentate": ("spliced", "unspliced", "clusters"),
    }
    if key in needed:
        for value in needed[key]:
            if value in {"new", "total", "spliced", "unspliced"} and value not in data.layers:
                raise KeyError(f"{key} missing layer {value!r}")
            if value == "clone_matrix" and value not in data.obsm:
                raise KeyError(f"{key} missing obsm[{value!r}]")
            if value in {"palantir_pseudotime", "clusters"} and value not in data.obs:
                raise KeyError(f"{key} missing obs[{value!r}]")
    if key.startswith(("v1_", "v2_")):
        for value in ("new", "total", "spliced", "unspliced"):
            if value not in data.layers:
                raise KeyError(f"{key} missing synthetic layer {value!r}")
        for value in ("day", "pulse_hours", "clone", "fate_true"):
            if value not in data.obs:
                raise KeyError(f"{key} missing synthetic obs[{value!r}]")


def append_stats(records: list[dict[str, Any]], *, group: str | None = None) -> None:
    """Append new records, replacing the complete logical group on an intentional rerun."""
    path = OUT / "stats.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["dataset", "kind", "statistic", "feature", "value", "n", "source"])
    new = pd.DataFrame(records)
    if not new.empty:
        if group is None:
            old = old[~old.dataset.isin(new.dataset.unique()) | ~old.statistic.isin(new.statistic.unique())]
        else:
            prefixes = {"counts_trajectory": ("C", "T"), "splicing": ("S",), "labelling": ("L",), "lineage": ("G",)}[group]
            prior = old.dataset.isin(new.dataset.unique()) & old.statistic.astype(str).str.startswith(prefixes)
            old = old.loc[~prior]
            for dataset in new.dataset.unique():
                fresh = set(new.loc[new.dataset == dataset, "statistic"].astype(str))
                for artifact in (OUT / "distributions").glob(f"{dataset}__*.csv"):
                    statistic = artifact.stem.split("__", 1)[1]
                    if statistic.startswith(prefixes) and statistic not in fresh:
                        artifact.unlink()
        old = pd.concat([old, new], ignore_index=True)
    old.to_csv(path, index=False)


def record_integrity(dataset: Dataset) -> dict[str, Any]:
    started = time.time()
    data = open_data(dataset, backed=True)
    try:
        require_schema(dataset, data)
        records: list[dict[str, Any]] = []
        checks: list[dict[str, Any]] = []
        for key in raw_layers(dataset, data):
            valid, dtype, n_values = is_integer_matrix(layer(data, key))
            checks.append({"layer": key, "integer": valid, "dtype": dtype, "n_nonzero_values": n_values})
            records.append({"dataset": dataset.key, "kind": dataset.kind, "statistic": "integrity_raw_integer", "feature": key, "value": float(valid), "n": n_values, "source": dataset.source})
            if not valid:
                raise RuntimeError(f"raw-count integrity failed for {dataset.key} layer {key}: non-integer values")
        records += [
            {"dataset": dataset.key, "kind": dataset.kind, "statistic": "integrity_shape", "feature": "n_obs", "value": float(data.n_obs), "n": data.n_obs, "source": dataset.source},
            {"dataset": dataset.key, "kind": dataset.kind, "statistic": "integrity_shape", "feature": "n_vars", "value": float(data.n_vars), "n": data.n_vars, "source": dataset.source},
        ]
        append_stats(records)
        return {"status": "complete", "dataset": dataset.key, "path": str(dataset.path), "sha256": sha256(dataset.path), "bytes": dataset.path.stat().st_size, "shape": [data.n_obs, data.n_vars], "raw_layers": checks, "started_utc": now(), "wall_seconds": round(time.time() - started, 3)}
    finally:
        if getattr(data, "isbacked", False):
            data.file.close()


def approved_download(key: str) -> dict[str, Any]:
    if key not in {"pancreas", "dentate"}:
        raise ValueError("only pancreas and dentate are approved downloads")
    DATA.mkdir(parents=True, exist_ok=True)
    destination = registry()[key].path
    started = time.time()
    import scvelo as scv
    if key == "pancreas":
        data = scv.datasets.pancreas(file_path=destination)
        url = "https://github.com/theislab/scvelo_notebooks/raw/master/data/Pancreas/endocrinogenesis_day15.h5ad"
    else:
        data = scv.datasets.dentategyrus(file_path=destination)
        url = "https://github.com/theislab/scvelo_notebooks/raw/master/data/DentateGyrus/10X43_1.h5ad"
    if not destination.is_file():
        # scvelo may honor a cache location but not the requested final name; persist only the returned object to the approved directory.
        data.write_h5ad(destination, compression="gzip")
    return {"dataset": key, "url": url, "path": str(destination), "sha256": sha256(destination), "bytes": destination.stat().st_size, "shape": [data.n_obs, data.n_vars], "started_utc": now(), "wall_seconds": round(time.time() - started, 3)}


def run_integrity(dataset: Dataset) -> None:
    payload = record_integrity(dataset)
    payload.update({"task": "orient/S08", "group": "integrity", "finished_utc": now(), "script_sha256": sha256(Path(__file__))})
    json_write(CHECKPOINTS / f"{dataset.key}_integrity.json", payload)
    print(json.dumps(payload, indent=2), flush=True)



def dense(matrix: Any) -> np.ndarray:
    if sp.issparse(matrix):
        return matrix.toarray()
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
    return np.asarray(matrix)


def base_counts(dataset: Dataset, data: ad.AnnData) -> Any:
    if dataset.key == "larry" or dataset.key.startswith(("v1_", "v2_")):
        return data.X
    if dataset.key == "scnt_hema":
        return data.layers["total"]
    if dataset.key in {"setty", "pancreas", "dentate"}:
        return data.layers["spliced"] + data.layers["unspliced"]
    raise ValueError(f"{dataset.key} has no raw-count pathway")


def subset_for_counts(dataset: Dataset) -> ad.AnnData:
    """Load raw counts, deterministically subsampling only LARRY's C/T path."""
    if dataset.key != "larry":
        return open_data(dataset)
    # AnnData 0.11 cannot row-index this backed CSR against the installed SciPy.
    # Read exactly the specified 20,000 rows from HDF5 CSR arrays instead.
    import h5py
    backed = open_data(dataset, backed=True)
    try:
        rng = np.random.default_rng(0)
        rows = np.sort(rng.choice(backed.n_obs, size=min(20_000, backed.n_obs), replace=False))
        obs, var = backed.obs.iloc[rows].copy(), backed.var.copy()
    finally:
        backed.file.close()
    with h5py.File(dataset.path, "r") as handle:
        group = handle["X"]
        indptr = np.asarray(group["indptr"], dtype=np.int64)
        chunks_data, chunks_indices, new_ptr = [], [], [0]
        for r in rows:
            lo, hi = int(indptr[r]), int(indptr[r + 1])
            chunks_data.append(group["data"][lo:hi]); chunks_indices.append(group["indices"][lo:hi])
            new_ptr.append(new_ptr[-1] + hi - lo)
    x = sp.csr_matrix((np.concatenate(chunks_data), np.concatenate(chunks_indices), np.asarray(new_ptr, dtype=np.int64)), shape=(len(rows), len(var)))
    return ad.AnnData(x, obs=obs, var=var)


def prepare(dataset: Dataset) -> tuple[ad.AnnData, np.ndarray, np.ndarray]:
    """One shared raw-count/HVG/log/PCA/neighbours/Leiden preprocessing path."""
    import scanpy as sc
    data = subset_for_counts(dataset)
    raw = base_counts(dataset, data)
    if hasattr(raw, "to_memory"):
        raw = raw.to_memory()
    if not sp.issparse(raw):
        raw = sp.csr_matrix(raw)
    raw = raw.tocsr().astype(np.float32)
    keep = np.asarray(raw.sum(axis=0)).ravel() >= 10
    if int(keep.sum()) < 800:
        raise RuntimeError(f"{dataset.key}: only {int(keep.sum())} genes remain after >=10 count filter")
    raw = raw[:, keep]
    obs = data.obs.copy()
    work = ad.AnnData(raw.copy(), obs=obs, var=pd.DataFrame(index=np.asarray(data.var_names)[keep]))
    work.layers["counts"] = raw.copy()
    n_top = min(800, work.n_vars)
    sc.pp.highly_variable_genes(work, flavor="seurat_v3", n_top_genes=n_top, layer="counts", inplace=True)
    hvg = np.asarray(work.var["highly_variable"], dtype=bool)
    if int(hvg.sum()) != n_top:
        raise RuntimeError(f"{dataset.key}: expected {n_top} HVGs, observed {int(hvg.sum())}")
    median_library = float(np.median(np.asarray(raw.sum(axis=1)).ravel()))
    sc.pp.normalize_total(work, target_sum=median_library)
    sc.pp.log1p(work)
    n_pcs = 30 if dataset.kind == "real" else 20
    n_pcs = min(n_pcs, work.n_obs - 1, int(hvg.sum()) - 1)
    sc.pp.pca(work, n_comps=n_pcs, use_highly_variable=True, svd_solver="arpack")
    sc.pp.neighbors(work, n_neighbors=min(30, work.n_obs - 1), use_rep="X_pca", random_state=0)
    sc.tl.leiden(work, resolution=1.0, random_state=0, key_added="s08_leiden")
    return work, keep, hvg


def num(value: Any) -> float:
    return float(value) if np.isfinite(value) else float("nan")


def row(dataset: Dataset, statistic: str, feature: str, value: float, n: int = 1) -> dict[str, Any]:
    return {"dataset": dataset.key, "kind": dataset.kind, "statistic": statistic, "feature": feature, "value": num(value), "n": int(n), "source": dataset.source}


def distribution(dataset: Dataset, statistic: str, values: np.ndarray, records: list[dict[str, Any]], *, feature: str = "value") -> None:
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    path = OUT / "distributions" / f"{dataset.key}__{statistic}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"value": values}).to_csv(path, index=False)
    if values.size:
        q = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
        for name, value in zip(("q05", "q25", "median", "q75", "q95"), q):
            records.append(row(dataset, statistic, name, value, values.size))
        records.append(row(dataset, statistic, "mean", float(values.mean()), values.size))


def count_time(dataset: Dataset, work: ad.AnnData) -> np.ndarray | None:
    for col in ("day", "age(days)", "time_hours", "Time point", "time"):
        if col in work.obs:
            return pd.to_numeric(work.obs[col], errors="coerce").to_numpy(float)
    return None


def palantir_summary(dataset: Dataset, work: ad.AnnData, records: list[dict[str, Any]]) -> None:
    """T2 with the frozen/Revision-1 root rules; no annotation substitute."""
    if dataset.key == "pancreas":
        records.append(row(dataset, "T2_not_applicable", "no_time_field", 1.0, work.n_obs))
        return
    import palantir
    z = np.asarray(work.obsm["X_pca"], dtype=float)
    if dataset.key == "setty":
        t = pd.to_numeric(work.obs["palantir_pseudotime"], errors="coerce").to_numpy(float)
    else:
        t = count_time(dataset, work)
    if t is None or not np.isfinite(t).any():
        raise RuntimeError(f"{dataset.key}: T2 requires the task-authorised time source")
    earliest = np.nanmin(t)
    eligible = np.flatnonzero(np.isfinite(t) & (t == earliest))
    centroid = z[eligible].mean(axis=0)
    # Deterministically choose the earliest-time cell nearest its earliest-time centroid.
    root_idx = int(eligible[np.argmin(np.sum((z[eligible] - centroid) ** 2, axis=1))])
    frame = pd.DataFrame(z, index=work.obs_names)
    dm = palantir.utils.run_diffusion_maps(frame, n_components=min(10, z.shape[1]), knn=min(30, z.shape[0] - 1), seed=0)
    ms = palantir.utils.determine_multiscale_space(dm)
    result = palantir.core.run_palantir(ms, early_cell=work.obs_names[root_idx], num_waypoints=min(500, z.shape[0]), knn=min(30, z.shape[0] - 1), seed=0)
    entropy = np.asarray(result.entropy, dtype=float)
    distribution(dataset, "T2_palantir_entropy", entropy, records)
    records.append(row(dataset, "T2_terminal_states", "count", int(result.branch_probs.shape[1]), work.n_obs))
    records.append(row(dataset, "T2_root_time", "value", float(earliest), work.n_obs))


def run_counts_trajectory(dataset: Dataset) -> list[dict[str, Any]]:
    if dataset.key == "scnt_neuron":
        return [row(dataset, "CT_not_applicable", "reason_normalised_ratio_only", 1.0)]
    work, _, hvg = prepare(dataset)
    raw = work.layers["counts"].tocsr()
    records: list[dict[str, Any]] = []
    library = np.asarray(raw.sum(axis=1)).ravel()
    distribution(dataset, "C1_log10_library", np.log10(np.maximum(library, 1.0)), records)
    records.append(row(dataset, "C2_genes_detected", "genes_kept", work.n_vars, work.n_obs))
    distribution(dataset, "C2_genes_detected", np.diff(raw.indptr), records, feature="per_cell")
    logx = dense(work.X)
    hvgx = logx[:, hvg]
    distribution(dataset, "C3_hvg_log_mean", hvgx.mean(axis=0), records)
    mean = np.asarray(raw.mean(axis=0)).ravel(); mean2 = np.asarray(raw.power(2).mean(axis=0)).ravel()
    dispersion = (mean2 - mean * mean) / np.maximum(mean, 1e-8)
    distribution(dataset, "C4_dispersion", dispersion[hvg], records)
    distribution(dataset, "C5_zero_fraction", 1.0 - np.asarray((raw > 0).mean(axis=0)).ravel()[hvg], records)
    x = hvgx - hvgx.mean(axis=0, keepdims=True)
    denom = np.sqrt(np.maximum((x * x).sum(axis=0), 1e-12)); rng = np.random.default_rng(0); p = x.shape[1]
    pairs = rng.integers(0, p, size=(200, 2)); pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    corr = np.array([np.dot(x[:, i], x[:, j]) / (denom[i] * denom[j]) for i, j in pairs])
    distribution(dataset, "C6_hvg_pair_correlation", corr, records)
    var_ratio = np.asarray(work.uns["pca"]["variance_ratio"], dtype=float)
    for i, value in enumerate(var_ratio, 1): records.append(row(dataset, "C7_pca_variance_ratio", f"PC{i}", value, work.n_obs))
    pr = (var_ratio.sum() ** 2) / np.maximum((var_ratio * var_ratio).sum(), 1e-12)
    records.append(row(dataset, "C7_participation_ratio", "value", pr, work.n_obs))
    clusters = work.obs["s08_leiden"].astype(str)
    for name, value in clusters.value_counts(normalize=True).sort_index().items(): records.append(row(dataset, "C8_leiden_share", str(name), value, work.n_obs))
    palantir_summary(dataset, work, records)
    return records


def run_splicing(dataset: Dataset) -> list[dict[str, Any]]:
    if dataset.key in {"larry", "scnt_neuron"}:
        return [row(dataset, "S_not_applicable", "no_spliced_unspliced", 1.0)]
    import scvelo as scv
    data = open_data(dataset)
    work, keep, hvg = prepare(dataset)
    selected = np.flatnonzero(keep)[hvg]
    spliced = data.layers["spliced"][:, selected]; unspliced = data.layers["unspliced"][:, selected]
    if not sp.issparse(spliced): spliced = sp.csr_matrix(spliced)
    if not sp.issparse(unspliced): unspliced = sp.csr_matrix(unspliced)
    velo = ad.AnnData(spliced.copy(), obs=work.obs.copy(), var=work.var.loc[hvg].copy())
    velo.layers["spliced"] = spliced.copy(); velo.layers["unspliced"] = unspliced.copy(); velo.obsm["X_pca"] = np.asarray(work.obsm["X_pca"])
    scv.pp.moments(velo, n_pcs=min(velo.obsm["X_pca"].shape[1], 30), n_neighbors=min(30, velo.n_obs - 1))
    scv.tl.recover_dynamics(velo, n_jobs=8, max_iter=10)
    records: list[dict[str, Any]] = []
    total_u = float(unspliced.sum()); total_s = float(spliced.sum())
    records.append(row(dataset, "S1_unspliced_fraction", "global", total_u / max(total_s + total_u, 1.0), velo.n_obs))
    ratio = np.asarray(unspliced.mean(axis=0)).ravel() / np.maximum(np.asarray(spliced.mean(axis=0)).ravel(), 1e-8)
    distribution(dataset, "S2_unspliced_spliced_ratio", ratio, records)
    # S3: direct scVelo 5%-extreme-quantile, zero-intercept steady-state fit.
    # For each gene, the fit cells are the upper 5% (including ties) of
    # Ms/max(Ms) + Mu/max(Mu); gamma' is sum(Ms*Mu)/sum(Ms^2) on those cells.
    # This uses kNN-30 first moments only and is independent of S4 dynamics.
    ms = np.asarray(velo.layers["Ms"], dtype=float)
    mu = np.asarray(velo.layers["Mu"], dtype=float)
    valid = np.isfinite(ms) & np.isfinite(mu)
    max_s = np.maximum(np.nanmax(np.where(valid, ms, np.nan), axis=0), 1e-3)
    max_u = np.maximum(np.nanmax(np.where(valid, mu, np.nan), axis=0), 1e-3)
    extreme = ms / max_s[None, :] + mu / max_u[None, :]
    cutoff = np.nanpercentile(extreme, 95, axis=0, method="linear")
    fit_cells = valid & (extreme >= cutoff[None, :])
    numerator = np.where(fit_cells, ms * mu, 0.0).sum(axis=0)
    denominator = np.where(fit_cells, ms * ms, 0.0).sum(axis=0)
    # When the selected spliced values are all zero, the unique zero-intercept
    # convention is gamma'=0: this still yields one of the four specified
    # classes (nonzero Mu is induction; all-zero moments are steady).
    gamma = np.divide(numerator, denominator, out=np.zeros(velo.n_vars, dtype=float), where=denominator > 0)
    residual = mu - gamma[None, :] * ms
    n_valid = valid.sum(axis=0)
    f_up = np.divide(((residual > 0.10 * mu) & valid).sum(axis=0), n_valid, out=np.zeros(velo.n_vars, dtype=float), where=n_valid > 0)
    f_down = np.divide(((-residual > 0.10 * mu) & valid).sum(axis=0), n_valid, out=np.zeros(velo.n_vars, dtype=float), where=n_valid > 0)
    classes = np.full(velo.n_vars, "steady", dtype=object)
    classes[(f_up >= 0.1) & (f_down >= 0.1)] = "cycle"
    classes[(f_up >= 0.1) & (f_down < 0.1)] = "induction"
    classes[(f_up < 0.1) & (f_down >= 0.1)] = "repression"
    distribution(dataset, "S3_gamma_prime", gamma, records)
    distribution(dataset, "S3_f_up", f_up, records)
    distribution(dataset, "S3_f_down", f_down, records)
    records.append(row(dataset, "S3_zero_slope_fraction", "fraction", np.mean(denominator <= 0), velo.n_vars))
    records.append(row(dataset, "S3_n_valid_cells", "median", float(np.median(n_valid)), velo.n_vars))
    records.append(row(dataset, "S3_n_fit_cells", "median", float(np.median(fit_cells.sum(axis=0))), velo.n_vars))
    for label in ("cycle", "induction", "repression", "steady"):
        records.append(row(dataset, "S3_phase_portrait_class", label, np.mean(classes == label), velo.n_vars))
    # The superseded likelihood/switching result is preserved under a distinct S3b name.
    likelihood = np.asarray(velo.var.get("fit_likelihood", np.full(velo.n_vars, np.nan)), dtype=float)
    distribution(dataset, "S4_dynamics_likelihood", likelihood, records)
    records.append(row(dataset, "S4_recovered_fraction", "fraction", np.mean(np.isfinite(likelihood)), velo.n_vars))
    records.append(row(dataset, "S4_likelihood_ge_0_3", "fraction", np.mean(likelihood >= 0.3), velo.n_vars))
    t = np.asarray(velo.var.get("fit_t_", np.zeros(velo.n_vars)), dtype=float)
    switch_classes = np.where(likelihood < 0.3, "low_fit", np.where(t < np.nanmedian(t), "early_switch", "late_switch"))
    for label in ("low_fit", "early_switch", "late_switch"):
        records.append(row(dataset, "S3b_scvelo_switch_class", label, np.mean(switch_classes == label), velo.n_vars))
    return records


def run_labelling(dataset: Dataset) -> list[dict[str, Any]]:
    if dataset.key == "larry" or dataset.key in {"setty", "pancreas", "dentate"}:
        return [row(dataset, "L_not_applicable", "no_new_total", 1.0)]
    data = open_data(dataset)
    if dataset.key == "scnt_neuron":
        new = dense(data.layers["new"]); total = dense(data.layers["total"]); frac = new.sum(axis=1) / np.maximum(total.sum(axis=1), 1e-8)
        records: list[dict[str, Any]] = []; distribution(dataset, "L1_new_total_fraction", frac, records)
        lib = total.sum(axis=1); bins = pd.qcut(lib, 4, duplicates="drop")
        for label, value in pd.Series(frac).groupby(bins, observed=True).median().items(): records.append(row(dataset, "L3_fraction_by_total_bin", str(label), value, int((bins == label).sum())))
        return records
    from scjdo2.labelled_rates import LabelledRateModel
    work, keep, hvg = prepare(dataset)
    selected = np.flatnonzero(keep)[hvg]; new = dense(data.layers["new"][:, selected]).astype(np.float32); total = dense(data.layers["total"][:, selected]).astype(np.float32)
    pulse_col = "pulse_hours" if "pulse_hours" in data.obs else "time"
    pulse = pd.to_numeric(data.obs[pulse_col], errors="coerce").to_numpy(np.float32)
    valid = np.isfinite(pulse) & (pulse > 0)
    if valid.sum() < 20: raise RuntimeError(f"{dataset.key}: insufficient positive labelling times")
    fit = LabelledRateModel(seed=0).fit(new[valid], total[valid], np.asarray(work.obsm["X_pca"])[valid], pulse[valid])
    gamma = np.asarray(fit.gamma, dtype=float); half_life = np.log(2.0) / np.maximum(gamma, 1e-12)
    frac = new.sum(axis=1) / np.maximum(total.sum(axis=1), 1e-8)
    records: list[dict[str, Any]] = []; distribution(dataset, "L2_gamma_half_life", half_life, records); distribution(dataset, "L1_new_total_fraction", frac, records)
    records.append(row(dataset, "L2_held_out_nll", "value", float(fit.best_val_nll), int(valid.sum())))
    for timepoint in sorted(np.unique(pulse[valid])):
        vals = frac[pulse == timepoint]; records.append(row(dataset, "L1_new_total_by_time", str(timepoint), np.median(vals), vals.size))
    lib = total.sum(axis=1); bins = pd.qcut(lib, 4, duplicates="drop")
    for label, value in pd.Series(frac).groupby(bins, observed=True).median().items(): records.append(row(dataset, "L3_fraction_by_total_bin", str(label), value, int((bins == label).sum())))
    return records


def lineage_labels(dataset: Dataset, data: ad.AnnData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if dataset.key == "larry":
        labels = np.asarray(data.obs["Cell type annotation"].astype(str)); time = pd.to_numeric(data.obs["Time point"], errors="coerce").to_numpy(float); matrix = data.obsm["clone_matrix"]
        if hasattr(matrix, "to_memory"): matrix = matrix.to_memory()
        if sp.issparse(matrix):
            present = matrix.getnnz(axis=1) > 0
            clone = np.asarray(matrix.argmax(axis=1)).ravel().astype(str)
        else:
            m = dense(matrix); present = m.sum(axis=1) > 0; clone = np.argmax(m, axis=1).astype(str)
        clone[~present] = ""
    elif dataset.key.startswith(("v1_", "v2_")):
        labels = np.asarray(data.obs["fate_true"].astype(str)); time = pd.to_numeric(data.obs["day"], errors="coerce").to_numpy(float); clone = np.asarray(data.obs["clone"].astype(str)); clone[pd.isna(data.obs["clone"]).to_numpy()] = ""
    else: raise ValueError("no lineage source")
    return clone, time, labels


def run_lineage(dataset: Dataset) -> list[dict[str, Any]]:
    if dataset.key not in {"larry"} and not dataset.key.startswith(("v1_", "v2_")):
        return [row(dataset, "G_not_applicable", "no_clone_source", 1.0)]
    data = open_data(dataset, backed=(dataset.key == "larry"))
    try:
        clone, times, labels = lineage_labels(dataset, data); valid = clone != ""; records: list[dict[str, Any]] = []
        records.append(row(dataset, "G1_barcoded_cell_fraction", "value", valid.mean(), len(clone)))
        ids, counts = np.unique(clone[valid], return_counts=True); distribution(dataset, "G2_clone_size", counts, records)
        records.append(row(dataset, "G2_singleton_fraction", "value", np.mean(counts == 1), len(counts)))
        multi = []; purity = []; pair_shared=[]
        for cid in ids:
            mask = clone == cid; ts = np.unique(times[mask][np.isfinite(times[mask])]); multi.append(len(ts) >= 2)
            vals, c = np.unique(labels[mask], return_counts=True); purity.append(c.max()/c.sum())
        for a in np.unique(times[np.isfinite(times)]):
            for b in np.unique(times[np.isfinite(times)]):
                if b <= a: continue
                ca=set(clone[(times==a)&valid]); cb=set(clone[(times==b)&valid]); pair_shared.append(len(ca&cb)/max(1,len(ca|cb)))
        records.append(row(dataset, "G3_multi_time_clone_fraction", "value", np.mean(multi), len(ids)))
        distribution(dataset, "G4_between_time_clone_overlap", np.asarray(pair_shared), records)
        distribution(dataset, "G5_clone_label_purity", np.asarray(purity), records)
        return records
    finally:
        if getattr(data, "isbacked", False): data.file.close()


def run_group(dataset: Dataset, group: str) -> None:
    started=time.time()
    if group == "integrity": payload=record_integrity(dataset); records=[]
    else:
        records = {"counts_trajectory":run_counts_trajectory, "splicing":run_splicing, "labelling":run_labelling, "lineage":run_lineage}[group](dataset)
        append_stats(records, group=group)
        payload={"status":"complete","dataset":dataset.key,"group":group,"path":str(dataset.path),"sha256":sha256(dataset.path),"records":len(records),"started_utc":now(),"wall_seconds":round(time.time()-started,3)}
    payload.update({"task":"orient/S08","finished_utc":now(),"script_sha256":sha256(Path(__file__))})
    json_write(CHECKPOINTS / f"{dataset.key}_{group}.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--dataset", choices=sorted(registry())); parser.add_argument("--group", choices=GROUPS); parser.add_argument("--download", choices=("pancreas","dentate")); args=parser.parse_args()
    if args.download:
        payload=approved_download(args.download); json_write(CHECKPOINTS / f"{args.download}_download.json", payload); print(json.dumps(payload,indent=2)); return
    if not args.dataset or not args.group: parser.error("supply --download DATASET or both --dataset and --group")
    run_group(registry()[args.dataset], args.group)


if __name__ == "__main__":
    main()
