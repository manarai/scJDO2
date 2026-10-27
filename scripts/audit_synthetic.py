#!/usr/bin/env python3
"""Frozen O1–O4 recoverability audit for synthetic multimodal generator v2.

The audit uses true rates only for retrospective scoring.  It never writes into
an input H5AD and imports S03's run script for the O4 scVelo settings without
modifying that script.
"""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors

from scjdo2.data.synthetic_multimodal import true_velocity_in_embedding

ROOT = Path(__file__).resolve().parents[1]
V1_DATA = ROOT / "benchmarks/synthetic_multimodal/data"
V2_DATA = V1_DATA / "v2"
DEFAULT_OUTPUT = ROOT / "benchmarks/synthetic_multimodal/v2_audit/audit.csv"


def _dense(layer: Any) -> np.ndarray:
    return np.asarray(layer.toarray() if hasattr(layer, "toarray") else layer, dtype=np.float64)


def _load_s03_module() -> Any:
    path = ROOT / "scripts/run_scvelo_palantir_s03.py"
    spec = importlib.util.spec_from_file_location("s03_settings_for_s05", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import S03 runner from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gene_factor(adata: ad.AnnData) -> np.ndarray:
    day = adata.obs["batch_day"].to_numpy(dtype=int) - 1
    return adata.obs["capture_factor"].to_numpy(dtype=np.float64)[:, None] * np.asarray(
        adata.uns["batch_gene_effects"], dtype=np.float64
    )[day]


def _map_observed_estimate(adata: ad.AnnData, estimate_counts: np.ndarray) -> np.ndarray:
    """Map an observed-count velocity estimate through the recorded PCA map."""
    clone = adata.copy()
    clone.layers["ds_dt_true"] = (estimate_counts / _gene_factor(adata)).astype(np.float32)
    return true_velocity_in_embedding(clone, "pca", at="observed").astype(np.float64)


def _score(estimate: np.ndarray, truth: np.ndarray, gate: float | None) -> dict[str, float | bool | int | None]:
    estimate = np.asarray(estimate, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    denom = np.linalg.norm(estimate, axis=1) * np.linalg.norm(truth, axis=1)
    valid = np.isfinite(denom) & (denom > 1e-12)
    if not np.any(valid):
        raise RuntimeError("no finite non-zero cosine values")
    cosine = np.sum(estimate[valid] * truth[valid], axis=1) / denom[valid]
    median = float(np.median(cosine))
    return {"median_cosine": median, "n_valid": int(valid.sum()), "threshold": None if gate is None else float(gate),
            "raw_pass": None if gate is None else bool(median >= gate)}



def realism_summary(adata: ad.AnnData) -> dict[str, object]:
    """Required descriptive checks, with no threshold tuning or interpretation."""
    spliced = _dense(adata.layers["spliced"])
    unspliced = _dense(adata.layers["unspliced"])
    new = _dense(adata.layers["new"])
    total = _dense(adata.layers["total"])
    library = spliced.sum(axis=1)
    scale = float(np.median(library[library > 0])) / np.maximum(library, 1e-12)
    normalized = spliced * scale[:, None]
    gene_mean = normalized.mean(axis=0)
    usable = gene_mean > 0.5
    detected = adata.obs["clone"].notna().to_numpy()
    clone_sizes = adata.obs.loc[detected, "clone"].value_counts()
    clone_sizes = clone_sizes[clone_sizes > 0]
    spans = adata.obs.loc[detected].groupby("clone", observed=True)["day"].nunique()
    z = np.asarray(adata.obsm["X_pca"], dtype=float)
    z_true = np.asarray(adata.obsm["z_true"], dtype=float)
    r2 = [float(LinearRegression().fit(z, z_true[:, dim]).score(z, z_true[:, dim])) for dim in range(z_true.shape[1])]
    return {
        "umi_per_cell_median": float(np.median(spliced.sum(axis=1) + unspliced.sum(axis=1))),
        "genes_detected_per_cell_median": float(np.median((spliced > 0).sum(axis=1))),
        "spliced_zero_fraction": float(np.mean(spliced == 0)),
        "unspliced_share": float(unspliced.sum() / np.maximum((spliced + unspliced).sum(), 1e-12)),
        "unspliced_zero_fraction": float(np.mean(unspliced == 0)),
        "normalized_var_to_mean_median": float(np.median(normalized.var(axis=0)[usable] / gene_mean[usable])),
        "new_to_total": float(new.sum() / np.maximum(total.sum(), 1e-12)),
        "barcode_fraction": float(detected.mean()),
        "detected_clone_size_median": float(clone_sizes.median()) if len(clone_sizes) else 0.0,
        "detected_clone_size_max": int(clone_sizes.max()) if len(clone_sizes) else 0,
        "detected_multi_day_clone_fraction": float((spans >= 2).mean()) if len(spans) else 0.0,
        "detected_multi_day_clone_count": int((spans >= 2).sum()),
        "pca_r2_z_true": r2,
    }

def oracle_o1(adata: ad.AnnData) -> dict[str, float | bool | int]:
    """Observed U/S, true beta/gamma, 30-neighbor moment smoothing, library norm."""
    s = _dense(adata.layers["spliced"])
    u = _dense(adata.layers["unspliced"])
    beta = adata.var["beta_initial"].to_numpy(dtype=np.float64)
    gamma = adata.var["gamma"].to_numpy(dtype=np.float64)
    eta = float(adata.uns["synthetic_multimodal"]["capture_eta_unspliced"])
    n_neighbors = min(30, adata.n_obs)
    neighbor_index = NearestNeighbors(n_neighbors=n_neighbors).fit(adata.obsm["X_pca"]).kneighbors(adata.obsm["X_pca"])[1]
    library = np.maximum(s.sum(axis=1, keepdims=True), 1e-12)
    norm = float(np.median(library)) / library
    u_moment = (u * norm)[neighbor_index].mean(axis=1)
    s_moment = (s * norm)[neighbor_index].mean(axis=1)
    estimate = (beta[None, :] * u_moment / eta - gamma[None, :] * s_moment) / norm
    truth = true_velocity_in_embedding(adata, "pca", at="expected")
    return _score(_map_observed_estimate(adata, estimate), truth, gate=0.40)


def oracle_o2(adata: ad.AnnData) -> dict[str, float | bool | int]:
    """Observed labelling estimate with true gamma and known pulse duration."""
    new = _dense(adata.layers["new"])
    total = _dense(adata.layers["total"])
    gamma = adata.var["gamma"].to_numpy(dtype=np.float64)
    pulse = adata.obs["pulse_hours"].to_numpy(dtype=np.float64)[:, None]
    efficiency = float(adata.uns["synthetic_multimodal"]["labelling_efficiency"])
    factor = np.maximum(1.0 - np.exp(-gamma[None, :] * pulse), 1e-12)
    estimate = gamma[None, :] * (new / efficiency) / factor - gamma[None, :] * total
    truth = true_velocity_in_embedding(adata, "pca", at="expected")
    return _score(_map_observed_estimate(adata, estimate), truth, gate=0.30)


def oracle_o3(adata: ad.AnnData) -> dict[str, float | bool | int]:
    """O1 moments with per-gene extreme-quantile steady-state gamma fit."""
    s = _dense(adata.layers["spliced"])
    u = _dense(adata.layers["unspliced"])
    n_neighbors = min(30, adata.n_obs)
    neighbor_index = NearestNeighbors(n_neighbors=n_neighbors).fit(adata.obsm["X_pca"]).kneighbors(adata.obsm["X_pca"])[1]
    library = np.maximum(s.sum(axis=1, keepdims=True), 1e-12)
    norm = float(np.median(library)) / library
    u_moment = (u * norm)[neighbor_index].mean(axis=1)
    s_moment = (s * norm)[neighbor_index].mean(axis=1)
    gamma_fit = np.empty(adata.n_vars, dtype=np.float64)
    for gene in range(adata.n_vars):
        x, y = s_moment[:, gene], u_moment[:, gene]
        lower, upper = np.quantile(x, [0.05, 0.95])
        mask = (x <= lower) | (x >= upper)
        gamma_fit[gene] = (x[mask] @ y[mask]) / max(x[mask] @ x[mask], 1e-12)
    estimate = (u_moment - gamma_fit[None, :] * s_moment) / norm
    truth = true_velocity_in_embedding(adata, "pca", at="expected")
    return _score(_map_observed_estimate(adata, estimate), truth, gate=None)


def oracle_o4(adata: ad.AnnData) -> dict[str, float | bool | int]:
    """Run S03's exact scVelo preprocessing/dynamical workflow, without Palantir."""
    s03 = _load_s03_module()
    source_pca = np.asarray(adata.obsm["X_pca"]).copy()
    work = adata.copy()
    work.X = work.layers["spliced"].copy()
    s03.scv.pp.filter_genes(work, min_shared_counts=20)
    s03.scv.pp.normalize_per_cell(work)
    s03.sc.pp.log1p(work)
    s03.sc.pp.neighbors(work, use_rep="X_pca", n_neighbors=30, random_state=0)
    connectivity_before = work.obsp["connectivities"].copy()
    s03.scv.pp.moments(work, n_neighbors=None, n_pcs=None, use_rep=None)
    if (work.obsp["connectivities"] != connectivity_before).nnz:
        raise RuntimeError("S03 moments altered the fixed Scanpy neighbor graph")
    if not np.array_equal(source_pca, np.asarray(work.obsm["X_pca"])):
        raise RuntimeError("S03 scVelo preprocessing changed X_pca")
    s03.scv.tl.recover_dynamics(work)
    s03.scv.tl.velocity(work, mode="dynamical")
    s03.scv.tl.velocity_graph(work)
    s03.scv.tl.velocity_embedding(work, basis="pca")
    truth = true_velocity_in_embedding(adata, "pca", at="expected")
    metric = _score(np.asarray(work.obsm["velocity_pca"]), truth, gate=None)
    by_day: dict[str, float] = {}
    for day in sorted(adata.obs["day"].unique()):
        mask = adata.obs["day"].to_numpy() == day
        by_day[str(int(day))] = float(_score(np.asarray(work.obsm["velocity_pca"])[mask], truth[mask], gate=None)["median_cosine"])
    metric["details_json"] = json.dumps({"by_day_median_cosine": by_day}, sort_keys=True)
    return metric


def audit_one(path: Path, run_scvelo: bool) -> list[dict[str, object]]:
    started = time.time()
    adata = ad.read_h5ad(path)
    name = path.stem
    rows: list[dict[str, object]] = [{
        "dataset": name, "oracle": "realism", "median_cosine": None, "n_valid": None,
        "threshold": None, "raw_pass": None, "details_json": json.dumps(realism_summary(adata), sort_keys=True),
    }]
    for name_oracle, fn in (("O1", oracle_o1), ("O2", oracle_o2), ("O3", oracle_o3)):
        metric = fn(adata)
        rows.append({"dataset": name, "oracle": name_oracle, "details_json": "", **metric})
    if run_scvelo:
        metric = oracle_o4(adata)
        rows.append({"dataset": name, "oracle": "O4", **metric})
    elapsed = time.time() - started
    # macOS reports ru_maxrss in bytes; Linux reports KiB.
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    rss_bytes = rss if sys.platform == "darwin" else rss * 1024
    is_v2_default = name in {f"v2_default_s{seed}" for seed in range(3)}
    for row in rows:
        is_gate = bool(is_v2_default and row["oracle"] in {"O1", "O2"})
        row["gate_status"] = ("pass" if row["raw_pass"] else "fail") if is_gate else "report_only"
        row["wall_seconds"] = round(elapsed, 3)
        row["peak_rss_bytes"] = rss_bytes
        row["path"] = str(path.relative_to(ROOT))
    return rows


def default_paths(include_references: bool) -> list[Path]:
    paths = [V2_DATA / f"v2_{preset}_s0.h5ad" for preset in ("default", "batch", "stress")]
    if include_references:
        paths = [V1_DATA / f"default_s{seed}.h5ad" for seed in range(3)] + paths
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", help="H5AD to audit; defaults to v1 references plus v2 default/batch/stress seed 0")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-scvelo", action="store_true", help="also run O4; required only for v2_default_s0")
    parser.add_argument("--no-v1-references", action="store_true")
    args = parser.parse_args()
    paths = args.dataset or default_paths(include_references=not args.no_v1_references)
    rows: list[dict[str, object]] = []
    for path in paths:
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        run_scvelo = bool(args.run_scvelo and path.name == "v2_default_s0.h5ad")
        rows.extend(audit_one(path, run_scvelo=run_scvelo))
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "oracle", "median_cosine", "n_valid", "threshold", "raw_pass", "gate_status", "details_json", "wall_seconds", "peak_rss_bytes", "path"]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(args.output.relative_to(ROOT)), "rows": rows}, indent=2))
    failed = [row for row in rows if row["gate_status"] == "fail"]
    if failed:
        raise SystemExit(f"recoverability audit gates failed: {failed}")


if __name__ == "__main__":
    main()
