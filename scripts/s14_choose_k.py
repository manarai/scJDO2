#!/usr/bin/env python3
"""orient/S14 Part A: choose PCA dimension k from observed inputs only.

This module deliberately separates its phases.  ``preflight``, ``prepare``,
``prepare-larry``, ``unit``, and ``select`` read only observed counts,
coordinates, sampling-day/time, and barcodes.  They never open a synthetic truth
field and emit no AUC.  ``diagnostic`` is a separately-invoked, post-selection
scoring stage; it refuses to run until the immutable truth-free selection record
exists.

Every compute action is designed to be launched as one detached timed command by
the caller.  Checkpoints record resource and input provenance under S14 only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

OUT = ROOT / "benchmarks/synthetic_multimodal/s14"
ARR = OUT / "arrays"
CK = OUT / "checkpoints"
LOG = OUT / "logs"
FIG = ROOT / "benchmarks/figures/orient/s14"
DATA = ROOT / "benchmarks/synthetic_multimodal/data/v3"
MANIFEST = ROOT / "benchmarks/synthetic_multimodal/MANIFEST_v3.csv"
S12ARR = ROOT / "benchmarks/synthetic_multimodal/data/s12"
S12CK = ROOT / "benchmarks/synthetic_multimodal/s12/checkpoints"
S10ARR = ROOT / "benchmarks/synthetic_multimodal/data/s10"
S13ARR = ROOT / "benchmarks/synthetic_multimodal/s13/arrays"

TASK = "orient/S14"
MEM_STOP = 24 * 1024 ** 3
KS = (5, 10, 15, 20, 30)
DEV_SYN = (("v3_transition", 0), ("v3_transition", 1),
           ("v3_default", 0), ("v3_default", 1))
DEV_DATASETS = tuple(f"{p}_s{s}" for p, s in DEV_SYN) + ("larry",)
REFERENCE = {
    "eps": 0.05,
    "last": "extrapolate",
    "hidden": [96, 96, 96],
    "wd": 1e-4,
    "ensemble": 1,
    "space": "whitened",
}


def utc(value: float | None = None) -> str:
    return datetime.fromtimestamp(time.time() if value is None else value, timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def cpu_seconds() -> float:
    a = resource.getrusage(resource.RUSAGE_SELF)
    b = resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime)


def rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def checkpoint(name: str, started: float, cpu0: float, payload: dict[str, Any]) -> dict[str, Any]:
    peak = rss_bytes()
    if peak > MEM_STOP:
        raise RuntimeError(f"MEMORY STOP: {peak / 1024 ** 3:.3f} GiB exceeds 24 GiB")
    record = {
        "task": TASK,
        "unit": name,
        "started_utc": utc(started),
        "finished_utc": utc(),
        "wall_seconds": round(time.time() - started, 3),
        "in_process_cpu_seconds": round(cpu_seconds() - cpu0, 3),
        "peak_rss_bytes": peak,
        **payload,
    }
    write_json(CK / f"{name}.json", record)
    return record


def module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def s12() -> Any:
    return module("s14_readonly_s12", SCRIPTS / "s12_step0.py")


def s13() -> Any:
    return module("s14_readonly_s13", SCRIPTS / "s13_whitened.py")


def s06() -> Any:
    return module("s14_readonly_s06", SCRIPTS / "s06_user_runs.py")


def s11() -> Any:
    return module("s14_readonly_s11", SCRIPTS / "s11_velocities.py")


def set_threads() -> None:
    n = min(12, int(os.environ.get("OMP_NUM_THREADS", "8")))
    os.environ["OMP_NUM_THREADS"] = str(n)
    try:
        import torch
        torch.set_num_threads(n)
        try:
            torch.set_num_interop_threads(min(4, n))
        except RuntimeError:
            pass
    except Exception:
        pass


def sid(preset: str, seed: int) -> str:
    return f"{preset}_s{int(seed)}"


def synthetic_id_parts(dataset: str) -> tuple[str, int]:
    if "_s" not in dataset:
        raise ValueError(f"not a synthetic dataset id: {dataset}")
    preset, raw_seed = dataset.rsplit("_s", 1)
    return preset, int(raw_seed)


def prepared_path(dataset: str) -> Path:
    return ARR / "prepared" / f"{dataset}.npz"


def choose_array_path(dataset: str, k: int) -> Path:
    return ARR / "choose" / f"{dataset}_k{k}.npz"


def unit_name(dataset: str, k: int) -> str:
    return f"choose_{dataset}_k{k}"


def require_preflight() -> dict[str, Any]:
    path = CK / "preflight.json"
    if not path.exists() or not read_json(path).get("passed"):
        raise RuntimeError("run S14 preflight successfully first")
    return read_json(path)


def manifest_row(preset: str, seed: int, verify: bool = True) -> dict[str, str]:
    return s12().manifest_row(preset, seed, verify=verify)


def _observed_pca50(src: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """S12's observed-spliced embed_linear path, PCA-50 only (never scVI)."""
    import anndata as ad
    import pandas as pd
    from scjdo2.data.synthetic_multimodal import embed_linear

    spliced = np.asarray(src.layers["spliced"].toarray() if hasattr(src.layers["spliced"], "toarray") else src.layers["spliced"], dtype=np.float32)
    work = ad.AnnData(X=spliced.copy(), obs=pd.DataFrame(index=src.obs_names.copy()), var=pd.DataFrame(index=src.var_names.copy()))
    work.layers["spliced"] = spliced
    embed_linear(work, "pca", n_comp=50)
    meta = work.uns["embedding_pca"]
    return (np.asarray(work.obsm["X_pca"], np.float32),
            np.asarray(meta["linear_map"], np.float64),
            np.asarray(meta["mean"], np.float64),
            float(meta["normalization_target"]))


def _s12_pca50_if_verified(preset: str, seed: int, input_sha: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, Any]] | None:
    path = S12ARR / sid(preset, seed) / "embeddings.npz"
    check = S12CK / f"embed_{preset}_s{seed}.json"
    if not path.exists() or not check.exists():
        return None
    record = read_json(check)
    if record.get("input_sha256") != input_sha or record.get("sha256") != sha256(path):
        raise RuntimeError(f"CHECKSUM FAILURE: S12 PCA-50 reuse artifact for {preset} s{seed}")
    with np.load(path, allow_pickle=False) as arr:
        required = {"z_pca50", "linear_map_pca50", "mean_pca50", "target_pca50"}
        if not required <= set(arr.files):
            raise RuntimeError(f"S12 PCA-50 artifact has incompatible keys: {path}")
        return (np.asarray(arr["z_pca50"], np.float32),
                np.asarray(arr["linear_map_pca50"], np.float64),
                np.asarray(arr["mean_pca50"], np.float64),
                float(arr["target_pca50"]),
                {"source": "S12 PCA-50", "path": str(path.relative_to(ROOT)), "sha256": sha256(path)})


def _strict_pseudotime(src: Any, preset: str, seed: int, names: np.ndarray, days: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Reuse S10 strict day-first r0 when present; otherwise run the fixed S10/S06 call."""
    old = S10ARR / sid(preset, seed) / "palantir.npz"
    first = int(np.min(days))
    key = f"day{first}_strict_r0"
    if old.exists():
        with np.load(old, allow_pickle=False) as arr:
            if key in arr and np.array_equal(arr["names"].astype(str), names.astype(str)):
                # S10 records all five deterministic candidates in its immutable
                # checkpoint.  Surface them in S14 so Part C can report the
                # strict r0 provenance alongside the newly-computed test runs.
                s10_ck = ROOT / "benchmarks" / "synthetic_multimodal" / "s10" / "checkpoints" / f"palantir_{preset}_s{seed}.json"
                provenance: dict[str, Any] = {}
                if s10_ck.exists():
                    prior = read_json(s10_ck)
                    strict = [r for r in prior.get("runs", []) if r.get("unit") == key]
                    group = [r for r in prior.get("runs", []) if r.get("mode") == "strict" and r.get("root_group") == f"day{first}"]
                    if strict:
                        provenance["given_root"] = str(strict[0]["given_cell"])
                        provenance["actual_start_cell"] = str(strict[0]["actual_start_cell"])
                    if group:
                        provenance["candidate_roots"] = [str(r["given_cell"]) for r in sorted(group, key=lambda r: int(r["root_index"]))]
                    provenance["s10_checkpoint"] = str(s10_ck.relative_to(ROOT))
                    provenance["s10_checkpoint_sha256"] = sha256(s10_ck)
                return np.asarray(arr[key], np.float64), {
                    "source": "reused S10 strict r0", "path": str(old.relative_to(ROOT)), "key": key,
                    "sha256": sha256(old), "root_rule": "default_rng(1000+seed).choice(earliest-day cells, 5, replace=False); strict r0",
                    **provenance,
                }
    # This is the exact S10 root draw and S06 strict mode, confined to a new
    # S14 checkpoint.  It consumes stored observed PCA coordinates and day only.
    import anndata as ad
    import pandas as pd
    roots = np.random.default_rng(1000 + int(seed)).choice(names[days == first], size=5, replace=False)
    given = str(roots[0])
    work = ad.AnnData(X=np.empty((len(names), 0), dtype=np.float32),
                      obs=pd.DataFrame({"day": days.astype(int)}, index=pd.Index(names.astype(str))),
                      obsm={"X_pca": np.asarray(src.obsm["X_pca"], np.float64)})
    output, details = s06().run_palantir(work, given, f"day{first}", "strict")
    return output.obs["palantir_pseudotime"].to_numpy(np.float64), {
        "source": "S14 fixed S06 strict r0", "key": key, "given_root": given,
        "candidate_roots": [str(root) for root in roots],
        "root_rule": "default_rng(1000+seed).choice(earliest-day cells, 5, replace=False); strict r0",
        "settings": {"n_components": 10, "knn": 30, "num_waypoints": 500, "seed": 0}, **details,
    }


def prepare_synthetic(preset: str, seed: int) -> None:
    """Prepare observed PCA-50 and strict pseudotime; no synthetic truth is opened."""
    import anndata as ad
    started, cpu0 = time.time(), cpu_seconds()
    row = manifest_row(preset, seed, verify=True)
    src = ad.read_h5ad(ROOT / row["path"])
    try:
        names = src.obs_names.to_numpy(str)
        days = src.obs["day"].to_numpy(int)
        hours = src.obs["time_hours"].to_numpy(np.float64)
        clone = src.obs["clone"].astype(object).to_numpy()
        clone_str = np.asarray(["" if value is None or value != value else str(value) for value in clone], dtype=str)
        reused = _s12_pca50_if_verified(preset, seed, row["sha256"])
        if reused is None:
            z50, pmap, pmean, target = _observed_pca50(src)
            pca_info: dict[str, Any] = {"source": "S14 observed S12 PCA code path", "path": None}
        else:
            z50, pmap, pmean, target, pca_info = reused
        pt, pt_info = _strict_pseudotime(src, preset, seed, names, days)
        dest = prepared_path(sid(preset, seed)); dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, names=names, day=days.astype(np.int16), hours=hours,
                            clone=clone_str, z_pca50=z50, pca_map=pmap, pca_mean=pmean,
                            normalization_target=np.float64(target), pseudotime_strict_r0=pt)
        checkpoint(f"prepare_{preset}_s{seed}", started, cpu0, {
            "part_a_truth_free": True,
            "input_path": row["path"], "input_sha256": row["sha256"],
            "prepared_path": str(dest.relative_to(ROOT)), "prepared_sha256": sha256(dest),
            "n_cells": int(len(names)), "pca50": pca_info, "strict_pseudotime": pt_info,
        })
    finally:
        src.file.close()


def prepare_larry() -> None:
    """Read S11's fixed sample and S13 common PCA-50 map in place; no raw mutation."""
    import anndata as ad
    from scipy import sparse
    started, cpu0 = time.time(), cpu_seconds()
    real = S10ARR / "real" / "larry.npz"
    maps_path = S13ARR / "larry_maps.npz"
    if not real.exists() or not maps_path.exists():
        raise RuntimeError("S10/S13 LARRY artifacts required for S14 reuse are missing")
    m = s11()
    raw_hash = m.check_input(m.LARRY_RAW)
    with np.load(real, allow_pickle=False) as base, np.load(maps_path, allow_pickle=False) as maps:
        idx = np.asarray(base["sample_index"], int)
        day = np.asarray(base["day"], np.int16)
        names = np.asarray(base["cell_id"], str)
        common_map = np.asarray(maps["common_map"], np.float64)
        common_mean = np.asarray(maps["common_mean"], np.float64)
        hvg = np.asarray(maps["hvg_index"], int)
        target = float(maps["normalization_target"])
        expected = np.concatenate([np.random.default_rng(0).choice(np.flatnonzero(np.asarray(base["day_all"], float) == d), 4000, replace=False) for d in (2.0, 4.0, 6.0)])
        # S11 draws consecutively from the same generator, not a fresh generator per day.
        rng = np.random.default_rng(0)
        expected = np.concatenate([rng.choice(np.flatnonzero(np.asarray(base["day_all"], float) == d), 4000, replace=False) for d in (2.0, 4.0, 6.0)])
        if not np.array_equal(idx, expected):
            raise RuntimeError("S11 fixed default_rng(0) LARRY sample mismatch")
        if not np.array_equal(np.asarray(maps["sample_index"], int), idx):
            raise RuntimeError("S13 LARRY common-map sample mismatch")
    raw = ad.read_h5ad(m.LARRY_RAW)
    try:
        library = np.asarray(raw.X.sum(axis=1)).ravel().astype(np.float64)
        X = m._transform_larry_rows(raw.X[:, hvg][idx], library[idx], target)
        z50 = ((X - common_mean[None, :]) @ common_map).astype(np.float32)
        clones = sparse.csr_matrix(raw.obsm["clone_matrix"][idx])
        dest = prepared_path("larry"); dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, names=names, day=day, hours=day.astype(np.float64) * 24.0,
                            z_pca50=z50, pca_map=common_map, pca_mean=common_mean,
                            normalization_target=np.float64(target), clone_indptr=clones.indptr,
                            clone_indices=clones.indices, clone_shape=np.asarray(clones.shape, dtype=np.int64))
        checkpoint("prepare_larry", started, cpu0, {
            "part_a_truth_free": True, "input_path": str(m.LARRY_RAW), "input_sha256": raw_hash,
            "s10_sample_path": str(real.relative_to(ROOT)), "s10_sample_sha256": sha256(real),
            "s13_map_path": str(maps_path.relative_to(ROOT)), "s13_map_sha256": sha256(maps_path),
            "prepared_path": str(dest.relative_to(ROOT)), "prepared_sha256": sha256(dest),
            "n_cells": int(len(idx)), "sample_per_day": 4000, "sample_seed": 0,
            "common_reference": "S13 PCA-50 on the fixed observed LARRY sample",
        })
    finally:
        raw.file.close()


def load_prepared(dataset: str, k: int) -> dict[str, Any]:
    if k not in KS:
        raise ValueError(f"k must be one of {KS}")
    path = prepared_path(dataset)
    if not path.exists():
        raise RuntimeError(f"prepare {dataset} first")
    with np.load(path, allow_pickle=False) as arr:
        out = {key: np.asarray(arr[key]) for key in arr.files}
    out["dataset"] = dataset
    out["path"] = path
    out["z"] = np.asarray(out["z_pca50"][:, :k], np.float64)
    out["linear_map"] = np.asarray(out["pca_map"][:, :k], np.float64)
    out["common_map"] = np.asarray(out["pca_map"], np.float64)
    return out


def fit_model(z: np.ndarray, v: np.ndarray, fit_rows: np.ndarray, hidden: tuple[int, ...], wd: float, seed: int) -> tuple[Any, dict[str, Any]]:
    from scjdo2.measured_drift import MeasuredDrift
    rows = np.asarray(fit_rows, int)
    vv = np.asarray(v, np.float32)[rows]
    good = np.isfinite(vv).all(axis=1)
    if int(good.sum()) < 10:
        raise RuntimeError("fewer than ten finite velocity rows")
    t0 = time.time()
    model = MeasuredDrift(np.asarray(z).shape[1], hidden=tuple(hidden), weight_decay=float(wd), seed=int(seed))
    model.fit(np.asarray(z, np.float32)[rows][good], vv[good], verbose=False)
    device = str(next(model._model.parameters()).device)
    model._model.to("cpu")
    return model, {"n_fit_cells": int(good.sum()), "hidden": list(hidden), "weight_decay": float(wd),
                   "seed": int(seed), "fit_device": device, "jacobian_device": "cpu",
                   "seconds": round(time.time() - t0, 3)}


def fit_flow(d: dict[str, Any], config: dict[str, Any], seeds: tuple[int, ...] | None = None) -> dict[str, Any]:
    """Fit OT in raw PCA-k then drift in configured raw or whitened coordinates."""
    set_threads()
    lib = s12()
    z = np.asarray(d["z"], np.float64)
    v, ot_info = lib.ot_velocity(z, np.asarray(d["day"], int), np.asarray(d["hours"], float),
                                 eps_factor=float(config["eps"]), backward_last=(config["last"] == "backward"))
    rows = np.flatnonzero(np.isfinite(v).all(axis=1))
    if config["space"] == "whitened":
        zw, W, WI, whitening = s13().whitening(z)
        fit_z, fit_v = zw, np.asarray(v, np.float64) @ W
    elif config["space"] == "raw":
        fit_z, fit_v = z, np.asarray(v, np.float64)
        W, WI = np.eye(z.shape[1]), np.eye(z.shape[1])
        whitening = {"status": "raw coordinate reference"}
    else:
        raise ValueError(f"unknown coordinate space: {config['space']}")
    use_seeds = tuple(range(int(config.get("ensemble", 1)))) if seeds is None else tuple(seeds)
    models, Js, vs, infos = [], [], [], []
    for seed in use_seeds:
        model, info = fit_model(fit_z, fit_v, rows, tuple(config["hidden"]), float(config["wd"]), int(seed))
        models.append(model)
        Js.append(np.asarray(model.jacobian_autodiff(np.asarray(fit_z, np.float32)), np.float32))
        vs.append(np.asarray(model.velocity(np.asarray(fit_z, np.float32)), np.float32))
        infos.append(info)
    return {
        "z": z, "fit_z": fit_z, "v_ot_raw": np.asarray(v, np.float32), "fit_rows": rows,
        "W": np.asarray(W, np.float64), "WI": np.asarray(WI, np.float64), "whitening": whitening,
        "models": models, "Js": Js, "J": np.mean(Js, axis=0, dtype=np.float64).astype(np.float32),
        "velocity_fit": np.mean(vs, axis=0, dtype=np.float64).astype(np.float32),
        "ot_info": ot_info, "model_info": infos, "config": dict(config),
    }


def predicted_velocity_raw(flow: dict[str, Any], x_raw: np.ndarray) -> np.ndarray:
    x = np.asarray(x_raw, np.float64)
    if flow["config"]["space"] == "whitened":
        fit_x = (x - np.asarray(flow["z"], np.float64).mean(axis=0)) @ flow["W"]
        vv = np.mean([np.asarray(model.velocity(np.asarray(fit_x, np.float32)), np.float64) for model in flow["models"]], axis=0)
        return vv @ flow["WI"]
    return np.mean([np.asarray(model.velocity(np.asarray(x, np.float32)), np.float64) for model in flow["models"]], axis=0)


def integrate_displacement(flow: dict[str, Any], x0: np.ndarray, gap_hours: float) -> np.ndarray:
    n_steps = int(round(float(gap_hours)))
    if n_steps < 1 or abs(n_steps - float(gap_hours)) > 1e-8:
        raise RuntimeError(f"Euler gap must be a positive integer number of hours, got {gap_hours}")
    x = np.asarray(x0, np.float64).copy()
    for _ in range(n_steps):
        x += predicted_velocity_raw(flow, x)
    return x - np.asarray(x0, np.float64)


def _lineage_pairs(d: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Observed clone-barcode pairs: current rows, future centroids, and their gap hours."""
    z, days, hours = np.asarray(d["z"], np.float64), np.asarray(d["day"], int), np.asarray(d["hours"], float)
    current, targets, gaps = [], [], []
    unique = np.unique(days)
    if "clone" in d:
        clone = np.asarray(d["clone"], str)
        for d0, d1 in zip(unique[:-1], unique[1:]):
            later = np.flatnonzero((days == d1) & (clone != ""))
            centroids: dict[str, np.ndarray] = {}
            for c in np.unique(clone[later]):
                centroids[c] = z[later[clone[later] == c]].mean(axis=0)
            earlier = np.flatnonzero((days == d0) & (clone != ""))
            for i in earlier:
                c = clone[i]
                if c in centroids:
                    current.append(int(i)); targets.append(centroids[c]); gaps.append(float(np.median(hours[days == d1]) - np.median(hours[days == d0])))
    else:
        indptr, indices = np.asarray(d["clone_indptr"], int), np.asarray(d["clone_indices"], int)
        for d0, d1 in zip(unique[:-1], unique[1:]):
            later_by_barcode: dict[int, list[int]] = {}
            for i in np.flatnonzero(days == d1):
                for barcode in indices[indptr[i]:indptr[i + 1]]:
                    later_by_barcode.setdefault(int(barcode), []).append(int(i))
            gap = float(np.median(hours[days == d1]) - np.median(hours[days == d0]))
            for i in np.flatnonzero(days == d0):
                for barcode in indices[indptr[i]:indptr[i + 1]]:
                    later = later_by_barcode.get(int(barcode))
                    if later:
                        current.append(int(i)); targets.append(z[later].mean(axis=0)); gaps.append(gap)
    if not current:
        raise RuntimeError("no barcoded earlier-day cells have same-clone later-day cells")
    return np.asarray(current, int), np.asarray(targets, np.float64), np.asarray(gaps, np.float64)


def lineage_agreement(d: dict[str, Any], flow: dict[str, Any]) -> dict[str, Any]:
    from scipy.stats import spearmanr
    rows, target, gaps = _lineage_pairs(d)
    z = np.asarray(d["z"], np.float64)
    pred = np.empty_like(target)
    for gap in np.unique(gaps):
        take = np.flatnonzero(gaps == gap)
        pred[take] = integrate_displacement(flow, z[rows[take]], float(gap))
    actual = target - z[rows]
    observed = float(spearmanr(pred.reshape(-1), actual.reshape(-1)).statistic)
    rng = np.random.default_rng(0)
    nulls = [float(spearmanr(pred[rng.permutation(len(pred))].reshape(-1), actual.reshape(-1)).statistic) for _ in range(5)]
    return {"lineage_spearman": observed, "lineage_nulls": nulls, "lineage_null_max": float(np.max(nulls)),
            "n_lineage_cell_barcode_pairs": int(len(rows)), "n_lineage_dimensions_pooled": int(pred.size),
            "definition": "Spearman across all PCA-coordinate components of observed clone-cell pairs; Euler integration uses 1-hour steps over the observed sampling gap"}


def stability(lambdas: list[np.ndarray]) -> dict[str, Any]:
    from scipy.stats import spearmanr
    if len(lambdas) != 3:
        raise ValueError("S14 stability requires drift seeds 0, 1, 2")
    values = [float(spearmanr(lambdas[i], lambdas[j]).statistic) for i, j in ((0, 1), (0, 2), (1, 2))]
    return {"lambda_seed_spearman_pairs": values, "lambda_seed_stability_median": float(np.median(values)),
            "definition": "median of the three whole-cell Spearman correlations of lambda1 between drift seeds 0, 1, 2"}


def choose_unit(dataset: str, k: int) -> None:
    """Truth-free one-dataset-by-k Part A unit."""
    require_preflight()
    started, cpu0 = time.time(), cpu_seconds()
    d = load_prepared(dataset, k)
    flow = fit_flow(d, REFERENCE, seeds=(0, 1, 2))
    lib = s12()
    # Extract per-seed spectra before averaging.  Each fit is observed-only.
    lambdas = [np.asarray(lib.spectra(J)["lam1"], np.float32) for J in flow["Js"]]
    line = lineage_agreement(d, flow)
    stable = stability(lambdas)
    z = np.asarray(d["z"], np.float64)
    zw, W, WI, _ = s13().whitening(z)
    held, held_info = s13().heldout(lib, z, zw, WI, z.mean(axis=0), np.asarray(d["day"], int),
                                    np.asarray(d["hours"], float), np.asarray(d["linear_map"], float), np.asarray(d["common_map"], float))
    dest = choose_array_path(dataset, k); dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, lambda_seed0=lambdas[0], lambda_seed1=lambdas[1], lambda_seed2=lambdas[2])
    checkpoint(unit_name(dataset, k), started, cpu0, {
        "part_a_truth_free": True, "dataset": dataset, "k": int(k), "n_cells": int(len(z)),
        "prepared_path": str(Path(d["path"]).relative_to(ROOT)), "prepared_sha256": sha256(Path(d["path"])),
        "reference_config": REFERENCE, "lineage": line, "stability": stable,
        "heldout_day": held, "heldout_details": held_info,
        "array_path": str(dest.relative_to(ROOT)), "array_sha256": sha256(dest),
        "ot_pairs": flow["ot_info"], "models": flow["model_info"],
    })


def _choose_records() -> list[dict[str, Any]]:
    records = []
    missing = []
    for dataset in DEV_DATASETS:
        for k in KS:
            path = CK / f"{unit_name(dataset, k)}.json"
            if not path.exists():
                missing.append(path.name)
            else:
                record = read_json(path)
                if not record.get("part_a_truth_free"):
                    raise RuntimeError(f"Part A record lacks truth-free provenance: {path}")
                records.append(record)
    if missing:
        raise RuntimeError("missing Part A units: " + ", ".join(missing))
    return records


def select_k() -> None:
    """Apply the fixed observed-only choice rule before any diagnostic AUC runs."""
    import pandas as pd
    started, cpu0 = time.time(), cpu_seconds()
    records = _choose_records()
    rows = []
    for record in records:
        line, stable = record["lineage"], record["stability"]
        held = {x["method"]: x["energy_distance"] for x in record["heldout_day"]}
        rows.append({"dataset": record["dataset"], "k": int(record["k"]),
                     "lineage_spearman": line["lineage_spearman"], "lineage_null_max": line["lineage_null_max"],
                     "lineage_above_null": bool(line["lineage_spearman"] > line["lineage_null_max"]),
                     "seed_stability_median": stable["lambda_seed_stability_median"],
                     "heldout_flow_energy": held.get("whitened_flow"),
                     "heldout_no_movement_energy": held.get("no_movement"),
                     "heldout_mean_displacement_energy": held.get("mean_displacement"),
                     "n_cells": record["n_cells"]})
    frame = pd.DataFrame(rows).sort_values(["k", "dataset"])
    summary = frame.groupby("k", as_index=False).agg(
        mean_lineage_agreement=("lineage_spearman", "mean"),
        mean_null_max=("lineage_null_max", "mean"),
        mean_seed_stability=("seed_stability_median", "mean"),
        n_datasets=("dataset", "nunique"),
    )
    if not (summary["n_datasets"] == len(DEV_DATASETS)).all():
        raise RuntimeError("each k must contain every development dataset exactly once")
    best = float(summary["mean_lineage_agreement"].max())
    tied = summary[summary["mean_lineage_agreement"] >= best - 0.02].sort_values(["mean_seed_stability", "k"], ascending=[False, True])
    k_star = int(tied.iloc[0]["k"])
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "choose_k_by_dataset.csv", index=False)
    summary.to_csv(OUT / "choose_k.csv", index=False)
    selection = {
        "k_star": k_star,
        "rule": "highest mean lineage agreement over five equally-weighted development datasets; candidates within 0.02 are broken by higher mean seed stability, then lower k deterministically",
        "best_mean_lineage_agreement": best,
        "within_0_02_candidates": tied.to_dict(orient="records"),
        "development_datasets": list(DEV_DATASETS),
        "truth_free": True,
        "truth_free_inputs": ["observed spliced counts", "sampling day/time", "barcodes", "S10/S12/S13 observed PCA artifacts"],
        "truth_keys_absent_from_selection": ["z_true", "fate_true", "jacobian_state", "ds_dt_true", "z1_crit"],
        "selection_locked_utc": utc(),
    }
    write_json(OUT / "k_selection.json", selection)
    checkpoint("select_k", started, cpu0, {"part_a_truth_free": True, **selection,
                                             "choose_k_csv": str((OUT / "choose_k.csv").relative_to(ROOT)),
                                             "choose_k_by_dataset_csv": str((OUT / "choose_k_by_dataset.csv").relative_to(ROOT))})
    print(json.dumps(jsonable(selection), indent=2))


def select_k_revision1() -> None:
    """Apply Revision 1's stability gate to the locked observed-only table.

    This is deliberately a table-only recomputation: it reads the committed
    Part A ``choose_k.csv`` and does not load a dataset or refit an operator.
    The task amendment was adopted after development AUCs were visible; that
    disclosure is retained in the selection provenance while test seeds remain
    absent from this stage.
    """
    import pandas as pd

    started, cpu0 = time.time(), cpu_seconds()
    source = OUT / "choose_k.csv"
    if not source.exists():
        raise RuntimeError("Revision 1 requires the locked Part A choose_k.csv")
    summary = pd.read_csv(source)
    required = {"k", "mean_lineage_agreement", "mean_seed_stability", "n_datasets"}
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise RuntimeError("choose_k.csv lacks required Revision 1 fields: " + ", ".join(missing))
    if set(summary["k"].astype(int)) != set(KS):
        raise RuntimeError("choose_k.csv does not contain the locked candidate k values")
    if not (summary["n_datasets"].astype(int) == len(DEV_DATASETS)).all():
        raise RuntimeError("choose_k.csv does not contain all development datasets for every k")

    summary = summary.sort_values("k").reset_index(drop=True)
    eligible = summary[summary["mean_seed_stability"] >= 0.5].copy()
    fallback = len(eligible) == 0
    pool = eligible if not fallback else summary
    best = float(pool["mean_lineage_agreement"].max())
    tied = pool[pool["mean_lineage_agreement"] >= best - 0.02].sort_values(
        ["mean_seed_stability", "k"], ascending=[False, True]
    )
    k_star = int(tied.iloc[0]["k"])
    selection = {
        "revision": 1,
        "k_star": k_star,
        "rule": "among k with mean seed stability >= 0.5, highest mean lineage agreement; ties within 0.02 by higher seed stability, then lower k; if none pass, use the most stable k",
        "stability_threshold": 0.5,
        "fallback_used": fallback,
        "all_candidates": summary.to_dict(orient="records"),
        "eligible_candidates": eligible.to_dict(orient="records"),
        "within_0_02_candidates": tied.to_dict(orient="records"),
        "development_datasets": list(DEV_DATASETS),
        "truth_free": True,
        "truth_free_inputs": ["locked Part A observed-only choose_k.csv"],
        "truth_keys_absent_from_selection": ["z_true", "fate_true", "jacobian_state", "ds_dt_true", "z1_crit"],
        "amendment_disclosure": "Rule fixed after development fork AUCs had been seen; development data are tuning data and test seeds were not accessed by this recomputation.",
        "source_choose_k_csv": str(source.relative_to(ROOT)),
        "source_choose_k_sha256": sha256(source),
        "selection_locked_utc": utc(),
    }
    dest = OUT / "k_selection_r1.json"
    write_json(dest, selection)
    checkpoint("select_k_r1", started, cpu0, {
        "part_a_truth_free": True,
        "no_refits": True,
        **selection,
        "k_selection_r1": str(dest.relative_to(ROOT)),
    })
    print(json.dumps(jsonable(selection), indent=2))

def diagnostic_auc() -> None:
    """Post-selection development AUC diagnostic only; never called by selection."""
    import anndata as ad
    import pandas as pd
    selection_path = OUT / "k_selection.json"
    if not selection_path.exists():
        raise RuntimeError("truth-free k selection must be committed before diagnostic AUC")
    selection = read_json(selection_path)
    if not selection.get("truth_free"):
        raise RuntimeError("k selection record is not truth-free")
    started, cpu0 = time.time(), cpu_seconds()
    lib = s12()
    rows = []
    for preset, seed in DEV_SYN:
        row = manifest_row(preset, seed, verify=True)
        src = ad.read_h5ad(ROOT / row["path"], backed="r")
        try:
            truth = lib.truth_for_scoring(src)  # scoring begins only after selection is fixed
            classes = lib.truth_classes(truth["z"], truth["z1_crit"])
            for k in KS:
                with np.load(choose_array_path(sid(preset, seed), k), allow_pickle=False) as arr:
                    lam = np.asarray(arr["lambda_seed0"], np.float64)
                rows.append({"dataset": sid(preset, seed), "preset": preset, "seed": seed, "k": k,
                             "auc_lam1": float(lib.auc(lam, classes)), "n_saddle": int(classes["saddle"].sum()),
                             "n_committed": int(classes["committed"].sum()),
                             "selection_k_star": int(selection["k_star"]), "diagnostic_only": True})
        finally:
            src.file.close()
    frame = pd.DataFrame(rows).sort_values(["preset", "seed", "k"])
    frame.to_csv(OUT / "dev_auc_by_k.csv", index=False)
    checkpoint("diagnostic_dev_auc", started, cpu0, {
        "post_selection_diagnostic": True, "k_selection_sha256": sha256(selection_path),
        "k_star": int(selection["k_star"]), "n_rows": int(len(frame)),
        "path": str((OUT / "dev_auc_by_k.csv").relative_to(ROOT)),
    })


def preflight() -> None:
    """Truth-free integrity and conservative 24 CPU-hour admission plan."""
    started, cpu0 = time.time(), cpu_seconds()
    lib = s12()
    inputs: dict[str, Any] = {}
    for preset, seed in DEV_SYN + (("v3_transition", 2), ("v3_default", 2)):
        row = lib.manifest_row(preset, seed, verify=True)
        inputs[sid(preset, seed)] = {"path": row["path"], "sha256": row["sha256"]}
    m = s11()
    larry_hash = m.check_input(m.LARRY_RAW)
    maps = S13ARR / "larry_maps.npz"
    if not maps.exists():
        raise RuntimeError("required S13 LARRY common PCA-50 map is missing")
    # Conservative admission: full v3 generation is charged at the observed S10
    # maximum, every network fit at the S13 PCA-50 maximum; ensemble multipliers
    # and preparation/Palantir allowance are explicit.
    s10 = list(csv.DictReader((ROOT / "benchmarks/synthetic_multimodal/s10/cpu_ledger.csv").open()))
    gen_cpu = max(float(r["cpu_s"]) for r in s10 if "generate_v3_transition" in r["log"])
    s13_rows = list(csv.DictReader((ROOT / "benchmarks/synthetic_multimodal/s13/cpu_ledger.csv").open()))
    fit_cpu = max(float(r["cpu_s"]) for r in s13_rows if "unit_v3_transition" in r["log"] and "pca50" in r["log"])
    n_generation = 6  # transition s3,s4; default s3; nofork s0,s1,s2
    n_part_a_model_fits = len(DEV_DATASETS) * len(KS) * 4  # seeds 0/1/2 + heldout fit
    n_oat = 15 * len(DEV_SYN) + len(DEV_SYN)  # 15 unique OAT + one combined per synthetic dev dataset
    n_test = 50  # selected/reference × requested k bands, nofork, report-only, and LARRY
    prep_allowance = 12 * 180.0 + 6 * 240.0 + 3600.0
    projected = n_generation * gen_cpu + (n_part_a_model_fits + n_oat + n_test) * fit_cpu + prep_allowance
    plan = {"basis": "S10 max full-v3 generation + S13 max PCA-50 operator unit; explicit one-network fit equivalents",
            "generation_units": n_generation, "generation_cpu_seconds_each": gen_cpu,
            "part_a_model_fit_equivalents": n_part_a_model_fits,
            "optimisation_and_combined_fit_equivalents": n_oat,
            "test_fit_equivalents": n_test, "operator_cpu_seconds_each": fit_cpu,
            "preparation_and_aggregation_allowance_seconds": prep_allowance,
            "projected_cpu_hours": projected / 3600.0, "budget_cpu_hours": 24.0,
            "admitted": bool(projected <= 24.0 * 3600.0)}
    if not plan["admitted"]:
        raise RuntimeError(f"COMPUTE STOP: projected {plan['projected_cpu_hours']:.3f} CPU-h exceeds 24")
    write_json(OUT / "admission_plan.json", plan)
    checkpoint("preflight", started, cpu0, {"passed": True, "part_a_truth_free": True,
                                              "synthetic_inputs": inputs, "larry_input_sha256": larry_hash,
                                              "s13_larry_map": {"path": str(maps.relative_to(ROOT)), "sha256": sha256(maps)},
                                              "admission_plan": plan, "threads_limit": 12, "memory_stop_gib": 24.0})
    print(json.dumps(jsonable(plan), indent=2))


def ledger() -> None:
    """Parse closed /usr/bin/time -l logs into an auditable task-local CPU ledger."""
    import re
    rows = []
    for path in sorted(LOG.glob("*.log")):
        text = path.read_text(errors="ignore")
        timing = re.search(r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys", text)
        started = re.search(r"^START (\S+)", text, re.M)
        ended = re.search(r"^EXIT (\d+) (\S+)", text, re.M)
        peak = re.search(r"(\d+)\s+maximum resident set size", text)
        rows.append({"log": str(path.relative_to(ROOT)), "start_utc": started.group(1) if started else "",
                     "end_utc": ended.group(2) if ended else "", "exit": int(ended.group(1)) if ended else "",
                     "wall_s": float(timing.group(1)) if timing else "",
                     "cpu_s": round(float(timing.group(2)) + float(timing.group(3)), 3) if timing else "",
                     "peak_rss_gb": round(int(peak.group(1)) / 1e9, 3) if peak else ""})
    if not rows:
        raise RuntimeError("no closed S14 timed logs")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "cpu_ledger.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    total = sum(float(row["cpu_s"]) for row in rows if row["cpu_s"] != "")
    print(json.dumps({"n_logs": len(rows), "cpu_hours": total / 3600.0}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "prepare", "prepare-larry", "unit", "select", "select-r1", "diagnostic", "ledger"))
    parser.add_argument("--preset")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset")
    parser.add_argument("--k", type=int, choices=KS)
    args = parser.parse_args()
    if args.stage == "preflight":
        preflight()
    elif args.stage == "prepare":
        if args.preset is None or args.seed is None:
            parser.error("prepare requires --preset and --seed")
        prepare_synthetic(args.preset, args.seed)
    elif args.stage == "prepare-larry":
        prepare_larry()
    elif args.stage == "unit":
        if args.dataset is None or args.k is None:
            parser.error("unit requires --dataset and --k")
        choose_unit(args.dataset, args.k)
    elif args.stage == "select":
        select_k()
    elif args.stage == "select-r1":
        select_k_revision1()
    elif args.stage == "diagnostic":
        diagnostic_auc()
    else:
        ledger()


if __name__ == "__main__":
    main()
