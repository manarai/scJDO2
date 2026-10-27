#!/usr/bin/env python3
"""orient/S10 Step 0 and Step 1: v3_transition generation, S08 statistics, fits and user runs.

Stages (one detached, timed command per unit; each writes its own checkpoint):

* ``generate --preset v3_transition --seed N``  Step 0 dataset + MANIFEST_v3 row.
* ``s08 --preset P --seed N``                   Step 0 S08 statistics (imported S08 functions).
* ``fit --preset P --seed N --method M``        Step 1 energy drift + J, four labelling-velocity
                                                variants (Step 0b), labelling flow operators.
* ``scvelo --preset P --seed N``                Step 1 scVelo (exact S03 block, imported from S06).
* ``palantir --preset P --seed N``              Step 1 Palantir: 5 day-2 + 5 latest-day roots x 2 modes
                                                (imported S06 functions); stationary cycle: one
                                                random-cell root (Step 4 null).
* ``flow --preset P --seed N --method M``       Step 1 flow operators from scVelo and from OT
                                                (balanced; growth-weighted sensitivity).

Only observed layers/metadata are passed to any fit: ``new``, ``total``,
``spliced``/``unspliced`` (scVelo), ``pulse_hours``, ``day``, stored embeddings.
Truth (``*_true``, ``z_true``, ``capture_factor``, ``batch_day``,
``batch_gene_effects``, ``core_dims``...) is never read here, except the true
growth used by the growth-weighted OT sensitivity (task Step 1: scoring-side
knowledge, report only) — which is the generator's division law, not a
per-cell truth array.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
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
DATA_V3 = ROOT / "benchmarks/synthetic_multimodal/data/v3"
MANIFEST = ROOT / "benchmarks/synthetic_multimodal/MANIFEST_v3.csv"
OUT = ROOT / "benchmarks/synthetic_multimodal/s10"
ARR = ROOT / "benchmarks/synthetic_multimodal/data/s10"          # large outputs (gitignored)
CKPT = OUT / "checkpoints"
LOCKS = OUT / ".locks"
FIG = ROOT / "benchmarks/figures/orient/s10"
PRESETS = ("v3_default", "v3_stress", "v3_batch", "v3_transition", "v3_stationary_cycle")
DEVELOPMENTAL = ("v3_default", "v3_stress", "v3_batch", "v3_transition")
METHODS = ("pca", "fa")
VARIANTS = ("raw_shared", "smoothed_shared", "raw_perpulse", "smoothed_perpulse")
PRIMARY_VARIANT = "smoothed_shared"
GATE_VARIANT = "raw_shared"            # gates fixed on "raw labelling" (S06 definition)
KNN = 30


# ---------------------------------------------------------------------------
# common helpers (imported by the other s10_* scripts)
# ---------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def utc(t: float | None = None) -> str:
    return datetime.fromtimestamp(time.time() if t is None else t, timezone.utc).isoformat().replace("+00:00", "Z")


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


@contextlib.contextmanager
def file_lock(name: str):
    LOCKS.mkdir(parents=True, exist_ok=True)
    with (LOCKS / f"{name}.lock").open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def cpu_seconds() -> float:
    a = resource.getrusage(resource.RUSAGE_SELF)
    b = resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime)


def manifest_rows() -> list[dict[str, str]]:
    with MANIFEST.open(newline="") as handle:
        return list(csv.DictReader(handle))


def manifest_row(preset: str, seed: int, verify: bool = True) -> dict[str, str]:
    for row in manifest_rows():
        if row["preset"] == preset and int(row["seed"]) == int(seed):
            if verify:
                path = ROOT / row["path"]
                if not path.is_file() or sha256(path) != row["sha256"]:
                    raise RuntimeError(f"CHECKSUM FAILURE: {path} does not match MANIFEST_v3 sha256")
            return row
    raise RuntimeError(f"{preset} seed {seed} has no MANIFEST_v3 row")


def dataset_path(preset: str, seed: int) -> Path:
    return ROOT / manifest_row(preset, seed, verify=False)["path"]


def unit_dir(preset: str, seed: int) -> Path:
    d = ARR / f"{preset}_s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def checkpoint(name: str, started_wall: float, started_cpu: float, payload: dict[str, Any]) -> dict[str, Any]:
    record = {"task": "orient/S10", "unit": name, "started_utc": utc(started_wall), "finished_utc": utc(),
              "wall_seconds": round(time.time() - started_wall, 3),
              "in_process_cpu_seconds": round(cpu_seconds() - started_cpu, 3),
              "peak_rss_bytes": rss_bytes(), **payload}
    write_json(CKPT / f"{name}.json", record)
    return record


def dense(x: Any, dtype: Any = np.float32) -> np.ndarray:
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=dtype)


# ---------------------------------------------------------------------------
# Step 0: generation and S08 statistics
# ---------------------------------------------------------------------------
def stage_generate(preset: str, seed: int) -> None:
    if preset != "v3_transition":
        raise SystemExit("Step 0 generates v3_transition only; the other v3 datasets are inputs")
    started, cpu0 = time.time(), cpu_seconds()
    writer = load_module("s10_writer", SCRIPTS / "make_synthetic_multimodal.py")
    row = writer.write_dataset(preset, seed, DATA_V3)
    with file_lock("manifest"):
        rows = [r for r in manifest_rows() if not (r["preset"] == preset and int(r["seed"]) == seed)]
        rows.append({k: str(v) for k, v in row.items()})
        order = {p: i for i, p in enumerate(("v3_default", "v3_stress", "v3_batch", "v3_stationary_cycle", "v3_transition"))}
        rows.sort(key=lambda r: (order.get(r["preset"], 99), int(r["seed"])))
        writer.write_manifest(rows, MANIFEST)
    checkpoint(f"generate_{preset}_s{seed}", started, cpu0, {"manifest_row": row})


def stage_s08(preset: str, seed: int) -> None:
    """S08's imported statistic functions on one v3 dataset, in a private S10 runtime copy."""
    started, cpu0 = time.time(), cpu_seconds()
    row = manifest_row(preset, seed)
    meas = load_module("s10_s09_measure", SCRIPTS / "s09_measure.py")
    meas.RUNTIME = ARR / "s08_runtime"          # private S08 runtime copy (gitignored data folder)
    meas.CHECKPOINTS = ARR / "s08_runtime" / "checkpoints"
    meas.LOCKS = LOCKS
    result = meas.run_s08_for_dataset(ROOT / row["path"], f"{preset}_s{seed}")
    checkpoint(f"s08_{preset}_s{seed}", started, cpu0, {"input_sha256": row["sha256"], "s08": result})


def stage_s08compare() -> None:
    """Report: v3_transition vs v3_default on the S09 13-target statistics and every S08 statistic."""
    import pandas as pd
    started, cpu0 = time.time(), cpu_seconds()
    cal = load_module("s10_s09_cal", SCRIPTS / "s09_calibrate.py")
    mine = pd.read_csv(ARR / "s08_runtime" / "stats.csv")
    s09 = pd.read_csv(ROOT / "benchmarks/synthetic_multimodal/v3_calibration/s08_runtime/stats.csv")     # read only
    rows = []
    for preset, table in (("v3_default", s09), ("v3_transition", mine)):
        for seed in range(3):
            key = f"{preset}_s{seed}"
            values = cal.statistic_values(table.loc[table["dataset"] == key, ["statistic", "feature", "value"]])
            for label, value in values.items():
                lo, hi = cal.INTERVALS_R5[label]
                rows.append({"preset": preset, "seed": seed, "statistic": label, "value": value, "target_lo": lo, "target_hi": hi,
                             "met": bool(pd.notna(value) and lo <= value <= hi)})
    t13 = pd.DataFrame(rows)
    t13.to_csv(OUT / "s08_transition_vs_default_targets.csv", index=False)
    scalar = []
    for preset, table in (("v3_default", s09), ("v3_transition", mine)):
        sub = table[table["dataset"].str.startswith(preset + "_s")]
        for (stat, feat), g in sub.groupby(["statistic", "feature"]):
            scalar.append({"preset": preset, "statistic": stat, "feature": feat, "median_over_seeds": float(g["value"].median()),
                           "min": float(g["value"].min()), "max": float(g["value"].max()), "n_seeds": int(g["dataset"].nunique())})
    allstats = pd.DataFrame(scalar).pivot_table(index=["statistic", "feature"], columns="preset", values="median_over_seeds").reset_index()
    allstats.to_csv(OUT / "s08_transition_vs_default_all.csv", index=False)
    met = t13.groupby(["preset", "seed"]).met.sum().unstack().to_dict()
    checkpoint("s08compare", started, cpu0, {"targets_met": met})


# ---------------------------------------------------------------------------
# Step 1: observed-data fits
# ---------------------------------------------------------------------------
def observed_work(src: Any, method: str) -> Any:
    """AnnData with observed layers/metadata only (the inputs of every fit)."""
    import anndata as ad
    import pandas as pd
    work = ad.AnnData(X=np.zeros((src.n_obs, 0), dtype=np.float32),
                      obs=pd.DataFrame({"pulse_hours": src.obs["pulse_hours"].to_numpy(np.float32),
                                        "day": src.obs["day"].to_numpy(int)}, index=src.obs_names.copy()))
    work = ad.AnnData(X=dense(src.layers["spliced"]), obs=work.obs, var=pd.DataFrame(index=src.var_names.copy()))
    for key in ("spliced", "new", "total"):
        work.layers[key] = dense(src.layers[key])
    work.uns[f"embedding_{method}"] = dict(src.uns[f"embedding_{method}"])
    work.obsm[f"X_{method}"] = np.asarray(src.obsm[f"X_{method}"], dtype=np.float32)
    return work


def knn_matrix(z: np.ndarray, k: int = KNN) -> Any:
    """Row-normalised kNN-k averaging operator (self included), as in labelling_diag_2026-10-19/diag.py."""
    from scipy import sparse
    from sklearn.neighbors import NearestNeighbors
    idx = NearestNeighbors(n_neighbors=k).fit(z).kneighbors(z)[1]
    n = len(z)
    return sparse.csr_matrix((np.full(n * k, 1.0 / k), (np.repeat(np.arange(n), k), idx.ravel())), shape=(n, n))


def labelling_gene_velocity(new: np.ndarray, total: np.ndarray, pulse: np.ndarray, gamma_per_cell: np.ndarray) -> np.ndarray:
    """alpha - gamma * total with alpha = gamma * new / (1 - exp(-gamma t)) (S02 formula)."""
    frac = -np.expm1(-pulse[:, None] * gamma_per_cell)
    return (gamma_per_cell * new / np.maximum(frac, 1e-12) - gamma_per_cell * total).astype(np.float32)


def flow_jacobian(z: np.ndarray, v: np.ndarray, fit_rows: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """J_flow = jacobian_autodiff of MeasuredDrift(d, seed=0) fitted on (z, v).

    The fit uses MeasuredDrift's default device; the fitted network is then
    moved to the CPU for ``jacobian_autodiff`` (same function, ~10x faster than
    per-cell autograd on MPS).  Returns (J float32, fitted velocity, details).
    """
    from scjdo2.measured_drift import MeasuredDrift
    z32 = np.asarray(z, dtype=np.float32)
    rows = np.arange(len(z32)) if fit_rows is None else np.asarray(fit_rows)
    vv = np.asarray(v, dtype=np.float32)[rows]
    ok = np.isfinite(vv).all(axis=1)
    t0 = time.time()
    md = MeasuredDrift(z32.shape[1], seed=0).fit(z32[rows][ok], vv[ok], verbose=False)
    device = str(next(md._model.parameters()).device)
    md._model.to("cpu")
    fitted = md.velocity(z32).astype(np.float32)
    J = md.jacobian_autodiff(z32).astype(np.float32)
    return J, fitted, {"fit_device": device, "jacobian_device": "cpu", "n_fit_cells": int(ok.sum()),
                       "n_dropped_nonfinite": int((~ok).sum()), "seconds": round(time.time() - t0, 2)}


def stage_fit(preset: str, seed: int, method: str) -> None:
    import anndata as ad
    import orient_end_to_end as s02
    from scjdo2.energy_drift import EnergyDriftAdapter, EnergyDriftConfig
    from scjdo2.labelled_rates import LabelledRateModel
    started, cpu0 = time.time(), cpu_seconds()
    row = manifest_row(preset, seed)
    src = ad.read_h5ad(ROOT / row["path"])
    work = observed_work(src, method)
    del src
    z = np.asarray(work.obsm[f"X_{method}"], dtype=np.float32)
    out: dict[str, np.ndarray] = {}
    details: dict[str, Any] = {"input_sha256": row["sha256"], "method": method, "n_cells": int(work.n_obs)}
    # --- score operator (exact S06 energy configuration)
    t0 = time.time()
    cfg = EnergyDriftConfig(hidden=96, depth=3, n_epochs=1500, seed=0, batch_size=256,
                            loss="esm", lr=5e-4, beta=1.0, esm_laplacian="exact")
    energy = EnergyDriftAdapter(z.shape[1], cfg).fit(z, verbose=False)
    symmetry = energy.symmetry_check(z, seed=0)
    if float(symmetry["max_asym"]) > 1e-4:
        raise RuntimeError(f"symmetry check {symmetry['max_asym']:.6g} exceeds frozen 1e-4 gate")
    out["J_score"] = np.asarray(energy.jacobian_autodiff(z), dtype=np.float32)
    out["X_drift_v"] = energy.velocity(z).astype(np.float32)
    out["energy"] = energy.energy(z).astype(np.float32)
    details["energy"] = {"symmetry_median": float(symmetry["median_asym"]), "symmetry_max": float(symmetry["max_asym"]),
                         "seconds": round(time.time() - t0, 2)}
    # --- labelling: shared gamma (S02/S06 LabelledRateModel), raw counts
    t0 = time.time()
    raw_gene, lab_details, model = s02.labelled_velocity(work, z)
    gamma_shared = np.asarray(model.gamma, dtype=np.float64)
    details["labelling_shared"] = {**lab_details, "seconds": round(time.time() - t0, 2)}
    new = np.asarray(work.layers["new"], dtype=np.float64)
    total = np.asarray(work.layers["total"], dtype=np.float64)
    pulse = work.obs["pulse_hours"].to_numpy(np.float64)
    # --- one gamma per pulse group (refit per pulse group)
    gamma_pulse: dict[float, np.ndarray] = {}
    for p in sorted(np.unique(pulse)):
        m = pulse == p
        t0 = time.time()
        fitted = LabelledRateModel(seed=0).fit(new[m].astype(np.float32), total[m].astype(np.float32), z[m], pulse[m].astype(np.float32))
        gamma_pulse[float(p)] = np.asarray(fitted.gamma, dtype=np.float64)
        details[f"labelling_pulse_{p:g}h"] = {"n_cells": int(m.sum()), "held_out_nll": float(fitted.best_val_nll),
                                               "gamma_min": float(gamma_pulse[float(p)].min()), "gamma_max": float(gamma_pulse[float(p)].max()),
                                               "seconds": round(time.time() - t0, 2)}
    gamma_cell_pp = np.vstack([gamma_pulse[float(p)] for p in pulse])
    gamma_cell_sh = np.broadcast_to(gamma_shared[None, :], new.shape)
    # --- kNN-30 smoothed normalised moments (labelling_diag diag.py), back to the cell's count units
    sf = total.sum(axis=1) / np.median(total.sum(axis=1))
    A = knn_matrix(z, KNN)
    Ns = np.asarray(A @ (new / sf[:, None]))
    Ts = np.asarray(A @ (total / sf[:, None]))
    gene_v = {
        "raw_shared": np.asarray(raw_gene, dtype=np.float32),
        "raw_perpulse": labelling_gene_velocity(new, total, pulse, gamma_cell_pp),
        "smoothed_shared": (labelling_gene_velocity(Ns, Ts, pulse, gamma_cell_sh) * sf[:, None]).astype(np.float32),
        "smoothed_perpulse": (labelling_gene_velocity(Ns, Ts, pulse, gamma_cell_pp) * sf[:, None]).astype(np.float32),
    }
    check = labelling_gene_velocity(new, total, pulse, gamma_cell_sh)
    details["raw_shared_formula_max_abs_diff"] = float(np.max(np.abs(check - gene_v["raw_shared"])))
    del Ns, Ts, check, gamma_cell_pp
    for name, gv in gene_v.items():
        v = s02.chain_rule_count_velocity(work, method, gv)
        if not np.isfinite(v).all():
            raise RuntimeError(f"non-finite labelling velocity {name}")
        out[f"v_lab_{name}"] = v.astype(np.float32)
    del gene_v
    # --- labelling flow operators (one MeasuredDrift per variant)
    for name in VARIANTS:
        J, fitted, info = flow_jacobian(z, out[f"v_lab_{name}"])
        out[f"J_flow_{name}"] = J
        out[f"v_lab_fit_{name}"] = fitted
        details[f"J_flow_{name}"] = info
    out["gamma_shared"] = gamma_shared.astype(np.float32)
    for p, g in gamma_pulse.items():
        out[f"gamma_pulse_{p:g}h"] = g.astype(np.float32)
    out["z"] = z
    dest = unit_dir(preset, seed) / f"fit_{method}.npz"
    np.savez(dest, **out)
    details.update({"path": str(dest.relative_to(ROOT)), "sha256": sha256(dest), "bytes": dest.stat().st_size,
                    "keys": sorted(out), "smoothing": "kNN-30 in the unit embedding, self included, size factor = total library / median; "
                    "smoothed gene velocity multiplied by the size factor before the S02 chain rule",
                    "velocity_labelling_s06": "v_lab_fit_raw_shared (MeasuredDrift on the raw labelling velocity, as S06)"})
    checkpoint(f"fit_{preset}_s{seed}_{method}", started, cpu0, details)


def stage_scvelo(preset: str, seed: int) -> None:
    """Exact S03 scVelo block via scripts/s06_user_runs.run_scvelo (imported, unedited).

    Deviation (runtime only): recover_dynamics and velocity_graph receive
    ``n_jobs=8`` (they parallelise over genes / cells; each gene fit is
    independent, so the fitted values do not depend on n_jobs).
    """
    import anndata as ad
    import scvelo as scv
    import s06_user_runs as s06u
    started, cpu0 = time.time(), cpu_seconds()
    row = manifest_row(preset, seed)
    src = ad.read_h5ad(ROOT / row["path"])
    # observed-only copy: counts, embeddings and their stored transforms, cell ids
    import pandas as pd
    obs = ad.AnnData(X=dense(src.layers["spliced"]), obs=pd.DataFrame(index=src.obs_names.copy()),
                     var=pd.DataFrame(index=src.var_names.copy()))
    for key in ("spliced", "unspliced"):
        obs.layers[key] = dense(src.layers[key])
    for key in ("X_pca", "X_fa"):
        obs.obsm[key] = np.asarray(src.obsm[key])
    del src
    # The S03 block calls recover_dynamics / velocity_graph without n_jobs (scVelo then
    # uses 1 core: > 60 min per v3 dataset).  Pass n_jobs=8 through thin wrappers;
    # every other argument is untouched (same pattern as the S09 recover_dynamics mutex).
    import functools
    originals = {name: getattr(scv.tl, name) for name in ("recover_dynamics", "velocity_graph")}
    for name, fn in originals.items():
        setattr(scv.tl, name, functools.partial(fn, n_jobs=8))
    try:
        with file_lock("scvelo"):
            result, stochastic, n_genes = s06u.run_scvelo(obs)
    finally:
        for name, fn in originals.items():
            setattr(scv.tl, name, fn)
    out = {key: np.asarray(result.obsm[key], dtype=np.float32) for key in result.obsm if key.startswith("velocity_")}
    dest = unit_dir(preset, seed) / "scvelo.npz"
    np.savez(dest, **out)
    checkpoint(f"scvelo_{preset}_s{seed}", started, cpu0, {
        "input_sha256": row["sha256"], "path": str(dest.relative_to(ROOT)), "sha256": sha256(dest),
        "keys": sorted(out), "n_genes_scvelo": n_genes, "stochastic": stochastic, "scvelo_n_jobs": 8,
        "settings": "exact S03 block (s06_user_runs.run_scvelo): filter_genes(min_shared_counts=20), normalize_per_cell, "
                    "log1p, neighbors(X_pca, 30, random_state=0), moments, recover_dynamics, velocity(dynamical), "
                    "velocity_graph, velocity_embedding(pca, fa); stochastic optional"})


def palantir_roots(days: np.ndarray, names: np.ndarray, seed: int) -> dict[str, list[str]]:
    """S06 root rule on v3 days: 5 earliest-day cells rng(1000+seed), 5 latest-day cells rng(2000+seed)."""
    first, last = int(days.min()), int(days.max())
    out = {}
    for label, day, rng_seed in ((f"day{first}", first, 1000 + seed), (f"day{last}", last, 2000 + seed)):
        pool = names[days == day]
        if len(pool) < 5:
            raise RuntimeError(f"{label} has only {len(pool)} cells")
        out[label] = list(np.random.default_rng(rng_seed).choice(pool, 5, replace=False))
    return out


def stage_palantir(preset: str, seed: int) -> None:
    """Palantir via scripts/s06_user_runs.run_palantir (imported, unedited; X_pca only)."""
    import anndata as ad
    import pandas as pd
    import s06_user_runs as s06u
    started, cpu0 = time.time(), cpu_seconds()
    row = manifest_row(preset, seed)
    src = ad.read_h5ad(ROOT / row["path"], backed="r")
    names = src.obs_names.to_numpy(str)
    days = src.obs["day"].to_numpy(int)
    obs = ad.AnnData(X=np.empty((len(names), 0), dtype=np.float32),
                     obs=pd.DataFrame({"day": days}, index=pd.Index(names)),
                     obsm={"X_pca": np.asarray(src.obsm["X_pca"], dtype=np.float64)})
    src.file.close()
    runs: list[dict[str, Any]] = []
    pseudo: dict[str, np.ndarray] = {}
    if preset == "v3_stationary_cycle":
        # Step 4 null: one Palantir run from a random cell (rng(1000+seed), default mode).
        plan = [("random", "default", 0, str(np.random.default_rng(1000 + seed).choice(names)))]
    else:
        roots = palantir_roots(days, names, seed)
        plan = [(group, mode, i, given) for group in roots for mode in ("default", "strict") for i, given in enumerate(roots[group])]
    for group, mode, i, given in plan:
        unit = f"{group}_{mode}_r{i}"
        try:
            output, info = s06u.run_palantir(obs, given, group, mode)
        except Exception as exc:  # retry once (task: retry a failed unit once)
            print(f"[S10 palantir] {unit} failed once: {exc!r}; retrying", flush=True)
            try:
                output, info = s06u.run_palantir(obs, given, group, mode)
                info["retried"] = True
            except Exception as exc2:
                runs.append({"unit": unit, "root_group": group, "mode": mode, "root_index": i, "given_cell": given,
                             "status": "failed_twice", "error": repr(exc2)})
                continue
        pseudo[unit] = output.obs["palantir_pseudotime"].to_numpy(np.float64)
        runs.append({"unit": unit, "root_group": group, "mode": mode, "root_index": i, "status": "ok", **info})
        print(f"[S10 palantir] {preset} s{seed} {unit}: {info['wall_seconds']}s", flush=True)
    dest = unit_dir(preset, seed) / "palantir.npz"
    np.savez(dest, names=names, **pseudo)
    checkpoint(f"palantir_{preset}_s{seed}", started, cpu0, {
        "input_sha256": row["sha256"], "path": str(dest.relative_to(ROOT)), "sha256": sha256(dest), "runs": runs,
        "settings": {"n_components": 10, "knn": 30, "num_waypoints": 500, "seed": 0, "input": "stored X_pca only"},
        "root_rule": "earliest day: default_rng(1000+seed).choice(5); latest day: default_rng(2000+seed).choice(5); "
                     "stationary cycle: default_rng(1000+seed).choice(1) of all cells, default mode"})


def ot_velocity(z: np.ndarray, days: np.ndarray, hours: np.ndarray, *, reg_m_factor: float | None = None,
                growth: float | None = None) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Entropic OT between consecutive days (POT), barycentric velocity / time gap.

    Balanced: ``ot.sinkhorn`` with uniform marginals, eps = 0.05 x median squared distance.
    Growth-weighted (sensitivity): ``ot.unbalanced.sinkhorn_unbalanced`` with source
    masses ``growth_factor / n_target`` (growth_factor = exp(dt log2 / 16 h) (1 - f_sample)),
    target masses ``1 / n_target`` and ``reg_m = reg_m_factor x eps``.
    """
    import ot
    v = np.full_like(z, np.nan, dtype=np.float64)
    info: list[dict[str, Any]] = []
    uniq = np.unique(days)
    for d0, d1 in zip(uniq[:-1], uniq[1:]):
        i0, i1 = np.flatnonzero(days == d0), np.flatnonzero(days == d1)
        x, y = z[i0].astype(np.float64), z[i1].astype(np.float64)
        dt = float(np.median(hours[i1]) - np.median(hours[i0]))
        C = ot.dist(x, y, metric="sqeuclidean")
        eps = float(0.05 * np.median(C))
        b = np.full(len(i1), 1.0 / len(i1))
        t0 = time.time()
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if reg_m_factor is None:
                a = np.full(len(i0), 1.0 / len(i0))
                P, log = ot.sinkhorn(a, b, C, reg=eps, log=True)
            else:
                g = float(growth(dt))
                a = np.full(len(i0), g / len(i1))
                P, log = ot.unbalanced.sinkhorn_unbalanced(a, b, C, reg=eps, reg_m=reg_m_factor * eps, log=True)
        rows = P.sum(axis=1, keepdims=True)
        bary = (P @ y) / np.maximum(rows, 1e-300)
        v[i0] = (bary - x) / dt
        info.append({"from_day": int(d0), "to_day": int(d1), "dt_hours": dt, "n_source": int(len(i0)), "n_target": int(len(i1)),
                     "epsilon": eps, "reg_m": None if reg_m_factor is None else reg_m_factor * eps,
                     "source_mass": float(a.sum()), "plan_mass": float(P.sum()), "n_iter": int(len(log.get("err", []))),
                     "final_err": float(log["err"][-1]) if len(log.get("err", [])) else None,
                     "warnings": [str(w.message)[:200] for w in caught], "rows_zero": int((rows[:, 0] <= 0).sum()),
                     "seconds": round(time.time() - t0, 2)})
    return v.astype(np.float32), np.isfinite(v).all(axis=1), info


def stage_flow(preset: str, seed: int, method: str) -> None:
    """J_flow_scvelo (MeasuredDrift on scVelo's embedded velocity) and J_flow_ot (time course alone)."""
    import anndata as ad
    started, cpu0 = time.time(), cpu_seconds()
    row = manifest_row(preset, seed)
    src = ad.read_h5ad(ROOT / row["path"], backed="r")
    z = np.asarray(src.obsm[f"X_{method}"], dtype=np.float32)
    days = src.obs["day"].to_numpy(int)
    hours = src.obs["time_hours"].to_numpy(np.float64)       # observed sampling time
    meta = src.uns["synthetic_multimodal"]
    f_sample = float(meta["f_sample"]); division_hours = float(meta["division_hours"])   # sensitivity only
    src.file.close()
    out: dict[str, np.ndarray] = {}
    details: dict[str, Any] = {"input_sha256": row["sha256"], "method": method}
    scv_path = unit_dir(preset, seed) / "scvelo.npz"
    if scv_path.exists():
        vs = np.load(scv_path)[f"velocity_{method}"]
        J, fitted, info = flow_jacobian(z, vs)
        out["J_flow_scvelo"], out["v_scvelo"] = J, vs
        details["J_flow_scvelo"] = {**info, "scvelo_sha256": sha256(scv_path)}
    else:
        details["J_flow_scvelo"] = {"status": "scvelo output missing"}
    if len(np.unique(days)) < 2:
        details["J_flow_ot"] = {"status": "not computable: a single sampling time"}
    else:
        import ot
        details["ot_library"] = {"name": "POT", "version": ot.__version__, "moscot": "not installed (no installs without approval)"}
        v, ok, info = ot_velocity(z, days, hours)
        J, _, jinfo = flow_jacobian(z, v, np.flatnonzero(ok))
        out["J_flow_ot"], out["v_ot"] = J, v
        details["J_flow_ot"] = {**jinfo, "pairs": info, "balanced": True}
        growth = lambda dt: np.exp(dt * np.log(2.0) / division_hours) * (1.0 - f_sample)
        for c in (0.1, 1.0, 10.0):
            key = f"J_flow_ot_ub{c:g}"
            v, ok, info = ot_velocity(z, days, hours, reg_m_factor=c, growth=growth)
            if ok.sum() < 10:
                details[key] = {"status": "too few finite OT velocities", "pairs": info}
                continue
            J, _, jinfo = flow_jacobian(z, v, np.flatnonzero(ok))
            out[key], out[f"v_ot_ub{c:g}"] = J, v
            details[key] = {**jinfo, "pairs": info, "reg_m_factor": c,
                            "growth_factor": "exp(dt log2 / division_hours) (1 - f_sample) from the generator (sensitivity only)"}
    dest = unit_dir(preset, seed) / f"flow_{method}.npz"
    np.savez(dest, **out)
    details.update({"path": str(dest.relative_to(ROOT)), "sha256": sha256(dest), "keys": sorted(out)})
    checkpoint(f"flow_{preset}_s{seed}_{method}", started, cpu0, details)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("generate", "s08", "s08compare", "fit", "scvelo", "palantir", "flow"))
    parser.add_argument("--preset", choices=PRESETS)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--method", choices=METHODS)
    args = parser.parse_args()
    if args.stage == "s08compare":
        stage_s08compare()
        return
    if args.preset is None or args.seed is None:
        parser.error("--preset and --seed are required")
    if args.stage == "generate":
        stage_generate(args.preset, args.seed)
    elif args.stage == "s08":
        stage_s08(args.preset, args.seed)
    elif args.stage == "fit":
        if args.method is None:
            parser.error("fit requires --method")
        stage_fit(args.preset, args.seed, args.method)
    elif args.stage == "scvelo":
        stage_scvelo(args.preset, args.seed)
    elif args.stage == "palantir":
        stage_palantir(args.preset, args.seed)
    elif args.stage == "flow":
        if args.method is None:
            parser.error("flow requires --method")
        stage_flow(args.preset, args.seed, args.method)


if __name__ == "__main__":
    main()
