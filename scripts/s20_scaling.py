#!/usr/bin/env python3
"""S20 LARRY OT solver agreement and scaling study.

The script is intentionally unit-oriented: ``prepare``, ``ot``, ``fit`` and
``finalize`` are separate commands so each expensive operation can be run
under ``/usr/bin/time -l`` with its own checkpoint.  The low-rank command is
run with the task-authorized ``.venv-baselines`` interpreter; all other
commands use ``.sandbox/bin/python``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks/scaling"
DATA = OUT / "data"
CHECKPOINTS = OUT / "checkpoints"
LOGS = OUT / "logs"
FIGURES = ROOT / "benchmarks/figures/scaling"
LARRY = Path("/Users/terooatt/Documents/scJDO2/benchmarks/data/larry/larry_inVitro_raw.h5ad")
EXPECTED_LARRY_SHA256 = "3e13ff16d0f3d1338932860abe15b1609749bc7f70d4b2596058ae3d7763a80e"
N_VALUES = (2000, 4000, 8000, 16000)
SOURCE_DAYS = (2.0, 4.0)
ALL_DAYS = (2.0, 4.0, 6.0)
BATCH_SIZE = 4000
MINIBATCH_COVERAGE = 10


def utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ensure_dirs() -> None:
    for path in (OUT, DATA, CHECKPOINTS, LOGS, FIGURES):
        path.mkdir(parents=True, exist_ok=True)


def checkpoint(unit: str, payload: dict[str, Any], *, started: float | None = None) -> None:
    ensure_dirs()
    result = {
        "task": "orient/S20",
        "unit": unit,
        "finished_utc": utc(),
        "status": "complete",
        **payload,
    }
    if started is not None:
        result["wall_seconds_internal"] = round(time.time() - started, 3)
    (CHECKPOINTS / f"{unit}.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def load_embedding() -> dict[str, np.ndarray]:
    path = DATA / "larry_embedding_all.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing prepared embedding: {path}")
    data = {key: value for key, value in np.load(path, allow_pickle=False).items()}
    if data["z_all"].ndim != 2 or data["z_all"].shape[1] != 30:
        raise RuntimeError(f"invalid embedding shape {data['z_all'].shape}")
    return data


def indices_for_n(embedding: dict[str, np.ndarray], n: int | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n == "full":
        day = embedding["day"].astype(float)
        return tuple(np.flatnonzero(day == value).astype(np.int64) for value in ALL_DAYS)  # type: ignore[return-value]
    n_int = int(n)
    parts = []
    for day in SOURCE_DAYS:
        key = f"sample_{n_int}_{int(day)}"
        if key not in embedding:
            raise KeyError(f"embedding lacks {key}")
        parts.append(embedding[key].astype(np.int64))
    day6 = embedding[f"sample_{n_int}_6"].astype(np.int64)
    return parts[0], parts[1], day6


def prepare() -> None:
    started = time.time()
    if not LARRY.is_file():
        raise FileNotFoundError(LARRY)
    input_hash = sha256(LARRY)
    if input_hash != EXPECTED_LARRY_SHA256:
        raise RuntimeError(f"LARRY sha256 mismatch: {input_hash} != {EXPECTED_LARRY_SHA256}")

    import anndata as ad
    import h5py
    import scanpy as sc
    import scipy.sparse as sp
    from sklearn.decomposition import PCA

    # Read the source through backed HDF5 in chunks.  The assembled sparse
    # count matrix is a task-local copy; the source file is never opened for
    # writing.
    backed = ad.read_h5ad(LARRY, backed="r")
    obs = backed.obs.copy()
    var = backed.var.copy()
    chunks = []
    # anndata's backed_csr_matrix slice path is incompatible with the local
    # SciPy/anndata pair, so read the same CSR arrays directly in row chunks.
    # This preserves the task's backed/in-chunks rule and never writes the
    # source H5AD.
    with h5py.File(LARRY, "r") as handle:
        x_group = handle["X"]
        shape = tuple(int(x) for x in x_group.attrs["shape"])
        indptr = np.asarray(x_group["indptr"], dtype=np.int64)
        for lo in range(0, shape[0], 4096):
            hi = min(lo + 4096, shape[0])
            ptr_lo, ptr_hi = int(indptr[lo]), int(indptr[hi])
            row_ptr = indptr[lo : hi + 1] - ptr_lo
            chunks.append(sp.csr_matrix(
                (np.asarray(x_group["data"][ptr_lo:ptr_hi]),
                 np.asarray(x_group["indices"][ptr_lo:ptr_hi], dtype=np.int64),
                 row_ptr), shape=(hi - lo, shape[1])))
    counts = sp.vstack(chunks, format="csr")
    backed.file.close()
    data = ad.AnnData(X=counts, obs=obs, var=var)
    days = pd.to_numeric(data.obs["Time point"], errors="raise").to_numpy(float)
    if tuple(sorted(np.unique(days))) != ALL_DAYS:
        raise RuntimeError(f"unexpected LARRY time points: {sorted(np.unique(days))}")
    cell_type = data.obs["Cell type annotation"].astype(str).to_numpy()

    sc.pp.highly_variable_genes(data, flavor="seurat_v3", n_top_genes=2000, subset=False)
    hvg = data.var["highly_variable"].to_numpy(bool)
    if int(hvg.sum()) != 2000:
        raise RuntimeError(f"expected exactly 2,000 HVGs, observed {int(hvg.sum())}")
    library = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
    if np.any(library <= 0):
        raise RuntimeError("non-positive LARRY library size")
    target = float(np.median(library))
    log_hvg = np.empty((data.n_obs, int(hvg.sum())), dtype=np.float32)
    for lo in range(0, data.n_obs, 4096):
        hi = min(lo + 4096, data.n_obs)
        block = counts[lo:hi, hvg].toarray().astype(np.float32)
        block *= (target / library[lo:hi])[:, None].astype(np.float32)
        log_hvg[lo:hi] = np.log1p(block)
    pca = PCA(n_components=30, svd_solver="randomized", random_state=0)
    z_all = pca.fit_transform(log_hvg).astype(np.float32)

    rng = np.random.default_rng(0)
    sample_max: dict[int, np.ndarray] = {}
    for day in ALL_DAYS:
        candidates = np.flatnonzero(days == day).astype(np.int64)
        if len(candidates) < 16000:
            raise RuntimeError(f"day {day:g} has only {len(candidates)} cells; need 16,000")
        sample_max[int(day)] = rng.choice(candidates, size=16000, replace=False).astype(np.int64)
    payload: dict[str, Any] = {
        "z_all": z_all,
        "day": days.astype(np.float32),
        "cell_type": cell_type.astype(str),
        "cell_id": data.obs_names.to_numpy(str),
        "hvg_index": np.flatnonzero(hvg).astype(np.int32),
        "pca_components": pca.components_.astype(np.float32),
        "pca_mean": pca.mean_.astype(np.float32),
        "normalization_target": np.asarray(target, dtype=np.float64),
        "input_sha256": np.asarray(input_hash),
    }
    for n in N_VALUES:
        for day in ALL_DAYS:
            payload[f"sample_{n}_{int(day)}"] = sample_max[int(day)][:n]
    np.savez_compressed(DATA / "larry_embedding_all.npz", **payload)
    checkpoint(
        "prepare_larry_all",
        {
            "input_sha256": input_hash,
            "n_cells": int(data.n_obs),
            "n_hvg": int(hvg.sum()),
            "embedding_dimension": 30,
            "pca_solver": "sklearn PCA(svd_solver=randomized, random_state=0) fitted on all cells",
            "normalization": "library-normalise to median then log1p",
            "normalization_target": target,
            "sample_seed": 0,
            "sample_per_time": {str(n): n for n in N_VALUES},
            "embedding_path": str((DATA / "larry_embedding_all.npz").relative_to(ROOT)),
            "source_read_mode": "backed HDF5 in 4096-cell chunks",
        },
        started=started,
    )


def distance_median(x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    """Return the fixed S11 epsilon scale without exceeding memory limits."""
    from scipy.spatial.distance import cdist

    total = int(x.shape[0]) * int(y.shape[0])
    if total <= 64_000_000:
        cost = cdist(x.astype(np.float64), y.astype(np.float64), metric="sqeuclidean")
        result = float(np.median(cost))
        del cost
        return result, "exact full pairwise median"
    rng = np.random.default_rng(0)
    sample_n = min(2_000_000, total)
    rows = rng.integers(0, x.shape[0], size=sample_n)
    cols = rng.integers(0, y.shape[0], size=sample_n)
    values = np.sum((x[rows].astype(np.float64) - y[cols].astype(np.float64)) ** 2, axis=1)
    return float(np.median(values)), f"deterministic {sample_n:,}-pair median (full pairwise matrix exceeds memory budget)"


def dense_bary(x: np.ndarray, y: np.ndarray, epsilon: float) -> tuple[np.ndarray, dict[str, Any]]:
    import ot

    cost = ot.dist(x.astype(np.float64), y.astype(np.float64), metric="sqeuclidean")
    a = np.full(x.shape[0], 1.0 / x.shape[0], dtype=np.float64)
    b = np.full(y.shape[0], 1.0 / y.shape[0], dtype=np.float64)
    plan = ot.sinkhorn(a, b, cost, reg=epsilon)
    rows = np.asarray(plan.sum(axis=1)).ravel()
    bary = np.asarray(plan @ y, dtype=np.float64) / np.maximum(rows[:, None], 1e-12)
    return bary.astype(np.float32), {"library": "POT", "version": getattr(ot, "__version__", "unknown"), "plan_shape": list(plan.shape)}


def lowrank_bary(x: np.ndarray, y: np.ndarray, epsilon: float, rank: int) -> tuple[np.ndarray, dict[str, Any]]:
    # This function is called only by the low-rank interpreter.  Keeping the
    # import local makes accidental use of moscot in the regular environment
    # fail loudly instead of silently changing the solver.
    import anndata as ad
    from moscot.problems.time import TemporalProblem

    xy = np.vstack([x, y]).astype(np.float32)
    obs = pd.DataFrame({"s20_time": np.r_[np.zeros(len(x)), np.ones(len(y))]})
    problem = TemporalProblem(ad.AnnData(xy, obs=obs)).prepare(
        time_key="s20_time", joint_attr=None, cost="sq_euclidean", policy="sequential"
    )
    problem.solve(
        epsilon=float(epsilon), tau_a=1.0, tau_b=1.0, rank=int(rank), scale_cost=1.0,
        device="cpu", max_iterations=1000,
    )
    solution = problem.solutions[(0.0, 1.0)] if (0.0, 1.0) in problem.solutions else next(iter(problem.solutions.values()))
    transported = np.asarray(solution.pull(y.astype(np.float32), scale_by_marginals=False))
    row_mass = np.asarray(solution.pull(np.ones(len(y), dtype=np.float32), scale_by_marginals=False)).ravel()
    bary = transported / np.maximum(row_mass[:, None], 1e-12)
    return bary.astype(np.float32), {
        "library": "moscot",
        "version": "0.5.2",
        "rank": int(rank),
        "solution_rank": int(solution.rank),
        "converged": bool(solution.converged),
        "plan_shape": list(solution.shape),
    }


def minibatch_bary(x: np.ndarray, y: np.ndarray, epsilon: float) -> tuple[np.ndarray, dict[str, Any]]:
    import ot

    rng = np.random.default_rng(0)
    n = len(x)
    # Ten exact source appearances per source cell, split into 4,000-cell
    # batches.  This guarantees coverage rather than relying on expectation.
    source_schedule = np.tile(np.arange(n, dtype=np.int64), MINIBATCH_COVERAGE)
    rng.shuffle(source_schedule)
    bary = np.zeros_like(x, dtype=np.float64)
    coverage = np.zeros(n, dtype=np.int32)
    batch_count = 0
    for lo in range(0, len(source_schedule), BATCH_SIZE):
        source_idx = source_schedule[lo : lo + BATCH_SIZE]
        target_replace = len(y) < BATCH_SIZE
        target_idx = rng.choice(len(y), size=BATCH_SIZE, replace=target_replace)
        xb, yb = x[source_idx].astype(np.float64), y[target_idx].astype(np.float64)
        cost = ot.dist(xb, yb, metric="sqeuclidean")
        a = np.full(len(xb), 1.0 / len(xb), dtype=np.float64)
        b = np.full(len(yb), 1.0 / len(yb), dtype=np.float64)
        plan = ot.sinkhorn(a, b, cost, reg=epsilon)
        mass = np.asarray(plan.sum(axis=1)).ravel()
        batch_bary = np.asarray(plan @ yb) / np.maximum(mass[:, None], 1e-12)
        np.add.at(bary, source_idx, batch_bary)
        np.add.at(coverage, source_idx, 1)
        batch_count += 1
    if int(coverage.min()) < 1:
        raise RuntimeError("mini-batch source coverage is incomplete")
    bary /= coverage[:, None]
    return bary.astype(np.float32), {
        "library": "POT",
        "version": getattr(ot, "__version__", "unknown"),
        "batch_size": BATCH_SIZE,
        "batch_count": batch_count,
        "coverage_min": int(coverage.min()),
        "coverage_median": float(np.median(coverage)),
        "coverage_max": int(coverage.max()),
        "target_sampling_with_replacement": target_replace,
    }


def solver_name(solver: str, rank: int | None = None) -> str:
    return f"lowrank_r{rank}" if solver == "lowrank" else solver


def velocity_path(solver: str, n: int | str, rank: int | None = None) -> Path:
    return DATA / f"velocity_{solver_name(solver, rank)}_{n}.npz"


def unit_name(stage: str, solver: str, n: int | str, rank: int | None = None) -> str:
    return f"{stage}_{solver_name(solver, rank)}_{n}"


def run_ot(solver: str, n: int | str, rank: int | None = None) -> None:
    started = time.time()
    ensure_dirs()
    emb = load_embedding()
    index_parts = indices_for_n(emb, n)
    z_all = emb["z_all"].astype(np.float32)
    source_z: list[np.ndarray] = []
    source_v: list[np.ndarray] = []
    source_index: list[np.ndarray] = []
    pair_details: list[dict[str, Any]] = []
    for pair_i, earlier in enumerate(SOURCE_DAYS):
        left = index_parts[pair_i]
        right = index_parts[pair_i + 1]
        x, y = z_all[left], z_all[right]
        median_cost, median_method = distance_median(x, y)
        epsilon = 0.05 * median_cost
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise RuntimeError(f"invalid epsilon {epsilon} for day {earlier:g}")
        if solver == "dense":
            bary, details = dense_bary(x, y, epsilon)
        elif solver == "lowrank":
            if rank not in (50, 200):
                raise ValueError("lowrank requires rank 50 or 200")
            bary, details = lowrank_bary(x, y, epsilon, rank)
        elif solver == "minibatch":
            bary, details = minibatch_bary(x, y, epsilon)
        else:
            raise ValueError(solver)
        source_z.append(x)
        source_v.append((bary - x) / 48.0)
        source_index.append(left)
        pair_details.append({
            "from_day": earlier,
            "to_day": earlier + 2.0,
            "n_source": int(len(left)),
            "n_target": int(len(right)),
            "epsilon": epsilon,
            "median_squared_distance": median_cost,
            "median_method": median_method,
            **details,
        })
    out = velocity_path(solver, n, rank)
    np.savez_compressed(
        out,
        z=np.vstack(source_z).astype(np.float32),
        velocity=np.vstack(source_v).astype(np.float32),
        source_index=np.concatenate(source_index).astype(np.int64),
    )
    checkpoint(unit_name("ot", solver, n, rank), {
        "stage": "ot",
        "solver": solver_name(solver, rank),
        "n_per_day": n,
        "rank": rank,
        "n_velocity_cells": int(sum(len(x) for x in source_index)),
        "velocity_path": str(out.relative_to(ROOT)),
        "pairs": pair_details,
        "wall_seconds": round(time.time() - started, 3),
    }, started=started)


def lambda1(jacobian: np.ndarray) -> np.ndarray:
    values = np.linalg.eigvals(np.asarray(jacobian, dtype=np.float64))
    return np.max(values.real, axis=1).astype(np.float32)


def run_fit(solver: str, n: int | str, rank: int | None = None) -> None:
    started = time.time()
    ensure_dirs()
    import torch

    torch.set_num_threads(int(os.environ.get("SCJDO2_TORCH_THREADS", "1")))
    from scjdo2.measured_drift import MeasuredDrift

    emb = load_embedding()
    path = velocity_path(solver, n, rank)
    if not path.is_file():
        raise FileNotFoundError(path)
    data = {key: value for key, value in np.load(path, allow_pickle=False).items()}
    z = data["z"].astype(np.float32)
    velocity = data["velocity"].astype(np.float32)
    source_index = data["source_index"].astype(np.int64)
    model = MeasuredDrift(30, seed=0).fit(z, velocity, verbose=False)
    eval_index = np.arange(len(emb["z_all"]), dtype=np.int64) if n == "full" else source_index
    jac = model.jacobian_autodiff(emb["z_all"][eval_index].astype(np.float32))
    values = lambda1(jac)
    out = DATA / f"lambda1_{solver_name(solver, rank)}_{n}.npz"
    np.savez_compressed(out, lambda1=values, eval_index=eval_index)
    checkpoint(unit_name("fit", solver, n, rank), {
        "stage": "fit_and_jacobian",
        "solver": solver_name(solver, rank),
        "n_per_day": n,
        "rank": rank,
        "n_fit_cells": int(len(z)),
        "n_eval_cells": int(len(eval_index)),
        "measured_drift": "MeasuredDrift(30, seed=0) defaults",
        "jacobian": "autodiff",
        "lambda1_path": str(out.relative_to(ROOT)),
        "wall_seconds": round(time.time() - started, 3),
    }, started=started)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {key: value for key, value in np.load(path, allow_pickle=False).items()}


def run_agreement() -> None:
    emb = load_embedding()
    rows: list[dict[str, Any]] = []
    for n in (2000, 4000, 8000):
        dense_v = load_npz(velocity_path("dense", n))
        dense_l = load_npz(DATA / f"lambda1_dense_{n}.npz")
        dense_by_index = {int(i): j for j, i in enumerate(dense_v["source_index"])}
        order = np.asarray([dense_by_index[int(i)] for i in dense_v["source_index"]])
        for solver in ("lowrank_r50", "lowrank_r200", "minibatch"):
            scale_v = load_npz(DATA / f"velocity_{solver}_{n}.npz")
            scale_l = load_npz(DATA / f"lambda1_{solver}_{n}.npz")
            if not np.array_equal(scale_v["source_index"], dense_v["source_index"]):
                raise RuntimeError(f"source index mismatch for {solver} n={n}")
            cosine = np.sum(dense_v["velocity"] * scale_v["velocity"], axis=1) / np.maximum(
                np.linalg.norm(dense_v["velocity"], axis=1) * np.linalg.norm(scale_v["velocity"], axis=1), 1e-12
            )
            rho = float(spearmanr(dense_l["lambda1"], scale_l["lambda1"]).statistic)
            rows.append({
                "n_per_day": n,
                "solver": solver,
                "median_velocity_cosine": float(np.median(cosine)),
                "velocity_cosine_q05": float(np.quantile(cosine, 0.05)),
                "velocity_cosine_q95": float(np.quantile(cosine, 0.95)),
                "flow_lambda1_spearman": rho,
                "n_cells_compared": int(len(cosine)),
                "G1_pass_at_n4000": bool(n == 4000 and np.median(cosine) >= 0.90 and rho >= 0.80),
            })
    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "agreement.csv", index=False)
    print(frame.to_string(index=False), flush=True)


def parse_time_log(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"log": str(path.relative_to(ROOT)), "status": "missing"}
    text = path.read_text(errors="replace")
    def number(pattern: str) -> float | None:
        match = re.search(pattern, text, flags=re.MULTILINE)
        return float(match.group(1)) if match else None
    rss = number(r"^\s*(\d+)\s+maximum resident set size")
    real = number(r"([0-9.]+)\s+real")
    user = number(r"([0-9.]+)\s+user")
    system = number(r"([0-9.]+)\s+sys")
    return {
        "log": str(path.relative_to(ROOT)),
        "status": "complete" if rss is not None and real is not None else "unparsed",
        "wall_seconds_external": real,
        "cpu_seconds_external": None if user is None or system is None else user + system,
        "max_rss_bytes": int(rss) if rss is not None else None,
    }


def run_finalize(best_solver: str | None) -> None:
    ensure_dirs()
    agreement = pd.read_csv(OUT / "agreement.csv") if (OUT / "agreement.csv").is_file() else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for path in sorted(CHECKPOINTS.glob("*.json")):
        item = json.loads(path.read_text())
        if item.get("stage") not in {"ot", "fit_and_jacobian"}:
            continue
        solver = str(item["solver"])
        n = item["n_per_day"]
        log_path = LOGS / f"{path.stem}.log"
        retry_path = LOGS / f"{path.stem}_retry.log"
        if retry_path.is_file():
            retry = parse_time_log(retry_path)
            if retry.get("status") == "complete":
                log_path = retry_path
        log = parse_time_log(log_path)
        rows.append({"solver": solver, "n_per_day": n, "stage": item["stage"], **log, "internal_wall_seconds": item.get("wall_seconds")})
    scaling = pd.DataFrame(rows)
    if not scaling.empty:
        scaling["_n_order"] = scaling["n_per_day"].map(lambda value: 10**9 if str(value) == "full" else int(value))
        scaling = scaling.sort_values(["solver", "_n_order", "stage"]).drop(columns="_n_order")
    scaling.to_csv(OUT / "scaling.csv", index=False)
    if best_solver is None:
        if agreement.empty:
            raise RuntimeError("cannot select best solver without agreement.csv")
        candidates = agreement[agreement.n_per_day == 4000].copy()
        candidates["gate"] = (candidates.median_velocity_cosine >= 0.90) & (candidates.flow_lambda1_spearman >= 0.80)
        best_solver = str(candidates.sort_values(["gate", "median_velocity_cosine", "flow_lambda1_spearman"], ascending=False).iloc[0].solver)
    best_path = DATA / f"lambda1_{best_solver}_full.npz"
    if best_path.is_file():
        emb = load_embedding()
        lam = load_npz(best_path)["lambda1"]
        frame = pd.DataFrame({
            "day": emb["day"].astype(float),
            "cell_type": emb["cell_type"].astype(str),
            "lambda1": lam.astype(float),
        })
        summary = frame.groupby(["day", "cell_type"], as_index=False).agg(
            n_cells=("lambda1", "size"), lambda1_median=("lambda1", "median"),
            lambda1_fraction_positive=("lambda1", lambda x: float(np.mean(np.asarray(x) > 0))),
        )
        s11_path = ROOT / "benchmarks/real_flow/by_cell_type.csv"
        if s11_path.is_file():
            ref = pd.read_csv(s11_path)
            ref = ref[(ref.dataset == "larry") & (ref.source == "ot") & (ref.operator == "measured")]
            ref = ref[["cell_type", "lambda1_median"]].rename(columns={"lambda1_median": "s11_lambda1_median"})
            summary = summary.merge(ref, on="cell_type", how="left")
            overall = frame.groupby("cell_type", as_index=False).lambda1.median().rename(columns={"lambda1": "lambda1_median"})
            common = overall.merge(ref, on="cell_type", how="inner")
            rank = float(spearmanr(common.lambda1_median, common.s11_lambda1_median).statistic) if len(common) >= 3 else float("nan")
        else:
            summary["s11_lambda1_median"] = np.nan
            rank = float("nan")
        summary.insert(0, "solver", best_solver)
        summary["s11_cell_type_lambda1_spearman"] = rank
        summary.to_csv(OUT / "larry_full_lambda1.csv", index=False)
    make_figures(scaling, agreement)
    print(json.dumps({"best_solver": best_solver, "scaling_rows": len(scaling), "agreement_rows": len(agreement)}, indent=2), flush=True)


def make_figures(scaling: pd.DataFrame, agreement: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"figure generation skipped: {exc}", flush=True)
        return
    if not scaling.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for solver, group in scaling[scaling.stage == "ot"].groupby("solver"):
            numeric = group[group.n_per_day.apply(lambda x: str(x).isdigit())]
            if numeric.empty:
                continue
            x = numeric.n_per_day.astype(int)
            axes[0].loglog(x, numeric.wall_seconds_external, marker="o", label=solver)
            axes[1].loglog(x, numeric.max_rss_bytes, marker="o", label=solver)
        axes[0].set(xlabel="cells per day", ylabel="wall seconds", title="OT runtime")
        axes[1].set(xlabel="cells per day", ylabel="peak RSS bytes", title="OT memory")
        for ax in axes: ax.legend(fontsize=7); ax.grid(True, which="both")
        fig.tight_layout(); fig.savefig(FIGURES / "runtime_memory_loglog.png", dpi=160); plt.close(fig)
    if not agreement.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for solver, group in agreement.groupby("solver"):
            ax.plot(group.n_per_day, group.median_velocity_cosine, marker="o", label=solver)
        ax.axhline(0.90, color="black", linestyle="--", linewidth=0.8)
        ax.set(xlabel="cells per day", ylabel="median velocity cosine", title="Agreement with dense OT")
        ax.legend(fontsize=7); ax.grid(True); fig.tight_layout(); fig.savefig(FIGURES / "agreement_vs_dense.png", dpi=160); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=("prepare", "ot", "fit", "agreement", "finalize"), required=True)
    parser.add_argument("--solver", choices=("dense", "lowrank", "minibatch"))
    parser.add_argument("--n", choices=("2000", "4000", "8000", "16000", "full"))
    parser.add_argument("--rank", type=int, choices=(50, 200))
    parser.add_argument("--best-solver")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(); return
    if args.action == "agreement":
        run_agreement(); return
    if args.action == "finalize":
        run_finalize(args.best_solver); return
    if args.solver is None or args.n is None:
        parser.error("--solver and --n are required for ot/fit")
    n: int | str = args.n if args.n == "full" else int(args.n)
    if args.solver == "lowrank" and args.rank is None:
        parser.error("lowrank requires --rank 50 or 200")
    if args.solver != "lowrank" and args.rank is not None:
        parser.error("--rank is only valid for lowrank")
    if args.action == "ot":
        run_ot(args.solver, n, args.rank)
    else:
        run_fit(args.solver, n, args.rank)


if __name__ == "__main__":
    main()
