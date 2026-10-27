#!/usr/bin/env python3
"""S09 calibration using only S08 statistics as the objective input.

One invocation is one checkpointable unit: an LHS batch, a single coordinate
sweep, or (Revision 2, parallel runs) a single evaluation selected with
``--index``.  Concurrent single-evaluation processes share the calibration log
under an exclusive file lock; each candidate gets a private S08 runtime
directory, and at most one process at a time is inside scVelo's
``recover_dynamics`` (S08 hard-codes ``n_jobs=8``), so the thread total stays
within the task's limit.  The runner imports S08's statistic functions rather
than duplicating them and redirects the imported module's runtime output below
``v3_calibration`` so no S08 artifact is modified.
"""
from __future__ import annotations

import argparse
import csv
import contextlib
import fcntl
import importlib.util
import json
import os
import resource
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import qmc

from scjdo2.data.synthetic_multimodal import (
    V3_DEFAULT_KNOBS,
    _gene_parameters_v3,
    _make_v3_config,
    _simulate_v3_latent,
    make_multimodal,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks/synthetic_multimodal/v3_calibration"
RUNTIME = OUT / "s08_runtime"
CHECKPOINTS = OUT / "checkpoints"
LOG = OUT / "calibration_log.csv"
COMPARE = ROOT / "benchmarks/real_reference/compare.csv"
LHS_SEED = 9009
N_LHS = 64
TARGET_REDUCED_CELLS = 1500
BUDGET_CPU_SECONDS = 40 * 60 * 60  # amended in review Revision 3 (whole process trees)

# This is the exact 13-knob order stated by S09.  Bounds here are not fitted or
# widened; they only define the prescribed LHS and coordinate values.
KNOBS = (
    "mu_b", "sigma_b", "f_constant", "f_switch", "n_prog", "theta0",
    "gamma0", "sigma_gamma", "beta0", "eta_u", "p_barcode", "n_founders", "f_sample",
)
BOUNDS: dict[str, tuple[float, float, str]] = {
    "mu_b": (-4.0, 1.0, "float"),
    "sigma_b": (0.5, 2.5, "float"),
    "f_constant": (0.0, 0.6, "float"),
    "f_switch": (0.0, 0.6, "float"),
    "n_prog": (0.0, 8.0, "int"),
    "theta0": (1.0, 50.0, "float"),
    "gamma0": (0.05, 0.5, "float"),
    "sigma_gamma": (0.3, 1.5, "float"),
    "beta0": (0.1, 3.0, "float"),
    "eta_u": (0.2, 1.0, "float"),
    "p_barcode": (0.2, 0.8, "float"),
    "n_founders": (200.0, 5000.0, "int"),
    "f_sample": (0.005, 0.3, "float"),
}

# Every objective value is an S08 statistic scalar or distribution median.
TARGET_STATS: dict[str, tuple[str, str]] = {
    "C2_genes_detected": ("C2_genes_detected", "median"),
    "C3_hvg_log_mean": ("C3_hvg_log_mean", "median"),
    "C4_dispersion": ("C4_dispersion", "median"),
    "C5_zero_fraction": ("C5_zero_fraction", "median"),
    "C7_participation_ratio": ("C7_participation_ratio", "value"),
    "S1_unspliced_fraction": ("S1_unspliced_fraction", "global"),
    "S3_steady_share": ("S3_phase_portrait_class", "steady"),
    "S4_recovered_fraction": ("S4_recovered_fraction", "fraction"),
    "L1_new_total_fraction": ("L1_new_total_fraction", "median"),
    "L2_gamma_half_life": ("L2_gamma_half_life", "median"),
    "G1_barcoded_cell_fraction": ("G1_barcoded_cell_fraction", "value"),
    "G2_clone_size": ("G2_clone_size", "median"),
    "G3_multi_time_clone_fraction": ("G3_multi_time_clone_fraction", "value"),
}


LOCKS = OUT / ".locks"


@contextlib.contextmanager
def file_lock(name: str):
    """Exclusive inter-process lock; never nested for the same name."""
    LOCKS.mkdir(parents=True, exist_ok=True)
    with (LOCKS / f"{name}.lock").open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


RECOVERY_TIMING: list[dict[str, float]] = []


def install_recovery_mutex() -> None:
    """Serialise scVelo recover_dynamics across processes (scheduling only).

    S08's ``run_splicing`` calls ``scv.tl.recover_dynamics(velo, n_jobs=8,
    max_iter=10)``; arguments and results are untouched, only entry is gated.
    """
    import scvelo.tools as tools
    import scvelo as scv

    original = tools.recover_dynamics
    if getattr(original, "_s09_mutex", False):
        return

    def gated(*args: Any, **kwargs: Any) -> Any:
        requested = time.time()
        with file_lock("recover_dynamics"):
            entered = time.time()
            try:
                return original(*args, **kwargs)
            finally:
                RECOVERY_TIMING.append({"wait_seconds": round(entered - requested, 3), "run_seconds": round(time.time() - entered, 3)})

    gated._s09_mutex = True  # type: ignore[attr-defined]
    tools.recover_dynamics = gated
    scv.tl.recover_dynamics = gated


def load_s08() -> Any:
    """Import the protected S08 module, then isolate its mutable runtime paths."""
    spec = importlib.util.spec_from_file_location("s08_for_s09", ROOT / "scripts/s08_reference_stats.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scripts/s08_reference_stats.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUT = RUNTIME
    module.CHECKPOINTS = RUNTIME / "checkpoints"
    return module


S08 = load_s08()


def rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def user_cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime + usage.ru_stime)


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def expected_collected_per_founder(f_sample: float, sampling_times: tuple[float, ...] = (48.0, 96.0, 144.0)) -> float:
    """Deterministic expectation used only to scale reduced non-lineage runs."""
    live = 1.0
    previous = 0.0
    collected = 0.0
    for t in sampling_times:
        live *= 2.0 ** ((t - previous) / 16.0)
        taken = f_sample * live
        collected += taken
        live -= taken
        previous = t
    return collected


def reduced_founders(vector: dict[str, Any]) -> int:
    # n_founders is intentionally not used for the reduced RNA/statistics path:
    # it is evaluated in the full-size lineage-only S08 run below.
    per_founder = expected_collected_per_founder(float(vector["f_sample"]))
    return max(2, int(round(TARGET_REDUCED_CELLS / max(per_founder, 1e-12))))


def coerce_vector(raw: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in KNOBS:
        lo, hi, kind = BOUNDS[key]
        value = float(raw[key])
        value = min(hi, max(lo, value))
        result[key] = int(round(value)) if kind == "int" else float(value)
    # Shares implicitly partition the genome.  This projection is not another
    # knob or a widened bound; it keeps a sampled vector valid under S09's
    # stated "rest linear" identity.
    result["f_switch"] = min(float(result["f_switch"]), 1.0 - float(result["f_constant"]))
    return result


def lhs_vectors() -> list[dict[str, Any]]:
    sampler = qmc.LatinHypercube(d=len(KNOBS), seed=LHS_SEED)
    sample = sampler.random(n=N_LHS)
    rows: list[dict[str, Any]] = []
    for unit in sample:
        raw = {key: BOUNDS[key][0] + value * (BOUNDS[key][1] - BOUNDS[key][0]) for key, value in zip(KNOBS, unit)}
        rows.append(coerce_vector(raw))
    return rows


def coordinate_values(key: str) -> list[Any]:
    lo, hi, kind = BOUNDS[key]
    values = np.linspace(lo, hi, 5)
    if kind == "int":
        return [int(round(value)) for value in values]
    return [float(value) for value in values]


def target_intervals() -> dict[str, tuple[float, float]]:
    """Freeze intervals from S08 compare.csv without reading generator outputs."""
    compare = pd.read_csv(COMPARE)
    intervals: dict[str, tuple[float, float]] = {}
    for label, (statistic, _feature) in TARGET_STATS.items():
        if label == "S1_unspliced_fraction":
            intervals[label] = (0.25, 0.35)  # S09's explicit 0.30 +/- 0.05 target.
            continue
        if label == "S3_steady_share":
            intervals[label] = (0.12, 0.46)  # S09's explicit real-reference range.
            continue
        rows = compare.loc[compare["statistic"] == statistic]
        if rows.empty:
            raise RuntimeError(f"S08 compare is missing required target statistic {statistic}")
        row = rows.iloc[0]
        lo, hi = float(row["real_range_min"]), float(row["real_range_max"])
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise RuntimeError(f"S08 has no finite real range for {statistic}")
        # A single applicable real reference has min=max.  S08's fixed
        # comparison tolerance supplies the frozen finite interval required by
        # S09's scalar/median target rule; log-ratio uses multiplicative limits.
        if np.isclose(lo, hi):
            tolerance = float(row["fixed_tolerance"])
            if str(row["metric"]) == "log_ratio":
                lo, hi = lo * float(np.exp(-tolerance)), hi * float(np.exp(tolerance))
            else:
                lo, hi = lo - tolerance, hi + tolerance
        intervals[label] = (float(lo), float(hi))
    return intervals


INTERVALS = target_intervals()


def _lineage_only_dataset(vector: dict[str, Any], path: Path) -> None:
    """Run full-size v3 genealogy without RNA/truth scoring, for S08 G1--G3."""
    config = _make_v3_config("v3_default", 0, dict(vector))
    rng = np.random.default_rng(config.seed)
    genes = _gene_parameters_v3(config, rng)
    first = _simulate_v3_latent(config, rng, genes["ou_tau"])
    sampled = first["sampled"]
    # This is a measurement-only barcode draw.  It deliberately does not use
    # z_true, fate truth, an operator, an oracle, or an orientation result.
    barcode_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 0x534F39, 1]))
    detected = barcode_rng.random(len(sampled["clone"])) < float(config.p_barcode)
    clone_ids = np.asarray([f"clone_{i:04d}" for i in range(config.n_founders)], dtype=object)
    clone = np.where(detected, clone_ids[sampled["clone"]], None)
    obs = pd.DataFrame(index=[f"lineage_{int(i):07d}" for i in sampled["cell_id"]])
    obs["day"] = sampled["day"].astype(np.int16)
    obs["clone"] = pd.Categorical(clone, categories=list(clone_ids))
    # S08's run_lineage needs a categorical label to produce its report-only G5
    # record.  A constant label prevents the calibration from accessing fate
    # truth; objective() filters G5 out before any scoring.
    obs["fate_true"] = pd.Categorical(np.repeat("not_used", len(obs)))
    data = ad.AnnData(X=sparse.csr_matrix((len(obs), 0), dtype=np.float32), obs=obs)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.write_h5ad(path, compression="gzip")


def s08_statistics(vector: dict[str, Any], candidate_dir: Path) -> pd.DataFrame:
    """Return the S08 statistic table for one vector, no copied calculations."""
    candidate_dir.mkdir(parents=True, exist_ok=True)
    # Private S08 runtime outputs per candidate: concurrent evaluations never
    # share a distribution file (S08 writes them; nothing here reads them).
    S08.OUT = candidate_dir / "s08_runtime"
    S08.CHECKPOINTS = S08.OUT / "checkpoints"
    reduced = dict(vector)
    reduced["n_founders"] = reduced_founders(vector)
    reduced_path = candidate_dir / "reduced.h5ad"
    data = make_multimodal("v3_default", seed=0, **reduced)
    data.write_h5ad(reduced_path, compression="gzip")
    del data
    lineage_path = candidate_dir / "lineage_full.h5ad"
    _lineage_only_dataset(vector, lineage_path)

    reduced_dataset = S08.Dataset("v2_s09_reduced", reduced_path, "synthetic", "S09 imported S08")
    lineage_dataset = S08.Dataset("v2_s09_lineage", lineage_path, "synthetic", "S09 imported S08")
    records: list[dict[str, Any]] = []
    # These are direct imports/calls of S08's functions.  The returned records,
    # not latent/RNA truth, are the sole inputs to objective().
    records.extend(S08.run_counts_trajectory(reduced_dataset))
    records.extend(S08.run_splicing(reduced_dataset))
    records.extend(S08.run_labelling(reduced_dataset))
    records.extend(S08.run_lineage(lineage_dataset))
    table = pd.DataFrame(records)
    table["dataset"] = "candidate"
    table.to_csv(candidate_dir / "s08_stats.csv", index=False)
    return table


def statistic_values(table: pd.DataFrame) -> dict[str, float]:
    values: dict[str, float] = {}
    for label, (statistic, feature) in TARGET_STATS.items():
        row = table.loc[(table["statistic"] == statistic) & (table["feature"].astype(str) == feature), "value"]
        values[label] = float(row.iloc[0]) if len(row) else float("nan")
    return values


def objective(table: pd.DataFrame) -> tuple[int, float, dict[str, float], dict[str, bool]]:
    """Score only a supplied S08 statistic table; no other input is accepted."""
    allowed = {"dataset", "kind", "statistic", "feature", "value", "n", "source"}
    if not isinstance(table, pd.DataFrame) or not set(table.columns).issubset(allowed):
        raise AssertionError("calibration objective accepts only an S08 statistic table")
    forbidden = ("orientation", "operator", "oracle", "truth")
    joined = " ".join(str(column).lower() for column in table.columns)
    if any(word in joined for word in forbidden):
        raise AssertionError("objective table contains a prohibited non-S08 input")
    values = statistic_values(table)
    met: dict[str, bool] = {}
    normalized = 0.0
    for key, value in values.items():
        lo, hi = INTERVALS[key]
        good = bool(np.isfinite(value) and lo <= value <= hi)
        met[key] = good
        if not good:
            scale = max(hi - lo, abs((hi + lo) / 2.0) * 0.05, 1e-8)
            if not np.isfinite(value):
                normalized += 1e6
            elif value < lo:
                normalized += (lo - value) / scale
            else:
                normalized += (value - hi) / scale
    return int(sum(not x for x in met.values())), float(normalized), values, met


LOG_FIELDS = [
    "uid", "stage", "pass", "batch", "candidate_index", "attempt", "status", "ranking_eligible",
    "knobs_json", "reduced_n_founders", "unmet_targets", "normalized_distance", "statistics_json",
    "targets_met_json", "wall_seconds", "cpu_seconds", "peak_rss_bytes", "error", "reused_from",
]


def log_rows() -> list[dict[str, str]]:
    if not LOG.exists():
        return []
    with LOG.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_log(rows: list[dict[str, Any]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    temporary = LOG.with_suffix(".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in LOG_FIELDS} for row in rows)
    temporary.replace(LOG)


def normalize_retry_log() -> None:
    with file_lock("log"):
        _normalize_retry_log()


def _normalize_retry_log() -> None:
    """Migrate pre-policy first failures to retry_pending without ranking them."""
    rows = log_rows()
    if not rows:
        return
    changed = False
    attempts: dict[str, int] = {}
    for row in rows:
        uid = row.get("uid", "")
        attempts[uid] = attempts.get(uid, 0) + 1
        attempt = attempts[uid]
        if row.get("attempt") != str(attempt):
            row["attempt"] = str(attempt)
            changed = True
        status = row.get("status", "")
        if status == "complete":
            if row.get("ranking_eligible") != "true":
                row["ranking_eligible"] = "true"
                changed = True
        elif status == "error" and attempt == 1:
            row["status"] = "retry_pending"
            row["ranking_eligible"] = "false"
            row["unmet_targets"] = ""
            row["normalized_distance"] = ""
            changed = True
        elif status in {"retry_pending", "error"}:
            if row.get("ranking_eligible") != "false":
                row["ranking_eligible"] = "false"
                changed = True
    # Once a retry completes, retain the original failure but mark it as
    # resolved instead of leaving a stale pending state in reports/ledgers.
    histories: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        histories.setdefault(row.get("uid", ""), []).append(row)
    for history in histories.values():
        if len(history) > 1 and history[-1].get("status") == "complete":
            for earlier in history[:-1]:
                if earlier.get("status") == "retry_pending":
                    earlier["status"] = "retry_resolved"
                    earlier["ranking_eligible"] = "false"
                    changed = True
    if changed:
        _write_log(rows)


def existing_uids() -> set[str]:
    return {row["uid"] for row in log_rows() if row.get("status") == "complete"}


def history_for(uid: str) -> list[dict[str, str]]:
    return [row for row in log_rows() if row.get("uid") == uid]


def append_row(row: dict[str, Any]) -> None:
    with file_lock("log"):
        previous = log_rows()
        previous.append(row)
        _write_log(previous)


def completed_identical(vector: dict[str, Any]) -> dict[str, str] | None:
    """A completed candidate with the identical knob vector, if any."""
    key = json.dumps(vector, sort_keys=True)
    for row in log_rows():
        if (row.get("status") == "complete" and row.get("knobs_json") == key and not row.get("reused_from")
                and row.get("stage") != "block3_lineage"):  # lineage-only rows carry no RNA statistics
            return row
    return None


def best_completed_row(exclude_prefix: str | None = None) -> dict[str, str]:
    candidates = []
    for row in log_rows():
        if row.get("status") != "complete":
            continue
        if exclude_prefix and row.get("uid", "").startswith(exclude_prefix):
            continue
        try:
            candidates.append((int(row["unmet_targets"]), float(row["normalized_distance"]), row["uid"], row))
        except (KeyError, ValueError):
            continue
    if not candidates:
        raise RuntimeError("no completed calibration candidates are available")
    return min(candidates, key=lambda value: (value[0], value[1], value[2]))[3]


def best_completed() -> dict[str, Any]:
    candidates = []
    for row in log_rows():
        if row.get("status") != "complete":
            continue
        try:
            candidates.append((int(row["unmet_targets"]), float(row["normalized_distance"]), row["uid"], json.loads(row["knobs_json"])))
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
    if not candidates:
        raise RuntimeError("no completed calibration candidates are available")
    return min(candidates, key=lambda value: (value[0], value[1], value[2]))[3]


def cpu_ledger() -> dict[str, float]:
    rows = log_rows()
    # A failed scVelo attempt consumed CPU even though it is ineligible for the
    # calibration rank, so every real attempt must be charged to S09's budget.
    spent = sum(float(row.get("cpu_seconds") or 0.0) for row in rows)
    latest: dict[str, dict[str, str]] = {}
    for row in rows:
        latest[row.get("uid", "")] = row
    completed = sum(row.get("status") == "complete" for row in latest.values())
    pending = sum(row.get("status") == "retry_pending" for row in latest.values())
    excluded = sum(row.get("status") == "error" for row in latest.values())
    resolved = sum(any(item.get("status") == "retry_resolved" for item in rows if item.get("uid") == uid) for uid in latest)
    return {
        "completed_evaluations": float(completed), "attempted_evaluations": float(len(rows)),
        "retry_pending": float(pending), "retries_resolved": float(resolved), "excluded_after_retry": float(excluded),
        "cpu_seconds": spent, "cpu_hours": spent / 3600.0, "budget_cpu_hours": BUDGET_CPU_SECONDS / 3600.0,
    }


def evaluate(
    uid: str, stage: str, pass_number: int, batch: str, candidate_index: int,
    vector: dict[str, Any], *, retry: bool = False,
) -> dict[str, Any]:
    history = history_for(uid)
    if uid in existing_uids():
        return {"uid": uid, "status": "already_complete"}
    if history and not retry:
        return {"uid": uid, "status": "retry_required"}
    if len(history) >= 2:
        return {"uid": uid, "status": "excluded_after_retry"}
    attempt = len(history) + 1
    identical = completed_identical(vector) if attempt == 1 else None
    if identical is not None:
        # Never rerun a completed candidate: an identical knob vector (e.g. a
        # refinement value equal to the current best) reuses its S08 table.
        row = {
            "uid": uid, "stage": stage, "pass": pass_number, "batch": batch,
            "candidate_index": candidate_index, "attempt": attempt, "status": "complete", "ranking_eligible": "true",
            "knobs_json": identical["knobs_json"], "reduced_n_founders": identical["reduced_n_founders"],
            "unmet_targets": identical["unmet_targets"], "normalized_distance": identical["normalized_distance"],
            "statistics_json": identical["statistics_json"], "targets_met_json": identical["targets_met_json"],
            "wall_seconds": 0.0, "cpu_seconds": 0.0, "peak_rss_bytes": rss_bytes(), "error": "",
            "reused_from": identical["uid"],
        }
        append_row(row)
        return row
    started_wall, started_cpu = time.time(), user_cpu_seconds()
    RECOVERY_TIMING.clear()
    candidate_dir = OUT / "candidates" / uid / f"attempt_{attempt}"
    try:
        table = s08_statistics(vector, candidate_dir)
        unmet, distance, values, met = objective(table)
        row = {
            "uid": uid, "stage": stage, "pass": pass_number, "batch": batch,
            "candidate_index": candidate_index, "attempt": attempt, "status": "complete", "ranking_eligible": "true",
            "knobs_json": json.dumps(vector, sort_keys=True), "reduced_n_founders": reduced_founders(vector), "unmet_targets": unmet,
            "normalized_distance": distance, "statistics_json": json.dumps(values, sort_keys=True),
            "targets_met_json": json.dumps(met, sort_keys=True),
            "wall_seconds": round(time.time() - started_wall, 3),
            "cpu_seconds": round(user_cpu_seconds() - started_cpu, 3), "peak_rss_bytes": rss_bytes(), "error": "",
            "reused_from": "",
        }
    except Exception as exc:  # a candidate failure is logged; its unit checkpoint remains inspectable.
        row = {
            "uid": uid, "stage": stage, "pass": pass_number, "batch": batch,
            "candidate_index": candidate_index, "attempt": attempt,
            "status": "retry_pending" if attempt == 1 else "error", "ranking_eligible": "false",
            "knobs_json": json.dumps(vector, sort_keys=True), "reduced_n_founders": reduced_founders(vector),
            # Failed measurements never receive an objective score or rank.
            "unmet_targets": "", "normalized_distance": "", "statistics_json": "{}", "targets_met_json": "{}",
            "wall_seconds": round(time.time() - started_wall, 3),
            "cpu_seconds": round(user_cpu_seconds() - started_cpu, 3), "peak_rss_bytes": rss_bytes(),
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}",
            "reused_from": "",
        }
    append_row(row)
    return row


def run_lhs(batch: int, limit: int | None, index: int | None = None) -> list[dict[str, Any]]:
    start = batch * 8
    vectors = lhs_vectors()[start:start + 8]
    if not vectors:
        raise ValueError("LHS batch must be in 0..7")
    if index is not None:
        if not 0 <= index < len(vectors):
            raise ValueError("LHS --index must be in 0..7")
        return [evaluate(f"lhs_{start + index:03d}", "lhs", 0, str(batch), start + index, vectors[index])]
    if limit is not None:
        vectors = vectors[:limit]
    return [evaluate(f"lhs_{start + i:03d}", "lhs", 0, str(batch), start + i, vector) for i, vector in enumerate(vectors)]


def sweep_order() -> list[tuple[int, str]]:
    return [(pass_number, knob) for pass_number in (1, 2) for knob in KNOBS]


def require_prior_units_settled(pass_number: int, knob: str) -> None:
    """Refinement is sequential: every earlier unit must be complete or excluded."""
    latest: dict[str, str] = {}
    for row in log_rows():
        latest[row.get("uid", "")] = row.get("status", "")
    required = [f"lhs_{i:03d}" for i in range(N_LHS)]
    for earlier in sweep_order() + [(3, "")]:
        if earlier == (pass_number, knob):
            break
        required.extend(f"refine_p{earlier[0]}_{earlier[1]}_{i}" for i in range(5))
    unsettled = [uid for uid in required if latest.get(uid) not in {"complete", "error"}]
    if unsettled:
        raise RuntimeError(f"refinement p{pass_number} {knob} needs earlier units settled first: {unsettled[:8]}")


def retry_one(uid: str) -> dict[str, Any]:
    history = history_for(uid)
    if not history:
        raise ValueError(f"no calibration history for {uid}")
    latest = history[-1]
    if latest.get("status") != "retry_pending":
        raise RuntimeError(f"{uid} is not a first-attempt retry_pending evaluation")
    if uid.startswith("r5block3_"):
        return run_r5_lineage_point(int(uid.split("_")[1]), retry=True)
    if uid.startswith("block"):
        return run_block_point(int(uid[5]), int(uid.split("_")[1]), retry=True)
    return evaluate(
        uid, latest["stage"], int(latest["pass"]), latest["batch"], int(latest["candidate_index"]),
        json.loads(latest["knobs_json"]), retry=True,
    )


def run_refinement(pass_number: int, knob: str, only_index: int | None = None) -> list[dict[str, Any]]:
    if pass_number not in (1, 2):
        raise ValueError("refinement pass must be 1 or 2")
    if knob not in KNOBS:
        raise ValueError(f"unknown calibration knob {knob!r}")
    require_prior_units_settled(pass_number, knob)
    # The base is the best of every unit before this sweep; this sweep's own
    # (possibly concurrently finishing) candidates are excluded, so the base
    # does not depend on concurrency.
    base_row = best_completed_row(exclude_prefix=f"refine_p{pass_number}_{knob}_")
    base = json.loads(base_row["knobs_json"])
    REFINE_BASE.update(uid=base_row["uid"], knobs=base)
    rows = []
    for index, value in enumerate(coordinate_values(knob)):
        if only_index is not None and index != only_index:
            continue
        vector = dict(base)
        vector[knob] = value
        vector = coerce_vector(vector)
        rows.append(evaluate(f"refine_p{pass_number}_{knob}_{index}", "refine", pass_number, knob, index, vector))
    return rows


REFINE_BASE: dict[str, Any] = {}
RESCALE = OUT / "rescale_n_founders.json"
TARGET_CELLS = 5000
CELL_WINDOW = (4000, 6000)


def full_size_collected(vector: dict[str, Any], n_founders: int, seed: int = 0) -> int:
    """Collected cells of the full-size v3_default genealogy (lineage-only pass)."""
    config = _make_v3_config("v3_default", seed, {**vector, "n_founders": int(n_founders)})
    rng = np.random.default_rng(config.seed)
    genes = _gene_parameters_v3(config, rng)
    first = _simulate_v3_latent(config, rng, genes["ou_tau"])
    return int(np.sum(first["n_collected_by_day"]))


def run_rescale() -> dict[str, Any]:
    """Revision 3 size rule: keep the selected knobs, set n_founders (within
    bounds) so the seed-0 full-size genealogy collects closest to 5 000 cells.

    Search: proportional updates from the expectation-based start, then the
    +/-1, +/-2 neighbours of the best value; ties go to the smaller value.
    """
    require_prior_units_settled(3, "")
    selected_row = best_completed_row()
    vector = json.loads(selected_row["knobs_json"])
    lo, hi = int(BOUNDS["n_founders"][0]), int(BOUNDS["n_founders"][1])
    clamp = lambda n: int(min(hi, max(lo, int(round(n)))))
    counts: dict[int, int] = {}

    def count(n: int) -> int:
        if n not in counts:
            counts[n] = full_size_collected(vector, n)
        return counts[n]

    n = clamp(TARGET_CELLS / max(expected_collected_per_founder(float(vector["f_sample"])), 1e-12))
    for _ in range(6):
        c = count(n)
        nxt = clamp(n * TARGET_CELLS / max(c, 1))
        if nxt in counts:
            break
        n = nxt
    best = min(counts, key=lambda k: (abs(counts[k] - TARGET_CELLS), k))
    for delta in (-2, -1, 1, 2):
        cand = best + delta
        if lo <= cand <= hi:
            count(cand)
    best = min(counts, key=lambda k: (abs(counts[k] - TARGET_CELLS), k))
    final = dict(vector)
    final["n_founders"] = int(best)
    # G1-G3 re-check at the rescaled size with S08's imported lineage function.
    candidate_dir = OUT / "candidates" / "rescale"
    lineage_path = candidate_dir / "lineage_full.h5ad"
    _lineage_only_dataset(final, lineage_path)
    S08.OUT = candidate_dir / "s08_runtime"
    S08.CHECKPOINTS = S08.OUT / "checkpoints"
    table = pd.DataFrame(S08.run_lineage(S08.Dataset("v2_s09_lineage", lineage_path, "synthetic", "S09 rescale")))
    lineage = {}
    for label in ("G1_barcoded_cell_fraction", "G2_clone_size", "G3_multi_time_clone_fraction"):
        statistic, feature = TARGET_STATS[label]
        rows = table.loc[(table["statistic"] == statistic) & (table["feature"].astype(str) == feature), "value"]
        value = float(rows.iloc[0]) if len(rows) else float("nan")
        lo_t, hi_t = INTERVALS[label]
        lineage[label] = {"value": value, "target": [lo_t, hi_t], "met": bool(np.isfinite(value) and lo_t <= value <= hi_t),
                          "met_at_calibration_size": json.loads(selected_row["targets_met_json"]).get(label),
                          "value_at_calibration_size": json.loads(selected_row["statistics_json"]).get(label)}
    payload = {
        "task": "orient/S09", "rule": "Revision 3 size rule (seed-0 lineage-only genealogy)",
        "selected_uid": selected_row["uid"], "selected_knobs_before": vector, "final_knobs": final,
        "n_founders_counts": {str(k): v for k, v in sorted(counts.items())},
        "collected_cells": counts[best], "within_4000_6000": CELL_WINDOW[0] <= counts[best] <= CELL_WINDOW[1],
        "n_founders_at_bound": best in (lo, hi), "lineage_recheck": lineage,
    }
    json_dump(RESCALE, payload)
    return payload


# ---------------------------------------------------------------------------
# Revision 4 block refinement (task "Dataset size and budget", amended).
BLOCK_START_UID = "refine_p2_n_founders_3"
BLOCK_STATE = OUT / "block_state.json"
LINEAGE_LABELS = ("G1_barcoded_cell_fraction", "G2_clone_size", "G3_multi_time_clone_fraction")
SIZE_WINDOW = (4000, 15000)


def block_grid(block: int) -> tuple[tuple[str, str], list[tuple[Any, Any]]]:
    if block == 1:
        keys, a, b = ("beta0", "eta_u"), np.geomspace(0.1, 3.0, 5), np.linspace(0.2, 1.0, 5)
    elif block == 2:
        keys, a, b = ("gamma0", "sigma_gamma"), np.linspace(0.05, 0.5, 5), np.linspace(0.3, 1.5, 5)
    elif block == 3:
        keys, a, b = ("n_founders", "f_sample"), np.asarray([200, 300, 500, 800, 1200]), np.geomspace(0.005, 0.3, 8)
    elif block == 4:
        keys, a, b = ("mu_b", "sigma_b"), np.linspace(-4.0, 1.0, 5), np.linspace(0.5, 2.5, 5)
    else:
        raise ValueError("block must be 1..4")
    cast = (lambda v: int(v)) if keys[0] == "n_founders" else float
    return keys, [(cast(x), float(y)) for x in a for y in b]


def block_state() -> dict[str, Any]:
    if BLOCK_STATE.exists():
        return json.loads(BLOCK_STATE.read_text())
    start = [row for row in log_rows() if row.get("uid") == BLOCK_START_UID and row.get("status") == "complete"]
    if not start:
        raise RuntimeError(f"{BLOCK_START_UID} is not complete")
    row = start[-1]
    state = {"task": "orient/S09", "start_uid": BLOCK_START_UID, "completed_blocks": [], "current_uid": BLOCK_START_UID,
             "current_knobs": json.loads(row["knobs_json"]), "current_unmet": int(row["unmet_targets"]),
             "current_distance": float(row["normalized_distance"]), "current_met": json.loads(row["targets_met_json"]),
             "history": []}
    json_dump(BLOCK_STATE, state)
    return state


def block_vector(block: int, index: int) -> tuple[str, dict[str, Any]]:
    state = block_state()
    if state["completed_blocks"] != list(range(1, block)):
        raise RuntimeError(f"block {block} requires blocks {list(range(1, block))} selected first; have {state['completed_blocks']}")
    keys, grid = block_grid(block)
    if not 0 <= index < len(grid):
        raise ValueError(f"block {block} index must be in 0..{len(grid) - 1}")
    vector = dict(state["current_knobs"])
    vector[keys[0]], vector[keys[1]] = grid[index]
    return f"block{block}_{index:02d}", coerce_vector(vector)


def lineage_values(table: pd.DataFrame) -> dict[str, float]:
    return {key: value for key, value in statistic_values(table).items() if key in LINEAGE_LABELS}


def evaluate_lineage_only(uid: str, block: int, index: int, vector: dict[str, Any], *, retry: bool = False) -> dict[str, Any]:
    """Block 3: full-size lineage-only genealogy and S08 run_lineage (G1-G3, cells)."""
    history = history_for(uid)
    if uid in existing_uids():
        return {"uid": uid, "status": "already_complete"}
    if history and not retry:
        return {"uid": uid, "status": "retry_required"}
    if len(history) >= 2:
        return {"uid": uid, "status": "excluded_after_retry"}
    attempt = len(history) + 1
    started_wall, started_cpu = time.time(), user_cpu_seconds()
    candidate_dir = OUT / "candidates" / uid / f"attempt_{attempt}"
    base = {"uid": uid, "stage": "block3_lineage", "pass": 4, "batch": f"block{block}", "candidate_index": index,
            "attempt": attempt, "ranking_eligible": "false", "knobs_json": json.dumps(vector, sort_keys=True),
            "reduced_n_founders": "", "unmet_targets": "", "normalized_distance": "", "reused_from": ""}
    try:
        path = candidate_dir / "lineage_full.h5ad"
        _lineage_only_dataset(vector, path)
        cells = int(ad.read_h5ad(path, backed="r").n_obs)
        S08.OUT = candidate_dir / "s08_runtime"
        S08.CHECKPOINTS = S08.OUT / "checkpoints"
        table = pd.DataFrame(S08.run_lineage(S08.Dataset("v2_s09_lineage", path, "synthetic", "S09 block 3")))
        table["dataset"] = "candidate"
        table.to_csv(candidate_dir / "s08_stats.csv", index=False)
        objective(table)  # same S08-table-only assertion; lineage subset scored below
        values = lineage_values(table)
        met = {key: bool(np.isfinite(v) and INTERVALS[key][0] <= v <= INTERVALS[key][1]) for key, v in values.items()}
        row = {**base, "status": "complete", "statistics_json": json.dumps({**values, "collected_cells": cells}, sort_keys=True),
               "targets_met_json": json.dumps(met, sort_keys=True), "error": ""}
    except Exception as exc:
        row = {**base, "status": "retry_pending" if attempt == 1 else "error", "statistics_json": "{}", "targets_met_json": "{}",
               "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}"}
    row.update(wall_seconds=round(time.time() - started_wall, 3), cpu_seconds=round(user_cpu_seconds() - started_cpu, 3), peak_rss_bytes=rss_bytes())
    append_row(row)
    return row


def run_block_point(block: int, index: int, *, retry: bool = False) -> dict[str, Any]:
    uid, vector = block_vector(block, index)
    if block == 3:
        return evaluate_lineage_only(uid, block, index, vector, retry=retry)
    return evaluate(uid, f"block{block}", 4, f"block{block}", index, vector, retry=retry)


def _distance(values: dict[str, float], labels: tuple[str, ...]) -> float:
    total = 0.0
    for key in labels:
        lo, hi = INTERVALS[key]
        value = values.get(key, float("nan"))
        if not np.isfinite(value):
            total += 1e6
        elif not lo <= value <= hi:
            scale = max(hi - lo, abs((hi + lo) / 2.0) * 0.05, 1e-8)
            total += (lo - value) / scale if value < lo else (value - hi) / scale
    return total


def select_block(block: int) -> dict[str, Any]:
    """Apply the Revision 4 acceptance rule for one block (no unmet increase)."""
    state = block_state()
    if state["completed_blocks"] != list(range(1, block)):
        raise RuntimeError(f"block {block} cannot be selected now; completed {state['completed_blocks']}")
    _, grid = block_grid(block)
    latest: dict[str, dict[str, str]] = {}
    for row in log_rows():
        latest[row["uid"]] = row
    uids = [f"block{block}_{i:02d}" for i in range(len(grid))]
    unsettled = [u for u in uids if latest.get(u, {}).get("status") not in {"complete", "error"}]
    if unsettled:
        raise RuntimeError(f"block {block} has unsettled grid points: {unsettled}")
    rows = [latest[u] for u in uids if latest[u]["status"] == "complete"]
    errors = [u for u in uids if latest[u]["status"] == "error"]
    current_unmet = int(state["current_unmet"])
    decision: dict[str, Any] = {"block": block, "grid_points": len(grid), "complete": len(rows), "errors": errors,
                                "base_uid": state["current_uid"], "base_unmet": current_unmet}
    chosen: dict[str, str] | None = None
    if block == 3:
        points = []
        for row in rows:
            stats = json.loads(row["statistics_json"])
            met = json.loads(row["targets_met_json"])
            cells = int(stats["collected_cells"])
            points.append({"uid": row["uid"], "knobs": json.loads(row["knobs_json"]), "cells": cells,
                           "g_unmet": sum(not met[k] for k in LINEAGE_LABELS),
                           "g_distance": _distance(stats, LINEAGE_LABELS), "in_window": SIZE_WINDOW[0] <= cells <= SIZE_WINDOW[1]})
        window = [pt for pt in points if pt["in_window"]]
        decision["points_in_window"] = len(window)
        if not window:
            decision["result"] = "no grid point has 4000-15000 cells; block 3 not applied"
        else:
            best = min(window, key=lambda pt: (pt["g_unmet"], pt["g_distance"], pt["uid"]))
            decision["lineage_choice"] = best
            # Acceptance over all 13 targets needs one standard (reduced-size
            # RNA + full-size lineage) evaluation of the chosen vector.
            confirm = evaluate("block3_confirm", "block3_confirm", 4, "block3", 0, best["knobs"])
            if confirm.get("status") == "retry_pending":
                confirm = evaluate("block3_confirm", "block3_confirm", 4, "block3", 0, best["knobs"], retry=True)
            decision["confirm"] = {k: confirm.get(k) for k in ("uid", "status", "unmet_targets", "normalized_distance", "targets_met_json")}
            if confirm.get("status") == "complete" and int(confirm["unmet_targets"]) <= current_unmet:
                chosen = {k: str(v) for k, v in confirm.items()}
            else:
                decision["result"] = "confirmation increased the unmet count (or failed); block 3 not applied"
    else:
        def met_of(row: dict[str, str]) -> dict[str, bool]:
            return json.loads(row["targets_met_json"])
        eligible = [r for r in rows if int(r["unmet_targets"]) <= current_unmet]
        if block == 1:
            s1 = [r for r in rows if met_of(r)["S1_unspliced_fraction"]]
            decision["points_meeting_S1"] = len(s1)
            if not s1:
                decision["result"] = "no grid point meets S1; block 1 not applied"
            else:
                best = min(s1, key=lambda r: (int(r["unmet_targets"]), float(r["normalized_distance"]), r["uid"]))
                decision["s1_choice"] = {"uid": best["uid"], "unmet": int(best["unmet_targets"])}
                if int(best["unmet_targets"]) <= current_unmet:
                    chosen = best
                else:
                    decision["result"] = "best S1-meeting point increases the unmet count; block 1 not applied"
        else:
            prefer = ("L1_new_total_fraction", "L2_gamma_half_life") if block == 2 else ("C2_genes_detected", "C5_zero_fraction")
            if block == 4 and state["current_met"].get("C3_hvg_log_mean"):
                eligible = [r for r in eligible if met_of(r)["C3_hvg_log_mean"]]
            decision["eligible_points"] = len(eligible)
            if eligible:
                chosen = min(eligible, key=lambda r: (-sum(met_of(r)[k] for k in prefer), int(r["unmet_targets"]),
                                                     float(r["normalized_distance"]), r["uid"]))
                current_pref = sum(bool(state["current_met"].get(k)) for k in prefer)
                if (-sum(met_of(chosen)[k] for k in prefer), int(chosen["unmet_targets"]), float(chosen["normalized_distance"])) >= (
                        -current_pref, current_unmet, float(state["current_distance"])):
                    decision["result"] = "no grid point improves on the base under the block preference; base kept"
                    chosen = None
            else:
                decision["result"] = "no grid point keeps the unmet count; base kept"
    if chosen is not None:
        decision["result"] = f"accepted {chosen['uid']}"
        state.update(current_uid=chosen["uid"], current_knobs=json.loads(chosen["knobs_json"]),
                     current_unmet=int(chosen["unmet_targets"]), current_distance=float(chosen["normalized_distance"]),
                     current_met=json.loads(chosen["targets_met_json"]))
    state["completed_blocks"].append(block)
    state["history"].append(decision)
    json_dump(BLOCK_STATE, state)
    json_dump(CHECKPOINTS / f"block{block}_select.json", decision)
    return decision


# ---------------------------------------------------------------------------
# Revision 5 re-ranking (no new calibration runs) and lineage step.
RERANK = OUT / "rerank_r5.json"
RERANK_CSV = OUT / "rerank_r5.csv"
R5_STATE = OUT / "r5_lineage_select.json"
SINGLE_REFERENCE = ("L1_new_total_fraction", "L2_gamma_half_life", "G1_barcoded_cell_fraction",
                    "G2_clone_size", "G3_multi_time_clone_fraction")
R5_TOLERANCE = 1.25  # S08 fixed log-ratio tolerance, log(1.25)


def intervals_r5() -> dict[str, tuple[float, float]]:
    """Single-reference targets: real value(s) x [1/1.25, 1.25] (L1: both scNT-seq
    values, min/1.25 .. max*1.25); multi-reference targets keep the real min-max;
    S1 keeps 0.30 +/- 0.05 and S3 its explicit range."""
    compare = pd.read_csv(COMPARE)
    out = dict(INTERVALS)
    for label in SINGLE_REFERENCE:
        statistic = TARGET_STATS[label][0]
        row = compare.loc[compare["statistic"] == statistic].iloc[0]
        out[label] = (float(row["real_range_min"]) / R5_TOLERANCE, float(row["real_range_max"]) * R5_TOLERANCE)
    return out


INTERVALS_R5 = intervals_r5()


def score_r5(values: dict[str, float], labels: tuple[str, ...] | None = None) -> tuple[int, float, dict[str, bool]]:
    labels = labels or tuple(TARGET_STATS)
    met: dict[str, bool] = {}
    distance = 0.0
    for key in labels:
        lo, hi = INTERVALS_R5[key]
        value = float(values.get(key, float("nan")))
        good = bool(np.isfinite(value) and lo <= value <= hi)
        met[key] = good
        if not good:
            scale = max(hi - lo, abs((hi + lo) / 2.0) * 0.05, 1e-8)
            distance += 1e6 if not np.isfinite(value) else ((lo - value) / scale if value < lo else (value - hi) / scale)
    return int(sum(not v for v in met.values())), float(distance), met


def run_rerank() -> dict[str, Any]:
    """Re-rank every logged reduced-size evaluation: S1 met, fewest unmet, distance."""
    seen: dict[str, dict[str, Any]] = {}
    for row in log_rows():
        if row.get("status") != "complete" or row.get("stage") == "block3_lineage" or row.get("reused_from"):
            continue
        values = json.loads(row["statistics_json"])
        if set(TARGET_STATS) - set(values):
            continue
        unmet, distance, met = score_r5(values)
        seen.setdefault(row["knobs_json"], {
            "uid": row["uid"], "stage": row["stage"], "s1_met": met["S1_unspliced_fraction"], "unmet": unmet,
            "distance": distance, "unmet_targets": sorted(k for k, v in met.items() if not v),
            "S1": values["S1_unspliced_fraction"], "knobs_json": row["knobs_json"], "statistics_json": row["statistics_json"]})
    ranked = sorted(seen.values(), key=lambda r: (not r["s1_met"], r["unmet"], r["distance"], r["uid"]))
    pd.DataFrame(ranked).to_csv(RERANK_CSV, index=False)
    s1 = [r for r in ranked if r["s1_met"]]
    payload = {"task": "orient/S09", "revision": 5, "intervals_r5": INTERVALS_R5, "intervals_previous": INTERVALS,
               "unique_vectors_ranked": len(ranked), "vectors_meeting_S1": len(s1),
               "top5": [{k: r[k] for k in ("uid", "stage", "s1_met", "unmet", "distance", "unmet_targets", "S1")} for r in ranked[:5]],
               "winner": ({"uid": ranked[0]["uid"], "knobs": json.loads(ranked[0]["knobs_json"]),
                           "statistics": json.loads(ranked[0]["statistics_json"])} if s1 else None),
               "closest_S1_if_none": None if s1 else min(ranked, key=lambda r: abs(r["S1"] - 0.30))["uid"]}
    json_dump(RERANK, payload)
    return payload


def r5_vector(index: int) -> tuple[str, dict[str, Any]]:
    winner = json.loads(RERANK.read_text())["winner"]
    if winner is None:
        raise RuntimeError("no re-ranked vector meets S1; stop and report")
    _, grid = block_grid(3)
    vector = dict(winner["knobs"])
    vector["n_founders"], vector["f_sample"] = grid[index]
    return f"r5block3_{index:02d}", coerce_vector(vector)


def run_r5_lineage_point(index: int, *, retry: bool = False) -> dict[str, Any]:
    uid, vector = r5_vector(index)
    return evaluate_lineage_only(uid, 3, index, vector, retry=retry)


def select_r5_lineage() -> dict[str, Any]:
    _, grid = block_grid(3)
    latest = {row["uid"]: row for row in log_rows()}
    uids = [f"r5block3_{i:02d}" for i in range(len(grid))]
    unsettled = [u for u in uids if latest.get(u, {}).get("status") not in {"complete", "error"}]
    if unsettled:
        raise RuntimeError(f"r5 lineage grid unsettled: {unsettled}")
    points = []
    for u in uids:
        row = latest[u]
        if row["status"] != "complete":
            continue
        stats = json.loads(row["statistics_json"])
        g_unmet, g_distance, g_met = score_r5(stats, LINEAGE_LABELS)
        cells = int(stats["collected_cells"])
        points.append({"uid": u, "knobs": json.loads(row["knobs_json"]), "cells": cells, "g_unmet": g_unmet,
                       "g_distance": g_distance, "g_met": g_met, "g_values": {k: stats[k] for k in LINEAGE_LABELS},
                       "in_window": SIZE_WINDOW[0] <= cells <= SIZE_WINDOW[1]})
    window = [pt for pt in points if pt["in_window"]]
    best = min(window, key=lambda pt: (pt["g_unmet"], pt["g_distance"], pt["uid"])) if window else None
    payload = {"task": "orient/S09", "revision": 5, "winner_uid": json.loads(RERANK.read_text())["winner"]["uid"],
               "points": len(points), "errors": [u for u in uids if latest[u]["status"] == "error"],
               "points_in_window": len(window), "chosen": best, "final_knobs": best["knobs"] if best else None}
    json_dump(R5_STATE, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("lhs", "refine", "retry", "plan", "rescale", "block", "block-select", "rerank", "r5lineage", "r5lineage-select"), required=True)
    parser.add_argument("--batch", type=int, help="0..7 LHS batch")
    parser.add_argument("--retry-uid", help="retry one first-attempt failed candidate exactly once")
    parser.add_argument("--limit", type=int, help="optional bounded candidates from an LHS batch")
    parser.add_argument("--pass-number", type=int, choices=(1, 2), help="coordinate-refinement pass")
    parser.add_argument("--knob", choices=KNOBS, help="coordinate-refinement knob")
    parser.add_argument("--block", type=int, choices=(1, 2, 3, 4), help="Revision 4 block")
    parser.add_argument("--index", type=int, help="run one evaluation only (LHS 0..7 within the batch; refine 0..4)")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    normalize_retry_log()
    if args.stage == "plan":
        print(json.dumps({"intervals": INTERVALS, "lhs_seed": LHS_SEED, "lhs_examples": lhs_vectors()[:2], "ledger": cpu_ledger(), "expected_evaluations": 64 + 13 * 5 * 2}, indent=2, sort_keys=True))
        return
    if cpu_ledger()["cpu_seconds"] >= BUDGET_CPU_SECONDS:
        raise RuntimeError("S09 calibration CPU budget is exhausted; stop and ask before further compute")
    if args.stage == "rescale":
        print(json.dumps(run_rescale(), indent=2, sort_keys=True))
        return
    if args.stage == "rerank":
        print(json.dumps(run_rerank(), indent=2, sort_keys=True, default=str))
        return
    if args.stage in ("r5lineage", "r5lineage-select"):
        rows = [run_r5_lineage_point(args.index) if args.stage == "r5lineage" else select_r5_lineage()]
        normalize_retry_log()
        name = f"eval_{rows[0]['uid']}.json" if args.stage == "r5lineage" else "r5_lineage_select_run.json"
        json_dump(CHECKPOINTS / name, {"task": "orient/S09", "stage": args.stage, "rows": rows, "peak_rss_bytes": rss_bytes(),
                                       "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        print(json.dumps(rows, indent=2, sort_keys=True, default=str))
        return
    if args.stage in ("block", "block-select"):
        if args.block is None:
            parser.error("--block is required")
        install_recovery_mutex()
        if args.stage == "block":
            if args.index is None:
                parser.error("--index is required for --stage block")
            rows = [run_block_point(args.block, args.index)]
            checkpoint = CHECKPOINTS / f"eval_{rows[0]['uid']}.json"
        else:
            rows = [select_block(args.block)]
            checkpoint = CHECKPOINTS / f"block{args.block}_select_run.json"
        normalize_retry_log()
        json_dump(checkpoint, {"task": "orient/S09", "stage": args.stage, "rows": rows, "ledger": cpu_ledger(),
                               "recover_dynamics_timing": list(RECOVERY_TIMING), "peak_rss_bytes": rss_bytes(),
                               "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        print(json.dumps(rows, indent=2, sort_keys=True, default=str))
        return
    if args.stage == "lhs":
        if args.batch is None:
            parser.error("--batch is required for --stage lhs")
        install_recovery_mutex()
        rows = run_lhs(args.batch, args.limit, args.index)
        checkpoint = CHECKPOINTS / (f"lhs_{args.batch:02d}.json" if args.index is None else f"eval_{rows[0]['uid']}.json")
    elif args.stage == "retry":
        if not args.retry_uid:
            parser.error("--retry-uid is required for --stage retry")
        install_recovery_mutex()
        rows = [retry_one(args.retry_uid)]
        checkpoint = CHECKPOINTS / f"retry_{args.retry_uid}.json"
    else:
        if args.pass_number is None or args.knob is None:
            parser.error("--pass-number and --knob are required for --stage refine")
        install_recovery_mutex()
        rows = run_refinement(args.pass_number, args.knob, args.index)
        checkpoint = CHECKPOINTS / (f"refine_p{args.pass_number}_{args.knob}.json" if args.index is None else f"eval_{rows[0]['uid']}.json")
    # Convert a first-attempt pending record to retry_resolved immediately
    # when the second, same-vector retry succeeds, before checkpointing ledger.
    normalize_retry_log()
    payload = {
        "task": "orient/S09", "stage": args.stage, "rows": rows, "ledger": cpu_ledger(),
        "recover_dynamics_timing": list(RECOVERY_TIMING), "refine_base": dict(REFINE_BASE),
        "threads_env": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "LOKY_MAX_CPU_COUNT")},
        "peak_rss_bytes": rss_bytes(), "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    json_dump(checkpoint, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
