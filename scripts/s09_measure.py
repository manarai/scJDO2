#!/usr/bin/env python3
"""One-dataset-at-a-time final S09 generation, S08 measurement, and S05 audit.

The protected S08 and S05 sources are imported in-process.  Their mutable output
locations are redirected to v3-specific artifacts, leaving all accepted S08
files untouched.  Invoke ``--dataset PRESET --seed N`` in one detached command
per final data set, then invoke ``--compare`` once all 12 checkpoints exist,
then ``--gates``.  Revision 2: several ``--dataset`` processes may run at once;
shared runtime/manifest/oracle files are updated under an exclusive file lock
and scVelo ``recover_dynamics`` (S08 hard-codes ``n_jobs=8``) is entered by at
most one process at a time.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import os
import importlib.util
import json
import resource
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CAL = ROOT / "benchmarks/synthetic_multimodal/v3_calibration"
DATA = ROOT / "benchmarks/synthetic_multimodal/data/v3"
MANIFEST = ROOT / "benchmarks/synthetic_multimodal/MANIFEST_v3.csv"
RUNTIME = CAL / "s08_runtime"
CHECKPOINTS = CAL / "measure_checkpoints"
ORACLES = CAL / "oracles_v3.csv"
COMPARE_OUT = CAL / "compare_v3.csv"
FIG = ROOT / "benchmarks/figures/real_reference/v3"
REFERENCE = ROOT / "benchmarks/real_reference"
V3 = ("v3_default", "v3_stress", "v3_batch", "v3_stationary_cycle")
BUDGET_CPU_SECONDS = 40 * 60 * 60  # amended in review Revision 3 (whole process trees)
GATES = CAL / "gates_v3.json"
TARGETS_BY_SEED = CAL / "targets_met_by_seed.csv"
LOCKS = CAL / ".locks"


@contextlib.contextmanager
def file_lock(name: str):
    LOCKS.mkdir(parents=True, exist_ok=True)
    with (LOCKS / f"{name}.lock").open("a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def install_recovery_mutex() -> None:
    """Gate entry to scVelo recover_dynamics across processes (arguments untouched)."""
    import scvelo as scv
    import scvelo.tools as tools
    original = tools.recover_dynamics
    if getattr(original, "_s09_mutex", False):
        return

    def gated(*args: Any, **kwargs: Any) -> Any:
        with file_lock("recover_dynamics"):
            return original(*args, **kwargs)

    gated._s09_mutex = True  # type: ignore[attr-defined]
    tools.recover_dynamics = gated
    scv.tl.recover_dynamics = gated


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_writer() -> Any:
    return load_module("s09_writer", ROOT / "scripts/make_synthetic_multimodal.py")


def load_s08() -> Any:
    module = load_module("s09_s08", ROOT / "scripts/s08_reference_stats.py")
    module.OUT = RUNTIME
    module.CHECKPOINTS = RUNTIME / "checkpoints"
    original_append = module.append_stats

    def locked_append(records: list[dict[str, Any]], *, group: str | None = None) -> None:
        with file_lock("runtime_stats"):
            original_append(records, group=group)

    module.append_stats = locked_append
    return module


def load_compare() -> Any:
    module = load_module("s09_compare", ROOT / "scripts/s08_compare.py")
    module.OUT = RUNTIME
    module.FIG = FIG
    module.SYN_PREFIXES = ("v1_", "v2_", "v3_")
    return module


def load_audit() -> Any:
    return load_module("s09_audit", ROOT / "scripts/audit_synthetic.py")


def rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime + usage.ru_stime)


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def ensure_runtime() -> None:
    """Seed a private S08 runtime copy with real/v1/v2 records and distributions."""
    distributions = RUNTIME / "distributions"
    if not (RUNTIME / "stats.csv").exists():
        RUNTIME.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REFERENCE / "stats.csv", RUNTIME / "stats.csv")
    if not distributions.exists():
        shutil.copytree(REFERENCE / "distributions", distributions)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)


def update_manifest(writer: Any, row: dict[str, object]) -> None:
    rows: list[dict[str, str]] = []
    if MANIFEST.exists():
        with MANIFEST.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    seed = str(row["seed"])
    rows = [old for old in rows if not (old["preset"] == str(row["preset"]) and old["seed"] == seed)]
    rows.append({key: str(value) for key, value in row.items()})
    rank = {preset: index for index, preset in enumerate(V3)}
    rows.sort(key=lambda value: (rank.get(value["preset"], 99), int(value["seed"])))
    writer.write_manifest(rows, MANIFEST)


def _rename_distributions(temp_key: str, final_key: str) -> None:
    directory = RUNTIME / "distributions"
    for source in directory.glob(f"{temp_key}__*.csv"):
        suffix = source.name[len(temp_key):]
        source.replace(directory / f"{final_key}{suffix}")


def _rewrite_integrity_key(temp_key: str, final_key: str) -> None:
    path = RUNTIME / "stats.csv"
    table = pd.read_csv(path)
    table.loc[table["dataset"] == temp_key, "dataset"] = final_key
    table.to_csv(path, index=False)


def run_s08_for_dataset(path: Path, final_key: str) -> dict[str, Any]:
    """Apply S08’s imported statistic functions to one v3 H5AD, privately."""
    with file_lock("ensure_runtime"):
        ensure_runtime()
    s08 = load_s08()
    temp_key = f"v2_{final_key}"  # makes S08 apply its unmodified synthetic schema branch.
    dataset = s08.Dataset(temp_key, path, "synthetic", "MANIFEST_v3.csv")
    integrity = s08.record_integrity(dataset)
    with file_lock("runtime_stats"):
        _rewrite_integrity_key(temp_key, final_key)
    group_rows: dict[str, int] = {}
    functions = {
        "counts_trajectory": s08.run_counts_trajectory,
        "splicing": s08.run_splicing,
        "labelling": s08.run_labelling,
        "lineage": s08.run_lineage,
    }
    for group, fn in functions.items():
        records = fn(dataset)
        for record in records:
            record["dataset"] = final_key
            record["source"] = "MANIFEST_v3.csv; imported S08 function"
        _rename_distributions(temp_key, final_key)
        s08.append_stats(records, group=group)
        group_rows[group] = len(records)
    return {"integrity": integrity, "group_rows": group_rows}


def write_oracles(path: Path, final_key: str) -> list[dict[str, object]]:
    """Run frozen S05 O1--O4 post hoc and apply the separate S09 gate labels."""
    audit = load_audit()
    rows = audit.audit_one(path, run_scvelo=True)
    for row in rows:
        row["dataset"] = final_key
        if row["oracle"] == "O1":
            row["s09_threshold"] = 0.30
            row["s09_gate_status"] = "pass" if float(row["median_cosine"]) >= 0.30 else "fail"
        elif row["oracle"] == "O2":
            row["s09_threshold"] = 0.20
            row["s09_gate_status"] = "pass" if float(row["median_cosine"]) >= 0.20 else "fail"
        else:
            row["s09_threshold"] = ""
            row["s09_gate_status"] = "report_only"
    with file_lock("oracles"):
        _merge_oracles(rows, final_key)
    return rows


def _merge_oracles(rows: list[dict[str, object]], final_key: str) -> None:
    prior: list[dict[str, str]] = []
    if ORACLES.exists():
        with ORACLES.open(newline="") as handle:
            prior = list(csv.DictReader(handle))
    prior = [row for row in prior if row.get("dataset") != final_key]
    fields = [
        "dataset", "oracle", "median_cosine", "n_valid", "threshold", "raw_pass", "gate_status",
        "s09_threshold", "s09_gate_status", "details_json", "wall_seconds", "peak_rss_bytes", "path",
    ]
    prior.extend({key: row.get(key, "") for key in fields} for row in rows)
    ORACLES.parent.mkdir(parents=True, exist_ok=True)
    with ORACLES.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(prior)


def calibration_cpu_seconds() -> float:
    path = CAL / "calibration_log.csv"
    if not path.exists():
        return 0.0
    # Every calibration attempt consumes the common S09 CPU budget, including
    # retried or terminally excluded scVelo failures.
    with path.open(newline="") as handle:
        return sum(float(row.get("cpu_seconds") or 0.0) for row in csv.DictReader(handle))


def measurement_cpu_seconds() -> float:
    total = 0.0
    for path in CHECKPOINTS.glob("*.json"):
        try:
            total += float(json.loads(path.read_text()).get("cpu_seconds", 0.0))
        except (ValueError, OSError):
            continue
    return total


def enforce_budget() -> None:
    spent = calibration_cpu_seconds() + measurement_cpu_seconds()
    if spent >= BUDGET_CPU_SECONDS:
        raise RuntimeError(f"S09 CPU budget exhausted ({spent / 3600:.3f} h); stop and ask")


def manifest_row(preset: str, seed: int) -> dict[str, str]:
    with MANIFEST.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["preset"] == preset and row["seed"] == str(seed):
                return row
    raise RuntimeError(f"{preset} seed {seed} has no MANIFEST_v3 row; run --part generate first")


def measure_one(preset: str, seed: int, part: str = "all") -> dict[str, Any]:
    """part=generate writes the H5AD + manifest row; part=measure runs S08 (+S05
    for v3_default) on it; part=all does both in one process."""
    enforce_budget()
    started_wall, started_cpu = time.time(), cpu_seconds()
    writer = load_writer()
    install_recovery_mutex()
    final_key = f"{preset}_s{seed}"
    if part in ("generate", "all"):
        row = writer.write_dataset(preset, seed, DATA)
        with file_lock("manifest"):
            update_manifest(writer, row)
        if part == "generate":
            payload = {"task": "orient/S09", "dataset": final_key, "part": "generate", "manifest_row": row,
                       "wall_seconds": round(time.time() - started_wall, 3), "cpu_seconds": round(cpu_seconds() - started_cpu, 3),
                       "peak_rss_bytes": rss_bytes(), "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            json_dump(CAL / "generate_checkpoints" / f"{final_key}.json", payload)
            return payload
    else:
        row = manifest_row(preset, seed)
        expected = ROOT / str(row["path"])
        if writer.sha256(expected) != row["sha256"]:
            raise RuntimeError(f"{expected} does not match its MANIFEST_v3 sha256")
    path = ROOT / str(row["path"])
    s08 = run_s08_for_dataset(path, final_key)
    oracle_rows: list[dict[str, object]] = []
    if preset == "v3_default":
        oracle_rows = write_oracles(path, final_key)
    payload = {
        "task": "orient/S09", "dataset": final_key, "path": str(row["path"]), "manifest_row": row,
        "s08": s08, "oracles": oracle_rows, "wall_seconds": round(time.time() - started_wall, 3),
        "cpu_seconds": round(cpu_seconds() - started_cpu, 3), "peak_rss_bytes": rss_bytes(),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cumulative_cpu_hours": (calibration_cpu_seconds() + measurement_cpu_seconds()) / 3600.0,
    }
    json_dump(CHECKPOINTS / f"{final_key}.json", payload)
    return payload


def compare_all() -> dict[str, Any]:
    required = {f"{preset}_s{seed}" for preset in V3 for seed in range(3)}
    completed = {path.stem for path in CHECKPOINTS.glob("*.json")}
    missing = sorted(required - completed)
    if missing:
        raise RuntimeError(f"cannot compare until all per-dataset checkpoints exist: {missing}")
    ensure_runtime()
    module = load_compare()
    module.main()
    shutil.copy2(RUNTIME / "compare.csv", COMPARE_OUT)
    table = pd.read_csv(COMPARE_OUT)
    summary = table.loc[table["synthetic"].str.startswith("v3_")].groupby("preset").agg(
        comparisons=("statistic", "size"), matched=("match", "sum"), match_rate=("match", "mean"),
    ).reset_index().to_dict(orient="records")
    payload = {"task": "orient/S09", "comparison": str(COMPARE_OUT.relative_to(ROOT)), "figures": len(list(FIG.glob("*.png"))), "summary": summary}
    json_dump(CHECKPOINTS / "compare.json", payload)
    return payload


def gates() -> dict[str, Any]:
    """Realism gate (13 statistics, v3_default seeds 0-2) and O1/O2 signal gate."""
    cal = load_module("s09_cal_targets", ROOT / "scripts/s09_calibrate.py")
    stats = pd.read_csv(RUNTIME / "stats.csv")
    rows: list[dict[str, Any]] = []
    met_by_seed: dict[str, dict[str, bool]] = {}
    for preset in V3:
        for seed in range(3):
            key = f"{preset}_s{seed}"
            table = stats.loc[stats["dataset"] == key, ["statistic", "feature", "value"]]
            values = cal.statistic_values(table)
            met_by_seed[key] = {}
            for label, value in values.items():
                # Revision 5: realism gate uses the re-ranking bands (single-reference
                # targets x [1/1.25, 1.25]); the earlier bands are kept for reference.
                lo, hi = cal.INTERVALS_R5[label]
                met = bool(pd.notna(value) and lo <= value <= hi)
                plo, phi = cal.INTERVALS[label]
                met_by_seed[key][label] = met
                rows.append({"dataset": key, "preset": preset, "seed": seed, "statistic": label, "value": value,
                             "target_lo": lo, "target_hi": hi, "met": met,
                             "met_previous_bands": bool(pd.notna(value) and plo <= value <= phi),
                             "gated": preset == "v3_default"})
    pd.DataFrame(rows).to_csv(TARGETS_BY_SEED, index=False)
    default = {k: v for k, v in met_by_seed.items() if k.startswith("v3_default_")}
    oracles = pd.read_csv(ORACLES)
    signal = oracles.loc[oracles["dataset"].str.startswith("v3_default_") & oracles["oracle"].isin(["O1", "O2"]),
                         ["dataset", "oracle", "median_cosine", "s09_threshold", "s09_gate_status"]]
    payload = {
        "task": "orient/S09",
        "realism": {"pass": bool(len(default) == 3 and all(all(v.values()) for v in default.values())),
                    "met_count_by_seed": {k: int(sum(v.values())) for k, v in default.items()},
                    "unmet_by_seed": {k: sorted(s for s, ok in v.items() if not ok) for k, v in default.items()},
                    "n_statistics": len(cal.TARGET_STATS)},
        "signal_O1_O2": {"pass": bool(len(signal) == 6 and (signal["s09_gate_status"] == "pass").all()),
                         "values": signal.to_dict(orient="records")},
        "targets_met_by_seed_all_presets": {k: int(sum(v.values())) for k, v in met_by_seed.items()},
    }
    json_dump(GATES, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=V3)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--gates", action="store_true")
    parser.add_argument("--part", choices=("all", "generate", "measure"), default="all")
    args = parser.parse_args()
    if args.gates:
        print(json.dumps(gates(), indent=2, sort_keys=True, default=str))
        return
    if args.compare:
        if args.dataset is not None or args.seed is not None:
            parser.error("--compare cannot be combined with --dataset/--seed")
        print(json.dumps(compare_all(), indent=2, sort_keys=True))
        return
    if args.dataset is None or args.seed is None:
        parser.error("supply --dataset PRESET --seed N, or --compare")
    print(json.dumps(measure_one(args.dataset, args.seed, args.part), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
