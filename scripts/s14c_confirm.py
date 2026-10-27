#!/usr/bin/env python3
"""S14c continuation: finish the frozen S14b confirmatory design safely."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import s14b_confirm as base

OUT = ROOT / "benchmarks/synthetic_multimodal/s14c"
CK = OUT / "checkpoints"
LOG = OUT / "logs"
MANIFEST = ROOT / "benchmarks/synthetic_multimodal/MANIFEST_v3.csv"
DATA = ROOT / "benchmarks/synthetic_multimodal/data/v3"
S14B = ROOT / "benchmarks/synthetic_multimodal/s14b"
REUSE_PREPARED = {
    "v3_nofork_s3": ROOT / "benchmarks/synthetic_multimodal/s14/arrays/prepared/v3_nofork_s3.npz",
    "v3_transition_s12": S14B / "arrays/prepared/v3_transition_s12.npz",
    "v3_transition_s16": S14B / "arrays/prepared/v3_transition_s16.npz",
}
REUSE_CHECKPOINTS = {
    "v3_nofork_s3": S14B / "checkpoints/prepare_v3_nofork_s3.json",
    "v3_transition_s12": S14B / "checkpoints/prepare_v3_transition_s12.json",
    "v3_transition_s16": S14B / "checkpoints/prepare_v3_transition_s16.json",
}
MAX_GENERATION_WALL = 60 * 60
MAX_GENERATION_RSS = 12_000_000_000
TRANSITION_RESERVE = range(17, 41)
NOFORK_SEEDS = range(3, 14)
DRIFT_SEEDS = (0, 1, 2)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def configure() -> None:
    base.OUT = OUT
    base.CK = CK
    base.LOG = LOG
    base.MANIFEST = MANIFEST
    base.DATA = DATA
    base.TRANSITION_RESERVE = TRANSITION_RESERVE
    base.core.OUT = OUT
    base.core.ARR = OUT / "arrays"
    base.core.CK = CK
    base.core.LOG = LOG
    base.core.prepared_path = prepared_path


def prepared_path(dataset: str) -> Path:
    """Read the explicitly reused S14/S14b arrays in place; new arrays stay in S14c."""
    if dataset in REUSE_PREPARED:
        path = REUSE_PREPARED[dataset]
        checkpoint = REUSE_CHECKPOINTS[dataset]
        if not path.is_file() or not checkpoint.is_file():
            raise RuntimeError(f"required reused preparation missing: {dataset}")
        record = json.loads(checkpoint.read_text())
        provenance = record.get("s14_checkpoint", record)
        expected = record.get("prepared_sha256") or provenance.get("prepared_sha256")
        if not expected or sha256(path) != expected:
            raise RuntimeError(f"reused prepared checksum mismatch: {dataset}")
        return path
    return OUT / "arrays" / "prepared" / f"{dataset}.npz"


def verify_reused_inputs() -> None:
    for dataset, path in REUSE_PREPARED.items():
        preset, seed = base.core.synthetic_id_parts(dataset)
        row = base.manifest_entry(preset, seed)
        if row is None:
            raise RuntimeError(f"reused manifest row missing: {dataset}")
        data_path = ROOT / row["path"]
        if not data_path.is_file() or sha256(data_path) != row["sha256"]:
            raise RuntimeError(f"reused data checksum mismatch: {dataset}")
        prepared_path(dataset)


def bootstrap_eligibility() -> Path:
    """Copy the committed S14b eligibility records into S14c once, read-only."""
    source = S14B / "g2_transition_eligibility.csv"
    dest = OUT / "g2_transition_eligibility.csv"
    if dest.exists():
        return dest
    if not source.is_file():
        raise RuntimeError("committed S14b eligibility table missing")
    OUT.mkdir(parents=True, exist_ok=True)
    with source.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if [int(r["seed"]) for r in rows if r["eligible"].lower() == "true"] != [12, 16]:
        raise RuntimeError("S14b eligibility table does not contain exactly reused eligible seeds 12 and 16")
    with dest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return dest


def record_generation_failure(preset: str, seed: int, status: str, message: str) -> None:
    if preset != "v3_transition":
        path = OUT / "generation_failures.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        if path.exists():
            with path.open(newline="") as f:
                rows = list(csv.DictReader(f))
        rows = [r for r in rows if not (r["preset"] == preset and int(r["seed"]) == seed)]
        rows.append({"preset": preset, "seed": str(seed), "status": status, "error": message})
        rows.sort(key=lambda r: (r["preset"], int(r["seed"])))
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["preset", "seed", "status", "error"], lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        return
    dest = bootstrap_eligibility()
    with dest.open(newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = [r for r in reader if int(r["seed"]) != seed]
    row = {k: "" for k in fields}
    row.update({"preset": "v3_transition", "seed": str(seed),
                "dataset": f"v3_transition_s{seed}", "eligible": "False",
                "generation_status": status, "generation_error": message})
    rows.append(row)
    rows.sort(key=lambda r: int(r["seed"]))
    with dest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def write_guard_checkpoint(preset: str, seed: int, status: str, started: float, cpu0: float,
                           peak_rss: int, attempts: int, message: str) -> None:
    CK.mkdir(parents=True, exist_ok=True)
    row = base.manifest_entry("v3_transition", seed)
    payload = {
        "task": "orient/S14c", "unit": f"generate_{preset}_s{seed}",
        "status": status, "started_unix": started, "finished_unix": time.time(),
        "wall_seconds": time.time() - started,
        "cpu_seconds": time.process_time() - cpu0,
        "peak_rss_bytes": int(peak_rss), "attempts": attempts,
        "dataset_created": bool(row), "manifest_row": row,
        "error": message,
    }
    (CK / f"generate_{preset}_s{seed}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sample_rss(pid: int) -> int:
    result = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                            capture_output=True, text=True, check=False)
    raw = result.stdout.strip()
    return int(raw.split()[0]) * 1024 if raw else 0


def run_generate_attempt(preset: str, seed: int, attempt: int) -> tuple[str, float, int, str]:
    """Run one generator child, enforcing the task's 60-minute/12-GB guard."""
    LOG.mkdir(parents=True, exist_ok=True)
    log_path = LOG / f"generate_{preset}_s{seed}_attempt{attempt}.log"
    command = [str(ROOT / ".sandbox/bin/python"), str(Path(__file__)),
               "generate-once", "--preset", preset, "--seed", str(seed)]
    started = time.monotonic()
    peak = 0
    with log_path.open("w") as log:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        status = "completed"
        detail = ""
        while proc.poll() is None:
            current = sample_rss(proc.pid)
            peak = max(peak, current)
            elapsed = time.monotonic() - started
            if current >= MAX_GENERATION_RSS:
                status = "guard_stop"
                detail = f"RSS reached guard threshold ({current} bytes >= {MAX_GENERATION_RSS})"
            elif elapsed >= MAX_GENERATION_WALL:
                status = "guard_stop"
                detail = f"wall time reached guard threshold ({elapsed:.2f} s >= {MAX_GENERATION_WALL})"
            else:
                time.sleep(0.5)
                continue
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            break
        return_code = proc.wait()
    elapsed = time.monotonic() - started
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    peak = max(peak, int(usage.ru_maxrss))
    if status == "completed" and return_code != 0:
        status = "exception"
        detail = f"generator process exited {return_code}; see {log_path.relative_to(ROOT)}"
    return status, elapsed, peak, detail


def generate_guarded(preset: str, seed: int) -> None:
    if preset == "v3_transition" and seed not in TRANSITION_RESERVE:
        raise ValueError(f"transition seed must be in {TRANSITION_RESERVE.start}–{TRANSITION_RESERVE.stop - 1}")
    if preset == "v3_nofork" and seed not in NOFORK_SEEDS:
        raise ValueError("no-fork seed must be in 3–13")
    if preset not in ("v3_transition", "v3_nofork"):
        raise ValueError(preset)
    configure()
    started, cpu0 = time.time(), time.process_time()
    existing = base.manifest_entry(preset, seed)
    if existing:
        base.generate(preset, seed)
        return
    attempts = 0
    total_wall = 0.0
    peak = 0
    last_error = ""
    while attempts < 2:
        attempts += 1
        status, elapsed, rss, detail = run_generate_attempt(preset, seed, attempts)
        total_wall += elapsed
        peak = max(peak, rss)
        if status == "completed":
            checkpoint = CK / f"generate_{preset}_s{seed}.json"
            if checkpoint.is_file():
                rec = json.loads(checkpoint.read_text())
                rec.update({"status": "completed", "attempts": attempts,
                            "guard_wall_limit_seconds": MAX_GENERATION_WALL,
                            "guard_rss_limit_bytes": MAX_GENERATION_RSS,
                            "observed_child_peak_rss_bytes": peak})
                checkpoint.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
            return
        if status == "guard_stop":
            last_error = detail
            write_guard_checkpoint(preset, seed, "guard_stop", started, cpu0, peak, attempts, detail)
            record_generation_failure(preset, seed, "guard_stop", detail)
            return
        last_error = detail
        # If the child completed and appended a checksum-valid manifest entry before raising,
        # preserve that completed dataset instead of trying to overwrite it.
        row = base.manifest_entry(preset, seed)
        if row:
            base.generate(preset, seed)
            return
    write_guard_checkpoint(preset, seed, "generator_error", started, cpu0, peak, attempts, last_error)
    record_generation_failure(preset, seed, "generator_error", last_error)


def generate_once(preset: str, seed: int) -> None:
    configure()
    base.generate(preset, seed)


def summarize() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    eligibility_path = OUT / "g2_transition_eligibility.csv"
    base.require_committed(eligibility_path)
    base.require_committed(OUT / "null_calibration.csv")
    with eligibility_path.open(newline="") as f:
        elig = [r for r in csv.DictReader(f) if r["eligible"].lower() == "true"]
    selected = [int(r["seed"]) for r in elig[:4]]
    if len(selected) != 4:
        raise RuntimeError(f"S14c requires four eligible transitions; found {len(selected)}")
    transitions = [f"v3_transition_s{s}" for s in selected]
    heldout = [f"v3_nofork_s{s}" for s in base.NULL_HELD_OUT]
    calibration = [f"v3_nofork_s{s}" for s in base.NULL_CALIBRATION]
    rows: list[dict[str, Any]] = []
    for phase, datasets in (("null", calibration), ("heldout", transitions + heldout)):
        for dataset in datasets:
            for drift_seed in DRIFT_SEEDS:
                rows.append(base.fit_record(phase, dataset, drift_seed))
    nulls = {d: max(float(r["R"]) for r in rows if r["phase"] == "null" and r["drift_seed"] == d)
             for d in DRIFT_SEEDS}
    for r in rows:
        r["T_null"] = nulls[r["drift_seed"]]
        r["exceeds_T_null"] = bool(r["R"] > r["T_null"])
        r["within_one_bin"] = bool(abs(int(r["detected_bin"]) - int(r["true_bin"])) <= 1)
    tests = [r for r in rows if r["phase"] == "heldout"]
    gates: dict[str, Any] = {}
    for d in DRIFT_SEEDS:
        dr = [r for r in tests if r["drift_seed"] == d]
        trans = [r for r in dr if r["dataset"] in transitions]
        nofork = [r for r in dr if r["dataset"] in heldout]
        gates[str(d)] = {
            "X1": bool(len(trans) == 4 and all(r["R"] > nulls[d] for r in trans)),
            "X2": bool(len(nofork) == 3 and all(r["R"] <= nulls[d] for r in nofork)),
            "X3": bool(len(trans) == 4 and all(abs(int(r["detected_bin"]) - int(r["true_bin"])) <= 1 for r in trans)),
            "transition_R": {r["dataset"]: r["R"] for r in trans},
            "transition_detected_true_bin": {r["dataset"]: [r["detected_bin"], r["true_bin"]] for r in trans},
            "heldout_nofork_R": {r["dataset"]: r["R"] for r in nofork},
            "T_null": nulls[d],
        }
    gates["X4"] = bool(all(gates[str(d)][k] for d in (1, 2) for k in ("X1", "X2", "X3")))
    gates["overall"] = bool(all(gates[str(d)][k] for d in DRIFT_SEEDS for k in ("X1", "X2", "X3")))
    gates["eligible_seeds"] = selected
    gates["null_calibration_seeds"] = list(base.NULL_CALIBRATION)
    gates["heldout_nofork_seeds"] = list(base.NULL_HELD_OUT)
    gates["eligible_seed_rows"] = [r for r in elig if int(r["seed"]) in selected]
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "test.csv", index=False)
    (OUT / "gates.json").write_text(json.dumps(gates, indent=2, sort_keys=True) + "\n")

    fig_dir = ROOT / "benchmarks/figures/orient/s14c"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    colors = {transitions[0]: "#1769aa", heldout[0]: "#d55e00"}
    for dataset in (transitions[0], heldout[0]):
        rec = base.fit_record("heldout", dataset, 0)
        profile = np.asarray(rec["bin_medians_lambda1"], float)
        scale = max(float(np.nanmedian(np.abs(profile))), 1e-12)
        ax.plot(np.arange(base.N_BINS), profile / scale, marker="o", label=dataset, color=colors[dataset])
    ax.axhline(nulls[0], color="black", linestyle="--", label=f"T_null (d=0): {nulls[0]:.3f}")
    ax.set(xlabel="S14 pseudotime bin", ylabel="Median λ₁ / median |bin median|",
           title="S14c reference flow: first eligible transition and held-out no-fork")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "profiles.png", dpi=160)
    plt.close(fig)


def main() -> None:
    configure()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("bootstrap")
    g = sub.add_parser("generate")
    g.add_argument("--preset", choices=("v3_transition", "v3_nofork"), required=True)
    g.add_argument("--seed", type=int, required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--preset", choices=("v3_transition", "v3_nofork"), required=True)
    p.add_argument("--seed", type=int, required=True)
    e = sub.add_parser("eligibility")
    e.add_argument("--seed", type=int, required=True)
    f = sub.add_parser("fit")
    f.add_argument("--dataset", required=True)
    f.add_argument("--drift-seed", type=int, choices=DRIFT_SEEDS, required=True)
    f.add_argument("--phase", choices=("null", "heldout"), required=True)
    sub.add_parser("calibrate")
    sub.add_parser("summarize")
    once = sub.add_parser("generate-once")
    once.add_argument("--preset", required=True)
    once.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.command == "generate-once":
        generate_once(args.preset, args.seed)
    elif args.command == "bootstrap":
        verify_reused_inputs()
        print(bootstrap_eligibility())
    elif args.command == "generate":
        generate_guarded(args.preset, args.seed)
    elif args.command == "prepare":
        dataset = f"{args.preset}_s{args.seed}"
        if dataset in REUSE_PREPARED:
            prepared_path(dataset)
            raise RuntimeError(f"refusing to re-prepare reused dataset {dataset}")
        base.prepare(args.preset, args.seed)
    elif args.command == "eligibility":
        bootstrap_eligibility()
        base.eligibility(args.seed)
    elif args.command == "fit":
        base.fit(args.dataset, args.drift_seed, args.phase)
    elif args.command == "calibrate":
        base.calibrate()
    elif args.command == "summarize":
        summarize()


if __name__ == "__main__":
    main()
