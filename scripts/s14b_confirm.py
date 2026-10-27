#!/usr/bin/env python3
"""Confirmatory S14b transition test with pre-fit transition eligibility."""
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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import s14_choose_k as core
import s14_optimise as opt
import s14_test as test

OUT = ROOT / "benchmarks/synthetic_multimodal/s14b"
CK = OUT / "checkpoints"
LOG = OUT / "logs"
MANIFEST = ROOT / "benchmarks/synthetic_multimodal/MANIFEST_v3.csv"
DATA = ROOT / "benchmarks/synthetic_multimodal/data/v3"
core.OUT = OUT
core.ARR = OUT / "arrays"
core.CK = CK
core.LOG = LOG
def require_s14_preflight() -> dict[str, Any]:
    path = ROOT / "benchmarks/synthetic_multimodal/s14/checkpoints/preflight.json"
    if not path.is_file() or not core.read_json(path).get("passed"):
        raise RuntimeError("the fixed S14 preflight checkpoint is required")
    return core.read_json(path)
core.require_preflight = require_s14_preflight
TRANSITION_RESERVE = range(5, 17)
NULL_CALIBRATION = range(3, 11)
NULL_HELD_OUT = range(11, 14)
DRIFT_SEEDS = (0, 1, 2)
K = 15
N_BINS = 12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def checkpoint(name: str, started: float, cpu0: float, payload: dict[str, Any]) -> None:
    CK.mkdir(parents=True, exist_ok=True)
    r = resource.getrusage(resource.RUSAGE_SELF)
    rss = int(r.ru_maxrss) if sys.platform == "darwin" else int(r.ru_maxrss) * 1024
    rec = {"task": "orient/S14b", "unit": name, "started_unix": started,
           "finished_unix": time.time(), "wall_seconds": time.time() - started,
           "cpu_seconds": time.process_time() - cpu0, "peak_rss_bytes": rss, **payload}
    (CK / f"{name}.json").write_text(json.dumps(jsonable(rec), indent=2, sort_keys=True) + "\n")


def manifest_entry(preset: str, seed: int) -> dict[str, str] | None:
    with MANIFEST.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["preset"] == preset and int(row["seed"]) == int(seed):
                return row
    return None


def generate(preset: str, seed: int) -> None:
    """Generate one full v3 dataset and append its manifest row."""
    if preset not in ("v3_transition", "v3_nofork"):
        raise ValueError(preset)
    started, cpu0 = time.time(), time.process_time()
    old = manifest_entry(preset, seed)
    if old is not None:
        path = ROOT / old["path"]
        if not path.is_file() or sha256(path) != old["sha256"]:
            raise RuntimeError(f"existing manifest entry fails sha256: {preset} s{seed}")
        checkpoint(f"generate_{preset}_s{seed}", started, cpu0,
                   {"reused_verified": True, "manifest_row": old})
        return
    path = DATA / f"{preset}_s{seed}.h5ad"
    if path.exists():
        raise RuntimeError(f"unmanifested output already exists; refusing to overwrite: {path}")
    spec = importlib.util.spec_from_file_location("s14b_writer", ROOT / "scripts/make_synthetic_multimodal.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the frozen v3 writer")
    writer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = writer
    spec.loader.exec_module(writer)
    row = writer.write_dataset(preset, seed, DATA)
    writer.append_manifest(row, MANIFEST)
    checkpoint(f"generate_{preset}_s{seed}", started, cpu0,
               {"reused_verified": False, "manifest_row": row})


def prepare(preset: str, seed: int) -> None:
    started, cpu0 = time.time(), time.process_time()
    core.prepare_synthetic(preset, seed)
    src = core.read_json(core.CK / f"prepare_{preset}_s{seed}.json")
    checkpoint(f"prepare_{preset}_s{seed}", started, cpu0,
               {"s14_checkpoint": src, "prepared_sha256": sha256(core.prepared_path(core.sid(preset, seed)))})


def eligibility_row(seed: int) -> dict[str, Any]:
    import anndata as ad

    dataset = f"v3_transition_s{seed}"
    prepared = core.prepared_path(dataset)
    prep_ck = core.CK / f"prepare_v3_transition_s{seed}.json"
    row = core.manifest_row("v3_transition", seed, verify=True)
    if not prepared.is_file() or not prep_ck.is_file():
        raise RuntimeError(f"prepare {dataset} before checking eligibility")
    prep = core.read_json(prep_ck)
    # `prepare` wraps the protected S14 helper checkpoint as `s14_checkpoint`;
    # accept either shape so root provenance is read from the actual prep record.
    provenance = prep.get("strict_pseudotime") or prep.get("s14_checkpoint", {}).get("strict_pseudotime", {})
    with np.load(prepared, allow_pickle=False) as p:
        names = np.asarray(p["names"]).astype(str)
        days = np.asarray(p["day"], int)
        pt = np.asarray(p["pseudotime_strict_r0"], float)
    bins = opt.day_weighted_bins(pt, days, n_bins=N_BINS)
    src = ad.read_h5ad(ROOT / row["path"], backed="r")
    try:
        truth_z = np.asarray(src.obsm["z_true"][:, :2], np.float64)
        meta = src.uns["synthetic_multimodal"]
        z1_crit = float(meta.get("z1_crit", 1.0))
    finally:
        src.file.close()
    signal = 0.5 * ((truth_z[:, 1] - z1_crit) - 3.0 * truth_z[:, 0] ** 2)
    med = np.asarray([np.median(signal[bins == b]) if np.any(bins == b) else np.nan
                      for b in range(N_BINS)], float)
    counts = np.bincount(bins, minlength=N_BINS)
    true_bin = int(np.nanargmax(med))
    rng = np.random.default_rng(seed)
    equal_peak = 0
    for _ in range(200):
        boot = np.full(N_BINS, np.nan)
        for b in range(N_BINS):
            idx = np.flatnonzero(bins == b)
            if len(idx):
                boot[b] = np.median(signal[rng.choice(idx, size=len(idx), replace=True)])
        if int(np.nanargmax(boot)) == true_bin:
            equal_peak += 1
    stability = equal_peak / 200.0
    given = str(provenance.get("given_root", ""))
    actual = str(provenance.get("actual_start_cell", ""))
    candidates = [str(x) for x in provenance.get("candidate_roots", [])]
    expected_key = f"day{int(days.min())}_strict_r0"
    provenance_complete = bool(
        provenance.get("key") == expected_key
        and provenance.get("root_rule")
        and given
        and actual
        and len(candidates) == 5
        and candidates[0] == given
        and set(candidates).issubset(set(names))
        and provenance.get("source")
        and provenance.get("settings")
    )
    criteria = {
        "strict_root_provenance_complete": provenance_complete,
        "all_bins_at_least_50_cells": bool(np.all(counts >= 50)),
        "true_peak_bin_2_to_9": bool(2 <= true_bin <= 9),
        "bootstrap_stability_at_least_90pct": bool(stability >= 0.90),
    }
    return {
        "preset": "v3_transition", "seed": seed, "dataset": dataset,
        "generation_status": "completed", "generation_error": "",
        "manifest_path": row["path"], "manifest_sha256": row["sha256"],
        "n_cells": int(len(names)), "given_root": given, "actual_start_cell": actual,
        "actual_differs_from_given": bool(actual and actual != given),
        "candidate_roots": json.dumps(candidates), "pseudotime_key": provenance.get("key", ""),
        "pseudotime_source": provenance.get("source", ""),
        "root_rule": provenance.get("root_rule", ""),
        "true_peak_bin": true_bin, "true_profile": json.dumps(jsonable(med)),
        "bin_counts": json.dumps(counts.tolist()), "bootstrap_equal_peak": equal_peak,
        "bootstrap_replicates": 200, "bootstrap_stability": stability,
        **criteria, "eligible": bool(all(criteria.values())),
    }


def eligibility(seed: int) -> None:
    started, cpu0 = time.time(), time.process_time()
    row = eligibility_row(seed)
    out = OUT / "g2_transition_eligibility.csv"
    existing = []
    if out.exists():
        with out.open(newline="") as f:
            existing = [r for r in csv.DictReader(f) if int(r["seed"]) != seed]
        for r in existing:
            r.setdefault("generation_status", "completed")
            r.setdefault("generation_error", "")
    rows = sorted(existing + [row], key=lambda r: int(r["seed"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        fields = list(rows[0])
        for r in rows[1:]:
            for key in r:
                if key not in fields:
                    fields.append(key)
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    checkpoint(f"eligibility_v3_transition_s{seed}", started, cpu0,
               {"row": row, "eligibility_csv_sha256": sha256(out)})


def transition_truth(dataset: str) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    import anndata as ad

    preset, seed = core.synthetic_id_parts(dataset)
    row = core.manifest_row(preset, seed, verify=True)
    ppath = core.prepared_path(dataset)
    with np.load(ppath, allow_pickle=False) as p:
        pt = np.asarray(p["pseudotime_strict_r0"], float)
        days = np.asarray(p["day"], int)
    src = ad.read_h5ad(ROOT / row["path"], backed="r")
    try:
        truth = core.s12().truth_for_scoring(src)
    finally:
        src.file.close()
    return truth, pt, {"days": days, "prepared_sha256": sha256(ppath), "input_sha256": row["sha256"]}


def require_committed(path: Path) -> None:
    import subprocess

    rel = str(path.relative_to(ROOT))
    subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT,
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT, check=True)


def fit(dataset: str, drift_seed: int, phase: str) -> None:
    if phase == "null" and dataset.startswith("v3_nofork_s"):
        require_committed(OUT / "g2_transition_eligibility.csv")
    elif phase == "heldout":
        require_committed(OUT / "g2_transition_eligibility.csv")
        require_committed(OUT / "null_calibration.csv")
    else:
        raise ValueError(phase)
    core.require_preflight()
    core.set_threads()
    started, cpu0 = time.time(), time.process_time()
    d = core.load_prepared(dataset, K)
    config = test.config_for("reference")
    flow = core.fit_flow(d, config, seeds=(drift_seed,))
    truth, pt, provenance = transition_truth(dataset)
    trans = opt.transition_score(core.s12(), flow["J"], truth, pt)
    scores, _ = opt.score_after_fit(dataset, d, flow)
    lams = np.asarray(core.s12().spectra(flow["J"])["lam1"], float)
    bins = opt.day_weighted_bins(pt, np.asarray(provenance["days"], int), n_bins=N_BINS)
    med = np.asarray([np.median(lams[bins == b]) if np.any(bins == b) else np.nan
                      for b in range(N_BINS)], float)
    record = {
        "phase": phase, "dataset": dataset, "preset": core.synthetic_id_parts(dataset)[0],
        "seed": core.synthetic_id_parts(dataset)[1], "drift_seed": drift_seed,
        "k": K, "configuration": config, "n_cells": int(len(lams)),
        "input_sha256": provenance["input_sha256"], "prepared_sha256": provenance["prepared_sha256"],
        "transition": trans, "fork_auc": scores.get("auc_lam1"),
        "bin_medians_lambda1": med.tolist(), "P": float(np.nanmax(med) - np.nanmedian(med)),
        "R": float(trans["peak_to_median"]), "detected_bin": int(trans["detected_bin"]),
        "true_bin": int(trans["true_bin"]), "n_fit_cells": flow["model_info"][0]["n_fit_cells"],
        "ot_pairs": flow["ot_info"], "model_info": flow["model_info"],
    }
    arr = OUT / "arrays" / f"{dataset}_d{drift_seed}.npz"
    arr.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arr, lambda1=lams.astype(np.float32), bins=bins.astype(np.int8),
                        bin_medians=med.astype(np.float64))
    checkpoint(f"fit_{phase}_{dataset}_d{drift_seed}", started, cpu0,
               {"result": record, "array_path": str(arr.relative_to(ROOT)), "array_sha256": sha256(arr)})


def fit_record(phase: str, dataset: str, drift_seed: int) -> dict[str, Any]:
    path = CK / f"fit_{phase}_{dataset}_d{drift_seed}.json"
    if not path.is_file():
        raise RuntimeError(f"missing fit checkpoint {path.name}")
    return json.loads(path.read_text())["result"]


def calibrate() -> None:
    rows = []
    for drift_seed in DRIFT_SEEDS:
        records = [fit_record("null", f"v3_nofork_s{s}", drift_seed) for s in NULL_CALIBRATION]
        tnull = max(float(r["R"]) for r in records)
        for r in records:
            rows.append({"drift_seed": drift_seed, "dataset": r["dataset"], "seed": r["seed"],
                         "R": r["R"], "T_null": tnull, "detected_bin": r["detected_bin"],
                         "fork_auc": r["fork_auc"], "P": r["P"]})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "null_calibration.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def summarize() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eligible_path = OUT / "g2_transition_eligibility.csv"
    require_committed(eligible_path)
    require_committed(OUT / "null_calibration.csv")
    with eligible_path.open(newline="") as f:
        elig = [r for r in csv.DictReader(f) if r["eligible"].lower() == "true"]
    selected = [int(r["seed"]) for r in elig[:4]]
    if len(selected) != 4:
        raise RuntimeError(f"S14b requires four eligible transitions; found {len(selected)}")
    transitions = [f"v3_transition_s{s}" for s in selected]
    heldout = [f"v3_nofork_s{s}" for s in NULL_HELD_OUT]
    calibration = [f"v3_nofork_s{s}" for s in NULL_CALIBRATION]
    rows: list[dict[str, Any]] = []
    for phase, datasets in (("null", calibration), ("heldout", transitions + heldout)):
        for dataset in datasets:
            for drift_seed in DRIFT_SEEDS:
                rows.append(fit_record(phase, dataset, drift_seed))
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
    gates["null_calibration_seeds"] = list(NULL_CALIBRATION)
    gates["heldout_nofork_seeds"] = list(NULL_HELD_OUT)
    gates["eligible_seed_rows"] = [r for r in elig if int(r["seed"]) in selected]

    OUT.mkdir(parents=True, exist_ok=True)
    test_path = OUT / "test.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(test_path, index=False)
    (OUT / "gates.json").write_text(json.dumps(jsonable(gates), indent=2, sort_keys=True) + "\n")

    fig_dir = ROOT / "benchmarks/figures/orient/s14b"
    fig_dir.mkdir(parents=True, exist_ok=True)
    first_transition, first_nofork = transitions[0], heldout[0]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    colors = {first_transition: "#1769aa", first_nofork: "#d55e00"}
    for dataset in (first_transition, first_nofork):
        r = fit_record("heldout", dataset, 0)
        profile = np.asarray(r["bin_medians_lambda1"], float)
        scale = max(float(np.nanmedian(np.abs(profile))), 1e-12)
        ax.plot(np.arange(N_BINS), profile / scale, marker="o", label=dataset, color=colors[dataset])
    ax.axhline(nulls[0], color="black", linestyle="--", label=f"T_null (d=0): {nulls[0]:.3f}")
    ax.set(xlabel="S14 pseudotime bin", ylabel="Median λ₁ / median |bin median|",
           title="S14b reference flow: first eligible transition and held-out no-fork")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "profiles.png", dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate"); g.add_argument("--preset", required=True); g.add_argument("--seed", type=int, required=True)
    prep = sub.add_parser("prepare"); prep.add_argument("--preset", required=True); prep.add_argument("--seed", type=int, required=True)
    e = sub.add_parser("eligibility"); e.add_argument("--seed", type=int, required=True)
    f = sub.add_parser("fit"); f.add_argument("--dataset", required=True); f.add_argument("--drift-seed", type=int, choices=DRIFT_SEEDS, required=True); f.add_argument("--phase", choices=("null", "heldout"), required=True)
    sub.add_parser("calibrate")
    sub.add_parser("summarize")
    a = p.parse_args()
    if a.cmd == "generate": generate(a.preset, a.seed)
    elif a.cmd == "prepare": prepare(a.preset, a.seed)
    elif a.cmd == "eligibility": eligibility(a.seed)
    elif a.cmd == "fit": fit(a.dataset, a.drift_seed, a.phase)
    elif a.cmd == "calibrate": calibrate()
    elif a.cmd == "summarize": summarize()


if __name__ == "__main__":
    main()
