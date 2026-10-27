#!/usr/bin/env python3
"""orient/S14 Part C: held-out evaluation after frozen k/configuration choices.

Every fit consumes only observed PCA coordinates, days, hours and barcodes.  For
synthetic datasets, truth is opened only by ``s14_optimise.score_after_fit``
after OT and every MeasuredDrift member has been fitted.  This script never
selects a configuration, k, model architecture, or test seed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in os.sys.path:
    os.sys.path.insert(0, str(SCRIPTS))
import s14_choose_k as core
import s14_optimise as opt

ROOT, OUT, ARR, CK, LOG, FIG = core.ROOT, core.OUT, core.ARR, core.CK, core.LOG, core.FIG
TRANSITION_TEST = ("v3_transition_s2", "v3_transition_s3", "v3_transition_s4")
DEFAULT_TEST = ("v3_default_s2", "v3_default_s3")
NOFORK_TEST = ("v3_nofork_s0", "v3_nofork_s1", "v3_nofork_s2")
REPORT_ONLY = tuple(f"v3_{preset}_s{seed}" for preset in ("stress", "batch") for seed in range(3))
SYNTHETIC_TEST = TRANSITION_TEST + DEFAULT_TEST + NOFORK_TEST + REPORT_ONLY
CONFIG_IDS = ("selected", "reference")


def _safe(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"unsafe identifier: {value}")
    return value


def selected_k() -> int:
    return opt.require_k()


def evaluation_ks() -> tuple[int, ...]:
    """k* and only its immediate neighbours in the frozen Part C grid."""
    k = selected_k()
    i = core.KS.index(k)
    return tuple(core.KS[max(0, i - 1): min(len(core.KS), i + 2)])


def config_for(config_id: str) -> dict[str, Any]:
    if config_id == "reference":
        # Part B's observed-only baseline is whitened; Part C's explicit
        # comparison reference is instead the raw-coordinate S10 default.
        config = dict(core.REFERENCE)
        config["space"] = "raw"
        return config
    if config_id == "selected":
        path = opt.combined_path()
        if not path.exists():
            raise RuntimeError("selected_configuration.json is required before Part C")
        record = core.read_json(path)
        if int(record.get("k_star", -1)) != selected_k():
            raise RuntimeError("selected configuration k does not match the locked truth-free k*")
        return dict(record["configuration"])
    raise ValueError(f"unknown configuration {config_id}")


def arr_path(dataset: str, k: int, config_id: str) -> Path:
    return ARR / "test" / f"{dataset}_k{k}_{config_id}.npz"


def unit_name(dataset: str, k: int, config_id: str) -> str:
    return f"test_{dataset}_k{k}_{config_id}"


def larry_unit_name(config_id: str) -> str:
    return f"test_larry_k{selected_k()}_{config_id}"


def checkpoint_path(name: str) -> Path:
    return CK / f"{name}.json"


def prepare(dataset: str) -> None:
    """Prepare observed inputs only for one held-out synthetic dataset or LARRY."""
    if dataset == "larry":
        core.prepare_larry()
        return
    if dataset not in SYNTHETIC_TEST:
        raise RuntimeError(f"not an S14 Part C synthetic dataset: {dataset}")
    preset, seed = core.synthetic_id_parts(dataset)
    core.prepare_synthetic(preset, seed)


def synthetic_unit(dataset: str, k: int, config_id: str) -> None:
    if dataset not in SYNTHETIC_TEST:
        raise RuntimeError(f"not an S14 Part C synthetic dataset: {dataset}")
    if k not in evaluation_ks():
        raise RuntimeError(f"Part C k must be selected k or neighbour: {evaluation_ks()}")
    _safe(dataset); _safe(config_id)
    core.require_preflight(); core.set_threads()
    started, cpu0 = time.time(), core.cpu_seconds()
    d = core.load_prepared(dataset, k)
    config = config_for(config_id)
    # Observed-only phase: raw PCA-k, day/hour OT, and MeasuredDrift members.
    flow = core.fit_flow(d, config)
    # Authorised scoring phase begins only after all observed fitting is complete.
    score, arrays = opt.score_after_fit(dataset, d, flow)
    dest = arr_path(dataset, k, config_id); dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **arrays)
    preset, seed = core.synthetic_id_parts(dataset)
    core.checkpoint(unit_name(dataset, k, config_id), started, cpu0, {
        "part": "C", "dataset": dataset, "preset": preset, "seed": seed, "k": int(k),
        "configuration_id": config_id, "configuration": config, "n_cells": int(len(d["z"])),
        "prepared_path": str(Path(d["path"]).relative_to(ROOT)),
        "prepared_sha256": core.sha256(Path(d["path"])),
        "observed_fit_complete_before_truth_scoring": True,
        "truth_scoring_only": ["fork AUC", "J_ref error", "fate-axis cosine", "strict-pseudotime transition"],
        "scores": score, "ot_pairs": flow["ot_info"], "models": flow["model_info"],
        "array_path": str(dest.relative_to(ROOT)), "array_sha256": core.sha256(dest),
    })


def _integrate_raw(flow: dict[str, Any], start: np.ndarray, gap_hours: float) -> np.ndarray:
    n_steps = int(round(float(gap_hours)))
    if n_steps < 1 or abs(n_steps - float(gap_hours)) > 1e-8:
        raise RuntimeError(f"invalid integer-hour held-out gap {gap_hours}")
    x = np.asarray(start, np.float64).copy()
    for _ in range(n_steps):
        x += core.predicted_velocity_raw(flow, x)
    return x


def larry_heldout(d: dict[str, Any], config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fit a configuration on days one/two and predict day three in PCA-k."""
    z, days, hours = np.asarray(d["z"], np.float64), np.asarray(d["day"], int), np.asarray(d["hours"], float)
    uniq = np.unique(days)
    if len(uniq) < 3:
        raise RuntimeError("LARRY Part C requires three sampling days")
    d0, d1, d2 = uniq[:3]
    train = (days == d0) | (days == d1)
    train_d = {"z": z[train], "day": days[train], "hours": hours[train]}
    flow = core.fit_flow(train_d, config)
    first, middle, last = days == d0, days == d1, days == d2
    gap = float(np.median(hours[last]) - np.median(hours[middle]))
    pred = _integrate_raw(flow, z[middle], gap)
    nomove = z[middle]
    meanmove = nomove + (z[middle].mean(axis=0) - z[first].mean(axis=0))
    inv = np.linalg.pinv(np.asarray(d["linear_map"], np.float64))
    common = np.asarray(d["common_map"], np.float64)
    observed = (z[last] @ inv) @ common
    rows = []
    for method, value in (("flow", pred), ("no_movement", nomove), ("mean_displacement", meanmove)):
        rows.append({"method": method, "energy_distance": float(core.s13().energy((value @ inv) @ common, observed)),
                     "n_middle": int(middle.sum()), "n_last": int(last.sum())})
    return rows, {"first_day": int(d0), "middle_day": int(d1), "last_day": int(d2), "gap_hours": gap,
                  "fit_ot_pairs": flow["ot_info"], "fit_models": flow["model_info"]}


def larry_unit(config_id: str) -> None:
    _safe(config_id); core.require_preflight(); core.set_threads()
    started, cpu0 = time.time(), core.cpu_seconds()
    k, config = selected_k(), config_for(config_id)
    d = core.load_prepared("larry", k)
    # Both full-data lineage and two-day held-out fits use observed inputs only.
    lineage_flow = core.fit_flow(d, config)
    lineage = core.lineage_agreement(d, lineage_flow)
    heldout, heldout_info = larry_heldout(d, config)
    core.checkpoint(larry_unit_name(config_id), started, cpu0, {
        "part": "C", "dataset": "larry", "k": k, "configuration_id": config_id,
        "configuration": config, "n_cells": int(len(d["z"])),
        "prepared_path": str(Path(d["path"]).relative_to(ROOT)),
        "prepared_sha256": core.sha256(Path(d["path"])),
        "truth_free": True, "lineage": lineage, "heldout_day": heldout,
        "heldout_details": heldout_info, "lineage_ot_pairs": lineage_flow["ot_info"],
        "lineage_models": lineage_flow["model_info"],
    })


def _synthetic_records() -> list[dict[str, Any]]:
    records = []
    missing = []
    for dataset in SYNTHETIC_TEST:
        for k in evaluation_ks():
            for config_id in CONFIG_IDS:
                path = checkpoint_path(unit_name(dataset, k, config_id))
                if not path.exists():
                    missing.append(path.name)
                else:
                    records.append(core.read_json(path))
    if missing:
        raise RuntimeError("missing Part C synthetic checkpoints: " + ", ".join(missing))
    return records


def _palantir_root_records() -> list[dict[str, Any]]:
    """Record fixed strict-r0 provenance for every synthetic Part C dataset."""
    rows, missing = [], []
    for dataset in SYNTHETIC_TEST:
        preset, seed = core.synthetic_id_parts(dataset)
        path = CK / f"prepare_{preset}_s{seed}.json"
        if not path.exists():
            missing.append(path.name)
            continue
        info = dict(core.read_json(path).get("strict_pseudotime", {}))
        rows.append({
            "dataset": dataset, "preset": preset, "seed": int(seed),
            "source": info.get("source", ""), "key": info.get("key", ""),
            "root_rule": info.get("root_rule", ""),
            "given_root": info.get("given_root", info.get("given_cell", "")),
            "actual_start_cell": info.get("actual_start_cell", ""),
            "candidate_roots": json.dumps(info.get("candidate_roots", [])),
            "settings": json.dumps(info.get("settings", {}), sort_keys=True),
            "preparation_checkpoint": str(path.relative_to(ROOT)),
        })
    if missing:
        raise RuntimeError("missing strict-Palantir preparation checkpoints: " + ", ".join(missing))
    return rows


def _larry_records() -> list[dict[str, Any]]:
    records, missing = [], []
    for config_id in CONFIG_IDS:
        path = checkpoint_path(larry_unit_name(config_id))
        if path.exists():
            records.append(core.read_json(path))
        else:
            missing.append(path.name)
    if missing:
        raise RuntimeError("missing Part C LARRY checkpoints: " + ", ".join(missing))
    return records


def _gate_summary(frame: pd.DataFrame, larry: list[dict[str, Any]]) -> dict[str, Any]:
    k = selected_k()
    selected = frame[(frame.configuration_id == "selected") & (frame.k == k)]
    trans = selected[selected.dataset.isin(TRANSITION_TEST)].copy()
    g1_rows = frame[(frame.configuration_id == "selected") & frame.dataset.isin(TRANSITION_TEST) & frame.k.isin(evaluation_ks())]
    g1 = {"rule": "selected fork AUC >= 0.85 on every transition test seed at k* and immediate grid neighbours",
          "k_values": list(evaluation_ks()), "minimum_auc": float(g1_rows.auc_lam1.min()),
          "n_required_rows": int(len(g1_rows)), "pass": bool(len(g1_rows) == len(TRANSITION_TEST) * len(evaluation_ks()) and (g1_rows.auc_lam1 >= 0.85).all())}
    informative = trans[trans.transition_informative.astype(bool)]
    trans_pass = bool(len(informative) == len(trans) and (informative.transition_abs_offset <= 1).all())
    nofork = selected[selected.dataset.isin(NOFORK_TEST)]
    peak_floor = float(trans.transition_peak_to_median.min()) if len(trans) else float("nan")
    nofork_pass = bool(len(nofork) == len(NOFORK_TEST) and (nofork.transition_peak_to_median < peak_floor).all())
    g2 = {"rule": "informative transition bin absolute offset <= 1; nofork peak/median below every transition-test ratio",
          "transition_abs_offsets": {str(r.dataset): int(r.transition_abs_offset) for r in trans.itertuples()},
          "transition_ratios": {str(r.dataset): float(r.transition_peak_to_median) for r in trans.itertuples()},
          "nofork_ratios": {str(r.dataset): float(r.transition_peak_to_median) for r in nofork.itertuples()},
          "transition_pass": trans_pass, "nofork_pass": nofork_pass, "pass": bool(trans_pass and nofork_pass)}
    lsel = next(r for r in larry if r["configuration_id"] == "selected")
    held = {r["method"]: float(r["energy_distance"]) for r in lsel["heldout_day"]}
    lineage = lsel["lineage"]
    g3 = {"rule": "selected LARRY lineage agreement exceeds all five nulls and held-out flow energy beats both baselines",
          "lineage_spearman": float(lineage["lineage_spearman"]), "lineage_null_max": float(lineage["lineage_null_max"]),
          "heldout_energy": held,
          "pass": bool(lineage["lineage_spearman"] > lineage["lineage_null_max"] and held["flow"] < held["no_movement"] and held["flow"] < held["mean_displacement"])}
    dev = []
    for dataset in ("v3_transition_s0", "v3_transition_s1"):
        path = CK / f"{opt.ck_name('combined', 'combined_selected', dataset, k)}.json"
        rec = core.read_json(path); dev.append({"dataset": dataset, "auc_lam1": float(rec["scores"]["auc_lam1"])})
    g4 = {"rule": "selected development fork AUC >= 0.85 on both transition development seeds at k*",
          "development": dev, "pass": bool(all(row["auc_lam1"] >= 0.85 for row in dev))}
    return {"gate_G1_auc_band": g1, "gate_G2_transition": g2, "gate_G3_larry": g3, "gate_G4_k_sanity": g4,
            "overall_pass": bool(g1["pass"] and g2["pass"] and g3["pass"] and g4["pass"])}


def _ledger() -> dict[str, Any]:
    rows = []
    for path in sorted(LOG.glob("*.log")):
        text = path.read_text(errors="ignore")
        tm = re.search(r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys", text)
        st = re.search(r"^START (\S+)", text, re.M); ex = re.search(r"^EXIT (\d+) (\S+)", text, re.M)
        rm = re.search(r"(\d+)\s+maximum resident set size", text)
        rows.append({"log": str(path.relative_to(ROOT)), "start_utc": st.group(1) if st else "", "end_utc": ex.group(2) if ex else "",
                     "exit": int(ex.group(1)) if ex else "", "wall_s": float(tm.group(1)) if tm else "",
                     "cpu_s": round(float(tm.group(2)) + float(tm.group(3)), 3) if tm else "",
                     "peak_rss_gb": round(int(rm.group(1)) / 1e9, 3) if rm else ""})
    pd.DataFrame(rows).to_csv(OUT / "cpu_ledger.csv", index=False)
    cpu = sum(float(row["cpu_s"]) for row in rows if row["cpu_s"] != "")
    return {"cpu_hours": cpu / 3600.0, "budget_cpu_hours": 24.0, "within_budget": bool(cpu / 3600.0 <= 24.0), "n_logs": len(rows)}


def finalise() -> None:
    core.require_preflight(); started, cpu0 = time.time(), core.cpu_seconds()
    recs = _synthetic_records(); larry = _larry_records(); rows = []
    for rec in recs:
        score = dict(rec["scores"])
        if "transition_bin_medians" in score:
            score["transition_bin_medians"] = json.dumps(score["transition_bin_medians"])
        rows.append({"kind": "synthetic", "dataset": rec["dataset"], "preset": rec["preset"], "seed": rec["seed"],
                     "k": rec["k"], "configuration_id": rec["configuration_id"], "n_cells": rec["n_cells"], **score,
                     "checkpoint": str((CK / f'{unit_name(rec["dataset"], int(rec["k"]), rec["configuration_id"])}.json').relative_to(ROOT))})
    for rec in larry:
        line = rec["lineage"]
        held = {x["method"]: x["energy_distance"] for x in rec["heldout_day"]}
        rows.append({"kind": "larry", "dataset": "larry", "preset": "larry", "seed": 0, "k": rec["k"],
                     "configuration_id": rec["configuration_id"], "n_cells": rec["n_cells"],
                     "lineage_spearman": line["lineage_spearman"], "lineage_null_max": line["lineage_null_max"],
                     "lineage_nulls": json.dumps(line["lineage_nulls"]), "heldout_flow_energy": held["flow"],
                     "heldout_no_movement_energy": held["no_movement"], "heldout_mean_displacement_energy": held["mean_displacement"],
                     "checkpoint": str((CK / f'{larry_unit_name(rec["configuration_id"])}.json').relative_to(ROOT))})
    frame = pd.DataFrame(rows).sort_values(["kind", "dataset", "k", "configuration_id"])
    frame.to_csv(OUT / "test.csv", index=False)
    roots = pd.DataFrame(_palantir_root_records()).sort_values(["dataset"])
    roots.to_csv(OUT / "palantir_roots_test.csv", index=False)
    syn = frame[frame.kind == "synthetic"].copy()
    gates = _gate_summary(syn, larry)
    core.write_json(OUT / "gates.json", gates)
    FIG.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot = syn[(syn.dataset.isin(TRANSITION_TEST)) & syn.configuration_id.isin(CONFIG_IDS)]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for dataset in TRANSITION_TEST:
        for config_id, marker in (("selected", "o"), ("reference", "s")):
            sub = plot[(plot.dataset == dataset) & (plot.configuration_id == config_id)].sort_values("k")
            ax.plot(sub.k, sub.auc_lam1, marker=marker, label=f"{dataset} {config_id}")
    ax.axhline(0.85, color="black", linestyle="--", linewidth=0.9, label="G1 threshold")
    ax.set(xlabel="PCA components (k)", ylabel="fork AUC (λ1)", ylim=(0, 1.03), title="S14 held-out transition test: selected vs reference")
    ax.legend(ncol=2, fontsize=7); fig.tight_layout(); fig.savefig(FIG / "test_auc_by_k.png", dpi=160); plt.close(fig)
    ledger = _ledger()
    lines = ["# orient/S14 — held-out Part C", "", "## Locked choices", f"* Truth-free Revision 1 k*: **{selected_k()}**.",
             f"* Selected development configuration: `{json.dumps(config_for('selected'), sort_keys=True)}`.",
             "* Synthetic truth was accessed only after each observed OT/drift fit completed; test seeds were not used to select k or configuration.",
             "", "## Gates"]
    for name, value in gates.items():
        if name.startswith("gate_"):
            lines.append(f"* **{name}: {'PASS' if value['pass'] else 'FAIL'}** — {value['rule']}")
    lines += ["", "## Test output", f"* `test.csv` has {len(frame)} rows; `gates.json` records all component values.",
              f"* Strict Palantir r0 provenance is in `palantir_roots_test.csv` ({len(roots)} synthetic test datasets).",
              f"* Timed CPU ledger: **{ledger['cpu_hours']:.3f} CPU-h** / 24.000 CPU-h (closed logs available at finalization).", ""]
    (OUT / "RESULT_S14.md").write_text("\n".join(lines))
    core.checkpoint("test_finalise", started, cpu0, {"part": "C", "k_star": selected_k(), "selected_configuration": config_for("selected"),
                    "n_test_rows": int(len(frame)), "gates": gates, "ledger": ledger,
                    "test_csv": str((OUT / "test.csv").relative_to(ROOT)),
                    "palantir_roots": str((OUT / "palantir_roots_test.csv").relative_to(ROOT)),
                    "figure": str((FIG / "test_auc_by_k.png").relative_to(ROOT))})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "finalise"))
    parser.add_argument("--dataset", choices=(*SYNTHETIC_TEST, "larry"))
    parser.add_argument("--k", type=int, choices=core.KS)
    parser.add_argument("--configuration", choices=CONFIG_IDS)
    args = parser.parse_args()
    if args.stage == "prepare":
        if args.dataset is None: parser.error("prepare requires --dataset")
        prepare(args.dataset)
    elif args.stage == "run":
        if args.dataset is None or args.configuration is None: parser.error("run requires --dataset and --configuration")
        if args.dataset == "larry":
            if args.k is not None and args.k != selected_k(): parser.error("LARRY Part C runs only at the locked k*")
            larry_unit(args.configuration)
        else:
            if args.k is None: parser.error("synthetic run requires --k")
            synthetic_unit(args.dataset, args.k, args.configuration)
    else:
        finalise()


if __name__ == "__main__":
    main()
