#!/usr/bin/env python3
"""orient/S14 Part B: fixed development-only optimisation at the locked k*.

Observed PCA coordinates, sampling day/time, and OT are consumed by
``s14_choose_k.fit_flow`` first.  Synthetic truth is opened only afterwards by
``score_after_fit`` for the explicitly-authorised development objective and its
frozen tie-breakers.  No test seed is accepted by this command.
"""
from __future__ import annotations

import argparse
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

ROOT = core.ROOT
OUT, ARR, CK, LOG = core.OUT, core.ARR, core.CK, core.LOG
DEV = ("v3_transition_s0", "v3_transition_s1", "v3_default_s0", "v3_default_s1")
REF_ID = "eps_0p05"


def _id_float(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def registry() -> dict[str, dict[str, Any]]:
    """Fifteen unique one-at-a-time candidates; duplicate reference points alias REF_ID."""
    base = dict(core.REFERENCE)
    out: dict[str, dict[str, Any]] = {}
    for eps in (0.0125, 0.025, 0.05, 0.1, 0.2):
        cfg = dict(base); cfg["eps"] = eps
        out[f"eps_{_id_float(eps)}"] = cfg
    cfg = dict(base); cfg["last"] = "backward"; out["last_backward"] = cfg
    for hidden in ((64, 64), (96, 96, 96), (128, 128, 128)):
        for wd in (1e-5, 1e-4, 1e-3):
            cfg = dict(base); cfg["hidden"] = list(hidden); cfg["wd"] = wd
            ident = "arch_" + "x".join(str(x) for x in hidden) + "_wd" + _id_float(wd)
            if cfg == out[REF_ID]:
                continue
            out[ident] = cfg
    cfg = dict(base); cfg["ensemble"] = 5; out["ensemble_5"] = cfg
    assert len(out) == 15, len(out)
    return out


def selection_path() -> Path:
    """Revision 1’s amended truth-free lock for all subsequent Part B work."""
    return OUT / "k_selection_r1.json"


def selected_oat_path() -> Path:
    return OUT / "oat_selection.json"


def combined_path() -> Path:
    return OUT / "selected_configuration.json"


def require_k() -> int:
    p = selection_path()
    if not p.exists():
        raise RuntimeError("truth-free k_selection_r1.json is required before Revision 1 Part B")
    s = core.read_json(p)
    if not s.get("truth_free") or "k_star" not in s:
        raise RuntimeError("S14 k selection is missing its truth-free lock")
    return int(s["k_star"])


def ck_name(mode: str, config_id: str, dataset: str, k: int | None = None) -> str:
    """k-qualified names retain superseded Revision 0 checkpoints in place."""
    selected_k = require_k() if k is None else int(k)
    return f"optimise_{mode}_k{selected_k}_{config_id}_{dataset}"


def arr_path(mode: str, config_id: str, dataset: str, k: int | None = None) -> Path:
    selected_k = require_k() if k is None else int(k)
    return ARR / "optimise" / f"{mode}_k{selected_k}_{config_id}_{dataset}.npz"


def _safe_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"unsafe identifier: {value}")
    return value


def day_weighted_bins(pseudotime: np.ndarray, days: np.ndarray, n_bins: int = 12) -> np.ndarray:
    """S14 fixed weighted quantiles: every sampling day has total weight one."""
    pt, day = np.asarray(pseudotime, np.float64), np.asarray(days, int)
    if len(pt) != len(day) or len(pt) == 0:
        raise ValueError("pseudotime/day arrays must be nonempty and aligned")
    counts = {int(d): int((day == d).sum()) for d in np.unique(day)}
    weights = np.asarray([1.0 / counts[int(d)] for d in day], np.float64)
    order = np.argsort(pt, kind="stable")
    midpoint = np.cumsum(weights[order]) - 0.5 * weights[order]
    bins = np.empty(len(pt), dtype=int)
    bins[order] = np.minimum((midpoint / weights.sum() * n_bins).astype(int), n_bins - 1)
    return bins


def transition_score(s12: Any, J: np.ndarray, truth: dict[str, Any], pt: np.ndarray) -> dict[str, Any]:
    """S14 day-weighted bins plus frozen S10 median, peak, and rising-run helpers."""
    import importlib.util
    path = ROOT / "scripts/s10_transitions.py"
    spec = importlib.util.spec_from_file_location("s14_readonly_s10_transitions", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen S10 transition helpers")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    bins = day_weighted_bins(pt, np.asarray(truth["day"], int), n_bins=12)
    lam = np.asarray(s12.spectra(J)["lam1"], np.float64)
    med = mod.bin_medians(lam, bins)
    ztrue = np.asarray(truth["z"], np.float64)
    eig = 0.5 * ((ztrue[:, 1] - float(truth["z1_crit"])) - 3.0 * ztrue[:, 0] ** 2)
    true_med = mod.bin_medians(eig, bins)
    true_bin = int(np.nanargmax(true_med))
    row = mod.score_signal({"signal": "lam1"}, lam, med, bins, true_bin)
    return {"detected_bin": int(row["detected_bin"]), "true_bin": int(true_bin),
            "informative": bool(row["informative"]), "abs_offset": int(row["abs_offset"]),
            "signed_offset": int(row["signed_offset"]),
            "n_rising_bins_before_detected": int(row["n_rising_bins_before_detected"]),
            "peak_to_median": float(row["peak_to_median"]), "bin_medians": row["bin_medians"],
            "pseudotime_order": "S10 strict earliest-day r0; 12 S14 day-weighted pseudotime-quantile bins (cell weight 1/n_cells(day))"}


def score_after_fit(dataset: str, d: dict[str, Any], flow: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Truth scoring only: invoked after OT and every drift member completed."""
    import anndata as ad
    preset, seed = core.synthetic_id_parts(dataset)
    s12 = core.s12()
    s13 = core.s13()
    row = s12.manifest_row(preset, seed, verify=True)
    src = ad.read_h5ad(ROOT / row["path"], backed="r")
    try:
        truth = s12.truth_for_scoring(src)
        zraw = np.asarray(d["z"], np.float64)
        axis = s12.fate_axis(zraw, truth["day"], truth["fate"])
        score, lam = s13.score_synthetic(s12, flow["J"], truth, axis, flow["WI"] if flow["config"]["space"] == "whitened" else None)
        vtrue = s12.true_expected_velocity(src, {"linear_map": d["linear_map"], "normalization_target": float(d["normalization_target"])})
        jref_raw = s12.jref(zraw, vtrue)
        if flow["config"]["space"] == "whitened":
            # y=(x-mu)W, so J_y = W J_x W^-1 for the conventional output-by-input Jacobian.
            jref = np.einsum("ab,nbc,cd->nad", flow["W"], jref_raw, flow["WI"], optimize=True)
        else:
            jref = jref_raw
        jerr = float(s12.jref_error(flow["J"], jref))
        trans = transition_score(s12, flow["J"], truth, np.asarray(d["pseudotime_strict_r0"], np.float64))
        score.update({"jref_relative_frobenius_median": jerr, **{f"transition_{k}": v for k, v in trans.items()}})
        return score, {"lambda1": lam.astype(np.float32), "jref_rel": np.linalg.norm(np.asarray(flow["J"], float) - jref, axis=(1, 2)).astype(np.float32)}
    finally:
        src.file.close()


def run(mode: str, config_id: str, dataset: str) -> None:
    if dataset not in DEV:
        raise RuntimeError(f"Part B accepts only fixed synthetic development datasets: {DEV}")
    k = require_k()
    all_configs = registry()
    if mode == "oat":
        if config_id not in all_configs:
            raise RuntimeError(f"unknown OAT config {config_id}; choices: {sorted(all_configs)}")
        config = all_configs[config_id]
    elif mode == "combined":
        if not combined_path().exists():
            raise RuntimeError("select-oat must finish before combined configuration runs")
        chosen = core.read_json(combined_path())
        if config_id != "combined_selected":
            raise RuntimeError("combined mode uses config id combined_selected")
        config = chosen["configuration"]
    else:
        raise ValueError(mode)
    _safe_id(config_id); _safe_id(dataset)
    started, cpu0 = time.time(), core.cpu_seconds()
    d = core.load_prepared(dataset, k)
    # All observed-only fitting completes within this call.
    flow = core.fit_flow(d, config)
    # Development truth is deliberately accessed after the complete fit only.
    scores, arrays = score_after_fit(dataset, d, flow)
    dest = arr_path(mode, config_id, dataset, k); dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **arrays)
    core.checkpoint(ck_name(mode, config_id, dataset, k), started, cpu0, {
        "part": "B", "mode": mode, "dataset": dataset, "k_star": k,
        "configuration_id": config_id, "configuration": config,
        "prepared_path": str(Path(d["path"]).relative_to(ROOT)), "prepared_sha256": core.sha256(Path(d["path"])),
        "observed_fit_complete_before_truth_scoring": True,
        "truth_scoring_only": ["fork AUC", "J_ref error", "strict-pseudotime transition error"],
        "scores": scores, "ot_pairs": flow["ot_info"], "models": flow["model_info"],
        "array_path": str(dest.relative_to(ROOT)), "array_sha256": core.sha256(dest),
    })


def _oat_rows() -> pd.DataFrame:
    k = require_k()
    records = []
    missing = []
    for config_id, config in registry().items():
        for dataset in DEV:
            p = CK / f"{ck_name('oat', config_id, dataset, k)}.json"
            if not p.exists():
                missing.append(p.name); continue
            rec = core.read_json(p)
            if int(rec.get("k_star", -1)) != k:
                raise RuntimeError(f"checkpoint k mismatch for Revision 1: {p}")
            score = rec["scores"]
            records.append({"stage": "oat", "configuration_id": config_id, "dataset": dataset, "k_star": rec["k_star"],
                            "auc_lam1": score["auc_lam1"], "jref_error": score["jref_relative_frobenius_median"],
                            "transition_informative": score["transition_informative"], "transition_abs_offset": score["transition_abs_offset"],
                            "eps_factor": config["eps"], "last_day": config["last"], "hidden": "x".join(map(str, config["hidden"])),
                            "weight_decay": config["wd"], "ensemble": config["ensemble"], "checkpoint": str(p.relative_to(ROOT))})
    if missing:
        raise RuntimeError("missing OAT checkpoints: " + ", ".join(missing))
    return pd.DataFrame(records)


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    out = []
    for config_id, sub in frame.groupby("configuration_id", sort=True):
        inf = sub[sub.transition_informative.astype(bool)]
        out.append({"configuration_id": config_id, "mean_development_fork_auc": float(sub.auc_lam1.mean()),
                    "mean_jref_error": float(sub.jref_error.mean()),
                    "mean_transition_abs_offset_informative": float(inf.transition_abs_offset.mean()) if len(inf) else np.nan,
                    "n_datasets": int(len(sub)), "n_informative": int(len(inf))})
    return pd.DataFrame(out)


def _choose(summary: pd.DataFrame, candidate_ids: list[str]) -> dict[str, Any]:
    sub = summary[summary.configuration_id.isin(candidate_ids)].copy()
    if len(sub) != len(candidate_ids):
        raise RuntimeError("candidate summary missing")
    best = float(sub.mean_development_fork_auc.max())
    tied = sub[sub.mean_development_fork_auc >= best - 0.01].sort_values(
        ["mean_jref_error", "mean_transition_abs_offset_informative", "configuration_id"], na_position="last")
    return {"selected_configuration_id": str(tied.iloc[0].configuration_id), "best_auc": best,
            "within_0_01": tied.to_dict(orient="records"),
            "tie_break": "within 0.01 fork-AUC: lower mean J_ref error, then lower mean transition-bin error on informative development datasets, then lexicographic id"}


def select_oat() -> None:
    k = require_k(); started, cpu0 = time.time(), core.cpu_seconds()
    frame = _oat_rows(); summary = _aggregate(frame); cfg = registry()
    component_candidates = {
        "epsilon": [f"eps_{_id_float(x)}" for x in (0.0125, 0.025, 0.05, 0.1, 0.2)],
        "last_day": [REF_ID, "last_backward"],
        "drift": [REF_ID] + [x for x in cfg if x.startswith("arch_")],
        "ensemble": [REF_ID, "ensemble_5"],
    }
    choices = {name: _choose(summary, ids) for name, ids in component_candidates.items()}
    eps_id, last_id, drift_id, ensemble_id = (choices["epsilon"]["selected_configuration_id"], choices["last_day"]["selected_configuration_id"],
                                               choices["drift"]["selected_configuration_id"], choices["ensemble"]["selected_configuration_id"])
    combined = dict(core.REFERENCE)
    combined["eps"] = cfg[eps_id]["eps"]
    combined["last"] = cfg[last_id]["last"]
    combined["hidden"] = cfg[drift_id]["hidden"]
    combined["wd"] = cfg[drift_id]["wd"]
    combined["ensemble"] = cfg[ensemble_id]["ensemble"]
    record = {"k_star": k, "objective": "mean development fork AUC over v3_transition/default s0,s1", "objective_datasets": list(DEV),
              "larry_role": "truth-free Part A k selection only; absent from synthetic fork-AUC Part B objective", "components": choices,
              "configuration": combined, "component_source_ids": {"epsilon": eps_id, "last_day": last_id, "drift": drift_id, "ensemble": ensemble_id},
              "selection_utc": core.utc()}
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "optimisation_oat_by_dataset.csv", index=False)
    summary.to_csv(OUT / "optimisation_oat_summary.csv", index=False)
    core.write_json(selected_oat_path(), record)
    core.write_json(combined_path(), record)
    core.checkpoint("optimise_select_oat", started, cpu0, {"part": "B", **record,
                    "oat_by_dataset": str((OUT / "optimisation_oat_by_dataset.csv").relative_to(ROOT)),
                    "oat_summary": str((OUT / "optimisation_oat_summary.csv").relative_to(ROOT))})
    print(json.dumps(core.jsonable(record), indent=2))


def finalise() -> None:
    if not combined_path().exists():
        raise RuntimeError("combined configuration has not been selected")
    k = require_k(); started, cpu0 = time.time(), core.cpu_seconds()
    oat = _oat_rows()
    rows = oat.to_dict(orient="records")
    missing = []
    for dataset in DEV:
        p = CK / f"{ck_name('combined', 'combined_selected', dataset, k)}.json"
        if not p.exists(): missing.append(p.name); continue
        rec = core.read_json(p); sc = rec["scores"]; cfg = rec["configuration"]
        rows.append({"stage": "combined", "configuration_id": "combined_selected", "dataset": dataset, "k_star": rec["k_star"],
                     "auc_lam1": sc["auc_lam1"], "jref_error": sc["jref_relative_frobenius_median"],
                     "transition_informative": sc["transition_informative"], "transition_abs_offset": sc["transition_abs_offset"],
                     "eps_factor": cfg["eps"], "last_day": cfg["last"], "hidden": "x".join(map(str, cfg["hidden"])),
                     "weight_decay": cfg["wd"], "ensemble": cfg["ensemble"], "checkpoint": str(p.relative_to(ROOT))})
    if missing: raise RuntimeError("missing combined checkpoints: " + ", ".join(missing))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "optimisation_log.csv", index=False)
    combined = frame[frame.stage == "combined"]
    summary = _aggregate(combined).iloc[0].to_dict()
    core.checkpoint("optimise_finalise", started, cpu0, {"part": "B", "k_star": k, "selected": core.read_json(combined_path()),
                    "combined_summary": summary, "optimisation_log": str((OUT / "optimisation_log.csv").relative_to(ROOT))})
    print(json.dumps(core.jsonable(summary), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "select-oat", "finalise"))
    parser.add_argument("--mode", choices=("oat", "combined"))
    parser.add_argument("--config")
    parser.add_argument("--dataset", choices=DEV)
    args = parser.parse_args()
    if args.stage == "run":
        if args.mode is None or args.config is None or args.dataset is None:
            parser.error("run requires --mode, --config, and --dataset")
        run(args.mode, args.config, args.dataset)
    elif args.stage == "select-oat":
        select_oat()
    else:
        finalise()


if __name__ == "__main__":
    main()
