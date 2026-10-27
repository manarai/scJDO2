#!/usr/bin/env python3
"""orient/S10 Step 2 (secondary): orientation scoring on v3 with the current layer defaults.

Imports the S06/S03 scoring helpers (unedited).  Every ``orient()`` input is a
minimal AnnData built from observed keys only (day, clone, pulse, stored
embeddings, fitted J / score / energy, fitted velocities, Palantir pseudotime
and its actual start cell).  Truth is read into a separate scoring copy.

Flow combinations do not depend on the Palantir run and are scored once per
unit (per labelling variant, reliability check on/off); development
combinations and the guard are scored for every Palantir run.

Developmental truth: ``forward_developmental_truth`` (protected) supports
generator v1/v2 only and raises for v3; no v3 forward truth exists in the
allowed additions.  Development agreement is therefore reported against the
clone chord truth (``chord_truth``, S06's diagnostic reference) as a
substitute, and the gate clause that needs the forward truth is reported as not
evaluable as written.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import s10_fit_and_runs as common  # noqa: E402
from orient_end_to_end_s03 import call_orient, chord_truth, expected_flow_truth, score_group, source_addition  # noqa: E402
from s06_score import COMBINATIONS, reliability_text  # noqa: E402
from scjdo2.orient import OrientConfig  # noqa: E402

LAB_VARIANTS = common.VARIANTS + ("s06_smoothed",)     # s06_smoothed = MeasuredDrift prediction of the raw velocity (S06 'smoothed')


def lab_key(v: str) -> str:
    return "velocity_labelling" if v == "s06_smoothed" else f"velocity_labelling_{v}"


def configs(variant: str, **extra: Any) -> dict[str, OrientConfig]:
    base = {"labelling_key": lab_key(variant), **extra}
    return {"on": OrientConfig(**base), "off": OrientConfig(**base, reliability_min={}),
            "lineage_pseudotime_check": OrientConfig(**base, reliability_min={"velocity": 0.6, "pseudotime": 0.0})}


def observed_input(preset: str, seed: int, method: str) -> tuple[Any, Any]:
    """(observed AnnData for orient, scoring copy with truth)."""
    import anndata as ad
    row = common.manifest_row(preset, seed)
    src = ad.read_h5ad(common.ROOT / row["path"])
    fit = np.load(common.ARR / f"{preset}_s{seed}" / f"fit_{method}.npz")
    scv = np.load(common.ARR / f"{preset}_s{seed}" / "scvelo.npz")
    obs = pd.DataFrame({"day": src.obs["day"].to_numpy(int), "clone": src.obs["clone"].copy(),
                        "pulse_hours": src.obs["pulse_hours"].to_numpy(np.float32)}, index=src.obs_names.copy())
    data = ad.AnnData(X=np.zeros((src.n_obs, 0), dtype=np.float32), obs=obs)
    data.obsm[f"X_{method}"] = np.asarray(src.obsm[f"X_{method}"], dtype=np.float32)
    data.uns["scjdo2"] = {"J": fit["J_score"]}
    data.obsm["X_drift_v"] = fit["X_drift_v"]
    data.obs["energy"] = fit["energy"]
    for v in common.VARIANTS:
        data.obsm[lab_key(v)] = fit[f"v_lab_{v}"]
    data.obsm["velocity_labelling_raw"] = fit["v_lab_raw_shared"]
    data.obsm["velocity_labelling"] = fit["v_lab_fit_raw_shared"]
    data.obsm[f"velocity_{method}"] = scv[f"velocity_{method}"]
    if not np.array_equal(np.asarray(fit["z"]), data.obsm[f"X_{method}"]):
        raise RuntimeError("fit embedding differs from the stored embedding")
    return data, src


def palantir_runs(preset: str, seed: int) -> tuple[list[dict[str, Any]], Any]:
    ck = json.loads((common.CKPT / f"palantir_{preset}_s{seed}.json").read_text())
    arr = np.load(common.ARR / f"{preset}_s{seed}" / "palantir.npz")
    return [r for r in ck["runs"] if r["status"] == "ok"], arr


def run_unit(preset: str, seed: int, method: str) -> dict[str, list[dict[str, Any]]]:
    data, src = observed_input(preset, seed, method)
    days = data.obs["day"].to_numpy(int)
    scoring = src  # truth copy, never passed to orient
    flow_truth = expected_flow_truth(scoring, method)
    dev_truth = chord_truth(scoring, method)
    fate = scoring.obs["fate_true"].astype(str).to_numpy()
    clone_true = scoring.obs["clone_true"].astype(str).to_numpy()
    out: dict[str, list[dict[str, Any]]] = {"results": [], "by_day": [], "diagnostics": [], "palantir": [], "lineage": []}
    base = {"preset": preset, "seed": seed, "method": method}
    # ---- flow combinations: once per unit
    flow_out: dict[tuple[str, str, str], Any] = {}
    for variant in LAB_VARIANTS:
        cfgs = configs(variant)
        for rel in ("on", "off"):
            for name in ("flow_labelling", "flow_velocity", "flow_combined"):
                d = data.copy()
                o = call_orient(d, COMBINATIONS[name][0], method, cfgs[rel], [])
                flow_out[(variant, rel, name)] = o
                metric, per_day = score_group(o, "flow", flow_truth, days)
                common_row = {**base, "labelling_variant": variant, "reliability_mode": rel, "root_group": "none", "palantir_mode": "none",
                              "root_index": -1, "group": "flow", "combination": COMBINATIONS[name][2], "truth": "expected_flow"}
                out["results"].append({**common_row, "reliability": reliability_text(o), **metric})
                out["by_day"] += [{**common_row, **x} for x in per_day]
            fc, fl = flow_out[(variant, rel, "flow_combined")], flow_out[(variant, rel, "flow_labelling")]
            out["diagnostics"].append({"row_type": "source_addition", **base, "labelling_variant": variant, "reliability_mode": rel,
                                       "group": "flow", **source_addition(fc, fl, "flow", "labelling", "velocity", flow_truth)})
    # ---- §17 sensitivity: lineage only (does not use pseudotime)
    clones_obs = data.obs["clone"].astype(object).to_numpy()
    both_fates = {}
    for c in np.unique(clone_true):
        idx = np.flatnonzero(clone_true == c)
        late = idx[days[idx] == days[idx].max()]
        f = set(fate[late]) - {"progenitor"}
        both_fates[c] = "both" if len(f) == 2 else ("one" if len(f) == 1 else "none")
    split = np.array([both_fates[c] for c in clone_true])
    for k in (1, 2, 3):
        d = data.copy()
        o = call_orient(d, ["lineage"], method, OrientConfig(labelling_key=lab_key(common.GATE_VARIANT), lineage_min_cells_per_end=k), [])
        metric, _ = score_group(o, "dev", dev_truth, days)
        sign = np.asarray(o["dev"]["sign"]); src_ = np.asarray(o["dev"]["source"]) > 0
        from orient_end_to_end_s03 import truth_masks
        tsign, moving, finite = truth_masks(np.asarray(o["axes"].U), dev_truth)
        row = {**base, "lineage_min_cells_per_end": k, "truth": "chord (substitute)", **metric}
        for label in ("one", "both"):
            m = src_ & (split == label)[:, None]
            row[f"agreement_clones_{label}_fate"] = float(np.mean(sign[m] == tsign[m])) if m.any() else np.nan
            row[f"n_pairs_clones_{label}_fate"] = int(m.sum())
        out["lineage"].append(row)
    # ---- every Palantir run: guard + development combinations
    runs, arr = palantir_runs(preset, seed)
    for r in runs:
        unit = r["unit"]
        pseudo = arr[unit]
        d0 = data.copy()
        d0.obs["palantir_pseudotime"] = pseudo
        d0.uns["palantir_start_cell"] = r["actual_start_cell"]
        guard_error = None
        try:
            guard = call_orient(d0.copy(), ["pseudotime"], method, configs(common.GATE_VARIANT)["on"], [])
        except ValueError as exc:
            guard, guard_error = None, str(exc)
        prow = {**base, "unit": unit, "root_group": r["root_group"], "palantir_mode": r["mode"], "root_index": r["root_index"],
                "given_cell": r["given_cell"], "actual_start_cell": r["actual_start_cell"], "actual_start_day": r["actual_start_day"],
                "given_differs_from_actual": r["given_differs_from_actual"],
                "spearman_pseudotime_day": float(spearmanr(pseudo, days).statistic),
                "guard_status": "refused" if guard_error else "accepted", "guard_error": guard_error}
        if guard is not None:
            prow["chord_agreement"] = score_group(guard, "dev", dev_truth, days)[0]["sign_agreement"]
        out["palantir"].append(prow)
        allowed = ["dev_lineage"] + (["dev_pseudotime", "dev_combined"] if guard is not None else [])
        dev_out = {}
        for rel in ("on", "off"):
            for name in allowed:
                o = call_orient(d0.copy(), COMBINATIONS[name][0], method, configs(common.GATE_VARIANT)[rel], [])
                dev_out[(rel, name)] = o
                metric, per_day = score_group(o, "dev", dev_truth, days)
                crow = {**base, "labelling_variant": "n/a", "reliability_mode": rel, "root_group": r["root_group"], "palantir_mode": r["mode"],
                        "root_index": r["root_index"], "group": "dev", "combination": COMBINATIONS[name][2], "truth": "chord (substitute)"}
                out["results"].append({**crow, "reliability": reliability_text(o), **metric})
                out["by_day"] += [{**crow, **x} for x in per_day]
        if guard is None:
            continue
        o = call_orient(d0.copy(), COMBINATIONS["dev_combined"][0], method, configs(common.GATE_VARIANT)["lineage_pseudotime_check"], [])
        metric, _ = score_group(o, "dev", dev_truth, days)
        out["results"].append({**base, "labelling_variant": "n/a", "reliability_mode": "lineage_pseudotime_check", "root_group": r["root_group"],
                               "palantir_mode": r["mode"], "root_index": r["root_index"], "group": "dev", "combination": "lineage+pseudotime",
                               "truth": "chord (substitute)", "reliability": reliability_text(o), **metric})
        dc, dl = dev_out[("on", "dev_combined")], dev_out[("on", "dev_lineage")]
        out["diagnostics"].append({"row_type": "source_addition", **base, "labelling_variant": "n/a", "reliability_mode": "on", "unit": unit,
                                   "group": "dev", **source_addition(dc, dl, "dev", "lineage", "pseudotime", dev_truth)})
        for variant in (common.GATE_VARIANT, common.PRIMARY_VARIANT):
            d4 = d0.copy()
            allfour = call_orient(d4, COMBINATIONS["all_four"][0], method, configs(variant)["on"], [])
            fc = flow_out[(variant, "on", "flow_combined")]
            out["diagnostics"].append({"row_type": "independence", **base, "labelling_variant": variant, "reliability_mode": "on", "unit": unit,
                                       "root_group": r["root_group"], "palantir_mode": r["mode"],
                                       "flow_identical": bool(np.array_equal(fc["flow"]["sign"], allfour["flow"]["sign"]) and np.array_equal(fc["flow"]["source"], allfour["flow"]["source"])),
                                       "dev_identical": bool(np.array_equal(dc["dev"]["sign"], allfour["dev"]["sign"]) and np.array_equal(dc["dev"]["source"], allfour["dev"]["source"])),
                                       "hash_unchanged": bool(allfour["meta"]["hash_unchanged"])})
    return out


def gates(results: pd.DataFrame, diag: pd.DataFrame, pt: pd.DataFrame) -> dict[str, Any]:
    sel = results[(results.preset == "v3_default") & (results.method == "pca") & (results.reliability_mode == "on")]
    lab = sel[(sel.group == "flow") & (sel.combination == "labelling") & (sel.labelling_variant == common.GATE_VARIANT)]
    lab_primary = sel[(sel.group == "flow") & (sel.combination == "labelling") & (sel.labelling_variant == common.PRIMARY_VARIANT)]
    dev = sel[(sel.group == "dev") & (sel.root_group == "day2")]
    sub = {}
    for mode in ("default", "strict"):
        dm = dev[dev.palantir_mode == mode]
        sub[mode] = {"lineage+pseudotime": dm[dm.combination == "lineage+pseudotime"].groupby("seed").sign_agreement.median().round(4).to_dict(),
                     "pseudotime": dm[dm.combination == "pseudotime"].groupby("seed").sign_agreement.median().round(4).to_dict(),
                     "lineage": dm[dm.combination == "lineage"].groupby("seed").sign_agreement.median().round(4).to_dict(),
                     "n_accepted_runs_per_seed": dm[dm.combination == "pseudotime"].groupby("seed").size().to_dict()}
    ind = diag[(diag.row_type == "independence") & (diag.preset == "v3_default") & (diag.method == "pca")]
    g_orient = {
        "labelling_agreement_by_seed": lab.sort_values("seed").sign_agreement.round(4).tolist(),
        "labelling_agreement_min": float(lab.sign_agreement.min()),
        "labelling_ge_0_80": bool((lab.sign_agreement >= 0.80).all() and len(lab) == 3),
        "primary_variant_labelling_agreement_by_seed": lab_primary.sort_values("seed").sign_agreement.round(4).tolist(),
        "lineage_pseudotime_vs_pseudotime": "not evaluable as written: needs the forward developmental truth, "
                                            "which the protected forward_developmental_truth does not provide for v3",
        "substitute_chord_truth_median_by_seed": sub,
        "independence_exact": bool(len(ind) > 0 and (ind.flow_identical & ind.dev_identical).all()),
        "J_unchanged": bool(results.hash_unchanged.dropna().all() and (len(ind) == 0 or ind.hash_unchanged.all())),
    }
    g_orient["pass"] = None
    g_orient["status"] = "not_evaluable_as_written" if g_orient["labelling_ge_0_80"] and g_orient["independence_exact"] and g_orient["J_unchanged"] else "fail"
    guard = {}
    v = pt[pt.preset == "v3_default"]
    for mode in ("default", "strict"):
        late = v[(v.root_group == "day6") & (v.palantir_mode == mode)]
        early = v[(v.root_group == "day2") & (v.palantir_mode == mode)]
        guard[mode] = {"latest_day_runs": int(len(late)), "latest_day_refused": int((late.guard_status == "refused").sum()),
                       "day2_runs": int(len(early)), "day2_accepted": int((early.guard_status == "accepted").sum()),
                       "day2_accepted_fraction": float((early.guard_status == "accepted").mean()) if len(early) else np.nan}
    guard_pass = all(g["latest_day_refused"] == g["latest_day_runs"] and g["latest_day_runs"] > 0 and g["day2_accepted_fraction"] >= 0.90 for g in guard.values())
    allp = {}
    for preset in sorted(pt.preset.unique()):
        p = pt[pt.preset == preset]
        allp[preset] = {f"{g}_{m}": f"{int((s.guard_status == 'accepted').sum())}/{len(s)} accepted"
                        for (g, m), s in p.groupby(["root_group", "palantir_mode"])}
    return {"gate_orientation": g_orient, "gate_guard_fresh": {"v3_default": guard, "pass": bool(guard_pass), "all_presets": allp}}


PARTS = common.OUT / "score_parts"
TABLES = ("results", "by_day", "diagnostics", "palantir", "lineage")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=common.DEVELOPMENTAL, help="score one preset (one timed unit)")
    parser.add_argument("--combine", action="store_true", help="merge the per-preset parts, compute gates")
    args = parser.parse_args()
    started, cpu0 = time.time(), common.cpu_seconds()
    PARTS.mkdir(parents=True, exist_ok=True)
    if args.preset:
        acc: dict[str, list[dict[str, Any]]] = {k: [] for k in TABLES}
        for seed in range(3):
            for method in common.METHODS:
                part = run_unit(args.preset, seed, method)
                for k in acc:
                    acc[k] += part[k]
                print(f"[S10 score] {args.preset} s{seed} {method}", flush=True)
        for k, v in acc.items():
            pd.DataFrame(v).to_csv(PARTS / f"{args.preset}_{k}.csv", index=False)
        common.checkpoint(f"score_orientation_{args.preset}", started, cpu0, {"n_results": len(acc["results"])})
        return
    if not args.combine:
        parser.error("use --preset P or --combine")
    frames = {k: pd.concat([pd.read_csv(PARTS / f"{p}_{k}.csv") for p in common.DEVELOPMENTAL], ignore_index=True) for k in TABLES}
    frames["results"].to_csv(common.OUT / "results.csv", index=False)
    frames["by_day"].to_csv(common.OUT / "by_day.csv", index=False)
    frames["diagnostics"].to_csv(common.OUT / "diagnostics.csv", index=False)
    frames["palantir"].to_csv(common.OUT / "palantir_runs.csv", index=False)
    frames["lineage"].to_csv(common.OUT / "lineage_sensitivity.csv", index=False)
    g = gates(frames["results"], frames["diagnostics"], frames["palantir"])
    common.write_json(common.OUT / "orientation_gates.json", g)
    common.checkpoint("score_orientation", started, cpu0, {"n_results": len(frames["results"])})
    print(json.dumps(common.jsonable(g), indent=1))


if __name__ == "__main__":
    main()
