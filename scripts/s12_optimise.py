#!/usr/bin/env python3
"""orient/S12 Step 1: flow-J optimisation on development data only.

STATUS: prepared before the Step 0 decision and NOT executed: Step 0's fixed
rule said stop (benchmarks/synthetic_multimodal/s12/RESULT_S12.md).  Only the
dynamo worker was probed on a toy linear field.

Development set: v3_default s0, s1 and v3_transition s0, s1 (never s2/s3 or any
test preset).  Start from the S10 default OT flow J and vary one component at a
time (OT eps; last-day coverage; estimator), then combine the best choice of
each component.  Flow sources other than OT (labelling, lineage) are a
comparison only.

Selection objective (fixed by the task): primary -- mean fork AUC over the
development set (PCA-20); ties within 0.01 broken by lower J_ref error, then by
the transition-bin error (|detected - true|, day-weighted bins, informative
datasets only).

Stages:
* ``jref --preset P --seed N --embedding E``        J_ref (scoring only) cache.
* ``run --phase {oat,combined} --preset P --seed N --embedding E``  every
                                                     configuration of the phase on one
                                                     dataset x embedding.
* ``dynamo_worker --inp X --out Y``                   (``.venv-dynamo`` only) SparseVFC.
* ``select --phase {oat,combined}``                   optimisation_log.csv, selection.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import s12_step0 as lib  # noqa: E402

ROOT = lib.ROOT
DEV = (("v3_default", 0), ("v3_default", 1), ("v3_transition", 0), ("v3_transition", 1))
N_BINS = 12
KAPPA = 0.5
DEFAULT = {"source": "ot", "eps": 0.05, "last": "extrapolate", "estimator": "md", "hidden": [96, 96, 96], "wd": 1e-4}
DYNAMO_PY = ROOT / ".venv-dynamo/bin/python"


def cid(cfg: dict[str, Any]) -> str:
    est = cfg["estimator"]
    if est == "md":
        e = f"md{'x'.join(map(str, cfg['hidden']))}_wd{cfg['wd']:g}"
    elif est == "ensemble":
        e = f"ens5_md{'x'.join(map(str, cfg['hidden']))}_wd{cfg['wd']:g}"
    elif est == "ridge":
        e = f"ridge_k{cfg['k']}"
    else:
        e = est
    if cfg["source"] != "ot":
        return f"src-{cfg['source']}__{e}"
    return f"ot_eps{cfg['eps']:g}__{cfg['last']}__{e}"


def oat_configs() -> list[dict[str, Any]]:
    out = [dict(DEFAULT)]
    for eps in (0.0125, 0.025, 0.1, 0.2):
        out.append({**DEFAULT, "eps": eps})
    out.append({**DEFAULT, "last": "backward"})
    for hidden in ([64, 64], [96, 96, 96], [128, 128, 128]):
        for wd in (1e-5, 1e-4, 1e-3):
            c = {**DEFAULT, "hidden": hidden, "wd": wd}
            if cid(c) != cid(DEFAULT):
                out.append(c)
    out.append({**DEFAULT, "estimator": "ensemble"})
    for k in (15, 30, 60):
        out.append({**DEFAULT, "estimator": "ridge", "k": k})
    out.append({**DEFAULT, "estimator": "dynamo"})
    out.append({**DEFAULT, "source": "labelling"})
    out.append({**DEFAULT, "source": "lineage"})
    return out


# ---------------------------------------------------------------------------
# day-weighted bins and transition scoring
# ---------------------------------------------------------------------------
def day_weighted_bins(pseudotime: np.ndarray, day: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """12 bins at weighted quantiles of pseudotime, cell weight 1 / n_cells(day); no cell removed.

    Cells are sorted by pseudotime; bin = floor(12 x (cumulative weight at the
    cell's midpoint) / total weight), capped at 11.
    """
    uniq, counts = np.unique(day, return_counts=True)
    w = 1.0 / counts[np.searchsorted(uniq, day)]
    order = np.argsort(pseudotime, kind="stable")
    cw = np.cumsum(w[order]) - 0.5 * w[order]
    b = np.empty(len(day), dtype=int)
    b[order] = np.minimum((cw / w.sum() * n_bins).astype(int), n_bins - 1)
    return b


def bin_medians(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([np.median(x[b == k]) if np.any(b == k) else np.nan for k in range(N_BINS)])


def peak_to_median(med: np.ndarray) -> float:
    return float(np.nanmax(med) / max(np.nanmedian(np.abs(med)), 1e-300))


def rising_run(med: np.ndarray, det: int) -> int:
    n, k = 0, det
    while k > 0 and med[k] > med[k - 1]:
        n += 1
        k -= 1
    return n


def strict_pseudotime(preset: str, seed: int, names: np.ndarray) -> tuple[np.ndarray, str]:
    """Palantir strict mode, earliest-day root r0 (S10 run, or S12 run for new datasets)."""
    for base in (lib.S10ARR, lib.ARR):
        p = base / f"{preset}_s{seed}" / "palantir.npz"
        if p.exists():
            arr = np.load(p)
            key = [k for k in arr.files if k.endswith("_strict_r0") and k.startswith("day2")]
            if key and np.array_equal(arr["names"].astype(str), names):
                return arr[key[0]], f"{p.relative_to(ROOT)}:{key[0]}"
    raise RuntimeError(f"no strict r0 Palantir run for {preset} s{seed}")


def transition_truth(src: Any, b: np.ndarray) -> tuple[int, np.ndarray]:
    meta = src.uns["synthetic_multimodal"]
    z1_crit = float(meta["z1_crit"]) if "z1_crit" in meta else 1.0
    zt = np.asarray(src.obsm["z_true"], np.float64)
    fe = KAPPA * ((zt[:, 1] - z1_crit) - 3.0 * zt[:, 0] ** 2)
    med = bin_medians(fe, b)
    return int(np.nanargmax(med)), med


def transition_scores(lam1: np.ndarray, b: np.ndarray, true_bin: int | None) -> dict[str, Any]:
    med = bin_medians(lam1, b)
    det = int(np.nanargmax(med))
    out = {"detected_bin": det, "peak_to_median": peak_to_median(med), "n_rising_bins_before_detected": rising_run(med, det),
           "lam1_bin_medians": json.dumps([round(float(x), 6) for x in med])}
    if true_bin is not None:
        out.update({"true_bin": true_bin, "informative": bool(2 <= true_bin <= 9), "signed_offset": det - true_bin,
                    "abs_offset": abs(det - true_bin)})
    return out


# ---------------------------------------------------------------------------
# flows and estimators
# ---------------------------------------------------------------------------
def lineage_velocity(z: np.ndarray, clone: np.ndarray, days: np.ndarray, hours: np.ndarray) -> np.ndarray:
    """S11 lineage displacement: (clone's next-day centroid - cell) / dt for cells with an observed clone."""
    v = np.full(z.shape, np.nan, dtype=np.float64)
    uniq = np.unique(days)
    has = np.array([c is not None and c == c for c in clone])
    for d0, d1 in zip(uniq[:-1], uniq[1:]):
        dt = float(np.median(hours[days == d1]) - np.median(hours[days == d0]))
        later = (days == d1) & has
        cent = {}
        for c in np.unique(clone[later].astype(str)):
            cent[c] = z[later & (clone.astype(str) == c)].mean(axis=0)
        for i in np.flatnonzero((days == d0) & has):
            c = str(clone[i])
            if c in cent:
                v[i] = (cent[c] - z[i]) / dt
    return v.astype(np.float32)


def dynamo_jacobian(z: np.ndarray, v: np.ndarray, fit_rows: np.ndarray, work: Path) -> tuple[np.ndarray, dict[str, Any]]:
    inp, out = work / "dynamo_in.npz", work / "dynamo_out.npz"
    np.savez(inp, z_fit=np.asarray(z, np.float32)[fit_rows], v_fit=np.asarray(v, np.float32)[fit_rows], z_all=np.asarray(z, np.float32))
    env = {**os.environ}
    t0 = time.time()
    res = subprocess.run([str(DYNAMO_PY), str(Path(__file__)), "dynamo_worker", "--inp", str(inp), "--out", str(out)],
                         env=env, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"dynamo worker failed: {res.stderr[-2000:]}")
    arr = np.load(out)
    info = json.loads(str(arr["info"]))
    J = np.asarray(arr["J"], np.float32)
    inp.unlink(); out.unlink()
    return J, {**info, "seconds": round(time.time() - t0, 2)}


def dynamo_worker(inp: str, out: str) -> None:
    import anndata as ad
    import dynamo
    from dynamo.vectorfield.utils import Jacobian_rkhs_gaussian
    arr = np.load(inp)
    a = ad.AnnData(X=np.zeros((len(arr["z_fit"]), 1), dtype=np.float32))
    a.obsm["X_pca"] = arr["z_fit"].astype(np.float32)
    a.obsm["velocity_pca"] = arr["v_fit"].astype(np.float32)
    dynamo.vf.VectorField(a, basis="pca", normalize=False, grid_velocity=False, cores=1)
    vf = a.uns["VecFld_pca"]
    z_all = arr["z_all"].astype(np.float64)
    J = np.empty((len(z_all), z_all.shape[1], z_all.shape[1]), dtype=np.float32)
    for lo in range(0, len(z_all), 1024):
        hi = min(lo + 1024, len(z_all))
        raw = np.asarray(Jacobian_rkhs_gaussian(z_all[lo:hi], vf), np.float64)       # (d, d, n)
        J[lo:hi] = np.transpose(raw, (2, 0, 1))
    info = {"dynamo_version": getattr(dynamo, "__version__", "?"), "method": "dynamo.vf.VectorField (SparseVFC defaults, normalize=False) "
            "on the fit cells + Jacobian_rkhs_gaussian at all cells", "n_fit": int(len(arr["z_fit"]))}
    np.savez(out, J=J, info=json.dumps(info))


def estimate(cfg: dict[str, Any], z: np.ndarray, v: np.ndarray, work: Path) -> tuple[np.ndarray, dict[str, Any]]:
    fit_rows = np.flatnonzero(np.isfinite(v).all(axis=1))
    est = cfg["estimator"]
    if est == "md":
        J, _, info = lib.md_jacobian(z, v, fit_rows, hidden=tuple(cfg["hidden"]), weight_decay=cfg["wd"], seed=0)
    elif est == "ensemble":
        Js, infos = [], []
        for s in range(5):
            Jk, _, ik = lib.md_jacobian(z, v, fit_rows, hidden=tuple(cfg["hidden"]), weight_decay=cfg["wd"], seed=s)
            Js.append(Jk.astype(np.float64)); infos.append(ik)
        J = np.mean(Js, axis=0).astype(np.float32)
        info = {"members": infos, "rule": "mean Jacobian of MeasuredDrift seeds 0-4"}
    elif est == "ridge":
        J, info = lib.local_ridge_jacobian(z, v, fit_rows, int(cfg["k"]))
    elif est == "dynamo":
        J, info = dynamo_jacobian(z, v, fit_rows, work)
    else:
        raise ValueError(est)
    return J, {**info, "n_fit_cells": int(len(fit_rows))}


def flow(cfg: dict[str, Any], preset: str, seed: int, emb: str, z: np.ndarray, src: Any) -> tuple[np.ndarray | None, dict[str, Any]]:
    days = src.obs["day"].to_numpy(int)
    hours = src.obs["time_hours"].to_numpy(np.float64)
    if cfg["source"] == "ot":
        v, info = lib.ot_velocity(z, days, hours, eps_factor=cfg["eps"], backward_last=(cfg["last"] == "backward"))
        return v, {"ot_pairs": [{k: x[k] for k in ("from_day", "to_day", "epsilon", "n_iter", "final_err")} for x in info]}
    if cfg["source"] == "lineage":
        v = lineage_velocity(z, src.obs["clone"].astype(object).to_numpy(), days, hours)
        return v, {"n_cells_with_lineage_velocity": int(np.isfinite(v).all(1).sum()),
                   "definition": "S11: (mean embedding of the clone's next-day cells - cell) / dt; observed clone labels"}
    if cfg["source"] == "labelling":
        if emb not in ("pca20", "fa20"):
            return None, {"status": "labelling velocity exists only for the S10 embeddings (pca20, fa20)"}
        fit = np.load(lib.S10ARR / f"{preset}_s{seed}" / f"fit_{emb[:-2]}.npz")
        return np.asarray(fit["v_lab_smoothed_shared"], np.float32), {"definition": "S10 v_lab_smoothed_shared (kNN-30 smoothed, shared gamma)"}
    raise ValueError(cfg["source"])


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def jref_path(preset: str, seed: int, emb: str) -> Path:
    return lib.unit_dir(preset, seed) / f"jref_{emb}.npz"


def stage_jref(preset: str, seed: int, emb: str) -> None:
    import anndata as ad
    started, cpu0 = time.time(), lib.cpu_seconds()
    row = lib.manifest_row(preset, seed)
    src = ad.read_h5ad(ROOT / row["path"])
    meta = lib.linear_meta(preset, seed, emb, src)
    if meta is None:
        lib.checkpoint(f"jref_{preset}_s{seed}_{emb}", started, cpu0, {"status": "not defined: embedding has no linear map"})
        return
    z = lib.load_embedding(preset, seed, emb, src)
    vt = lib.true_expected_velocity(src, meta)
    Jr = lib.jref(z, vt)
    dest = jref_path(preset, seed, emb)
    np.savez(dest, J_ref=Jr, v_true_expected=vt)
    lib.checkpoint(f"jref_{preset}_s{seed}_{emb}", started, cpu0, {"input_sha256": row["sha256"], "path": str(dest.relative_to(ROOT)),
                                                                   "sha256": lib.sha256(dest),
                                                                   "definition": "local_jacobian(k=30, OrientConfig().ridge) of true_velocity_in_embedding(at='expected')"})


def score_operator(J: np.ndarray, z: np.ndarray, src: Any, preset: str, seed: int, emb: str, b: np.ndarray | None,
                   true_bin: int | None) -> tuple[dict[str, Any], np.ndarray]:
    truth = lib.truth_for_scoring(src)
    res, arrays = lib.fork_scores(J, z, truth, with_extras=False)
    jp = jref_path(preset, seed, emb)
    res["jref_error"] = lib.jref_error(J, np.load(jp)["J_ref"]) if jp.exists() else np.nan
    if b is not None:
        res.update(transition_scores(arrays["lam1"].astype(np.float64), b, true_bin))
    return res, arrays["lam1"]


def run_configs(configs: list[dict[str, Any]], preset: str, seed: int, emb: str, tag: str, save_J: set[str] | None = None) -> list[dict[str, Any]]:
    import anndata as ad
    import torch
    torch.set_num_threads(lib.threads())
    row = lib.manifest_row(preset, seed)
    src = ad.read_h5ad(ROOT / row["path"], backed="r")
    z = lib.load_embedding(preset, seed, emb, src).astype(np.float64)
    names = src.obs_names.to_numpy(str)
    days = src.obs["day"].to_numpy(int)
    b, true_bin, pt_key = None, None, None
    if len(np.unique(days)) > 1:
        pt, pt_key = strict_pseudotime(preset, seed, names)
        b = day_weighted_bins(pt, days)
        if preset != "v3_nofork":
            true_bin, _ = transition_truth(src, b)
    rows = []
    work = lib.unit_dir(preset, seed)
    for cfg in configs:
        t0, c0 = time.time(), lib.cpu_seconds()
        name = cid(cfg)
        base = {"config_id": name, "preset": preset, "seed": seed, "embedding": emb, "tag": tag, "config": json.dumps(cfg)}
        try:
            v, finfo = flow(cfg, preset, seed, emb, z, src)
            if v is None:
                rows.append({**base, "status": "skipped", "note": finfo["status"]})
                continue
            if cfg["source"] == "labelling" and emb in ("pca20", "fa20"):
                # S10 J_flow_smoothed_shared is MeasuredDrift(d, seed=0) on this velocity: reuse, do not refit
                J = np.load(lib.S10ARR / f"{preset}_s{seed}" / f"fit_{emb[:-2]}.npz")["J_flow_smoothed_shared"]
                einfo = {"reused": "S10 J_flow_smoothed_shared"}
            else:
                J, einfo = estimate(cfg, z, v, work)
            res, lam1 = score_operator(J, z, src, preset, seed, emb, b, true_bin)
            if save_J and name in save_J:
                np.savez(work / f"J_{tag}_{emb}_{name}.npz", J=J.astype(np.float32), v=v)
            rows.append({**base, "status": "ok", **res, "pseudotime": pt_key, "n_fit_cells": einfo.get("n_fit_cells"),
                         "flow_info": json.dumps(lib.jsonable(finfo))[:2000], "estimator_info": json.dumps(lib.jsonable(einfo))[:2000],
                         "seconds": round(time.time() - t0, 1), "cpu_seconds_in_process": round(lib.cpu_seconds() - c0, 1)})
        except Exception as exc:  # retry once, then mark
            print(f"[S12] {name} failed once: {exc!r}; retrying", flush=True)
            try:
                v, finfo = flow(cfg, preset, seed, emb, z, src)
                J, einfo = estimate(cfg, z, v, work)
                res, lam1 = score_operator(J, z, src, preset, seed, emb, b, true_bin)
                rows.append({**base, "status": "ok_after_retry", **res, "first_error": repr(exc)[:300]})
            except Exception as exc2:
                rows.append({**base, "status": "failed_twice", "error": repr(exc2)[:500]})
        print(f"[S12] {preset} s{seed} {emb} {name}: {rows[-1].get('auc_lam1')} ({time.time() - t0:.0f}s)", flush=True)
    src.file.close()
    return rows


def phase_configs(phase: str) -> list[dict[str, Any]]:
    if phase == "oat":
        return oat_configs()
    sel = json.loads((lib.OUT / "selection_oat.json").read_text())
    return [c for c in sel["combined_configs"]]


def stage_run(phase: str, preset: str, seed: int, emb: str) -> None:
    if (preset, seed) not in DEV:
        raise SystemExit("Step 1 uses the development set only")
    started, cpu0 = time.time(), lib.cpu_seconds()
    rows = run_configs(phase_configs(phase), preset, seed, emb, phase)
    parts = lib.OUT / "opt_parts"
    parts.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(parts / f"{phase}_{preset}_s{seed}_{emb}.csv", index=False)
    lib.checkpoint(f"opt_{phase}_{preset}_s{seed}_{emb}", started, cpu0, {"n_configs": len(rows),
                                                                         "status_counts": {s: sum(r["status"] == s for r in rows) for s in {r["status"] for r in rows}}})


def objective(frame: Any, config_id: str) -> dict[str, Any]:
    sub = frame[(frame.config_id == config_id) & (frame.embedding == "pca20") & frame.status.str.startswith("ok")]
    dev = sub[[(p, s) in DEV for p, s in zip(sub.preset, sub.seed)]]
    inf = dev[dev.get("informative", False) == True]  # noqa: E712
    return {"config_id": config_id, "n_datasets": int(len(dev)), "mean_auc_pca20": float(dev.auc_lam1.mean()) if len(dev) == 4 else np.nan,
            "mean_jref_error_pca20": float(dev.jref_error.mean()) if len(dev) else np.nan,
            "mean_abs_bin_error_informative": float(inf.abs_offset.mean()) if len(inf) else np.nan,
            "n_informative": int(len(inf))}


def rank(objs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Primary mean AUC; ties within 0.01 of the best broken by J_ref error, then by bin error."""
    ok = [o for o in objs if np.isfinite(o["mean_auc_pca20"])]
    best = max(o["mean_auc_pca20"] for o in ok)
    tied = [o for o in ok if o["mean_auc_pca20"] >= best - 0.01]
    tied.sort(key=lambda o: (o["mean_jref_error_pca20"] if np.isfinite(o["mean_jref_error_pca20"]) else np.inf,
                             o["mean_abs_bin_error_informative"] if np.isfinite(o["mean_abs_bin_error_informative"]) else np.inf))
    return tied


def stage_select(phase: str) -> None:
    import glob
    import pandas as pd
    started, cpu0 = time.time(), lib.cpu_seconds()
    frames = [pd.read_csv(p) for p in sorted(glob.glob(str(lib.OUT / "opt_parts" / "*.csv")))]
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(lib.OUT / "optimisation_log.csv", index=False)
    configs = {r.config_id: json.loads(r.config) for r in frame.drop_duplicates("config_id").itertuples()}
    ot_ids = [c for c, cfg in configs.items() if cfg["source"] == "ot"]
    objs = {c: objective(frame, c) for c in configs}
    pd.DataFrame(list(objs.values())).to_csv(lib.OUT / f"objective_{phase}.csv", index=False)
    out: dict[str, Any] = {"phase": phase, "objective": "mean fork AUC over the development set (PCA-20); ties within 0.01 "
                           "broken by lower J_ref error, then by the transition bin error (informative datasets only)"}
    if phase == "oat":
        default_id = cid(DEFAULT)
        best_by_component = {}
        comps = {"eps": [c for c in ot_ids if configs[c]["last"] == "extrapolate" and configs[c]["estimator"] == "md"
                         and configs[c]["hidden"] == DEFAULT["hidden"] and configs[c]["wd"] == DEFAULT["wd"]],
                 "last": [c for c in ot_ids if configs[c]["eps"] == DEFAULT["eps"] and configs[c]["estimator"] == "md"
                          and configs[c]["hidden"] == DEFAULT["hidden"] and configs[c]["wd"] == DEFAULT["wd"]],
                 "estimator": [c for c in ot_ids if configs[c]["eps"] == DEFAULT["eps"] and configs[c]["last"] == "extrapolate"]}
        for comp, ids in comps.items():
            ranked = rank([objs[c] for c in ids])
            best_by_component[comp] = {"best": ranked[0]["config_id"], "candidates": [objs[c] for c in ids]}
        b = {k: configs[v["best"]] for k, v in best_by_component.items()}
        combined = {**DEFAULT, "eps": b["eps"]["eps"], "last": b["last"]["last"],
                    **{k: b["estimator"][k] for k in ("estimator", "hidden", "wd", "k") if k in b["estimator"]}}
        out.update({"default": default_id, "best_by_component": best_by_component, "combined_configs": [combined],
                    "combined_config_id": cid(combined), "combined_is_already_evaluated": cid(combined) in configs})
    else:
        ranked = rank([objs[c] for c in ot_ids])
        out.update({"ranking_ot_configs_tied_block": ranked, "selected_config_id": ranked[0]["config_id"],
                    "selected_config": configs[ranked[0]["config_id"]],
                    "rule_note": "selection over every OT configuration recorded (default, one-at-a-time and combined)",
                    "all_ot_objectives_sorted": sorted([objs[c] for c in ot_ids], key=lambda o: -o["mean_auc_pca20"] if np.isfinite(o["mean_auc_pca20"]) else 0)})
    lib.write_json(lib.OUT / f"selection_{phase}.json", out)
    print(json.dumps(lib.jsonable({k: out[k] for k in out if k in ("combined_config_id", "selected_config_id", "selected_config")}), indent=1))
    lib.checkpoint(f"select_{phase}", started, cpu0, {})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("jref", "run", "select", "dynamo_worker"))
    parser.add_argument("--phase", choices=("oat", "combined"))
    parser.add_argument("--preset")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--embedding")
    parser.add_argument("--inp")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.stage == "dynamo_worker":
        dynamo_worker(args.inp, args.out)
    elif args.stage == "jref":
        stage_jref(args.preset, args.seed, args.embedding)
    elif args.stage == "run":
        stage_run(args.phase, args.preset, args.seed, args.embedding)
    elif args.stage == "select":
        stage_select(args.phase)


if __name__ == "__main__":
    main()
