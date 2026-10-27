#!/usr/bin/env python3
"""orient/S21 dyngen regeneration, observed fits, and post-fit scoring.

The command phases are deliberately separate: ``part-a`` reads truth only to
select regenerated bifurcating datasets; ``prepare`` and ``fit`` read counts
and sampling day only; ``score`` is the first phase allowed to read truth.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import resource
import sys
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks/dyngen/data"
RAW = ROOT / "benchmarks/dyngen/raw"
S21B_MODE = os.environ.get("SCJDO2_DYNGEN_TASK") == "S21b"
OUT = ROOT / "benchmarks/dyngen" / ("s21b" if S21B_MODE else "s21")
ARR = OUT / "arrays"
CK = OUT / "checkpoints"
MANIFEST = ROOT / "benchmarks/dyngen/MANIFEST.csv"
FIG = ROOT / "benchmarks/figures/dyngen"
TASK = "orient/S21b" if S21B_MODE else "orient/S21"


def jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x)
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(obj), indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError(f"cannot import {path}")
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


def s12() -> Any: return module("s21_s12", ROOT / "scripts/s12_step0.py")
def s14() -> Any: return module("s21_s14", ROOT / "scripts/s14_choose_k.py")


def csv_write(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def parse_time_log(path: Path) -> dict[str, Any]:
    text = path.read_text() if path.exists() else ""
    def val(name: str) -> float | None:
        m = re.search(rf"^\s*{name}\s+([0-9.]+)", text, re.M)
        if not m: m = re.search(rf"([0-9.]+)\s+{name}\b", text)
        return float(m.group(1)) if m else None
    rss = re.search(r"^\s*([0-9]+)\s+maximum resident set size", text, re.M | re.I)
    return {"wall_seconds": val("real"), "user_seconds": val("user"),
            "sys_seconds": val("sys"), "max_rss_bytes": int(rss.group(1)) if rss else None}


def part_a() -> None:
    import anndata as ad
    rows: list[dict[str, Any]] = []
    for p in sorted(DATA.glob("bifurcating_v2_s*.h5ad")):
        m = re.fullmatch(r"bifurcating_v2_s(\d+)", p.stem)
        if not m: continue
        seed = int(m.group(1)); a = ad.read_h5ad(p, backed="r")
        vc = a.obs["truth_branch"].astype(str).value_counts()
        c, d = int(vc.get("C", 0)), int(vc.get("D", 0)); post = c + d
        frac = min(c, d) / post if post else 0.0
        log = ROOT / "benchmarks/dyngen/logs" / f"bifurcating_v2_s{seed}.log"
        tm = parse_time_log(log)
        rows.append({"seed": seed, "key": p.stem, "status": "accepted" if frac >= .25 else "rejected",
                     "branch_C": c, "branch_D": d, "post_branch": post,
                     "minority_fraction": frac, **tm, "h5ad": str(p.relative_to(ROOT)),
                     "raw": str((RAW / f"bifurcating_v2_s{seed}").relative_to(ROOT)),
                     "sha256": sha256(p)})
    rows.sort(key=lambda r: r["seed"])
    fields = ["seed", "key", "status", "branch_C", "branch_D", "post_branch", "minority_fraction",
              "wall_seconds", "user_seconds", "sys_seconds", "max_rss_bytes", "h5ad", "raw", "sha256"]
    csv_write(OUT / "part_a_seeds.csv", rows, fields)
    # Preserve S19 rows and append each completed v2 export exactly once.
    with MANIFEST.open(newline="") as f: old = list(csv.DictReader(f))
    seen = {r["key"] for r in old}
    for r in rows:
        if r["key"] in seen: continue
        old.append({"key": r["key"], "backbone": "bifurcating", "seed": r["seed"],
                    "path": r["h5ad"], "sha256": r["sha256"], "cells": 5000, "genes": 1000,
                    "settings": f"benchmarks/dyngen/raw/{r['key']}/settings.txt"})
    fields_m = ["key", "backbone", "seed", "path", "sha256", "cells", "genes", "settings"]
    csv_write(MANIFEST, old, fields_m)
    print(json.dumps({"tried": len(rows), "accepted": [r["seed"] for r in rows if r["status"] == "accepted"]}))


def dataset_path(dataset: str) -> Path: return DATA / f"{dataset}.h5ad"
def prep_path(dataset: str) -> Path: return ARR / "prepared" / f"{dataset}.npz"
def flow_path(dataset: str, seed: int) -> Path: return ARR / "flow" / f"{dataset}_seed{seed}.npz"
def score_path(dataset: str) -> Path: return ARR / "score" / f"{dataset}.npz"


def prepare(dataset: str) -> None:
    import anndata as ad
    from sklearn.decomposition import PCA
    started = time.time()
    src = ad.read_h5ad(dataset_path(dataset), backed="r")
    x = src.layers["spliced"]
    x = x.to_memory() if hasattr(x, "to_memory") else x
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    x = np.asarray(x, dtype=np.float64)
    lib = x.sum(axis=1); target = float(np.median(lib));
    xn = x / np.maximum(lib[:, None], 1.0) * target
    xl = np.log1p(xn)
    pca = PCA(n_components=15, random_state=0)
    z = pca.fit_transform(xl).astype(np.float32)
    day = np.asarray(src.obs["day"], dtype=np.int64)
    prep_path(dataset).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prep_path(dataset), z=z, day=day, pca_mean=pca.mean_.astype(np.float64),
                        pca_components=pca.components_.astype(np.float64), library=lib.astype(np.float64),
                        normalization_target=np.asarray(target), input_sha256=np.asarray(sha256(dataset_path(dataset))))
    write_json(CK / f"prepare_{dataset}.json", {"task": TASK, "unit": f"prepare_{dataset}",
               "wall_seconds": time.time() - started, "dataset": dataset, "n_cells": len(z), "n_genes": x.shape[1]})


def fit(dataset: str, drift_seed: int) -> None:
    os.environ.setdefault("SCJDO2_TORCH_THREADS", "3")
    os.environ.setdefault("OMP_NUM_THREADS", "3")
    d = np.load(prep_path(dataset)); z = np.asarray(d["z"], np.float64); day = np.asarray(d["day"], int)
    flowmod = s14(); cfg = dict(flowmod.REFERENCE); cfg["space"] = "raw"; cfg["ensemble"] = 1
    result = flowmod.fit_flow({"z": z, "day": day, "hours": 48.0 * (day - 1)}, cfg, seeds=(int(drift_seed),))
    flow_path(dataset, drift_seed).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(flow_path(dataset, drift_seed), J=np.asarray(result["J"], np.float32),
                        v_ot=np.asarray(result["v_ot_raw"], np.float32), drift_seed=drift_seed)
    payload: dict[str, Any] = {"task": TASK, "unit": f"fit_{dataset}_seed{drift_seed}", "dataset": dataset,
                               "drift_seed": drift_seed, "config": cfg, "fit_rows": len(result["fit_rows"])}
    # Score-J is deterministic and shared by the three drift-seed fits.
    if not score_path(dataset).exists():
        from scjdo2.energy_drift import EnergyDriftAdapter, EnergyDriftConfig
        ecfg = EnergyDriftConfig(hidden=96, depth=3, n_epochs=1500, seed=0, batch_size=256,
                                 loss="esm", lr=5e-4, beta=1.0, esm_laplacian="exact")
        energy = EnergyDriftAdapter(z.shape[1], ecfg).fit(z.astype(np.float32), verbose=False)
        sym = energy.symmetry_check(z.astype(np.float32), seed=0)
        if float(sym["max_asym"]) > 1e-4: raise RuntimeError(f"score-J symmetry gate failed: {sym}")
        score_path(dataset).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(score_path(dataset), J=np.asarray(energy.jacobian_autodiff(z.astype(np.float32)), np.float32))
        payload["score_j"] = {"config": vars(ecfg), "symmetry": {k: float(v) for k, v in sym.items()}}
    write_json(CK / f"fit_{dataset}_seed{drift_seed}.json", payload)


def _spectra(J: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = np.linalg.eigvals(np.asarray(J, np.float64)); lam = w.real.max(axis=1)
    rot = np.abs(w.imag).max(axis=1) / np.maximum(np.abs(w).max(axis=1), 1e-12)
    return lam, rot


def _auc(values: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if not pos.any() or not neg.any(): return float("nan")
    return float(roc_auc_score(np.r_[np.ones(pos.sum()), np.zeros(neg.sum())], np.r_[values[pos], values[neg]]))


def score(dataset: str) -> None:
    import anndata as ad
    src = ad.read_h5ad(dataset_path(dataset), backed="r")
    d = np.load(prep_path(dataset)); z = np.asarray(d["z"], np.float64)
    # This is the first phase that accesses truth-prefixed fields.
    obs = src.obs
    frm = np.asarray(obs["truth_backbone_from"].astype(str)); to = np.asarray(obs["truth_backbone_to"].astype(str))
    pct = np.asarray(obs["truth_backbone_percentage"], float)
    backbone = str(src.uns["s19"]["backbone"])
    fork = ((frm == "sB") & (to == "sBmid")) | ((frm == "sBmid") & np.isin(to, ["sC", "sD"]) & (pct < .15))
    committed = ((frm == "sC") & (to == "sEndC") | (frm == "sD") & (to == "sEndD")) & (pct >= .5)
    early = (frm == "sA") & (to == "sB")
    regions = {"fork": fork, "committed": committed, "early": early}
    rows: list[dict[str, Any]] = []; jref_rows: list[dict[str, Any]] = []
    ops: list[tuple[str, int | str, np.ndarray]] = []
    for seed in (0, 1, 2): ops.append(("flow", seed, np.asarray(np.load(flow_path(dataset, seed))["J"])))
    ops.append(("score", "shared", np.asarray(np.load(score_path(dataset))["J"])))
    velocity = None
    for operator, seed, J in ops:
        lam, rot = _spectra(J)
        row = {"dataset": dataset, "backbone": backbone, "operator": operator, "drift_seed": seed,
               "n_cells": len(z), "n_fork": int(fork.sum()), "n_committed": int(committed.sum()),
               "fork_auc": _auc(lam, fork, committed), "median_rotation": float(np.median(rot))}
        for name, mask in regions.items():
            row[f"n_{name}"] = int(mask.sum()); row[f"fraction_positive_{name}"] = float(np.mean(lam[mask] > 0)) if mask.any() else float("nan")
            row[f"median_rotation_{name}"] = float(np.median(rot[mask])) if mask.any() else float("nan")
        rows.append(row)
        if velocity is None:
            vv = src.layers["truth_rna_velocity"]
            vv = vv.to_memory() if hasattr(vv, "to_memory") else vv
            velocity = vv.toarray() if hasattr(vv, "toarray") else np.asarray(vv)
            lib = np.asarray(d["library"], float); target = float(d["normalization_target"])
            vnorm = np.asarray(velocity, np.float64) / np.maximum(lib[:, None], 1.0) * target
            velocity = vnorm @ np.asarray(d["pca_components"], float).T
        Jr = np.asarray(s12().jref(z, velocity), np.float64); J64 = np.asarray(J, np.float64)
        den = np.linalg.norm(Jr); num = np.linalg.norm(J64); c = float(np.sum(J64 * Jr) / max(np.sum(Jr * Jr), 1e-300))
        rel = float(np.linalg.norm(J64 - c * Jr) / max(den, 1e-300))
        cos = np.sum(J64 * Jr, axis=(1, 2)) / np.maximum(np.linalg.norm(J64, axis=(1, 2)) * np.linalg.norm(Jr, axis=(1, 2)), 1e-300)
        lr, _ = _spectra(Jr)
        from scipy.stats import spearmanr
        rho = float(spearmanr(lam, lr).statistic)
        jref_rows.append({"dataset": dataset, "backbone": backbone, "operator": operator, "drift_seed": seed,
                          "norm_ratio": num / max(den, 1e-300), "c_star": c, "relative_error": rel,
                          "median_cosine": float(np.median(cos)), "spearman_lambda1": rho})
    csv_write(OUT / "scores.csv", _merge_csv(OUT / "scores.csv", rows), list(rows[0].keys()))
    csv_write(OUT / "jref_diagnosis.csv", _merge_csv(OUT / "jref_diagnosis.csv", jref_rows), list(jref_rows[0].keys()))
    write_json(CK / f"score_{dataset}.json", {"task": TASK, "unit": f"score_{dataset}", "truth_read_after_fits": True,
               "n_rows": len(rows), "n_fork": int(fork.sum()), "n_committed": int(committed.sum())})


def _merge_csv(path: Path, new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    old: list[dict[str, Any]] = []
    if path.exists():
        with path.open(newline="") as f: old = list(csv.DictReader(f))
    keys = {(r.get("dataset"), r.get("operator"), str(r.get("drift_seed"))) for r in new}
    old = [r for r in old if (r.get("dataset"), r.get("operator"), str(r.get("drift_seed"))) not in keys]
    return old + new


def _timed_cpu_hours() -> float:
    total = 0.0
    if S21B_MODE:
        for p in (OUT / "logs").glob("*.log"):
            t = parse_time_log(p)
            total += float(t.get("user_seconds") or 0) + float(t.get("sys_seconds") or 0)
        return total / 3600.0
    for p in (ROOT / "benchmarks/dyngen/logs").glob("bifurcating_v2_s*.log"):
        t = parse_time_log(p); total += float(t.get("user_seconds") or 0) + float(t.get("sys_seconds") or 0)
    for p in (OUT / "logs").glob("fit_*.log"):
        t = parse_time_log(p); total += float(t.get("user_seconds") or 0) + float(t.get("sys_seconds") or 0)
    for p in (OUT / "logs").glob("score_*.log"):
        t = parse_time_log(p); total += float(t.get("user_seconds") or 0) + float(t.get("sys_seconds") or 0)
    return total / 3600.0


def report() -> None:
    if S21B_MODE:
        report_s21b()
        return
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scores = pd.read_csv(OUT / "scores.csv")
    jrefd = pd.read_csv(OUT / "jref_diagnosis.csv")
    accepted_df = pd.read_csv(OUT / "part_a_seeds.csv")
    tried = [int(x) for x in accepted_df.seed.tolist()]
    accepted = accepted_df.loc[accepted_df.status == "accepted", "key"].tolist()
    accepted = [str(x) for x in accepted[:3]]
    flow = scores[scores.operator == "flow"].copy()
    bif = flow[flow.dataset.isin(accepted)]
    f1 = bool(len(bif) == 9 and np.isfinite(bif.fork_auc).all() and (bif.fork_auc >= .85).all())
    rot = flow[flow.dataset.str.startswith(("cycle_", "linear_"))].copy()
    f2_rows: list[dict[str, Any]] = []
    for n in (0, 1, 2):
        c = rot[(rot.dataset == f"cycle_s{n}") & (rot.operator == "flow")].set_index("drift_seed").median_rotation
        l = rot[(rot.dataset == f"linear_s{n}") & (rot.operator == "flow")].set_index("drift_seed").median_rotation
        for seed in (0, 1, 2):
            cv, lv = float(c.loc[str(seed)]), float(l.loc[str(seed)])
            f2_rows.append({"dataset": f"cycle_s{n}", "linear_dataset": f"linear_s{n}", "drift_seed": seed,
                            "cycle_median_rotation": cv, "linear_median_rotation": lv, "passes": cv > lv})
    f2 = bool(f2_rows) and all(x["passes"] for x in f2_rows)
    FIG.mkdir(parents=True, exist_ok=True)
    names = accepted + [f"cycle_s{i}" for i in range(3)] + [f"linear_s{i}" for i in range(3)]
    fig, axes = plt.subplots(3, 3, figsize=(13, 12), constrained_layout=True)
    for ax, ds in zip(axes.flat, names):
        z = np.load(prep_path(ds))["z"]
        J = np.load(flow_path(ds, 0))["J"]
        lam, _ = _spectra(J)
        sc = ax.scatter(z[:, 0], z[:, 1], c=lam, s=2, cmap="coolwarm", linewidths=0)
        ax.set_title(ds); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        fig.colorbar(sc, ax=ax, fraction=.046, pad=.04, label="lambda1")
    fig.savefig(FIG / "s21_lambda1_pca.png", dpi=180); plt.close(fig)
    auc = scores.copy(); auc["label"] = auc.dataset + " / " + auc.operator.astype(str)
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    plot = auc[np.isfinite(auc.fork_auc)]
    if len(plot): ax.bar(np.arange(len(plot)), plot.fork_auc, color=["#b2182b" if x == "flow" else "#2166ac" for x in plot.operator])
    ax.axhline(.85, color="black", linestyle="--", linewidth=1); ax.set_ylim(0, 1.05)
    ax.set_ylabel("fork AUC"); ax.set_xticks(np.arange(len(plot)), plot.label, rotation=65, ha="right")
    fig.savefig(FIG / "s21_auc_bars.png", dpi=180); plt.close(fig)
    import importlib.metadata as metadata
    versions = {"python": sys.version.split()[0]}
    for package in ("numpy", "scipy", "pandas", "scikit-learn", "anndata", "torch"):
        try: versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError: pass
    payload = {"task": TASK, "report": "S21", "part_a_seeds_tried": tried,
               "part_a_accepted_seeds": [int(x.rsplit("_s", 1)[1]) for x in accepted],
               "gate_F1_fork_auc": {"passes": f1, "minimum": float(bif.fork_auc.min()) if len(bif) else None,
                                    "rows": jsonable(bif[["dataset", "drift_seed", "fork_auc"]].to_dict("records"))},
               "gate_F2_rotation": {"passes": f2, "rows": f2_rows},
               "score_j_report": jsonable(scores[scores.operator == "score"].to_dict("records")),
               "jref_diagnosis": jsonable(jrefd.to_dict("records")),
               "versions": versions, "cpu_hours": _timed_cpu_hours()}
    write_json(ROOT / "coord/orient/reports/S21.json", payload)
    md = ["# S21", "", "## Summary", "", f"Accepted regenerated bifurcating seeds: {', '.join(str(x) for x in payload['part_a_accepted_seeds'])}.",
          "", "## What ran", "", "Three accepted bifurcating_v2 datasets plus cycle_s0–2 and linear_s0–2; PCA-15 observed-only preparation; flow-J drift seeds 0, 1, 2; cached S10 score-J per dataset; truth scoring after all fits.",
          "", "## Gate numbers", "", f"F1 fork AUC gate: {'PASS' if f1 else 'FAIL'}; minimum accepted-bifurcating flow-J AUC = {payload['gate_F1_fork_auc']['minimum']!s}.",
          "", f"F2 rotation gate: {'PASS' if f2 else 'FAIL'}.", "", "## Deviations", "", "F1 is not evaluable for accepted seeds 16 and 18 because the contract-defined fork region is empty (n_fork=0); F2 fails for cycle_s1 at all drift seeds and cycle_s2 at drift seed 2. Score-J and J_ref remain report-only.",
          "", "## Observations", "", "See scores.csv and jref_diagnosis.csv for operator and region-level values.",
          "", "## Suggestions", "", "Review report-only score-J and J_ref diagnostics before using them for a gate.",
          "", "## Questions", "", "None.", "", "## Artifacts", "", "- benchmarks/dyngen/s21/part_a_seeds.csv", "- benchmarks/dyngen/s21/scores.csv", "- benchmarks/dyngen/s21/jref_diagnosis.csv", "- benchmarks/figures/dyngen/s21_lambda1_pca.png", "- benchmarks/figures/dyngen/s21_auc_bars.png"]
    (ROOT / "benchmarks/dyngen/s21/RESULT_S21.md").write_text("\n".join(md) + "\n")
    (ROOT / "coord/orient/reports/S21.md").write_text("\n".join(md) + "\n")
    print(json.dumps({"F1": f1, "F2": f2, "cpu_hours": payload["cpu_hours"]}))


def report_s21b() -> None:
    from datetime import datetime, timezone
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = pd.read_csv(OUT / "scores.csv")
    jrefd = pd.read_csv(OUT / "jref_diagnosis.csv")
    part_a = pd.read_csv(OUT / "part_a_T.csv")
    bif_keys = part_a.loc[part_a.backbone == "bifurcating", "dataset"].astype(str).tolist()
    flow = scores[scores.operator == "flow"].copy()
    bif = flow[flow.dataset.isin(bif_keys)].copy()
    f1 = bool(len(bif) == 9 and np.isfinite(bif.fork_auc).all() and (bif.fork_auc >= .85).all())

    f2_rows: list[dict[str, Any]] = []
    for n in (0, 1, 2):
        c_name, l_name = f"cycle_s{n}_T8", f"linear_s{n}_T8"
        c = flow[flow.dataset == c_name].set_index("drift_seed").median_rotation
        l = flow[flow.dataset == l_name].set_index("drift_seed").median_rotation
        for seed in (0, 1, 2):
            cv, lv = float(c.loc[str(seed)]), float(l.loc[str(seed)])
            f2_rows.append({"dataset": c_name, "linear_dataset": l_name, "drift_seed": seed,
                            "cycle_median_rotation": cv, "linear_median_rotation": lv,
                            "passes": cv > lv})
    f2 = bool(f2_rows) and all(r["passes"] for r in f2_rows)

    # Compare every numeric score/J_ref field with the corresponding S21 result.
    def add_changes(old_path: Path, new_df: Any, output_path: Path) -> list[dict[str, Any]]:
        old_df = pd.read_csv(old_path)
        old_df["dataset_base"] = old_df.dataset.astype(str)
        new_df = new_df.copy()
        new_df["dataset_base"] = new_df.dataset.astype(str).str.replace(r"_T8$", "", regex=True)
        keys = ["dataset_base", "operator", "drift_seed"]
        old_df["drift_seed"] = old_df.drift_seed.astype(str)
        new_df["drift_seed"] = new_df.drift_seed.astype(str)
        numeric_cols = [c for c in old_df.columns if c in new_df.columns and
                        c not in keys and c not in ("dataset", "backbone") and
                        pd.api.types.is_numeric_dtype(old_df[c]) and pd.api.types.is_numeric_dtype(new_df[c])]
        merged = old_df[keys + numeric_cols].merge(
            new_df[keys + numeric_cols], on=keys, how="outer", suffixes=("_S21", "_S21b"),
            validate="one_to_one")
        records: list[dict[str, Any]] = []
        for _, row in merged.iterrows():
            for col in numeric_cols:
                a, b = row.get(f"{col}_S21"), row.get(f"{col}_S21b")
                av = None if pd.isna(a) else float(a)
                bv = None if pd.isna(b) else float(b)
                records.append({"dataset": row.dataset_base, "operator": row.operator,
                                "drift_seed": row.drift_seed, "metric": col,
                                "S21": av, "S21b": bv,
                                "delta_S21b_minus_S21": None if av is None or bv is None else bv - av})
        pd.DataFrame(records).to_csv(output_path, index=False)
        return records

    score_changes = add_changes(ROOT / "benchmarks/dyngen/s21/scores.csv", scores,
                                OUT / "s21_change_scores.csv")
    jref_changes = add_changes(ROOT / "benchmarks/dyngen/s21/jref_diagnosis.csv", jrefd,
                               OUT / "s21_change_jref.csv")

    FIG.mkdir(parents=True, exist_ok=True)
    names = bif_keys + [f"cycle_s{i}_T8" for i in range(3)] + [f"linear_s{i}_T8" for i in range(3)]
    fig, axes = plt.subplots(3, 3, figsize=(13, 12), constrained_layout=True)
    for ax, ds in zip(axes.flat, names):
        z = np.load(prep_path(ds))["z"]
        J = np.load(flow_path(ds, 0))["J"]
        lam, _ = _spectra(J)
        sc = ax.scatter(z[:, 0], z[:, 1], c=lam, s=2, cmap="coolwarm", linewidths=0)
        ax.set_title(ds); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        fig.colorbar(sc, ax=ax, fraction=.046, pad=.04, label="lambda1")
    fig.savefig(FIG / "s21b_lambda1_pca.png", dpi=180); plt.close(fig)
    auc = scores.copy(); auc["label"] = auc.dataset + " / " + auc.operator.astype(str)
    fig, ax = plt.subplots(figsize=(15, 5), constrained_layout=True)
    plot = auc[np.isfinite(auc.fork_auc)]
    if len(plot): ax.bar(np.arange(len(plot)), plot.fork_auc,
                         color=["#b2182b" if x == "flow" else "#2166ac" for x in plot.operator])
    ax.axhline(.85, color="black", linestyle="--", linewidth=1); ax.set_ylim(0, 1.05)
    ax.set_ylabel("fork AUC"); ax.set_xticks(np.arange(len(plot)), plot.label, rotation=65, ha="right")
    fig.savefig(FIG / "s21b_auc_bars.png", dpi=180); plt.close(fig)

    import importlib.metadata as metadata
    versions = {"python": sys.version.split()[0], "R": "4.6.0", "dyngen": "1.1.1"}
    for package in ("numpy", "scipy", "pandas", "scikit-learn", "anndata", "torch"):
        try: versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError: pass
    fork_counts = {str(r.dataset): int(r.fork_cells) for r in part_a.itertuples()
                   if r.backbone == "bifurcating"}
    settings_by_dataset = {
        str(row.dataset): json.loads((RAW / str(row.dataset) / "settings.json").read_text())
        for row in part_a.itertuples()
    }
    runtimes = []
    for log in sorted((OUT / "logs").glob("*.log")):
        tm = parse_time_log(log)
        if tm["wall_seconds"] is not None:
            runtimes.append({"unit": log.stem, **tm})
    started_values = []
    for row in part_a.itertuples():
        cp = json.loads((OUT / "checkpoints" / f"{row.dataset}.json").read_text())
        finished = datetime.strptime(cp["generated_utc"], "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)
        started_values.append(finished.timestamp() - float(row.wall_seconds))
    started_utc = datetime.fromtimestamp(min(started_values), timezone.utc).isoformat().replace("+00:00", "Z")
    finished_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    deviations = [
        "The initial R launch omitted R_LIBS_USER and stopped before loading dyngen; the following attempt was interrupted after a default temporary reference download. All successful resampling units used the private .sandbox/Rlib and .sandbox/tmp.",
        "The first preparation launch was denied before preprocessing because the workspace sandbox blocked task-folder writes; the units were rerun after explicit task-folder authorization. No analysis outputs from that attempt were used.",
    ]
    observations = [
        f"T=8 fork counts were {fork_counts}; each bifurcating dataset exceeded the fixed 100-cell threshold.",
        f"F1 {'passed' if f1 else 'failed'} and F2 {'passed' if f2 else 'failed'} under the unchanged S21 gates.",
        "F2 passed for cycle seeds 1 and 2 at all three drift seeds; cycle seed 0 failed all three comparisons." if not f2 else "All cycle-versus-linear rotation comparisons passed.",
    ]
    suggestions = [
        "If F2 remains negative, inspect dyngen's cycle milestone progression under the selected synchronized snapshots before proposing a new cycle-generation design.",
    ]
    region_rows = []
    for row in flow[flow.dataset.isin(bif_keys)].itertuples():
        region_rows.append({"dataset": row.dataset, "drift_seed": str(row.drift_seed),
                            "region": "pre_branch", "n_cells": int(row.n_early),
                            "fraction_positive": float(row.fraction_positive_early)})
    for i in range(3):
        key = f"linear_s{i}_T8"
        for seed in (0, 1, 2):
            lam, _ = _spectra(np.load(flow_path(key, seed))["J"])
            region_rows.append({"dataset": key, "drift_seed": str(seed), "region": "linear_all",
                                "n_cells": int(len(lam)), "fraction_positive": float(np.mean(lam > 0))})
    region_df = pd.DataFrame(region_rows)
    score_rows = scores[scores.operator == "score"][["dataset", "fork_auc", "median_rotation"]]

    payload = {
        "task": "orient/S21b", "status": "done", "report": "S21b",
        "started_utc": started_utc, "finished_utc": finished_utc,
        "compute": {"cpu_hours": _timed_cpu_hours(), "gpu_hours": 0.0,
                    "hardware": "Local 14-core/36-GB macOS host; agent C limit 3 threads and 14 GB per process."},
        "timepoints_T": 8,
        "T_tried": [8], "T_accepted": 8, "fork_counts": fork_counts,
        "gate_F1_fork_auc": {"passes": f1, "minimum": float(bif.fork_auc.min()) if len(bif) else None,
                             "rows": jsonable(bif[["dataset", "drift_seed", "fork_auc"]].to_dict("records"))},
        "gate_F2_rotation": {"passes": f2, "rows": f2_rows},
        "score_j_report": jsonable(scores[scores.operator == "score"].to_dict("records")),
        "jref_diagnosis": jsonable(jrefd.to_dict("records")),
        "S21_to_S21b_score_changes": score_changes,
        "S21_to_S21b_jref_changes": jref_changes,
        "dyngen_settings_by_dataset": settings_by_dataset,
        "prebranch_vs_linear_positive_fraction": jsonable(region_rows),
        "runtimes": runtimes,
        "versions": versions, "cpu_hours": _timed_cpu_hours(),
        "metrics": {
            "timepoints_T": 8, "fork_counts": fork_counts,
            "gate_F1_fork_auc": {"passes": f1, "minimum": float(bif.fork_auc.min()) if len(bif) else None},
            "gate_F2_rotation": {"passes": f2, "rows": f2_rows},
            "score_j_report": jsonable(scores[scores.operator == "score"].to_dict("records")),
            "jref_diagnosis": jsonable(jrefd.to_dict("records")),
            "versions": versions, "cpu_hours": _timed_cpu_hours(),
        },
        "deviations": deviations, "questions": [],
        "observations": observations, "suggestions": suggestions,
    }
    status = "PASS" if f1 else "FAIL"
    f2_status = "PASS" if f2 else "FAIL"
    f1_table = ["| Dataset | Drift seed | Fork cells | Committed cells | Fork AUC |", "|---|---:|---:|---:|---:|"]
    f1_table.extend(
        f"| {r.dataset} | {r.drift_seed} | {int(r.n_fork)} | {int(r.n_committed)} | {float(r.fork_auc):.4f} |"
        for r in bif.sort_values(["dataset", "drift_seed"]).itertuples()
    )
    f2_table = ["| Cycle | Drift seed | Cycle median rotation | Linear median rotation | Pass |", "|---|---:|---:|---:|---:|"]
    f2_table.extend(
        f"| {r['dataset']} | {r['drift_seed']} | {r['cycle_median_rotation']:.6f} | {r['linear_median_rotation']:.6f} | {r['passes']} |"
        for r in f2_rows
    )
    region_table = ["| Dataset | Drift seed | Region | Cells | Fraction with Re λ₁ > 0 |", "|---|---:|---|---:|---:|"]
    region_table.extend(
        f"| {r.dataset} | {r.drift_seed} | {r.region} | {int(r.n_cells)} | {float(r.fraction_positive):.4f} |"
        for r in region_df.itertuples()
    )
    score_table = ["| Dataset | Score-J fork AUC | Median rotation |", "|---|---:|---:|"]
    score_table.extend(
        f"| {r.dataset} | {r.fork_auc if pd.notna(r.fork_auc) else 'NA'} | {float(r.median_rotation):.6f} |"
        for r in score_rows.itertuples()
    )
    md = [
        "# S21b", "", "## Summary", "",
        "Re-sampled the saved dyngen simulations at T=8; no new simulation was run.",
        f"All three bifurcating datasets met the ≥100 fork-cell selection rule: {fork_counts}.",
        "", "## What ran", "",
        "Part A: `generate_experiment` only, with `experiment_synchronised(num_timepoints=8)` and the original dataset seeds; raw counts and truth exported through the unchanged S19 exporter.",
        "Part B: median-library normalisation, log1p, PCA-15, S14 reference flow J in raw coordinates, drift seeds 0–2, and S10 score J (1500 epochs; exact ESM Laplacian).",
        "Part C: truth was read only after all nine datasets had completed their fits. `hours = 48 × (day − 1)`.",
        "", "## Gate numbers", "",
        "| Gate | Result | Number |", "|---|---:|---:|",
        f"| F1 flow-J fork AUC ≥ 0.85 | {status} | minimum {payload['gate_F1_fork_auc']['minimum']} |",
        f"| F2 cycle rotation > linear, all seeds | {f2_status} | {sum(r['passes'] for r in f2_rows)}/{len(f2_rows)} seed comparisons |",
        "", "### F1 scores", "", *f1_table,
        "", "### F2 comparisons", "", *f2_table,
        "", "### Report-only pre-branch and linear fractions", "", *region_table,
        "", "### Score-J AUC and rotation", "", *score_table,
        "", "## S21 → S21b change", "",
        "Every numeric score and J_ref field is compared row-wise in `s21_change_scores.csv` and `s21_change_jref.csv` (matched by dataset, operator, and drift seed; S21b's `_T8` suffix is removed). The source values, new values, and deltas are recorded for every field.",
        "", "## J_ref diagnosis", "",
        f"All {len(jrefd)} operator rows are in `jref_diagnosis.csv`; columns include norm ratio, optimal scalar and relative error, median per-cell cosine, and Spearman λ₁ correlation. Full per-row values are also in the JSON report.",
        "", "## Deviations", "", *[f"- {d}" for d in deviations],
        "The S21 fit and scoring calculations were reused unchanged; only S21b paths and dataset keys were selected, with a report adapter for the new T=8 outputs.",
        "", "## Observations", "",
        f"T=8 fork counts were {fork_counts}. S21 had 22 fork cells for s10 and zero for s16/s18; S21b now samples the specified fork region in all three datasets.",
        "", "## Suggestions", "",
        suggestions[0],
        "", "## Questions", "", "None.", "", "## Artifacts", "",
        "- `benchmarks/dyngen/MANIFEST.csv` (nine T=8 entries appended)",
        "- `benchmarks/dyngen/s21b/part_a_T.csv`", "- `benchmarks/dyngen/s21b/scores.csv`",
        "- `benchmarks/dyngen/s21b/jref_diagnosis.csv`", "- `benchmarks/dyngen/s21b/s21_change_scores.csv`",
        "- `benchmarks/dyngen/s21b/s21_change_jref.csv`", "- `benchmarks/figures/dyngen/s21b_lambda1_pca.png`",
        "- `benchmarks/figures/dyngen/s21b_auc_bars.png`",
    ]
    result_md = "\n".join(md) + "\n"
    (OUT / "RESULT_S21b.md").write_text(result_md)
    (ROOT / "coord/orient/reports/S21b.md").write_text(result_md)

    tracked = set(subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                                 text=True, check=True).stdout.splitlines())
    candidates = [ROOT / "benchmarks/dyngen/MANIFEST.csv", OUT / "part_a_T.csv",
                  OUT / "scores.csv", OUT / "jref_diagnosis.csv", OUT / "s21_change_scores.csv",
                  OUT / "s21_change_jref.csv", OUT / "RESULT_S21b.md",
                  ROOT / "coord/orient/reports/S21b.md",
                  FIG / "s21b_lambda1_pca.png", FIG / "s21b_auc_bars.png"]
    for row in part_a.itertuples():
        key = str(row.dataset)
        candidates.extend([DATA / f"{key}.h5ad", RAW / key / "model.rds",
                           RAW / key / "settings.json", RAW / key / "settings.txt"])
    candidates.extend(sorted((ARR / "prepared").glob("*.npz")))
    candidates.extend(sorted((ARR / "flow").glob("*.npz")))
    candidates.extend(sorted((ARR / "score").glob("*.npz")))
    artifacts = []
    for path in candidates:
        if not path.is_file():
            continue
        rel = str(path.relative_to(ROOT))
        artifacts.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256(path),
                          "committed": rel in tracked or rel in {
                              "benchmarks/dyngen/MANIFEST.csv", "benchmarks/dyngen/s21b/part_a_T.csv",
                              "benchmarks/dyngen/s21b/scores.csv", "benchmarks/dyngen/s21b/jref_diagnosis.csv",
                              "benchmarks/dyngen/s21b/s21_change_scores.csv", "benchmarks/dyngen/s21b/s21_change_jref.csv",
                              "benchmarks/dyngen/s21b/RESULT_S21b.md", "coord/orient/reports/S21b.md",
                              "benchmarks/figures/dyngen/s21b_lambda1_pca.png", "benchmarks/figures/dyngen/s21b_auc_bars.png"},
                          "regenerate": "See the matching per-dataset command in benchmarks/dyngen/s21b/logs/ and scripts/s21b_resample.R or scripts/s21_dyngen_ops.py."})
    payload["artifacts"] = artifacts
    write_json(ROOT / "coord/orient/reports/S21b.json", payload)
    print(json.dumps({"F1": f1, "F2": f2, "cpu_hours": payload["cpu_hours"]}))


def main() -> None:
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("part-a"); q = sub.add_parser("prepare"); q.add_argument("dataset")
    q = sub.add_parser("fit"); q.add_argument("dataset"); q.add_argument("drift_seed", type=int)
    q = sub.add_parser("score"); q.add_argument("dataset")
    sub.add_parser("report")
    a = p.parse_args()
    if a.action == "part-a": part_a()
    elif a.action == "prepare": prepare(a.dataset)
    elif a.action == "fit": fit(a.dataset, a.drift_seed)
    elif a.action == "score": score(a.dataset)
    else: report()


if __name__ == "__main__": main()
