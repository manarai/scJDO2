#!/usr/bin/env python3
"""orient/S10 Step 4: critical-transition detection without truth, scored against truth.

Per dataset x embedding x operator (``J_score``, ``J_flow_<variant>``,
``J_flow_scvelo``, ``J_flow_ot`` and the growth-weighted OT sensitivities):

* signals per cell: leading real eigenvalue ``lam1``; curvature ``a^T J a``
  along the data-derived soft axis ``a`` (leading eigenvector averaged over
  the 10 % of cells with the largest ``lam1``, sign-aligned to the first of
  them); local kNN-30 variance of ``z . a``.  Baseline: critical-state index
  (Mojtahedi et al. 2016) on the top-200 HVGs (log-normalised spliced counts),
  per bin.
* ordering: 12 equal-count bins of Palantir pseudotime (earliest-day,
  default-mode root r0); per-bin medians; detected transition = argmax bin.
* truth (scoring only): true bin = argmax of the per-bin median
  ``kappa (mu - 3 z0^2)``, ``mu = z1 - z1_crit``; informative iff true bin in 2..9.
* null: ``v3_stationary_cycle`` binned by the random-cell Palantir run;
  peak-to-median ratio = max(bin medians) / median(|bin medians|).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import s10_fit_and_runs as common  # noqa: E402
import time as _time  # noqa: E402
STARTED, CPU0 = _time.time(), common.cpu_seconds()
import s10_operators as ops_mod  # noqa: E402

N_BINS = 12
KAPPA = 0.5
SIGNALS = ("lam1", "soft_curv", "var_soft")


def bins_from(pseudotime: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    rank = np.argsort(np.argsort(pseudotime, kind="stable"), kind="stable")
    return np.minimum(rank * n_bins // len(pseudotime), n_bins - 1)


def bin_medians(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([np.median(x[b == k]) if np.any(b == k) else np.nan for k in range(N_BINS)])


def peak_to_median(med: np.ndarray) -> float:
    return float(np.nanmax(med) / max(np.nanmedian(np.abs(med)), 1e-300))


def rising_run(med: np.ndarray, det: int) -> int:
    """Number of consecutive bin-to-bin increases ending at the detected bin."""
    n = 0
    k = det
    while k > 0 and med[k] > med[k - 1]:
        n += 1
        k -= 1
    return n


def soft_axis(sp: dict[str, np.ndarray]) -> np.ndarray:
    lam, vec = sp["lam1"], sp["vec1"]
    top = np.flatnonzero(lam >= np.quantile(lam, 0.9))
    ref = vec[top[np.argmax(lam[top])]]
    aligned = vec[top] * np.sign(vec[top] @ ref)[:, None]
    a = aligned.mean(axis=0)
    return a / max(np.linalg.norm(a), 1e-12)


def critical_state_index(X: np.ndarray, b: np.ndarray) -> np.ndarray:
    hv = np.argsort(X.var(axis=0))[-200:]
    Xh = X[:, hv]
    out = []
    for k in range(N_BINS):
        rows = np.flatnonzero(b == k)
        if len(rows) < 30:
            out.append(np.nan)
            continue
        A = Xh[rows]
        A = A[:, A.std(axis=0) > 0]
        rows_ok = A.std(axis=1) > 0
        A = A[rows_ok]
        g = np.abs(np.corrcoef(A.T)[np.triu_indices(A.shape[1], 1)]).mean()
        c = np.abs(np.corrcoef(A)[np.triu_indices(A.shape[0], 1)]).mean()
        out.append(g / c)
    return np.asarray(out)


def load_unit_inputs(preset: str, seed: int, mode: str = "default") -> dict[str, Any]:
    import anndata as ad
    src = ad.read_h5ad(common.dataset_path(preset, seed))
    X = common.dense(src.layers["spliced"], np.float64)
    X = np.log1p(X / np.maximum(X.sum(axis=1, keepdims=True), 1e-12) * np.median(X.sum(axis=1)))
    meta = src.uns["synthetic_multimodal"]
    z1_crit = float(meta["z1_crit"]) if "z1_crit" in meta else 1.0
    zt = np.asarray(src.obsm["z_true"], dtype=np.float64)
    pal = np.load(common.ARR / f"{preset}_s{seed}" / "palantir.npz")
    if preset == "v3_stationary_cycle":
        key = "random_default_r0"
    else:
        key = f"day{int(src.obs['day'].min())}_{mode}_r0"
    if not np.array_equal(pal["names"], src.obs_names.to_numpy(str)):
        raise RuntimeError("Palantir cell order differs")
    return {"X_log": X, "pseudotime": pal[key], "pseudotime_key": key,
            "Z": {m: np.asarray(src.obsm[f"X_{m}"], dtype=np.float64) for m in common.METHODS},
            # truth, scoring only
            "truth_fate_eig": KAPPA * ((zt[:, 1] - z1_crit) - 3.0 * zt[:, 0] ** 2), "mu": zt[:, 1] - z1_crit,
            "z1_crit": z1_crit}


def score_signal(name_meta: dict[str, Any], x: np.ndarray, med: np.ndarray, b: np.ndarray,
                 true_bin: int | None) -> dict[str, Any]:
    det = int(np.nanargmax(med))
    row = {**name_meta, "detected_bin": det, "peak_to_median": peak_to_median(med),
           "n_rising_bins_before_detected": rising_run(med, det),
           "bin_medians": json.dumps([None if not np.isfinite(v) else round(float(v), 6) for v in med])}
    if true_bin is None:
        row.update({"true_bin": np.nan, "informative": np.nan, "abs_offset": np.nan, "signed_offset": np.nan,
                    "rho_before_true_bins": np.nan, "rho_before_true_cells": np.nan})
        return row
    row.update({"true_bin": true_bin, "informative": bool(2 <= true_bin <= 9),
                "abs_offset": abs(det - true_bin), "signed_offset": det - true_bin})
    if true_bin >= 2:
        row["rho_before_true_bins"] = float(spearmanr(np.arange(true_bin), med[:true_bin]).statistic)
        cells = b < true_bin
        row["rho_before_true_cells"] = float(spearmanr(b[cells], x[cells]).statistic) if x is not None else np.nan
    else:
        row["rho_before_true_bins"] = np.nan
        row["rho_before_true_cells"] = np.nan
    return row


def run_dataset(preset: str, seed: int, mode: str = "default") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inp = load_unit_inputs(preset, seed, mode)
    b = bins_from(inp["pseudotime"])
    null = preset == "v3_stationary_cycle"
    truth_med = bin_medians(inp["truth_fate_eig"], b)
    mu_med = bin_medians(inp["mu"], b)
    true_bin = None if null else int(np.nanargmax(truth_med))
    rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    for k in range(N_BINS):
        bin_rows.append({"preset": preset, "seed": seed, "ordering": "primary" if mode == "default" else "sensitivity_strict_root", "bin": k, "n": int((b == k).sum()), "truth_fate_eig_median": truth_med[k],
                         "mu_median": mu_med[k], "pseudotime_key": inp["pseudotime_key"]})
    base = {"preset": preset, "seed": seed, "pseudotime_key": inp["pseudotime_key"], "ordering": "primary" if mode == "default" else "sensitivity_strict_root",
            "truth_peak_to_median": peak_to_median(truth_med), "mu_bin_medians": json.dumps([round(float(v), 4) for v in mu_med]),
            "truth_bin_medians": json.dumps([round(float(v), 5) for v in truth_med])}
    csi = critical_state_index(inp["X_log"], b)
    rows.append(score_signal({**base, "method": "counts", "operator": "baseline", "signal": "critical_state_index"}, None, csi, b, true_bin))
    for method in common.METHODS:
        z = inp["Z"][method]
        nbr = NearestNeighbors(n_neighbors=common.KNN).fit(z).kneighbors(z)[1]
        for name, (J, sym) in ops_mod.operators(preset, seed, method).items():
            sp = ops_mod.spectra(J, symmetric=sym)
            a = soft_axis(sp)
            proj = z @ a
            sig = {"lam1": sp["lam1"], "soft_curv": np.einsum("d,nde,e->n", a, np.asarray(J, np.float64), a),
                   "var_soft": proj[nbr].var(axis=1)}
            for s, x in sig.items():
                med = bin_medians(x, b)
                rows.append(score_signal({**base, "method": method, "operator": name, "signal": s}, x, med, b, true_bin))
                for k in range(N_BINS):
                    bin_rows[k][f"{method}:{name}:{s}"] = med[k]
    return rows, bin_rows


def gates(frame: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    frame = frame[frame.ordering == "primary"]
    for op_key, label in (("J_flow_raw_shared", "gate_critical_transition"), ("J_flow_ot", "gate_critical_transition_ot"),
                          ("J_flow_smoothed_shared", "primary_variant_critical_transition")):
        sel = frame[(frame.preset == "v3_transition") & (frame.method == "pca") & (frame.operator == op_key) & (frame.signal == "lam1")].sort_values("seed")
        null = frame[(frame.preset == "v3_stationary_cycle") & (frame.method == "pca") & (frame.operator == op_key) & (frame.signal == "lam1")].sort_values("seed")
        per_seed = []
        for seed in range(3):
            r = sel[sel.seed == seed]
            n = null[null.seed == seed]
            if r.empty:
                per_seed.append({"seed": seed, "status": "missing"})
                continue
            r = r.iloc[0]
            null_ratio = float(n.peak_to_median.iloc[0]) if len(n) else None
            per_seed.append({"seed": seed, "true_bin": int(r.true_bin), "detected_bin": int(r.detected_bin),
                             "signed_offset": int(r.signed_offset), "abs_offset": int(r.abs_offset),
                             "informative": bool(r.informative), "peak_to_median_transition": float(r.peak_to_median),
                             "peak_to_median_null": null_ratio,
                             "null_below_transition": None if null_ratio is None else bool(null_ratio < float(r.peak_to_median)),
                             "n_rising_bins_before_detected": int(r.n_rising_bins_before_detected),
                             "rho_before_true_bins": r.rho_before_true_bins})
        ok = [p for p in per_seed if "true_bin" in p]
        all_inf = len(ok) == 3 and all(p["informative"] for p in ok)
        offs = len(ok) == 3 and all(p["abs_offset"] <= 1 for p in ok)
        null_ok = len(ok) == 3 and all(p["null_below_transition"] is True for p in ok)
        null_eval = len(ok) == 3 and all(p["null_below_transition"] is not None for p in ok)
        status = ("fail_untestable" if not all_inf else
                  "not_evaluable_null_missing" if not null_eval else
                  "pass" if (offs and null_ok) else "fail")
        out[label] = {"operator": op_key, "signal": "lam1", "embedding": "pca", "per_seed": per_seed,
                      "every_seed_informative": all_inf, "abs_offset_le_1_every_seed": offs,
                      "null_below_transition_every_seed": null_ok if null_eval else None, "status": status,
                      "pass": status == "pass"}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets", nargs="*", default=list(common.PRESETS))
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    bins: list[dict[str, Any]] = []
    for preset in args.presets:
        for seed in range(3):
            for mode in (("default",) if preset == "v3_stationary_cycle" else ("default", "strict")):
                r, b = run_dataset(preset, seed, mode)
                rows += r
                bins += b
            print(f"[S10 transitions] {preset} s{seed}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(common.OUT / "transitions.csv", index=False)
    pd.DataFrame(bins).to_csv(common.OUT / "transition_bins.csv", index=False)
    g = gates(frame)
    common.write_json(common.OUT / "transition_gates.json", g)
    print(json.dumps(common.jsonable({k: v["status"] for k, v in g.items()}), indent=1))
    common.checkpoint("transitions", STARTED, CPU0, {})


if __name__ == "__main__":
    main()
