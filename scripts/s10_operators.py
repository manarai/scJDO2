#!/usr/bin/env python3
"""orient/S10 Step 3: operators for forks and basins (S07 Step 1 on v3).

For the score operator ``J`` and the flow operators (labelling ``J_flow`` per
Step 0b variant, ``J_flow_scvelo``; ``J_flow_ot`` added as report-only
columns), per truth class (S06 Step 4 definitions on the core dims, with
``mu = z1 - z1_crit``): median leading real eigenvalue, fraction > 0, AUC
saddle vs committed, |cos| of the leading eigenvector with the fate axis at
saddle cells.  ``v3_stationary_cycle``: median max |imag| of each flow
operator vs the truth.  Truth (``z_true``, ``fate_true``, ``jacobian_state``,
``z1_crit``) is read here for scoring only, after every operator was fitted.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import s10_fit_and_runs as common  # noqa: E402
import time as _time  # noqa: E402
STARTED, CPU0 = _time.time(), common.cpu_seconds()

ROOT = common.ROOT
OUT = common.OUT


def spectra(J: np.ndarray, symmetric: bool = False) -> dict[str, np.ndarray]:
    """Leading real eigenvalue, its (real, unit) eigenvector, and max |imag| per cell."""
    J = np.asarray(J, dtype=np.float64)
    n = len(J)
    if symmetric:
        w, V = np.linalg.eigh(0.5 * (J + np.swapaxes(J, 1, 2)))
        return {"lam1": w[:, -1], "vec1": V[:, :, -1], "max_imag": np.zeros(n)}
    w, V = np.linalg.eig(J)
    lead = np.argmax(w.real, axis=1)
    vec = V[np.arange(n), :, lead].real
    vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12)
    return {"lam1": w.real[np.arange(n), lead], "vec1": vec, "max_imag": np.abs(w.imag).max(axis=1)}


def operators(preset: str, seed: int, method: str) -> dict[str, tuple[np.ndarray, bool]]:
    """All fitted operators of one unit: name -> (J, symmetric)."""
    d = common.ARR / f"{preset}_s{seed}"
    fit = np.load(d / f"fit_{method}.npz")
    ops: dict[str, tuple[np.ndarray, bool]] = {"J_score": (fit["J_score"], True)}
    for v in common.VARIANTS:
        ops[f"J_flow_{v}"] = (fit[f"J_flow_{v}"], False)
    flow_path = d / f"flow_{method}.npz"
    if flow_path.exists():
        flow = np.load(flow_path)
        for key in flow.files:
            if key.startswith("J_flow_"):
                ops[key] = (flow[key], False)
    return ops


def truth_for_scoring(preset: str, seed: int) -> dict[str, Any]:
    import anndata as ad
    src = ad.read_h5ad(common.dataset_path(preset, seed), backed="r")
    meta = src.uns["synthetic_multimodal"]
    z1_crit = float(meta["z1_crit"]) if "z1_crit" in meta else 1.0
    out = {"z": np.asarray(src.obsm["z_true"], dtype=np.float64)[:, :6], "z1_crit": z1_crit,
           "fate": src.obs["fate_true"].astype(str).to_numpy(), "day": src.obs["day"].to_numpy(int),
           "jacobian_state": np.asarray(src.obsm["jacobian_state"], dtype=np.float64),
           "X": {m: np.asarray(src.obsm[f"X_{m}"], dtype=np.float64) for m in common.METHODS}}
    src.file.close()
    return out


def truth_classes(z: np.ndarray, z1_crit: float) -> dict[str, np.ndarray]:
    """S06 Step 4 definitions on the core dims, mu = z1 - z1_crit."""
    mu = z[:, 1] - z1_crit
    r = np.sqrt(np.maximum(mu, 0))
    return {"saddle": (mu >= 0.1) & (np.abs(z[:, 0]) < 0.3 * r),
            "committed": (mu >= 0.3) & (np.abs(np.abs(z[:, 0]) - r) < 0.3 * r),
            "progenitor": z[:, 1] < z1_crit - 0.2}


def fate_axis(X: np.ndarray, day: np.ndarray, fate: np.ndarray) -> np.ndarray:
    last = day == day.max()
    ax = X[last & (fate == "positive")].mean(0) - X[last & (fate == "negative")].mean(0)
    return ax / max(np.linalg.norm(ax), 1e-12)


def unit_rows(preset: str, seed: int, method: str, truth: dict[str, Any]) -> list[dict[str, Any]]:
    classes = truth_classes(truth["z"], truth["z1_crit"])
    axis = fate_axis(truth["X"][method], truth["day"], truth["fate"])
    rows = []
    true_spec = spectra(truth["jacobian_state"])
    true_auc = auc(true_spec["lam1"], classes)
    for name, (J, sym) in operators(preset, seed, method).items():
        sp = spectra(J, symmetric=sym)
        a = auc(sp["lam1"], classes)
        align = np.abs(sp["vec1"][classes["saddle"]] @ axis)
        for label, mask in classes.items():
            rows.append({"preset": preset, "seed": seed, "method": method, "operator": name, "truth_class": label,
                         "n": int(mask.sum()),
                         "median_lead": float(np.median(sp["lam1"][mask])) if mask.any() else np.nan,
                         "fraction_lead_positive": float(np.mean(sp["lam1"][mask] > 0)) if mask.any() else np.nan,
                         "auc_saddle_vs_committed": a,
                         "median_abs_cosine_fate_axis_at_saddle": float(np.median(align)) if align.size else np.nan,
                         "median_max_imag": float(np.median(sp["max_imag"][mask])) if mask.any() else np.nan,
                         "truth_median_lead": float(np.median(true_spec["lam1"][mask])) if mask.any() else np.nan,
                         "truth_auc_saddle_vs_committed": true_auc})
    return rows


def auc(lam: np.ndarray, classes: dict[str, np.ndarray]) -> float:
    s, c = classes["saddle"], classes["committed"]
    if not s.any() or not c.any():
        return float("nan")
    return float(roc_auc_score(np.r_[np.ones(s.sum()), np.zeros(c.sum())], np.r_[lam[s], lam[c]]))


def cycle_rows(seed: int, method: str, truth: dict[str, Any]) -> list[dict[str, Any]]:
    true_imag = spectra(truth["jacobian_state"])["max_imag"]
    rows = []
    for name, (J, sym) in operators("v3_stationary_cycle", seed, method).items():
        sp = spectra(J, symmetric=sym)
        rows.append({"preset": "v3_stationary_cycle", "seed": seed, "method": method, "operator": name, "truth_class": "all",
                     "n": int(len(J)), "median_lead": float(np.median(sp["lam1"])),
                     "fraction_lead_positive": float(np.mean(sp["lam1"] > 0)),
                     "median_max_imag": float(np.median(sp["max_imag"])),
                     "truth_median_max_imag": float(np.median(true_imag)),
                     "truth_median_lead": float(np.median(spectra(truth["jacobian_state"])["lam1"]))})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets", nargs="*", default=list(common.PRESETS))
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    for preset in args.presets:
        for seed in range(3):
            truth = truth_for_scoring(preset, seed)
            for method in common.METHODS:
                if preset == "v3_stationary_cycle":
                    rows += cycle_rows(seed, method, truth)
                else:
                    rows += unit_rows(preset, seed, method, truth)
            print(f"[S10 operators] {preset} s{seed}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "operators.csv", index=False)
    print(frame.shape)
    common.checkpoint("operators", STARTED, CPU0, {})


if __name__ == "__main__":
    main()
