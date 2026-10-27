"""S14 addendum (post hoc, review response): LARRY lineage agreement and held-out day.

Re-runs the S14 Part C reference fit on LARRY (PCA-15, raw coordinates) and reports:

1. The pre-registered lineage statistic (Spearman between the drift-integrated
   displacement and each barcoded cell's observed clonal displacement, pooled over
   PCA coordinates) with 1,000 row permutations instead of five, so a p-value exists.
2. Controls that separate "moves like the population" from cell-specific agreement:
   a population mean-shift predictor; the same statistic after centring each PCA
   coordinate per sampling gap; raw OT displacement without a drift; displacement
   magnitudes; and lambda_1 against displacement magnitude.
3. Bootstrap 95 % intervals for the held-out day-6 energy distances.

Everything here is exploratory and was not pre-registered.

    python scripts/s14_lineage_addendum.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import s14_choose_k as core  # noqa: E402
import s14_test as st  # noqa: E402

OUT = core.ROOT / "benchmarks/synthetic_multimodal/s14/addendum_lineage.json"
N_PERM = 1000
N_BOOT = 200


def pooled(a: np.ndarray, b: np.ndarray) -> float:
    return float(spearmanr(a.reshape(-1), b.reshape(-1)).statistic)


def centred(x: np.ndarray, gaps: np.ndarray) -> np.ndarray:
    out = x.copy()
    for g in np.unique(gaps):
        m = gaps == g
        out[m] -= out[m].mean(axis=0)
    return out


def perm_test(pred: np.ndarray, actual: np.ndarray, rng: np.random.Generator) -> dict:
    obs = pooled(pred, actual)
    null = np.array([pooled(pred[rng.permutation(len(pred))], actual) for _ in range(N_PERM)])
    return {"spearman": obs, "null_median": float(np.median(null)), "null_q99": float(np.quantile(null, 0.99)),
            "null_max": float(null.max()), "p_one_sided": float((1 + np.sum(null >= obs)) / (N_PERM + 1))}


def main() -> None:
    t0 = time.time()
    core.set_threads()
    d = core.load_prepared("larry", 15)
    config = st.config_for("reference")
    flow = core.fit_flow(d, config)
    z = np.asarray(d["z"], np.float64)
    days = np.asarray(d["day"], int)
    rows, target, gaps = core._lineage_pairs(d)
    actual = target - z[rows]
    pred = np.empty_like(actual)
    for g in np.unique(gaps):
        k = np.flatnonzero(gaps == g)
        pred[k] = core.integrate_displacement(flow, z[rows[k]], float(g))
    # population mean shift between the two days of each pair
    uniq = np.unique(days)
    mean_shift = np.empty_like(actual)
    for d0, d1 in zip(uniq[:-1], uniq[1:]):
        k = days[rows] == d0
        mean_shift[k] = z[days == d1].mean(axis=0) - z[days == d0].mean(axis=0)
    v_ot = np.asarray(flow["v_ot_raw"], np.float64)[rows] * gaps[:, None]
    lam1 = np.linalg.eigvals(np.asarray(flow["J"], np.float64)).real.max(axis=1)[rows]

    rng = np.random.default_rng(0)
    res = {
        "n_cell_barcode_pairs": int(len(rows)), "n_permutations": N_PERM,
        "drift_pooled": perm_test(pred, actual, rng),
        "mean_shift_pooled": {"spearman": pooled(mean_shift, actual)},
        "ot_displacement_pooled": perm_test(v_ot, actual, rng),
        "drift_centred": perm_test(centred(pred, gaps), centred(actual, gaps), rng),
        "ot_displacement_centred": perm_test(centred(v_ot, gaps), centred(actual, gaps), rng),
        "magnitude_drift_vs_actual": float(spearmanr(np.linalg.norm(pred, axis=1), np.linalg.norm(actual, axis=1)).statistic),
        "magnitude_ot_vs_actual": float(spearmanr(np.linalg.norm(v_ot, axis=1), np.linalg.norm(actual, axis=1)).statistic),
        "lambda1_vs_actual_magnitude": float(spearmanr(lam1, np.linalg.norm(actual, axis=1)).statistic),
        "per_cell_cosine_drift_actual_median": float(np.median(np.sum(pred * actual, 1) /
                                                               (np.linalg.norm(pred, axis=1) * np.linalg.norm(actual, axis=1)))),
    }

    # held-out day: same construction as s14_test.larry_heldout, plus a bootstrap
    hours = np.asarray(d["hours"], float)
    d0, d1, d2 = uniq[:3]
    train = (days == d0) | (days == d1)
    hflow = core.fit_flow({"z": z[train], "day": days[train], "hours": hours[train]}, config)
    first, middle, last = days == d0, days == d1, days == d2
    gap = float(np.median(hours[last]) - np.median(hours[middle]))
    preds = {"flow": st._integrate_raw(hflow, z[middle], gap), "no_movement": z[middle],
             "mean_displacement": z[middle] + (z[middle].mean(0) - z[first].mean(0))}
    inv = np.linalg.pinv(np.asarray(d["linear_map"], np.float64))
    common = np.asarray(d["common_map"], np.float64)
    observed = (z[last] @ inv) @ common
    proj = {m: (p @ inv) @ common for m, p in preds.items()}
    energy = core.s13().energy
    point = {m: float(energy(p, observed)) for m, p in proj.items()}
    boots = {m: [] for m in proj}
    for _ in range(N_BOOT):
        i = rng.integers(0, len(observed), len(observed))
        j = rng.integers(0, middle.sum(), middle.sum())
        for m, p in proj.items():
            boots[m].append(energy(p[j], observed[i]))
    b = {m: np.asarray(v) for m, v in boots.items()}
    res["heldout_energy"] = {
        "point": point, "n_boot": N_BOOT,
        "ci95": {m: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for m, v in b.items()},
        "diff_vs_flow_ci95": {m: [float(np.quantile(b[m] - b["flow"], 0.025)), float(np.quantile(b[m] - b["flow"], 0.975))]
                              for m in ("no_movement", "mean_displacement")},
        "frac_boot_flow_best": float(np.mean((b["flow"] < b["no_movement"]) & (b["flow"] < b["mean_displacement"]))),
        "train_days": [int(d0), int(d1)], "pca_fitted_on": "all three days (S14 prepare step)"}
    res["seconds"] = round(time.time() - t0, 1)
    res["note"] = "post hoc, exploratory; S14 Part C reference configuration, raw PCA-15"
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
