#!/usr/bin/env python3
"""orient/S10 Step 6: diffusion in the embedding — truth, measurement noise, three estimators.

Stages (one detached, timed unit each):

* ``truth --preset P --seed N --chunk k``  scoring-only truth: per-cell covariance of
  expected-count embedding positions after 6, 24, 48 h (``forward_displacement_covariance``,
  32 paths, all latent dims), cells split into ``N_CHUNKS`` chunks (seed = chunk index) to
  respect the 60-min unit limit.
* ``meas --preset P --seed N``  measurement-noise reference: counts redrawn from the true
  pools (Gamma/Poisson only, noise seed 1), embedded with the stored transform;
  ``Sigma_meas = d d^T / 2`` per cell, ``d`` = stored position - redrawn position.
* ``score``  estimators (no truth) and their comparison with the truth.
* ``real``   report-only estimators on LARRY (2, 3) and scNT-seq (1).

Estimators (no truth used):
1. score-flow: per cell, kNN-30 least squares ``f ~ D s`` (``f`` labelling velocity per
   Step 0b variant, ``s = X_drift_v / beta``, ``beta = 1``), all dims and along the soft axis.
2. clone dispersion: per observed clone with >= 3 cells at both of two consecutive days,
   ``(Cov_t2 - Cov_t1) / (2 dt)``; reported uncorrected, literally minus ``Sigma_meas`` (task
   text) and minus ``(Sigma_meas_t2 - Sigma_meas_t1) / (2 dt)``; assigned to the clone's cells.
3. OT noise level: balanced entropic OT between consecutive days for
   ``eps in {0.25, 0.5, 1, 2, 4} x eps0`` (``eps0`` = Step 1 value); the eps is chosen by a
   held-out (20 % of the later day, ``rng(0)``) Sinkhorn divergence — executor's construction,
   see ``OT_SELECTION`` — and ``D = eps / (4 dt)``; uncorrected and minus ``trace(Sigma_meas)/d``.
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
from sklearn.neighbors import NearestNeighbors

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import s10_fit_and_runs as common  # noqa: E402

TRUTH_PRESETS = ("v3_default",)
HORIZONS = (6.0, 24.0, 48.0)
N_PATHS = 32
N_CHUNKS = 12           # ~1 060 cells per unit (each unit < 60 min wall; 4 chunks exceeded it)
EPS_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
OT_SELECTION = ("for each eps: balanced Sinkhorn plan between the earlier day and the 80 % training part of the later "
                "day; prediction of the later day = barycentric image of each earlier cell plus Gaussian noise with "
                "the Brownian-kernel variance eps/2 per dim (rng(0)); criterion = Sinkhorn divergence (reg = eps0; "
                "transport-cost form <P, C>) between the prediction and the 20 % held-out later cells; argmin over the grid")


def diff_dir(preset: str, seed: int) -> Path:
    d = common.unit_dir(preset, seed) / "diffusion"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# truth and measurement-noise reference (scoring only)
# ---------------------------------------------------------------------------
def stage_truth(preset: str, seed: int, chunk: int) -> None:
    import anndata as ad
    from scjdo2.data.synthetic_multimodal import forward_displacement_covariance
    started, cpu0 = time.time(), common.cpu_seconds()
    row = common.manifest_row(preset, seed)
    src = ad.read_h5ad(common.ROOT / row["path"])
    cells = np.array_split(np.arange(src.n_obs), N_CHUNKS)[chunk]
    cov = forward_displacement_covariance(src, common.METHODS, HORIZONS, n_paths=N_PATHS, seed=chunk, cells=cells)
    dest = diff_dir(preset, seed) / f"truth_chunk{chunk}.npz"
    np.savez(dest, cells=cells, **{f"cov_{m}_{h:g}h": v for (m, h), v in cov.items()})
    common.checkpoint(f"dtruth_{preset}_s{seed}_c{chunk}", started, cpu0, {
        "input_sha256": row["sha256"], "path": str(dest.relative_to(common.ROOT)), "sha256": common.sha256(dest),
        "n_cells": int(len(cells)), "n_paths": N_PATHS, "seed": chunk, "horizons": HORIZONS})


def stage_meas(preset: str, seed: int) -> None:
    import anndata as ad
    from scjdo2.data.synthetic_multimodal import embed_with_stored_transform, resample_spliced_counts
    started, cpu0 = time.time(), common.cpu_seconds()
    row = common.manifest_row(preset, seed)
    src = ad.read_h5ad(common.ROOT / row["path"])
    redraw = resample_spliced_counts(src, noise_seed=1)
    out = {}
    for m in common.METHODS:
        x0 = np.asarray(src.obsm[f"X_{m}"], dtype=np.float64)
        recon = embed_with_stored_transform(src, src.layers["spliced"], m).astype(np.float64)
        x1 = embed_with_stored_transform(src, redraw, m).astype(np.float64)
        d = x0 - x1
        out[f"sigma_meas_{m}"] = (np.einsum("ni,nj->nij", d, d) / 2.0).astype(np.float32)
        out[f"recon_max_abs_{m}"] = np.float64(np.max(np.abs(recon - x0)))
    dest = diff_dir(preset, seed) / "sigma_meas.npz"
    np.savez(dest, **out)
    common.checkpoint(f"dmeas_{preset}_s{seed}", started, cpu0, {
        "input_sha256": row["sha256"], "path": str(dest.relative_to(common.ROOT)), "sha256": common.sha256(dest),
        "noise_seed": 1, "stored_transform_reconstruction_max_abs": {m: float(out[f"recon_max_abs_{m}"]) for m in common.METHODS}})


def load_truth(preset: str, seed: int, method: str, n: int) -> dict[float, np.ndarray]:
    d = diff_dir(preset, seed)
    out = {}
    for h in HORIZONS:
        cov = np.full((n, 20, 20), np.nan, dtype=np.float32)
        for k in range(N_CHUNKS):
            z = np.load(d / f"truth_chunk{k}.npz")
            cov[z["cells"]] = z[f"cov_{method}_{h:g}h"]
        out[h] = cov / (2.0 * h)           # D_eff(dt) = Cov / (2 dt)
    return out


# ---------------------------------------------------------------------------
# estimators (observed data only)
# ---------------------------------------------------------------------------
def score_flow(z: np.ndarray, f: np.ndarray, s: np.ndarray, axis: np.ndarray | None = None, k: int = common.KNN) -> np.ndarray:
    nbr = NearestNeighbors(n_neighbors=k).fit(z).kneighbors(z)[1]
    if axis is None:
        num = np.einsum("nkd,nkd->n", f[nbr], s[nbr])
        den = np.einsum("nkd,nkd->n", s[nbr], s[nbr])
    else:
        fa, sa = f @ axis, s @ axis
        num = (fa[nbr] * sa[nbr]).sum(axis=1)
        den = (sa[nbr] ** 2).sum(axis=1)
    return num / np.maximum(den, 1e-300)


def clone_dispersion(z: np.ndarray, clone: np.ndarray, day: np.ndarray, hours: np.ndarray,
                     sigma_meas: np.ndarray | None) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    d = z.shape[1]
    rows = []
    per_cell: dict[str, list[list[float]]] = {k: [[] for _ in range(len(z))] for k in ("uncorrected", "literal", "differenced")}
    days = np.unique(day)
    valid = pd.notna(clone)
    for d0, d1 in zip(days[:-1], days[1:]):
        dt = float(np.median(hours[day == d1]) - np.median(hours[day == d0]))
        for c in pd.unique(clone[valid]):
            i0 = np.flatnonzero((clone == c) & (day == d0))
            i1 = np.flatnonzero((clone == c) & (day == d1))
            if len(i0) < 3 or len(i1) < 3:
                continue
            raw = (np.cov(z[i1].T) - np.cov(z[i0].T)) / (2.0 * dt)
            vals = {"uncorrected": np.trace(raw) / d}
            if sigma_meas is not None:
                both = np.r_[i0, i1]
                vals["literal"] = np.trace(raw - sigma_meas[both].mean(axis=0)) / d
                vals["differenced"] = np.trace(raw - (sigma_meas[i1].mean(axis=0) - sigma_meas[i0].mean(axis=0)) / (2.0 * dt)) / d
            rows.append({"clone": str(c), "from_day": int(d0), "to_day": int(d1), "dt_hours": dt, "n_t1": len(i0), "n_t2": len(i1), **vals})
            for key, v in vals.items():
                for i in np.r_[i0, i1]:
                    per_cell[key][i].append(v)
    cell_vals = {k: np.array([np.mean(v) if v else np.nan for v in lists]) for k, lists in per_cell.items()}
    return pd.DataFrame(rows), cell_vals


def _plan(a: np.ndarray, b: np.ndarray, C: np.ndarray, reg: float) -> tuple[np.ndarray, str]:
    """POT Sinkhorn (default method); log-domain only if the default returns non-finite values."""
    import ot
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        P = ot.sinkhorn(a, b, C, reg=reg)
    if np.isfinite(P).all() and P.sum() > 0:
        return P, "sinkhorn"
    return ot.sinkhorn(a, b, C, reg=reg, method="sinkhorn_log"), "sinkhorn_log"


def _sinkhorn_divergence(X: np.ndarray, Y: np.ndarray, reg: float) -> float:
    import ot
    a, b = np.full(len(X), 1 / len(X)), np.full(len(Y), 1 / len(Y))
    out = []
    for p, q, A, B in ((a, b, X, Y), (a, a, X, X), (b, b, Y, Y)):
        C = ot.dist(A, B)
        P, _ = _plan(p, q, C, reg)
        out.append(float((P * C).sum()))
    return out[0] - 0.5 * out[1] - 0.5 * out[2]


def ot_noise(z: np.ndarray, day: np.ndarray, hours: np.ndarray) -> list[dict[str, Any]]:
    import ot
    out = []
    days = np.unique(day)
    for d0, d1 in zip(days[:-1], days[1:]):
        i0, i1 = np.flatnonzero(day == d0), np.flatnonzero(day == d1)
        dt = float(np.median(hours[i1]) - np.median(hours[i0]))
        rng = np.random.default_rng(0)
        test = np.zeros(len(i1), dtype=bool)
        test[rng.choice(len(i1), size=int(round(0.2 * len(i1))), replace=False)] = True
        X, Ytr, Yte = z[i0].astype(np.float64), z[i1[~test]].astype(np.float64), z[i1[test]].astype(np.float64)
        C = ot.dist(X, Ytr)
        eps0 = float(0.05 * np.median(ot.dist(X, z[i1].astype(np.float64))))     # Step 1 value (full later day)
        a, b = np.full(len(X), 1 / len(X)), np.full(len(Ytr), 1 / len(Ytr))
        grid = []
        for factor in EPS_GRID:
            eps = factor * eps0
            P, solver = _plan(a, b, C, eps)
            bary = (P @ Ytr) / np.maximum(P.sum(axis=1, keepdims=True), 1e-300)
            noise_rng = np.random.default_rng(0)
            pred = bary + noise_rng.normal(scale=np.sqrt(eps / 2.0), size=bary.shape)
            grid.append({"factor": factor, "eps": eps, "solver": solver, "divergence": _sinkhorn_divergence(pred, Yte, eps0)})
        best = min(grid, key=lambda g: g["divergence"])
        out.append({"from_day": int(d0), "to_day": int(d1), "dt_hours": dt, "eps0": eps0, "grid": grid,
                    "eps_hat": best["eps"], "factor_hat": best["factor"], "D_hat": best["eps"] / (4.0 * dt),
                    "n_source": int(len(i0)), "n_train": int((~test).sum()), "n_test": int(test.sum())})
    return out


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def unit_scores(preset: str, seed: int, method: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import anndata as ad
    import s10_operators as ops_mod
    import s10_transitions as tr
    src = ad.read_h5ad(common.dataset_path(preset, seed), backed="r")
    z = np.asarray(src.obsm[f"X_{method}"], dtype=np.float64)
    day = src.obs["day"].to_numpy(int)
    hours = src.obs["time_hours"].to_numpy(np.float64)
    clone = src.obs["clone"].astype(object).to_numpy()
    zt = np.asarray(src.obsm["z_true"], dtype=np.float64)[:, :6]                        # truth, scoring only
    meta = src.uns["synthetic_multimodal"]
    z1_crit = float(meta["z1_crit"]) if "z1_crit" in meta else 1.0
    src.file.close()
    n, d = z.shape
    fit = np.load(common.ARR / f"{preset}_s{seed}" / f"fit_{method}.npz")
    s = fit["X_drift_v"].astype(np.float64) / 1.0                                         # beta = 1
    J_score = fit["J_score"].astype(np.float64)
    J_flow = fit["J_flow_raw_shared"].astype(np.float64)
    axis = tr.soft_axis(ops_mod.spectra(J_flow))                                          # Step 4 soft axis (J_flow, raw)
    D_true = load_truth(preset, seed, method, n)
    sig = np.load(diff_dir(preset, seed) / "sigma_meas.npz")[f"sigma_meas_{method}"].astype(np.float64)
    tr_true = {h: np.trace(D_true[h], axis1=1, axis2=2) / d for h in HORIZONS}
    ax_true = {h: np.einsum("i,nij,j->n", axis, D_true[h], axis) for h in HORIZONS}
    meas_level = float(np.median(np.trace(sig, axis1=1, axis2=2) / d))
    classes = ops_mod.truth_classes(zt, z1_crit)
    rows: list[dict[str, Any]] = []
    base = {"preset": preset, "seed": seed, "method": method}

    def compare(est_name: str, variant: str, values: np.ndarray, truth: np.ndarray, horizon: float, quantity: str, extra: dict[str, Any]):
        ok = np.isfinite(values) & np.isfinite(truth)
        r = {**base, "estimator": est_name, "variant": variant, "quantity": quantity, "horizon_hours": horizon,
             "n_cells": int(ok.sum()), "median_estimate": float(np.median(values[ok])) if ok.any() else np.nan,
             "median_truth_matched": float(np.median(truth[ok])) if ok.any() else np.nan,
             "median_truth_all": float(np.nanmedian(truth)),
             "spearman_all": float(spearmanr(values[ok], truth[ok]).statistic) if ok.sum() > 2 else np.nan, **extra}
        r["ratio_to_truth_all"] = r["median_estimate"] / r["median_truth_all"]
        r["ratio_to_truth_matched"] = r["median_estimate"] / r["median_truth_matched"] if ok.any() else np.nan
        for label, mask in classes.items():
            m = ok & mask
            r[f"n_{label}"] = int(m.sum())
            r[f"median_estimate_{label}"] = float(np.median(values[m])) if m.any() else np.nan
            r[f"median_truth_{label}"] = float(np.median(truth[m])) if m.any() else np.nan
            r[f"spearman_{label}"] = float(spearmanr(values[m], truth[m]).statistic) if m.sum() > 2 else np.nan
        rows.append(r)

    per_cell_D1 = {}
    for v in common.VARIANTS:
        f = fit[f"v_lab_{v}"].astype(np.float64)
        D1 = score_flow(z, f, s)
        D1a = score_flow(z, f, s, axis)
        per_cell_D1[v] = D1
        compare("score_flow", v, D1, tr_true[6.0], 6.0, "trace/d", {})
        compare("score_flow", v, D1a, ax_true[6.0], 6.0, "axis", {})
    tab, cell_vals = clone_dispersion(z, clone, day, hours, sig)
    for key, vals in cell_vals.items():
        compare("clone_dispersion", key, vals, tr_true[48.0], 48.0, "trace/d",
                {"n_clone_pairs": int(len(tab)), "median_over_clone_pairs": float(tab[key].median()) if len(tab) else np.nan})
    ot_rows = ot_noise(z, day, hours)
    for key, sub in (("uncorrected", 0.0), ("minus_trace_sigma_meas_over_d", meas_level)):
        D3 = np.array([o["D_hat"] - sub for o in ot_rows])
        tvals = [float(np.median(tr_true[48.0][day == o["from_day"]])) for o in ot_rows]
        r = {**base, "estimator": "ot_noise", "variant": key, "quantity": "trace/d", "horizon_hours": 48.0,
             "median_estimate": float(np.median(D3)), "median_truth_all": float(np.nanmedian(tr_true[48.0])),
             "per_pair": json.dumps([{"pair": f"{o['from_day']}->{o['to_day']}", "D_hat": float(x), "truth_median_source_day": t,
                                       "factor_hat": o["factor_hat"]} for o, x, t in zip(ot_rows, D3, tvals)]),
             "spearman_over_day_pairs": float(spearmanr(D3, tvals).statistic) if len(D3) > 1 else np.nan,
             "selection": OT_SELECTION}
        r["ratio_to_truth_all"] = r["median_estimate"] / r["median_truth_all"]
        rows.append(r)
    # payoff: D-hat x J_score vs J_flow
    lam_flow = ops_mod.spectra(J_flow)["lam1"]
    nf = np.linalg.norm(J_flow, axis=(1, 2))
    payoff: dict[str, Any] = {}

    def pay(label: str, scale: np.ndarray | float):
        Js = J_score * (np.asarray(scale)[:, None, None] if np.ndim(scale) else scale)
        rel = np.linalg.norm(Js - J_flow, axis=(1, 2)) / np.maximum(nf, 1e-300)
        lam = ops_mod.spectra(Js, symmetric=True)["lam1"]
        ok = np.isfinite(rel) & np.isfinite(lam)
        payoff[label] = {"median_rel_frobenius": float(np.median(rel[ok])),
                         "spearman_lambda1": float(spearmanr(lam[ok], lam_flow[ok]).statistic)}
    pay("unscaled", 1.0)
    pay("score_flow_per_cell_raw_shared", per_cell_D1["raw_shared"])
    pay("score_flow_median_raw_shared", float(np.median(per_cell_D1["raw_shared"])))
    cd = [r for r in rows if r["estimator"] == "clone_dispersion" and r["variant"] == "literal"]
    if cd and np.isfinite(cd[0]["median_estimate"]):
        pay("clone_dispersion_median_literal", float(cd[0]["median_estimate"]))
    cu = [r for r in rows if r["estimator"] == "clone_dispersion" and r["variant"] == "uncorrected"]
    if cu and np.isfinite(cu[0]["median_estimate"]):
        pay("clone_dispersion_median_uncorrected", float(cu[0]["median_estimate"]))
    ou = [r for r in rows if r["estimator"] == "ot_noise" and r["variant"] == "uncorrected"]
    pay("ot_noise_median_uncorrected", float(ou[0]["median_estimate"]))
    extras = {"payoff": payoff, "sigma_meas_median_trace_over_d": meas_level, "ot_pairs": ot_rows,
              "truth_median_trace_over_d": {f"{h:g}h": float(np.nanmedian(tr_true[h])) for h in HORIZONS},
              "n_clone_pairs": int(len(tab))}
    return rows, extras


def gate(frame: pd.DataFrame) -> dict[str, Any]:
    sel = frame[(frame.preset == "v3_default") & (frame.method == "pca")]
    per_seed = []
    for seed in range(3):
        s = sel[sel.seed == seed]
        cd = s[(s.estimator == "clone_dispersion") & (s.variant == "literal")]
        ot_ = s[(s.estimator == "ot_noise") & (s.variant == "minus_trace_sigma_meas_over_d")]
        sf = s[(s.estimator == "score_flow") & (s.variant == common.GATE_VARIANT) & (s.quantity == "trace/d")]
        if cd.empty or ot_.empty or sf.empty:
            per_seed.append({"seed": seed, "status": "missing"})
            continue
        rc, ro, rho = float(cd.ratio_to_truth_all.iloc[0]), float(ot_.ratio_to_truth_all.iloc[0]), float(sf.spearman_committed.iloc[0])
        per_seed.append({"seed": seed, "clone_ratio": rc, "ot_ratio": ro, "score_flow_spearman_committed": rho,
                         "clone_within_2": bool(0.5 <= rc <= 2.0), "ot_within_2": bool(0.5 <= ro <= 2.0), "score_flow_ge_0_5": bool(rho >= 0.5)})
    ok = [p for p in per_seed if "clone_ratio" in p]
    passed = len(ok) == 3 and all(p["clone_within_2"] and p["ot_within_2"] and p["score_flow_ge_0_5"] for p in ok)
    return {"definition": "v3_default, PCA, every seed: clone (literal Sigma_meas correction) and OT (minus trace(Sigma_meas)/d) "
                          "median estimates within x2 of the all-cell median truth trace(D_eff(48 h))/d; score-flow (raw labelling) "
                          "Spearman >= 0.5 with trace(D_eff(6 h))/d over committed cells",
            "ot_selection_is_executor_construction": True, "per_seed": per_seed, "pass": passed}


def stage_score() -> None:
    started, cpu0 = time.time(), common.cpu_seconds()
    rows: list[dict[str, Any]] = []
    extras: dict[str, Any] = {}
    for preset in TRUTH_PRESETS:
        for seed in range(3):
            for method in common.METHODS:
                r, e = unit_scores(preset, seed, method)
                rows += r
                extras[f"{preset}_s{seed}_{method}"] = e
                print(f"[S10 diffusion] {preset} s{seed} {method}", flush=True)
    frame = pd.DataFrame(rows)
    real_path = common.OUT / "diffusion_real_rows.json"
    if real_path.exists():
        frame = pd.concat([frame, pd.DataFrame(json.loads(real_path.read_text()))], ignore_index=True)
    frame.to_csv(common.OUT / "diffusion.csv", index=False)
    g = gate(frame)
    common.write_json(common.OUT / "diffusion_summary.json", {"gate_diffusion": g, "units": extras})
    common.checkpoint("diffusion_score", started, cpu0, {"gate_pass": g["pass"]})
    print(json.dumps(common.jsonable(g), indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("truth", "meas", "score", "real"))
    parser.add_argument("--preset", choices=common.PRESETS)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--chunk", type=int, choices=range(N_CHUNKS))
    args = parser.parse_args()
    if args.stage == "truth":
        stage_truth(args.preset, args.seed, args.chunk)
    elif args.stage == "meas":
        stage_meas(args.preset, args.seed)
    elif args.stage == "score":
        stage_score()
    else:
        import s10_real_transitions as real
        real.stage_diffusion_real()


if __name__ == "__main__":
    main()
