#!/usr/bin/env python3
"""orient/S12 Step 0: is OT flow J real across embeddings? (gate for Steps 1-2)

Also the shared library of the S12 scripts (``s12_optimise.py`` and
``s12_test.py`` import it).

Definitions (task S12):

* OT flow J (S10 default): balanced entropic OT (POT ``ot.sinkhorn``,
  eps = 0.05 x median squared distance) between consecutive sampling days in
  the embedding -> barycentric velocity / dt for earlier-day cells ->
  ``MeasuredDrift(d, seed=0)`` defaults -> ``jacobian_autodiff`` on all cells.
* Fork AUC: AUC of the leading real eigenvalue, true saddle vs committed cells
  (S06 Step 4 classes on the core dims, mu = z1 - z1_crit).

Stages (one detached, timed unit each; checkpoints in ``s12/checkpoints``):

* ``embed --preset P --seed N``                 PCA-10, PCA-50 (``embed_linear``,
                                                same preprocessing) and scVI-10.
* ``ot --preset P --seed N --embedding E``      OT flow J in embedding E
                                                (pca20, fa20, pca10, pca50, scvi10,
                                                fa20w = OT cost whitened by the FA
                                                noise variances; velocities and J in raw FA).
* ``diag --preset P --seed N``                  FA diagnosis items (a)-(c).
* ``summary``                                   step0.csv, step0_decision.json.

Only observed layers/metadata enter any fit (spliced counts, day, time_hours,
stored embeddings).  Truth (``z_true``, ``fate_true``, ``jacobian_state``,
``z1_crit``) is read for scoring only, after the operator is fitted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
DATA_V3 = ROOT / "benchmarks/synthetic_multimodal/data/v3"
MANIFEST = ROOT / "benchmarks/synthetic_multimodal/MANIFEST_v3.csv"
S10ARR = ROOT / "benchmarks/synthetic_multimodal/data/s10"       # S10 fits (read only)
S10OUT = ROOT / "benchmarks/synthetic_multimodal/s10"            # S10 tables (read only)
ARR = ROOT / "benchmarks/synthetic_multimodal/data/s12"          # large outputs (gitignored)
OUT = ROOT / "benchmarks/synthetic_multimodal/s12"
CKPT = OUT / "checkpoints"
FIG = ROOT / "benchmarks/figures/orient/s12"
TASK = "orient/S12"
STEP0_PRESETS = ("v3_default", "v3_transition")
STEP0_SEEDS = (0, 1, 2)
EMBEDDINGS = ("pca20", "fa20", "pca10", "pca50", "scvi10", "fa20w")
KNN = 30
EPS_FACTOR = 0.05


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def utc(t: float | None = None) -> str:
    return datetime.fromtimestamp(time.time() if t is None else t, timezone.utc).isoformat().replace("+00:00", "Z")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def cpu_seconds() -> float:
    a = resource.getrusage(resource.RUSAGE_SELF)
    b = resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime)


def rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def checkpoint(name: str, started: float, cpu0: float, payload: dict[str, Any]) -> dict[str, Any]:
    record = {"task": TASK, "unit": name, "started_utc": utc(started), "finished_utc": utc(),
              "wall_seconds": round(time.time() - started, 3), "in_process_cpu_seconds": round(cpu_seconds() - cpu0, 3),
              "peak_rss_bytes": rss_bytes(), **payload}
    write_json(CKPT / f"{name}.json", record)
    return record


def manifest_rows() -> list[dict[str, str]]:
    with MANIFEST.open(newline="") as handle:
        return list(csv.DictReader(handle))


def manifest_row(preset: str, seed: int, verify: bool = True) -> dict[str, str]:
    for row in manifest_rows():
        if row["preset"] == preset and int(row["seed"]) == int(seed):
            if verify:
                path = ROOT / row["path"]
                if not path.is_file() or sha256(path) != row["sha256"]:
                    raise RuntimeError(f"CHECKSUM FAILURE: {path} does not match MANIFEST_v3 sha256")
            return row
    raise RuntimeError(f"{preset} seed {seed} has no MANIFEST_v3 row")


def dataset_path(preset: str, seed: int) -> Path:
    return ROOT / manifest_row(preset, seed, verify=False)["path"]


def unit_dir(preset: str, seed: int) -> Path:
    d = ARR / f"{preset}_s{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def dense(x: Any, dtype: Any = np.float32) -> np.ndarray:
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=dtype)


# ---------------------------------------------------------------------------
# embeddings
# ---------------------------------------------------------------------------
def embedding_file(preset: str, seed: int) -> Path:
    return unit_dir(preset, seed) / "embeddings.npz"


def fit_extra_embeddings(src: Any, n_threads: int = 4, with_scvi: bool = True) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """PCA-10 / PCA-50 by the generator's embed_linear (same preprocessing) and scVI-10.

    Input: observed spliced counts only.
    """
    import anndata as ad
    import pandas as pd
    from scjdo2.data.synthetic_multimodal import embed_linear
    out: dict[str, np.ndarray] = {}
    info: dict[str, Any] = {}
    spliced = dense(src.layers["spliced"], np.float32)
    for n in (10, 50):
        t0 = time.time()
        work = ad.AnnData(X=spliced.copy(), obs=pd.DataFrame(index=src.obs_names.copy()), var=pd.DataFrame(index=src.var_names.copy()))
        work.layers["spliced"] = spliced
        embed_linear(work, "pca", n_comp=n)
        meta = work.uns["embedding_pca"]
        out[f"z_pca{n}"] = np.asarray(work.obsm["X_pca"], np.float32)
        out[f"mean_pca{n}"] = np.asarray(meta["mean"], np.float32)
        out[f"linear_map_pca{n}"] = np.asarray(meta["linear_map"], np.float32)
        out[f"target_pca{n}"] = np.float64(meta["normalization_target"])
        info[f"pca{n}"] = {"seconds": round(time.time() - t0, 2), "n_comp": int(meta["n_comp"])}
    if with_scvi:
        import torch
        import scvi
        torch.set_num_threads(n_threads)
        scvi.settings.seed = 0
        scvi.settings.num_threads = n_threads
        t0 = time.time()
        work = ad.AnnData(X=spliced.copy(), obs=pd.DataFrame(index=src.obs_names.copy()), var=pd.DataFrame(index=src.var_names.copy()))
        work.layers["counts"] = spliced
        scvi.model.SCVI.setup_anndata(work, layer="counts")
        model = scvi.model.SCVI(work, n_latent=10)
        model.train(accelerator="cpu", devices=1, enable_progress_bar=False)
        out["z_scvi10"] = np.asarray(model.get_latent_representation(), np.float32)
        hist = model.history
        info["scvi10"] = {"seconds": round(time.time() - t0, 2), "scvi_version": scvi.__version__, "n_latent": 10,
                          "settings": "SCVI defaults (n_hidden 128, n_layers 1, gene_likelihood zinb), layer = observed spliced counts, "
                                      "all genes, seed 0, default max_epochs, accelerator cpu; latent = posterior mean",
                          "n_epochs": int(len(hist["elbo_train"])) if "elbo_train" in hist else None,
                          "final_elbo_train": float(hist["elbo_train"].iloc[-1, 0]) if "elbo_train" in hist else None}
    return out, info


def load_embedding(preset: str, seed: int, name: str, src: Any | None = None) -> np.ndarray:
    """Coordinates of embedding ``name`` (pca20 / fa20 stored by the generator; others from ``embed``)."""
    if name in ("pca20", "fa20", "fa20w"):
        import anndata as ad
        key = "X_pca" if name == "pca20" else "X_fa"
        if src is None:
            a = ad.read_h5ad(dataset_path(preset, seed), backed="r")
            z = np.asarray(a.obsm[key], np.float32)
            a.file.close()
            return z
        return np.asarray(src.obsm[key], np.float32)
    arr = np.load(embedding_file(preset, seed))
    return np.asarray(arr[f"z_{name}"], np.float32)


def linear_meta(preset: str, seed: int, name: str, src: Any) -> dict[str, Any] | None:
    """Stored linear transform of a linear embedding (for the J_ref truth), None for scVI."""
    if name in ("pca20", "fa20", "fa20w"):
        m = src.uns["embedding_pca" if name == "pca20" else "embedding_fa"]
        return {"mean": np.asarray(m["mean"], np.float64), "linear_map": np.asarray(m["linear_map"], np.float64),
                "normalization_target": float(m["normalization_target"])}
    if name.startswith("pca"):
        arr = np.load(embedding_file(preset, seed))
        return {"mean": np.asarray(arr[f"mean_{name}"], np.float64), "linear_map": np.asarray(arr[f"linear_map_{name}"], np.float64),
                "normalization_target": float(arr[f"target_{name}"])}
    return None


# ---------------------------------------------------------------------------
# OT flow
# ---------------------------------------------------------------------------
def ot_velocity(z: np.ndarray, days: np.ndarray, hours: np.ndarray, eps_factor: float = EPS_FACTOR,
                backward_last: bool = False, cost_coords: np.ndarray | None = None
                ) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Balanced entropic OT between consecutive days; barycentric velocity / dt.

    ``cost_coords``: coordinates in which the squared-Euclidean cost is computed
    (default: ``z``); barycentres are always taken in ``z``.
    ``backward_last``: last-day cells get the backward-OT velocity of the last
    coupling (day d-1 -> d): ``(y - barycentre of its preimage) / dt``.
    Returns v (NaN where undefined) and per-pair info.
    """
    import ot
    import warnings
    c = z if cost_coords is None else cost_coords
    v = np.full(z.shape, np.nan, dtype=np.float64)
    info: list[dict[str, Any]] = []
    uniq = np.unique(days)
    pairs = list(zip(uniq[:-1], uniq[1:]))
    for k, (d0, d1) in enumerate(pairs):
        i0, i1 = np.flatnonzero(days == d0), np.flatnonzero(days == d1)
        x, y = z[i0].astype(np.float64), z[i1].astype(np.float64)
        C = ot.dist(c[i0].astype(np.float64), c[i1].astype(np.float64), metric="sqeuclidean")
        eps = float(eps_factor * np.median(C))
        dt = float(np.median(hours[i1]) - np.median(hours[i0]))
        a = np.full(len(i0), 1.0 / len(i0))
        b = np.full(len(i1), 1.0 / len(i1))
        t0 = time.time()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            P, log = ot.sinkhorn(a, b, C, reg=eps, log=True)
        rows = P.sum(axis=1, keepdims=True)
        v[i0] = ((P @ y) / np.maximum(rows, 1e-300) - x) / dt
        rec = {"from_day": int(d0), "to_day": int(d1), "dt_hours": dt, "n_source": int(len(i0)), "n_target": int(len(i1)),
               "epsilon": eps, "eps_factor": eps_factor, "plan_mass": float(P.sum()), "n_iter": int(len(log.get("err", []))),
               "final_err": float(log["err"][-1]) if len(log.get("err", [])) else None,
               "warnings": [str(w.message)[:200] for w in caught], "rows_zero": int((rows[:, 0] <= 0).sum()),
               "seconds": round(time.time() - t0, 2)}
        if backward_last and k == len(pairs) - 1:
            cols = P.sum(axis=0)[:, None]
            pre = (P.T @ x) / np.maximum(cols, 1e-300)
            v[i1] = (y - pre) / dt
            rec["backward_last_day"] = True
            rec["cols_zero"] = int((cols[:, 0] <= 0).sum())
        info.append(rec)
        del C, P
    return v.astype(np.float32), info


def whitening_coords(z_fa: np.ndarray, meta: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Coordinates whose squared distance is ||Psi^{-1/2} W (z_i - z_j)||^2 (FA noise-variance whitened cost)."""
    W = np.asarray(meta["loadings"], np.float64)
    psi = np.asarray(meta["noise_variance"], np.float64)
    M = W.T @ (W / psi[:, None])
    L = np.linalg.cholesky(M)                # M = L L^T ;  ||L^T dz||^2 = dz^T M dz
    return (np.asarray(z_fa, np.float64) @ L).astype(np.float64), {
        "metric": "M = W^T Psi^-1 W (FA loadings W, FA noise variances Psi); cost = (z_i - z_j)^T M (z_i - z_j)",
        "M_eigenvalues_min_max": [float(np.linalg.eigvalsh(M).min()), float(np.linalg.eigvalsh(M).max())]}


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------
def md_jacobian(z: np.ndarray, v: np.ndarray, fit_rows: np.ndarray, *, hidden: tuple[int, ...] = (96, 96, 96),
                weight_decay: float = 1e-4, seed: int = 0) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """MeasuredDrift fit on (z, v)[fit_rows] (default device), jacobian_autodiff on all cells (CPU)."""
    from scjdo2.measured_drift import MeasuredDrift
    z32 = np.asarray(z, np.float32)
    rows = np.asarray(fit_rows)
    vv = np.asarray(v, np.float32)[rows]
    ok = np.isfinite(vv).all(axis=1)
    t0 = time.time()
    md = MeasuredDrift(z32.shape[1], hidden=tuple(hidden), weight_decay=float(weight_decay), seed=int(seed))
    md.fit(z32[rows][ok], vv[ok], verbose=False)
    device = str(next(md._model.parameters()).device)
    md._model.to("cpu")
    fitted = md.velocity(z32).astype(np.float32)
    J = md.jacobian_autodiff(z32).astype(np.float32)
    return J, fitted, {"fit_device": device, "jacobian_device": "cpu", "n_fit_cells": int(ok.sum()), "hidden": list(hidden),
                       "weight_decay": weight_decay, "seed": seed, "seconds": round(time.time() - t0, 2)}


def local_ridge_jacobian(z: np.ndarray, v: np.ndarray, fit_rows: np.ndarray, k: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Local ridge Jacobian at every cell from its k nearest cells with a velocity.

    Ridge least squares of ``v_n - mean(v) ~ J (z_n - mean(z))`` over the k
    nearest fit cells (neighbourhood-centred; the ridge is
    ``OrientConfig().ridge x trace(A^T A)/d`` as ``scjdo2.orient.base``).  For
    centres that carry a velocity this differs from ``local_jacobian`` only by
    centring on the neighbourhood mean instead of the centre cell, and it also
    defines J at cells without a velocity (the last day).
    """
    from sklearn.neighbors import NearestNeighbors
    from scjdo2.orient import OrientConfig
    ridge = OrientConfig().ridge
    rows = np.asarray(fit_rows)
    zf = np.asarray(z, np.float64)[rows]
    vf = np.asarray(v, np.float64)[rows]
    nn = NearestNeighbors(n_neighbors=k).fit(zf)
    d = z.shape[1]
    J = np.empty((len(z), d, d), dtype=np.float32)
    for lo in range(0, len(z), 2048):
        hi = min(lo + 2048, len(z))
        idx = nn.kneighbors(np.asarray(z[lo:hi], np.float64), return_distance=False)
        A = zf[idx]
        A = A - A.mean(axis=1, keepdims=True)
        B = vf[idx]
        B = B - B.mean(axis=1, keepdims=True)
        AtA = np.einsum("nkd,nke->nde", A, A)
        lam = ridge * np.trace(AtA, axis1=1, axis2=2)[:, None, None] / d
        JT = np.linalg.solve(AtA + lam * np.eye(d)[None], np.einsum("nkd,nke->nde", A, B))
        J[lo:hi] = np.transpose(JT, (0, 2, 1))
    return J, {"k": k, "ridge": ridge, "n_fit_cells": int(len(rows))}


# ---------------------------------------------------------------------------
# truth (scoring only)
# ---------------------------------------------------------------------------
def truth_for_scoring(src: Any) -> dict[str, Any]:
    meta = src.uns["synthetic_multimodal"]
    z1_crit = float(meta["z1_crit"]) if "z1_crit" in meta else 1.0
    return {"z": np.asarray(src.obsm["z_true"], np.float64)[:, :6], "z1_crit": z1_crit,
            "fate": src.obs["fate_true"].astype(str).to_numpy(), "day": src.obs["day"].to_numpy(int)}


def truth_classes(z: np.ndarray, z1_crit: float) -> dict[str, np.ndarray]:
    """S06 Step 4 definitions on the core dims, mu = z1 - z1_crit (as S10)."""
    mu = z[:, 1] - z1_crit
    r = np.sqrt(np.maximum(mu, 0))
    return {"saddle": (mu >= 0.1) & (np.abs(z[:, 0]) < 0.3 * r),
            "committed": (mu >= 0.3) & (np.abs(np.abs(z[:, 0]) - r) < 0.3 * r),
            "progenitor": z[:, 1] < z1_crit - 0.2}


def fate_axis(X: np.ndarray, day: np.ndarray, fate: np.ndarray) -> np.ndarray:
    """S10 fate axis: latest-day positive minus negative centroid, normalised (scoring only)."""
    last = day == day.max()
    ax = X[last & (fate == "positive")].mean(0) - X[last & (fate == "negative")].mean(0)
    return ax / max(np.linalg.norm(ax), 1e-12)


def auc(lam: np.ndarray, classes: dict[str, np.ndarray]) -> float:
    from sklearn.metrics import roc_auc_score
    s, c = classes["saddle"], classes["committed"]
    if not s.any() or not c.any():
        return float("nan")
    return float(roc_auc_score(np.r_[np.ones(s.sum()), np.zeros(c.sum())], np.r_[lam[s], lam[c]]))


def spectra(J: np.ndarray) -> dict[str, np.ndarray]:
    """Leading real eigenvalue, its real unit eigenvector, max |imag| (as S10 s10_operators.spectra)."""
    J = np.asarray(J, dtype=np.float64)
    n = len(J)
    w, V = np.linalg.eig(J)
    lead = np.argmax(w.real, axis=1)
    vec = V[np.arange(n), :, lead].real
    vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12)
    return {"lam1": w.real[np.arange(n), lead], "vec1": vec, "max_imag": np.abs(w.imag).max(axis=1)}


def soft_axis(sp: dict[str, np.ndarray]) -> np.ndarray:
    """S10 Step 4 data-derived soft axis (leading eigenvector averaged over the top-10 % lam1 cells)."""
    lam, vec = sp["lam1"], sp["vec1"]
    top = np.flatnonzero(lam >= np.quantile(lam, 0.9))
    ref = vec[top[np.argmax(lam[top])]]
    aligned = vec[top] * np.sign(vec[top] @ ref)[:, None]
    a = aligned.mean(axis=0)
    return a / max(np.linalg.norm(a), 1e-12)


def tangent_lam1(J: np.ndarray, z: np.ndarray, r: int = 5, k: int = KNN) -> np.ndarray:
    """lam1 of U^T J U, U = top-r eigenvectors of the kNN-k local data covariance of each cell."""
    from sklearn.neighbors import NearestNeighbors
    z = np.asarray(z, np.float64)
    idx = NearestNeighbors(n_neighbors=k).fit(z).kneighbors(z, return_distance=False)
    out = np.empty(len(z))
    for lo in range(0, len(z), 2048):
        hi = min(lo + 2048, len(z))
        A = z[idx[lo:hi]]
        A = A - A.mean(axis=1, keepdims=True)
        C = np.einsum("nkd,nke->nde", A, A) / (k - 1)
        _, V = np.linalg.eigh(C)
        U = V[:, :, -r:]
        R = np.einsum("ndr,nde,nes->nrs", U, np.asarray(J[lo:hi], np.float64), U)
        out[lo:hi] = np.linalg.eigvals(R).real.max(axis=1)
    return out


def fork_scores(J: np.ndarray, z: np.ndarray, truth: dict[str, Any], with_extras: bool = True) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    classes = truth_classes(truth["z"], truth["z1_crit"])
    sp = spectra(J)
    axis = fate_axis(np.asarray(z, np.float64), truth["day"], truth["fate"])
    align = np.abs(sp["vec1"][classes["saddle"]] @ axis)
    res = {"auc_lam1": auc(sp["lam1"], classes), "n_saddle": int(classes["saddle"].sum()),
           "n_committed": int(classes["committed"].sum()), "fraction_saddle": float(classes["saddle"].mean()),
           "fate_axis_abs_cos_at_saddle": float(np.median(align)) if align.size else np.nan,
           "median_lam1_saddle": float(np.median(sp["lam1"][classes["saddle"]])) if classes["saddle"].any() else np.nan,
           "median_lam1_committed": float(np.median(sp["lam1"][classes["committed"]])) if classes["committed"].any() else np.nan}
    arrays = {"lam1": sp["lam1"].astype(np.float32)}
    if with_extras:
        a = soft_axis(sp)
        curv = np.einsum("d,nde,e->n", a, np.asarray(J, np.float64), a)
        tl = tangent_lam1(J, z)
        res.update({"auc_soft_axis_curvature": auc(curv, classes), "auc_tangent5_lam1": auc(tl, classes),
                    "soft_axis_abs_cos_fate_axis": float(abs(a @ axis))})
        arrays.update({"soft_curv": curv.astype(np.float32), "tangent5_lam1": tl.astype(np.float32)})
    return res, arrays


def jref(z: np.ndarray, v_true: np.ndarray) -> np.ndarray:
    """J_ref (scoring only): local_jacobian (k = 30) of the true expected velocity in the embedding."""
    from scjdo2.orient import OrientConfig
    from scjdo2.orient.base import local_jacobian, neighbours
    return local_jacobian(np.asarray(z, np.float64), np.asarray(v_true, np.float64), neighbours(np.asarray(z, np.float64), 30),
                          OrientConfig().ridge).astype(np.float32)


def jref_error(J: np.ndarray, Jr: np.ndarray) -> float:
    num = np.linalg.norm(np.asarray(J, np.float64) - Jr, axis=(1, 2))
    den = np.linalg.norm(np.asarray(Jr, np.float64), axis=(1, 2))
    return float(np.median(num / np.maximum(den, 1e-300)))


def true_expected_velocity(src: Any, meta: dict[str, Any]) -> np.ndarray:
    """true_velocity_in_embedding(..., at='expected') with a given linear transform (scoring only)."""
    import anndata as ad
    import pandas as pd
    from scjdo2.data.synthetic_multimodal import true_velocity_in_embedding
    tmp = ad.AnnData(X=np.zeros((src.n_obs, 0), np.float32), obs=pd.DataFrame(
        {"capture_factor": src.obs["capture_factor"].to_numpy(), "batch_day": src.obs["batch_day"].to_numpy()}, index=src.obs_names.copy()))
    tmp = ad.AnnData(X=np.zeros((src.n_obs, src.n_vars), np.float32), obs=tmp.obs)
    tmp.layers["ds_dt_true"] = np.asarray(src.layers["ds_dt_true"])
    tmp.layers["s_true"] = np.asarray(src.layers["s_true"])
    tmp.uns["batch_gene_effects"] = np.asarray(src.uns["batch_gene_effects"])
    tmp.uns["embedding_x"] = {"linear_map": meta["linear_map"], "normalization_target": meta["normalization_target"]}
    tmp.obsm["X_x"] = np.zeros((src.n_obs, meta["linear_map"].shape[1]), np.float32)
    return np.asarray(true_velocity_in_embedding(tmp, "x", at="expected"), np.float32)


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def threads() -> int:
    return int(os.environ.get("OMP_NUM_THREADS", "2"))


def stage_embed(preset: str, seed: int) -> None:
    import anndata as ad
    started, cpu0 = time.time(), cpu_seconds()
    row = manifest_row(preset, seed)
    src = ad.read_h5ad(ROOT / row["path"])
    out, info = fit_extra_embeddings(src, n_threads=threads())
    dest = embedding_file(preset, seed)
    np.savez(dest, **out)
    checkpoint(f"embed_{preset}_s{seed}", started, cpu0, {"input_sha256": row["sha256"], "path": str(dest.relative_to(ROOT)),
                                                          "sha256": sha256(dest), "keys": sorted(out), "details": info})


def stage_ot(preset: str, seed: int, emb: str) -> None:
    import anndata as ad
    import torch
    torch.set_num_threads(threads())
    started, cpu0 = time.time(), cpu_seconds()
    row = manifest_row(preset, seed)
    src = ad.read_h5ad(ROOT / row["path"], backed="r")
    days = src.obs["day"].to_numpy(int)
    hours = src.obs["time_hours"].to_numpy(np.float64)          # observed sampling time
    z = load_embedding(preset, seed, emb, src).astype(np.float64)
    details: dict[str, Any] = {"input_sha256": row["sha256"], "embedding": emb, "dim": int(z.shape[1])}
    cost = None
    if emb == "fa20w":
        cost, winfo = whitening_coords(z, src.uns["embedding_fa"])
        details["whitening"] = winfo
    import ot
    details["ot_library"] = {"name": "POT", "version": ot.__version__}
    v, pinfo = ot_velocity(z, days, hours, cost_coords=cost)
    fit_rows = np.flatnonzero(np.isfinite(v).all(axis=1))
    J, fitted, jinfo = md_jacobian(z, v, fit_rows)
    details.update({"ot_pairs": pinfo, "measured_drift": jinfo})
    # ---- scoring (truth read after the fit)
    truth = truth_for_scoring(src)
    res, arrays = fork_scores(J, z, truth, with_extras=True)
    vel_scale = float(np.median(np.linalg.norm(v[fit_rows], axis=1)))
    details["median_ot_speed"] = vel_scale
    src.file.close()
    dest = unit_dir(preset, seed) / f"step0_ot_{emb}.npz"
    np.savez(dest, J=J, v_ot=v, v_fit=fitted, **arrays)
    # S10 stored operator (exact reproduction of the S10 number) for pca20 / fa20
    if emb in ("pca20", "fa20"):
        method = emb[:-2]
        s10 = np.load(S10ARR / f"{preset}_s{seed}" / f"flow_{method}.npz")
        classes = truth_classes(truth["z"], truth["z1_crit"])
        res["auc_lam1_s10_stored_operator"] = auc(spectra(s10["J_flow_ot"])["lam1"], classes)
        vs = np.asarray(s10["v_ot"], np.float64)
        both = np.isfinite(vs).all(1) & np.isfinite(v).all(1)
        res["max_abs_diff_ot_velocity_vs_s10"] = float(np.max(np.abs(vs[both] - v[both])))
    checkpoint(f"ot_{preset}_s{seed}_{emb}", started, cpu0, {**details, "scores": res, "path": str(dest.relative_to(ROOT)),
                                                             "sha256": sha256(dest)})


def gene_space(v: np.ndarray, linear_map: np.ndarray) -> np.ndarray:
    """Map score-space vectors to gene space by the linear map's pseudo-inverse: g = v pinv(L)."""
    return np.asarray(v, np.float64) @ np.linalg.pinv(np.asarray(linear_map, np.float64))


def cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-300)


def stage_diag(preset: str, seed: int) -> None:
    """FA diagnosis (a) truth classes and fate axis, (b) OT velocity agreement in gene space."""
    import anndata as ad
    started, cpu0 = time.time(), cpu_seconds()
    row = manifest_row(preset, seed)
    src = ad.read_h5ad(ROOT / row["path"], backed="r")
    truth = truth_for_scoring(src)
    classes = truth_classes(truth["z"], truth["z1_crit"])
    Lp = np.asarray(src.uns["embedding_pca"]["linear_map"], np.float64)
    Lf = np.asarray(src.uns["embedding_fa"]["linear_map"], np.float64)
    zp = np.asarray(src.obsm["X_pca"], np.float64)
    zf = np.asarray(src.obsm["X_fa"], np.float64)
    src.file.close()
    ap = fate_axis(zp, truth["day"], truth["fate"])
    af = fate_axis(zf, truth["day"], truth["fate"])
    gp, gf = gene_space(ap[None], Lp)[0], gene_space(af[None], Lf)[0]
    res: dict[str, Any] = {
        "a_n_saddle": int(classes["saddle"].sum()), "a_n_committed": int(classes["committed"].sum()),
        "a_fraction_saddle": float(classes["saddle"].mean()),
        "a_note": "truth classes are defined on z_true and are identical in every embedding",
        "a_fate_axis_gene_space_abs_cos_pca_vs_fa": float(abs(gp @ gf) / (np.linalg.norm(gp) * np.linalg.norm(gf))),
    }
    # fate-axis separation of the classes in each embedding (projection on the fate axis)
    for name, z, a in (("pca20", zp, ap), ("fa20", zf, af)):
        proj = z @ a
        res[f"a_{name}_fate_projection_median_saddle"] = float(np.median(proj[classes["saddle"]]))
        res[f"a_{name}_fate_projection_abs_median_committed"] = float(np.median(np.abs(proj[classes["committed"]] - np.median(proj[classes["saddle"]]))))
    vp = np.load(unit_dir(preset, seed) / "step0_ot_pca20.npz")["v_ot"].astype(np.float64)
    vf = np.load(unit_dir(preset, seed) / "step0_ot_fa20.npz")["v_ot"].astype(np.float64)
    ok = np.isfinite(vp).all(1) & np.isfinite(vf).all(1)
    c = cos_rows(gene_space(vp[ok], Lp), gene_space(vf[ok], Lf))
    days = truth["day"][ok]
    res.update({"b_n_cells": int(ok.sum()), "b_gene_space_cos_median": float(np.median(c)),
                "b_gene_space_cos_fraction_positive": float(np.mean(c > 0)),
                "b_gene_space_cos_median_by_day": {int(d): float(np.median(c[days == d])) for d in np.unique(days)}})
    sad = classes["saddle"][ok]
    res["b_gene_space_cos_median_saddle"] = float(np.median(c[sad])) if sad.any() else None
    # fitted (MeasuredDrift) velocities at all cells, same comparison
    fp = np.load(unit_dir(preset, seed) / "step0_ot_pca20.npz")["v_fit"].astype(np.float64)
    ff = np.load(unit_dir(preset, seed) / "step0_ot_fa20.npz")["v_fit"].astype(np.float64)
    cf = cos_rows(gene_space(fp, Lp), gene_space(ff, Lf))
    res.update({"b_fitted_gene_space_cos_median_all": float(np.median(cf)),
                "b_fitted_gene_space_cos_median_saddle": float(np.median(cf[classes["saddle"]])) if classes["saddle"].any() else None,
                "b_fitted_gene_space_cos_median_committed": float(np.median(cf[classes["committed"]]))})
    checkpoint(f"diag_{preset}_s{seed}", started, cpu0, {"input_sha256": row["sha256"], "diag": res})


def stage_summary() -> None:
    import pandas as pd
    started, cpu0 = time.time(), cpu_seconds()
    rows = []
    s10ops = pd.read_csv(S10OUT / "operators.csv")
    for preset in STEP0_PRESETS:
        for seed in STEP0_SEEDS:
            for emb in EMBEDDINGS:
                p = CKPT / f"ot_{preset}_s{seed}_{emb}.json"
                if not p.exists():
                    rows.append({"preset": preset, "seed": seed, "embedding": emb, "status": "missing"})
                    continue
                ck = json.loads(p.read_text())
                r = {"preset": preset, "seed": seed, "embedding": emb, "status": "ok", "dim": ck["dim"], **ck["scores"],
                     "median_ot_speed": ck["median_ot_speed"], "eps_pairs": json.dumps([round(x["epsilon"], 6) for x in ck["ot_pairs"]])}
                if emb in ("pca20", "fa20"):
                    s = s10ops[(s10ops.preset == preset) & (s10ops.seed == seed) & (s10ops.method == emb[:-2]) &
                               (s10ops.operator == "J_flow_ot") & (s10ops.truth_class == "saddle")]
                    r["auc_s10_reported"] = float(s.auc_saddle_vs_committed.iloc[0]) if len(s) else np.nan
                rows.append(r)
    frame = pd.DataFrame(rows)
    diag_rows = []
    for preset in STEP0_PRESETS:
        for seed in STEP0_SEEDS:
            p = CKPT / f"diag_{preset}_s{seed}.json"
            if p.exists():
                d = json.loads(p.read_text())["diag"]
                diag_rows.append({"preset": preset, "seed": seed, **{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in d.items()}})
    diag = pd.DataFrame(diag_rows)
    frame.to_csv(OUT / "step0.csv", index=False)
    diag.to_csv(OUT / "step0_fa_diagnosis.csv", index=False)
    # ---- decision (fixed rule)
    others = [e for e in ("fa20", "pca10", "pca50", "scvi10") if e in set(frame.embedding)]
    per_emb = {}
    for e in EMBEDDINGS:
        sub = frame[(frame.embedding == e) & (frame.status == "ok")]
        vals = {f"{r.preset}_s{r.seed}": float(r.auc_lam1) for r in sub.itertuples()}
        per_emb[e] = {"auc_by_dataset": vals, "n": len(vals), "min": min(vals.values()) if vals else None,
                      "every_seed_ge_0_85": bool(len(vals) == 6 and min(vals.values()) >= 0.85)}
    passing = [e for e in others if per_emb[e]["every_seed_ge_0_85"]]
    fa_fix = per_emb["fa20w"]["every_seed_ge_0_85"]
    proceed_a = len(passing) >= 2
    decision = {"rule": "proceed if OT-flow-J fork AUC >= 0.85 on every seed in at least two embeddings other than PCA-20 "
                        "(fa20, pca10, pca50, scvi10), or if (a)-(d) identify the FA inversion as a defect of the FA coordinates "
                        "with a fix that brings FA to >= 0.85",
                "per_embedding": per_emb, "embeddings_other_than_pca20_passing": passing,
                "clause_1_two_embeddings": proceed_a, "clause_2_fa_fix_whitened_cost_ge_0_85_every_seed": fa_fix,
                "proceed": bool(proceed_a or fa_fix)}
    write_json(OUT / "step0_decision.json", decision)
    # ---- figure: fork AUC per embedding x dataset
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ds = [f"{p}_s{s}" for p in STEP0_PRESETS for s in STEP0_SEEDS]
    for j, e in enumerate(EMBEDDINGS):
        vals = [per_emb[e]["auc_by_dataset"].get(d, np.nan) for d in ds]
        ax.plot(np.arange(len(ds)) + (j - 2.5) * 0.06, vals, "o", label=e)
    ax.axhline(0.85, color="k", lw=0.8, ls="--")
    ax.set_xticks(range(len(ds)), [d.replace("v3_", "") for d in ds], rotation=20)
    ax.set_ylabel("fork AUC (lam1, saddle vs committed)")
    ax.set_ylim(0, 1.02)
    ax.legend(ncol=6, fontsize=7, loc="lower center")
    ax.set_title("S12 Step 0: OT flow J fork AUC by embedding")
    fig.tight_layout()
    fig.savefig(FIG / "step0_auc_by_embedding.png", dpi=130)
    plt.close(fig)
    print(json.dumps(jsonable({k: decision[k] for k in ("embeddings_other_than_pca20_passing", "clause_1_two_embeddings",
                                                         "clause_2_fa_fix_whitened_cost_ge_0_85_every_seed", "proceed")}), indent=1))
    checkpoint("step0_summary", started, cpu0, {"proceed": decision["proceed"]})


def stage_ledger() -> None:
    """Whole-process-tree CPU per unit from the /usr/bin/time -l logs -> cpu_ledger.csv."""
    import re
    rows = []
    for log in sorted((OUT / "logs").glob("*.log")):
        text = log.read_text(errors="ignore")
        m = re.search(r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys", text)
        st = re.search(r"^START (\S+)", text, re.M)
        ex = re.search(r"^EXIT (\d+) (\S+)", text, re.M)
        rss = re.search(r"(\d+)\s+maximum resident set size", text)
        rows.append({"log": str(log.relative_to(ROOT)), "start_utc": st.group(1) if st else "", "end_utc": ex.group(2) if ex else "",
                     "exit": int(ex.group(1)) if ex else "", "wall_s": float(m.group(1)) if m else "",
                     "cpu_s": round(float(m.group(2)) + float(m.group(3)), 2) if m else "",
                     "peak_rss_gb": round(int(rss.group(1)) / 1e9, 3) if rss else ""})
    with (OUT / "cpu_ledger.csv").open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    total = sum(r["cpu_s"] for r in rows if r["cpu_s"] != "")
    print(f"{len(rows)} units, {total / 3600:.3f} CPU-h")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("embed", "ot", "diag", "summary", "ledger"))
    parser.add_argument("--preset")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--embedding", choices=EMBEDDINGS)
    args = parser.parse_args()
    if args.stage == "summary":
        stage_summary()
    elif args.stage == "ledger":
        stage_ledger()
    elif args.stage == "embed":
        stage_embed(args.preset, args.seed)
    elif args.stage == "ot":
        stage_ot(args.preset, args.seed, args.embedding)
    elif args.stage == "diag":
        stage_diag(args.preset, args.seed)


if __name__ == "__main__":
    main()
