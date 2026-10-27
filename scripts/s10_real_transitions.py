#!/usr/bin/env python3
"""orient/S10 Step 5 (and Step 6 real data): transition signals on real data, report only.

Real inputs are read in place from the S08/S11 paths (checksums verified by the
imported S11 helpers) and never modified.  S11's embeddings and velocities are
rebuilt with S11's own functions (``scripts/s11_velocities.py``, imported,
outputs redirected into this task's data folder), because S11's arrays are not
on disk.

* LARRY in vitro (days 2-6): S11's 4 000-cells-per-day sample (rng(0)), PCA-30
  (S11) and FA-30 (same transformed HVG matrix); score ``J`` (S06 energy
  configuration) and ``J_flow_ot`` (MeasuredDrift on S11's balanced-OT velocity of
  the day-2 and day-4 cells).
* scNT-seq haematopoiesis: S11's map-aware PCA-30 of total counts and FA-30;
  score ``J`` and labelling ``J_flow`` in the four Step 0b variants.
* fit seeds 0, 1, 2 (energy and MeasuredDrift seeds); Palantir on PCA-30 from
  one root (LARRY: earliest-day cell, rng(1000); scNT-seq: an HSC cell, rng(1000)).
* signals and binning as Step 4; report the detected bin, the cell types in it,
  and agreement across embeddings and fit seeds.
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

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import s10_fit_and_runs as common  # noqa: E402
import time as _time  # noqa: E402
STARTED, CPU0 = _time.time(), common.cpu_seconds()

REAL = common.ARR / "real"
FIT_SEEDS = (0, 1, 2)
EMBEDDINGS = ("pca", "fa")


def s11() -> Any:
    mod = common.load_module("s10_s11_velocities", SCRIPTS / "s11_velocities.py")
    mod.ARRAYS = REAL / "arrays"
    mod.EMBED = mod.ARRAYS / "embeddings"
    mod.VELOCITIES = mod.ARRAYS / "velocities"
    mod.SCVELO_FITS = mod.ARRAYS / "scvelo_fits"
    mod.CHECKPOINTS = REAL / "s11_checkpoints"
    # MAP stays the committed S11 gene map (read only)
    return mod


def fa_map(transformed: np.ndarray, n_comp: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Factor analysis exactly as the generator's embed_linear(method='fa')."""
    from sklearn.decomposition import FactorAnalysis
    model = FactorAnalysis(n_components=n_comp, random_state=0, max_iter=100, tol=1e-4).fit(transformed)
    W = model.components_.T
    psi = np.maximum(model.noise_variance_, 1e-8)
    inv_psi_W = W / psi[:, None]
    post = inv_psi_W @ np.linalg.inv(np.eye(n_comp) + W.T @ inv_psi_W)
    return ((transformed - model.mean_[None, :]) @ post).astype(np.float32), post, model.mean_


def stage_prepare() -> None:
    """Rebuild S11 embeddings/velocities (S11 functions) and add FA-30 + labelling variants."""
    import anndata as ad
    from scipy import sparse
    started, cpu0 = time.time(), common.cpu_seconds()
    m = s11()
    m.prepare_scnt()
    m.prepare_larry()
    m.velocity_larry_ot()
    m.velocity_scnt_labelling()
    out: dict[str, Any] = {}
    # ---- scNT-seq: FA-30 on the same transformed total counts; four labelling variants in both embeddings
    raw, _processed, raw_idx, emb, counts = m._scnt_common()
    target = float(emb["normalization_target"])
    transformed = np.log1p(counts * (target / counts.sum(axis=1))[:, None])
    z_fa, post, fa_mean = fa_map(transformed)
    new = m.dense(raw.layers["new"][:, raw_idx], np.float64)
    pulse = raw.obs["time"].to_numpy(np.float64)
    from scjdo2.labelled_rates import LabelledRateModel
    import orient_end_to_end as s02
    scnt = {"z_pca": emb["z"].astype(np.float32), "z_fa": z_fa, "cell_type": raw.obs["cell_type"].to_numpy(str),
            "cell_id": raw.obs_names.to_numpy(str), "pulse_hours": pulse.astype(np.float32)}
    for emb_name, z, lmap in (("pca", emb["z"].astype(np.float32), emb["linear_map"]), ("fa", z_fa, post)):
        work = ad.AnnData(X=counts.astype(np.float32), obs=pd.DataFrame({"pulse_hours": pulse.astype(np.float32)}, index=raw.obs_names.copy()))
        work.layers["total"] = counts.astype(np.float32)
        work.layers["new"] = new.astype(np.float32)
        raw_gene, det, model = s02.labelled_velocity(work, z)
        g_sh = np.asarray(model.gamma, dtype=np.float64)
        g_pp = {}
        for p in sorted(np.unique(pulse)):
            sel = pulse == p
            g_pp[float(p)] = np.asarray(LabelledRateModel(seed=0).fit(new[sel].astype(np.float32), counts[sel].astype(np.float32), z[sel], pulse[sel].astype(np.float32)).gamma, dtype=np.float64)
        gpp = np.vstack([g_pp[float(p)] for p in pulse])
        gsh = np.broadcast_to(g_sh[None, :], counts.shape)
        sf = counts.sum(axis=1) / np.median(counts.sum(axis=1))
        A = common.knn_matrix(z, common.KNN)
        Ns, Ts = np.asarray(A @ (new / sf[:, None])), np.asarray(A @ (counts / sf[:, None]))
        gv = {"raw_shared": raw_gene, "raw_perpulse": common.labelling_gene_velocity(new, counts, pulse, gpp),
              "smoothed_shared": common.labelling_gene_velocity(Ns, Ts, pulse, gsh) * sf[:, None],
              "smoothed_perpulse": common.labelling_gene_velocity(Ns, Ts, pulse, gpp) * sf[:, None]}
        for v, g in gv.items():
            scnt[f"v_{emb_name}_{v}"] = m._chain_rule_total(counts, g, lmap, target)
        out[f"scnt_{emb_name}_labelling"] = det
    np.savez(REAL / "scnt.npz", **scnt)
    # ---- LARRY: FA-30 on the sampled cells (same HVG transform as S11's PCA); OT velocity in both embeddings
    data = ad.read_h5ad(m.LARRY_RAW)
    lemb = m.load_larry_embedding()
    idx = lemb["sample_index"].astype(int)
    hvg = lemb["hvg_index"].astype(int)
    library = np.asarray(data.X.sum(axis=1)).ravel().astype(np.float64)
    ltarget = float(lemb["normalization_target"])
    Xs = m._transform_larry_rows(data.X[:, hvg][idx], library[idx], ltarget)
    lz_fa, lpost, _ = fa_map(Xs)
    clone = data.obsm["clone_matrix"].tocsr()
    larry = {"z_pca": lemb["z_all"][idx].astype(np.float32), "z_fa": lz_fa, "day": lemb["day"][idx],
             "cell_type": lemb["cell_type"][idx], "cell_id": lemb["cell_id"][idx], "sample_index": idx,
             "z_all_pca": lemb["z_all"].astype(np.float32), "day_all": lemb["day"],
             "clone_indptr": clone.indptr, "clone_indices": clone.indices, "clone_shape": np.asarray(clone.shape)}
    del data
    import ot
    for emb_name in EMBEDDINGS:
        z = larry[f"z_{emb_name}"].astype(np.float64)
        v = np.full_like(z, np.nan)
        pairs = []
        for d0 in (2.0, 4.0):
            i0, i1 = np.flatnonzero(larry["day"] == d0), np.flatnonzero(larry["day"] == d0 + 2.0)
            C = ot.dist(z[i0], z[i1], metric="sqeuclidean")
            eps = float(0.05 * np.median(C))
            P = ot.sinkhorn(np.full(len(i0), 1 / len(i0)), np.full(len(i1), 1 / len(i1)), C, reg=eps)
            v[i0] = ((P @ z[i1]) / np.maximum(P.sum(axis=1, keepdims=True), 1e-300) - z[i0]) / 48.0
            pairs.append({"from_day": d0, "epsilon": eps})
        larry[f"v_ot_{emb_name}"] = v.astype(np.float32)
        out[f"larry_ot_{emb_name}"] = pairs
    s11_ot = np.load(m.VELOCITIES / "larry_ot.npz")
    pos = {c: i for i, c in enumerate(idx)}
    rows = np.array([pos[c] for c in s11_ot["cell_index"]])
    out["larry_ot_pca_matches_s11_function"] = float(np.nanmax(np.abs(larry["v_ot_pca"][rows] - s11_ot["velocity"])))
    np.savez(REAL / "larry.npz", **larry)
    common.checkpoint("real_prepare", started, cpu0, {"details": out})


def stage_fit(dataset: str, emb_name: str, seed: int) -> None:
    from scjdo2.energy_drift import EnergyDriftAdapter, EnergyDriftConfig
    started, cpu0 = time.time(), common.cpu_seconds()
    arr = np.load(REAL / f"{dataset}.npz", allow_pickle=False)
    z = arr[f"z_{emb_name}"].astype(np.float32)
    cfg = EnergyDriftConfig(hidden=96, depth=3, n_epochs=1500, seed=seed, batch_size=256, loss="esm", lr=5e-4, beta=1.0, esm_laplacian="exact")
    energy = EnergyDriftAdapter(z.shape[1], cfg).fit(z, verbose=False)
    sym = energy.symmetry_check(z, seed=0)
    out = {"J_score": np.asarray(energy.jacobian_autodiff(z), np.float32), "X_drift_v": energy.velocity(z).astype(np.float32)}
    info: dict[str, Any] = {"symmetry_max": float(sym["max_asym"])}
    from scjdo2.measured_drift import MeasuredDrift
    keys = ["v_ot"] if dataset == "larry" else [f"v_{v}" for v in common.VARIANTS]
    for key in keys:
        v = arr[f"v_ot_{emb_name}"] if dataset == "larry" else arr[f"v_{emb_name}_{key[2:]}"]
        ok = np.isfinite(v).all(axis=1)
        md = MeasuredDrift(z.shape[1], seed=seed).fit(z[ok], v[ok].astype(np.float32), verbose=False)
        md._model.to("cpu")
        name = "J_flow_ot" if dataset == "larry" else f"J_flow_{key[2:]}"
        out[name] = md.jacobian_autodiff(z).astype(np.float32)
        info[name] = {"n_fit_cells": int(ok.sum())}
    dest = REAL / f"fit_{dataset}_{emb_name}_seed{seed}.npz"
    np.savez(dest, **out)
    common.checkpoint(f"real_fit_{dataset}_{emb_name}_seed{seed}", started, cpu0, {"path": str(dest.relative_to(common.ROOT)), "sha256": common.sha256(dest), **info})


def stage_palantir(dataset: str) -> None:
    import anndata as ad
    import s06_user_runs as s06u
    started, cpu0 = time.time(), common.cpu_seconds()
    arr = np.load(REAL / f"{dataset}.npz", allow_pickle=False)
    names = arr["cell_id"].astype(str)
    if dataset == "larry":
        day = arr["day"]
        obs = pd.DataFrame({"day": day.astype(int)}, index=pd.Index(names))
        pool = names[day == day.min()]
        rule = "default_rng(1000).choice(earliest-day cells)"
    else:
        obs = pd.DataFrame({"day": np.zeros(len(names), int)}, index=pd.Index(names))
        pool = names[arr["cell_type"].astype(str) == "HSC"]
        rule = "default_rng(1000).choice(HSC cells)"
    given = str(np.random.default_rng(1000).choice(pool))
    work = ad.AnnData(X=np.empty((len(names), 0), dtype=np.float32), obs=obs, obsm={"X_pca": arr["z_pca"].astype(np.float64)})
    output, info = s06u.run_palantir(work, given, "root", "default")
    dest = REAL / f"palantir_{dataset}.npz"
    np.savez(dest, names=names, pseudotime=output.obs["palantir_pseudotime"].to_numpy(np.float64))
    common.checkpoint(f"real_palantir_{dataset}", started, cpu0, {"root_rule": rule, **info, "path": str(dest.relative_to(common.ROOT))})


def stage_score() -> None:
    import s10_operators as ops_mod
    import s10_transitions as tr
    from sklearn.neighbors import NearestNeighbors
    rows: list[dict[str, Any]] = []
    for dataset in ("larry", "scnt"):
        arr = np.load(REAL / f"{dataset}.npz", allow_pickle=False)
        pt = np.load(REAL / f"palantir_{dataset}.npz")["pseudotime"]
        b = tr.bins_from(pt)
        ctype = arr["cell_type"].astype(str)
        for emb_name in EMBEDDINGS:
            z = arr[f"z_{emb_name}"].astype(np.float64)
            nbr = NearestNeighbors(n_neighbors=common.KNN).fit(z).kneighbors(z)[1]
            for seed in FIT_SEEDS:
                fit = np.load(REAL / f"fit_{dataset}_{emb_name}_seed{seed}.npz")
                for name in fit.files:
                    if not name.startswith("J_"):
                        continue
                    J = fit[name].astype(np.float64)
                    sp = ops_mod.spectra(J, symmetric=(name == "J_score"))
                    a = tr.soft_axis(sp)
                    sig = {"lam1": sp["lam1"], "soft_curv": np.einsum("d,nde,e->n", a, J, a), "var_soft": (z @ a)[nbr].var(axis=1)}
                    for s, x in sig.items():
                        med = tr.bin_medians(x, b)
                        det = int(np.nanargmax(med))
                        in_bin = pd.Series(ctype[b == det]).value_counts(normalize=True)
                        rows.append({"dataset": dataset, "embedding": emb_name, "fit_seed": seed, "operator": name, "signal": s,
                                     "detected_bin": det, "peak_to_median": tr.peak_to_median(med),
                                     "n_rising_bins_before_detected": tr.rising_run(med, det),
                                     "cell_types_in_detected_bin": json.dumps({k: round(float(v), 3) for k, v in in_bin.items()}),
                                     "top_cell_type": str(in_bin.index[0]),
                                     "bin_medians": json.dumps([round(float(v), 6) for v in med])})
        # baseline: critical-state index on log-normalised counts
        if dataset == "larry":
            m = s11()
            import anndata as ad
            data = ad.read_h5ad(m.LARRY_RAW)          # read only (backed sparse row indexing fails in this anndata/scipy pair)
            idx = arr["sample_index"].astype(int)
            X = data.X[idx]
            Xs = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
            del data
        else:
            m = s11()
            raw, _p, raw_idx, _emb, Xs = m._scnt_common()
        Xs = np.asarray(Xs, dtype=np.float64)
        keep = Xs.sum(axis=0) > 0
        Xs = Xs[:, keep]
        Xl = np.log1p(Xs / np.maximum(Xs.sum(axis=1, keepdims=True), 1e-12) * np.median(Xs.sum(axis=1)))
        csi = tr.critical_state_index(Xl, b)
        det = int(np.nanargmax(csi))
        in_bin = pd.Series(ctype[b == det]).value_counts(normalize=True)
        rows.append({"dataset": dataset, "embedding": "counts", "fit_seed": np.nan, "operator": "baseline", "signal": "critical_state_index",
                     "detected_bin": det, "peak_to_median": tr.peak_to_median(csi), "n_rising_bins_before_detected": tr.rising_run(csi, det),
                     "cell_types_in_detected_bin": json.dumps({k: round(float(v), 3) for k, v in in_bin.items()}),
                     "top_cell_type": str(in_bin.index[0]), "bin_medians": json.dumps([round(float(v), 6) for v in csi])})
        # cell types per bin (context)
        for k in range(tr.N_BINS):
            vc = pd.Series(ctype[b == k]).value_counts(normalize=True)
            rows.append({"dataset": dataset, "embedding": "palantir_bins", "operator": "cell_types_by_bin", "signal": f"bin_{k}",
                         "detected_bin": k, "cell_types_in_detected_bin": json.dumps({kk: round(float(v), 3) for kk, v in vc.items()}),
                         "top_cell_type": str(vc.index[0])})
    frame = pd.DataFrame(rows)
    # agreement of the detected bin across embeddings and fit seeds, per dataset x operator x signal
    sig_rows = frame[frame.operator.str.startswith("J_")]
    agree = sig_rows.groupby(["dataset", "operator", "signal"]).agg(
        detected_bins=("detected_bin", lambda s: json.dumps(sorted(int(x) for x in s))),
        n_distinct=("detected_bin", "nunique"), range_bins=("detected_bin", lambda s: int(s.max() - s.min()))).reset_index()
    frame.to_csv(common.OUT / "real_transitions.csv", index=False)
    agree.to_csv(common.OUT / "real_transitions_agreement.csv", index=False)
    print(agree.to_string())
    common.checkpoint("real_score", STARTED, CPU0, {})


def stage_diffusion_real() -> None:
    """Step 6 real data, report only: LARRY clone dispersion (2) and OT noise (3); scNT score-flow (1)."""
    import s10_diffusion as dif
    from scipy import sparse
    started, cpu0 = time.time(), common.cpu_seconds()
    rows: list[dict[str, Any]] = []
    larry = np.load(REAL / "larry.npz", allow_pickle=False)
    z_all = larry["z_all_pca"].astype(np.float64)
    day_all = larry["day_all"].astype(int)
    C = sparse.csr_matrix((np.ones(len(larry["clone_indices"])), larry["clone_indices"], larry["clone_indptr"]), shape=tuple(larry["clone_shape"]))
    # one clone label per cell (cells with exactly one barcode clone)
    nnz = np.diff(C.indptr)
    clone = np.full(len(z_all), None, dtype=object)
    single = np.flatnonzero(nnz == 1)
    clone[single] = C.indices[C.indptr[single]].astype(str)
    hours = day_all * 24.0
    tab, cell_vals = dif.clone_dispersion(z_all, clone, day_all, hours, None)
    v = cell_vals["uncorrected"]
    rows.append({"preset": "larry", "seed": 0, "method": "pca", "estimator": "clone_dispersion", "variant": "uncorrected",
                 "quantity": "trace/d", "horizon_hours": 48.0, "n_cells": int(np.isfinite(v).sum()), "n_clone_pairs": int(len(tab)),
                 "median_estimate": float(np.nanmedian(v)), "median_over_clone_pairs": float(tab["uncorrected"].median()) if len(tab) else np.nan,
                 "note": "all LARRY cells with exactly one clone barcode; no Sigma_meas on real data"})
    z = larry["z_pca"].astype(np.float64)
    ot_rows = dif.ot_noise(z, larry["day"].astype(int), larry["day"].astype(float) * 24.0)
    rows.append({"preset": "larry", "seed": 0, "method": "pca", "estimator": "ot_noise", "variant": "uncorrected", "quantity": "trace/d",
                 "horizon_hours": 48.0, "median_estimate": float(np.median([o["D_hat"] for o in ot_rows])),
                 "per_pair": json.dumps([{"pair": f"{o['from_day']}->{o['to_day']}", "D_hat": o["D_hat"], "factor_hat": o["factor_hat"]} for o in ot_rows]),
                 "selection": dif.OT_SELECTION, "note": "S11 sample, 4000 cells per day"})
    scnt = np.load(REAL / "scnt.npz", allow_pickle=False)
    fit = np.load(REAL / "fit_scnt_pca_seed0.npz")
    zs = scnt["z_pca"].astype(np.float64)
    for var in common.VARIANTS:
        D1 = dif.score_flow(zs, scnt[f"v_pca_{var}"].astype(np.float64), fit["X_drift_v"].astype(np.float64))
        ct = scnt["cell_type"].astype(str)
        rows.append({"preset": "scnt_hema", "seed": 0, "method": "pca", "estimator": "score_flow", "variant": var, "quantity": "trace/d",
                     "horizon_hours": np.nan, "n_cells": int(len(D1)), "median_estimate": float(np.median(D1)),
                     "fraction_negative": float(np.mean(D1 < 0)),
                     "by_cell_type": json.dumps({c: round(float(np.median(D1[ct == c])), 6) for c in sorted(set(ct))})})
    (common.OUT / "diffusion_real_rows.json").write_text(json.dumps(common.jsonable(rows), indent=1))
    common.checkpoint("real_diffusion", started, cpu0, {"n_rows": len(rows)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("prepare", "fit", "palantir", "score", "diffusion"))
    parser.add_argument("--dataset", choices=("larry", "scnt"))
    parser.add_argument("--embedding", choices=EMBEDDINGS)
    parser.add_argument("--seed", type=int, choices=FIT_SEEDS)
    args = parser.parse_args()
    REAL.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        stage_prepare()
    elif args.stage == "fit":
        stage_fit(args.dataset, args.embedding, args.seed)
    elif args.stage == "palantir":
        stage_palantir(args.dataset)
    elif args.stage == "score":
        stage_score()
    else:
        stage_diffusion_real()


if __name__ == "__main__":
    main()
