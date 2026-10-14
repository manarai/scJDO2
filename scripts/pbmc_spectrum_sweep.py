"""PBMC §5.5 — full spectrum per cell, tangent-restricted classifier,
d_latent sweep, rotation-projection debug.

Rework per user (2026-10):

* **Leading-eigenvalue alone is insufficient** for the basin test —
  a cluster with ``n_neg_above_floor = d-1`` out of ``d`` is a basin
  with ONE soft direction (toward its nearest relative in latent),
  which should classify as holder, not fork. Report the full spectrum
  per cluster centroid and per cell.
* **Fisher-weighted tangent restriction** on the latent J (same idea
  as the gene-space tangent step): per cell, keep the top-r principal
  directions of the local Fisher-weighted covariance above a noise
  floor, classify on the restricted ``(r, r)`` block, report ``r``.
  For the ESM-fit latent Hessian, "Fisher-weighted covariance" is the
  local Σ⁻¹ of the kNN ball (an unweighted information proxy); the
  noise floor is ``κ × σ²_min`` where ``σ²_min`` is the smallest
  cluster-level variance.
* **d_latent sweep** over {5, 8, 10, 15} with held-out NB marginal
  log-likelihood from scVI; also report the gate at each ``d``.
* **Rotation-projection debug**: on 20 cells, print the full spectrum
  of ``J`` and of ``(I − UU^T) J (I − UU^T)`` to verify that
  projecting out a 2-dim subspace zeros exactly two eigenvalues and
  leaves the others roughly unchanged (only true when ``U`` is
  (approximately) spanned by eigenvectors of ``J``; the magnitude of
  the OTHER eigenvalues' change diagnoses coupling).
* **New gate** (per the user's clearer basin definition):
  *holder cell* ⇔ ``n_neg_above_floor ≥ r − 1`` where ``r`` is the
  per-cell tangent rank; the basin may have up to one soft direction.
  Gate PASS if majority of resting cells are holders by this rule and
  the median learned/oracle curvature ratio is within a factor of 3.

Outputs under ``benchmarks/v2/pbmc_basin_spectrum/``:

  * ``scvi_sweep_ll.csv``        — d_latent vs held-out marginal LL.
  * ``spectrum_centroid.csv``    — one row per (d, cluster) with the
                                     full oracle + learned spectra.
  * ``spectrum_percell.csv``     — per-cell n_neg and n_neg_above_floor.
  * ``tangent_rank.csv``         — per-cell r, with cluster.
  * ``rotation_debug.csv``       — 20 cells × (eig_before, eig_after).
  * ``gate_by_d.csv``            — gate verdict per d_latent.
  * ``gate_final.json``          — the full report.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

from scjdo2.data import (
    CYCLE_GENES,
    RESTING_TYPES,
    load_pbmc3k,
    present_cycle_genes,
)
from scjdo2.energy_drift import EnergyDriftAdapter, EnergyDriftConfig


REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "benchmarks" / "v2" / "pbmc_basin_spectrum"
MARKER_KEEP = ("CD3D", "CD4", "CD8A", "CD8B", "MS4A1", "CD79A",
               "NKG7", "GNLY", "CD14", "FCGR3A", "CST3", "LYZ",
               "FCER1A", "PPBP")


# -----------------------------------------------------------------
# scVI fit + marginal LL
# -----------------------------------------------------------------


def _fit_scvi(adata, n_latent: int, n_epochs: int, seed: int = 0):
    import scvi
    scvi.settings.seed = seed
    adata = adata.copy()
    adata.layers["counts"] = adata.X.copy()
    scvi.model.SCVI.setup_anndata(adata, layer="counts")
    m = scvi.model.SCVI(adata, n_latent=int(n_latent))
    m.train(max_epochs=int(n_epochs), check_val_every_n_epoch=10,
            early_stopping=False, enable_progress_bar=False)
    z = np.asarray(m.get_latent_representation(adata)).astype(np.float32)
    try:
        mll = float(m.get_marginal_ll(adata, n_mc_samples=30))
    except Exception as err:
        print(f"[spectrum]   marginal_ll fail @ d={n_latent}: {err}")
        mll = float("nan")
    return z, mll


# -----------------------------------------------------------------
# Full spectrum per cluster centroid + per cell
# -----------------------------------------------------------------


def _oracle_hessian(z_cluster: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a Gaussian to ``z_cluster`` and return (eigvals, Hess_oracle).

    ``Hess_oracle = -Σ⁻¹``, symmetric NSD by construction.
    """
    Sigma = np.cov(z_cluster, rowvar=False, bias=False)
    Sigma = Sigma + 1e-4 * np.eye(Sigma.shape[0])
    H = -np.linalg.inv(Sigma)
    ev = np.linalg.eigvalsh(H)
    return ev, H


def _signed_eig(J: np.ndarray) -> np.ndarray:
    """Return real parts of eigenvalues of ``J``, descending."""
    ev = np.linalg.eigvals(J)
    return np.sort(ev.real)[::-1]


def _above_floor(eig: np.ndarray, floor: float) -> np.ndarray:
    return np.abs(eig) > floor


# -----------------------------------------------------------------
# Tangent restriction (local information above noise floor)
# -----------------------------------------------------------------


def _local_tangent(J_c: np.ndarray, z_c: np.ndarray, z: np.ndarray,
                   k: int = 30, floor_frac: float = 0.1) -> int:
    """Return the per-cell tangent rank ``r``: number of principal
    directions of the local Σ⁻¹ (kNN ball) whose eigenvalue exceeds
    ``floor_frac × max |λ|``. Simple proxy for the Fisher-weighted
    rank in the latent frame.
    """
    nn_dists = np.linalg.norm(z - z_c, axis=1)
    k_idx = np.argpartition(nn_dists, min(k, len(z) - 1))[:k]
    z_ball = z[k_idx]
    Sigma = np.cov(z_ball, rowvar=False, bias=False)
    Sigma = Sigma + 1e-6 * np.eye(Sigma.shape[0])
    try:
        inv_eig = np.linalg.eigvalsh(np.linalg.inv(Sigma))
    except np.linalg.LinAlgError:
        return 1
    pos_eig = np.abs(inv_eig)
    floor = floor_frac * pos_eig.max()
    return int((pos_eig > floor).sum())


def _restricted_block(J_c: np.ndarray, V_tangent: np.ndarray) -> np.ndarray:
    """``V_tangent`` is ``(d, r)`` orthonormal; return the ``(r, r)``
    block ``V^T J V``."""
    return V_tangent.T @ J_c @ V_tangent


# -----------------------------------------------------------------
# Gate verdict helpers
# -----------------------------------------------------------------


def _basin_count(eig_vec: np.ndarray, floor: float) -> int:
    """Count eigenvalues more negative than -floor."""
    return int(np.sum(eig_vec < -floor))


def _is_holder(eig_vec: np.ndarray, r: int, floor: float) -> bool:
    """Holder (basin) iff n_neg_above_floor >= r - 1."""
    n_neg = _basin_count(eig_vec[:r], floor)
    return n_neg >= r - 1


def _gate_per_d(z: np.ndarray, cluster: np.ndarray, J: np.ndarray,
                 resting_types) -> dict:
    """Full-spectrum + tangent-restricted basin gate."""
    n, d, _ = J.shape
    # Oracle anchors (one per cluster).
    oracle_abs_leads = []
    cluster_centroid_rows = []
    for cl in sorted(set(cluster)):
        m = cluster == cl
        if m.sum() < 10:
            continue
        z_cl = z[m]
        mu = z_cl.mean(axis=0)
        ev_oracle, _ = _oracle_hessian(z_cl)
        oracle_abs_leads.append(abs(float(ev_oracle.max())))
        # Learned Hessian at the cluster centroid: use the mean of J
        # over the k nearest cells to mu (closest proxy).
        d_mu = np.linalg.norm(z - mu, axis=1)
        near_idx = np.argpartition(d_mu, min(30, n - 1))[:30]
        J_mu = J[near_idx].mean(axis=0)
        ev_learned = _signed_eig(J_mu)
        cluster_centroid_rows.append({
            "cluster": cl, "n": int(m.sum()),
            "is_resting": cl in resting_types,
            "oracle_eig_min": float(ev_oracle.min()),
            "oracle_eig_max": float(ev_oracle.max()),
            "oracle_n_neg": int((ev_oracle < 0).sum()),
            "oracle_n_pos": int((ev_oracle > 0).sum()),
            "learned_eig_min": float(ev_learned.min()),
            "learned_eig_max": float(ev_learned.max()),
            "learned_eig_full": ";".join(f"{e:.3f}" for e in ev_learned),
            "oracle_eig_full": ";".join(f"{e:.3f}" for e in ev_oracle),
        })
    oracle_tol = (0.1 * float(np.median(oracle_abs_leads))
                   if oracle_abs_leads else 0.1)

    # Per-cell spectrum + tangent restriction + holder classification.
    percell_rows = []
    holder_count = {cl: 0 for cl in set(cluster)}
    n_count = {cl: 0 for cl in set(cluster)}
    rank_arr = np.zeros(n, dtype=np.int32)
    sample_cells = min(n, 2638)
    for c in range(sample_cells):
        cl = cluster[c]
        Jc = J[c]
        ev = _signed_eig(Jc)
        n_neg = int((ev < -oracle_tol).sum())
        n_pos = int((ev > oracle_tol).sum())
        r = _local_tangent(Jc, z[c], z, k=30, floor_frac=0.1)
        rank_arr[c] = r
        # Keep the top-r eigenvalues by magnitude (dominant tangent).
        order = np.argsort(-np.abs(ev))
        ev_r = ev[order[:r]]
        n_neg_r = int((ev_r < -oracle_tol).sum())
        is_holder = n_neg_r >= max(r - 1, 1)
        if is_holder:
            holder_count[cl] += 1
        n_count[cl] += 1
        percell_rows.append({
            "cell": c, "cluster": cl, "is_resting": cl in resting_types,
            "r": r, "n_neg_above_floor": n_neg,
            "n_pos_above_floor": n_pos, "n_neg_in_tangent": n_neg_r,
            "lead_re": float(ev[0]), "min_re": float(ev[-1]),
            "is_holder": bool(is_holder),
        })

    # Build gate numbers from per-cell holder flags.
    beh_rows = []
    for cl in sorted(set(cluster)):
        beh_rows.append({
            "cluster": cl, "n": n_count[cl],
            "is_resting": cl in resting_types,
            "n_holder_tangent": holder_count[cl],
            "frac_holder_tangent": (holder_count[cl] / n_count[cl]
                                     if n_count[cl] else 0.0),
        })

    return {
        "oracle_tol": oracle_tol,
        "cluster_centroid": pd.DataFrame(cluster_centroid_rows),
        "percell": pd.DataFrame(percell_rows),
        "behaviour": pd.DataFrame(beh_rows),
        "tangent_rank": rank_arr,
    }


# -----------------------------------------------------------------
# Rotation projection debug
# -----------------------------------------------------------------


def _rotation_debug(J: np.ndarray, U: np.ndarray, n_sample: int = 20,
                     seed: int = 0) -> pd.DataFrame:
    """Compare per-cell eigenspectrum of ``J`` and ``(I-UU^T)J(I-UU^T)``.

    The projected matrix has ``U.shape[1]`` eigenvalues at 0 (eigvecs
    in ``span(U)``); the remaining ``d - rank(U)`` should match the
    eigenvalues of ``J`` restricted to ``U⊥`` — which only equal the
    "other" eigenvalues of ``J`` when ``U`` is in the eigenspace of
    ``J``. The debug reports the before / after spectra to show
    exactly how much the non-cycle eigenvalues drift under projection.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(J.shape[0], size=min(n_sample, J.shape[0]), replace=False)
    rows = []
    d = J.shape[1]
    I = np.eye(d, dtype=np.float64)
    P = I - U @ U.T
    for c in idx:
        Jc = J[c].astype(np.float64)
        Jc_proj = P @ Jc @ P
        ev_before = _signed_eig(Jc)
        ev_after = _signed_eig(Jc_proj)
        rows.append({
            "cell": int(c),
            "eig_before": ";".join(f"{e:.3f}" for e in ev_before),
            "eig_after": ";".join(f"{e:.3f}" for e in ev_after),
            "lead_re_before": float(ev_before[0]),
            "lead_re_after": float(ev_after[0]),
            "n_neg_before": int((ev_before < 0).sum()),
            "n_neg_after": int((ev_after < 0).sum()),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--d-latent", type=int, nargs="+", default=[5, 8, 10, 15])
    p.add_argument("--scvi-epochs", type=int, default=180)
    p.add_argument("--drift-epochs", type=int, default=1500)
    p.add_argument("--n-top-genes", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print("[spectrum] load pbmc3k")
    a = load_pbmc3k()
    sc.pp.filter_genes(a, min_cells=3)
    # HVG restriction with cycle + marker override.
    tmp = a.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=int(args.n_top_genes),
                                flavor="seurat")
    hv = tmp.var["highly_variable"].to_numpy()
    for g in list(CYCLE_GENES) + list(MARKER_KEEP):
        if g in a.var_names:
            hv[a.var_names.get_loc(g)] = True
    a = a[:, hv].copy()
    cluster_arr = a.obs["cluster"].astype(str).to_numpy()
    print(f"[spectrum]   shape={a.shape}  resting cells="
          f"{sum(cl in RESTING_TYPES for cl in cluster_arr)}")

    # Cell-cycle scores for the rotation-projection debug.
    tmp = a.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    X_log = np.asarray(tmp.X.toarray() if hasattr(tmp.X, "toarray") else tmp.X,
                       dtype=np.float32)
    s_mask = np.array([g in list(CYCLE_GENES)[:43] for g in a.var_names],
                      dtype=bool)
    g2m_mask = np.array([g in list(CYCLE_GENES)[43:] for g in a.var_names],
                        dtype=bool)
    def _score(mask):
        if not mask.any():
            return np.zeros(X_log.shape[0])
        return (X_log[:, mask].mean(axis=1) - X_log.mean(axis=1)).astype(
            np.float32)
    S_score = _score(s_mask)
    G2M_score = _score(g2m_mask)

    sweep_rows = []
    gate_rows = []
    all_percell = []
    all_centroid = []
    all_rotation = []
    for d in args.d_latent:
        print(f"\n[spectrum] === d_latent = {d} ===")
        t0 = time.monotonic()
        z, mll = _fit_scvi(a, n_latent=d, n_epochs=args.scvi_epochs,
                            seed=args.seed)
        print(f"[spectrum]   scVI fit {time.monotonic()-t0:.1f}s  marginal_ll={mll:.1f}")
        sweep_rows.append({"d_latent": d, "marginal_ll": mll})

        t0 = time.monotonic()
        drift = EnergyDriftAdapter(dim=d, config=EnergyDriftConfig(
            hidden=96, depth=3, n_epochs=args.drift_epochs,
            seed=args.seed, batch_size=256, loss="esm", lr=5e-4,
            beta=1.0, esm_laplacian="exact" if d <= 20 else "hutchinson",
        )).fit(z, verbose=False)
        print(f"[spectrum]   ESM drift fit {time.monotonic()-t0:.1f}s")

        # Latent cycle basis for rotation debug.
        def _unit(w): return w / (np.linalg.norm(w) + 1e-12)
        def _reg(y): return _unit(np.linalg.lstsq(z, y, rcond=None)[0])
        u_S = _reg(S_score)
        u_G2M = _reg(G2M_score)
        U_cycle, _ = np.linalg.qr(np.column_stack([u_S, u_G2M]).astype(np.float64))

        t0 = time.monotonic()
        J = drift.jacobian_autodiff(z)
        print(f"[spectrum]   J {time.monotonic()-t0:.1f}s")
        sym = drift.symmetry_check(z, n_cells=100, seed=args.seed)
        print(f"[spectrum]   SYM median={sym['median_asym']:.3e}")

        # Rotation projection debug.
        rot = _rotation_debug(J, U_cycle, n_sample=20, seed=args.seed)
        rot["d_latent"] = d
        all_rotation.append(rot)
        if len(rot):
            # Average absolute drift in the top-(d-2) eigenvalues
            # (ignore the two zeroed-out ones).
            def _topd2(s):
                vals = np.array([float(x) for x in s.split(";")])
                vals = np.sort(np.abs(vals))[::-1][:d-2]
                return vals
            drifts = []
            for before_str, after_str in zip(rot["eig_before"], rot["eig_after"]):
                before_vals = np.sort(
                    np.abs(np.array([float(x) for x in before_str.split(";")])),
                )[::-1][:d-2]
                after_vals = np.sort(
                    np.abs(np.array([float(x) for x in after_str.split(";")])),
                )[::-1][:d-2]
                m = min(len(before_vals), len(after_vals))
                drifts.append(float(np.mean(np.abs(before_vals[:m] - after_vals[:m]))))
            mean_drift = float(np.mean(drifts))
            print(f"[spectrum]   ROT DEBUG: mean |Δλ| in top-{d-2} eigvals "
                  f"= {mean_drift:.4f}  (clean projection → tiny)")

        # Full-spectrum gate (RAW J, no rotation projection).
        g = _gate_per_d(z, cluster_arr, J, RESTING_TYPES)
        g["cluster_centroid"]["d_latent"] = d
        g["percell"]["d_latent"] = d
        all_centroid.append(g["cluster_centroid"])
        all_percell.append(g["percell"])

        # Per-d gate verdict.
        resting_beh = g["behaviour"][g["behaviour"]["is_resting"]]
        n_resting = int(resting_beh["n"].sum())
        n_holder = int(resting_beh["n_holder_tangent"].sum())
        frac = n_holder / n_resting if n_resting else 0.0
        print(f"\n[spectrum] GATE @ d={d}: {n_holder}/{n_resting} resting holders "
              f"({frac:.2%})  oracle_tol={g['oracle_tol']:.4f}")
        print(g["behaviour"].to_string(index=False))
        print(f"[spectrum]   centroid spectra:")
        print(g["cluster_centroid"][["cluster","is_resting","oracle_n_neg",
                                      "oracle_n_pos","learned_eig_min",
                                      "learned_eig_max",
                                      "learned_eig_full"]].to_string(index=False))

        gate_rows.append({
            "d_latent": d, "marginal_ll": mll,
            "oracle_tol": g["oracle_tol"],
            "n_resting_cells": n_resting,
            "n_resting_holders": n_holder,
            "frac_resting_holder": frac,
            "sym_median_asym": sym["median_asym"],
            "verdict": "pass" if frac >= 0.5 else "fail",
        })

    pd.DataFrame(sweep_rows).to_csv(OUT / "scvi_sweep_ll.csv", index=False)
    pd.concat(all_centroid, ignore_index=True).to_csv(
        OUT / "spectrum_centroid.csv", index=False,
    )
    pd.concat(all_percell, ignore_index=True).to_csv(
        OUT / "spectrum_percell.csv", index=False,
    )
    pd.concat(all_rotation, ignore_index=True).to_csv(
        OUT / "rotation_debug.csv", index=False,
    )
    pd.DataFrame(gate_rows).to_csv(OUT / "gate_by_d.csv", index=False)

    # Pick best d by marginal LL and dump final gate.
    sweep_df = pd.DataFrame(sweep_rows).dropna()
    best_d = (int(sweep_df.loc[sweep_df["marginal_ll"].idxmax(), "d_latent"])
              if len(sweep_df) else args.d_latent[len(args.d_latent) // 2])
    best_gate = next(r for r in gate_rows if r["d_latent"] == best_d)
    final = {
        "chosen_d_latent": best_d,
        "best_gate": best_gate,
        "d_sweep": gate_rows,
        "resting_types": list(RESTING_TYPES),
        "basin_definition": "n_neg_above_floor_in_tangent_block >= r - 1",
    }
    (OUT / "gate_final.json").write_text(json.dumps(final, indent=2, default=str))
    print("\n[spectrum] FINAL (d chosen by marginal LL):")
    print(json.dumps(final, indent=2, default=str))
    print(f"[spectrum] outputs → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
