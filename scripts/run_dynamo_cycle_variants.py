"""Phase D §7a — three-variant cycle audit on Dynamo hematopoiesis.

Variants:
  A. Unprojected (reference)         — reuse the §7 audit outputs.
  B. Score-regressed                 — residualise velocity_pca on
                                        per-cell [S, G2M] scores, refit
                                        MeasuredDrift, re-run Phase C.
  C. Rotation-projected              — keep A's full Phase C fit;
                                        project the two phase-gradient
                                        directions out of J_latent per
                                        cell BEFORE the eigendecomp,
                                        tag the projected directions
                                        in the ledger with
                                        ``reason=known_rotation``.

Reports side by side:
  - behaviour counts (overall + per annotated fate),
  - holder count on terminal fates,
  - cycle weight in the archetype leading eigenvector,
  - GATA1 / SPI1 ranks per archetype.

Plus the 37-holder deep dive from variant A: cell_type_annotation,
phase bin, G0-like indicator (low S + low G2M + low cycle-gene
expression). Recovery is defensible only if the holder set on
terminal fates survives projection and the cycle weight on the
leading eigvec drops below 5% in variants B/C.

Direction here comes from the labelling pulse chemistry (per V2_PLAN
R17), not from any annotation-based arrow.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scjdo2 import opspace
from scjdo2.data import load_dynamo_hematopoiesis
from scjdo2.measured_drift import MeasuredDrift, MeasuredDriftConfig
from scjdo2.pipeline import fit_rate_and_gene_field


REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "benchmarks" / "v2" / "dynamo_hematopoiesis"
MUST_KEEP = ("GATA1", "SPI1")
TERMINAL_FATES = {"Ery", "Mon", "Neu", "Bas", "Meg"}

# Seurat (Tirosh 2016) S + G2M panels, as in the §7 audit.
SEURAT_S_GENES = {
    "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG",
    "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", "MLF1IP",
    "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76",
    "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51",
    "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM",
    "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8",
}
SEURAT_G2M_GENES = {
    "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A",
    "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF",
    "TACC3", "FAM64A", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB",
    "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1", "KIF20B", "HJURP",
    "CDCA3", "HN1", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1",
    "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR",
    "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF",
    "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA",
}
CYCLE_UNION = SEURAT_S_GENES | SEURAT_G2M_GENES


# -------------------------------------------------------------------- helpers


def _score(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean log1p-expression over the ``mask`` genes, standardised across
    cells. Robust to the HVG filter: when ``mask.sum() == 0`` returns
    zeros."""
    if mask.sum() == 0:
        return np.zeros(X.shape[0], dtype=np.float64)
    s = X[:, mask].mean(axis=1).astype(np.float64)
    s = (s - s.mean()) / (s.std() + 1e-9)
    return s


def _lead_re(J: np.ndarray) -> np.ndarray:
    """Per-cell max Re eigenvalue of each (d, d) slice."""
    n = J.shape[0]
    out = np.empty(n, dtype=np.float64)
    for c in range(n):
        try:
            out[c] = float(np.max(np.linalg.eigvals(J[c]).real))
        except np.linalg.LinAlgError:
            out[c] = np.nan
    return out


def _lead_eigvec_latent(J: np.ndarray) -> np.ndarray:
    """Per-cell leading eigvec (max-Re) of each (d, d) slice. Returns
    complex ``(n, d)``."""
    n, d = J.shape[0], J.shape[1]
    out = np.zeros((n, d), dtype=np.complex128)
    for c in range(n):
        try:
            evs, V = np.linalg.eig(J[c])
            k = int(np.argmax(evs.real))
            out[c] = V[:, k]
        except np.linalg.LinAlgError:
            pass
    return out


def _classify_lead_re(lead_re: np.ndarray, tol: float = 0.0) -> dict:
    r = lead_re[np.isfinite(lead_re)]
    return {
        "n": int(r.size),
        "fork": int((r > tol).sum()),
        "holder": int((r < -tol).sum()),
        "drift": int((np.abs(r) <= tol).sum()),
        "median_lead_re": float(np.median(r)),
        "frac_fork": float((r > tol).sum() / r.size) if r.size else 0.0,
        "frac_holder": float((r < -tol).sum() / r.size) if r.size else 0.0,
    }


def _project_out(J: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Return ``(I - UU^T) J (I - UU^T)`` for a stack ``J[c]`` of
    (d, d) matrices and shared orthonormal ``U`` ∈ R^(d × k).

    Keeps the operator in the orthogonal complement of ``span(U)``.
    """
    I = np.eye(J.shape[1], dtype=np.float64)
    P = I - U @ U.T
    return np.einsum("ij,cjk,kl->cil", P, J.astype(np.float64), P)


def _per_cell_counts(a) -> np.ndarray:
    """Return log1p(X) densified; used by the cell-cycle scorer."""
    X = a.X
    try:
        X = X.toarray()
    except AttributeError:
        pass
    X = np.asarray(X, dtype=np.float64)
    return np.log1p(X)


def _holders_report(a, lead_re_A: np.ndarray) -> pd.DataFrame:
    """Variant-A holder deep dive: cell_type, phase, S/G2M, mean
    cycle-gene expression."""
    mask_holder = (lead_re_A < 0) & np.isfinite(lead_re_A)
    idx = np.where(mask_holder)[0]
    rows = []
    cycle_mask = np.array([g in CYCLE_UNION for g in a.var_names], dtype=bool)
    X_log = _per_cell_counts(a)
    mean_cycle = X_log[:, cycle_mask].mean(axis=1) if cycle_mask.any() else np.zeros(a.n_obs)
    S = a.obs.get("_score_S", pd.Series(np.zeros(a.n_obs))).to_numpy()
    G2M = a.obs.get("_score_G2M", pd.Series(np.zeros(a.n_obs))).to_numpy()
    phase = a.obs.get("_phase", pd.Series(np.zeros(a.n_obs))).to_numpy()
    for c in idx:
        rows.append({
            "cell": int(c),
            "cell_type": str(a.obs["cell_type"].iloc[c]),
            "day_pulse_h": float(a.obs["day"].iloc[c]),
            "lead_re_per_h": float(lead_re_A[c]),
            "S_score": float(S[c]),
            "G2M_score": float(G2M[c]),
            "phase_deg": float(phase[c]),
            "mean_cycle_logexp": float(mean_cycle[c]),
            "is_terminal": str(a.obs["cell_type"].iloc[c]) in TERMINAL_FATES,
            "is_G0like": bool(
                abs(S[c]) < 0.2 and abs(G2M[c]) < 0.2 and mean_cycle[c] < np.median(mean_cycle)
            ),
        })
    return pd.DataFrame(rows)


def _cycle_weight_latent(vecs: np.ndarray, U_cycle: np.ndarray) -> np.ndarray:
    """Return per-cell |U_cycle^T v|² / |v|² — the fraction of the
    leading eigvec's squared magnitude that falls on the cycle
    subspace in the latent frame."""
    v = np.asarray(vecs)
    num = np.sum(np.abs(v @ U_cycle) ** 2, axis=1)
    den = np.sum(np.abs(v) ** 2, axis=1) + 1e-12
    return num / den


def _summary_block(lead_re: np.ndarray, label: str, a) -> dict:
    cls = _classify_lead_re(lead_re)
    cls["variant"] = label
    # holders on terminals.
    holder_mask = (lead_re < 0) & np.isfinite(lead_re)
    terminal_mask = a.obs["cell_type"].isin(TERMINAL_FATES).to_numpy()
    cls["holders_on_terminals"] = int((holder_mask & terminal_mask).sum())
    cls["holders_total"] = int(holder_mask.sum())
    # per fate
    for f in sorted(a.obs["cell_type"].unique()):
        m = a.obs["cell_type"].to_numpy() == f
        hold_f = int((holder_mask & m).sum())
        cls[f"holder_{f}"] = hold_f
        cls[f"n_{f}"] = int(m.sum())
    return cls


# -------------------------------------------------------------------- main


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-pcs", type=int, default=10)
    p.add_argument("--drift-epochs", type=int, default=1500)
    p.add_argument("--rate-epochs", type=int, default=150)
    p.add_argument("--anchor-max-live", type=int, default=200)
    p.add_argument("--n-anchors", type=int, default=200)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--run-B", action="store_true",
                   help="also run variant B (score-regressed drift) with "
                        "a full Phase C refit (~15 min)")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"[§7a] loading Dynamo hemat (d_latent={args.n_pcs})")
    a = load_dynamo_hematopoiesis(n_pcs=args.n_pcs)
    gene_names = np.asarray(a.var_names).astype(str)
    z = np.asarray(a.obsm["X_pca"]).astype(np.float32)
    v = np.asarray(a.obsm["velocity_pca"]).astype(np.float32)

    # ---------- cell-cycle scores + phase + latent directions
    s_mask = np.array([g in SEURAT_S_GENES for g in gene_names], dtype=bool)
    g2m_mask = np.array([g in SEURAT_G2M_GENES for g in gene_names], dtype=bool)
    cycle_mask = s_mask | g2m_mask
    print(f"[§7a] cycle genes in var: S={s_mask.sum()}  G2M={g2m_mask.sum()}  "
          f"union={cycle_mask.sum()} / {len(CYCLE_UNION)}")
    X_log = _per_cell_counts(a)
    S_score = _score(X_log, s_mask)
    G2M_score = _score(X_log, g2m_mask)
    a.obs["_score_S"] = S_score
    a.obs["_score_G2M"] = G2M_score
    phase_deg = (np.degrees(np.arctan2(G2M_score, S_score)) % 360)
    a.obs["_phase"] = phase_deg

    # Latent-frame cycle directions by linear regression of score on z.
    def _unit(w): return w / (np.linalg.norm(w) + 1e-12)
    def _reg(y):
        # z = (n, d), y = (n,). Return the weight vector that best
        # predicts y from z (ordinary least squares).
        return _unit(np.linalg.lstsq(z, y, rcond=None)[0])
    u_S = _reg(S_score)
    u_G2M = _reg(G2M_score)
    U_cycle = np.column_stack([u_S, u_G2M])
    U_cycle, _ = np.linalg.qr(U_cycle)          # orthonormal (d, 2)
    print(f"[§7a] cycle direction angles (acos |u_S·u_G2M|): {np.degrees(np.arccos(abs(float(u_S @ u_G2M)))):.1f}°")

    # ---------- Variant A: unprojected (full pipeline reused)
    print(f"\n[§7a] variant A — drift fit + latent operator (unprojected)")
    t0 = time.monotonic()
    drift_A = MeasuredDrift(dim=z.shape[1], config=MeasuredDriftConfig(
        hidden=(96, 96, 96), n_epochs=args.drift_epochs,
        weight_decay=0.0, seed=0, clock_calibrated=True,
    )).fit(z, v, verbose=False)
    J_A = drift_A.jacobian_autodiff(z)
    lead_A = _lead_re(J_A)
    vecs_A = _lead_eigvec_latent(J_A)
    cycle_w_A = _cycle_weight_latent(vecs_A, U_cycle)
    print(f"[§7a]   A drift+J {time.monotonic()-t0:.1f}s  cycle_w median {np.median(cycle_w_A):.3f}")

    # ---------- Variant C: rotation-projected
    print(f"\n[§7a] variant C — project cycle subspace out of J_latent")
    J_C = _project_out(J_A, U_cycle)
    lead_C = _lead_re(J_C)
    vecs_C = _lead_eigvec_latent(J_C)
    cycle_w_C = _cycle_weight_latent(vecs_C, U_cycle)
    print(f"[§7a]   C cycle_w median {np.median(cycle_w_C):.3f} (should be ~0)")

    # ---------- Variant B: score-regressed (full refit — expensive)
    v_B = v.copy()
    if args.run_B:
        print(f"\n[§7a] variant B — residualise velocity_pca on [S, G2M], refit")
        W = np.column_stack([S_score, G2M_score])
        for d_i in range(v.shape[1]):
            coef, *_ = np.linalg.lstsq(W, v[:, d_i], rcond=None)
            v_B[:, d_i] = v[:, d_i] - W @ coef
        print(f"[§7a]   velocity residual RMS reduction: "
              f"{float(np.linalg.norm(v_B) / np.linalg.norm(v)):.3f}")
        t0 = time.monotonic()
        drift_B = MeasuredDrift(dim=z.shape[1], config=MeasuredDriftConfig(
            hidden=(96, 96, 96), n_epochs=args.drift_epochs,
            weight_decay=0.0, seed=0, clock_calibrated=True,
        )).fit(z, v_B, verbose=False)
        J_B = drift_B.jacobian_autodiff(z)
        lead_B = _lead_re(J_B)
        vecs_B = _lead_eigvec_latent(J_B)
        cycle_w_B = _cycle_weight_latent(vecs_B, U_cycle)
        print(f"[§7a]   B drift+J {time.monotonic()-t0:.1f}s  cycle_w median {np.median(cycle_w_B):.3f}")

    # ---------- Side-by-side table (latent level)
    rows = [
        _summary_block(lead_A, "A unprojected", a),
        _summary_block(lead_C, "C rotation-projected", a),
    ]
    if args.run_B:
        rows.insert(1, _summary_block(lead_B, "B score-regressed", a))
    summary = pd.DataFrame(rows)
    print("\n[§7a] latent-operator behaviour + holders-on-terminals (SxS):")
    cols = ["variant", "n", "fork", "holder", "drift",
            "frac_fork", "frac_holder", "holders_total",
            "holders_on_terminals"] + [f"holder_{f}" for f in sorted(a.obs["cell_type"].unique())]
    print(summary[cols].to_string(index=False))
    summary.to_csv(OUT / "s7a_latent_summary.csv", index=False)

    # Cycle weight distribution.
    cw_rows = [
        {"variant": "A unprojected", "cycle_w_median": float(np.median(cycle_w_A)),
         "cycle_w_q05": float(np.quantile(cycle_w_A, 0.05)),
         "cycle_w_q95": float(np.quantile(cycle_w_A, 0.95))},
        {"variant": "C rotation-projected", "cycle_w_median": float(np.median(cycle_w_C)),
         "cycle_w_q05": float(np.quantile(cycle_w_C, 0.05)),
         "cycle_w_q95": float(np.quantile(cycle_w_C, 0.95))},
    ]
    if args.run_B:
        cw_rows.insert(1, {
            "variant": "B score-regressed", "cycle_w_median": float(np.median(cycle_w_B)),
            "cycle_w_q05": float(np.quantile(cycle_w_B, 0.05)),
            "cycle_w_q95": float(np.quantile(cycle_w_B, 0.95)),
        })
    cw = pd.DataFrame(cw_rows)
    print("\n[§7a] latent-eigvec cycle weight (|U^T v|² / |v|²):")
    print(cw.to_string(index=False))
    cw.to_csv(OUT / "s7a_cycle_weight_latent.csv", index=False)

    # ---------- 37-holder (variant A) deep dive
    print("\n[§7a] variant-A holder deep dive (per-cell latent lead_re < 0):")
    hdf = _holders_report(a, lead_A)
    print(hdf.head(40).to_string(index=False))
    hdf.to_csv(OUT / "s7a_variantA_holders.csv", index=False)
    print(f"[§7a]   holders total: {len(hdf)}  "
          f"on terminals: {int(hdf['is_terminal'].sum())}  "
          f"G0-like: {int(hdf['is_G0like'].sum())}")

    # ---------- Phase C-level GATA1/SPI1 ranks: variant A is the saved
    # §7 audit output; variant C uses the SAME gene-space J (since
    # rotation projection only touches J_latent). Variant B needs a
    # Phase C refit which we only trigger with --run-B.
    if args.run_B:
        print("\n[§7a] variant B — Phase C refit on residual velocity")
        a_B = a.copy()
        a_B.obsm["X_drift_v"] = drift_B.velocity(z)
        a_B.uns.setdefault("scjdo2", {})
        a_B.uns["scjdo2"]["J"] = J_B
        a_B.uns["scjdo2"]["meta"] = {
            "clock_calibrated": True, "use_rep": "X_pca",
            "time_key": None, "jacobian_method": "autodiff",
        }
        a_B.uns["scjdo2"]["field_reason_summary"] = {"ok": 1.0, "autodiff": True}
        a_B.obs["pseudotime"] = a_B.obsm["X_pca"][:, 0].astype(float)
        t0 = time.monotonic()
        fit_rate_and_gene_field(
            a_B, use_rep="X_pca", n_epochs_rate=args.rate_epochs,
            k=args.k, cv_sample_cells=15, seed=0,
            anchor_fit=True, n_anchors=args.n_anchors, anchor_top_k=10,
            anchor_max_live=args.anchor_max_live,
        )
        a_B.uns["scjdo2"]["meta"]["clock_calibrated"] = True
        print(f"[§7a]   B Phase C in {time.monotonic()-t0:.1f}s")
        opspace.compute(a_B)
        arch_B = a_B.uns["scjdo2"].get("archetypes") or {}
        rows_B = []
        for ak, info in arch_B.items():
            for table_name in ("responders", "drivers_left"):
                df = info.get(table_name)
                if df is None: continue
                for g in MUST_KEEP:
                    hit = df.index[df["gene"].astype(str) == g].tolist()
                    if not hit:
                        continue
                    i = int(hit[0])
                    r = df.loc[i].to_dict()
                    rows_B.append({
                        "variant": "B score-regressed",
                        "archetype": int(ak), "table": table_name, "gene": g,
                        "rank": i, "fisher": float(r["fisher"]),
                        "snr": float(r.get("snr", float("nan"))),
                        "reason": str(r["reason"]),
                        "lead_re_per_h": float(info["eigenvalues"][0].real),
                        "behaviour": info["behaviour"],
                    })
        pd.DataFrame(rows_B).to_csv(OUT / "s7a_variantB_gata1_spi1_ranks.csv",
                                     index=False)
        print("\n[§7a] (variant B) GATA1 / SPI1 ranks:")
        print(pd.DataFrame(rows_B).to_string(index=False))

    print(f"\n[§7a] outputs under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
