"""Phase D §7b — pulse-duration-aware clock scale + per-batch splits.

(1) Per-cell clock scale ``t_c`` from the dynamo batch metadata (``obs['day']``,
    which holds the labelling pulse duration in hours: 3 h for JL12_0 +
    JL12_1 batches, 5 h for JL_10). The dynamo processed artefact fit
    ``velocity_pca`` assuming a single pulse duration (``uns['dynamics']['t']``
    is a constant 5 for every cell), so 3-h batch cells carry a per-hour
    velocity that is under-scaled by ``t_c / t_ref`` relative to the 5-h
    cells. We correct by **velocity → velocity * (t_ref / t_c)**, which
    is the clock-scale needed to make synthesis + degradation estimates
    consistent in hours across batches. The scale is recorded as
    ``obs['clock_scale']``.

(2) Run variants A (unprojected) and C (rotation-projected) with the
    clock-scaled velocity on all cells.

(3) Split by batch (3-h-only and 5-h-only), refit per batch, re-run
    variants A and C per batch. Report whether holders appear in both
    batches, one, or neither.

(4) For every holder cell found above, dump ``total_UMI`` and
    ``n_genes_detected`` against the dataset median. If holders are
    the low-count tail, say so.

Comparison with §7a (``s7a_latent_summary.csv``): §7a's full-cell
variant-A fit on dynamo's as-stored velocity reports 37 holders, all
3-h-batch. §7b asks whether that was a pulse-duration artifact (expect
balance between batches after scaling) or terminal-state biology (expect
holders on terminals of BOTH batches).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scjdo2.data import load_dynamo_hematopoiesis
from scjdo2.measured_drift import MeasuredDrift, MeasuredDriftConfig


REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "benchmarks" / "v2" / "dynamo_hematopoiesis"
TERMINAL_FATES = {"Ery", "Mon", "Neu", "Bas", "Meg"}

# Seurat panels (same as §7a).
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


def _score(X_log: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return np.zeros(X_log.shape[0], dtype=np.float64)
    s = X_log[:, mask].mean(axis=1).astype(np.float64)
    s = (s - s.mean()) / (s.std() + 1e-9)
    return s


def _per_cell_counts(a) -> np.ndarray:
    X = a.X
    try:
        X = X.toarray()
    except AttributeError:
        pass
    X = np.asarray(X, dtype=np.float64)
    return np.log1p(X)


def _lead_re(J: np.ndarray) -> np.ndarray:
    out = np.empty(J.shape[0], dtype=np.float64)
    for c in range(J.shape[0]):
        try:
            out[c] = float(np.max(np.linalg.eigvals(J[c]).real))
        except np.linalg.LinAlgError:
            out[c] = np.nan
    return out


def _lead_eigvec(J: np.ndarray) -> np.ndarray:
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


def _project_out(J: np.ndarray, U: np.ndarray) -> np.ndarray:
    I = np.eye(J.shape[1], dtype=np.float64)
    P = I - U @ U.T
    return np.einsum("ij,cjk,kl->cil", P, J.astype(np.float64), P)


def _fit_cycle_dirs(z: np.ndarray, X_log: np.ndarray,
                    gene_names: np.ndarray) -> np.ndarray:
    """Return orthonormal (d, 2) matrix of latent cycle directions."""
    s_mask = np.array([g in SEURAT_S_GENES for g in gene_names], dtype=bool)
    g_mask = np.array([g in SEURAT_G2M_GENES for g in gene_names], dtype=bool)
    S = _score(X_log, s_mask)
    G2M = _score(X_log, g_mask)
    u_S = np.linalg.lstsq(z, S, rcond=None)[0]
    u_G = np.linalg.lstsq(z, G2M, rcond=None)[0]
    u_S = u_S / (np.linalg.norm(u_S) + 1e-12)
    u_G = u_G / (np.linalg.norm(u_G) + 1e-12)
    U, _ = np.linalg.qr(np.column_stack([u_S, u_G]))
    return U, S, G2M


def _fit_and_classify(z, v, U_cycle, label, epochs, cell_types, out_cols):
    """Fit a MeasuredDrift on (z, v), compute per-cell lead_re under the
    unprojected and rotation-projected variants, return summary row for
    each.
    """
    drift = MeasuredDrift(dim=z.shape[1], config=MeasuredDriftConfig(
        hidden=(96, 96, 96), n_epochs=epochs, weight_decay=0.0,
        seed=0, clock_calibrated=True,
    )).fit(z, v, verbose=False)
    J = drift.jacobian_autodiff(z)
    rows = []
    for variant_name, J_use in (("A", J), ("C", _project_out(J, U_cycle))):
        lead = _lead_re(J_use)
        vec = _lead_eigvec(J_use)
        num = np.sum(np.abs(vec @ U_cycle) ** 2, axis=1)
        den = np.sum(np.abs(vec) ** 2, axis=1) + 1e-12
        cycle_w = num / den
        r = lead[np.isfinite(lead)]
        rec = {
            "fit": label, "variant": variant_name,
            "n": int(r.size),
            "n_fork": int((r > 0).sum()),
            "n_holder": int((r < 0).sum()),
            "frac_holder": float((r < 0).sum() / r.size) if r.size else 0.0,
            "median_lead_re": float(np.median(r)),
            "cycle_w_median": float(np.median(cycle_w)),
            "cycle_w_q95": float(np.quantile(cycle_w, 0.95)),
        }
        # Holder counts per terminal fate.
        holder_mask = (lead < 0) & np.isfinite(lead)
        for f in TERMINAL_FATES:
            m = cell_types == f
            rec[f"holder_{f}"] = int((holder_mask & m).sum())
            rec[f"n_{f}"] = int(m.sum())
        # Record which cells are holders (indices in the fit's local
        # index frame — caller maps back to the parent dataset).
        rec["_holder_local_idx"] = np.where(holder_mask)[0].tolist()
        rec["_lead_re_full"] = lead
        rec["_cycle_w_full"] = cycle_w
        rows.append(rec)
    return rows


def _holder_quality(a, local_idx: list[int], global_idx: np.ndarray,
                    label: str) -> list[dict]:
    """Depth report for holder cells: total UMI + genes detected vs
    dataset median."""
    X = a.X
    try:
        X = X.toarray()
    except AttributeError:
        pass
    X = np.asarray(X)
    total_umi = np.asarray(X.sum(axis=1)).ravel()
    n_genes = (X > 0).sum(axis=1)
    med_umi = float(np.median(total_umi))
    med_ng = float(np.median(n_genes))
    rows = []
    for i in local_idx:
        g = int(global_idx[i])
        rows.append({
            "fit": label, "global_cell": g,
            "cell_type": str(a.obs["cell_type"].iloc[g]),
            "day_h": float(a.obs["day"].iloc[g]),
            "clock_scale": float(a.obs.get("clock_scale",
                                            pd.Series([1.0] * a.n_obs)).iloc[g]),
            "total_UMI": int(total_umi[g]),
            "n_genes_detected": int(n_genes[g]),
            "UMI_ratio_vs_median": float(total_umi[g] / med_umi) if med_umi else float("nan"),
            "ngenes_ratio_vs_median": float(n_genes[g] / med_ng) if med_ng else float("nan"),
        })
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-pcs", type=int, default=10)
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--t-ref", type=float, default=5.0,
                   help="reference pulse (hours) for the clock scale "
                        "normalisation; dynamo fit at t=5 h")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"[§7b] loading Dynamo hemat (d_latent={args.n_pcs})")
    a = load_dynamo_hematopoiesis(n_pcs=args.n_pcs)
    gene_names = np.asarray(a.var_names).astype(str)
    z = np.asarray(a.obsm["X_pca"]).astype(np.float32)
    v_raw = np.asarray(a.obsm["velocity_pca"]).astype(np.float32)
    cell_types = a.obs["cell_type"].astype(str).to_numpy()
    day_h = a.obs["day"].astype(float).to_numpy()
    print(f"[§7b]   pulse-duration distribution:")
    print(a.obs["day"].value_counts().sort_index().to_string())

    # (1) Clock scale: v_cell_per_hour = v_raw * (t_ref / t_c).
    clock_scale = args.t_ref / np.clip(day_h, 1e-6, None)
    a.obs["clock_scale"] = clock_scale
    v_scaled = v_raw * clock_scale.astype(np.float32)[:, None]
    print(f"\n[§7b] clock_scale: {args.t_ref}/t_c  (3h cells scaled by "
          f"{args.t_ref/3.0:.3f}, 5h cells by {args.t_ref/5.0:.3f})")
    print(f"[§7b]   velocity norm ratio median_3h/median_5h "
          f"before scaling = "
          f"{float(np.median(np.linalg.norm(v_raw[day_h==3.0], axis=1)) / np.median(np.linalg.norm(v_raw[day_h==5.0], axis=1))):.3f}  "
          f"after = "
          f"{float(np.median(np.linalg.norm(v_scaled[day_h==3.0], axis=1)) / np.median(np.linalg.norm(v_scaled[day_h==5.0], axis=1))):.3f}")

    # Cycle directions (fit once on all cells, reused across every variant).
    X_log = _per_cell_counts(a)
    U_cycle, _S, _G2M = _fit_cycle_dirs(z, X_log, gene_names)

    all_summary: list[dict] = []
    all_holders: list[dict] = []

    def _run_split(label, idx):
        global_idx = np.asarray(idx, dtype=np.int64)
        print(f"\n[§7b] {label} — fit on {global_idx.size} cells")
        if global_idx.size < 50:
            print(f"[§7b]   too few cells, skipping")
            return
        t0 = time.monotonic()
        rows = _fit_and_classify(
            z[global_idx], v_scaled[global_idx], U_cycle, label,
            args.epochs, cell_types[global_idx], TERMINAL_FATES,
        )
        for r in rows:
            local_idx = r.pop("_holder_local_idx")
            r.pop("_lead_re_full")
            r.pop("_cycle_w_full")
            print(f"[§7b]   {r['fit']} / {r['variant']}: "
                  f"n_holder={r['n_holder']} ({r['frac_holder']*100:.2f}%)  "
                  f"cycle_w_med={r['cycle_w_median']:.3f}  "
                  f"per-fate: " +
                  "  ".join(f"{f}={r[f'holder_{f}']}/{r[f'n_{f}']}"
                             for f in sorted(TERMINAL_FATES)))
            all_summary.append(r)
            all_holders.extend(_holder_quality(a, local_idx, global_idx,
                                               f"{label}/{r['variant']}"))
        print(f"[§7b]   {time.monotonic()-t0:.1f}s")

    # (2) All cells, clock-scaled.
    _run_split("all_clock_scaled", np.arange(a.n_obs))

    # (3) Per-batch (same clock-scaled velocity).
    _run_split("batch_3h", np.where(day_h == 3.0)[0])
    _run_split("batch_5h", np.where(day_h == 5.0)[0])

    # Comparison with the §7 (unscaled) baseline from §7a:
    print("\n[§7b] baseline (unscaled velocity, from §7a):")
    print("  all_unscaled / A: n_holder=37 (1.9%) cycle_w_med=0.079")
    print("  all_unscaled / C: n_holder=8  (0.4%) cycle_w_med=0.0")

    summary = pd.DataFrame(all_summary)
    summary.to_csv(OUT / "s7b_summary.csv", index=False)
    pd.DataFrame(all_holders).to_csv(OUT / "s7b_holder_quality.csv", index=False)

    print("\n[§7b] SUMMARY TABLE:")
    cols = ["fit", "variant", "n", "n_holder", "frac_holder",
            "cycle_w_median", "cycle_w_q95"]
    cols += [f"holder_{f}" for f in sorted(TERMINAL_FATES)]
    print(summary[cols].to_string(index=False))

    # (4) Depth report.
    depth = pd.DataFrame(all_holders)
    if len(depth):
        print("\n[§7b] HOLDER DEPTH (UMI + n_genes vs dataset median):")
        print(depth.groupby(["fit"]).agg(
            n_holder=("global_cell", "count"),
            UMI_ratio_median=("UMI_ratio_vs_median", "median"),
            UMI_ratio_q05=("UMI_ratio_vs_median", lambda s: s.quantile(0.05)),
            UMI_ratio_q95=("UMI_ratio_vs_median", lambda s: s.quantile(0.95)),
            ngenes_ratio_median=("ngenes_ratio_vs_median", "median"),
        ).to_string())
    print(f"\n[§7b] outputs under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
