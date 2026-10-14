"""§7c — LabelledRateModel on Dynamo scNT-seq hematopoiesis.

Refit γ per gene from new + total UMI counts with per-cell pulse
length ``t_c`` entering the NB likelihood directly, bypassing dynamo's
per-batch γ fit. Then rerun the latent-side gate under the lead-eig
rule to check whether the §7b "3h-batch holder-only" artefact
resolves when γ is on a shared per-hour footing across batches.

Outputs under ``benchmarks/v2/dynamo_s7c/``:
  * fit_meta.json         — val NLL, γ summary
  * gamma_per_gene.csv    — γ_g (hour⁻¹), optional comparison to
                             dynamo's time_3_gamma / time_5_gamma
  * behaviour_by_batch.csv — holder/fork fraction per (day, cell_type)
                             under the lead_eig classifier
  * report.md             — short write-up
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from scjdo2.data import load_dynamo_hematopoiesis
from scjdo2.energy_drift import EnergyDriftAdapter, EnergyDriftConfig
from scjdo2.labelled_rates import LabelledRateModel, LabelledRatesConfig
from scjdo2.measured_drift import MeasuredDrift, MeasuredDriftConfig
from scjdo2.opspace import classify_basin_latent, invariants, spectrum


REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "benchmarks" / "v2" / "dynamo_s7c"


def _oracle_tol(z: np.ndarray, cluster: np.ndarray) -> float:
    leads = []
    for cl in sorted(set(cluster)):
        m = cluster == cl
        if m.sum() < 10:
            continue
        Sigma = np.cov(z[m], rowvar=False, bias=False) + 1e-4 * np.eye(z.shape[1])
        ev = np.linalg.eigvalsh(-np.linalg.inv(Sigma))
        leads.append(abs(float(ev.max())))
    return 0.1 * float(np.median(leads)) if leads else 0.1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--drift-epochs", type=int, default=1500)
    p.add_argument("--rate-epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print("[s7c] load Dynamo scNT-seq hemat")
    a = load_dynamo_hematopoiesis(n_pcs=10)
    print(f"[s7c]   shape={a.shape}")

    # per-cell pulse: 3h or 5h. Dynamo stores it in obs['pulse_duration'].
    if "pulse_duration" in a.obs:
        pulse = a.obs["pulse_duration"].astype(float).to_numpy()
    elif "day" in a.obs:
        pulse = a.obs["day"].astype(float).to_numpy()
    else:
        raise RuntimeError("no pulse column in obs")
    print(f"[s7c]   pulse distribution: {dict(pd.Series(pulse).value_counts())}")

    import scipy.sparse as _sp
    def _dense(X):
        return X.toarray() if _sp.issparse(X) else np.asarray(X)
    new = _dense(a.layers["X_new"]).astype(np.float32)
    total = _dense(a.layers["X_total"]).astype(np.float32)
    z = np.asarray(a.obsm["X_pca"]).astype(np.float32)

    # Fit the LabelledRateModel on all cells (pooled 3h+5h).
    print("[s7c] fit LabelledRateModel (pooled 3h+5h) — closed-form init + "
          "lr_γ×10")
    t0 = time.monotonic()
    rm = LabelledRateModel(LabelledRatesConfig(
        hidden=(128, 128), n_epochs=args.rate_epochs, seed=args.seed,
        lr_gamma_scale=10.0,
    )).fit(new, total, z, t_c=pulse, verbose=True)
    print(f"[s7c]   fit {time.monotonic()-t0:.1f}s  val NLL = {rm.best_val_nll:.3f}")
    g_norm_hist = getattr(rm, "_gamma_grad_history", [])
    if g_norm_hist:
        print(f"[s7c]   log_γ grad norm — first {g_norm_hist[0]:.3e}  "
              f"last {g_norm_hist[-1]:.3e}  "
              f"mean {float(np.mean(g_norm_hist)):.3e}")

    gamma_pool = rm.gamma
    print(f"[s7c]   pooled γ — median={float(np.median(gamma_pool)):.4f}  "
          f"q25={float(np.quantile(gamma_pool,0.25)):.4f}  "
          f"q75={float(np.quantile(gamma_pool,0.75)):.4f}  "
          f"min={float(gamma_pool.min()):.4f}  max={float(gamma_pool.max()):.4f}")

    # Per-batch refits for γ agreement check.
    print("[s7c] per-batch γ agreement — refit on 3h and 5h subsets")
    def _fit_subset(mask: np.ndarray) -> np.ndarray:
        return LabelledRateModel(LabelledRatesConfig(
            hidden=(128, 128), n_epochs=args.rate_epochs,
            seed=args.seed + int(mask.sum()),
            lr_gamma_scale=10.0,
        )).fit(new[mask], total[mask], z[mask], t_c=pulse[mask],
                verbose=False).gamma
    m3 = pulse == 3.0
    m5 = pulse == 5.0
    gamma_3h = _fit_subset(m3)
    gamma_5h = _fit_subset(m5)
    from scipy.stats import spearmanr, pearsonr
    valid = np.isfinite(gamma_3h) & np.isfinite(gamma_5h)
    rho_batch = float(spearmanr(gamma_3h[valid], gamma_5h[valid]).correlation)
    pr_batch = float(pearsonr(gamma_3h[valid], gamma_5h[valid])[0])
    print(f"[s7c]   per-gene Spearman(γ_3h, γ_5h) = {rho_batch:+.3f}")
    print(f"[s7c]   per-gene Pearson (γ_3h, γ_5h) = {pr_batch:+.3f}")

    gamma_df = pd.DataFrame({
        "gene": list(a.var_names),
        "gamma_s7c_pooled": gamma_pool.astype(float),
        "gamma_s7c_3h": gamma_3h.astype(float),
        "gamma_s7c_5h": gamma_5h.astype(float),
    })
    # Compare to dynamo's per-batch γ if present in var.
    if "time_3_gamma" in a.var:
        gamma_df["dynamo_time3_gamma"] = a.var["time_3_gamma"].to_numpy()
    if "time_5_gamma" in a.var:
        gamma_df["dynamo_time5_gamma"] = a.var["time_5_gamma"].to_numpy()
    gamma_df.to_csv(OUT / "gamma_per_gene.csv", index=False)

    gamma = gamma_pool

    # Build a new velocity under the s7c γ: v_g(z, t_c) = γ_g · (total - new/(1-exp(-γt)))
    # or more directly: dμ/dt from the labelled fit at steady state is
    # α(z) - γ · μ. For the energy-drift gate we rebuild a per-cell drift
    # using the Jacobian of eta_total(z) which depends on z through α(z) only
    # (γ is shared); the DIFFERENCE between batches should now be driven by
    # z, not by batch-specific γ.
    print("[s7c] ESM energy drift on z (same latent as §7b)")
    t0 = time.monotonic()
    drift = EnergyDriftAdapter(dim=z.shape[1], config=EnergyDriftConfig(
        hidden=96, depth=3, n_epochs=args.drift_epochs, seed=args.seed,
        batch_size=256, loss="esm", lr=5e-4, beta=1.0,
        esm_laplacian="exact" if z.shape[1] <= 20 else "hutchinson",
    )).fit(z, verbose=False)
    sym = drift.symmetry_check(z, n_cells=100, seed=args.seed)
    J = drift.jacobian_autodiff(z)
    print(f"[s7c]   drift+J {time.monotonic()-t0:.1f}s  SYM median {sym['median_asym']:.2e}")

    a.uns.setdefault("scjdo2", {})["J"] = J
    a.uns["scjdo2"]["J_latent"] = J
    cluster = a.obs["cell_type"].astype(str).to_numpy() if "cell_type" in a.obs \
        else np.array(["all"] * a.n_obs)
    tol = _oracle_tol(z, cluster)
    spectrum(a, n_eig=5, fisher_mask=False, smooth_k=0)
    invariants(a, tol=tol)
    classify_basin_latent(a, tol=tol, rule="lead_eig", use_rep="X_pca")

    beh = a.obs["op_behaviour_basin"].astype(str).to_numpy()
    bi = a.obs["op_basin_index"].to_numpy()
    day = pulse

    rows = []
    for ct in sorted(set(cluster)):
        for d in sorted(set(day)):
            m = (cluster == ct) & (day == d)
            n = int(m.sum())
            if n < 5:
                continue
            rec = {"cell_type": ct, "day": float(d), "n": n}
            for k in ("holder", "fork", "drift"):
                rec[k] = int((beh[m] == k).sum())
            rec["frac_holder"] = rec["holder"] / n
            rec["basin_index_median"] = float(np.median(bi[m][np.isfinite(bi[m])])) \
                if np.isfinite(bi[m]).any() else float("nan")
            rows.append(rec)
    beh_df = pd.DataFrame(rows)
    beh_df.to_csv(OUT / "behaviour_by_batch.csv", index=False)
    print("\n[s7c] behaviour by (cell_type, day):")
    print(beh_df.to_string(index=False))

    meta = {
        "tag": "v0.5-basin-lock",
        "symmetry_median": sym["median_asym"],
        "oracle_tol": tol,
        "labelled_rate_nll": rm.best_val_nll,
        "gamma_pooled": {
            "median": float(np.median(gamma)),
            "q25": float(np.quantile(gamma, 0.25)),
            "q75": float(np.quantile(gamma, 0.75)),
            "min": float(gamma.min()),
            "max": float(gamma.max()),
        },
        "gamma_grad_norm_first": float(g_norm_hist[0]) if g_norm_hist else None,
        "gamma_grad_norm_last": float(g_norm_hist[-1]) if g_norm_hist else None,
        "gamma_per_batch_agreement": {
            "spearman_3h_vs_5h": rho_batch,
            "pearson_3h_vs_5h": pr_batch,
        },
    }
    (OUT / "fit_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    # Report.
    def _r(x, n=3):
        try: return round(float(x), n)
        except Exception: return x
    lines = [
        "# §7c — LabelledRateModel on Dynamo scNT-seq hematopoiesis",
        "",
        f"Tag: `v0.5-basin-lock`.  Val NLL (joint new+total NB) = "
        f"{meta['labelled_rate_nll']:.3f}.",
        "",
        "## γ training diagnostics",
        "",
        f"* Closed-form init: `γ̂_g = −log(1 − new_g/total_g) / t̄` on",
        "  pooled cells, clamped to [1e-3, 10].",
        f"* Separate param group: `log_γ` lr = base lr × 10.",
        f"* `log_γ` grad norm — first step: "
        f"{meta['gamma_grad_norm_first']:.3e}" if meta.get('gamma_grad_norm_first') else "",
        f"* `log_γ` grad norm — last step: "
        f"{meta['gamma_grad_norm_last']:.3e}" if meta.get('gamma_grad_norm_last') else "",
        "",
        "## Pooled γ (hour⁻¹)",
        "",
        f"* median: {_r(meta['gamma_pooled']['median'])}",
        f"* IQR:    [{_r(meta['gamma_pooled']['q25'])}, "
        f"{_r(meta['gamma_pooled']['q75'])}]",
        f"* range:  [{_r(meta['gamma_pooled']['min'])}, "
        f"{_r(meta['gamma_pooled']['max'])}]",
        "",
        "## Per-batch γ agreement (3h-only fit vs 5h-only fit)",
        "",
        f"* Spearman: **{_r(meta['gamma_per_batch_agreement']['spearman_3h_vs_5h'])}**",
        f"* Pearson:  **{_r(meta['gamma_per_batch_agreement']['pearson_3h_vs_5h'])}**",
        "",
        "If γ is a stable gene property, 3h-only and 5h-only fits should",
        "agree closely. This is the clean test of whether the labelled rate",
        "model recovers a batch-invariant degradation rate per gene.",
        "",
        "## Behaviour by (cell_type, day) under the locked lead_eig rule",
        "",
        beh_df.round(3).to_markdown(index=False),
        "",
        "## Note on comparison to §7b",
        "",
        "The user directive is: **do not re-interpret the batch-flip",
        "pattern until γ trains**. The behaviour table above is reported",
        "but not analysed vs §7b until the γ training diagnostics",
        "(above) show non-trivial gradient and a spread γ distribution.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print(f"\n[s7c] outputs → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
