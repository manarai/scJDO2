"""Can the DriftField architecture + autodiff Jacobian represent the toggle
dynamics correctly if we feed the analytic drift as the velocity prior?

If YES → the training signal (pseudotime velocity prior, DSM) is the bottleneck.
If NO  → there is a deeper architecture / autodiff issue.

Diagnostic only. Does not affect the pipeline.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from scjdo2 import opspace
from scjdo2.data import generate_pitchfork
from scjdo2.drift import DriftAdapter, DriftConfig
from scjdo2.field import FieldConfig


REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "benchmarks" / "phase_b_gate" / "convergence_audit"


def _toggle_drift_true(z_true: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    x, y = z_true[:, 0], z_true[:, 1]
    fx = alpha / (1.0 + y ** 4) - x
    fy = alpha / (1.0 + x ** 4) - y
    return np.stack([fx, fy], axis=1)


def main() -> int:
    print("[arch-check] generate")
    adata = generate_pitchfork(
        tau_crit=0.5, seed=0,
        n_cells=1200, n_genes=200, n_latent=20,
        noise_sigma=0.3, T_total=10.0, dt=0.02, sigma_sde=0.20,
    )
    meta = adata.uns["pitchfork"]
    alpha = meta["alpha_min"] + adata.obs["pseudotime"].to_numpy() * (
        meta["alpha_max"] - meta["alpha_min"]
    )

    # Analytic toggle drift in 2D true space; project to 20D PCA via lstsq.
    z_true = np.asarray(adata.obsm["X_true"], dtype=np.float64)
    z_pca = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
    v_true = _toggle_drift_true(z_true, alpha)

    # Fit a linear map B: 2 → 20 by least squares (z_pca ≈ z_true @ B).
    B, *_ = np.linalg.lstsq(z_true, z_pca, rcond=None)   # (2, 20)
    v_pca_ref = v_true @ B

    t = adata.obs["pseudotime"].to_numpy(dtype=np.float32)
    drift = DriftAdapter(dim=20, config=DriftConfig(
        hidden=128, depth=3, n_epochs=5000, seed=0,
        use_spectral_norm=False, batch_size=256,
        inject_velocity_prior=True,
        vel_scale=4.0,
    ))
    t0 = time.monotonic()
    drift.fit(z_pca.astype(np.float32), t=t, V_ref=v_pca_ref.astype(np.float32), verbose=True)
    print(f"  fit {time.monotonic() - t0:.1f}s")

    # Autodiff Jacobian at every cell at its own pseudotime
    J = drift.jacobian_autodiff(z_pca.astype(np.float32), t)
    v_learned = drift.velocity(z_pca.astype(np.float32), t)

    # True-space cos-sim
    A, *_ = np.linalg.lstsq(z_pca, z_true, rcond=None)   # (20, 2)
    v_learned_true = v_learned @ A
    cos = float(np.mean(
        np.sum(v_learned_true * v_true, axis=1)
        / (np.linalg.norm(v_learned_true, axis=1) * np.linalg.norm(v_true, axis=1) + 1e-12)
    ))
    print(f"\n  cos_sim(learned_v, analytic_v) in true space = {cos:+.3f}")

    # Project J back to true 2D
    J_true = np.einsum("ia,nab,bj->nij", A.T, J, A)

    # Sign tally in TRUE 2D space (should match the dynamical signature)
    tops = np.stack([
        np.sort(np.linalg.eigvals(J_true[i]).real)[::-1] for i in range(J_true.shape[0])
    ])
    print(f"  frac(both neg) = {float(np.mean((tops[:,0] < 0) & (tops[:,1] < 0))):.3f}  "
          f"frac(one pos)  = {float(np.mean((tops[:,0] > 0) & (tops[:,1] < 0))):.3f}  "
          f"frac(both pos) = {float(np.mean((tops[:,0] > 0) & (tops[:,1] > 0))):.3f}")

    # And in PCA space (what opspace actually operates on)
    tops_pca = np.stack([
        np.sort(np.linalg.eigvals(J[i]).real)[::-1][:2] for i in range(J.shape[0])
    ])
    print(f"  PCA-space:   frac(top1>0) = {float(np.mean(tops_pca[:,0] > 0)):.3f}  "
          f"frac(top2<0) = {float(np.mean(tops_pca[:,1] < 0)):.3f}")

    # Fork/holder on behaviour label
    adata.obsm["X_drift_v"] = v_learned
    adata.uns["scjdo2"] = {"J": J, "meta": {"clock_calibrated": False}}
    opspace.spectrum(adata, n_eig=5, fisher_mask=False)
    opspace.invariants(adata)
    opspace.behaviour(adata)
    beh = adata.obs["op_behaviour"].astype(str).to_numpy()
    tt = adata.obs["pseudotime"].to_numpy()
    pred_fork = beh == "fork"
    pred_holder = beh == "holder"
    true_fork = np.abs(tt - 0.5) < 0.05
    true_holder = np.abs(tt - 0.5) >= 0.15
    def pr(pred, truth):
        tp = int(np.sum(pred & truth)); fp = int(np.sum(pred & ~truth)); fn = int(np.sum(~pred & truth))
        p = tp / (tp + fp) if (tp + fp) else float("nan")
        r = tp / (tp + fn) if (tp + fn) else float("nan")
        return p, r
    fp_p, fp_r = pr(pred_fork, true_fork)
    hp_p, hp_r = pr(pred_holder, true_holder)
    print(f"\n  fork  P={fp_p:.2f} R={fp_r:.2f}")
    print(f"  holder P={hp_p:.2f} R={hp_r:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
