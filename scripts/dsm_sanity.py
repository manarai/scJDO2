"""DSM sanity checks before any multi-σ tuning.

Section A — Isotropic Gaussian unit test
    Sample ``z ~ N(0, s² I)`` in the same latent dim (20D), fit the drift
    with ``inject_velocity_prior=False`` (DSM only), then compute

        ∂ model.score / ∂z   at test points

    The exact score of a unit isotropic Gaussian is ``-z / s²`` so the
    analytic Jacobian is ``-I / s²``. Report what the module returns:

        ≈ -I / s²   → DSM module learns the score correctly
                       → pitchfork failure is in the mixture / time
                         conditioning path.
        ≈ +I / s²   → sign error between score and drift.
        ≈ 0         → score isn't being learned (loss sits at the floor).

Section B — Theoretical DSM floor
    v1's DSM loss is ``F.mse_loss(s, -ε/σ)`` with ``s = model.score(x+σε, t)``,
    averaged element-wise. If ``s_θ ≡ 0`` the loss collapses to
    ``E[ε²] / σ² = 1 / σ²`` (independent of ``d``). For ``σ = 0.1`` that
    floor is ``100``. Compare to the ~94 plateau on the pitchfork.

    If plateau ≈ floor: the network output on the score path is ~0, so the
    autodiff Jacobian of the drift comes entirely from the velocity prior
    and the residual MLP. Dump each term's Jacobian separately.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from scjdo2 import opspace
from scjdo2.data import generate_pitchfork_equilibrated
from scjdo2.drift import DriftAdapter, DriftConfig
from scjdo2.pipeline import fit_latent_field


REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "benchmarks" / "phase_b_gate" / "dsm_sanity"


# --------------------------------------------------- A. Gaussian unit test

def section_a(d: int, s: float, n_cells: int, n_epochs: int, seed: int,
              verbose: bool) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    z = (s * rng.standard_normal(size=(n_cells, d))).astype(np.float32)
    t = np.linspace(0.0, 1.0, n_cells, dtype=np.float32)

    drift = DriftAdapter(dim=d, config=DriftConfig(
        hidden=128, depth=3, n_epochs=n_epochs, seed=seed,
        use_spectral_norm=False, batch_size=256,
        inject_velocity_prior=False,
        alpha_control=0.0,            # no control pressure
        sigma=0.1,
    ))
    t0 = time.monotonic()
    drift.fit(z, t=t, verbose=verbose)
    dt_fit = time.monotonic() - t0

    # Jacobian of model.score (NOT model.forward) at 30 random test points.
    device = drift._device
    model = drift._model
    model.eval()
    rng2 = np.random.default_rng(seed + 1)
    test_z = (s * rng2.standard_normal(size=(30, d))).astype(np.float32)
    test_t = np.full(30, 0.5, dtype=np.float32)
    xg = torch.as_tensor(test_z, dtype=torch.float32, device=device).requires_grad_(True)
    tg = torch.as_tensor(test_t, dtype=torch.float32, device=device)

    # Per-point ∂score/∂z — manual loop so we isolate model.score (not model.forward).
    J_score_list = []
    for i in range(30):
        xi = xg[i:i + 1].detach().requires_grad_(True)
        ti = tg[i:i + 1]
        s_out = model.score(xi, ti)                            # (1, d)
        Ji = torch.zeros(d, d, device=device)
        for k in range(d):
            g = torch.autograd.grad(s_out[0, k], xi, create_graph=False,
                                    retain_graph=(k < d - 1))[0][0]
            Ji[k] = g
        J_score_list.append(Ji.detach().cpu().numpy())
    J_score = np.stack(J_score_list, axis=0)                   # (30, d, d)

    J_mean = J_score.mean(axis=0)
    diag_mean = np.mean(np.diag(J_mean))
    off_diag_mean = (np.sum(J_mean) - np.trace(J_mean)) / (d * d - d)
    analytic_diag = -1.0 / (s ** 2)
    frob_err = float(np.linalg.norm(J_mean - analytic_diag * np.eye(d), ord="fro")
                     / (np.linalg.norm(analytic_diag * np.eye(d), ord="fro") + 1e-12))

    # Verdict
    if abs(diag_mean - analytic_diag) / abs(analytic_diag) < 0.3:
        verdict = "CORRECT  (score ≈ analytic; pitchfork failure is in the mixture/conditioning)"
    elif diag_mean > 0.3 * abs(analytic_diag):
        verdict = f"SIGN ERROR (diag {diag_mean:+.2f}, expected {analytic_diag:+.2f})"
    elif abs(diag_mean) < 0.1 * abs(analytic_diag):
        verdict = f"NOT LEARNING (diag {diag_mean:+.3f} ≈ 0, expected {analytic_diag:+.2f})"
    else:
        verdict = f"PARTIAL (diag {diag_mean:+.2f}, expected {analytic_diag:+.2f})"

    return {
        "d": d, "s": s, "n_cells": n_cells, "n_epochs": n_epochs,
        "fit_seconds": dt_fit,
        "analytic_diag": float(analytic_diag),
        "learned_diag_mean": float(diag_mean),
        "learned_off_diag_mean": float(off_diag_mean),
        "frob_err_vs_analytic": frob_err,
        "verdict": verdict,
    }


# ------------------------------------- B. floor + per-term Jacobian split

def section_b(tau_crit: float, n_cells: int, n_alpha: int, n_epochs: int,
              seed: int, sigma: float, verbose: bool) -> dict[str, Any]:
    # Compute theoretical floor
    floor = 1.0 / (sigma ** 2)
    print(f"\n[B] Theoretical DSM floor for σ={sigma}: {floor:.1f}")

    # Fit the pitchfork so we can inspect the trained model
    adata = generate_pitchfork_equilibrated(
        tau_crit=tau_crit, seed=seed,
        n_cells=n_cells, n_alpha=n_alpha,
        n_genes=200, n_latent=20,
        noise_sigma=0.3,
        T_burn=20.0, T_sample=5.0, dt=0.02, sigma_sde=0.20,
        init_box=2.5,
    )
    cfg = DriftConfig(
        hidden=128, depth=3, n_epochs=n_epochs, seed=seed,
        use_spectral_norm=False, batch_size=256,
        inject_velocity_prior=True,     # standard pitchfork fit
        sigma=sigma,
        alpha_control=1e-3,
    )
    fit_latent_field(adata, use_rep="X_pca", time_key="pseudotime",
                     drift_config=cfg, jacobian_method="autodiff", verbose=verbose)

    # Decompose J at 30 random test points
    # drift = β · score + residual + vel_scale · gate · v_hat
    # Compute ∂term/∂z for each term.
    model = _get_last_model()
    if model is None:
        print("[B] WARNING: could not retrieve trained model; skipping decomposition")
        return {"floor_theoretical": floor, "note": "model handle unavailable"}
    device = next(model.parameters()).device
    beta = float(getattr(model.cfg, "beta", 0.1))
    vel_scale = float(getattr(model.cfg, "vel_scale", 2.0))

    rng2 = np.random.default_rng(seed + 2)
    idx = rng2.choice(adata.n_obs, size=30, replace=False)
    z = adata.obsm["X_pca"][idx].astype(np.float32)
    t = adata.obs["pseudotime"].to_numpy(dtype=np.float32)[idx]
    d = z.shape[1]

    J_score = np.zeros((30, d, d))
    J_residual = np.zeros((30, d, d))
    J_vel = np.zeros((30, d, d))
    J_total = np.zeros((30, d, d))
    for i in range(30):
        xi = torch.as_tensor(z[i:i + 1], device=device).requires_grad_(True)
        ti = torch.as_tensor(t[i:i + 1], device=device)

        # Term 1 — β · score(x, t)
        s_out = model.score(xi, ti)
        for k in range(d):
            g = torch.autograd.grad(s_out[0, k], xi, retain_graph=True)[0][0]
            J_score[i, k] = g.detach().cpu().numpy() * beta

        xi.grad = None
        # Term 2 — residual(x, t)
        r_out = model.residual(xi, ti)
        for k in range(d):
            g = torch.autograd.grad(r_out[0, k], xi, retain_graph=True)[0][0]
            J_residual[i, k] = g.detach().cpu().numpy()

        xi.grad = None
        # Term 3 — vel_scale · gate · v_hat (if applicable)
        if model.vel is not None:
            v_hat, conf = model.vel(xi)
            gate = conf.pow(model.cfg.vel_conf_power) * model._gate(ti)
            v_contrib = (vel_scale * gate).unsqueeze(-1) * v_hat
            for k in range(d):
                try:
                    g = torch.autograd.grad(v_contrib[0, k], xi, retain_graph=True,
                                            allow_unused=True)[0]
                    if g is None:
                        g = torch.zeros_like(xi[0])
                except RuntimeError:
                    g = torch.zeros_like(xi[0])
                J_vel[i, k] = g.detach().cpu().numpy() if hasattr(g, "detach") else np.asarray(g)

        # Term 4 — forward total
        xi2 = torch.as_tensor(z[i:i + 1], device=device).requires_grad_(True)
        u_out = model(xi2, ti)
        for k in range(d):
            g = torch.autograd.grad(u_out[0, k], xi2, retain_graph=True)[0][0]
            J_total[i, k] = g.detach().cpu().numpy()

    def _summary(J):
        tops = np.array([np.sort(np.linalg.eigvals(J[i]).real)[-1] for i in range(J.shape[0])])
        return {
            "frob_mean": float(np.mean([np.linalg.norm(J[i], ord="fro") for i in range(J.shape[0])])),
            "lead_re_mean": float(np.mean(tops)),
            "lead_re_frac_pos": float(np.mean(tops > 0)),
        }

    return {
        "tau_crit": tau_crit,
        "n_epochs": n_epochs,
        "sigma": sigma,
        "floor_theoretical": floor,
        "term_beta_score": _summary(J_score),
        "term_residual": _summary(J_residual),
        "term_vel_prior": _summary(J_vel),
        "term_total_drift": _summary(J_total),
        "note": (
            "term_beta_score is β·∂score/∂z (DSM-trained). term_residual is ∂residual/∂z "
            "(trained only via control_energy on the full forward). term_vel_prior is "
            "∂(vel_scale·gate·v_hat)/∂z (non-trainable buffer: kNN interpolation). "
            "term_total_drift is the Jacobian the pipeline actually uses."
        ),
    }


_last_model_ref: list = []


def _get_last_model():
    """Trick: expose the trained v1 DriftField through DriftAdapter's cache."""
    # We don't have a clean handle; use the drift adapter from a fresh fit instead.
    # Caller runs section_b_direct(); this function is a placeholder.
    return _last_model_ref[-1] if _last_model_ref else None


def section_b_direct(tau_crit: float, n_cells: int, n_alpha: int, n_epochs: int,
                     seed: int, sigma: float, verbose: bool) -> dict[str, Any]:
    """Same as section_b but fits via DriftAdapter directly so we keep a handle."""
    adata = generate_pitchfork_equilibrated(
        tau_crit=tau_crit, seed=seed,
        n_cells=n_cells, n_alpha=n_alpha,
        n_genes=200, n_latent=20,
        noise_sigma=0.3,
        T_burn=20.0, T_sample=5.0, dt=0.02, sigma_sde=0.20,
        init_box=2.5,
    )
    z = adata.obsm["X_pca"].astype(np.float32)
    tt = adata.obs["pseudotime"].to_numpy(dtype=np.float32)
    d = z.shape[1]

    adapter = DriftAdapter(dim=d, config=DriftConfig(
        hidden=128, depth=3, n_epochs=n_epochs, seed=seed,
        use_spectral_norm=False, batch_size=256,
        inject_velocity_prior=True, sigma=sigma, alpha_control=1e-3,
    ))
    adapter.fit(z, t=tt, verbose=verbose)
    _last_model_ref.append(adapter._model)
    device = adapter._device
    model = adapter._model
    beta = float(model.cfg.beta)
    vel_scale = float(model.cfg.vel_scale)

    rng2 = np.random.default_rng(seed + 2)
    idx = rng2.choice(adata.n_obs, size=30, replace=False)
    zi_all = z[idx]
    ti_all = tt[idx]
    floor = 1.0 / (sigma ** 2)

    J_score = np.zeros((30, d, d))
    J_residual = np.zeros((30, d, d))
    J_vel = np.zeros((30, d, d))
    J_total = np.zeros((30, d, d))
    for i in range(30):
        xi = torch.as_tensor(zi_all[i:i + 1], device=device).requires_grad_(True)
        ti = torch.as_tensor(ti_all[i:i + 1], device=device)

        s_out = model.score(xi, ti)
        for k in range(d):
            g = torch.autograd.grad(s_out[0, k], xi, retain_graph=True)[0][0]
            J_score[i, k] = g.detach().cpu().numpy() * beta

        xi = torch.as_tensor(zi_all[i:i + 1], device=device).requires_grad_(True)
        r_out = model.residual(xi, ti)
        for k in range(d):
            g = torch.autograd.grad(r_out[0, k], xi, retain_graph=True)[0][0]
            J_residual[i, k] = g.detach().cpu().numpy()

        if model.vel is not None:
            xi = torch.as_tensor(zi_all[i:i + 1], device=device).requires_grad_(True)
            v_hat, conf = model.vel(xi)
            gate = conf.pow(model.cfg.vel_conf_power) * model._gate(ti)
            v_contrib = (vel_scale * gate).unsqueeze(-1) * v_hat
            for k in range(d):
                try:
                    g_tuple = torch.autograd.grad(v_contrib[0, k], xi,
                                                  retain_graph=True, allow_unused=True)
                    g = g_tuple[0]
                    if g is None:
                        g_np = np.zeros(d)
                    else:
                        g_np = g.detach().cpu().numpy()[0]
                except RuntimeError:
                    g_np = np.zeros(d)
                J_vel[i, k] = g_np

        xi = torch.as_tensor(zi_all[i:i + 1], device=device).requires_grad_(True)
        u_out = model(xi, ti)
        for k in range(d):
            g = torch.autograd.grad(u_out[0, k], xi, retain_graph=True)[0][0]
            J_total[i, k] = g.detach().cpu().numpy()

    def _summary(J):
        tops = np.array([np.sort(np.linalg.eigvals(J[i]).real)[-1] for i in range(J.shape[0])])
        return {
            "frob_mean": float(np.mean([np.linalg.norm(J[i], ord="fro") for i in range(J.shape[0])])),
            "lead_re_mean": float(np.mean(tops)),
            "lead_re_frac_pos": float(np.mean(tops > 0)),
        }

    return {
        "tau_crit": tau_crit,
        "n_epochs": n_epochs,
        "sigma": sigma,
        "floor_theoretical": floor,
        "term_beta_score": _summary(J_score),
        "term_residual": _summary(J_residual),
        "term_vel_prior": _summary(J_vel),
        "term_total_drift": _summary(J_total),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-epochs-a", type=int, default=2000)
    p.add_argument("--n-epochs-b", type=int, default=5000)
    p.add_argument("--d", type=int, default=20)
    p.add_argument("--s", type=float, default=1.0)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--n-cells", type=int, default=1200)
    p.add_argument("--n-alpha", type=int, default=30)
    p.add_argument("--tau-crit", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    print(f"\n==== A. Isotropic Gaussian DSM unit test (d={args.d}, s={args.s}) ====")
    a = section_a(args.d, args.s, args.n_cells, args.n_epochs_a, args.seed, args.verbose)
    print(json.dumps(a, indent=2, default=float))

    print(f"\n==== B. DSM floor + per-term Jacobian (pitchfork τ={args.tau_crit}) ====")
    b = section_b_direct(args.tau_crit, args.n_cells, args.n_alpha, args.n_epochs_b,
                         args.seed, args.sigma, args.verbose)
    print(json.dumps(b, indent=2, default=float))

    out_json = OUT / "report.json"
    out_json.write_text(json.dumps(
        {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
         "args": vars(args), "section_a": a, "section_b": b},
        indent=2, default=float))
    print(f"\n[dsm sanity] wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
