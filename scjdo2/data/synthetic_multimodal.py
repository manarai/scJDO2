"""Seeded synthetic multimodal lineage trajectories with exact dynamical truth.

The generator intentionally supplies count-level inputs for external scVelo and
Palantir runs without invoking either package.  It also retains the noiseless
state, RNA, and embedding-velocity truth needed to score an orientation layer.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from math import log
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from sklearn.decomposition import FactorAnalysis, PCA

D = 6
DEFAULT_GENES = 800
DEFAULT_DT = 0.1
Z1_MAX = 3.0
Z1_CRIT = 1.0
KAPPA = 0.5
RHO0 = 1.0
RADIAL_LAMBDA = 0.1
OMEGA = 2.0 * np.pi / 20.0
C_EXIT = 0.2
C_STABLE = 0.2
ETA_U = 0.3


@dataclass(frozen=True)
class _Config:
    preset: str
    seed: int
    n_founders: int = 300
    n_genes: int = DEFAULT_GENES
    dt: float = DEFAULT_DT
    sigma: float = 0.05
    poisson: bool = True
    division_hours: float = 16.0
    f_sample: float = 0.20
    sampling_times: tuple[float, ...] = (24.0, 48.0, 72.0, 96.0)
    p_barcode: float = 0.60
    labelling_efficiency: float = 0.90
    pulse_hours: tuple[float, ...] = (2.0,)
    stress_beta_fraction: float = 0.0
    batch_effect: bool = False
    stationary_cycle: bool = False


def _make_config(preset: str, seed: int, overrides: dict[str, Any]) -> _Config:
    if preset not in {"default", "stress", "batch", "stationary_cycle"}:
        raise ValueError(f"unknown preset {preset!r}")
    values: dict[str, Any] = {"preset": preset, "seed": int(seed)}
    if preset == "stress":
        values.update(p_barcode=0.30, labelling_efficiency=0.60,
                      pulse_hours=(1.0, 2.0, 4.0), stress_beta_fraction=0.20)
    elif preset == "batch":
        values.update(batch_effect=True)
    elif preset == "stationary_cycle":
        # One 24-h collection; 5 000 founders → ≈ 2 800 cells, enough for local
        # regressions (k = 30) in a 20-D embedding (review orient/S01, note 2).
        values.update(stationary_cycle=True, sampling_times=(24.0,), n_founders=5000)
    legal = set(_Config.__dataclass_fields__)
    unknown = sorted(set(overrides) - legal)
    if unknown:
        raise TypeError(f"unknown make_multimodal overrides: {unknown}")
    values.update(overrides)
    config = _Config(**values)
    if config.n_founders < 1 or config.n_genes < 2:
        raise ValueError("n_founders must be positive and n_genes must be at least 2")
    if config.dt <= 0 or config.division_hours <= 0 or not (0 < config.f_sample <= 1):
        raise ValueError("dt, division_hours, and f_sample must be positive")
    if not config.sampling_times or any(t <= 0 for t in config.sampling_times):
        raise ValueError("sampling_times must be positive and nonempty")
    if any(abs(t / config.dt - round(t / config.dt)) > 1e-8 for t in config.sampling_times):
        raise ValueError("sampling_times must lie on the fixed dt grid")
    if any(p <= 0 or p > min(config.sampling_times) for p in config.pulse_hours):
        raise ValueError("each pulse duration must be positive and precede collection")
    return config


def _sigmoid_gate(z1: np.ndarray, z1_crit: float = Z1_CRIT) -> np.ndarray:
    x = np.clip((z1 - float(z1_crit)) / 0.1, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(x))


def _state_drift(z: np.ndarray, stationary_cycle: bool, z1_crit: float = Z1_CRIT) -> np.ndarray:
    # ``z1_crit`` (orient/S10 Step 0) moves the pitchfork and the cycle-exit
    # gate together; the default reproduces v1/v2/v3 exactly.
    z1_crit = float(z1_crit)
    z = np.asarray(z, dtype=np.float32)
    out = np.empty_like(z)
    z0, z1, z2, z3, z4, z5 = (z[:, i] for i in range(D))
    mu = z1 - z1_crit
    gate = _sigmoid_gate(z1, z1_crit).astype(np.float32)
    r2 = z2 * z2 + z3 * z3
    h2 = RADIAL_LAMBDA * (RHO0 * RHO0 - r2) * z2 - OMEGA * z3
    h3 = RADIAL_LAMBDA * (RHO0 * RHO0 - r2) * z3 + OMEGA * z2
    out[:, 0] = KAPPA * (mu * z0 - z0 ** 3)
    out[:, 1] = 0.0 if stationary_cycle else 0.04 * (1.0 - z1 / Z1_MAX)
    out[:, 2] = gate * h2 - (1.0 - gate) * C_EXIT * z2
    out[:, 3] = gate * h3 - (1.0 - gate) * C_EXIT * z3
    out[:, 4] = -C_STABLE * z4
    out[:, 5] = -C_STABLE * z5
    return out


def _state_jacobian(z: np.ndarray, stationary_cycle: bool, z1_crit: float = Z1_CRIT) -> np.ndarray:
    """Analytic ``∂F/∂z`` for every state in ``z``."""
    z1_crit = float(z1_crit)
    z = np.asarray(z, dtype=np.float32)
    n = len(z)
    J = np.zeros((n, D, D), dtype=np.float32)
    z0, z1, z2, z3 = (z[:, i] for i in range(4))
    gate = _sigmoid_gate(z1, z1_crit).astype(np.float32)
    gate_prime = (-10.0 * gate * (1.0 - gate)).astype(np.float32)
    r2 = z2 * z2 + z3 * z3
    radial = RADIAL_LAMBDA * (RHO0 * RHO0 - r2)
    h2 = radial * z2 - OMEGA * z3
    h3 = radial * z3 + OMEGA * z2
    J[:, 0, 0] = KAPPA * ((z1 - z1_crit) - 3.0 * z0 * z0)
    J[:, 0, 1] = KAPPA * z0
    J[:, 1, 1] = 0.0 if stationary_cycle else -0.04 / Z1_MAX
    J[:, 2, 1] = gate_prime * (h2 + C_EXIT * z2)
    J[:, 3, 1] = gate_prime * (h3 + C_EXIT * z3)
    J[:, 2, 2] = gate * RADIAL_LAMBDA * (RHO0 * RHO0 - r2 - 2.0 * z2 * z2) - (1.0 - gate) * C_EXIT
    J[:, 2, 3] = gate * (-2.0 * RADIAL_LAMBDA * z2 * z3 - OMEGA)
    J[:, 3, 2] = gate * (-2.0 * RADIAL_LAMBDA * z2 * z3 + OMEGA)
    J[:, 3, 3] = gate * RADIAL_LAMBDA * (RHO0 * RHO0 - r2 - 2.0 * z3 * z3) - (1.0 - gate) * C_EXIT
    J[:, 4, 4] = -C_STABLE
    J[:, 5, 5] = -C_STABLE
    return J


def _gene_parameters(config: _Config, rng: np.random.Generator) -> dict[str, np.ndarray]:
    n_loadings = rng.integers(1, 4, size=config.n_genes)
    W = np.zeros((config.n_genes, D), dtype=np.float32)
    for gene, k in enumerate(n_loadings):
        dims = rng.choice(D, size=int(k), replace=False)
        W[gene, dims] = rng.normal(0.0, 1.0, size=int(k)).astype(np.float32)
    b = rng.normal(log(2.0), 0.7, size=config.n_genes).astype(np.float32)
    beta_initial = rng.lognormal(log(3.0), 0.5, size=config.n_genes).astype(np.float32)
    gamma = rng.lognormal(log(0.2), 0.6, size=config.n_genes).astype(np.float32)
    changed = np.zeros(config.n_genes, dtype=bool)
    if config.stress_beta_fraction:
        changed[rng.choice(config.n_genes, size=int(round(config.n_genes * config.stress_beta_fraction)), replace=False)] = True
    beta_final = beta_initial.copy()
    beta_final[changed] *= 2.0
    return {"W": W, "b": b, "beta_initial": beta_initial, "beta_final": beta_final, "gamma": gamma, "beta_changed": changed}


def _alpha(z: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.exp(z @ W.T + b[None, :]).astype(np.float32)


def _rna_step(u: np.ndarray, s: np.ndarray, alpha: np.ndarray, beta: np.ndarray,
              gamma: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact linear RNA update over ``dt`` with alpha frozen at the step start."""
    eb = np.exp(-beta * dt).astype(np.float32)
    eg = np.exp(-gamma * dt).astype(np.float32)
    denom = gamma - beta
    ib = np.empty_like(beta)
    near = np.abs(denom) < 1e-6
    ib[near] = dt * eg[near]
    ib[~near] = (eb[~near] - eg[~near]) / denom[~near]
    i0 = (1.0 - eg) / gamma
    u_next = u * eb[None, :] + alpha * ((1.0 - eb) / beta)[None, :]
    s_next = s * eg[None, :] + u * (beta * ib)[None, :] + alpha * (i0 - ib)[None, :]
    return u_next.astype(np.float32), s_next.astype(np.float32)


def _append_daughters(live: dict[str, np.ndarray], divide: np.ndarray, time: float,
                      division_rate: float, rng: np.random.Generator) -> None:
    if not np.any(divide):
        return
    idx = np.flatnonzero(divide)
    for key in ("z", "u", "s", "u_new", "s_new", "clone", "cell_id"):
        live[key] = np.concatenate((live[key], live[key][idx].copy()), axis=0)
    live["next_div"][idx] = time + rng.exponential(1.0 / division_rate, size=len(idx))
    daughters = time + rng.exponential(1.0 / division_rate, size=len(idx))
    live["next_div"] = np.concatenate((live["next_div"], daughters.astype(np.float32)))


def _remove_indices(live: dict[str, np.ndarray], indices: np.ndarray) -> None:
    keep = np.ones(len(live["z"]), dtype=bool)
    keep[indices] = False
    for key in tuple(live):
        live[key] = live[key][keep]


def _batch_gene_effects(config: _Config, rng: np.random.Generator) -> np.ndarray:
    if config.batch_effect:
        return rng.lognormal(0.0, 0.3, size=(len(config.sampling_times), config.n_genes)).astype(np.float32)
    return np.ones((len(config.sampling_times), config.n_genes), dtype=np.float32)


def _one_step_expression(s: np.ndarray, ds_dt: np.ndarray, dt: float) -> np.ndarray:
    """Keep float64 precision for the infinitesimal validation step."""
    return np.maximum(np.asarray(s, dtype=np.float64) + dt * np.asarray(ds_dt, dtype=np.float64), 0.0)


def _make_multimodal_v1(preset: str = "default", seed: int = 0, **overrides: Any) -> AnnData:
    """Generate a seeded count-level lineage time series with exact truth.

    ``n_founders`` is the supported size override.  Other documented model
    parameters are accepted as keyword overrides for deterministic validity
    checks; default preset values are the fixed S01 design.
    """
    config = _make_config(preset, seed, overrides)
    rng = np.random.default_rng(config.seed)
    genes = _gene_parameters(config, rng)
    W, b = genes["W"], genes["b"]
    beta_initial, beta_final, gamma = genes["beta_initial"], genes["beta_final"], genes["gamma"]
    phase = rng.uniform(0.0, 2.0 * np.pi, size=config.n_founders)
    radius = rng.normal(1.0, 0.05, size=config.n_founders).astype(np.float32)
    z = np.zeros((config.n_founders, D), dtype=np.float32)
    z[:, 0] = rng.normal(0.0, 0.05, size=config.n_founders)
    z[:, 1] = 0.0 if config.stationary_cycle else rng.normal(0.0, 0.1, size=config.n_founders)
    z[:, 2] = radius * np.cos(phase)
    z[:, 3] = radius * np.sin(phase)
    z[:, 4:] = rng.normal(0.0, 0.3, size=(config.n_founders, 2))
    alpha0 = _alpha(z, W, b)
    u0 = alpha0 / beta_initial[None, :]
    s0 = alpha0 / gamma[None, :]
    division_rate = log(2.0) / config.division_hours
    live: dict[str, np.ndarray] = {
        "z": z, "u": u0.astype(np.float32), "s": s0.astype(np.float32),
        "u_new": np.zeros_like(u0, dtype=np.float32), "s_new": np.zeros_like(s0, dtype=np.float32),
        "clone": np.arange(config.n_founders, dtype=np.int32),
        "cell_id": np.arange(config.n_founders, dtype=np.int64),
        "next_div": rng.exponential(1.0 / division_rate, size=config.n_founders).astype(np.float32),
    }
    next_cell_id = int(config.n_founders)
    time_to_day = {float(t): day for day, t in enumerate(config.sampling_times, start=1)}
    pulse_for_time = {float(t): float(config.pulse_hours[(day - 1) % len(config.pulse_hours)])
                      for day, t in enumerate(config.sampling_times, start=1)}
    pulse_starts = {round(t - pulse_for_time[t], 8) for t in config.sampling_times}
    horizon = float(max(config.sampling_times))
    n_steps = int(round(horizon / config.dt))
    sqrt_dt = float(np.sqrt(config.dt))
    batch_factors = _batch_gene_effects(config, rng)
    collected: dict[str, list[np.ndarray]] = {key: [] for key in (
        "z", "u", "s", "u_new", "s_new", "s_prev", "clone", "day", "time", "pulse", "capture", "batch_index", "cell_id",
    )}
    collected_counts: list[int] = []
    beta_regime: list[int] = []
    for step in range(1, n_steps + 1):
        t_prev = round((step - 1) * config.dt, 8)
        t = round(step * config.dt, 8)
        if t_prev in pulse_starts:
            live["u_new"].fill(0.0)
            live["s_new"].fill(0.0)
        beta = beta_final if (config.stress_beta_fraction and t_prev >= horizon / 2.0) else beta_initial
        alpha = _alpha(live["z"], W, b)
        s_before = live["s"].copy()
        live["u"], live["s"] = _rna_step(live["u"], live["s"], alpha, beta, gamma, config.dt)
        live["u_new"], live["s_new"] = _rna_step(live["u_new"], live["s_new"], alpha, beta, gamma, config.dt)
        drift = _state_drift(live["z"], config.stationary_cycle)
        noise = config.sigma * sqrt_dt * rng.standard_normal(live["z"].shape).astype(np.float32)
        if config.stationary_cycle:
            noise[:, 1] = 0.0
        live["z"] = (live["z"] + config.dt * drift + noise).astype(np.float32)
        if config.stationary_cycle:
            live["z"][:, 1] = 0.0
        divide = live["next_div"] <= t
        n_before = len(live["z"])
        _append_daughters(live, divide, t, division_rate, rng)
        if len(live["z"]) > n_before:
            daughter_from = np.flatnonzero(divide)
            s_before = np.concatenate((s_before, s_before[daughter_from].copy()), axis=0)
            new_count = len(live["z"]) - n_before
            live["cell_id"][-new_count:] = np.arange(next_cell_id, next_cell_id + new_count, dtype=np.int64)
            next_cell_id += new_count
        if t not in time_to_day:
            continue
        day = time_to_day[t]
        n_take = int(round(config.f_sample * len(live["z"])))
        if n_take < 1:
            raise RuntimeError("sampling fraction produced no collected cells")
        take = np.sort(rng.choice(len(live["z"]), size=n_take, replace=False))
        capture = rng.lognormal(log(0.1), 0.3, size=n_take).astype(np.float32)
        for key, source in (("z", "z"), ("u", "u"), ("s", "s"), ("u_new", "u_new"), ("s_new", "s_new"), ("clone", "clone"), ("cell_id", "cell_id")):
            collected[key].append(live[source][take].copy())
        collected["s_prev"].append(s_before[take].copy())
        collected["day"].append(np.full(n_take, day, dtype=np.int16))
        collected["time"].append(np.full(n_take, t, dtype=np.float32))
        collected["pulse"].append(np.full(n_take, pulse_for_time[t], dtype=np.float32))
        collected["capture"].append(capture)
        collected["batch_index"].append(np.full(n_take, day - 1, dtype=np.int16))
        collected_counts.append(n_take)
        beta_regime.extend([int(config.stress_beta_fraction and t >= horizon / 2.0)] * n_take)
        _remove_indices(live, take)
    if not collected_counts:
        raise RuntimeError("no samples collected")
    sampled = {key: np.concatenate(values, axis=0) for key, values in collected.items()}
    beta_regime_arr = np.asarray(beta_regime, dtype=np.int8)
    beta_per_cell = np.where(beta_regime_arr[:, None].astype(bool), beta_final[None, :], beta_initial[None, :])
    alpha_final = _alpha(sampled["z"], W, b)
    ds_dt = beta_per_cell * sampled["u"] - gamma[None, :] * sampled["s"]
    jac = _state_jacobian(sampled["z"], config.stationary_cycle)
    omega_state = 0.5 * (jac - np.swapaxes(jac, 1, 2))
    committed = sampled["z"][:, 1] > (Z1_CRIT + 0.2)
    fate = np.where(committed, np.where(sampled["z"][:, 0] >= 0.0, "positive", "negative"), "progenitor")
    gene_effect = batch_factors[sampled["batch_index"]]
    capture_scale = sampled["capture"][:, None] * gene_effect
    if config.poisson:
        spliced = rng.poisson(np.maximum(capture_scale * sampled["s"], 0.0)).astype(np.int32)
        unspliced = rng.poisson(np.maximum(capture_scale * ETA_U * sampled["u"], 0.0)).astype(np.int32)
    else:
        spliced = (capture_scale * sampled["s"]).astype(np.float32)
        unspliced = (capture_scale * ETA_U * sampled["u"]).astype(np.float32)
    total = spliced + unspliced
    fraction_new = np.divide(sampled["u_new"] + sampled["s_new"], sampled["u"] + sampled["s"],
                             out=np.zeros_like(sampled["s"], dtype=np.float32), where=(sampled["u"] + sampled["s"]) > 0)
    fraction_new = np.clip(fraction_new, 0.0, 1.0)
    if config.poisson:
        new_before_efficiency = rng.binomial(total.astype(np.int64), fraction_new).astype(np.int32)
        new = rng.binomial(new_before_efficiency, config.labelling_efficiency).astype(np.int32)
    else:
        new = (config.labelling_efficiency * total * fraction_new).astype(np.float32)
    clone_ids = np.asarray([f"clone_{i:04d}" for i in range(config.n_founders)], dtype=object)
    detected = rng.random(len(sampled["clone"])) < config.p_barcode
    clone_values = np.where(detected, clone_ids[sampled["clone"]], None)
    obs = pd.DataFrame(index=[f"cell_{int(i):07d}" for i in sampled["cell_id"]])
    obs["day"] = sampled["day"].astype(np.int16)
    obs["time_hours"] = sampled["time"].astype(np.float32)
    obs["pulse_hours"] = sampled["pulse"].astype(np.float32)
    obs["clone"] = pd.Categorical(clone_values, categories=list(clone_ids))
    obs["clone_true"] = pd.Categorical(clone_ids[sampled["clone"]], categories=list(clone_ids))
    obs["fate_true"] = pd.Categorical(fate, categories=["progenitor", "negative", "positive"])
    obs["capture_factor"] = sampled["capture"].astype(np.float32)
    obs["batch_day"] = sampled["batch_index"].astype(np.int16) + 1
    obs["beta_regime"] = beta_regime_arr
    var = pd.DataFrame(index=[f"gene_{i:04d}" for i in range(config.n_genes)])
    var["transcription_intercept"] = b
    var["beta_initial"] = beta_initial
    var["beta_final"] = beta_final
    var["gamma"] = gamma
    var["assumption_breaking_kinetics"] = genes["beta_changed"]
    adata = AnnData(X=spliced.copy(), obs=obs, var=var)
    adata.layers["spliced"] = spliced.copy()
    adata.layers["unspliced"] = unspliced
    adata.layers["total"] = total
    adata.layers["new"] = new
    adata.layers["u_true"] = sampled["u"].astype(np.float32)
    adata.layers["s_true"] = sampled["s"].astype(np.float32)
    adata.layers["s_prev_true"] = sampled["s_prev"].astype(np.float32)
    adata.layers["u_new_true"] = sampled["u_new"].astype(np.float32)
    adata.layers["s_new_true"] = sampled["s_new"].astype(np.float32)
    adata.layers["new_fraction_true"] = fraction_new.astype(np.float32)
    adata.layers["ds_dt_true"] = ds_dt.astype(np.float32)
    adata.obsm["z_true"] = sampled["z"].astype(np.float32)
    adata.obsm["v_state"] = _state_drift(sampled["z"], config.stationary_cycle).astype(np.float32)
    adata.obsm["jacobian_state"] = jac.astype(np.float32)
    adata.obsm["omega_state"] = omega_state.astype(np.float32)
    rows = np.flatnonzero(detected)
    clone_matrix = sparse.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, sampled["clone"][rows])), shape=(adata.n_obs, config.n_founders))
    adata.obsm["clone_matrix"] = clone_matrix
    adata.varm["W"] = W
    earliest = np.flatnonzero(obs["day"].to_numpy() == int(obs["day"].min()))
    start = earliest[int(np.argmin(np.abs(sampled["z"][earliest, 1])))]
    adata.uns["suggested_start_cell"] = str(adata.obs_names[start])
    adata.uns["clone_ids"] = clone_ids.astype(str)
    adata.uns["batch_gene_effects"] = batch_factors
    adata.uns["synthetic_multimodal"] = {
        **asdict(config),
        "sampling_times": np.asarray(config.sampling_times, dtype=np.float32),
        "pulse_hours": np.asarray(config.pulse_hours, dtype=np.float32),
        "n_collected_by_day": np.asarray(collected_counts, dtype=np.int32),
        "n_living_after_final_collection": int(len(live["z"])),
        "n_clones_detected": int(len(np.unique(sampled["clone"][detected]))),
        "assumption_breaking_gene_indices": np.flatnonzero(genes["beta_changed"]).astype(np.int32),
        "capture_eta_unspliced": float(ETA_U),
        "omega": float(OMEGA),
    }
    return adata


def _normalised_spliced(adata: AnnData) -> tuple[np.ndarray, np.ndarray, float]:
    spliced = np.asarray(adata.layers["spliced"], dtype=np.float64)
    library = spliced.sum(axis=1)
    positive = library[library > 0]
    if len(positive) == 0:
        raise ValueError("spliced counts have zero library size for every cell")
    target = float(np.median(positive))
    factors = target / np.maximum(library, 1e-12)
    return np.log1p(spliced * factors[:, None]), factors.astype(np.float64), target


def embed_linear(adata: AnnData, method: str = "pca", n_comp: int = 20) -> AnnData:
    """Fit PCA or factor analysis to normalised/log1p observed spliced counts."""
    if method not in {"pca", "fa"}:
        raise ValueError("method must be 'pca' or 'fa'")
    transformed, factors, target = _normalised_spliced(adata)
    n_comp = min(int(n_comp), adata.n_obs - 1, adata.n_vars)
    if n_comp < 1:
        raise ValueError("not enough cells/features for a linear embedding")
    key = f"embedding_{method}"
    if method == "pca":
        model = PCA(n_components=n_comp, svd_solver="full", random_state=0)
        scores = model.fit_transform(transformed)
        mapping = model.components_.T
        meta = {"mean": model.mean_.astype(np.float32), "loadings": model.components_.T.astype(np.float32),
                "linear_map": mapping.astype(np.float32)}
    else:
        model = FactorAnalysis(n_components=n_comp, random_state=0, max_iter=100, tol=1e-4)
        model.fit(transformed)
        W = model.components_.T
        psi = np.maximum(model.noise_variance_, 1e-8)
        inv_psi_W = W / psi[:, None]
        posterior_map = inv_psi_W @ np.linalg.inv(np.eye(n_comp) + W.T @ inv_psi_W)
        scores = (transformed - model.mean_[None, :]) @ posterior_map
        meta = {"mean": model.mean_.astype(np.float32), "loadings": W.astype(np.float32),
                "noise_variance": psi.astype(np.float32), "posterior_map": posterior_map.astype(np.float32),
                "linear_map": posterior_map.astype(np.float32)}
    meta.update({"normalization_factors": factors.astype(np.float32), "normalization_target": float(target),
                 "input_layer": "spliced", "n_comp": int(n_comp)})
    adata.obsm[f"X_{method}"] = scores.astype(np.float32)
    adata.uns[key] = meta
    return adata


def true_velocity_in_embedding(adata: AnnData, method: str, at: str = "observed") -> np.ndarray:
    """Map the exact spliced expression velocity through preprocessing and the
    embedding's linear map.

    ``at="observed"`` evaluates the chain rule at the observed (Poisson)
    spliced counts and stores ``obsm["velocity_true_<method>"]``.
    ``at="expected"`` evaluates it at the noise-free expected counts
    (capture × ``s_true`` × batch effect), so the truth does not carry
    sampling noise, and stores ``obsm["velocity_true_expected_<method>"]``.
    Score orientation against the expected truth: at observed counts the
    sign differs on a few percent of (cell, axis) pairs (review orient/S01).
    """
    if at not in {"observed", "expected"}:
        raise ValueError("at must be 'observed' or 'expected'")
    key = f"embedding_{method}"
    if key not in adata.uns or f"X_{method}" not in adata.obsm:
        raise ValueError(f"run embed_linear(..., method={method!r}) first")
    meta = adata.uns[key]
    ds_dt = np.asarray(adata.layers["ds_dt_true"], dtype=np.float64)
    capture = adata.obs["capture_factor"].to_numpy(dtype=np.float64)
    day = adata.obs["batch_day"].to_numpy(dtype=int) - 1
    batch_effects = np.asarray(adata.uns["batch_gene_effects"], dtype=np.float64)
    gene_factor = capture[:, None] * batch_effects[day]
    observed_velocity = ds_dt * gene_factor
    if at == "observed":
        spliced = np.asarray(adata.layers["spliced"], dtype=np.float64)
    else:
        spliced = np.asarray(adata.layers["s_true"], dtype=np.float64) * gene_factor
    library = spliced.sum(axis=1)
    target = float(meta["normalization_target"])
    scaled = spliced * target / np.maximum(library[:, None], 1e-12)
    dlibrary = observed_velocity.sum(axis=1)
    dscaled = target * (observed_velocity / np.maximum(library[:, None], 1e-12) - spliced * dlibrary[:, None] / np.maximum(library[:, None] ** 2, 1e-12))
    dlog = dscaled / (1.0 + scaled)
    output = (dlog @ np.asarray(meta["linear_map"], dtype=np.float64)).astype(np.float32)
    out_key = f"velocity_true_{method}" if at == "observed" else f"velocity_true_expected_{method}"
    adata.obsm[out_key] = output
    return output


def finite_difference_embedding_velocity(adata: AnnData, method: str, dt: float | None = None) -> np.ndarray:
    """One-step deterministic embedding finite difference used by validity tests."""
    key = f"embedding_{method}"
    meta = adata.uns[key]
    dt = float(dt if dt is not None else adata.uns["synthetic_multimodal"]["dt"])
    spliced = np.asarray(adata.layers["spliced"], dtype=np.float64)
    ds_dt = np.asarray(adata.layers["ds_dt_true"], dtype=np.float64)
    capture = adata.obs["capture_factor"].to_numpy(dtype=np.float64)
    day = adata.obs["batch_day"].to_numpy(dtype=int) - 1
    batch_effects = np.asarray(adata.uns["batch_gene_effects"], dtype=np.float64)
    observed_velocity = ds_dt * capture[:, None] * batch_effects[day]
    next_spliced = _one_step_expression(spliced, observed_velocity, dt)
    target = float(meta["normalization_target"])
    current = np.log1p(spliced * target / np.maximum(spliced.sum(axis=1, keepdims=True), 1e-12))
    future = np.log1p(next_spliced * target / np.maximum(next_spliced.sum(axis=1, keepdims=True), 1e-12))
    linear_map = np.asarray(meta["linear_map"], dtype=np.float64)
    return ((future - current) @ linear_map / dt).astype(np.float32)

# ---------------------------------------------------------------------------
# Generator v2
# ---------------------------------------------------------------------------
# The original S01 implementation above is deliberately retained as a separate
# v1 function.  Its config, RNG draws, and AnnData serialization therefore stay
# byte-for-byte stable.  V2 has a separate config and two-pass implementation:
# latent genealogy/sampling first, then RNA pools only for ancestral paths that
# terminate in a collected cell.

V2_PRESETS = ("v2_default", "v2_stress", "v2_batch", "v2_stationary_cycle")


@dataclass(frozen=True)
class _V2Config:
    preset: str
    seed: int
    n_founders: int = 5000
    n_genes: int = DEFAULT_GENES
    dt: float = DEFAULT_DT
    sigma: float = 0.05
    poisson: bool = True
    division_hours: float = 16.0
    f_sample: float = 0.01
    sampling_times: tuple[float, ...] = (24.0, 48.0, 72.0, 96.0)
    p_barcode: float = 0.60
    labelling_efficiency: float = 0.90
    pulse_hours: tuple[float, ...] = (2.0,)
    stress_beta_fraction: float = 0.0
    batch_effect: bool = False
    stationary_cycle: bool = False
    capture_eta_unspliced: float = 0.60


def _make_v2_config(preset: str, seed: int, overrides: dict[str, Any]) -> _V2Config:
    if preset not in V2_PRESETS:
        raise ValueError(f"unknown v2 preset {preset!r}")
    values: dict[str, Any] = {"preset": preset, "seed": int(seed)}
    if preset == "v2_stress":
        values.update(p_barcode=0.30, labelling_efficiency=0.60,
                      pulse_hours=(1.0, 2.0, 4.0), stress_beta_fraction=0.20)
    elif preset == "v2_batch":
        values.update(batch_effect=True)
    elif preset == "v2_stationary_cycle":
        values.update(stationary_cycle=True, sampling_times=(24.0,), n_founders=5000,
                      f_sample=0.20)
    legal = set(_V2Config.__dataclass_fields__)
    unknown = sorted(set(overrides) - legal)
    if unknown:
        raise TypeError(f"unknown make_multimodal overrides: {unknown}")
    values.update(overrides)
    config = _V2Config(**values)
    if config.n_founders < 1 or config.n_genes < 2:
        raise ValueError("n_founders must be positive and n_genes must be at least 2")
    if config.dt <= 0 or config.division_hours <= 0 or not (0 < config.f_sample <= 1):
        raise ValueError("dt, division_hours, and f_sample must be positive")
    if not config.sampling_times or any(t <= 0 for t in config.sampling_times):
        raise ValueError("sampling_times must be positive and nonempty")
    if any(abs(t / config.dt - round(t / config.dt)) > 1e-8 for t in config.sampling_times):
        raise ValueError("sampling_times must lie on the fixed dt grid")
    if any(p <= 0 or p > min(config.sampling_times) for p in config.pulse_hours):
        raise ValueError("each pulse duration must be positive and precede collection")
    return config


def _gene_parameters_v2(config: _V2Config, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """V2 kinetic, switching, and overdispersion parameters from the gene RNG."""
    n_loadings = rng.integers(1, 4, size=config.n_genes)
    W = np.zeros((config.n_genes, D), dtype=np.float32)
    for gene, k in enumerate(n_loadings):
        dims = rng.choice(D, size=int(k), replace=False)
        W[gene, dims] = rng.normal(0.0, 1.0, size=int(k)).astype(np.float32)
    b = rng.normal(log(2.0), 0.7, size=config.n_genes).astype(np.float32)
    beta_initial = rng.lognormal(log(0.6), 0.5, size=config.n_genes).astype(np.float32)
    gamma = rng.lognormal(log(0.2), 0.6, size=config.n_genes).astype(np.float32)

    kind = np.full(config.n_genes, "linear", dtype=object)
    theta = np.full(config.n_genes, np.nan, dtype=np.float32)
    delta = np.full(config.n_genes, np.nan, dtype=np.float32)
    fate_sign = np.full(config.n_genes, np.nan, dtype=np.float32)
    # The prescribed benchmark has exactly 800 genes and therefore exactly
    # 80 on, 80 on_off, and 80 fate genes.  Small test-only overrides retain
    # equal classes by nearest whole-class allocation.
    n_each = 80 if config.n_genes == DEFAULT_GENES else max(1, int(round(0.30 * config.n_genes / 3.0)))
    picked = rng.choice(config.n_genes, size=3 * n_each, replace=False)
    on, on_off, fate = picked[:n_each], picked[n_each:2 * n_each], picked[2 * n_each:]
    kind[on] = "on"
    kind[on_off] = "on_off"
    kind[fate] = "fate"
    theta[on] = rng.uniform(0.3, 1.6, size=n_each).astype(np.float32)
    theta[on_off] = rng.uniform(0.3, 1.0, size=n_each).astype(np.float32)
    delta[on_off] = rng.uniform(0.4, 0.8, size=n_each).astype(np.float32)
    fate_sign[fate] = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n_each)

    changed = np.zeros(config.n_genes, dtype=bool)
    if config.stress_beta_fraction:
        changed[rng.choice(config.n_genes,
                           size=int(round(config.n_genes * config.stress_beta_fraction)),
                           replace=False)] = True
    beta_final = beta_initial.copy()
    beta_final[changed] *= 2.0
    nb_theta = rng.lognormal(log(20.0), 0.5, size=config.n_genes).astype(np.float32)
    return {
        "W": W, "b": b, "beta_initial": beta_initial, "beta_final": beta_final,
        "gamma": gamma, "beta_changed": changed, "gene_kind": kind,
        "theta": theta, "delta": delta, "fate_sign": fate_sign, "nb_theta": nb_theta,
    }


def _v2_alpha(z: np.ndarray, genes: dict[str, np.ndarray]) -> np.ndarray:
    """Transcription field for linear and the three fixed v2 switching modes."""
    W = genes["W"]
    b = genes["b"]
    out = np.exp(np.asarray(z, dtype=np.float32) @ W.T + b[None, :]).astype(np.float32)
    kind = genes["gene_kind"]
    width = np.float32(0.05)
    lo = np.exp(b - np.float32(1.5)).astype(np.float32)
    hi = np.exp(b + np.float32(1.5)).astype(np.float32)

    def logistic(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -60.0, 60.0)
        return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)

    on = kind == "on"
    if np.any(on):
        h = logistic((z[:, 1, None] - genes["theta"][None, on]) / width)
        out[:, on] = lo[None, on] + (hi[None, on] - lo[None, on]) * h
    on_off = kind == "on_off"
    if np.any(on_off):
        th = genes["theta"][on_off]
        de = genes["delta"][on_off]
        h = logistic((z[:, 1, None] - th[None, :]) / width)
        h *= 1.0 - logistic((z[:, 1, None] - th[None, :] - de[None, :]) / width)
        out[:, on_off] = lo[None, on_off] + (hi[None, on_off] - lo[None, on_off]) * h
    fate = kind == "fate"
    if np.any(fate):
        sign = genes["fate_sign"][fate]
        h = logistic((z[:, 0, None] * sign[None, :]) / width)
        h *= logistic((z[:, 1, None] - np.float32(1.2)) / width)
        out[:, fate] = lo[None, fate] + (hi[None, fate] - lo[None, fate]) * h
    return out


def _initial_latent_population(config: _V2Config, rng: np.random.Generator) -> tuple[dict[str, np.ndarray], int]:
    phase = rng.uniform(0.0, 2.0 * np.pi, size=config.n_founders)
    radius = rng.normal(1.0, 0.05, size=config.n_founders).astype(np.float32)
    z = np.zeros((config.n_founders, D), dtype=np.float32)
    z[:, 0] = rng.normal(0.0, 0.05, size=config.n_founders)
    z[:, 1] = 0.0 if config.stationary_cycle else rng.normal(0.0, 0.1, size=config.n_founders)
    z[:, 2] = radius * np.cos(phase)
    z[:, 3] = radius * np.sin(phase)
    z[:, 4:] = rng.normal(0.0, 0.3, size=(config.n_founders, 2))
    division_rate = log(2.0) / config.division_hours
    live = {
        "z": z,
        "clone": np.arange(config.n_founders, dtype=np.int32),
        "cell_id": np.arange(config.n_founders, dtype=np.int64),
        "next_div": rng.exponential(1.0 / division_rate, size=config.n_founders).astype(np.float32),
    }
    return live, config.n_founders


def _latent_divide(live: dict[str, np.ndarray], divide: np.ndarray, time: float,
                   division_rate: float, rng: np.random.Generator,
                   next_cell_id: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Append daughter latent cells in the same order and RNG draw pattern as v1."""
    idx = np.flatnonzero(divide)
    if len(idx) == 0:
        return idx, np.empty(0, dtype=np.int64), next_cell_id
    parent_ids = live["cell_id"][idx].copy()
    live["z"] = np.concatenate((live["z"], live["z"][idx].copy()), axis=0)
    live["clone"] = np.concatenate((live["clone"], live["clone"][idx].copy()), axis=0)
    live["cell_id"] = np.concatenate((live["cell_id"],
                                        np.arange(next_cell_id, next_cell_id + len(idx), dtype=np.int64)))
    live["next_div"][idx] = time + rng.exponential(1.0 / division_rate, size=len(idx))
    daughters = time + rng.exponential(1.0 / division_rate, size=len(idx))
    live["next_div"] = np.concatenate((live["next_div"], daughters.astype(np.float32)))
    return idx, parent_ids, next_cell_id + len(idx)


def _remove_latent_indices(live: dict[str, np.ndarray], indices: np.ndarray) -> None:
    keep = np.ones(len(live["z"]), dtype=bool)
    keep[indices] = False
    for key in tuple(live):
        live[key] = live[key][keep]


def _v2_schedule(config: _V2Config) -> tuple[dict[float, int], dict[float, float], set[float]]:
    time_to_day = {float(t): day for day, t in enumerate(config.sampling_times, start=1)}
    pulse_for_time = {float(t): float(config.pulse_hours[(day - 1) % len(config.pulse_hours)])
                      for day, t in enumerate(config.sampling_times, start=1)}
    pulse_starts = {round(t - pulse_for_time[t], 8) for t in config.sampling_times}
    return time_to_day, pulse_for_time, pulse_starts


def _simulate_v2_latent(config: _V2Config, rng: np.random.Generator) -> dict[str, Any]:
    """First pass: exact latent genealogy and destructive collections, no RNA pools."""
    live, next_cell_id = _initial_latent_population(config, rng)
    batch_factors = _batch_gene_effects(config, rng)
    division_rate = log(2.0) / config.division_hours
    time_to_day, pulse_for_time, _ = _v2_schedule(config)
    horizon = float(max(config.sampling_times))
    sqrt_dt = float(np.sqrt(config.dt))
    n_steps = int(round(horizon / config.dt))
    parent: list[int] = [-1] * config.n_founders
    collected: dict[str, list[np.ndarray]] = {key: [] for key in (
        "z", "clone", "day", "time", "pulse", "capture", "batch_index", "cell_id",
    )}
    collected_counts: list[int] = []
    beta_regime: list[np.ndarray] = []
    for step in range(1, n_steps + 1):
        t = round(step * config.dt, 8)
        drift = _state_drift(live["z"], config.stationary_cycle)
        noise = config.sigma * sqrt_dt * rng.standard_normal(live["z"].shape).astype(np.float32)
        if config.stationary_cycle:
            noise[:, 1] = 0.0
        live["z"] = (live["z"] + config.dt * drift + noise).astype(np.float32)
        if config.stationary_cycle:
            live["z"][:, 1] = 0.0
        divide = live["next_div"] <= t
        _, parent_ids, next_cell_id = _latent_divide(live, divide, t, division_rate, rng, next_cell_id)
        parent.extend(int(v) for v in parent_ids)
        if t not in time_to_day:
            continue
        day = time_to_day[t]
        n_take = int(round(config.f_sample * len(live["z"])))
        if n_take < 1:
            raise RuntimeError("sampling fraction produced no collected cells")
        take = np.sort(rng.choice(len(live["z"]), size=n_take, replace=False))
        capture = rng.lognormal(log(0.1), 0.3, size=n_take).astype(np.float32)
        for key in ("z", "clone", "cell_id"):
            collected[key].append(live[key][take].copy())
        collected["day"].append(np.full(n_take, day, dtype=np.int16))
        collected["time"].append(np.full(n_take, t, dtype=np.float32))
        collected["pulse"].append(np.full(n_take, pulse_for_time[t], dtype=np.float32))
        collected["capture"].append(capture)
        collected["batch_index"].append(np.full(n_take, day - 1, dtype=np.int16))
        collected_counts.append(n_take)
        beta_regime.append(np.full(n_take, int(config.stress_beta_fraction and t >= horizon / 2.0), dtype=np.int8))
        _remove_latent_indices(live, take)
    if not collected_counts:
        raise RuntimeError("no samples collected")
    sampled = {key: np.concatenate(values, axis=0) for key, values in collected.items()}
    return {
        "sampled": sampled,
        "parent": np.asarray(parent, dtype=np.int64),
        "n_collected_by_day": np.asarray(collected_counts, dtype=np.int32),
        "beta_regime": np.concatenate(beta_regime),
        "n_living_after_final_collection": int(len(live["z"])),
        "batch_factors": batch_factors,
    }


def _needed_gene_cells(parent: np.ndarray, sampled_ids: np.ndarray, eager: bool) -> np.ndarray:
    if eager:
        return np.ones(len(parent), dtype=bool)
    needed = np.zeros(len(parent), dtype=bool)
    for cell in np.asarray(sampled_ids, dtype=np.int64):
        current = int(cell)
        while current >= 0 and not needed[current]:
            needed[current] = True
            current = int(parent[current])
    return needed


def _replay_v2_rna(config: _V2Config, first: dict[str, Any], integration: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Replay latent RNG exactly and integrate pools only for selected ancestry."""
    if integration not in {"lazy", "eager"}:
        raise ValueError("integration must be 'lazy' or 'eager'")
    sampled = first["sampled"]
    parent = np.asarray(first["parent"], dtype=np.int64)
    needed = _needed_gene_cells(parent, sampled["cell_id"], eager=(integration == "eager"))
    slots = np.full(len(parent), -1, dtype=np.int64)
    slots[np.flatnonzero(needed)] = np.arange(int(needed.sum()), dtype=np.int64)
    n_slots = int(needed.sum())

    rng = np.random.default_rng(config.seed)
    genes = _gene_parameters_v2(config, rng)
    live, next_cell_id = _initial_latent_population(config, rng)
    batch_factors = _batch_gene_effects(config, rng)
    if not np.array_equal(batch_factors, first["batch_factors"]):
        raise RuntimeError("v2 latent replay changed batch-factor RNG state")
    division_rate = log(2.0) / config.division_hours
    time_to_day, pulse_for_time, pulse_starts = _v2_schedule(config)
    horizon = float(max(config.sampling_times))
    sqrt_dt = float(np.sqrt(config.dt))
    n_steps = int(round(horizon / config.dt))

    pools = {name: np.zeros((n_slots, config.n_genes), dtype=np.float32)
             for name in ("u", "s", "u_new", "s_new")}
    previous_s = np.zeros((n_slots, config.n_genes), dtype=np.float32)
    founder_slots = slots[live["cell_id"]]
    alpha0 = _v2_alpha(live["z"], genes)
    pools["u"][founder_slots] = alpha0 / genes["beta_initial"][None, :]
    pools["s"][founder_slots] = alpha0 / genes["gamma"][None, :]
    replay: dict[str, list[np.ndarray]] = {key: [] for key in ("u", "s", "u_new", "s_new", "s_prev")}
    offset = 0

    for step in range(1, n_steps + 1):
        t_prev = round((step - 1) * config.dt, 8)
        t = round(step * config.dt, 8)
        live_slots = slots[live["cell_id"]]
        active_pos = np.flatnonzero(live_slots >= 0)
        active_slots = live_slots[active_pos]
        if len(active_slots):
            if t_prev in pulse_starts:
                pools["u_new"][active_slots] = 0.0
                pools["s_new"][active_slots] = 0.0
            previous_s[active_slots] = pools["s"][active_slots]
            beta = genes["beta_final"] if (config.stress_beta_fraction and t_prev >= horizon / 2.0) else genes["beta_initial"]
            alpha = _v2_alpha(live["z"][active_pos], genes)
            pools["u"][active_slots], pools["s"][active_slots] = _rna_step(
                pools["u"][active_slots], pools["s"][active_slots], alpha, beta, genes["gamma"], config.dt)
            pools["u_new"][active_slots], pools["s_new"][active_slots] = _rna_step(
                pools["u_new"][active_slots], pools["s_new"][active_slots], alpha, beta, genes["gamma"], config.dt)

        drift = _state_drift(live["z"], config.stationary_cycle)
        noise = config.sigma * sqrt_dt * rng.standard_normal(live["z"].shape).astype(np.float32)
        if config.stationary_cycle:
            noise[:, 1] = 0.0
        live["z"] = (live["z"] + config.dt * drift + noise).astype(np.float32)
        if config.stationary_cycle:
            live["z"][:, 1] = 0.0
        divide = live["next_div"] <= t
        _, parent_ids, next_cell_id = _latent_divide(live, divide, t, division_rate, rng, next_cell_id)
        if len(parent_ids):
            child_ids = np.arange(next_cell_id - len(parent_ids), next_cell_id, dtype=np.int64)
            parent_slots = slots[parent_ids]
            child_slots = slots[child_ids]
            child_needed = child_slots >= 0
            if np.any(child_needed):
                if np.any(parent_slots[child_needed] < 0):
                    raise RuntimeError("needed descendant lacks a needed ancestor")
                for name in pools:
                    pools[name][child_slots[child_needed]] = pools[name][parent_slots[child_needed]]
                previous_s[child_slots[child_needed]] = previous_s[parent_slots[child_needed]]

        if t not in time_to_day:
            continue
        n_take = int(round(config.f_sample * len(live["z"])))
        take = np.sort(rng.choice(len(live["z"]), size=n_take, replace=False))
        capture = rng.lognormal(log(0.1), 0.3, size=n_take).astype(np.float32)
        expected_ids = sampled["cell_id"][offset:offset + n_take]
        if not np.array_equal(live["cell_id"][take], expected_ids):
            raise RuntimeError("v2 RNA replay changed sampled cell identities")
        if not np.array_equal(capture, sampled["capture"][offset:offset + n_take]):
            raise RuntimeError("v2 RNA replay changed capture-factor RNG state")
        take_slots = slots[live["cell_id"][take]]
        if np.any(take_slots < 0):
            raise RuntimeError("a collected cell was omitted from lazy ancestry")
        for name in ("u", "s", "u_new", "s_new"):
            replay[name].append(pools[name][take_slots].copy())
        replay["s_prev"].append(previous_s[take_slots].copy())
        offset += n_take
        _remove_latent_indices(live, take)
    if offset != len(sampled["cell_id"]):
        raise RuntimeError("v2 RNA replay did not visit every collection")
    return {key: np.concatenate(values, axis=0) for key, values in replay.items()}, genes


def _make_multimodal_v2(config: _V2Config, integration: str = "lazy") -> AnnData:
    """Generate a v2 dataset using two latent-identical passes and lazy pools."""
    # First pass begins immediately after v2 parameter construction; gene pools
    # consume no draws from this latent/sampling RNG stream.
    rng = np.random.default_rng(config.seed)
    _gene_parameters_v2(config, rng)
    # `_simulate_v2_latent` starts from the post-gene RNG state and constructs the
    # initial population itself; use this exact stream rather than gene pools.
    first = _simulate_v2_latent(config, rng)
    pools, genes = _replay_v2_rna(config, first, integration=integration)
    sampled = first["sampled"]
    beta_regime = np.asarray(first["beta_regime"], dtype=np.int8)
    beta_per_cell = np.where(beta_regime[:, None].astype(bool), genes["beta_final"][None, :], genes["beta_initial"][None, :])
    ds_dt = beta_per_cell * pools["u"] - genes["gamma"][None, :] * pools["s"]
    jac = _state_jacobian(sampled["z"], config.stationary_cycle)
    omega_state = 0.5 * (jac - np.swapaxes(jac, 1, 2))
    committed = sampled["z"][:, 1] > (Z1_CRIT + 0.2)
    fate = np.where(committed, np.where(sampled["z"][:, 0] >= 0.0, "positive", "negative"), "progenitor")
    gene_effect = first["batch_factors"][sampled["batch_index"]]
    capture_scale = sampled["capture"][:, None] * gene_effect

    measurement_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 0x534F35]))
    if config.poisson:
        theta = genes["nb_theta"]
        multiplier = measurement_rng.gamma(theta[None, :], (1.0 / theta)[None, :],
                                           size=pools["s"].shape).astype(np.float32)
        spliced = measurement_rng.poisson(np.maximum(multiplier * capture_scale * pools["s"], 0.0)).astype(np.int32)
        unspliced = measurement_rng.poisson(np.maximum(multiplier * capture_scale * config.capture_eta_unspliced * pools["u"], 0.0)).astype(np.int32)
    else:
        spliced = (capture_scale * pools["s"]).astype(np.float32)
        unspliced = (capture_scale * config.capture_eta_unspliced * pools["u"]).astype(np.float32)
    total = spliced + unspliced
    fraction_new = np.divide(pools["u_new"] + pools["s_new"], pools["u"] + pools["s"],
                             out=np.zeros_like(pools["s"], dtype=np.float32), where=(pools["u"] + pools["s"]) > 0)
    fraction_new = np.clip(fraction_new, 0.0, 1.0)
    if config.poisson:
        new_before_efficiency = measurement_rng.binomial(total.astype(np.int64), fraction_new).astype(np.int32)
        new = measurement_rng.binomial(new_before_efficiency, config.labelling_efficiency).astype(np.int32)
    else:
        new = (config.labelling_efficiency * total * fraction_new).astype(np.float32)

    clone_ids = np.asarray([f"clone_{i:04d}" for i in range(config.n_founders)], dtype=object)
    detected = measurement_rng.random(len(sampled["clone"])) < config.p_barcode
    clone_values = np.where(detected, clone_ids[sampled["clone"]], None)
    obs = pd.DataFrame(index=[f"cell_{int(i):07d}" for i in sampled["cell_id"]])
    obs["day"] = sampled["day"].astype(np.int16)
    obs["time_hours"] = sampled["time"].astype(np.float32)
    obs["pulse_hours"] = sampled["pulse"].astype(np.float32)
    obs["clone"] = pd.Categorical(clone_values, categories=list(clone_ids))
    obs["clone_true"] = pd.Categorical(clone_ids[sampled["clone"]], categories=list(clone_ids))
    obs["fate_true"] = pd.Categorical(fate, categories=["progenitor", "negative", "positive"])
    obs["capture_factor"] = sampled["capture"].astype(np.float32)
    obs["batch_day"] = sampled["batch_index"].astype(np.int16) + 1
    obs["beta_regime"] = beta_regime
    var = pd.DataFrame(index=[f"gene_{i:04d}" for i in range(config.n_genes)])
    var["transcription_intercept"] = genes["b"]
    var["beta_initial"] = genes["beta_initial"]
    var["beta_final"] = genes["beta_final"]
    var["gamma"] = genes["gamma"]
    var["assumption_breaking_kinetics"] = genes["beta_changed"]
    var["gene_kind"] = pd.Categorical(genes["gene_kind"], categories=["linear", "on", "on_off", "fate"])
    var["theta"] = genes["theta"]
    var["delta"] = genes["delta"]
    var["fate_sign"] = genes["fate_sign"]
    var["nb_theta"] = genes["nb_theta"]
    adata = AnnData(X=spliced.copy(), obs=obs, var=var)
    adata.layers["spliced"] = spliced.copy()
    adata.layers["unspliced"] = unspliced
    adata.layers["total"] = total
    adata.layers["new"] = new
    adata.layers["u_true"] = pools["u"].astype(np.float32)
    adata.layers["s_true"] = pools["s"].astype(np.float32)
    adata.layers["s_prev_true"] = pools["s_prev"].astype(np.float32)
    adata.layers["u_new_true"] = pools["u_new"].astype(np.float32)
    adata.layers["s_new_true"] = pools["s_new"].astype(np.float32)
    adata.layers["new_fraction_true"] = fraction_new.astype(np.float32)
    adata.layers["ds_dt_true"] = ds_dt.astype(np.float32)
    adata.obsm["z_true"] = sampled["z"].astype(np.float32)
    adata.obsm["v_state"] = _state_drift(sampled["z"], config.stationary_cycle).astype(np.float32)
    adata.obsm["jacobian_state"] = jac.astype(np.float32)
    adata.obsm["omega_state"] = omega_state.astype(np.float32)
    rows = np.flatnonzero(detected)
    adata.obsm["clone_matrix"] = sparse.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, sampled["clone"][rows])),
                                                     shape=(adata.n_obs, config.n_founders))
    adata.varm["W"] = genes["W"]
    earliest = np.flatnonzero(obs["day"].to_numpy() == int(obs["day"].min()))
    start = earliest[int(np.argmin(np.abs(sampled["z"][earliest, 1])))]
    adata.uns["suggested_start_cell"] = str(adata.obs_names[start])
    adata.uns["clone_ids"] = clone_ids.astype(str)
    adata.uns["batch_gene_effects"] = first["batch_factors"]
    adata.uns["synthetic_multimodal"] = {
        **asdict(config),
        "version": "v2",
        "sampling_times": np.asarray(config.sampling_times, dtype=np.float32),
        "pulse_hours": np.asarray(config.pulse_hours, dtype=np.float32),
        "n_collected_by_day": np.asarray(first["n_collected_by_day"], dtype=np.int32),
        "n_living_after_final_collection": int(first["n_living_after_final_collection"]),
        "n_clones_detected": int(len(np.unique(sampled["clone"][detected]))),
        "assumption_breaking_gene_indices": np.flatnonzero(genes["beta_changed"]).astype(np.int32),
        "capture_eta_unspliced": float(config.capture_eta_unspliced),
        "omega": float(OMEGA),
        "rna_integration": integration,
        "n_gene_ancestral_paths": int((np.asarray(_needed_gene_cells(first["parent"], sampled["cell_id"], eager=False))).sum()),
    }
    return adata



def forward_developmental_truth(
    adata: AnnData,
    method: str,
    horizon_hours: float = 24.0,
    n_paths: int = 8,
    seed: int = 0,
) -> np.ndarray:
    """Return deterministic expected-count developmental displacement truth.

    This is a scoring-only oracle.  Starting from every cell's stored latent
    and RNA truth, it advances ``n_paths`` SDE continuations for the requested
    horizon, with no division, uses the collection-time beta regime, and maps
    current/future expected spliced pools through the stored linear embedding.
    It never mutates ``adata`` and must not be used as an orientation input.
    """
    if method not in {"pca", "fa"}:
        raise ValueError("method must be 'pca' or 'fa'")
    if not np.isfinite(horizon_hours) or horizon_hours <= 0:
        raise ValueError("horizon_hours must be positive and finite")
    if int(n_paths) != n_paths or int(n_paths) < 1:
        raise ValueError("n_paths must be a positive integer")
    required_layers = {"u_true", "s_true"}
    missing_layers = required_layers - set(adata.layers)
    if missing_layers:
        raise KeyError(f"missing truth layers: {sorted(missing_layers)}")
    required_obs = {"capture_factor", "batch_day", "beta_regime"}
    missing_obs = required_obs - set(adata.obs.columns)
    if missing_obs:
        raise KeyError(f"missing truth metadata: {sorted(missing_obs)}")
    meta_key = f"embedding_{method}"
    if meta_key not in adata.uns:
        raise KeyError(f"missing stored embedding metadata {meta_key!r}")
    if "synthetic_multimodal" not in adata.uns:
        raise KeyError("missing synthetic_multimodal metadata")
    synthetic = adata.uns["synthetic_multimodal"]
    version = str(synthetic.get("version", "v1"))
    dt = float(synthetic["dt"])
    if abs(horizon_hours / dt - round(horizon_hours / dt)) > 1e-8:
        raise ValueError("horizon_hours must lie on the generator dt grid")
    n_steps = int(round(horizon_hours / dt))
    stationary_cycle = bool(synthetic["stationary_cycle"])
    sigma = np.float32(synthetic["sigma"])
    z_all = np.asarray(adata.obsm["z_true"], dtype=np.float32)
    u_all = np.asarray(adata.layers["u_true"], dtype=np.float32)
    s_all = np.asarray(adata.layers["s_true"], dtype=np.float32)
    if z_all.shape[0] != adata.n_obs or u_all.shape != s_all.shape:
        raise ValueError("stored truth arrays have incompatible shapes")
    if u_all.shape[0] != adata.n_obs or u_all.shape[1] != adata.n_vars:
        raise ValueError("stored RNA truth does not match AnnData shape")
    gene_kind = None
    if version == "v2":
        required_var = {"gene_kind", "theta", "delta", "fate_sign"}
        missing_var = required_var - set(adata.var.columns)
        if missing_var:
            raise KeyError(f"v2 truth is missing gene metadata: {sorted(missing_var)}")
        genes: dict[str, np.ndarray] = {
            "W": np.asarray(adata.varm["W"], dtype=np.float32),
            "b": adata.var["transcription_intercept"].to_numpy(dtype=np.float32),
            "beta_initial": adata.var["beta_initial"].to_numpy(dtype=np.float32),
            "beta_final": adata.var["beta_final"].to_numpy(dtype=np.float32),
            "gamma": adata.var["gamma"].to_numpy(dtype=np.float32),
            "gene_kind": adata.var["gene_kind"].astype(str).to_numpy(dtype=object),
            "theta": adata.var["theta"].to_numpy(dtype=np.float32),
            "delta": adata.var["delta"].to_numpy(dtype=np.float32),
            "fate_sign": adata.var["fate_sign"].to_numpy(dtype=np.float32),
        }
        alpha_fn = lambda z: _v2_alpha(z, genes)
    elif version in {"v1", ""}:
        genes = {
            "W": np.asarray(adata.varm["W"], dtype=np.float32),
            "b": adata.var["transcription_intercept"].to_numpy(dtype=np.float32),
            "beta_initial": adata.var["beta_initial"].to_numpy(dtype=np.float32),
            "beta_final": adata.var["beta_final"].to_numpy(dtype=np.float32),
            "gamma": adata.var["gamma"].to_numpy(dtype=np.float32),
        }
        alpha_fn = lambda z: _alpha(z, genes["W"], genes["b"])
    else:
        raise ValueError(f"unsupported synthetic generator version {version!r}")
    if genes["W"].shape != (adata.n_vars, D):
        raise ValueError("stored gene loading matrix has invalid shape")
    capture = adata.obs["capture_factor"].to_numpy(dtype=np.float32)
    batch_day = adata.obs["batch_day"].to_numpy(dtype=int) - 1
    batch = np.asarray(adata.uns["batch_gene_effects"], dtype=np.float32)
    if batch.ndim != 2 or batch.shape[1] != adata.n_vars or np.any(batch_day < 0) or np.any(batch_day >= len(batch)):
        raise ValueError("stored batch-effect metadata has invalid shape or index")
    beta_regime = adata.obs["beta_regime"].to_numpy(dtype=bool)
    embedding = adata.uns[meta_key]
    target = float(embedding["normalization_target"])
    mean = np.asarray(embedding["mean"], dtype=np.float32)
    linear_map = np.asarray(embedding["linear_map"], dtype=np.float32)
    if mean.shape != (adata.n_vars,) or linear_map.shape[0] != adata.n_vars:
        raise ValueError("stored embedding transform has invalid dimensions")

    def map_expected(pool: np.ndarray, indices: np.ndarray) -> np.ndarray:
        expected = pool * capture[indices, None] * batch[batch_day[indices]]
        scaled = expected * target / np.maximum(expected.sum(axis=1, keepdims=True), 1e-12)
        return (np.log1p(scaled) - mean[None, :]) @ linear_map

    current = map_expected(s_all, np.arange(adata.n_obs))
    future = np.empty_like(current, dtype=np.float32)
    rng = np.random.default_rng(seed)
    # Bound the working set while retaining a fixed traversal and RNG sequence.
    cells_per_batch = 128
    sqrt_dt = np.float32(np.sqrt(dt))
    paths = int(n_paths)
    for start in range(0, adata.n_obs, cells_per_batch):
        stop = min(start + cells_per_batch, adata.n_obs)
        batch_idx = np.arange(start, stop)
        # `_rna_step` deliberately accepts one beta vector.  Cells are therefore
        # continued in their stored collection-time beta regime without changing
        # its exact update law.
        for regime in (False, True):
            idx = batch_idx[beta_regime[batch_idx] == regime]
            if len(idx) == 0:
                continue
            n = len(idx)
            z = np.repeat(z_all[idx], paths, axis=0).copy()
            u = np.repeat(u_all[idx], paths, axis=0).copy()
            s = np.repeat(s_all[idx], paths, axis=0).copy()
            beta = genes["beta_final"] if regime else genes["beta_initial"]
            for _ in range(n_steps):
                alpha = alpha_fn(z)
                u, s = _rna_step(u, s, alpha, beta, genes["gamma"], dt)
                drift = _state_drift(z, stationary_cycle)
                noise = (sigma * sqrt_dt * rng.standard_normal(z.shape)).astype(np.float32)
                if stationary_cycle:
                    noise[:, 1] = 0.0
                z = (z + np.float32(dt) * drift + noise).astype(np.float32)
                if stationary_cycle:
                    z[:, 1] = 0.0
            mean_future = s.reshape(n, paths, adata.n_vars).mean(axis=1, dtype=np.float32)
            future[idx] = map_expected(mean_future, idx).astype(np.float32)
    return (future - current).astype(np.float32)



def make_multimodal(preset: str = "default", seed: int = 0, **overrides: Any) -> AnnData:
    """Generate a seeded v1 or v2 count-level lineage time series.

    V1 presets keep their original implementation and arrays.  V2 presets use
    lazy ancestral-path RNA integration; ``_v2_integration='eager'`` is a
    test-only equivalence mode for bounded small configurations.
    """
    if preset in V2_PRESETS:
        integration = str(overrides.pop("_v2_integration", "lazy"))
        return _make_multimodal_v2(_make_v2_config(preset, seed, overrides), integration=integration)
    if preset in V3_PRESETS:
        integration = str(overrides.pop("_v3_integration", "lazy"))
        return _make_multimodal_v3(_make_v3_config(preset, seed, overrides), integration=integration)
    return _make_multimodal_v1(preset=preset, seed=seed, **overrides)
# ---------------------------------------------------------------------------
# Generator v3
# ---------------------------------------------------------------------------
# V3 is deliberately additive.  The v1 and v2 RNG streams above must remain
# untouched because their saved baselines are regression fixtures.  V3 retains
# v2's two-pass ancestry/replay layout, but makes the model changes specified
# for S09 in its own config and helpers.
V3_PRESETS = ("v3_default", "v3_stress", "v3_batch", "v3_stationary_cycle", "v3_transition", "v3_nofork")

# Calibrated v3_default constants (orient/S09 Revision 5): the re-ranked winner
# (S1 required, then fewest unmet, then distance, with single-reference bands
# x [1/1.25, 1.25]; v3_calibration/rerank_r5.json, uid block1_09) with
# n_founders and f_sample from the full-size lineage-only grid
# (v3_calibration/r5_lineage_select.json, uid r5block3_27).  Fitted only to
# S08 statistic tables.
V3_DEFAULT_KNOBS: dict[str, Any] = {
    "mu_b": -2.75,
    "sigma_b": 2.5,
    "f_constant": 0.15,
    "f_switch": 0.15,
    "n_prog": 5,
    "theta0": 37.75,
    "gamma0": 0.1625,
    "sigma_gamma": 0.8999999999999999,
    "beta0": 0.2340347319320716,
    "eta_u": 1.0,
    "p_barcode": 0.35000000000000003,
    "n_founders": 800,
    "f_sample": 0.0289091169431116,
}


@dataclass(frozen=True)
class _V3Config:
    preset: str
    seed: int
    n_founders: int = 5000
    n_genes: int = 4000
    dt: float = DEFAULT_DT
    sigma: float = 0.05
    poisson: bool = True
    division_hours: float = 16.0
    f_sample: float = 0.005
    sampling_times: tuple[float, ...] = (48.0, 96.0, 144.0)
    p_barcode: float = 0.60
    labelling_efficiency: float = 0.90
    pulse_hours: tuple[float, float] = (3.0, 5.0)
    stress_beta_fraction: float = 0.0
    batch_effect: bool = False
    stationary_cycle: bool = False
    mu_b: float = 0.0
    sigma_b: float = 1.0
    f_constant: float = 0.20
    f_switch: float = 0.30
    n_prog: int = 4
    theta0: float = 20.0
    gamma0: float = 0.20
    sigma_gamma: float = 0.60
    beta0: float = 0.60
    eta_u: float = 0.60
    # orient/S10 Step 0 / 0b additions.  Defaults reproduce v3 exactly.
    z1_crit: float = Z1_CRIT
    pulse_batch_effect: bool = False


# Fields added after the S09 v3 datasets were written; omitted from the stored
# metadata when at their defaults so v3 outputs stay byte-identical.
_V3_ADDED_FIELD_DEFAULTS: dict[str, Any] = {"z1_crit": Z1_CRIT, "pulse_batch_effect": False}


def _make_v3_config(preset: str, seed: int, overrides: dict[str, Any]) -> _V3Config:
    if preset not in V3_PRESETS:
        raise ValueError(f"unknown v3 preset {preset!r}")
    values: dict[str, Any] = {"preset": preset, "seed": int(seed), **V3_DEFAULT_KNOBS}
    if preset == "v3_stress":
        # V3's scNT-seq 3/5-hour paired pulse design supersedes v2's cyclic
        # stress pulse durations.  The non-conflicting v2 stress interventions
        # are retained exactly.
        values.update(p_barcode=0.30, labelling_efficiency=0.60, stress_beta_fraction=0.20)
    elif preset == "v3_batch":
        values.update(batch_effect=True)
    elif preset == "v3_stationary_cycle":
        # Keep v2's stationary one-day collection design while retaining the
        # v3 genes, Gamma--Poisson noise, and paired scNT-seq labelling design.
        values.update(stationary_cycle=True, sampling_times=(24.0,), n_founders=5000, f_sample=0.20)
    elif preset == "v3_transition":
        # orient/S10 Step 0: v3_default knobs with the pitchfork moved to
        # z1 = 2 (mu crosses 0 at ~82 h: day 2 before, day 4 just after, day 6
        # committed).  Step 0b: separate per-gene capture factors for the 3 h
        # and 5 h pulse groups of each day (scNT-seq pulses are batches).
        values.update(z1_crit=2.0, pulse_batch_effect=True)
    elif preset == "v3_nofork":
        # orient/S14: exactly the transition measurement law, but move the
        # pitchfork beyond the latent z1 support so 48/96/144 h contain no fork.
        values.update(z1_crit=5.0, pulse_batch_effect=True)
    legal = set(_V3Config.__dataclass_fields__)
    unknown = sorted(set(overrides) - legal)
    if unknown:
        raise TypeError(f"unknown v3 make_multimodal overrides: {unknown}")
    values.update(overrides)
    config = _V3Config(**values)
    if config.n_founders < 1 or config.n_genes < 2:
        raise ValueError("n_founders must be positive and n_genes must be at least 2")
    if config.dt <= 0 or config.division_hours <= 0 or not (0 < config.f_sample <= 1):
        raise ValueError("dt, division_hours, and f_sample must be positive")
    if not config.sampling_times or any(t <= 0 for t in config.sampling_times):
        raise ValueError("sampling_times must be positive and nonempty")
    if any(abs(t / config.dt - round(t / config.dt)) > 1e-8 for t in config.sampling_times):
        raise ValueError("sampling_times must lie on the fixed dt grid")
    if tuple(sorted(config.pulse_hours)) != (3.0, 5.0):
        raise ValueError("v3 requires the fixed 3-hour and 5-hour paired pulse design")
    if not (0.0 <= config.f_constant <= 1.0 and 0.0 <= config.f_switch <= 1.0):
        raise ValueError("v3 programme shares must lie in [0, 1]")
    if config.f_constant + config.f_switch > 1.0 + 1e-12:
        raise ValueError("v3 constant and switching shares cannot exceed all genes")
    if int(config.n_prog) != config.n_prog or not (0 <= int(config.n_prog) <= 8):
        raise ValueError("n_prog must be an integer in [0, 8]")
    for key in ("sigma_b", "theta0", "gamma0", "sigma_gamma", "beta0", "eta_u", "p_barcode"):
        if float(getattr(config, key)) <= 0:
            raise ValueError(f"{key} must be positive")
    # S14's v3_nofork is the sole permitted beyond-support threshold.  The
    # historical and calibrated presets retain their original strict Z1_MAX
    # validation so their law and output behavior are unchanged.
    if config.preset == "v3_nofork":
        if not (0.0 < float(config.z1_crit) <= 5.0):
            raise ValueError("v3_nofork z1_crit must lie in (0, 5]")
    elif not (0.0 < float(config.z1_crit) < Z1_MAX):
        raise ValueError("z1_crit must lie strictly between 0 and Z1_MAX")
    return config


def _gene_parameters_v3(config: _V3Config, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Draw v3 gene parameters from an independent, replayable gene stream."""
    n_dims = D + int(config.n_prog)
    n_loadings = rng.integers(1, 4, size=config.n_genes)
    W = np.zeros((config.n_genes, n_dims), dtype=np.float32)
    for gene, k in enumerate(n_loadings):
        dims = rng.choice(n_dims, size=int(k), replace=False)
        W[gene, dims] = rng.normal(0.0, 1.0, size=int(k)).astype(np.float32)
    b = rng.normal(float(config.mu_b), float(config.sigma_b), size=config.n_genes).astype(np.float32)
    beta_initial = rng.lognormal(log(float(config.beta0)), 0.5, size=config.n_genes).astype(np.float32)
    gamma = rng.lognormal(log(float(config.gamma0)), float(config.sigma_gamma), size=config.n_genes).astype(np.float32)
    kind = np.full(config.n_genes, "linear", dtype=object)
    theta = np.full(config.n_genes, np.nan, dtype=np.float32)
    delta = np.full(config.n_genes, np.nan, dtype=np.float32)
    fate_sign = np.full(config.n_genes, np.nan, dtype=np.float32)
    # Discrete genes are assigned by the nearest deterministic whole-gene
    # allocation.  The three switching kinds always receive equal counts.
    n_constant = int(round(config.n_genes * float(config.f_constant)))
    n_each_switch = int(round(config.n_genes * float(config.f_switch) / 3.0))
    n_switch = 3 * n_each_switch
    if n_constant + n_switch > config.n_genes:
        # Decimal shares may sum to exactly one while independent rounding adds
        # one extra gene.  Preserve the constant count and equal switching
        # split, assigning the unavoidable remainder to the stated linear pool.
        n_each_switch = (config.n_genes - n_constant) // 3
        n_switch = 3 * n_each_switch
    picked = rng.choice(config.n_genes, size=n_constant + n_switch, replace=False)
    constant = picked[:n_constant]
    on = picked[n_constant:n_constant + n_each_switch]
    on_off = picked[n_constant + n_each_switch:n_constant + 2 * n_each_switch]
    fate = picked[n_constant + 2 * n_each_switch:]
    kind[constant] = "constant"
    kind[on] = "on"
    kind[on_off] = "on_off"
    kind[fate] = "fate"
    theta[on] = rng.uniform(0.3, 1.6, size=n_each_switch).astype(np.float32)
    theta[on_off] = rng.uniform(0.3, 1.0, size=n_each_switch).astype(np.float32)
    delta[on_off] = rng.uniform(0.4, 0.8, size=n_each_switch).astype(np.float32)
    fate_sign[fate] = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n_each_switch)
    changed = np.zeros(config.n_genes, dtype=bool)
    if config.stress_beta_fraction:
        changed[rng.choice(config.n_genes, size=int(round(config.n_genes * config.stress_beta_fraction)), replace=False)] = True
    beta_final = beta_initial.copy()
    beta_final[changed] *= 2.0
    nb_theta = rng.lognormal(log(float(config.theta0)), 0.5, size=config.n_genes).astype(np.float32)
    # The time constants belong to nuisance latent programmes and are drawn
    # with the gene parameters so both replay passes receive the same values.
    ou_tau = rng.uniform(6.0, 48.0, size=int(config.n_prog)).astype(np.float32)
    return {
        "W": W, "b": b, "beta_initial": beta_initial, "beta_final": beta_final,
        "gamma": gamma, "beta_changed": changed, "gene_kind": kind,
        "theta": theta, "delta": delta, "fate_sign": fate_sign, "nb_theta": nb_theta,
        "ou_tau": ou_tau,
    }


def _v3_alpha(z: np.ndarray, genes: dict[str, np.ndarray], z1_crit: float = Z1_CRIT) -> np.ndarray:
    """V3 transcription field with constant, linear, and v2-shaped switches.

    The fate genes' commitment gate sits at ``z1_crit + 0.2`` (1.2 by default,
    as in v2/v3).
    """
    z = np.asarray(z, dtype=np.float32)
    W = genes["W"]
    b = genes["b"]
    out = np.exp(np.clip(z @ W.T + b[None, :], -60.0, 60.0)).astype(np.float32)
    kind = genes["gene_kind"]
    constant = kind == "constant"
    if np.any(constant):
        out[:, constant] = np.exp(np.clip(b[None, constant], -60.0, 60.0)).astype(np.float32)
    width = np.float32(0.05)
    lo = np.exp(b - np.float32(1.5)).astype(np.float32)
    hi = np.exp(b + np.float32(1.5)).astype(np.float32)

    def logistic(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -60.0, 60.0)
        return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)

    on = kind == "on"
    if np.any(on):
        h = logistic((z[:, 1, None] - genes["theta"][None, on]) / width)
        out[:, on] = lo[None, on] + (hi[None, on] - lo[None, on]) * h
    on_off = kind == "on_off"
    if np.any(on_off):
        th = genes["theta"][on_off]
        de = genes["delta"][on_off]
        h = logistic((z[:, 1, None] - th[None, :]) / width)
        h *= 1.0 - logistic((z[:, 1, None] - th[None, :] - de[None, :]) / width)
        out[:, on_off] = lo[None, on_off] + (hi[None, on_off] - lo[None, on_off]) * h
    fate = kind == "fate"
    if np.any(fate):
        sign = genes["fate_sign"][fate]
        h = logistic((z[:, 0, None] * sign[None, :]) / width)
        h *= logistic((z[:, 1, None] - np.float32(float(z1_crit) + 0.2)) / width)
        out[:, fate] = lo[None, fate] + (hi[None, fate] - lo[None, fate]) * h
    return out


def _initial_v3_latent_population(config: _V3Config, rng: np.random.Generator) -> tuple[dict[str, np.ndarray], int]:
    phase = rng.uniform(0.0, 2.0 * np.pi, size=config.n_founders)
    radius = rng.normal(1.0, 0.05, size=config.n_founders).astype(np.float32)
    z = np.zeros((config.n_founders, D + int(config.n_prog)), dtype=np.float32)
    z[:, 0] = rng.normal(0.0, 0.05, size=config.n_founders)
    z[:, 1] = 0.0 if config.stationary_cycle else rng.normal(0.0, 0.1, size=config.n_founders)
    z[:, 2] = radius * np.cos(phase)
    z[:, 3] = radius * np.sin(phase)
    z[:, 4:D] = rng.normal(0.0, 0.3, size=(config.n_founders, D - 4))
    if config.n_prog:
        # Stationary N(0, 1) initial distribution for every independent OU dim.
        z[:, D:] = rng.normal(0.0, 1.0, size=(config.n_founders, int(config.n_prog)))
    division_rate = log(2.0) / config.division_hours
    live = {
        "z": z,
        "clone": np.arange(config.n_founders, dtype=np.int32),
        "cell_id": np.arange(config.n_founders, dtype=np.int64),
        "next_div": rng.exponential(1.0 / division_rate, size=config.n_founders).astype(np.float32),
    }
    return live, config.n_founders


def _v3_latent_step(z: np.ndarray, config: _V3Config, ou_tau: np.ndarray, rng: np.random.Generator, sqrt_dt: float) -> np.ndarray:
    """Advance six developmental coordinates plus independent OU nuisances."""
    z = np.asarray(z, dtype=np.float32)
    core = z[:, :D]
    core_noise = config.sigma * sqrt_dt * rng.standard_normal(core.shape).astype(np.float32)
    if config.stationary_cycle:
        core_noise[:, 1] = 0.0
    next_core = (core + config.dt * _state_drift(core, config.stationary_cycle, config.z1_crit) + core_noise).astype(np.float32)
    if config.stationary_cycle:
        next_core[:, 1] = 0.0
    if not config.n_prog:
        return next_core
    nuisance = z[:, D:]
    tau = np.asarray(ou_tau, dtype=np.float32)[None, :]
    # s = sqrt(2/tau) makes the continuous-time OU stationary standard deviation 1.
    noise = (np.sqrt(2.0 / tau).astype(np.float32) * sqrt_dt * rng.standard_normal(nuisance.shape)).astype(np.float32)
    next_nuisance = (nuisance - config.dt * nuisance / tau + noise).astype(np.float32)
    return np.concatenate((next_core, next_nuisance), axis=1)


def _v3_pulse_assignment(n: int, rng: np.random.Generator) -> np.ndarray:
    """Seeded, disjoint 3 h / 5 h allocation with half per sampling day."""
    pulse = np.full(n, 5.0, dtype=np.float32)
    if n:
        pulse[rng.choice(n, size=n // 2, replace=False)] = 3.0
    return pulse


def _v3_schedule(config: _V3Config) -> tuple[dict[float, int], set[float]]:
    time_to_day = {float(t): int(round(t / 24.0)) for t in config.sampling_times}
    pulse_starts = {round(float(t) - pulse, 8) for t in config.sampling_times for pulse in config.pulse_hours}
    return time_to_day, pulse_starts


def _simulate_v3_latent(config: _V3Config, rng: np.random.Generator, ou_tau: np.ndarray) -> dict[str, Any]:
    """First v3 pass: latent genealogy, paired-pulse assignments, collections."""
    live, next_cell_id = _initial_v3_latent_population(config, rng)
    batch_factors = _batch_gene_effects(config, rng)
    division_rate = log(2.0) / config.division_hours
    time_to_day, _ = _v3_schedule(config)
    horizon = float(max(config.sampling_times))
    sqrt_dt = float(np.sqrt(config.dt))
    n_steps = int(round(horizon / config.dt))
    parent: list[int] = [-1] * config.n_founders
    collected: dict[str, list[np.ndarray]] = {key: [] for key in (
        "z", "clone", "day", "time", "pulse", "capture", "batch_index", "cell_id",
    )}
    collected_counts: list[int] = []
    beta_regime: list[np.ndarray] = []
    for step in range(1, n_steps + 1):
        t = round(step * config.dt, 8)
        live["z"] = _v3_latent_step(live["z"], config, ou_tau, rng, sqrt_dt)
        divide = live["next_div"] <= t
        _, parent_ids, next_cell_id = _latent_divide(live, divide, t, division_rate, rng, next_cell_id)
        parent.extend(int(v) for v in parent_ids)
        if t not in time_to_day:
            continue
        day = time_to_day[t]
        n_take = int(round(config.f_sample * len(live["z"])))
        if n_take < 1:
            raise RuntimeError("sampling fraction produced no collected cells")
        take = np.sort(rng.choice(len(live["z"]), size=n_take, replace=False))
        capture = rng.lognormal(log(0.1), 0.3, size=n_take).astype(np.float32)
        pulse = _v3_pulse_assignment(n_take, rng)
        for key in ("z", "clone", "cell_id"):
            collected[key].append(live[key][take].copy())
        collected["day"].append(np.full(n_take, day, dtype=np.int16))
        collected["time"].append(np.full(n_take, t, dtype=np.float32))
        collected["pulse"].append(pulse)
        collected["capture"].append(capture)
        collected["batch_index"].append(np.full(n_take, len(collected_counts), dtype=np.int16))
        collected_counts.append(n_take)
        beta_regime.append(np.full(n_take, int(config.stress_beta_fraction and t >= horizon / 2.0), dtype=np.int8))
        _remove_latent_indices(live, take)
    if not collected_counts:
        raise RuntimeError("no samples collected")
    sampled = {key: np.concatenate(values, axis=0) for key, values in collected.items()}
    return {
        "sampled": sampled,
        "parent": np.asarray(parent, dtype=np.int64),
        "n_collected_by_day": np.asarray(collected_counts, dtype=np.int32),
        "beta_regime": np.concatenate(beta_regime),
        "n_living_after_final_collection": int(len(live["z"])),
        "batch_factors": batch_factors,
    }


def _replay_v3_rna(config: _V3Config, first: dict[str, Any], integration: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Replay v3 latent draws exactly and integrate only selected ancestral RNA."""
    if integration not in {"lazy", "eager"}:
        raise ValueError("integration must be 'lazy' or 'eager'")
    sampled = first["sampled"]
    parent = np.asarray(first["parent"], dtype=np.int64)
    needed = _needed_gene_cells(parent, sampled["cell_id"], eager=(integration == "eager"))
    slots = np.full(len(parent), -1, dtype=np.int64)
    slots[np.flatnonzero(needed)] = np.arange(int(needed.sum()), dtype=np.int64)
    n_slots = int(needed.sum())
    rng = np.random.default_rng(config.seed)
    genes = _gene_parameters_v3(config, rng)
    live, next_cell_id = _initial_v3_latent_population(config, rng)
    batch_factors = _batch_gene_effects(config, rng)
    if not np.array_equal(batch_factors, first["batch_factors"]):
        raise RuntimeError("v3 latent replay changed batch-factor RNG state")
    division_rate = log(2.0) / config.division_hours
    time_to_day, pulse_starts = _v3_schedule(config)
    horizon = float(max(config.sampling_times))
    sqrt_dt = float(np.sqrt(config.dt))
    n_steps = int(round(horizon / config.dt))
    pools = {name: np.zeros((n_slots, config.n_genes), dtype=np.float32) for name in (
        "u", "s", "u_new_3", "s_new_3", "u_new_5", "s_new_5",
    )}
    previous_s = np.zeros((n_slots, config.n_genes), dtype=np.float32)
    founder_slots = slots[live["cell_id"]]
    alpha0 = _v3_alpha(live["z"], genes, config.z1_crit)
    pools["u"][founder_slots] = alpha0 / genes["beta_initial"][None, :]
    pools["s"][founder_slots] = alpha0 / genes["gamma"][None, :]
    replay: dict[str, list[np.ndarray]] = {key: [] for key in ("u", "s", "u_new", "s_new", "s_prev")}
    offset = 0
    for step in range(1, n_steps + 1):
        t_prev = round((step - 1) * config.dt, 8)
        t = round(step * config.dt, 8)
        live_slots = slots[live["cell_id"]]
        active_pos = np.flatnonzero(live_slots >= 0)
        active_slots = live_slots[active_pos]
        if len(active_slots):
            if t_prev in pulse_starts:
                for duration in config.pulse_hours:
                    if round(t_prev + duration, 8) in {round(x, 8) for x in config.sampling_times}:
                        pools[f"u_new_{int(duration)}"][active_slots] = 0.0
                        pools[f"s_new_{int(duration)}"][active_slots] = 0.0
            previous_s[active_slots] = pools["s"][active_slots]
            beta = genes["beta_final"] if (config.stress_beta_fraction and t_prev >= horizon / 2.0) else genes["beta_initial"]
            alpha = _v3_alpha(live["z"][active_pos], genes, config.z1_crit)
            pools["u"][active_slots], pools["s"][active_slots] = _rna_step(
                pools["u"][active_slots], pools["s"][active_slots], alpha, beta, genes["gamma"], config.dt)
            for duration in config.pulse_hours:
                ukey, skey = f"u_new_{int(duration)}", f"s_new_{int(duration)}"
                pools[ukey][active_slots], pools[skey][active_slots] = _rna_step(
                    pools[ukey][active_slots], pools[skey][active_slots], alpha, beta, genes["gamma"], config.dt)
        live["z"] = _v3_latent_step(live["z"], config, genes["ou_tau"], rng, sqrt_dt)
        divide = live["next_div"] <= t
        _, parent_ids, next_cell_id = _latent_divide(live, divide, t, division_rate, rng, next_cell_id)
        if len(parent_ids):
            child_ids = np.arange(next_cell_id - len(parent_ids), next_cell_id, dtype=np.int64)
            parent_slots = slots[parent_ids]
            child_slots = slots[child_ids]
            child_needed = child_slots >= 0
            if np.any(child_needed):
                if np.any(parent_slots[child_needed] < 0):
                    raise RuntimeError("needed descendant lacks a needed ancestor")
                for name in pools:
                    pools[name][child_slots[child_needed]] = pools[name][parent_slots[child_needed]]
                previous_s[child_slots[child_needed]] = previous_s[parent_slots[child_needed]]
        if t not in time_to_day:
            continue
        n_take = int(round(config.f_sample * len(live["z"])))
        take = np.sort(rng.choice(len(live["z"]), size=n_take, replace=False))
        capture = rng.lognormal(log(0.1), 0.3, size=n_take).astype(np.float32)
        pulse = _v3_pulse_assignment(n_take, rng)
        expected = slice(offset, offset + n_take)
        if not np.array_equal(live["cell_id"][take], sampled["cell_id"][expected]):
            raise RuntimeError("v3 RNA replay changed sampled cell identities")
        if not np.array_equal(capture, sampled["capture"][expected]):
            raise RuntimeError("v3 RNA replay changed capture-factor RNG state")
        if not np.array_equal(pulse, sampled["pulse"][expected]):
            raise RuntimeError("v3 RNA replay changed paired-pulse RNG state")
        take_slots = slots[live["cell_id"][take]]
        if np.any(take_slots < 0):
            raise RuntimeError("a collected cell was omitted from lazy ancestry")
        replay["u"].append(pools["u"][take_slots].copy())
        replay["s"].append(pools["s"][take_slots].copy())
        u_new = np.empty((n_take, config.n_genes), dtype=np.float32)
        s_new = np.empty((n_take, config.n_genes), dtype=np.float32)
        for duration in config.pulse_hours:
            mask = pulse == np.float32(duration)
            if np.any(mask):
                u_new[mask] = pools[f"u_new_{int(duration)}"][take_slots[mask]]
                s_new[mask] = pools[f"s_new_{int(duration)}"][take_slots[mask]]
        replay["u_new"].append(u_new)
        replay["s_new"].append(s_new)
        replay["s_prev"].append(previous_s[take_slots].copy())
        offset += n_take
        _remove_latent_indices(live, take)
    if offset != len(sampled["cell_id"]):
        raise RuntimeError("v3 RNA replay did not visit every collection")
    return {key: np.concatenate(values, axis=0) for key, values in replay.items()}, genes


def _make_multimodal_v3(config: _V3Config, integration: str = "lazy") -> AnnData:
    """Generate a v3 dataset with full nuisance state and core-only dynamics truth."""
    rng = np.random.default_rng(config.seed)
    genes_first = _gene_parameters_v3(config, rng)
    first = _simulate_v3_latent(config, rng, genes_first["ou_tau"])
    pools, genes = _replay_v3_rna(config, first, integration=integration)
    sampled = first["sampled"]
    beta_regime = np.asarray(first["beta_regime"], dtype=np.int8)
    beta_per_cell = np.where(beta_regime[:, None].astype(bool), genes["beta_final"][None, :], genes["beta_initial"][None, :])
    ds_dt = beta_per_cell * pools["u"] - genes["gamma"][None, :] * pools["s"]
    core_z = sampled["z"][:, :D]
    jac = _state_jacobian(core_z, config.stationary_cycle, config.z1_crit)
    omega_state = 0.5 * (jac - np.swapaxes(jac, 1, 2))
    committed = core_z[:, 1] > (float(config.z1_crit) + 0.2)
    fate = np.where(committed, np.where(core_z[:, 0] >= 0.0, "positive", "negative"), "progenitor")
    batch_effects = first["batch_factors"]
    batch_index = np.asarray(sampled["batch_index"], dtype=np.int16)
    pulse_factors = None
    if config.pulse_batch_effect:
        # orient/S10 Step 0b: one LogNormal(0, 0.3) per-gene capture factor for
        # each (collection, pulse) group, from its own seeded stream so the
        # latent, gene and measurement streams are untouched.  The effective
        # per-group factor (collection factor x pulse factor) is stored as
        # ``batch_gene_effects`` and indexed by ``batch_day`` so every truth
        # function maps expected counts with the factor the cell received.
        pulse_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 0x50554C53]))
        n_collections = len(first["n_collected_by_day"])
        pulse_factors = pulse_rng.lognormal(0.0, 0.3, size=(2 * n_collections, config.n_genes)).astype(np.float32)
        group = (2 * batch_index.astype(np.int64) + (sampled["pulse"] == np.float32(5.0)).astype(np.int64)).astype(np.int16)
        batch_effects = (np.repeat(first["batch_factors"], 2, axis=0) * pulse_factors).astype(np.float32)
        batch_index = group
    gene_effect = batch_effects[batch_index]
    capture_scale = sampled["capture"][:, None] * gene_effect
    measurement_rng = np.random.default_rng(np.random.SeedSequence([config.seed, 0x534F39]))
    if config.poisson:
        theta = genes["nb_theta"]
        multiplier = measurement_rng.gamma(theta[None, :], (1.0 / theta)[None, :], size=pools["s"].shape).astype(np.float32)
        spliced = measurement_rng.poisson(np.maximum(multiplier * capture_scale * pools["s"], 0.0)).astype(np.int32)
        unspliced = measurement_rng.poisson(np.maximum(multiplier * capture_scale * config.eta_u * pools["u"], 0.0)).astype(np.int32)
    else:
        spliced = (capture_scale * pools["s"]).astype(np.float32)
        unspliced = (capture_scale * config.eta_u * pools["u"]).astype(np.float32)
    total = spliced + unspliced
    fraction_new = np.divide(pools["u_new"] + pools["s_new"], pools["u"] + pools["s"], out=np.zeros_like(pools["s"], dtype=np.float32), where=(pools["u"] + pools["s"]) > 0)
    fraction_new = np.clip(fraction_new, 0.0, 1.0)
    if config.poisson:
        new_before_efficiency = measurement_rng.binomial(total.astype(np.int64), fraction_new).astype(np.int32)
        new = measurement_rng.binomial(new_before_efficiency, config.labelling_efficiency).astype(np.int32)
    else:
        new = (config.labelling_efficiency * total * fraction_new).astype(np.float32)
    clone_ids = np.asarray([f"clone_{i:04d}" for i in range(config.n_founders)], dtype=object)
    detected = measurement_rng.random(len(sampled["clone"])) < config.p_barcode
    clone_values = np.where(detected, clone_ids[sampled["clone"]], None)
    obs = pd.DataFrame(index=[f"cell_{int(i):07d}" for i in sampled["cell_id"]])
    obs["day"] = sampled["day"].astype(np.int16)
    obs["time_hours"] = sampled["time"].astype(np.float32)
    obs["pulse_hours"] = sampled["pulse"].astype(np.float32)
    obs["clone"] = pd.Categorical(clone_values, categories=list(clone_ids))
    obs["clone_true"] = pd.Categorical(clone_ids[sampled["clone"]], categories=list(clone_ids))
    obs["fate_true"] = pd.Categorical(fate, categories=["progenitor", "negative", "positive"])
    obs["capture_factor"] = sampled["capture"].astype(np.float32)
    obs["batch_day"] = batch_index.astype(np.int16) + 1
    obs["beta_regime"] = beta_regime
    var = pd.DataFrame(index=[f"gene_{i:04d}" for i in range(config.n_genes)])
    var["transcription_intercept"] = genes["b"]
    var["beta_initial"] = genes["beta_initial"]
    var["beta_final"] = genes["beta_final"]
    var["gamma"] = genes["gamma"]
    var["assumption_breaking_kinetics"] = genes["beta_changed"]
    var["gene_kind"] = pd.Categorical(genes["gene_kind"], categories=["constant", "linear", "on", "on_off", "fate"])
    var["theta"] = genes["theta"]
    var["delta"] = genes["delta"]
    var["fate_sign"] = genes["fate_sign"]
    var["nb_theta"] = genes["nb_theta"]
    adata = AnnData(X=spliced.copy(), obs=obs, var=var)
    adata.layers["spliced"] = spliced.copy()
    adata.layers["unspliced"] = unspliced
    adata.layers["total"] = total
    adata.layers["new"] = new
    adata.layers["u_true"] = pools["u"].astype(np.float32)
    adata.layers["s_true"] = pools["s"].astype(np.float32)
    adata.layers["s_prev_true"] = pools["s_prev"].astype(np.float32)
    adata.layers["u_new_true"] = pools["u_new"].astype(np.float32)
    adata.layers["s_new_true"] = pools["s_new"].astype(np.float32)
    adata.layers["new_fraction_true"] = fraction_new.astype(np.float32)
    adata.layers["ds_dt_true"] = ds_dt.astype(np.float32)
    adata.obsm["z_true"] = sampled["z"].astype(np.float32)
    adata.obsm["v_state"] = _state_drift(core_z, config.stationary_cycle, config.z1_crit).astype(np.float32)
    adata.obsm["jacobian_state"] = jac.astype(np.float32)
    adata.obsm["omega_state"] = omega_state.astype(np.float32)
    rows = np.flatnonzero(detected)
    adata.obsm["clone_matrix"] = sparse.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, sampled["clone"][rows])), shape=(adata.n_obs, config.n_founders))
    adata.varm["W"] = genes["W"]
    earliest = np.flatnonzero(obs["day"].to_numpy() == int(obs["day"].min()))
    start = earliest[int(np.argmin(np.abs(core_z[earliest, 1])))]
    adata.uns["suggested_start_cell"] = str(adata.obs_names[start])
    adata.uns["clone_ids"] = clone_ids.astype(str)
    adata.uns["batch_gene_effects"] = batch_effects
    if pulse_factors is not None:
        adata.uns["pulse_batch_gene_effects"] = pulse_factors
        adata.uns["collection_batch_gene_effects"] = first["batch_factors"]
    stored_config = asdict(config)
    for key, default in _V3_ADDED_FIELD_DEFAULTS.items():
        if stored_config[key] == default:
            del stored_config[key]
    if pulse_factors is not None:
        stored_config["batch_index_definition"] = "batch_day - 1 = 2 * collection_index + (pulse_hours == 5)"
    adata.uns["synthetic_multimodal"] = {
        **stored_config,
        "version": "v3",
        "sampling_times": np.asarray(config.sampling_times, dtype=np.float32),
        "pulse_hours": np.asarray(config.pulse_hours, dtype=np.float32),
        "n_collected_by_day": np.asarray(first["n_collected_by_day"], dtype=np.int32),
        "n_living_after_final_collection": int(first["n_living_after_final_collection"]),
        "n_clones_detected": int(len(np.unique(sampled["clone"][detected]))),
        "assumption_breaking_gene_indices": np.flatnonzero(genes["beta_changed"]).astype(np.int32),
        "capture_eta_unspliced": float(config.eta_u),
        "omega": float(OMEGA),
        "rna_integration": integration,
        "n_gene_ancestral_paths": int(np.asarray(_needed_gene_cells(first["parent"], sampled["cell_id"], eager=False)).sum()),
        "core_dims": list(range(D)),
        "ou_tau_hours": genes["ou_tau"].astype(np.float32),
        "pulse_assignment": "seeded_disjoint_half_per_sampling_day",
    }
    return adata


# ---------------------------------------------------------------------------
# orient/S10 Step 6 additions: diffusion truth and measurement-noise reference
# ---------------------------------------------------------------------------
def _v3_truth_inputs(adata: AnnData, method: str) -> dict[str, Any]:
    """Collect the stored v3 truth needed to continue every cell (scoring only)."""
    synthetic = adata.uns["synthetic_multimodal"]
    if str(synthetic.get("version", "")) != "v3":
        raise ValueError("this S10 truth helper supports generator v3 only")
    meta_key = f"embedding_{method}"
    if meta_key not in adata.uns:
        raise KeyError(f"missing stored embedding metadata {meta_key!r}")
    genes = {
        "W": np.asarray(adata.varm["W"], dtype=np.float32),
        "b": adata.var["transcription_intercept"].to_numpy(dtype=np.float32),
        "beta_initial": adata.var["beta_initial"].to_numpy(dtype=np.float32),
        "beta_final": adata.var["beta_final"].to_numpy(dtype=np.float32),
        "gamma": adata.var["gamma"].to_numpy(dtype=np.float32),
        "gene_kind": adata.var["gene_kind"].astype(str).to_numpy(dtype=object),
        "theta": adata.var["theta"].to_numpy(dtype=np.float32),
        "delta": adata.var["delta"].to_numpy(dtype=np.float32),
        "fate_sign": adata.var["fate_sign"].to_numpy(dtype=np.float32),
        "nb_theta": adata.var["nb_theta"].to_numpy(dtype=np.float32),
    }
    n_prog = int(synthetic["n_prog"])
    from types import SimpleNamespace
    cfg = SimpleNamespace(sigma=float(synthetic["sigma"]), dt=float(synthetic["dt"]),
                          stationary_cycle=bool(synthetic["stationary_cycle"]), n_prog=n_prog,
                          z1_crit=float(synthetic.get("z1_crit", Z1_CRIT)))
    capture = adata.obs["capture_factor"].to_numpy(dtype=np.float32)
    batch_day = adata.obs["batch_day"].to_numpy(dtype=int) - 1
    batch = np.asarray(adata.uns["batch_gene_effects"], dtype=np.float32)
    embedding = adata.uns[meta_key]
    return {"genes": genes, "cfg": cfg, "ou_tau": np.asarray(synthetic["ou_tau_hours"], dtype=np.float32),
            "gene_factor": capture[:, None] * batch[batch_day], "target": float(embedding["normalization_target"]),
            "mean": np.asarray(embedding["mean"], dtype=np.float32),
            "linear_map": np.asarray(embedding["linear_map"], dtype=np.float32)}


def embed_with_stored_transform(adata: AnnData, spliced: np.ndarray, method: str) -> np.ndarray:
    """Embed spliced counts with a dataset's stored normalisation, mean and linear map."""
    meta = adata.uns[f"embedding_{method}"]
    x = np.asarray(spliced, dtype=np.float64)
    scaled = x * float(meta["normalization_target"]) / np.maximum(x.sum(axis=1, keepdims=True), 1e-12)
    return ((np.log1p(scaled) - np.asarray(meta["mean"], dtype=np.float64)) @ np.asarray(meta["linear_map"], dtype=np.float64)).astype(np.float32)


def resample_spliced_counts(adata: AnnData, noise_seed: int) -> np.ndarray:
    """Redraw the v3 spliced counts from the stored true pools with a new noise seed.

    Only the Gamma multiplier and the Poisson draw are resampled; the capture
    factors, batch effects and pools are the stored truth.  Measurement-noise
    reference for orient/S10 Step 6 (scoring only).
    """
    synthetic = adata.uns["synthetic_multimodal"]
    if str(synthetic.get("version", "")) != "v3" or not bool(synthetic["poisson"]):
        raise ValueError("resampling is defined for Gamma-Poisson v3 data only")
    rng = np.random.default_rng(np.random.SeedSequence([int(synthetic["seed"]), 0x4E4F4953, int(noise_seed)]))
    theta = adata.var["nb_theta"].to_numpy(dtype=np.float32)
    capture = adata.obs["capture_factor"].to_numpy(dtype=np.float32)
    batch = np.asarray(adata.uns["batch_gene_effects"], dtype=np.float32)
    scale = capture[:, None] * batch[adata.obs["batch_day"].to_numpy(dtype=int) - 1]
    s = np.asarray(adata.layers["s_true"], dtype=np.float32)
    multiplier = rng.gamma(theta[None, :], (1.0 / theta)[None, :], size=s.shape).astype(np.float32)
    return rng.poisson(np.maximum(multiplier * scale * s, 0.0)).astype(np.int32)


def forward_displacement_covariance(
    adata: AnnData,
    method: str | tuple[str, ...],
    horizon_hours: float | tuple[float, ...],
    n_paths: int = 32,
    seed: int = 0,
    cells: np.ndarray | None = None,
) -> Any:
    """Covariance of expected-count embedding positions after ``horizon_hours``.

    Scoring-only truth (orient/S10 Step 6).  From each cell's stored latent
    state (all dims, nuisance programmes included) and RNA pools, ``n_paths``
    continuations of the v3 SDE and RNA kinetics are simulated (no division,
    collection-time beta regime); every path's expected spliced counts
    (capture x batch effect x pool) are mapped through the stored embedding,
    and the across-path covariance (ddof = 1) is returned, ``n x d x d``.
    ``D_eff(dt) = Cov / (2 dt)``.  ``method`` and ``horizon_hours`` may be
    tuples; then a dict ``{(method, horizon): cov}`` is returned from one set
    of simulated paths (the horizons are nested time points of the same paths).
    """
    methods = (method,) if isinstance(method, str) else tuple(method)
    horizons = (float(horizon_hours),) if np.isscalar(horizon_hours) else tuple(float(h) for h in horizon_hours)
    if int(n_paths) != n_paths or int(n_paths) < 2:
        raise ValueError("n_paths must be an integer >= 2")
    inputs = {m: _v3_truth_inputs(adata, m) for m in methods}
    first = inputs[methods[0]]
    genes, cfg, ou_tau = first["genes"], first["cfg"], first["ou_tau"]
    dt = cfg.dt
    steps = {}
    for h in horizons:
        if not np.isfinite(h) or h <= 0 or abs(h / dt - round(h / dt)) > 1e-8:
            raise ValueError("each horizon must be positive and on the generator dt grid")
        steps[int(round(h / dt))] = h
    n_steps = max(steps)
    z_all = np.asarray(adata.obsm["z_true"], dtype=np.float32)
    u_all = np.asarray(adata.layers["u_true"], dtype=np.float32)
    s_all = np.asarray(adata.layers["s_true"], dtype=np.float32)
    beta_regime = adata.obs["beta_regime"].to_numpy(dtype=bool)
    rows = np.arange(adata.n_obs) if cells is None else np.asarray(cells, dtype=int)
    out = {(m, h): np.zeros((len(rows), inputs[m]["linear_map"].shape[1], inputs[m]["linear_map"].shape[1]), dtype=np.float32)
           for m in methods for h in horizons}
    rng = np.random.default_rng(seed)
    sqrt_dt = float(np.sqrt(dt))
    paths = int(n_paths)
    per_batch = 32         # rows per simulated block = per_batch x n_paths (cache-friendlier than 128)
    for start in range(0, len(rows), per_batch):
        local = np.arange(start, min(start + per_batch, len(rows)))
        for regime in (False, True):
            sel = local[beta_regime[rows[local]] == regime]
            if len(sel) == 0:
                continue
            idx = rows[sel]
            n = len(idx)
            z = np.repeat(z_all[idx], paths, axis=0).copy()
            u = np.repeat(u_all[idx], paths, axis=0).copy()
            s = np.repeat(s_all[idx], paths, axis=0).copy()
            beta = genes["beta_final"] if regime else genes["beta_initial"]
            for step in range(1, n_steps + 1):
                alpha = _v3_alpha(z, genes, cfg.z1_crit)
                u, s = _rna_step(u, s, alpha, beta, genes["gamma"], dt)
                z = _v3_latent_step(z, cfg, ou_tau, rng, sqrt_dt)
                if step in steps:
                    for m in methods:
                        inp = inputs[m]
                        expected = s * np.repeat(inp["gene_factor"][idx], paths, axis=0)
                        scaled = expected * inp["target"] / np.maximum(expected.sum(axis=1, keepdims=True), 1e-12)
                        pos = ((np.log1p(scaled) - inp["mean"][None, :]) @ inp["linear_map"]).reshape(n, paths, -1).astype(np.float64)
                        centred = pos - pos.mean(axis=1, keepdims=True)
                        out[(m, steps[step])][sel] = (np.einsum("npi,npj->nij", centred, centred) / (paths - 1)).astype(np.float32)
    if len(methods) == 1 and len(horizons) == 1:
        return out[(methods[0], horizons[0])]
    return out
