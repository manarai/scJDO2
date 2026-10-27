"""Validity gates for the count-level synthetic multimodal generator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scjdo2.data.synthetic_multimodal import (
    OMEGA,
    embed_linear,
    finite_difference_embedding_velocity,
    make_multimodal,
    true_velocity_in_embedding,
    forward_developmental_truth,
)
from scjdo2.labelled_rates import LabelledRateModel


@pytest.fixture(scope="module")
def default_data():
    """The fixed default dataset; reused across the costlier gates."""
    return make_multimodal("default", seed=0)


@pytest.fixture(scope="module")
def noise_free_data():
    # Size overrides make the analytic checks inexpensive without changing the
    # default preset's dynamics, kinetics, collection schedule, or readout law.
    return make_multimodal(
        "default", seed=11, n_founders=30, n_genes=80, sigma=0.0, poisson=False,
    )


def _analytic_rna_step(u, s, alpha, beta, gamma, dt):
    """Float64 analytic local continuation for finite-difference verification."""
    u = np.asarray(u, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    eb = np.exp(-beta * dt)
    eg = np.exp(-gamma * dt)
    denom = gamma - beta
    ib = np.where(np.abs(denom) < 1e-12, dt * eg, (eb - eg) / denom)
    i0 = (1.0 - eg) / gamma
    u_next = u * eb[None, :] + alpha * ((1.0 - eb) / beta)[None, :]
    s_next = s * eg[None, :] + u * (beta * ib)[None, :] + alpha * (i0 - ib)[None, :]
    return u_next, s_next


def _spliced_lag(adata):
    beta = adata.var["beta_initial"].to_numpy(dtype=float)
    gamma = adata.var["gamma"].to_numpy(dtype=float)
    u = np.asarray(adata.layers["u_true"], dtype=float)
    s = np.asarray(adata.layers["s_true"], dtype=float)
    return np.abs(gamma[None, :] * s - beta[None, :] * u) / np.maximum(beta[None, :] * u, 1e-12)


def test_determinism_same_seed_has_identical_arrays():
    kwargs = dict(preset="default", seed=7, n_founders=16, n_genes=48)
    first = make_multimodal(**kwargs)
    second = make_multimodal(**kwargs)
    assert first.obs.equals(second.obs)
    assert first.var.equals(second.var)
    assert np.array_equal(first.X, second.X)
    for key in first.layers:
        assert np.array_equal(first.layers[key], second.layers[key]), key
    for key in ("z_true", "v_state", "jacobian_state", "omega_state"):
        assert np.array_equal(first.obsm[key], second.obsm[key]), key
    assert (first.obsm["clone_matrix"] != second.obsm["clone_matrix"]).nnz == 0


def test_kinetic_velocity_matches_a_cell_history_finite_difference(noise_free_data):
    a = noise_free_data
    z = np.asarray(a.obsm["z_true"], dtype=float)
    alpha = np.exp(z @ np.asarray(a.varm["W"], dtype=float).T + a.var["transcription_intercept"].to_numpy(dtype=float)[None, :])
    beta = a.var["beta_initial"].to_numpy(dtype=float)
    gamma = a.var["gamma"].to_numpy(dtype=float)
    delta = 1e-5
    _, s_later = _analytic_rna_step(a.layers["u_true"], a.layers["s_true"], alpha, beta, gamma, delta)
    finite_difference = (s_later - np.asarray(a.layers["s_true"], dtype=float)) / delta
    velocity = np.asarray(a.layers["ds_dt_true"], dtype=float)
    rel = np.linalg.norm(finite_difference - velocity) / np.linalg.norm(velocity)
    assert rel <= 1e-3


def test_spliced_lag_has_scvelo_scale_signal(default_data):
    lag = _spliced_lag(default_data)
    qualifying = lag > 0.2
    genes_with_support = qualifying.mean(axis=0) >= 0.10
    assert qualifying.mean() >= 0.10
    assert genes_with_support.sum() >= 200
    assert genes_with_support.mean() >= 0.25
    by_day = {
        int(day): float(qualifying[default_data.obs["day"].to_numpy() == day].mean())
        for day in sorted(default_data.obs["day"].unique())
    }
    assert all(np.isfinite(v) for v in by_day.values())
    # The review records the biological pattern: support fades as cells settle.
    assert by_day[1] > by_day[4]


@pytest.mark.parametrize("method", ("pca", "fa"))
def test_embedding_truth_mapping_matches_finite_difference(noise_free_data, method):
    a = noise_free_data.copy()
    embed_linear(a, method=method, n_comp=12)
    velocity = true_velocity_in_embedding(a, method)
    finite_difference = finite_difference_embedding_velocity(a, method, dt=1e-5)
    cosine = np.sum(velocity * finite_difference, axis=1) / (
        np.linalg.norm(velocity, axis=1) * np.linalg.norm(finite_difference, axis=1) + 1e-12
    )
    assert np.median(cosine) >= 0.99


def test_lineage_truth_and_detection_gate(default_data):
    a = default_data
    detected = a.obs["clone"].notna().to_numpy()
    assert abs(float(detected.mean()) - 0.60) <= 0.03
    clone = a.obs.loc[detected, "clone"].astype(str)
    truth = a.obs.loc[detected, "clone_true"].astype(str)
    assert np.array_equal(clone.to_numpy(), truth.to_numpy())
    observed_days = a.obs.loc[detected].groupby("clone", observed=True)["day"].nunique()
    assert float((observed_days >= 2).mean()) >= 0.50
    matrix = a.obsm["clone_matrix"].tocsr()
    assert matrix.shape == (a.n_obs, len(a.uns["clone_ids"]))
    assert matrix.nnz == int(detected.sum())
    assert np.all(np.diff(matrix.indptr) == detected.astype(np.int64))


def test_labelling_fraction_and_rate_fit(noise_free_data, default_data):
    a = noise_free_data
    total = np.asarray(a.layers["total"], dtype=float)
    expected = np.asarray(a.layers["new_fraction_true"], dtype=float)
    observed = np.asarray(a.layers["new"], dtype=float) / (0.9 * np.maximum(total, 1e-12))
    rel = np.abs(observed - expected) / np.maximum(expected, 1e-8)
    assert np.max(rel[total > 0]) <= 1e-3

    # A few deterministic CPU epochs exercise the documented public interface
    # on default-preset counts while keeping the test suite bounded.
    take = np.arange(min(512, default_data.n_obs))
    model = LabelledRateModel(
        n_epochs=3, patience=3, batch_size=128, hidden=(16,), device="cpu", seed=0,
    ).fit(
        np.asarray(default_data.layers["new"])[take],
        np.asarray(default_data.layers["total"])[take],
        np.asarray(default_data.obsm["z_true"])[take],
        default_data.obs["pulse_hours"].to_numpy(dtype=float)[take],
    )
    assert np.isfinite(model.gamma).all()


def test_time_dynamics_and_stationary_rotation(default_data):
    a = default_data
    assert np.array_equal(a.obs["time_hours"].to_numpy(), a.obs["day"].to_numpy(dtype=float) * 24.0)
    committed = a.obs.groupby("day", observed=True)["fate_true"].apply(lambda x: float((x != "progenitor").mean()))
    assert np.all(np.diff(committed.to_numpy()) >= 0.0)

    stationary = make_multimodal("stationary_cycle", seed=3, n_founders=40, n_genes=48)
    z1 = np.asarray(stationary.obsm["z_true"])[:, 1]
    gate = 1.0 / (1.0 + np.exp((z1 - 1.0) / 0.1))
    omega = np.asarray(stationary.obsm["omega_state"])
    assert np.allclose(z1, 0.0)
    assert np.all(gate > 0.99)
    assert np.allclose(omega[:, 3, 2], gate * OMEGA, rtol=1e-5, atol=1e-6)
    assert np.allclose(omega[:, 2, 3], -gate * OMEGA, rtol=1e-5, atol=1e-6)


def test_sizes_layers_founder_override_and_presets(default_data):
    a = default_data
    required_layers = {
        "spliced", "unspliced", "total", "new", "u_true", "s_true", "ds_dt_true",
        "u_new_true", "s_new_true", "new_fraction_true",
    }
    assert 2900 <= a.n_obs <= 3900
    assert a.n_vars == 800
    assert required_layers <= set(a.layers)
    assert np.array_equal(a.layers["total"], a.layers["spliced"] + a.layers["unspliced"])
    assert np.all(a.layers["new"] <= a.layers["total"])
    small = make_multimodal("default", seed=3, n_founders=20, n_genes=48)
    large = make_multimodal("default", seed=3, n_founders=40, n_genes=48)
    assert large.n_obs > small.n_obs

    stress = make_multimodal("stress", seed=4, n_founders=16, n_genes=100)
    assert stress.uns["synthetic_multimodal"]["p_barcode"] == 0.30
    assert stress.uns["synthetic_multimodal"]["labelling_efficiency"] == 0.60
    assert stress.var["assumption_breaking_kinetics"].sum() == 20
    assert set(stress.obs["pulse_hours"].unique()) == {1.0, 2.0, 4.0}
    batch = make_multimodal("batch", seed=4, n_founders=16, n_genes=48)
    assert not np.allclose(batch.uns["batch_gene_effects"], 1.0)



def test_expected_truth_is_noise_free_and_close_to_observed(default_data, noise_free_data):
    """Expected-count truth ignores the Poisson spliced layer, agrees with the
    observed-count truth in direction, and equals it when there is no noise."""
    a = default_data.copy()
    embed_linear(a, method="pca", n_comp=12)
    exp_v = true_velocity_in_embedding(a, "pca", at="expected").astype(float)
    obs_v = true_velocity_in_embedding(a, "pca", at="observed").astype(float)
    cos = np.sum(exp_v * obs_v, 1) / (np.linalg.norm(exp_v, axis=1) * np.linalg.norm(obs_v, axis=1))
    assert np.median(cos) >= 0.95
    b = a.copy()
    b.layers["spliced"] = np.random.default_rng(0).poisson(np.asarray(b.layers["spliced"], float) + 1.0)
    assert np.allclose(true_velocity_in_embedding(b, "pca", at="expected"), exp_v, atol=1e-6)
    n = noise_free_data.copy()
    embed_linear(n, method="pca", n_comp=12)
    assert np.allclose(true_velocity_in_embedding(n, "pca", at="expected"),
                       true_velocity_in_embedding(n, "pca", at="observed"), rtol=1e-4, atol=1e-6)


def test_stationary_cycle_default_size():
    a = make_multimodal("stationary_cycle", seed=0)
    assert 2400 <= a.n_obs <= 3300, a.n_obs

# S05 generator-v2 regression coverage.  V1 remains in the original tests
# above; these tests intentionally use the historical writer shape for the
# stored default H5AD (without the later expected-count velocity keys).
from pathlib import Path
import anndata as ad
from scipy import sparse as scipy_sparse
from scjdo2.data.synthetic_multimodal import _v2_alpha


V2_DATA_PATH = Path(__file__).resolve().parents[1] / "benchmarks/synthetic_multimodal/data/v2/v2_default_s0.h5ad"


def _v2_gene_dict(adata):
    return {
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


@pytest.fixture(scope="module")
def v2_default_data():
    """Use the required local v2-default artifact when available.

    The fallback keeps the test self-contained for a fresh worktree; the task
    workflow writes the seed-zero artifact before this suite is invoked.
    """
    if V2_DATA_PATH.exists():
        return ad.read_h5ad(V2_DATA_PATH)
    return make_multimodal("v2_default", seed=0)


@pytest.fixture(scope="module")
def v2_noise_free_data():
    return make_multimodal(
        "v2_default", seed=11, n_founders=60, f_sample=0.20, poisson=False,
    )


def test_v1_default_seed_zero_matches_the_stored_writer_arrays():
    """The v1 dispatcher must recreate every stored default layer/obsm/obs array."""
    stored_path = Path(__file__).resolve().parents[1] / "benchmarks/synthetic_multimodal/data/default_s0.h5ad"
    stored = ad.read_h5ad(stored_path)
    regenerated = make_multimodal("default", seed=0)
    for method in ("pca", "fa"):
        embed_linear(regenerated, method=method, n_comp=20)
        true_velocity_in_embedding(regenerated, method=method)
    assert stored.obs.equals(regenerated.obs)
    assert stored.var.equals(regenerated.var)
    assert np.array_equal(stored.X, regenerated.X)
    assert set(stored.layers) == set(regenerated.layers)
    for key in stored.layers:
        assert np.array_equal(stored.layers[key], regenerated.layers[key]), key
    assert set(stored.obsm) == set(regenerated.obsm)
    for key in stored.obsm:
        left, right = stored.obsm[key], regenerated.obsm[key]
        if scipy_sparse.issparse(left):
            assert (left != right).nnz == 0, key
        else:
            assert np.array_equal(left, right), key


def test_v2_lazy_ancestral_pools_equal_full_population_integration():
    kwargs = dict(preset="v2_default", seed=17, n_founders=60, f_sample=0.20, poisson=False)
    lazy = make_multimodal(**kwargs)
    eager = make_multimodal(**kwargs, _v2_integration="eager")
    assert lazy.obs.equals(eager.obs)
    assert lazy.var.equals(eager.var)
    for key in ("u_true", "s_true", "u_new_true", "s_new_true", "s_prev_true", "ds_dt_true"):
        assert np.allclose(lazy.layers[key], eager.layers[key], rtol=1e-5, atol=1e-6), key


def test_v2_default_determinism_and_switch_metadata(v2_noise_free_data):
    first = make_multimodal("v2_default", seed=7, n_founders=60, f_sample=0.20, poisson=False)
    second = make_multimodal("v2_default", seed=7, n_founders=60, f_sample=0.20, poisson=False)
    assert first.obs.equals(second.obs)
    assert first.var.equals(second.var)
    for key in first.layers:
        assert np.array_equal(first.layers[key], second.layers[key]), key
    kinds = v2_noise_free_data.var["gene_kind"].value_counts().to_dict()
    assert kinds == {"linear": 560, "on": 80, "on_off": 80, "fate": 80}
    for column, used_by in (("theta", {"on", "on_off"}), ("delta", {"on_off"}), ("fate_sign", {"fate"})):
        values = v2_noise_free_data.var[column]
        kinds_array = v2_noise_free_data.var["gene_kind"].astype(str)
        assert values[kinds_array.isin(used_by)].notna().all(), column
        assert values[~kinds_array.isin(used_by)].isna().all(), column
    assert float(v2_noise_free_data.uns["synthetic_multimodal"]["capture_eta_unspliced"]) == 0.60


def test_v2_kinetics_and_labelling_are_analytic(v2_noise_free_data):
    a = v2_noise_free_data
    genes = _v2_gene_dict(a)
    alpha = _v2_alpha(np.asarray(a.obsm["z_true"], dtype=np.float32), genes)
    beta = a.var["beta_initial"].to_numpy(dtype=float)
    gamma = a.var["gamma"].to_numpy(dtype=float)
    _, later = _analytic_rna_step(a.layers["u_true"], a.layers["s_true"], alpha, beta, gamma, 1e-5)
    finite_difference = (later - np.asarray(a.layers["s_true"], dtype=float)) / 1e-5
    velocity = np.asarray(a.layers["ds_dt_true"], dtype=float)
    assert np.linalg.norm(finite_difference - velocity) / np.linalg.norm(velocity) <= 1e-3
    total = np.asarray(a.layers["total"], dtype=float)
    expected = np.asarray(a.layers["new_fraction_true"], dtype=float)
    observed = np.asarray(a.layers["new"], dtype=float) / (0.9 * np.maximum(total, 1e-12))
    assert np.max(np.abs(observed - expected)[total > 0]) <= 1e-3


@pytest.mark.parametrize("method", ("pca", "fa"))
def test_v2_truth_mapping_matches_finite_difference(v2_noise_free_data, method):
    a = v2_noise_free_data.copy()
    embed_linear(a, method=method, n_comp=12)
    velocity = true_velocity_in_embedding(a, method)
    finite_difference = finite_difference_embedding_velocity(a, method, dt=1e-5)
    cosine = np.sum(velocity * finite_difference, axis=1) / (
        np.linalg.norm(velocity, axis=1) * np.linalg.norm(finite_difference, axis=1) + 1e-12
    )
    assert np.median(cosine) >= 0.99


def test_v2_default_lineage_time_size_and_realism_gates(v2_default_data):
    a = v2_default_data
    assert 3800 <= a.n_obs <= 5800
    assert np.array_equal(a.obs["time_hours"].to_numpy(), a.obs["day"].to_numpy(dtype=float) * 24.0)
    detected = a.obs["clone"].notna().to_numpy()
    assert abs(float(detected.mean()) - 0.60) <= 0.03
    assert np.array_equal(a.obs.loc[detected, "clone"].astype(str), a.obs.loc[detected, "clone_true"].astype(str))
    spans = a.obs.loc[detected].groupby("clone", observed=True)["day"].nunique()
    assert int((spans >= 2).sum()) >= 50
    committed = a.obs.groupby("day", observed=True)["fate_true"].apply(lambda x: float((x != "progenitor").mean()))
    assert np.all(np.diff(committed.to_numpy()) >= 0.0)
    s = np.asarray(a.layers["spliced"], dtype=float)
    u = np.asarray(a.layers["unspliced"], dtype=float)
    share = float(u.sum() / (s.sum() + u.sum()))
    assert 0.10 <= share <= 0.30


def test_v2_on_genes_reach_their_high_steady_states(v2_default_data):
    a = v2_default_data
    kinds = a.var["gene_kind"].astype(str).to_numpy()
    on = np.flatnonzero(kinds == "on")
    z1 = np.asarray(a.obsm["z_true"], dtype=float)[:, 1]
    theta = a.var["theta"].to_numpy(dtype=float)
    gamma = a.var["gamma"].to_numpy(dtype=float)
    intercept = a.var["transcription_intercept"].to_numpy(dtype=float)
    s_true = np.asarray(a.layers["s_true"], dtype=float)
    qualifying = []
    for gene in on:
        cells = z1 > theta[gene] + 0.5
        target = np.exp(intercept[gene] + 1.5) / gamma[gene]
        close = np.abs(s_true[cells, gene] - target) / target < 0.10
        qualifying.append(bool(cells.any() and close.mean() >= 0.50))
    assert np.mean(qualifying) >= 0.50

def _expected_embedded_positions_for_test(adata, method):
    """Expected-count positions used for the amended forward-truth validation."""
    meta = adata.uns[f"embedding_{method}"]
    s_true = np.asarray(adata.layers["s_true"], dtype=np.float64)
    capture = adata.obs["capture_factor"].to_numpy(dtype=np.float64)
    day_index = adata.obs["batch_day"].to_numpy(dtype=int) - 1
    expected = s_true * capture[:, None] * np.asarray(adata.uns["batch_gene_effects"], dtype=np.float64)[day_index]
    transformed = np.log1p(expected * float(meta["normalization_target"]) / np.maximum(expected.sum(axis=1, keepdims=True), 1e-12))
    return (transformed - np.asarray(meta["mean"], dtype=np.float64)) @ np.asarray(meta["linear_map"], dtype=np.float64)


def _chord_truth_for_test(adata, method):
    z = _expected_embedded_positions_for_test(adata, method)
    clones = adata.obs["clone_true"].astype(str).to_numpy()
    days = adata.obs["day"].to_numpy(dtype=int)
    out = np.full_like(z, np.nan)
    for clone in np.unique(clones):
        idx = np.flatnonzero(clones == clone)
        observed = np.unique(days[idx])
        if len(observed) < 2:
            continue
        out[idx] = z[idx[days[idx] == observed.max()]].mean(axis=0) - z[idx[days[idx] == observed.min()]].mean(axis=0)
    return out


def _cosine(left, right):
    return float(np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-12))


def _dense_chord_cells(adata, minimum):
    clones = adata.obs["clone_true"].astype(str).to_numpy()
    days = adata.obs["day"].to_numpy(dtype=int)
    keep = np.zeros(adata.n_obs, dtype=bool)
    for clone in np.unique(clones):
        idx = np.flatnonzero(clones == clone)
        observed = np.unique(days[idx])
        if len(observed) < 2:
            continue
        early = int(observed.min())
        late = int(observed.max())
        if int((days[idx] == early).sum()) >= minimum and int((days[idx] == late).sum()) >= minimum:
            keep[idx] = True
    return keep


@pytest.mark.parametrize("path", (
    Path(__file__).resolve().parents[1] / "benchmarks/synthetic_multimodal/data/default_s0.h5ad",
    V2_DATA_PATH,
), ids=("v1_default_s0", "v2_default_s0"))
def test_forward_developmental_truth_amended_validation(path):
    """S06 Revision 1: population gate for both versions; dense chords for v1."""
    a = ad.read_h5ad(path)
    forward = forward_developmental_truth(a, "pca", horizon_hours=24.0, n_paths=8, seed=0)
    expected = _expected_embedded_positions_for_test(a, "pca")
    days = a.obs["day"].to_numpy(dtype=int)
    population = []
    for day in sorted(np.unique(days))[:-1]:
        next_day = day + 1
        if not np.any(days == next_day):
            continue
        score = _cosine(forward[days == day].mean(axis=0), expected[days == next_day].mean(axis=0) - expected[days == day].mean(axis=0))
        population.append((int(day), int(next_day), score))
        assert score >= 0.95
    chord = _chord_truth_for_test(a, "pca")
    finite = np.isfinite(chord).all(axis=1)
    cosine = np.sum(forward[finite] * chord[finite], axis=1) / np.maximum(
        np.linalg.norm(forward[finite], axis=1) * np.linalg.norm(chord[finite], axis=1), 1e-12
    )
    dense = _dense_chord_cells(a, minimum=3) & finite
    dense_cosine = np.sum(forward[dense] * chord[dense], axis=1) / np.maximum(
        np.linalg.norm(forward[dense], axis=1) * np.linalg.norm(chord[dense], axis=1), 1e-12
    )
    version = str(a.uns["synthetic_multimodal"].get("version", "v1"))
    print({
        "version": version,
        "population_day_pair_cosines": population,
        "all_chord_median": float(np.median(cosine)),
        "all_chord_n": int(finite.sum()),
        "dense_chord_median_min3": float(np.median(dense_cosine)) if dense.any() else float("nan"),
        "dense_chord_n_min3": int(dense.sum()),
    })
    if version == "v1":
        assert dense.any()
        assert float(np.median(dense_cosine)) >= 0.85

# S09 generator-v3 coverage.  These bounded overrides exercise the public
# v3_default law without turning the normal test suite into the full calibration.
from scjdo2.data.synthetic_multimodal import (
    _gene_parameters_v3, _make_v3_config, _state_drift, _state_jacobian, _v3_alpha,
)


@pytest.fixture(scope="module")
def v3_noise_free_data():
    return make_multimodal(
        "v3_default", seed=29, n_founders=18, n_genes=72, f_sample=0.05,
        poisson=False, n_prog=3,
    )


def test_v3_lazy_ancestral_pools_equal_full_population_integration():
    kwargs = dict(
        preset="v3_default", seed=31, n_founders=16, n_genes=64,
        f_sample=0.05, poisson=False, n_prog=2,
    )
    lazy = make_multimodal(**kwargs)
    eager = make_multimodal(**kwargs, _v3_integration="eager")
    assert lazy.obs.equals(eager.obs)
    assert lazy.var.equals(eager.var)
    for key in ("u_true", "s_true", "u_new_true", "s_new_true", "s_prev_true", "ds_dt_true"):
        assert np.allclose(lazy.layers[key], eager.layers[key], rtol=1e-5, atol=1e-6), key


def test_v3_default_determinism_kinetics_and_programmes(v3_noise_free_data):
    a = v3_noise_free_data
    repeat = make_multimodal(
        "v3_default", seed=29, n_founders=18, n_genes=72, f_sample=0.05,
        poisson=False, n_prog=3,
    )
    assert a.obs.equals(repeat.obs)
    for key in a.layers:
        assert np.array_equal(a.layers[key], repeat.layers[key]), key
    genes = _v2_gene_dict(a)
    alpha = _v3_alpha(np.asarray(a.obsm["z_true"], dtype=np.float32), genes)
    beta = a.var["beta_initial"].to_numpy(dtype=float)
    gamma = a.var["gamma"].to_numpy(dtype=float)
    _, later = _analytic_rna_step(a.layers["u_true"], a.layers["s_true"], alpha, beta, gamma, 1e-5)
    finite_difference = (later - np.asarray(a.layers["s_true"], dtype=float)) / 1e-5
    velocity = np.asarray(a.layers["ds_dt_true"], dtype=float)
    assert np.linalg.norm(finite_difference - velocity) / np.linalg.norm(velocity) <= 1e-3
    kinds = a.var["gene_kind"].astype(str).to_numpy()
    assert {"constant", "linear", "on", "on_off", "fate"} <= set(kinds)
    constant = np.flatnonzero(kinds == "constant")
    assert len(constant)
    swapped = np.asarray(a.obsm["z_true"], dtype=np.float32).copy()
    swapped[:, :] += 3.0
    assert np.allclose(_v3_alpha(swapped, genes)[:, constant], alpha[:, constant])


def test_v3_programme_rounding_keeps_a_nonnegative_linear_pool_at_simplex_boundary():
    config = _make_v3_config(
        "v3_default", 0,
        {"n_genes": 4000, "f_constant": 0.5921983516839394, "f_switch": 0.4078016483160606},
    )
    genes = _gene_parameters_v3(config, np.random.default_rng(0))
    counts = pd.Series(genes["gene_kind"]).value_counts()
    assert int(counts.sum()) == 4000
    assert int(counts.get("linear", 0)) >= 0
    assert int(counts.get("on", 0)) == int(counts.get("on_off", 0)) == int(counts.get("fate", 0))


def test_v3_labelling_lineage_and_nuisance_core_truth(v3_noise_free_data):
    a = v3_noise_free_data
    total = np.asarray(a.layers["total"], dtype=float)
    expected = np.asarray(a.layers["new_fraction_true"], dtype=float)
    observed = np.asarray(a.layers["new"], dtype=float) / (0.9 * np.maximum(total, 1e-12))
    assert np.max(np.abs(observed - expected)[total > 0]) <= 1e-5
    assert set(a.obs["pulse_hours"].unique()) == {3.0, 5.0}
    for day in sorted(a.obs["day"].unique()):
        pulses = a.obs.loc[a.obs["day"] == day, "pulse_hours"].to_numpy()
        assert abs(int((pulses == 3).sum()) - int((pulses == 5).sum())) <= 1
    detected = a.obs["clone"].notna().to_numpy()
    assert np.array_equal(a.obs.loc[detected, "clone"].astype(str), a.obs.loc[detected, "clone_true"].astype(str))
    matrix = a.obsm["clone_matrix"].tocsr()
    assert matrix.shape == (a.n_obs, len(a.uns["clone_ids"]))
    assert matrix.nnz == int(detected.sum())
    z = np.asarray(a.obsm["z_true"], dtype=np.float32)
    assert z.shape[1] == 9
    assert a.uns["synthetic_multimodal"]["core_dims"] == list(range(6))
    assert np.array_equal(a.obsm["v_state"], _state_drift(z[:, :6], False))
    assert np.array_equal(a.obsm["jacobian_state"], _state_jacobian(z[:, :6], False))
    assert a.obsm["omega_state"].shape == (a.n_obs, 6, 6)


def test_v3_truth_mapping_matches_finite_difference(v3_noise_free_data):
    a = v3_noise_free_data.copy()
    embed_linear(a, method="pca", n_comp=12)
    velocity = true_velocity_in_embedding(a, "pca", at="expected")
    finite_difference = finite_difference_embedding_velocity(a, "pca", dt=1e-5)
    cosine = np.sum(velocity * finite_difference, axis=1) / (
        np.linalg.norm(velocity, axis=1) * np.linalg.norm(finite_difference, axis=1) + 1e-12
    )
    assert np.median(cosine) >= 0.99


# orient/S10 Step 0 / 0b / 6 additions.
from scjdo2.data.synthetic_multimodal import (  # noqa: E402
    embed_with_stored_transform, forward_displacement_covariance, resample_spliced_counts,
)


def _arrays_equal(a, b):
    assert a.obs.equals(b.obs)
    assert a.var.equals(b.var)
    for key in a.layers:
        assert np.array_equal(np.asarray(a.layers[key]), np.asarray(b.layers[key])), key
    for key in ("z_true", "v_state", "jacobian_state", "omega_state"):
        assert np.array_equal(a.obsm[key], b.obsm[key]), key
    assert np.array_equal(a.uns["batch_gene_effects"], b.uns["batch_gene_effects"])


def test_s10_z1_crit_default_leaves_v1_v2_v3_unchanged():
    small = dict(seed=29, n_founders=18, n_genes=72, f_sample=0.05, poisson=False, n_prog=3)
    default = make_multimodal("v3_default", **small)
    explicit = make_multimodal("v3_default", z1_crit=1.0, pulse_batch_effect=False, **small)
    _arrays_equal(default, explicit)
    meta = default.uns["synthetic_multimodal"]
    assert "z1_crit" not in meta and "pulse_batch_effect" not in meta   # stored v3 metadata unchanged
    z = np.asarray(default.obsm["z_true"], dtype=np.float32)[:, :6]
    assert np.array_equal(_state_drift(z, False), _state_drift(z, False, 1.0))
    assert np.array_equal(_state_jacobian(z, False), _state_jacobian(z, False, 1.0))
    genes = _v2_gene_dict(default)
    assert np.array_equal(_v3_alpha(np.asarray(default.obsm["z_true"]), genes), _v3_alpha(np.asarray(default.obsm["z_true"]), genes, 1.0))


def test_s10_v3_transition_moves_the_fork_and_adds_pulse_batches():
    a = make_multimodal("v3_transition", seed=1, n_founders=30, n_genes=80, f_sample=0.05, n_prog=2, poisson=False)
    meta = a.uns["synthetic_multimodal"]
    assert float(meta["z1_crit"]) == 2.0 and bool(meta["pulse_batch_effect"])
    z = np.asarray(a.obsm["z_true"], dtype=np.float32)
    jac = a.obsm["jacobian_state"]
    assert np.allclose(jac[:, 0, 0], 0.5 * ((z[:, 1] - 2.0) - 3.0 * z[:, 0] ** 2), atol=1e-5)
    assert np.allclose(a.obsm["v_state"][:, 0], 0.5 * ((z[:, 1] - 2.0) * z[:, 0] - z[:, 0] ** 3), atol=1e-5)
    committed = z[:, 1] > 2.2
    assert np.array_equal(a.obs["fate_true"].astype(str).to_numpy() != "progenitor", committed)
    # mean z1 by day: day 2 before the fork, day 4 just after, day 6 beyond
    by_day = {d: float(z[a.obs["day"].to_numpy() == d, 1].mean()) for d in (2, 4, 6)}
    assert by_day[2] < 2.0 < by_day[4] < by_day[6]
    # cycle exit gate and fate-gene commitment gate follow z1_crit
    genes = _v2_gene_dict(a)
    probe = np.zeros((2, z.shape[1]), dtype=np.float32); probe[:, 0] = 1.0; probe[:, 1] = (1.5, 2.6)
    fate = np.flatnonzero(a.var["gene_kind"].astype(str).to_numpy() == "fate")
    lo = np.exp(genes["b"][fate] - 1.5)
    alpha = _v3_alpha(probe, genes, 2.0)[:, fate]
    assert np.allclose(alpha[0], lo, rtol=1e-4)            # z1 = 1.5 < 2.2: fate genes off
    assert not np.allclose(alpha[1], lo, rtol=1e-2)        # z1 = 2.6 > 2.2: switched on one side
    assert np.allclose(_state_drift(probe[:, :6], False, 2.0)[:, 2:4],
                       _state_drift(probe[:, :6] - np.array([0, 1, 0, 0, 0, 0], np.float32), False, 1.0)[:, 2:4], atol=1e-6)
    # pulse batches: 6 groups (day x pulse), separate per-gene factors, used for the counts
    groups = a.obs.groupby(["day", "pulse_hours"], observed=True)["batch_day"].unique()
    assert all(len(v) == 1 for v in groups) and sorted(int(v[0]) for v in groups) == [1, 2, 3, 4, 5, 6]
    effects = np.asarray(a.uns["batch_gene_effects"])
    assert effects.shape == (6, a.n_vars)
    assert np.allclose(effects, np.asarray(a.uns["pulse_batch_gene_effects"]))   # no day batch in v3_transition
    expected = a.layers["s_true"] * a.obs["capture_factor"].to_numpy()[:, None] * effects[a.obs["batch_day"].to_numpy() - 1]
    assert np.allclose(a.layers["spliced"], expected, rtol=1e-5)


def test_s14_v3_nofork_is_transition_measurement_law_without_a_sampled_fork():
    a = make_multimodal(
        "v3_nofork", seed=17, n_founders=24, n_genes=72, f_sample=0.05,
        n_prog=2, poisson=False,
    )
    cfg = _make_v3_config("v3_nofork", 17, {})
    assert float(cfg.z1_crit) == 5.0 and bool(cfg.pulse_batch_effect)
    meta = a.uns["synthetic_multimodal"]
    assert float(meta["z1_crit"]) == 5.0 and bool(meta["pulse_batch_effect"])
    z = np.asarray(a.obsm["z_true"], dtype=np.float32)
    assert float(z[:, 1].max()) < 5.0
    assert set(a.obs["fate_true"].astype(str)) == {"progenitor"}
    assert np.array_equal(a.obsm["v_state"], _state_drift(z[:, :6], False, 5.0))
    assert np.array_equal(a.obsm["jacobian_state"], _state_jacobian(z[:, :6], False, 5.0))

def test_s10_forward_displacement_covariance_and_noise_resampling():
    a = make_multimodal("v3_default", seed=4, n_founders=16, n_genes=60, f_sample=0.05, n_prog=2)
    embed_linear(a, method="pca", n_comp=8)
    cov = forward_displacement_covariance(a, ("pca",), (0.5, 1.0), n_paths=8, seed=0)
    c1, c2 = cov[("pca", 0.5)], cov[("pca", 1.0)]
    assert c1.shape == (a.n_obs, 8, 8)
    assert np.allclose(c1, np.swapaxes(c1, 1, 2), atol=1e-6)
    assert np.all(np.linalg.eigvalsh(c2.astype(float)) > -1e-6)
    assert np.median(np.trace(c2, axis1=1, axis2=2)) > np.median(np.trace(c1, axis1=1, axis2=2))
    again = forward_displacement_covariance(a, ("pca",), (0.5, 1.0), n_paths=8, seed=0)
    assert np.array_equal(again[("pca", 1.0)], c2)
    # no noise anywhere -> zero spread
    quiet = make_multimodal("v3_default", seed=4, n_founders=16, n_genes=60, f_sample=0.05, n_prog=0, sigma=0.0)
    embed_linear(quiet, method="pca", n_comp=8)
    assert np.allclose(forward_displacement_covariance(quiet, "pca", 1.0, n_paths=4), 0.0, atol=1e-10)
    # resampled counts: same law, different draw; stored transform reproduces the stored embedding
    redraw = resample_spliced_counts(a, noise_seed=1)
    assert redraw.shape == a.shape and not np.array_equal(redraw, np.asarray(a.layers["spliced"]))
    assert np.allclose(embed_with_stored_transform(a, a.layers["spliced"], "pca"), a.obsm["X_pca"], atol=1e-3)
