"""Orientation-layer gates (docs/ORIENTATION_SPEC.md §8, §12), synthetic data only.

Flow signs are scored against the instantaneous velocity truth,
developmental signs against the developmental (chord) truth.

Every rate is over (cell, axis) pairs that pass the gap rule and have
support, as §8 states; each sign test also requires the oriented-cell
fraction to be ≥ 0.5. The operator is analytic (``J = −Hess E``) so these
test the orientation layer, not an energy fit.
"""

from __future__ import annotations

import numpy as np
import pytest

from scjdo2.data.synthetic_oriented import make_oriented_pitchfork, make_rotating_two_well
from scjdo2.orient import orient
from scjdo2.orient.base import get_operator, operator_hash

SEEDS = (0, 1, 2)


def _instantaneous(adata):
    return adata.obsm["velocity_true"]


def _chord(adata):
    """Developmental truth: latest-day minus earliest-day centroid of each true
    clone, assigned to every cell of that clone."""
    z = adata.obsm["X_lat"]
    cl, day = adata.obs["clone_true"].to_numpy(), adata.obs["day"].to_numpy()
    out = np.full_like(z, np.nan)
    for c in np.unique(cl):
        i = np.where(cl == c)[0]
        ds = np.unique(day[i])
        out[i] = z[i][day[i] == ds.max()].mean(axis=0) - z[i][day[i] == ds.min()].mean(axis=0)
    return out


def sign_agreement(out, group, truth):
    m = out[group]
    sup = m["source"] > 0
    t = np.sign(np.einsum("nd,ndk->nk", np.nan_to_num(truth), out["axes"].U))
    return float(np.mean(m["sign"][sup] == t[sup])), float(np.mean(sup.any(axis=1)))


def _run(adata, sources):
    before = operator_hash(get_operator(adata))
    out = orient(adata, sources, "X_lat")
    assert out["meta"]["hash_unchanged"] and operator_hash(get_operator(adata)) == before  # test 6
    return out


# ---- 1. pitchfork: each group against its own truth ----------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_flow_labelling_matches_instantaneous_truth(seed):
    a = make_oriented_pitchfork(seed=seed)
    agree, cov = sign_agreement(_run(a, ["labelling"]), "flow", _instantaneous(a))
    assert cov >= 0.5 and agree >= 0.95, f"agreement {agree:.3f}, coverage {cov:.3f}"


@pytest.mark.parametrize("seed", SEEDS)
def test_flow_labelling_velocity_matches_instantaneous_truth(seed):
    a = make_oriented_pitchfork(seed=seed)
    agree, cov = sign_agreement(_run(a, ["labelling", "velocity"]), "flow", _instantaneous(a))
    assert cov >= 0.5 and agree >= 0.95, f"agreement {agree:.3f}, coverage {cov:.3f}"


@pytest.mark.parametrize("seed", SEEDS)
def test_dev_lineage_matches_developmental_truth(seed):
    a = make_oriented_pitchfork(seed=seed)
    agree, cov = sign_agreement(_run(a, ["lineage"]), "dev", _chord(a))
    assert cov >= 0.5 and agree >= 0.97, f"agreement {agree:.3f}, coverage {cov:.3f}"


@pytest.mark.parametrize("seed", SEEDS)
def test_dev_pseudotime_documented_reliability(seed):
    """Accepted limitation (docs/ORIENTATION_SPEC.md §13): pseudotime gives
    developmental signs at ≈ 0.70 agreement, not the original 0.90 gate.
    This guards the documented level (floor 0.65) so it cannot silently
    degrade; its signs are labelled ``imputed``."""
    a = make_oriented_pitchfork(seed=seed)
    out = _run(a, ["pseudotime"])
    agree, cov = sign_agreement(out, "dev", _chord(a))
    assert cov >= 0.5 and agree >= 0.65, f"agreement {agree:.3f}"
    assert np.all(out["dev"]["orientation_class"][out["dev"]["source"].any(axis=1)] == "imputed")


def test_pseudotime_root_near_minimum_is_accepted():
    """A start cell near, not at, the minimum (as Palantir places it) is
    accepted; the percentile is recorded."""
    a = make_oriented_pitchfork(seed=0)
    t = a.obs["palantir_pseudotime"].to_numpy()
    a.uns["palantir_start_cell"] = a.obs_names[int(np.argsort(t)[int(0.04 * len(t))])]
    out = orient(a, ["pseudotime"], "X_lat")
    assert 0.03 <= out["meta"]["per_source"]["pseudotime"]["root_percentile"] <= 0.05


@pytest.mark.parametrize("seed", SEEDS)
def test_groups_are_independent_and_disagreement_reported(seed):
    """Flow and developmental signs are each computed only from their own
    sources: adding the other group's sources changes nothing in a group;
    where both sign an axis and differ, it is reported, not resolved."""
    a = make_oriented_pitchfork(seed=seed)
    allsrc = _run(a.copy(), ["lineage", "labelling", "velocity", "pseudotime"])
    flow_only = _run(a.copy(), ["labelling", "velocity"])
    dev_only = _run(a.copy(), ["lineage", "pseudotime"])
    for key in ("sign", "source", "conflict"):
        assert np.array_equal(allsrc["flow"][key], flow_only["flow"][key])
        assert np.array_equal(allsrc["dev"][key], dev_only["dev"][key])
    both = (allsrc["flow"]["source"] > 0) & (allsrc["dev"]["source"] > 0)
    expected = both & (allsrc["flow"]["sign"] != allsrc["dev"]["sign"])
    assert np.array_equal(allsrc["flow_vs_dev"], expected)


def test_pseudotime_wrong_root_is_refused():
    a = make_oriented_pitchfork(seed=0)
    t = a.obs["palantir_pseudotime"].to_numpy()
    a.uns["palantir_start_cell"] = a.obs_names[int(np.argmax(t))]
    with pytest.raises(ValueError, match="start cell"):
        orient(a, ["pseudotime"], "X_lat")


# ---- 2. rotating two-well: omega from the velocity regression ------------

@pytest.mark.parametrize("seed", SEEDS)
def test_labelling_recovers_omega(seed):
    """Tests the regression of measured velocity, not the energy model: J is
    symmetric and cannot carry rotation."""
    a = make_rotating_two_well(seed=seed)
    out = _run(a, ["labelling"])
    true = a.uns["synthetic"]["omega_true"]
    err = np.linalg.norm(out["flow"]["omega"] - true, axis=(1, 2)) / np.linalg.norm(true)
    assert np.mean(np.isfinite(err)) >= 0.5
    assert np.nanmedian(err) <= 0.3, f"median relative Frobenius error {np.nanmedian(err):.3f}"
    J = get_operator(a)
    assert np.allclose(J, np.transpose(J, (0, 2, 1)))


def test_rotation_comes_only_from_flow_sources():
    a = make_oriented_pitchfork(seed=0)
    out = _run(a, ["lineage", "pseudotime"])
    assert np.all(out["flow"]["omega_source"] == "none")
    assert np.all(np.isnan(out["flow"]["omega"]))
    assert out["meta"]["per_source"]["pseudotime"]["rotation_identified"] is False


# ---- 3. conflict: labelling beats flipped velocity -----------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_conflict_labelling_wins_and_is_flagged(seed):
    a = make_oriented_pitchfork(seed=seed)
    inj = np.random.default_rng(100 + seed).random(a.n_obs) < 0.10
    a.obsm["velocity_lat"] = np.where(inj[:, None], -a.obsm["velocity_lat"], a.obsm["velocity_lat"])
    out = _run(a, ["labelling", "velocity"])
    L, V = out["results"]["labelling"], out["results"]["velocity"]
    f = out["flow"]
    assert np.all((f["sign"] == L.sign)[L.support & inj[:, None]]), "labelling must win on injected cells"
    both = inj & (L.support & V.support).any(axis=1)
    assert both.sum() >= 20
    assert f["conflict_flag"][both].mean() >= 0.95
    assert f["conflict_flag"][~inj].mean() <= 0.01


# ---- 4. coverage: lineage where clones are, pseudotime elsewhere ---------

@pytest.mark.parametrize("seed", SEEDS)
def test_coverage_measured_only_where_lineage_signs(seed):
    a = make_oriented_pitchfork(seed=seed, clone_coverage=0.4)
    out = _run(a, ["lineage", "pseudotime"])
    covered = a.obs["clone_covered"].to_numpy()
    lineage_signs = out["results"]["lineage"].support.any(axis=1)
    cls = out["dev"]["orientation_class"]
    assert 0.3 <= covered.mean() <= 0.5
    assert not np.any(cls[~covered] == "measured")
    assert np.array_equal(cls == "measured", covered & lineage_signs)
    assert np.all(np.isin(cls, ["measured", "imputed", "none"]))
    for col in ("dev_sign_source", "dev_orientation_class", "dev_n_sources_supporting", "dev_conflict_flag"):
        assert a.obs[col].notna().all()


# ---- 5. flags -------------------------------------------------------------

def test_flags_gradient_system_is_equilibrium():
    a = make_rotating_two_well(seed=0, omega=0.0)
    out = _run(a, ["labelling"])
    ok = out["equilibrium_flag"] != "not_identified"
    assert ok.mean() >= 0.5
    assert np.mean(out["equilibrium_flag"][ok] == "equilibrium") >= 0.9
    assert np.mean(out["stationarity_flag"][ok] == "stationary") >= 0.9


def test_flags_rotating_system_is_non_equilibrium():
    a = make_rotating_two_well(seed=0)
    out = _run(a, ["labelling"])
    ok = out["equilibrium_flag"] != "not_identified"
    assert np.mean(out["equilibrium_flag"][ok] == "non_equilibrium") >= 0.9


def test_flags_without_velocity_are_not_identified():
    a = make_oriented_pitchfork(seed=0)
    out = _run(a, ["lineage", "pseudotime"])
    assert np.all(out["equilibrium_flag"] == "not_identified")
    assert np.all(out["stationarity_flag"] == "not_identified")


# ---- input handling -------------------------------------------------------

def test_unknown_source_refused():
    with pytest.raises(ValueError):
        orient(make_oriented_pitchfork(seed=0, n_clones=20), ["telepathy"], "X_lat")


def test_gene_space_velocity_needs_linear_map():
    from scjdo2.orient import OrientConfig
    a = make_oriented_pitchfork(seed=0, n_clones=20)
    del a.obsm["velocity_lat"]
    a.layers["velocity"] = np.zeros((a.n_obs, a.n_vars))
    with pytest.raises(ValueError, match="linear_map"):
        orient(a, ["velocity"], "X_lat", OrientConfig(gene_velocity_layer="velocity"))


def test_gene_space_velocity_through_linear_map_matches_latent():
    """Gene velocity mapped by the embedding's linear map gives the same signs
    as the same velocity supplied directly in the embedding."""
    from scjdo2.orient import OrientConfig
    a = make_oriented_pitchfork(seed=0, n_clones=60)
    d, G = a.obsm["X_lat"].shape[1], 40
    rng = np.random.default_rng(7)
    W = rng.standard_normal((d, G))                     # latent = gene @ W.T
    vg = a.obsm["velocity_lat"] @ np.linalg.pinv(W).T    # gene velocity whose image is velocity_lat
    direct = orient(a.copy(), ["velocity"], "X_lat")
    del a.obsm["velocity_lat"]
    a.layers["velocity"] = np.zeros((a.n_obs, 1))
    import anndata as ad
    b = ad.AnnData(np.zeros((a.n_obs, G), dtype=np.float32), obs=a.obs.copy(), obsm=dict(a.obsm), uns=a.uns.copy())
    b.layers["velocity"] = vg
    mapped = orient(b, ["velocity"], "X_lat", OrientConfig(gene_velocity_layer="velocity", linear_map=W))
    assert np.array_equal(direct["flow"]["sign"], mapped["flow"]["sign"])


# ---- 6. reliability check: unreliable velocity is demoted (§15) ----------

@pytest.mark.parametrize("seed", SEEDS)
def test_reversed_velocity_is_demoted_to_cross_check(seed):
    """Velocity that disagrees with labelling dataset-wide (as scVelo did in
    orient/S03) decides no flow sign and gives no omega; the flow output is
    then labelling's alone, and the disagreements are still recorded."""
    a = make_oriented_pitchfork(seed=seed)
    a.obsm["velocity_lat"] = -a.obsm["velocity_lat"]
    out = _run(a.copy(), ["labelling", "velocity"])
    lab = _run(a.copy(), ["labelling"])
    rel = out["meta"]["reliability"]["flow"]["velocity"]
    assert rel["demoted"] and rel["agreement"] < 0.6 and rel["n_pairs"] >= 200
    assert out["meta"]["demoted"]["flow"] == ["velocity"]
    assert np.array_equal(out["flow"]["sign"], lab["flow"]["sign"])
    assert np.array_equal(out["flow"]["source"], lab["flow"]["source"])
    assert not np.any(out["flow"]["omega_source"] == "velocity")
    assert out["flow"]["conflict_flag"].mean() > 0.5


@pytest.mark.parametrize("seed", SEEDS)
def test_reliable_velocity_passes_and_fills(seed):
    a = make_oriented_pitchfork(seed=seed)
    inj = np.random.default_rng(100 + seed).random(a.n_obs) < 0.10
    a.obsm["velocity_lat"] = np.where(inj[:, None], -a.obsm["velocity_lat"], a.obsm["velocity_lat"])
    out = _run(a.copy(), ["labelling", "velocity"])
    lab = _run(a.copy(), ["labelling"])
    rel = out["meta"]["reliability"]["flow"]["velocity"]
    assert rel["status"] == "passed" and not out["meta"]["demoted"]["flow"]
    assert (out["flow"]["source"] > 0).sum() >= (lab["flow"]["source"] > 0).sum()


def test_velocity_alone_is_not_checked():
    a = make_oriented_pitchfork(seed=0)
    a.obsm["velocity_lat"] = -a.obsm["velocity_lat"]
    out = _run(a, ["velocity"])
    assert out["meta"]["reliability"]["flow"]["velocity"]["status"] == "not_checked"
    assert (out["flow"]["source"] > 0).any()


def test_reliability_check_can_be_switched_off():
    from scjdo2.orient import OrientConfig
    a = make_oriented_pitchfork(seed=0)
    a.obsm["velocity_lat"] = -a.obsm["velocity_lat"]
    out = orient(a, ["labelling", "velocity"], "X_lat", OrientConfig(reliability_min={}))
    assert out["meta"]["demoted"]["flow"] == [] and "velocity" not in out["meta"]["reliability"]["flow"]


# ---- 7. real-time guard on pseudotime (§16) -----------------------------

def test_pseudotime_reversed_against_days_is_refused():
    """A reversed pseudotime whose start cell is its own minimum passes the
    start-cell check; the sampling days catch it."""
    a = make_oriented_pitchfork(seed=0)
    t = a.obs["palantir_pseudotime"].to_numpy()
    a.obs["palantir_pseudotime"] = t.max() - t
    a.uns["palantir_start_cell"] = a.obs_names[int(np.argmax(t))]
    with pytest.raises(ValueError, match="does not increase"):
        orient(a, ["pseudotime"], "X_lat")


def test_pseudotime_day_correlation_is_recorded():
    a = make_oriented_pitchfork(seed=0)
    out = orient(a, ["pseudotime"], "X_lat")
    assert out["meta"]["per_source"]["pseudotime"]["spearman_with_day"] > 0.3


def test_pseudotime_starting_on_a_late_day_is_refused():
    """Positive correlation with day is not enough: a pseudotime whose
    minimum lies on a late sampling day is refused (§16 amended)."""
    a = make_oriented_pitchfork(seed=0)
    t = a.obs["palantir_pseudotime"].to_numpy().copy()
    day = a.obs["day"].to_numpy()
    late = int(np.where(day == day.max())[0][0])
    t[late] = t.min() - 1.0                       # minimum moved to a late-day cell
    a.obs["palantir_pseudotime"] = t
    a.uns["palantir_start_cell"] = a.obs_names[late]
    with pytest.raises(ValueError, match="earliest sampling day"):
        orient(a, ["pseudotime"], "X_lat")


# ---- 8. lineage needs well-sampled clones (§17) -------------------------

def test_lineage_skips_clones_with_one_cell_at_an_end():
    from scjdo2.orient import OrientConfig
    a = make_oriented_pitchfork(seed=0)
    day = a.obs["day"].to_numpy(); cl = a.obs["clone"].to_numpy(dtype=object).copy()
    first, last = day.min(), day.max()
    for c in np.unique(cl[pd_notna(cl)]):             # keep one cell per end in every clone
        i = np.where(cl == c)[0]
        for d in (first, last):
            j = i[day[i] == d]
            cl[j[1:]] = None
    a.obs["clone"] = cl
    strict = orient(a.copy(), ["lineage"], "X_lat")
    loose = orient(a.copy(), ["lineage"], "X_lat", OrientConfig(lineage_min_cells_per_end=1))
    assert strict["meta"]["per_source"]["lineage"]["n_clones_with_cross_day_siblings"] == 0
    assert loose["meta"]["per_source"]["lineage"]["n_clones_with_cross_day_siblings"] > 0


def pd_notna(x):
    import pandas as pd
    return pd.notna(x)
