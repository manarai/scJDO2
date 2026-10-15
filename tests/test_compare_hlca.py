"""Synthetic checks for the continuous-age, donor-level tests
(PREREG_aging_hlca.md §4–§6; scjdo2.compare.*_trend).

Donors have ages 10–76, their own random scale and centroid jitter
(donor-to-donor variation), and Gaussian cells with per-cell Jacobians
``J_i = −Σ^{-1/2}(I + E_i)Σ^{-1/2}``. Scenarios: no aging; uniform
widening with age; single-axis widening with age; centroid drift along
the young soft axis with age.
"""

from __future__ import annotations

import numpy as np

from scjdo2.compare import (anticipation_trend, centroid_operator,
                            donor_operator, isotropy_trend, stiffness_trend)

D = 10
SIG = np.logspace(0, 1, D)                       # softest axis = last


def _cells(sigma, mu, n, rng, noise=0.15):
    z = mu + rng.standard_normal((n, D)) * sigma
    E = rng.standard_normal((n, D, D)) * noise
    E = 0.5 * (E + np.transpose(E, (0, 2, 1)))
    s = 1.0 / sigma
    return z, -(s[None, :, None] * (np.eye(D)[None] + E) * s[None, None, :])


def simulate(scenario, seed, n_donors=40, n_cells=80, n_common=50, n_draws=10):
    rng = np.random.default_rng(seed)
    ages = rng.uniform(10, 76, n_donors)
    ops, cents, young_z, young_J = [], [], [], []
    for a in ages:
        sig = SIG * np.exp(rng.normal(0, 0.05))            # donor jitter
        mu = rng.normal(0, 0.3, D) * SIG                   # donor offset
        t = a - 10
        if scenario == "uniform":
            sig = sig * (1 + 0.01 * t)
        elif scenario == "single_axis":
            sig = sig.copy(); sig[-1] *= 1 + 0.02 * t
        elif scenario == "drift":
            mu = mu.copy(); mu[-1] += 0.03 * t * SIG[-1]
        z, J = _cells(sig, mu, n_cells, rng)
        op, c = donor_operator(z, J, n_common, n_draws=n_draws, seed=int(rng.integers(1 << 31)))
        ops.append(op); cents.append(c)
        if a < 45:
            young_z.append(z); young_J.append(J)
    ref = centroid_operator(np.vstack(young_z), np.vstack(young_J), 30)
    return ops, cents, ages, ref


def run(scenario, seed, n_perm=200):
    ops, cents, ages, ref = simulate(scenario, seed)
    return (isotropy_trend(ops, ages, ref, n_perm=n_perm, seed=seed),
            anticipation_trend(cents, ages, ref, n_perm=n_perm, seed=seed),
            stiffness_trend(ops, ages, n_perm=n_perm, seed=seed))


def test_no_aging_fails_all():
    iso, ant, st = run("none", 0)
    assert not iso.passes and not ant.passes and not st.passes


def test_uniform_widening_fails_isotropy_but_softens():
    iso, ant, st = run("uniform", 1)
    assert not iso.passes, f"S={iso.statistic:.4f} q95={iso.null_q95:.4f}"
    assert st.passes and st.statistic < 0           # softer with age


def test_single_axis_widening_passes_isotropy():
    iso, ant, st = run("single_axis", 2)
    assert iso.passes, f"S={iso.statistic:.4f} q95={iso.null_q95:.4f}"


def test_drift_along_soft_axis_passes_anticipation():
    iso, ant, st = run("drift", 3)
    assert ant.passes, f"cos={ant.statistic:.3f} q95={ant.null_q95:.3f}"
    assert abs(ant.detail["u_ref"][-1]) > 0.9


def test_donor_operator_equal_count_guard():
    rng = np.random.default_rng(4)
    z, J = _cells(SIG, np.zeros(D), 40, rng)
    try:
        donor_operator(z, J, 50)
    except ValueError:
        return
    raise AssertionError("expected ValueError for n < n_common")


def test_strata_none_is_identical_to_primary():
    ops, cents, ages, ref = simulate("single_axis", 7)
    a = isotropy_trend(ops, ages, ref, n_perm=100, seed=0)
    b = isotropy_trend(ops, ages, ref, n_perm=100, seed=0, strata=None)
    assert a.statistic == b.statistic and np.array_equal(a.null, b.null)


def test_stratified_permutation_keeps_ages_within_stratum():
    from scjdo2.compare import _perm_ages
    ages = np.array([20., 30., 40., 50., 60., 70.])
    strata = np.array(["s1", "s1", "s1", "s2", "s2", "s2"])
    P = _perm_ages(ages, 50, 0, strata)
    assert np.all(np.sort(P[:, :3], axis=1) == [20., 30., 40.])
    assert np.all(np.sort(P[:, 3:], axis=1) == [50., 60., 70.])


def test_stratified_null_removes_study_confounded_age_signal():
    """Two studies with different stiffness and different age ranges:
    unstratified stiffness_trend sees a slope; stratified does not."""
    rng = np.random.default_rng(8)
    ops, ages, strata = [], [], []
    for study, (lo, hi, scale) in {"young_study": (10, 40, 1.0), "old_study": (50, 76, 1.5)}.items():
        for a in rng.uniform(lo, hi, 15):
            z, J = _cells(SIG * scale * np.exp(rng.normal(0, 0.03)), np.zeros(D), 60, rng)
            ops.append(donor_operator(z, J, 50, n_draws=5, seed=int(rng.integers(1 << 31)))[0])
            ages.append(a); strata.append(study)
    assert stiffness_trend(ops, ages, n_perm=200, seed=0).passes
    assert not stiffness_trend(ops, ages, n_perm=200, seed=0, strata=strata).passes
