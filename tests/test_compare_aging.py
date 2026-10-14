"""Unit tests for scjdo2.compare — isotropy_test + anticipation_test.

PREREG §4a amendment (2026-10-14): isotropy_test now uses Spearman
correlation of matched |λ| against a random-pairing null; the old
CV-based gate was permutation-degenerate.

Three synthetic scenarios:
  * ISO: isotropic widening — Spearman null degenerate on ties → does
    NOT pass; anticipation also fails (no displacement).
  * COHERENT AXIS SET: both young and old have structured spectra
    with coherent ordering (soft stays soft; stiff stays stiff) +
    displacement along the widest axis — both tests PASS.
  * identity sanity (match_directions on J_a == J_b).

J is constructed as `-Σ⁻¹` (basin Hessian of a Gaussian), matching
what the lead-eig rule reads in the real pipeline.
"""

from __future__ import annotations

import numpy as np

from scjdo2.compare import (
    anticipation_test,
    isotropy_test,
    match_directions,
)


def _sigma_to_J(Sigma: np.ndarray) -> np.ndarray:
    return -np.linalg.inv(Sigma)


def test_isotropic_widening_fails_both():
    rng = np.random.default_rng(0)
    d = 10
    # Both Gaussians isotropic; "old" wider by factor sqrt(2) uniformly.
    sigma_a = 1.0
    sigma_b = sigma_a * np.sqrt(2.0)
    Sigma_a = (sigma_a ** 2) * np.eye(d)
    Sigma_b = (sigma_b ** 2) * np.eye(d)
    J_a = _sigma_to_J(Sigma_a)
    J_b = _sigma_to_J(Sigma_b)

    iso = isotropy_test(J_a, J_b, n_null=500, seed=0)
    # Spearman on all-tied inputs is NaN → passes stays False.
    assert not iso.passes, (
        f"isotropy_test should NOT pass for uniform widening — got "
        f"ρ_λ={iso.rho_lambda:.3e}, null_q95={iso.null_q95:.3e}"
    )

    # Zero displacement → |cos(u_a, d)| within the random-split null.
    n = 1000
    z_a = rng.multivariate_normal(np.zeros(d), Sigma_a, size=n)
    z_b = rng.multivariate_normal(np.zeros(d), Sigma_b, size=n)
    ant = anticipation_test(z_a, z_b, J_a, n_null=500, seed=0)
    assert not ant.passes, (
        f"anticipation_test should FAIL for zero-mean-shift isotropic — "
        f"got |cos|={ant.cos_abs:.3f}, null_q95={ant.null_q95:.3f}"
    )


def test_coherent_widening_plus_displacement_passes_both():
    """Both young and old have STRUCTURED spectra with coherent
    ordering — small-eigenvalue (soft) directions in young remain
    soft in old. Displacement along the softest axis.
    isotropy_test PASSES (Spearman > null_q95), anticipation PASSES.
    """
    rng = np.random.default_rng(1)
    d = 10
    # Young Σ: anisotropic log-spaced spectrum, sigma_a_i from 1 to 10
    # along successive axes.
    sigma_a = np.logspace(0, 1, d)
    # Old Σ: widen each axis coherently — softer axes widen MORE
    # (anticipatory loss), stiffer axes widen less. Preserves the
    # ordering of eigenvalues.
    widen = np.linspace(1.1, 3.0, d)                                   # softer first
    sigma_b = sigma_a * widen
    Sigma_a = np.diag(sigma_a ** 2)
    Sigma_b = np.diag(sigma_b ** 2)
    J_a = _sigma_to_J(Sigma_a)
    J_b = _sigma_to_J(Sigma_b)

    iso = isotropy_test(J_a, J_b, n_null=500, seed=0)
    assert iso.passes, (
        f"isotropy_test should PASS for direction-coherent widening — got "
        f"ρ_λ={iso.rho_lambda:.3f}, null_q95={iso.null_q95:.3f}"
    )

    # Soft axis = axis d-1 (largest sigma_a → smallest |λ_a|). Displace
    # old along this direction.
    v = np.zeros(d); v[-1] = 1.0
    mu_b = np.zeros(d); mu_b[-1] = 3.0 * sigma_a[-1]
    n = 1000
    z_a = rng.multivariate_normal(np.zeros(d), Sigma_a, size=n)
    z_b = rng.multivariate_normal(mu_b, Sigma_b, size=n)
    ant = anticipation_test(z_a, z_b, J_a, n_null=500, seed=0)
    assert ant.passes, (
        f"anticipation_test should PASS for coherent + displacement — got "
        f"|cos|={ant.cos_abs:.3f}, null_q95={ant.null_q95:.3f}"
    )
    assert abs(float(np.dot(ant.u_a, v))) > 0.9, (
        f"u_a should point along the widening/soft axis; "
        f"cos={np.dot(ant.u_a, v):.3f}"
    )


def test_match_directions_identity_is_identity():
    """When J_a == J_b, every direction matches itself with cos = 1."""
    rng = np.random.default_rng(2)
    d = 6
    A = rng.standard_normal((d, d))
    J = -0.5 * (A @ A.T + np.eye(d))                                # NSD
    md = match_directions(J, J)
    for i_a, i_b, c in md.pairs:
        assert i_a == i_b
        assert abs(c - 1.0) < 1e-6
    assert np.allclose(md.eig_a, md.eig_b)
