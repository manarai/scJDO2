"""The measured-flow figure path must stay intact for datasets whose
Jacobian is non-symmetric (RNA velocity, labelled flow, organoid
time-series). Snapshot defaults route around it but nothing is
removed from ``scjdo2.plot``.

Test: render ``figure_operator_space(measured_flow=True)`` on the
equilibrated pitchfork with a SYNTHETIC asymmetric J (random
antisymmetric perturbation added to the symmetric part). The
measured-flow Panel B (complex plane) and Panel A's oscillator ring
must fire without errors.
"""

from __future__ import annotations

import numpy as np
import pytest

from scjdo2.data import generate_pitchfork_equilibrated


@pytest.fixture(scope="module")
def asym_pitchfork_adata():
    """Equilibrated pitchfork with a synthetic ASYMMETRIC J so the
    measured-flow Panel B has real work to do."""
    a = generate_pitchfork_equilibrated(
        tau_crit=0.3, seed=0, n_cells=200, n_alpha=5,
        n_genes=50, n_latent=5, noise_sigma=0.3,
        T_burn=10.0, T_sample=2.0, dt=0.05, sigma_sde=0.2,
        init_box=2.5,
    )
    n, d = a.n_obs, a.obsm["X_pca"].shape[1]
    rng = np.random.default_rng(0)
    J_sym = -np.eye(d) * 0.5 + 0.1 * rng.standard_normal((d, d))
    J_sym = 0.5 * (J_sym + J_sym.T)
    # Antisymmetric perturbation per cell → non-zero lead_im.
    A = 0.3 * rng.standard_normal((n, d, d))
    A = 0.5 * (A - A.transpose(0, 2, 1))
    J = J_sym[None, :, :] + A
    a.uns.setdefault("scjdo2", {})["J_latent"] = J.astype(np.float32)
    a.uns["scjdo2"]["J"] = J.astype(np.float32)
    a.uns["scjdo2"]["meta"] = {
        "use_rep": "X_pca", "clock_calibrated": False,
        "jacobian_method": "autodiff", "drift_param": "test_asymmetric",
    }
    from scjdo2 import opspace
    opspace.spectrum(a, n_eig=5, fisher_mask=False, smooth_k=0)
    opspace.invariants(a)
    opspace.behaviour(a)
    return a


def test_measured_flow_operator_space_renders(asym_pitchfork_adata, tmp_path):
    """Panel B complex-plane + Panel A oscillator ring both exercise
    code paths that only fire when lead_im > 0."""
    import matplotlib
    matplotlib.use("Agg")
    from scjdo2.plot import figure_operator_space

    a = asym_pitchfork_adata
    assert float(np.abs(a.obs["lead_im"].to_numpy()).max()) > 1e-6, \
        "fixture should have non-zero lead_im"

    fig = figure_operator_space(a, use_rep="X_pca", by="lead_re",
                                 measured_flow=True,
                                 out=tmp_path / "measured_flow_pitchfork")
    assert fig is not None
    axes = fig.get_axes()
    assert len(axes) >= 3, "operator_space has three panels"
    # Panel B axes should contain scatter points (complex plane); its
    # xlabel references lead_re.
    panel_b = axes[1]
    assert panel_b.get_xlabel() or panel_b.collections, \
        "Panel B must have content under measured_flow=True"
    # PDFs exist.
    assert (tmp_path / "measured_flow_pitchfork.pdf").exists()
    assert (tmp_path / "measured_flow_pitchfork.png").exists()


def test_snapshot_default_routes_to_spectrum_ladder(asym_pitchfork_adata,
                                                     tmp_path):
    """Under ``measured_flow=False`` (default) Panel B is the
    spectrum-ladder, not the complex plane. Guard that the alt path
    still renders without error."""
    import matplotlib
    matplotlib.use("Agg")
    from scjdo2.plot import figure_operator_space

    fig = figure_operator_space(
        asym_pitchfork_adata, use_rep="X_pca", by="lead_re",
        measured_flow=False, out=tmp_path / "snapshot_default_pitchfork",
    )
    assert fig is not None
    assert (tmp_path / "snapshot_default_pitchfork.pdf").exists()
