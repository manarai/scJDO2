"""Opspace / plot tests on the synthetic pitchfork, with scJDO2's one drift
(the energy drift, ``f = −∇E_θ``).

    - spectrum returns conjugate-free output.
    - figure_operator_space produces a non-empty PDF.
    - Removing the clock flag changes every axis label to say "relative".
    - drivers() and archetype_operators() stay tracked as pending (xfail).

The former v1-drift tests of field_gradient and fork behaviour against the
pitchfork's bifurcation time (τ_crit) were removed with the v1 drift: they
tested a time-dependent drift fitted with a pseudotime anchor, which the
time-independent energy drift does not have.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pytest

from scjdo2 import opspace, plot
from scjdo2.data import generate_pitchfork
from scjdo2.energy_drift import EnergyDriftConfig
from scjdo2.pipeline import fit_latent_field


pytestmark = pytest.mark.slow


def _fit(adata, n_epochs: int = 300, seed: int = 0):
    return fit_latent_field(
        adata, use_rep="X_pca",
        drift_config=EnergyDriftConfig(hidden=64, depth=3, n_epochs=n_epochs,
                                       seed=seed, batch_size=256),
    )


def _run_opspace(adata):
    opspace.spectrum(adata, n_eig=5, fisher_mask=False)
    opspace.invariants(adata)
    opspace.behaviour(adata)
    opspace.field_gradient(adata, k=15)
    return adata


# ----------------------------------------------------------------------
# Spectrum: conjugate-free output
# ----------------------------------------------------------------------


def test_spectrum_conjugate_free():
    ad = generate_pitchfork(tau_crit=0.5, seed=0, n_cells=400,
                            n_genes=100, n_latent=10)
    _fit(ad, n_epochs=200, seed=0)
    opspace.spectrum(ad, n_eig=5, fisher_mask=False)

    eigs = np.asarray(ad.obsm["op_eigvals"])
    for c in range(ad.n_obs):
        row = eigs[c][~np.isnan(eigs[c].real)]
        if row.size < 2:
            continue
        # Pair-wise: no two should be complex conjugates (unless both near-real).
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                if abs(row[i].imag) < 1e-10 and abs(row[j].imag) < 1e-10:
                    continue
                assert not (
                    abs(row[i].real - row[j].real) < 1e-9
                    and abs(row[i].imag + row[j].imag) < 1e-9
                ), f"cell {c}: {row[i]} and {row[j]} are conjugates"


# ----------------------------------------------------------------------
# figure_operator_space produces a non-empty PDF
# ----------------------------------------------------------------------


def test_figure_operator_space_saves_pdf(tmp_path):
    ad = generate_pitchfork(tau_crit=0.5, seed=0, n_cells=500,
                            n_genes=100, n_latent=10)
    _fit(ad, n_epochs=300, seed=0)
    _run_opspace(ad)

    out = tmp_path / "op"
    fig = plot.figure_operator_space(ad, out=out)
    assert (tmp_path / "op.pdf").exists()
    assert (tmp_path / "op.png").exists()
    assert (tmp_path / "op.pdf").stat().st_size > 1024
    plt.close(fig)


# ----------------------------------------------------------------------
# Clock flag changes axis labels
# ----------------------------------------------------------------------


def test_axis_labels_relative_when_not_clock_calibrated():
    ad = generate_pitchfork(tau_crit=0.5, seed=0, n_cells=200,
                            n_genes=50, n_latent=10)
    _fit(ad, n_epochs=100, seed=0)
    _run_opspace(ad)

    fig = plot.figure_operator_space(ad)
    axes = fig.axes
    labels = [ax.get_xlabel() + ax.get_ylabel() for ax in axes]
    joined = " ".join(labels)
    assert "relative" in joined, "expected 'relative' in axis labels when not clock-calibrated"
    plt.close(fig)


def test_axis_labels_strip_relative_when_clock_calibrated():
    ad = generate_pitchfork(tau_crit=0.5, seed=0, n_cells=200,
                            n_genes=50, n_latent=10)
    _fit(ad, n_epochs=100, seed=0)
    _run_opspace(ad)
    ad.uns["scjdo2"]["meta"]["clock_calibrated"] = True

    fig = plot.figure_operator_space(ad)
    for ax in fig.axes:
        assert "relative" not in ax.get_xlabel() + ax.get_ylabel()
    plt.close(fig)


# ----------------------------------------------------------------------
# Phase C xfails — kept visible so the gap is tracked.
# ----------------------------------------------------------------------


@pytest.mark.xfail(
    reason="drivers() needs gene-space Jacobians — Phase C work (see V2_PLAN R11)",
    strict=True,
)
def test_drivers_fork_archetype_pending_phase_c():
    ad = generate_pitchfork(tau_crit=0.5, seed=0, n_cells=200,
                            n_genes=50, n_latent=10)
    _fit(ad, n_epochs=50, seed=0)
    opspace.spectrum(ad, n_eig=3, fisher_mask=False)
    opspace.drivers(ad, which="lead", top=10)  # raises NotImplementedError


@pytest.mark.xfail(
    reason="archetype_operators() needs scjdo2.archetypes adapter — Phase B Step 8",
    strict=True,
)
def test_archetype_operators_pending_step_8():
    ad = generate_pitchfork(tau_crit=0.5, seed=0, n_cells=200,
                            n_genes=50, n_latent=10)
    _fit(ad, n_epochs=50, seed=0)
    opspace.spectrum(ad, n_eig=3, fisher_mask=False)
    opspace.archetype_operators(ad)
