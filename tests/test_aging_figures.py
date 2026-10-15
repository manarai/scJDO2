"""Synthetic tests for the A04 aging figure extensions."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData

from scjdo2 import plot


def _toy_adata() -> AnnData:
    rng = np.random.default_rng(17)
    labels: list[str] = []
    ages: list[str] = []
    points: list[np.ndarray] = []
    for cell_type, centre in (("A", np.array([0.0, 0.0])), ("B", np.array([3.0, 1.0]))):
        for age, shift in (("young", np.array([0.0, 0.0])), ("old", np.array([0.45, 0.15]))):
            for _ in range(14):
                labels.append(cell_type)
                ages.append(age)
                points.append(centre + shift + 0.11 * rng.normal(size=2))
    z = np.asarray(points, dtype=np.float32)
    a = AnnData(X=np.zeros((len(z), 3), dtype=np.float32))
    a.obsm["X_scVI_shared"] = z
    a.obs["cell_type"] = labels
    a.obs["age_group"] = ages
    a.obs["op_basin_index"] = np.linspace(0.2, 1.1, len(z), dtype=np.float32)
    a.obs["op_behaviour_basin"] = ["holder"] * len(z)
    J = np.tile(np.diag([-0.8, -0.25]).astype(np.float32), (len(z), 1, 1))
    J[np.asarray(labels) == "B", 0, 0] = -0.4
    a.uns["scjdo2"] = {"J": J, "J_latent": J,
                         "meta": {"use_rep": "X_scVI_shared", "clock_calibrated": False}}
    return a


def _quadratic_energy(z: np.ndarray) -> np.ndarray:
    return np.sum((np.asarray(z) - np.array([0.6, 0.0])) ** 2, axis=1)


def test_spectrum_ladder_groups_draws_matched_direction_lines(tmp_path):
    fig = plot.spectrum_ladder(_toy_adata(), by="cell_type", groups=("young", "old"),
                               use_rep="X_scVI_shared", out=tmp_path / "ladder")
    assert (tmp_path / "ladder.pdf").exists() and (tmp_path / "ladder.png").exists()
    assert len(fig.axes[0].lines) >= 5  # zero line plus matched directions
    plt.close(fig)


def test_soft_axis_map_adds_old_centroid_displacements(tmp_path):
    fig = plot.soft_axis_map(_toy_adata(), by="cell_type", use_rep="X_scVI_shared",
                             groups=("young", "old"), displacement_arrow=True,
                             out=tmp_path / "soft")
    assert (tmp_path / "soft.pdf").exists() and (tmp_path / "soft.png").exists()
    labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    assert "u_young" in labels and "old − young centroid" in labels
    plt.close(fig)


def test_aging_vectors_uses_log_axes_and_writes_both_formats(tmp_path):
    results = pd.DataFrame({
        "tissue": ["lung", "marrow"], "cell_type": ["A", "B"], "mode": ["mixed", "mixed"],
        "young_median_abs_lambda": [2.0, 0.2], "old_median_abs_lambda": [2.4, 0.25],
        "young_anisotropy": [0.4, 0.6], "old_anisotropy": [0.5, 0.8],
    })
    fig = plot.aging_vectors(results, out=tmp_path / "vectors")
    assert (tmp_path / "vectors.pdf").exists() and (tmp_path / "vectors.png").exists()
    assert all(ax.get_xscale() == "log" and ax.get_yscale() == "log" for ax in fig.axes)
    plt.close(fig)


def test_energy_transect_renders_stored_energy_surface(tmp_path):
    fig = plot.energy_transect(_toy_adata(), "A", "B", _quadratic_energy,
                               use_rep="X_scVI_shared", out=tmp_path / "transect")
    assert (tmp_path / "transect.pdf").exists() and (tmp_path / "transect.png").exists()
    assert "energy transect" in fig.axes[0].get_title()
    assert any("ΔE" in t.get_text() for t in fig.axes[0].texts)
    plt.close(fig)


def test_aging_vectors_spans_a_decade_and_labels_stiffness_ratio(tmp_path):
    results = pd.DataFrame({
        "tissue": ["lung"], "cell_type": ["A"], "mode": ["mixed"],
        "young_median_abs_lambda": [321.7], "old_median_abs_lambda": [325.1],
        "young_anisotropy": [46.7], "old_anisotropy": [46.6],
    })
    fig = plot.aging_vectors(results, out=tmp_path / "vectors_decade")
    ax = fig.axes[0]
    assert ax.get_xlim()[1] / ax.get_xlim()[0] >= 10 - 1e-9
    assert ax.get_ylim()[1] / ax.get_ylim()[0] >= 10 - 1e-9
    assert any("1.01×" in text.get_text() for text in ax.texts)
    plt.close(fig)


def test_energy_transect_marks_only_a_strict_interior_saddle(tmp_path):
    a = _toy_adata()
    # Make the stored leading direction the x-axis, between the two A floors.
    J = np.tile(np.diag([-0.25, -0.8]).astype(np.float32), (a.n_obs, 1, 1))
    a.uns["scjdo2"]["J"] = J
    a.uns["scjdo2"]["J_latent"] = J

    def double_well(z):
        x = np.asarray(z)[:, 0]
        # The old centroid is near t≈0.15, while this maximum lies near t≈0.5;
        # it proves the saddle search uses the young-to-neighbour endpoints.
        return (x * (x - 3.0)) ** 2

    fig = plot.energy_transect(a, "A", "B", double_well,
                               use_rep="X_scVI_shared", out=tmp_path / "saddle")
    labels = fig.axes[0].get_legend_handles_labels()[1]
    assert "intervening saddle" in labels
    assert any("barrier height" in text.get_text() for text in fig.axes[0].texts)
    plt.close(fig)


def test_energy_transect_omits_triangle_for_monotone_curve(tmp_path):
    a = _toy_adata()

    def monotone(z):
        return np.asarray(z)[:, 0]

    fig = plot.energy_transect(a, "A", "B", monotone,
                               use_rep="X_scVI_shared", out=tmp_path / "monotone")
    labels = fig.axes[0].get_legend_handles_labels()[1]
    assert "intervening saddle" not in labels
    assert any("no barrier along this transect" in text.get_text() for text in fig.axes[0].texts)
    plt.close(fig)
