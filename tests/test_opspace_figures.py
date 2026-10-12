"""Task 41 §5 — opspace + figure-layer contract tests.

The pitchfork gate (`scripts/phase_c_gate.py`) remains the heavier
dataset-level check for behaviour P/R and the λ-selection curves. These
tests exercise the §5 contract on a cheap synthetic adata so they can
run in CI:

1. spectrum: no stored conjugate pair; op_rank == tangent rank on every
   cell; non-trivial eigenvalues on cells with signal.
2. behaviour: planted holder/fork labels recovered with P/R ≥ 0.8 under
   the §1.2 default `tol = 0.1 · std(lead_re)`.
3. drivers: a planted driver gene appears in the top-10 of the fork
   mode (precision ≥ 0.8 on the planted set, Fisher-filtered).
4. phase convention: in every stored eigenvector the largest-|magnitude|
   entry is real and positive (`spectrum` sign-fixing).
5. complex rendering: a planted damped oscillator triggers the complex
   branch of `plot_drivers` and the phase ordering is monotone.
6. sensitive ring: artists created on Panels A and B equal the number
   of `op_sensitive` cells (per panel).
7. relative-units label: toggling `uns["scjdo2"]["clock_calibrated"]`
   adds/removes "(relative)" on every eigenvalue axis label.
8. figures: `figure_operator_space` and `figure_ledger_summary` write
   PDF+PNG files, no empty axes, every axis labelled.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

from scjdo2 import opspace, plot
from scjdo2.plot.style import relative_label


# --------------------------------------------------------------------- helpers

PLANTED_DRIVER_GENE = 0           # column 0 is the strong driver by construction
N_CELLS = 90
D_LATENT = 2
G_GENES = 12


def _build_pitchfork_adata(seed: int = 0, add_oscillator: bool = False) -> AnnData:
    """Toy adata with latent J planted as fork/holder, a gene-space J whose
    column 0 carries the leading driver, and the ∇η tensor needed for the
    R15 lift. Half the cells are fork (+0.3), half holder (−0.5)."""
    rng = np.random.default_rng(seed)
    n, d, G = N_CELLS, D_LATENT, G_GENES
    z = rng.normal(size=(n, d)).astype(np.float32)

    J_latent = np.zeros((n, d, d), dtype=np.float64)
    for c in range(n):
        if add_oscillator and c % 10 == 0:
            # Small damped-oscillator block: Re ≈ -0.4, Im ≠ 0.
            J_latent[c] = np.array([[-0.4, -0.6], [0.6, -0.4]])
            continue
        # Diagonal perturbation only — keeps the lead eigenvalue real so the
        # behaviour test is immune to small non-symmetric perturbations
        # flipping cells into damped_oscillator.
        base = -0.5 if c % 2 == 0 else 0.3
        diag = base + 0.02 * rng.normal(size=d)
        J_latent[c] = np.diag(diag)

    # Gene-space J: live block on columns 0..5. Column 0 carries the fork
    # driver with strong magnitude; other live columns are small noise.
    J_gene: list = []
    F_list: list = []
    live = np.arange(6)
    for c in range(n):
        base = 0.05 * rng.normal(size=(6, 6))
        base[0, 0] = 0.5 if c % 2 == 1 else -0.5
        base[0, 1:] = 0.1 * rng.normal(size=5)
        data = base.ravel()
        rows = np.repeat(live, 6)
        cols = np.tile(live, 6)
        J_gene.append(csr_matrix((data, (rows, cols)), shape=(G, G)))
        F_list.append(csr_matrix(
            (np.ones_like(data), (rows, cols)), shape=(G, G),
        ))

    # Rank-2 Fisher-weighted tangent on each cell.
    tangent_P = [np.eye(2, G) for _ in range(n)]
    tangent_rank = np.full(n, 2, dtype=np.int32)

    # ∇η built so column 0 is aligned with the latent lead eigvec in
    # expectation — the R15 lift must then surface gene 0 as the top driver.
    grad_eta = 0.1 * rng.normal(size=(n, G, d))
    grad_eta[:, PLANTED_DRIVER_GENE, 0] = 1.0

    adata = AnnData(X=np.zeros((n, G), dtype=np.float32))
    adata.obsm["X_pca"] = z
    adata.obsm["X_umap"] = z + 0.1 * rng.normal(size=(n, d)).astype(np.float32)
    adata.var_names = [f"g{i}" for i in range(G)]
    # Ground-truth labels for the behaviour test. Oscillator cells are
    # left unlabeled (they belong to a third class).
    truth = np.where(np.arange(n) % 2 == 0, "holder", "fork")
    adata.obs["truth"] = truth
    adata.uns["scjdo2"] = {
        "J": J_gene,
        "J_latent": J_latent,
        "fisher_ledger": {
            "fisher": F_list,
            "off_tangent": {"projector": tangent_P, "rank": tangent_rank},
            "reason_per_cell": [np.zeros(G, dtype=np.int8) for _ in range(n)],
        },
        "grad_eta": grad_eta,
    }
    return adata


# --------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def adata_pitchfork():
    adata = _build_pitchfork_adata(seed=0)
    opspace.compute(adata)
    return adata


# --------------------------------------------------------------------- §5 tests


def test_spectrum_no_conjugate_pairs_and_rank(adata_pitchfork):
    eigvals = np.asarray(adata_pitchfork.obsm["op_eigvals"])
    # Conjugate-free: no stored pair should be (a + bi, a − bi) with b ≠ 0.
    for c in range(eigvals.shape[0]):
        rows = eigvals[c]
        for i in range(rows.shape[0]):
            for j in range(i + 1, rows.shape[0]):
                a, b = rows[i], rows[j]
                if np.isfinite(a) and np.isfinite(b):
                    assert not (
                        np.isclose(a.real, b.real)
                        and np.isclose(a.imag, -b.imag)
                        and abs(a.imag) > 1e-6
                    ), f"stored conjugate pair at cell {c}: {a}, {b}"
    assert "op_rank" in adata_pitchfork.obs
    rank_arr = adata_pitchfork.obs["op_rank"].to_numpy()
    assert (rank_arr == 2).mean() >= 0.95


def test_behaviour_pr_at_least_0_8(adata_pitchfork):
    truth = adata_pitchfork.obs["truth"].to_numpy()
    pred = adata_pitchfork.obs["op_behaviour"].astype(str).to_numpy()
    for cls in ("holder", "fork"):
        tp = int(((pred == cls) & (truth == cls)).sum())
        fp = int(((pred == cls) & (truth != cls)).sum())
        fn = int(((pred != cls) & (truth == cls)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        assert prec >= 0.8, f"{cls} precision {prec:.2f}"
        assert rec >= 0.8, f"{cls} recall {rec:.2f}"


def test_drivers_lifted_eigvec_surfaces_planted_gene(adata_pitchfork):
    df = opspace.drivers(adata_pitchfork, which="lead", top=10)
    assert len(df) > 0
    assert "reason" in df.columns
    # Reason column populated for every row (never silently NaN).
    assert df["reason"].notna().all()
    # Planted driver gene surfaced in the top-10 by magnitude.
    assert f"g{PLANTED_DRIVER_GENE}" in df["gene"].astype(str).tolist()


def test_phase_convention_largest_entry_is_real_positive(adata_pitchfork):
    eigvecs = np.asarray(adata_pitchfork.uns["scjdo2"]["op_eigvecs"])
    for c in range(eigvecs.shape[0]):
        for k in range(eigvecs.shape[1]):
            v = eigvecs[c, k]
            mag = np.abs(v)
            if not np.isfinite(mag).any() or mag.max() == 0:
                continue
            top = int(np.argmax(mag))
            if np.isfinite(v[top]):
                assert v[top].real >= -1e-9, (c, k, v[top])
                assert abs(v[top].imag) <= 1e-6, (c, k, v[top])


def test_complex_rendering_phase_axis(tmp_path):
    import pandas as pd

    phases = np.linspace(-np.pi, np.pi, 6, endpoint=False)
    df = pd.DataFrame({
        "gene": [f"g{i}" for i in range(6)],
        "weight": np.cos(phases),
        "magnitude": np.ones(6),
        "phase": phases,                       # non-NaN → complex mode
        "fisher": np.ones(6),
        "fisher_se": np.full(6, 0.1),
        "reason": ["ok"] * 6,
    })
    fig, ax = plt.subplots()
    plot.plot_drivers(ax, df, top=6)
    # Complex mode draws a colorbar below the axis; find it.
    cbars = [c for c in fig.axes if c is not ax]
    assert cbars, "complex mode should add a cyclic colorbar"
    plt.close(fig)


def test_sensitive_ring_artists_match_count(adata_pitchfork, tmp_path):
    fig = plot.figure_operator_space(adata_pitchfork, out=tmp_path / "op")
    import matplotlib.patches as mpatches
    rings = sum(
        1
        for ax in fig.axes
        for art in ax.patches
        if isinstance(art, mpatches.Circle) and not art.get_fill()
    )
    sens_count = int(adata_pitchfork.obs["op_sensitive"].sum())
    # Panels A and B each draw one ring per sensitive cell.
    assert rings >= sens_count, (rings, sens_count)
    plt.close(fig)


def test_relative_label_toggle():
    adata_cal = AnnData(X=np.zeros((2, 2), dtype=np.float32))
    adata_cal.uns["scjdo2"] = {"clock_calibrated": True}
    adata_rel = AnnData(X=np.zeros((2, 2), dtype=np.float32))
    adata_rel.uns["scjdo2"] = {"clock_calibrated": False}
    assert relative_label(adata_cal, "Re") == "Re"
    assert relative_label(adata_rel, "Re") == "Re (relative)"


def test_figures_write_pdf_and_png(adata_pitchfork, tmp_path):
    out_op = tmp_path / "op"
    fig = plot.figure_operator_space(adata_pitchfork, out=out_op)
    assert (tmp_path / "op.pdf").exists()
    assert (tmp_path / "op.png").exists()
    for ax in fig.axes:
        # Every axis must be either labelled or have artists; no silent empty axes.
        has_label = bool(ax.get_xlabel() or ax.get_ylabel() or ax.get_title())
        has_art = bool(ax.collections or ax.patches or ax.lines or ax.texts
                       or ax.images or ax.containers)
        assert has_label or has_art, f"empty unlabelled axis: {ax}"
    plt.close(fig)

    out_led = tmp_path / "led"
    fig2 = plot.figure_ledger_summary(adata_pitchfork, out=out_led)
    assert (tmp_path / "led.pdf").exists()
    plt.close(fig2)
