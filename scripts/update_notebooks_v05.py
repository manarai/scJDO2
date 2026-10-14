"""Append v0.5 figure cells to the two existing notebooks + execute.

Non-destructive: existing cells are preserved. New cells are appended
with markdown headers citing the tag + rule.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


REPO = Path(__file__).resolve().parent.parent
NOTEBOOKS = REPO / "notebooks"


def _mkd(text: str) -> dict:
    return nbf.v4.new_markdown_cell(text)


def _code(src: str) -> dict:
    return nbf.v4.new_code_cell(src)


PBMC_CELLS = [
    _mkd(
        "## v0.5 figure 1 — spectrum ladder (per cluster)\n\n"
        "**Produced by:** `scjdo2.plot.spectrum_ladder(adata, by=\"cluster\", oracle=True)` "
        "under tag `v0.5-basin-lock`, locked lead-eig rule.\n\n"
        "Vertical ladders: eigenvalues of the per-cluster centroid Jacobian "
        "(nearest 30 cells to the cluster mean). Hollow markers = eigenvalues "
        "of the Gaussian oracle `−Σ⁻¹`. Soft band `|λ| < tol` shaded. "
        "`n_soft` and learned/oracle negative-direction curvature ratio "
        "annotated per cluster.",
    ),
    _code(
        "from scjdo2.plot import spectrum_ladder\n"
        "spectrum_ladder(a, by='cluster', oracle=True,\n"
        "                out=REPO / 'benchmarks' / 'figures' / 'v0.5_lock' / "
        "'pbmc_spectrum_ladder_notebook')\n"
    ),
    _mkd(
        "## v0.5 figure 2 — soft-axis map\n\n"
        "**Produced by:** `scjdo2.plot.soft_axis_map(adata)` under tag "
        "`v0.5-basin-lock`.\n\n"
        "scVI embedding coloured by `basin_index`. Each cluster centroid "
        "shows an arrow along its most-positive eigenvector of the centroid "
        "Jacobian (length ∝ eigenvalue). Annotation names the nearest other "
        "cluster centroid and the `|cos|` of the soft axis with the "
        "centroid→neighbour direction.",
    ),
    _code(
        "from scjdo2.plot import soft_axis_map\n"
        "soft_axis_map(a, by='cluster', use_rep='X_umap',\n"
        "              out=REPO / 'benchmarks' / 'figures' / 'v0.5_lock' / "
        "'pbmc_soft_axis_map_notebook')\n"
    ),
    _mkd(
        "## v0.5 figure 3 — ledger map\n\n"
        "**Produced by:** `scjdo2.plot.ledger_map(adata)` under tag "
        "`v0.5-basin-lock`.\n\n"
        "Embedding coloured by provenance class per cell: basin (measured), "
        "saddle region, drift (near-flat), or not_determinable. Legend "
        "reports per-class counts.",
    ),
    _code(
        "from scjdo2.plot import ledger_map\n"
        "ledger_map(a, use_rep='X_umap',\n"
        "           out=REPO / 'benchmarks' / 'figures' / 'v0.5_lock' / "
        "'pbmc_ledger_map_notebook')\n"
    ),
    _mkd(
        "## v0.5 figure 6 — basin_index by cell type\n\n"
        "**Produced by:** `scjdo2.plot.basin_index_by_type(adata, "
        "resting_types=RESTING_TYPES)` under tag `v0.5-basin-lock`.\n\n"
        "Per-type strip plot of `basin_index`, sorted by median; "
        "per-type basin fraction bar below; dashed line = dataset "
        "resting basin fraction (**47.2 %** under the locked rule, "
        "not 92.6 % of the retired r-1 rule).",
    ),
    _code(
        "from scjdo2.plot import basin_index_by_type\n"
        "from scjdo2.data import RESTING_TYPES\n"
        "basin_index_by_type(a, by='cluster', resting_types=RESTING_TYPES,\n"
        "                    out=REPO / 'benchmarks' / 'figures' / "
        "'v0.5_lock' / 'pbmc_basin_index_by_type_notebook')\n"
    ),
]


K562_CELLS = [
    _mkd(
        "## v0.5 figure 5 — K562 paired + generic decomposition\n\n"
        "**Produced by:** `scjdo2.plot.perturb_paired({per_kd, generic_table})` "
        "under tag `v0.5-basin-lock`.\n\n"
        "Left: operator cosine vs DE-only cosine as paired points per KD; "
        "diagonal + win fraction annotation. "
        "Right: generic-response decomposition — mean cosine for full vs "
        "residual (after subtracting the mean δx_obs direction) for "
        "operator and DE-only. Honest effect sizes: full operator **+0.049**, "
        "residual **+0.014**.",
    ),
    _code(
        "import pandas as pd\n"
        "from pathlib import Path\n"
        "from scjdo2.plot import perturb_paired\n"
        "OUT = REPO / 'benchmarks' / 'perturbseq_k562'\n"
        "pk = pd.read_csv(OUT / 'per_knockdown.csv')\n"
        "gen = pd.read_csv(OUT / 'generic_control_comparison.csv') if (OUT / 'generic_control_comparison.csv').exists() else None\n"
        "perturb_paired({'per_kd': pk, 'generic_table': gen},\n"
        "               out=REPO / 'benchmarks' / 'figures' / 'v0.5_lock' / "
        "'k562_perturb_paired_notebook')\n"
    ),
]


def _append(path: Path, new_cells: list) -> None:
    nb = nbf.read(path, as_version=4)
    nb["cells"].extend(new_cells)
    nbf.write(nb, path)


def main() -> int:
    _append(NOTEBOOKS / "basin_analysis.ipynb", PBMC_CELLS)
    _append(NOTEBOOKS / "perturbseq_k562.ipynb", K562_CELLS)
    print("appended v0.5 cells; now re-execute via nbconvert")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
