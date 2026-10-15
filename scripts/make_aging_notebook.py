#!/usr/bin/env python3
"""Build and execute the A04 aging-basins notebook from stored artifacts."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
import pandas as pd
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "aging_basins.ipynb"
FIG_ROOT = ROOT / "benchmarks" / "figures" / "aging"


def md(source: str):
    return nbf.v4.new_markdown_cell(source)


def code(source: str):
    return nbf.v4.new_code_cell(source)


def main() -> int:
    manifest = pd.read_csv(FIG_ROOT / "MANIFEST.csv")
    images = manifest[manifest.path.str.endswith(".png")].copy()
    images = images.drop_duplicates("path").sort_values("path")
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "scJDO2 A04 (.venv)", "language": "python", "name": "scjdo2-a04"},
        "language_info": {"name": "python", "version": "3.13"},
    }
    cells = [
        md("""# Aging basins — stored-results figure atlas (A04)

This executed notebook renders **no new models**. It documents and displays the A04 figure set generated solely from the checksum-verified outputs of A01–A03.

**Citations.** Tabula Muris Senis (Schaum *et al.*, 2020); scVI (Lopez *et al.*, 2018); scArches (Lotfollahi *et al.*, 2022)."""),
        code("""from pathlib import Path
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display, Markdown
from scjdo2 import plot

ROOT = Path.cwd()
AGING = ROOT / 'benchmarks' / 'aging'
FIGURES = ROOT / 'benchmarks' / 'figures' / 'aging'
manifest = pd.read_csv(FIGURES / 'MANIFEST.csv')
print(f"Manifest: {len(manifest)} rows; {manifest.path.str.endswith('.png').sum()} PNG figures")"""),
        md("## Preprocessing and donor balancing\nThe A01 processed matrices apply the frozen young/old mapping and balanced sampling before the ≥200-cells-per-age eligibility gate. The tables below are stored bookkeeping, not recomputed filtering."),
        code("""for tissue in ('lung', 'marrow'):
    before = pd.read_csv(AGING / 'qc' / f'{tissue}_donor_type_counts_before.csv')
    after = pd.read_csv(AGING / 'qc' / f'{tissue}_donor_type_counts_after.csv')
    print(f'\\n{tissue}: before={before.n_cells_before.sum():,}; after={after.n_cells_after.sum():,}')
    display(after.head())"""),
        md("## scVI/scArches mapping provenance\nA02 used the frozen 15-dimensional young-reference mapping; the old data were mapped with `SCVI.load_query_data`. Only the stored final validation ELBOs are available—per-epoch curves were not retained."),
        code("""for tissue in ('lung', 'marrow'):
    metrics = pd.read_csv(AGING / 'qc' / f'{tissue}_scvi_metrics.csv')
    print(tissue); display(metrics)"""),
        md("## Symmetry and stored-drift checks\nThe Jacobians are the stored A02 symmetric-operator outputs. The summary below reports the already-recorded Frobenius symmetry residuals and final energy-score losses."),
        code("""for tissue in ('lung', 'marrow'):
    print(tissue); display(pd.read_csv(AGING / 'qc' / f'{tissue}_drift_metrics.csv'))"""),
        md("## Per-type shared-space comparison\nThese are the locked A03 tables. `lambda_ratio_soft_axis` is interpreted only for types with young holder fraction ≥0.5; others are shown as an em dash in the summary figure."),
        code("""tables = []
for tissue in ('lung', 'marrow'):
    tables.append(pd.read_csv(AGING / f'{tissue}_compare_shared.csv'))
compare = pd.concat(tables, ignore_index=True)
display(compare[['tissue','cell_type','n_young','n_old','iso_pass','ant_pass','basin_fraction_young','basin_fraction_old','correction_artifact']])
print('non-artifact types:', int((~compare.correction_artifact).sum()))"""),
        md("## Live plotting API audit\nThe following calls operate on stored shared-space arrays only and show the same plotted data used in the saved figure inventory."),
        code("""lung = ad.read_h5ad(AGING / 'lung_shared.h5ad')
# Display aliases use stored fields only; no fitting or parameter update occurs.
lung.obsm['X_pca'] = np.asarray(lung.obsm['X_scVI_shared'])
lung.obs['lead_re'] = -lung.obs['op_basin_index'].to_numpy(float)
lung.obs['lead_im'] = 0.0
lung.obs['op_behaviour'] = lung.obs['op_behaviour_basin'].astype(str).to_numpy()
fig = plot.spectrum_ladder(lung, by='cell_type', groups=('young','old'), use_rep='X_scVI_shared')
display(fig); plt.close(fig)
fig = plot.soft_axis_map(lung, by='cell_type', groups=('young','old'), displacement_arrow=True, use_rep='X_scVI_shared')
display(fig); plt.close(fig)"""),
        md("## Figure atlas\nEach entry gives the manifest path, the specification item, and the frozen rule/context it represents. PNGs are linked inline; matching 200-dpi PDFs are recorded alongside them in `MANIFEST.csv`."),
    ]
    for row in images.itertuples(index=False):
        rel_from_notebook = Path('..') / Path(row.path)
        cells.append(md(
            f"### `{row.spec_item}` — {row.tissue}, {row.age}\n\n"
            f"**What it shows.** Stored-result rendering for the A04 item `{row.spec_item}`.  "
            f"**Frozen rule/context.** v0.5 basin lock; `classify_basin_latent(rule='lead_eig')`; shared scArches space where applicable.  "
            f"**Source files.** `{row.source_files}`.\n\n"
            f"![{Path(row.path).stem}]({rel_from_notebook.as_posix()})"
        ))
    cells += [
        md("## Deviations\nThe three no-refit deviations are intentionally documented rather than approximated or simulated."),
        code("""print((FIGURES / 'DEVIATIONS.md').read_text())"""),
        md("## Frozen outcome\nThe outcome header below is copied verbatim from A03 and is not reinterpreted in A04."),
        code("""outcome_header = 'isotropic noise increase; no aging claim beyond stiffness'
print(outcome_header)"""),
    ]
    nb.cells = cells
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    client = NotebookClient(nb, timeout=180, kernel_name="scjdo2-a04", resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    nbf.write(nb, OUT)
    print(f"executed notebook: {OUT} ({len(cells)} cells, {len(images)} inline figures)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
