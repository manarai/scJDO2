#!/usr/bin/env python3
"""Build and execute the H05 HLCA results notebook in place."""
from __future__ import annotations

import os
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "aging_hlca.ipynb"
FIGURE_DIR = Path("benchmarks/figures/aging_hlca")


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def figure_cell(name: str, explanation: str):
    return [
        markdown(explanation),
        code(f"display(Image(filename=str(FIGURE_DIR / '{name}.png'), width=1000))"),
    ]


def build() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    cells = [
        markdown("# HLCA aging basins — H05 figures and mouse–human comparison\n\nThis executed notebook renders the committed H05 figure files. It does not refit a model or rerun the H04 tests."),
        code("""from pathlib import Path
import anndata as ad
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd()
HLCA_DATA = ROOT / 'benchmarks' / 'aging_hlca' / 'data'
FIGURE_DIR = ROOT / 'benchmarks' / 'figures' / 'aging_hlca'
primary = pd.read_csv(ROOT / 'benchmarks' / 'aging_hlca' / 'tests_primary.csv')
sensitivity = pd.read_csv(ROOT / 'benchmarks' / 'aging_hlca' / 'tests_sensitivity.csv')
mouse_human = pd.read_csv(ROOT / 'benchmarks' / 'aging_hlca' / 'mouse_human.csv')
manifest = pd.read_csv(FIGURE_DIR / 'MANIFEST.csv')
print({'primary_types': len(primary), 'sensitivity_rows': len(sensitivity), 'matched_pairs': len(mouse_human), 'figure_files': len(manifest)})"""),
        markdown("## Frozen primary outcome\n\nThe H04 outcome is retained verbatim. H05 is a rendering and descriptive-comparison task; sensitivity analyses do not alter the primary header."),
        code("print('isotropic noise increase; no aging claim beyond stiffness — no majority stiffness change')\ndisplay(primary[['ann_level_3', 'iso_pass', 'ant_pass', 'stiff_class']])"),
    ]
    cells += figure_cell("tests_overview", "## Primary test overview\n\nShows the 13 frozen-eligible types. Isotropy and anticipation compare their stored statistic with the H04 stored 95% permutation threshold; stiffness shows the stored slope and its call. The stiffness null q95 was not persisted by H04, so no band is fabricated.")
    cells += figure_cell("sensitivity_heatmap", "## Sensitivity calls\n\nShows the stored primary and §7(a–f) calls as isotropy / anticipation / stiffness. Grey cells are types that became ineligible in the restricted subset.")
    cells += figure_cell("stiffness_vs_age", "## Donor stiffness against age\n\nShows stored primary donor-level `log median |λ|` values against age, coloured by `study_or_dataset`, with a display-only OLS line. T cell lineage and Monocytes carry the mandated H04-review lead flag.")
    cells += [
        markdown("## EC-capillary anticipation display\n\nShows primary stored donor centroids projected onto the young-reference soft axis against age. The source H5AD is accessed only via `HLCA_DATA`, as specified."),
        code("""drift = ad.read_h5ad(HLCA_DATA / 'h03_primary_drift.h5ad', backed='r')
try:
    print({'stored_drift_cells': drift.n_obs, 'representation': list(drift.obsm.keys())})
finally:
    drift.file.close()
display(Image(filename=str(FIGURE_DIR / 'ec_capillary_anticipation.png'), width=1000))"""),
    ]
    cells += figure_cell("mouse_human", "## Pre-registered mouse–human comparison\n\nShows only the two §8 matched pairs. Mouse points are the stored A03 old/young stiffness ratio with the X04 shared-seed range; human points are `exp(stiff_slope)` from H04 §7(f). Calls are text only. This is descriptive; no statistic is computed.")
    cells += figure_cell("qc_scvi_history", "## QC: scVI history\n\nShows the stored young-reference and older-query validation ELBO histories from H03.")
    cells += figure_cell("qc_query_vs_ref_elbo", "## QC: final query versus reference ELBO\n\nShows the two stored best validation ELBO values reported by H03.")
    cells += figure_cell("qc_symmetry", "## QC: energy-drift symmetry\n\nShows the stored primary and secondary maximum Jacobian asymmetry values against the H03 symmetry gate.")
    cells += figure_cell("qc_eligibility_counts", "## QC: eligibility counts\n\nShows the stored H03/H04 count of eligible types for primary, Banovich-only, parenchyma-only, and never-smoker analyses.")
    cells += [
        markdown("## QC: donor composition\n\nShows unique donors by age bin and study. The figure is derived from stored H03 drift metadata through the `HLCA_DATA` path variable."),
        code("""drift = ad.read_h5ad(HLCA_DATA / 'h03_primary_drift.h5ad', backed='r')
try:
    donor_meta = drift.obs[['donor_id', 'age', 'study_or_dataset']].drop_duplicates('donor_id')
finally:
    drift.file.close()
print({'unique_donors_in_drift_sample': len(donor_meta), 'donors_age_ge_60': int((donor_meta['age'] >= 60).sum())})
display(Image(filename=str(FIGURE_DIR / 'qc_donor_composition.png'), width=1000))"""),
    ]
    cells += [
        markdown("## Mouse–human values\n\nThe committed two-row descriptive comparison table used by the figure."),
        code("display(mouse_human)"),
        markdown("## Figure manifest\n\nEvery PDF and PNG delivered by H05, with the committed source files used for that rendering."),
        code("display(manifest)"),
    ]
    nb.cells = cells
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK)
    os.environ["JUPYTER_PATH"] = str(ROOT / ".venv" / "share" / "jupyter")
    client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    nb.metadata["execution"] = {"status": "executed", "working_directory": str(ROOT), "data_path_variable": "HLCA_DATA"}
    nbf.write(nb, NOTEBOOK)


def main() -> int:
    build()
    print(NOTEBOOK)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
