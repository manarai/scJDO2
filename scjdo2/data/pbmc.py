"""10x PBMC positive-control loader for the scJDO2 basin arm.

Default source: ``scanpy.datasets.pbmc3k`` (the preprocessed-but-raw-counts
3k PBMC dataset from 10x v1), merged with the louvain cell-type labels
from ``scanpy.datasets.pbmc3k_processed``. 2638 cells × 32738 genes
with raw UMIs; eight louvain types (CD4 T, CD8 T, B, CD14+ Monocytes,
FCGR3A+ Monocytes, NK, Dendritic cells, Megakaryocytes). Chosen as the
fallback the user suggested; the 10x PBMC 10k v3 target was not
required because pbmc3k already contains the resting mature types we
need (CD4 T, CD8 T, B, classical mono) as the dominant populations.

The dataset is a homeostatic bulk-blood snapshot — resting mature
cells sit at or near fixed points of their respective lineages, which
is exactly what the §5 PBMC basin arm needs to test for holder
recovery (contrast with the four progenitor-compartment haemat
datasets in §1-§4, where zero holders is biology, not a positive
test).

Cycle gene list is the Tirosh 2016 S + G2M set (97 genes), as used
by Seurat's CellCycleScoring and scanpy's ``score_genes_cell_cycle``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import scanpy as sc
from anndata import AnnData


RESTING_TYPES = ("CD4 T cells", "CD8 T cells", "B cells", "CD14+ Monocytes")
"""Mature resting cell types expected to be holders if the pipeline
recovers basins. Non-classical monos (FCGR3A+) and NK are more
proliferative/activated and may or may not land as holders."""

NK_PROLIF_TYPES = ("NK cells", "FCGR3A+ Monocytes", "Dendritic cells",
                   "Megakaryocytes")
"""Minority / more active types where the holder expectation is
weaker."""

TIROSH_S_PHASE = (
    "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG",
    "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", "MLF1IP",
    "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76",
    "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51",
    "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM",
    "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1",
    "E2F8",
)

TIROSH_G2M_PHASE = (
    "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A",
    "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF",
    "TACC3", "FAM64A", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB",
    "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1", "KIF20B", "HJURP",
    "CDCA3", "HN1", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1",
    "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR",
    "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF",
    "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA",
)

CYCLE_GENES = tuple(TIROSH_S_PHASE) + tuple(TIROSH_G2M_PHASE)


def load_pbmc3k() -> AnnData:
    """Load pbmc3k (raw UMIs) and merge the louvain labels from
    pbmc3k_processed by barcode. Returns a (2638, 32738) AnnData whose
    ``.X`` holds raw UMI counts and ``.obs['cluster']`` holds the
    8-type louvain annotation."""
    raw = sc.datasets.pbmc3k()
    proc = sc.datasets.pbmc3k_processed()
    annotated = raw.obs_names.isin(proc.obs_names)
    a = raw[annotated].copy()
    label = proc.obs["louvain"].astype(str)
    a.obs["cluster"] = label.reindex(a.obs_names).to_numpy()
    a.obs["is_resting_type"] = a.obs["cluster"].isin(RESTING_TYPES).to_numpy()
    # Carry over the processed tSNE / UMAP for figures.
    for key in ("X_umap", "X_tsne", "X_pca"):
        if key in proc.obsm:
            sub = proc[proc.obs_names.isin(a.obs_names)].obsm[key]
            # Row order of processed matches the louvain index on a.
            idx = {bc: i for i, bc in enumerate(proc.obs_names)}
            order = np.array([idx[bc] for bc in a.obs_names], dtype=int)
            a.obsm[key] = np.asarray(proc.obsm[key])[order].astype(np.float32)
    a.var_names_make_unique()
    return a


def present_cycle_genes(adata: AnnData,
                        extra: Sequence[str] = ()) -> list[str]:
    """Return the subset of CYCLE_GENES (+ ``extra``) present in
    ``adata.var_names`` — the "known rotation" basis for the
    projection variant."""
    want = set(CYCLE_GENES) | set(extra)
    return [g for g in want if g in adata.var_names]
