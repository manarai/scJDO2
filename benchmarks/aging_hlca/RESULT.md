# HLCA aging basins — H03 executed record

This record documents the H03 data, preprocessing, and shared-space/drift
execution only.  **No §5 statistic, outcome header, or sensitivity analysis was
computed in H03.**

## §1 Data

- **Source:** local H01-verified HLCA core raw counts and metadata
  (`hlca_core_full_metadata.h5ad:adata.raw.X`; 2025-11-08 Census/CELLxGENE
  version), with the local Zenodo 7599104 published HLCA embedding.
- **Cohort:** 106 donors with numeric age, 581,517 cells.  The pre-registered
  young reference contains 266,145 cells from 48 donors aged <45; the query
  contains 315,372 cells from 58 donors aged ≥45.
- **Drift input:** the frozen deterministic sample has 56,904 unique cells,
  drawn at ≤60 cells per donor × `ann_level_3` type using
  `crc32(donor|type)`.  Rare cells remained in preprocessing and scVI but were
  excluded from the eligibility/analysis sample as specified.
- All H01 source checksums were verified before H03 preprocessing.  The
  published secondary embedding was matched one-to-one to all 56,904 sampled
  cell IDs and has 30 dimensions.

## §2 Preprocessing

- Raw counts were retained in `layers["counts"]`; donors without a numeric age
  were excluded before any H03 model input was constructed.
- The frozen `seurat_v3` selection used all 581,517 retained cells,
  `batch_key="study_or_dataset"`, `span=1.0`, and selected 2,000 HVGs.
- The accepted full-data equivalence check independently ran Scanpy's
  in-memory selection and matched all **2,000 / 2,000** genes exactly.  Its
  peak RSS was 14.195 GB (decimal), below the ~28 GB stop threshold.
- The Revision 2 plumbing dry run used deterministic 2,000-cell reference and
  query samples for two CPU epochs each.  Both authoritative `model.history`
  files contain finite `elbo_train` and `elbo_validation` values for epochs 1
  and 2; the frozen-encoder audit was exactly 0 and the non-authoritative audit
  hooks raised no exceptions.

## §3 Shared spaces and drift

### Primary young-reference scVI space

- The primary model used the frozen young-only reference/scArches query design:
  `n_latent=15`, `n_layers=2`, negative-binomial likelihood,
  `batch_key="study_or_dataset"`, CPU, seed 0, and `max_epochs=400` with
  patience 30.  The query used `SCVI.load_query_data(..., accelerator="cpu",
  device="auto")` and `weight_decay=0`.
- The reference completed 400 epochs (best validation ELBO 550.0310); the
  query stopped after 88 epochs (best validation ELBO 655.2349).  The frozen
  latent encoder and BatchNorm-buffer audit covered 277,022 parameters and
  514 buffers and had maximum absolute change **0.0**.  The full-run audit
  trail recorded **0** hook exceptions.
- Exact-score-matching energy drift was fit on all 56,904 sampled primary
  15-D latents with `n_epochs=1500`, seed 0, β=1.0, learning rate 5e-4, and
  exact ESM Laplacian.  The `lead_eig` basin tolerance was 0.1471630 and
  `sym_max=3.8248e-07`, passing the ≤1e-4 gate.  Calls were 15,379 holder,
  34,323 fork, and 7,202 drift cells.

### Secondary published HLCA embedding

- The same 56,904 cell IDs were analyzed in the 30-D published HLCA embedding
  (Zenodo 7599104).  This secondary space is recorded beside the primary space
  and never decides the frozen outcome.
- Its exact-score-matching drift fit used the same locked settings.  The
  `lead_eig` tolerance was 0.1663502 and `sym_max=6.7317e-08`, passing the
  ≤1e-4 gate.  Calls were 22,963 holder, 29,559 fork, and 4,382 drift cells.

### Eligibility for H04

The equal-count criterion (`n_common=50`; ≥20 contributing donors including
≥5 aged <45 and ≥5 aged ≥60) retains 13 primary types, 12 parenchyma-only
types, 6 never-smoker types, and 2 Banovich_Kropski_2020-only types.  The
complete counts and type lists are in `eligibility.csv`; H04 must reapply those
fixed eligibility sets before its donor-level tests.
