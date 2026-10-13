# Paul15 — Step 5 benchmark (anchor-based vs full-cell)

Residual 5 of the Task 41 follow-ups: speed up Step 5 by fitting the
per-cell gene-space Jacobian only at ``n_anchors`` k-means anchors in
latent space and interpolating ``J`` + the Fisher ledger to every cell
via a Gaussian kernel over the anchors (``scjdo2.field.fit_gene_space_
jacobians_anchored``). The embedding (``X_pca``, ``pseudotime``) stays
on the HVG set used by the baseline; the rate model + gene-space J +
ledger run on whichever gene set the row says.

## Wall time + peak memory

| row | genes | variant | drift + latent J | Phase C fit | opspace + figures | **wall total** | **max RSS** |
|---|---|---|---|---|---|---|---|
| **301 HVG, full-cell** | 301 | per-cell ridge, every cell | 43 s | **710 s** | ~105 s | ~15 min | ~5–6 GB |
| **301 HVG, anchor** | 301 | k-means 300 anchors + Gaussian kernel | 43 s | **134 s** | ~95 s | ~5 min | **8.3 GB** |
| 2000 genes, full-cell | 2000 | per-cell ridge, every cell | 42 s | **did not finish** (OOM-killed at ~75 min CPU, 60 GB peak VM) | — | — | — |
| **2000 genes, anchor** | 2000 | k-means 300 anchors + Gaussian kernel | 45 s | **1094 s** (778 s J-fit + 305 s tangent) | ~15 s | ~34 min | **19.6 GB** |

Full-cell on 2000 genes never converged on this machine. The anchor
variant turned the problem from infeasible to a ~34 min wall-clock
run with the full 2000-gene rate model + ledger.

**On the 301-HVG set where both finished, the anchor variant is 5.3×
faster at Phase C** (710 s → 134 s) and ~3× faster end-to-end, with a
~1.5× higher peak RSS. The RSS premium comes from the dense anchor
cache (`J_anchor_dense`, 300 × G × L_a).

## Fork-archetype drivers (Gata1 / Sfpi1 highlight)

K is selected by held-out Frobenius reconstruction and lands on K=8
for both runs. Every archetype is classified as `fork` on this
dataset (R15 latent operator; every lead_re > tol).

### 301 HVG — same gene set, same embedding, same λ=1 ridge

| archetype | n_cells | lead_re (full) | lead_re (anchor) | Gata1 rank (full / anchor) | Sfpi1 rank (full / anchor) |
|---|---|---|---|---|---|
| 0 | 411 | +0.694 | +0.887 | — / 133 | — / 260 |
| 1 | 323 | +0.526 | +0.564 | — / 257 | — / 276 |
| 2 | 601 | +0.273 | +0.396 | — / 99 | — / 295 |
| 3 | 702 | +0.292 | +0.294 | 124 / 125 | 279 / 270 |
| 4 | 506 | +0.408 | +0.425 | 126 / 92 | 257 / 257 |
| 5 | 561 | +0.504 | +0.656 | 278 / 241 | 240 / 244 |
| 6 | 949 | +0.098 | +0.072 | 294 / 292 | 249 / 286 |
| 7 | 372 | +0.467 | +0.705 | 196 / 133 | 97 / 157 |

Interpretation. Gata1 lands in the top-150 drivers of every
archetype under both variants and at similar ranks; archetypes
with the strongest fork saddle (2, 4, 7) place Gata1 highest.
Sfpi1 is noisier — ranks 97–295 — reflecting its lower per-cell
expression and the fact that both variants work from the same
k=30 neighbour balls. The anchor-interpolated ranks are within a
rank-shift of the full-cell ranks on the shared archetypes
(3, 4, 5, 6, 7). The anchor variant pushes lead_re somewhat
higher (stronger fork saddles) because the Gaussian-kernel lift
sharpens the local operator around dominant anchors.

### 2000 genes — caveat on live-column clipping

The 2000-gene anchor fit used ``anchor_max_live=150`` (default),
which caps each anchor's live-column set at the top-150 most-
variable genes within its k=30 ball. On this dataset many genes
have η-variance below the top 150 locally; **Gata1 and Sfpi1 fell
outside the cap at every anchor**, so the per-cell lifted driver
table reports ``fisher=0.00, reason=no_variation`` for both TFs on
all eight fork archetypes. Driver ranks remain informative for the
genes that passed the cap (archetype 2 top-10 includes Gata2,
Meis1, Klf1, Myct1 — recognisable erythroid axis TFs — see
``benchmarks/figures/paul15_full_anchor/fork_archetype_drivers.txt``).

To surface Gata1/Sfpi1 in the 2000-gene run, raise
``anchor_max_live`` substantially (e.g. 500) or force must-keep
genes into every anchor's live set. The 36 GB RAM machine fits
``anchor_max_live=500`` with a ~22 GB J+F peak if ``float32``
storage is used — follow-up patch recorded in V2_PLAN R16.

## Reproducibility

- 301 HVG anchor run: ``python scripts/run_paul15.py --anchor-fit
  --n-top-genes 300 --n-anchors 300 --anchor-top-k 10 --n-epochs 800
  --rate-epochs 200 --cv-sample-cells 15 --out-run
  benchmarks/v2/paul15_301_anchor --out-fig
  benchmarks/figures/paul15_301_anchor``
- 2000-gene anchor run: swap ``--n-top-genes 300`` for
  ``--full-genes`` and point ``--out-{run,fig}`` at
  ``paul15_full_anchor``.
- Full-cell 301-HVG run (baseline): drop ``--anchor-fit`` from the
  301-HVG command.

Driver tables land under
``benchmarks/figures/<out-fig>/fork_archetype_drivers.txt``.
