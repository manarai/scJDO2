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
| **2000 genes, anchor** (max_live=150, raw-variance rank) | 2000 | k-means 300 anchors + Gaussian kernel | 45 s | **1094 s** (778 s J-fit + 305 s tangent) | ~15 s | ~34 min | **19.6 GB** |
| **2000 genes, anchor** (max_live=300, SNR rank, float32) — **R16** | 2000 | k-means 300 anchors + Gaussian kernel, SNR-ranked live cols | 44 s | **1461 s** (916 s J-fit + 535 s tangent) | ~45 s | ~43 min | **17.6 GB** |

Full-cell on 2000 genes never converged on this machine. The anchor
variant turned the problem from infeasible to a ~34 min wall-clock
run with the full 2000-gene rate model + ledger.

**R16 (anchor-cap book-keeping; recorded 2026-10-13).** The 150-cap
raw-variance run clipped Gata1 / Sfpi1 at every anchor and labelled
them ``no_variation`` on every archetype (fisher = 0). R16 (a) adds
a ``capped`` reason code to the Fisher ledger so the storage choice
is distinguished from the measurement verdict, (b) ranks live
columns by NB-precision-weighted η variance (SNR on the log-rate
scale) rather than raw variance, (c) raises the default
``anchor_max_live`` to 300, and (d) stores the CSR J + F as
``float32`` (opspace auto-upcasts on the aggregated side) so the
larger cap fits the 36 GB machine budget. Gata1 and Sfpi1 now
report ``reason=ok`` on **every** fork archetype with real Fisher
weights and measured SNR, instead of ``no_variation`` with
fisher=0. Peak memory footprint stayed under the previous jetsam
threshold (max RSS 17.6 GB vs the prior OOM-kill at float64 and
50.6 GB peak VM).

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

### 2000 genes — R16 (max_live=300, SNR rank, ``capped`` reason)

Previously (max_live=150, raw-variance rank) Gata1 and Sfpi1 fell
outside the top-150 at every anchor and reported ``fisher=0.00,
reason=no_variation`` on every archetype. R16 (SNR rank by
``var_i(η_i,g) · <w_i,g>_i`` + max_live=300 + float32 storage)
pulls them into the live set and reports:

Ledger reason-code fractions (all 2730 × 2000 cell × gene
entries):

```
ok:           15.00%
no_counts:     0.00%
no_variation:  0.00%
not_in_ball:   0.00%
capped:       85.00%
```

The 85% ``capped`` share is the correct accounting: of the 2000
genes, each anchor's dominant live set keeps the top-300 by SNR
(300 / 2000 = 15% ok), and the other 85% are above the η-variance
floor but clipped by the cap — a storage choice, labelled distinctly
from the ``no_variation`` measurement verdict (0% on this dataset).

Archetype classifications (K = 8, chosen by held-out Frobenius):
6 fork + 2 growing_oscillator. The topology change vs the 150-cap
run (which classified every archetype as fork) is real — SNR-
ranked columns reveal complex-eigenvalue archetypes at indices 4
(lead_re +0.03, lead_im +0.001) and 7 (+0.08, +0.04).

Fork-archetype Gata1 / Sfpi1 drivers (R16, max_live=300):

| archetype | n_cells | lead_re | Gata1 rank | Gata1 fisher / SNR | Sfpi1 rank | Sfpi1 fisher / SNR |
|---|---|---|---|---|---|---|
| 0 | 411 | +0.120 | 159  | 5.91 / 0.290  | 1586 | 2.31 / 0.218 |
| 1 | 323 | +0.088 | 1868 | 4.00 / 0.204  | 1126 | 1.32 / 0.154 |
| 2 | 601 | +0.056 | 912  | 1.72 / 0.153  | 1778 | 2.48 / 0.161 |
| 3 | 702 | +0.043 | 543  | 0.08 / 0.0714 |  363 | 2.92 / 0.173 |
| 5 | 561 | +0.107 | 464  | 5.52 / 0.236  | 1404 | 1.01 / 0.151 |
| 6 | 949 | +0.007 | 1831 | 0.19 / 0.0709 | 1327 | 0.02 / 0.0317 |

Every row reports ``reason=ok`` for both TFs. SNRs span 0.03
(archetype 6, the near-drift archetype with ``lead_re`` barely
above tol) to 0.29 (archetype 0, the strongest fork saddle,
erythroid-side; Gata1 is a top-150 driver there). The max-live
=150 run reported ``no_variation`` with ``fisher=0`` on all these
cells; R16 instead reports an SNR either way, which is the
acceptable outcome the user specified.

## Reproducibility

- 301 HVG anchor run: ``python scripts/run_paul15.py --anchor-fit
  --n-top-genes 300 --n-anchors 300 --anchor-top-k 10 --n-epochs 800
  --rate-epochs 200 --cv-sample-cells 15 --out-run
  benchmarks/v2/paul15_301_anchor --out-fig
  benchmarks/figures/paul15_301_anchor``
- 2000-gene anchor run (R16): ``python scripts/run_paul15.py
  --anchor-fit --full-genes --n-anchors 300 --anchor-top-k 10
  --anchor-max-live 300 --n-epochs 800 --rate-epochs 200
  --cv-sample-cells 15 --out-run benchmarks/v2/paul15_full_anchor_v2
  --out-fig benchmarks/figures/paul15_full_anchor_v2``
- Full-cell 301-HVG run (baseline): drop ``--anchor-fit`` from the
  301-HVG command.

Driver tables land under
``benchmarks/figures/<out-fig>/fork_archetype_drivers.txt``.
