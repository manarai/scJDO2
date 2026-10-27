# orient/S12 — result (executor: local Claude agent)

**Status: stopped at Step 0 by the task's fixed decision rule. Steps 1 and 2 were not run.**

**v3 caveats (S09):** unspliced share 0.40–0.52 (above the 0.25–0.35 band, inside real 0.08–0.50); splicing oracle 0.21–0.28; steady genes ≈ 1 %; scVelo recovers more genes than on real data; multi-time clones 0.44–0.49; genes detected ≈ 830 / cell. No number below is splicing-based. No number below is lineage-based. Step 0 uses only the time course (day, time_hours) and spliced counts. Truth (z_true, fate_true) is used for scoring only.

Tables: `step0.csv` (one row per dataset × embedding), `step0_fa_diagnosis.csv` (items a–b), `step0_decision.json`, `cpu_ledger.csv`. Figure: `benchmarks/figures/orient/s12/step0_auc_by_embedding.png`.

## Gates (verbatim from the task) and results

> **G0 (Step 0 decision)** as defined above.
>
> 4. **Decision (fixed):** proceed to Steps 1–2 if the OT-flow-J fork AUC is ≥ 0.85 on every seed in **at least two** embeddings other than PCA-20, or if (a)–(d) identify the FA inversion as a defect of the FA coordinates with a fix that brings FA to ≥ 0.85 (state it). Otherwise **stop and report** — no optimisation.

**Result: G0 = STOP.**
* Clause 1 is not met. Of FA-20, PCA-10, PCA-50 and scVI-10, only PCA-10 has AUC ≥ 0.85 on all 6 datasets (minimum 0.875). FA-20 has minimum 0.043, PCA-50 0.559 and scVI-10 0.310.
* Clause 2 is not met. The fix tested in (d), an FA OT cost whitened by the FA noise variances, gives AUC 0.207 / 0.970 / 0.897 / 0.625 / 0.787 / 0.801 (minimum 0.207). The tangent-restricted λ₁ and the soft-axis curvature in (c) stay at or below 0.61 on raw FA.

> **G1** selected OT flow J fork AUC ≥ 0.90 on every test dataset in {v3_default s2, s3, v3_transition s2, s3} (PCA-20), and ≥ 0.85 in the second embedding.

**Not evaluated** (Step 2 not run: G0 = stop).

> **G2** selected `J_ref` error lower than the S10 default's on ≥ 3 of the 4.

**Not evaluated** (Step 2 not run).

> **G3 transition** on `v3_transition` s2, s3 (both must be informative): |detected − true bin| ≤ 1 with the selected OT flow J; and on `v3_nofork` s0–2 the λ₁ peak-to-median ratio below that at the detected transition.

**Not evaluated** (Step 2 not run).

> **G4 real** LARRY held-out prediction: the selected flow's energy distance below both baselines in PCA-30.

**Not evaluated** (Step 2 not run).

## What ran

All units ran as detached `/usr/bin/time -l` processes, one log each in `s12/logs/` and one checkpoint each in `s12/checkpoints/`. Limits: ≤ 12 threads in total, `LOKY_MAX_CPU_COUNT=2`, no scVelo, no Palantir.

1. `python -m pytest -q tests/test_orient.py tests/test_smoke.py tests/test_synthetic_multimodal.py`: 77 passed (before; no code under test changed afterwards).
2. `python scripts/s12_step0.py ot --preset P --seed N --embedding E` for P ∈ {v3_default, v3_transition}, N ∈ {0, 1, 2}, E ∈ {pca20, fa20, fa20w}: 18 units.
3. `python scripts/s12_step0.py embed --preset P --seed N`: PCA-10 and PCA-50 from `embed_linear` (same preprocessing), and scVI-10. 6 units.
4. `python scripts/s12_step0.py ot ... --embedding E` for E ∈ {pca10, pca50, scvi10}: 18 units. Then `diag --preset P --seed N`: 6 units.
5. `python scripts/s12_step0.py summary` (step0.csv, decision, figure) and `ledger` (cpu_ledger.csv).

OT flow J is the S10 default in every embedding. It uses POT 0.9.7 `ot.sinkhorn`, balanced, with ε = 0.05 × median squared distance per day pair (day 2→4 and 4→6). The barycentric velocity is divided by Δt (48 h) for the day-2 and day-4 cells. `MeasuredDrift(d, seed=0)` defaults are fitted on those cells (MPS), and `jacobian_autodiff` is evaluated on all cells (CPU).

## Step 0.1 — reproduction of S10 (PCA-20, FA-20)

| dataset | PCA-20 AUC (S12 refit) | S10 stored operator | S10 reported | FA-20 AUC (S12 refit) | S10 stored | S10 reported |
|---|---|---|---|---|---|---|
| v3_default s0 | 0.9738 | 0.9738 | 0.9738 | 0.0472 | 0.0472 | 0.0472 |
| v3_default s1 | 0.9905 | 0.9905 | 0.9905 | 0.2004 | 0.2004 | 0.2004 |
| v3_default s2 | 0.9990 | 0.9990 | 0.9990 | 0.0432 | 0.0432 | 0.0432 |
| v3_transition s0 | 0.8181 | 0.8181 | 0.8181 | 0.4747 | 0.4747 | 0.4747 |
| v3_transition s1 | 0.9103 | 0.9103 | 0.9103 | 0.3016 | 0.3016 | 0.3016 |
| v3_transition s2 | 0.9668 | 0.9668 | 0.9668 | 0.5263 | 0.5263 | 0.5263 |

The OT velocity is identical to S10's on every unit (max |Δv| = 0). The refitted MeasuredDrift operator gives the same AUC to 4 decimals.

## Step 0.3 — fork AUC of OT flow J by embedding (λ₁, saddle vs committed)

| dataset | n saddle / committed | PCA-20 | FA-20 | PCA-10 | PCA-50 | scVI-10 | FA-20 whitened cost (d) |
|---|---|---|---|---|---|---|---|
| v3_default s0 | 22 / 12 594 | 0.974 | 0.047 | 0.955 | 0.637 | 0.828 | 0.207 |
| v3_default s1 | 28 / 12 550 | 0.990 | 0.200 | 0.980 | 0.968 | 0.717 | 0.970 |
| v3_default s2 | 29 / 12 781 | 0.999 | 0.043 | 0.997 | 0.559 | 0.964 | 0.897 |
| v3_transition s0 | 350 / 8 998 | 0.818 | 0.475 | 0.875 | 0.891 | 0.310 | 0.625 |
| v3_transition s1 | 368 / 8 859 | 0.910 | 0.302 | 0.895 | 0.787 | 0.522 | 0.787 |
| v3_transition s2 | 364 / 9 186 | 0.967 | 0.526 | 0.963 | 0.967 | 0.556 | 0.801 |
| every seed ≥ 0.85 | | no (0.818) | no | **yes** | no | no | no |

Median |cos| of the leading eigenvector with the fate axis at saddle cells:

| dataset | PCA-20 | FA-20 | PCA-10 | PCA-50 | scVI-10 | FA-20 whitened |
|---|---|---|---|---|---|---|
| v3_default s0 | 0.981 | 0.106 | 0.976 | 0.899 | 0.920 | 0.640 |
| v3_default s1 | 0.972 | 0.351 | 0.981 | 0.945 | 0.959 | 0.926 |
| v3_default s2 | 0.992 | 0.284 | 0.972 | 0.870 | 0.965 | 0.851 |
| v3_transition s0 | 0.952 | 0.270 | 0.926 | 0.946 | 0.532 | 0.900 |
| v3_transition s1 | 0.962 | 0.550 | 0.960 | 0.925 | 0.215 | 0.949 |
| v3_transition s2 | 0.967 | 0.688 | 0.973 | 0.959 | 0.166 | 0.968 |

scVI-10: `scvi-tools` 1.5.1 was importable in `.venv`. SCVI defaults were used (1 hidden layer of 128, ZINB) with n_latent = 10, observed spliced counts, all 4 000 genes, seed 0, the default 400 epochs on CPU, and the posterior-mean latent. Training took 17–21 min per dataset.

## Step 0.2 — FA inversion diagnosis

**(a) Truth classes and fate axis.** The truth classes are defined on `z_true`, so they are the same cells in every embedding. Fraction of saddle cells: v3_default 0.0017 / 0.0022 / 0.0023; v3_transition 0.028 / 0.029 / 0.028. The fate axes were mapped to gene space by the pseudo-inverse of each linear map. Their |cos| (PCA vs FA) is 0.912 / 0.860 / 0.889 (v3_default) and 0.797 / 0.780 / 0.757 (v3_transition). On each embedding's own fate axis, the median saddle projection is near the midpoint and committed cells lie away from it:

| dataset | PCA: saddle median | PCA: committed median distance | FA: saddle median | FA: committed median distance |
|---|---|---|---|---|
| v3_default s0 / s1 / s2 | −0.26 / 0.66 / 0.09 | 10.7 / 9.5 / 10.1 | 0.02 / −0.18 / −0.01 | 1.01 / 0.94 / 1.01 |
| v3_transition s0 / s1 / s2 | 0.09 / 0.48 / −0.06 | 7.4 / 6.9 / 7.0 | 0.02 / 0.02 / −0.00 | 1.15 / 1.15 / 1.15 |

**(b) OT velocity agreement, gene space.** Each embedding's OT velocity was mapped to gene space by its linear map's pseudo-inverse. The per-cell cosine between PCA-20 and FA-20 was computed over the cells with an OT velocity (1 626–1 647 day-2 and day-4 cells):

| dataset | median cos | fraction cos > 0 | median cos, saddle cells | MeasuredDrift-fitted velocity, all cells / saddle / committed |
|---|---|---|---|---|
| v3_default s0 | 0.524 | 0.999 | 0.508 | 0.546 / 0.505 / 0.546 |
| v3_default s1 | 0.542 | 1.000 | 0.438 | 0.545 / 0.455 / 0.546 |
| v3_default s2 | 0.563 | 1.000 | 0.532 | 0.521 / 0.533 / 0.521 |
| v3_transition s0 | 0.563 | 1.000 | 0.570 | 0.276 / 0.365 / 0.250 |
| v3_transition s1 | 0.538 | 1.000 | 0.559 | 0.090 / 0.235 / 0.064 |
| v3_transition s2 | 0.554 | 1.000 | 0.556 | 0.271 / 0.411 / 0.227 |

Median OT speed (embedding units per h): PCA-20 0.17–0.28, FA-20 0.050–0.096, FA-20 whitened 0.066–0.103, PCA-10 0.15–0.23, PCA-50 0.19–0.30, scVI-10 0.036–0.061.

**(c) Low-variance axes.** U5 is the top-5 eigenvectors of each cell's kNN-30 local data covariance in the embedding. The table gives the AUC of λ₁ of U5ᵀJU5, then the AUC of the soft-axis curvature aᵀJa (S10 Step 4 soft axis).

| dataset | PCA-20 tangent / curvature | FA-20 tangent / curvature | FA-20 whitened tangent / curvature | PCA-10 | PCA-50 | scVI-10 |
|---|---|---|---|---|---|---|
| v3_default s0 | 0.876 / 0.822 | 0.085 / 0.083 | 0.225 / 0.043 | 0.966 / 0.979 | 0.732 / 0.390 | 0.502 / 0.722 |
| v3_default s1 | 0.873 / 0.997 | 0.118 / 0.025 | 0.152 / 0.962 | 0.982 / 0.992 | 0.701 / 0.952 | 0.577 / 0.494 |
| v3_default s2 | 0.976 / 0.999 | 0.091 / 0.203 | 0.327 / 0.760 | 0.995 / 0.991 | 0.915 / 0.554 | 0.628 / 0.928 |
| v3_transition s0 | 0.643 / 0.910 | 0.543 / 0.379 | 0.600 / 0.690 | 0.729 / 0.967 | 0.663 / 0.901 | 0.557 / 0.194 |
| v3_transition s1 | 0.781 / 0.955 | 0.568 / 0.416 | 0.650 / 0.797 | 0.785 / 0.929 | 0.624 / 0.848 | 0.776 / 0.535 |
| v3_transition s2 | 0.882 / 0.970 | 0.605 / 0.558 | 0.672 / 0.708 | 0.905 / 0.983 | 0.809 / 0.967 | 0.539 / 0.519 |

**(d) OT cost whitened by the FA noise variances vs raw FA.** The cost was (zᵢ − zⱼ)ᵀ Wᵀ Ψ⁻¹ W (zᵢ − zⱼ), i.e. ‖Ψ^{-1/2} W Δz‖² (W = FA loadings, Ψ = FA noise variances). ε = 0.05 × the median of this cost. Barycentres, MeasuredDrift and J were computed in raw FA coordinates. Fork AUC, whitened vs raw: 0.207 vs 0.047, 0.970 vs 0.200, 0.897 vs 0.043, 0.625 vs 0.475, 0.787 vs 0.302 and 0.801 vs 0.526 (v3_default s0–s2, v3_transition s0–s2).

## Compute

* CPU **5.0 h of 60** over 49 timed process trees (`cpu_ledger.csv`), including the 0.06 h pre-test. Short probes outside the timed units (a dynamo SparseVFC toy check and a local-ridge toy check, each < 10 s) add an estimated < 0.02 h.
* The embed units (scVI training plus PCA-10/50) took 4.52 h of the total.
* Peak RSS was 6.6 GB. The longest unit took 21.3 min (scVI, v3_transition s1).
