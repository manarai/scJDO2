---
title: "scJDO2: label-free flow operators of single-cell dynamics from sampled time courses"
---

David Redd, Sam Green, Tommy W. Terooatea\*

Genomics & Bioinformatics Center, Department of Biology, Brigham Young University, Provo, UT, USA
\*To whom correspondence should be addressed: tommy.terooatea@byu.edu

Application Note — Systems biology / Gene expression

## Abstract

**Motivation:** Local Jacobians of single-cell dynamics are used to find cells near fate decisions, but they require a velocity, and splicing or metabolic-labeling velocities are often unavailable or disagree with each other. Many single-cell studies have sampling times. Methods that learn dynamics from time courses exist; scJDO2 integrates transport-derived velocity estimation, drift fitting and per-cell operator analysis in one AnnData-compatible call, with documented diagnostics and limitations.

**Results:** scJDO2 provides `scjdo2.flow`, a single call that estimates a velocity by entropic optimal transport between sampling times, fits a smooth drift and returns the per-cell flow operator, its leading eigenvalue λ₁ (largest real part) and magnitude-free operator-space descriptors in an AnnData object. With a-priori settings, λ₁ ranked fate-bifurcation cells above committed cells on held-out synthetic data (AUC 0.89–1.00). On LARRY lineage-traced hematopoiesis, the fitted drift predicted the clonal displacement of barcoded cells beyond population movement and a held-out sampling day better than baselines. With mini-batch transport, the full pipeline processed all 130,887 cells in ≈ 26 min with a peak memory of 2.5 GB. The operator does not recover the values of the true Jacobian and depends on the transport metric; the documented default is an unwhitened PCA with 10–20 components.

**Availability and implementation:** Python (≥ 3.10), MIT license. Source, documentation, test data and reproduction scripts: https://github.com/manarai/scJDO2. Archived version (v0.6.0): Zenodo, doi:10.5281/zenodo.23699152.

**Contact:** tommy.terooatea@byu.edu

**Supplementary information:** The full evaluation, pre-registered test specifications and supporting analyses are in the companion preprint (Redd *et al.*, 2026).

## 1 Introduction

A Jacobian of the velocity field describes how a cell's neighbourhood expands, contracts or rotates. Its leading eigenvalue λ₁ has been used to locate unstable cells and transition genes (Qiu *et al.*, 2022; Bocci *et al.*, 2022; Barcenas *et al.*, 2024). These methods differentiate RNA velocity (La Manno *et al.*, 2018; Bergen *et al.*, 2020) or metabolic-labeling velocity, which are not always available and can disagree: on scNT-seq hematopoiesis, labeling, scVelo and Dynamo velocities had median per-cell cosines between −0.05 and 0.15 (Redd *et al.*, 2026).

Many experiments instead sample cells at several times. Optimal transport (OT) between sampling times gives cell-level couplings and velocities without labels (Schiebinger *et al.*, 2019; Klein *et al.*, 2025), and several methods learn continuous dynamics from such data and differentiate them for gene regulatory networks (Sha *et al.*, 2024; Zhao *et al.*, 2025; Liu *et al.*, 2026). scJDO2 packages these steps into one call that returns a per-cell operator in AnnData format, scales to full datasets and documents what the operator does and does not measure. Beyond the expression matrix, it requires only sampling-time metadata, not splicing or metabolic-labeling measurements (hence "label-free"). The returned operator is the Jacobian of a fitted, transport-derived drift, not an identified regulatory Jacobian; its leading eigenvalue is evaluated here as an empirical branch-point score. It implements the configuration that was evaluated, with pre-registered tests, in the companion preprint (Redd *et al.*, 2026).

## 2 Implementation

`scjdo2.flow(adata, time_key=...)` runs four steps (Fig. 1a).

*OT velocity.* Cells of consecutive sampling times are coupled by balanced entropic OT (Sinkhorn; Cuturi, 2013; POT, Flamary *et al.*, 2021) in unwhitened PCA scores of log-normalized counts (each component keeps its variance; default 15 components), with regularization ε = 0.05 × the median squared distance of the pair; the coupling is the static Schrödinger bridge between the two snapshots with a Brownian reference whose noise level is set by ε (Léonard, 2014). Each cell of the earlier time receives the barycentric velocity (barycentre of its image − z)/Δt. When a pair exceeds 64 × 10⁶ cell pairs, transport switches to mini-batches of 4,000 cells in which every source cell appears ten times (Fatras *et al.*, 2020), and the barycentres are averaged. Cells of the last time point receive no velocity.

*Drift.* A multilayer perceptron *f(z)* (three hidden layers of 96 units; AdamW, weight decay 10⁻⁴) is fitted to the velocities by mean-squared error for 1,500 mini-batch optimizer steps (`n_epochs`; steps, not passes through the data; PyTorch; Paszke *et al.*, 2019) and evaluated at every cell. The drift is autonomous: time enters only through which cells are paired and through Δt, one state-dependent *f(z)* is fitted to all intervals, and operators at the last time point are extrapolated without direct velocity supervision.

*Flow operator.* $J_{\mathrm{flow}}(z) = \partial f/\partial z$ is computed for every cell by batched automatic differentiation (`torch.func`). It is generally non-symmetric and can carry rotation.

*Descriptors.* λ₁ and eight magnitude-free descriptors are returned, the descriptors both as raw values and as within-dataset rank quantiles: six functions of the eigenvalues (sign of Re λ₁, number of eigenvalues with positive real part, Re λ₁/max|λ|, rotational index max|Im λ|/max|λ|, spectral gap and trace normalized by the spectrum), which are invariant under any linear change of coordinates, and Henrici non-normality and transient gain, which are invariant only under orthogonal changes. These invariances hold for a change of coordinates of one fitted field (*J* ↦ *AJA*⁻¹); recomputing transport in another representation changes the fitted field itself (Section 4).

Outputs are written to the AnnData object (Virshup *et al.*, 2024): `obs["flow_lambda1"]`, `obsm["flow_velocity"]`, `obsm["flow_drift"]`, `obsm["flow_descriptors"]` (raw), `obsm["flow_descriptor_ranks"]`, `obs["flow_has_velocity"]`, the operators in `uns["scjdo2"]["J"]` (read by the scJDO2 operator-space and plotting modules) and the parameters, per-pair OT diagnostics and drift-fit summary in `uns["flow"]`. Defaults are the a-priori reference configuration; parameters are documented in `docs/flow.md`, and a quickstart with test data (`examples/flow_quickstart.py`) runs in seconds on a laptop CPU within Scanpy workflows (Wolf *et al.*, 2018).

## 3 Evaluation

We summarize the evaluation of the companion preprint (Redd *et al.*, 2026) and verify the released function. Run through the public API on the three held-out transition test datasets (12,784–12,884 cells), `scjdo2.flow` reproduced the stored fork AUCs exactly (0.960, 0.960, 0.924; per-cell λ₁ Spearman 1.000 with the analysis code) in 11–13 s each on an Apple M3 Max (script `scripts/appnote_validate_api.py`). This checks the code, not robustness; results below use fit seed 0, with refits (seeds 1 and 2) alongside.

*Branching cells, untuned.* On a synthetic benchmark calibrated to six real datasets, λ₁ of the flow operator separated cells at the true saddle from committed cells with a fork AUC (area under the ROC curve of λ₁ for branch-point versus committed cells) of 0.89–1.00 (Fig. 1b). The 11 test datasets are held-out seeds of four presets (three each of transition, batch and stress, two of default), each scored at 10, 15 and 20 components; the generator calibration and the settings were fixed before these seeds were used. Across fit seeds 0–2 the fork AUC was 0.82–1.00 (one of 99 values below 0.85), but per-cell λ₁ rankings agreed only moderately between fits (median Spearman 0.56, minimum −0.04): λ₁ separates groups of cells more reproducibly than it ranks individual cells. A configuration tuned on two development seeds did worse on the same held-out seeds (0.42–0.95), so the released defaults are the untuned ones. On an independent simulator (dyngen; Cannoodt *et al.*, 2021) with eight time points, λ₁ detected two of three bifurcations, the rotational index separated two of three cycles from linear trajectories, and no cell of any linear trajectory had Re λ₁ > 0.

*Lineage and prediction.* On LARRY, a lineage-barcoded in vitro hematopoiesis time course (Weinreb *et al.*, 2020), integrating the drift over each 48-h interval predicted the displacement of barcoded cells to the centroid of their clone at the next sampled time point (554 cell–clone pairs in 405 clone–interval units; Spearman 0.56, pooled over the 15 PCA coordinates; maximum of 1,000 permutations of cell rows 0.15; Fig. 1c). Because cells of a clone share a target, we also resampled clones (95 % interval 0.53–0.59) and permuted clone means within intervals (0.55; maximum of 1,000 permutations 0.18). These intervals are conditional on the fitted drift; refits with seeds 1 and 2 gave 0.55 and 0.56. Agreement persisted after centring each coordinate within each interval (0.50; null maximum 0.04), whereas a population mean-shift predictor reached 0.19, so the drift captures cell-level movement beyond the population trend. λ₁ was unrelated to displacement magnitude (−0.04): lineage validates the velocity and drift, not λ₁. Fitted on days 2 and 4 only, the drift predicted the held-out day-6 distribution better than mean-displacement and no-movement baselines (energy distance, a metric between point clouds that is zero only for identical distributions; Székely and Rizzo, 2013: 0.69, 95 % interval from resampling cells 0.66–0.73, vs 0.87 [0.85–0.89] and 0.88 [0.83–0.94]; seeds 1 and 2: 0.75 and 0.73; Fig. 1d). We did not compare against other learned time-course models.

*Scaling.* Mini-batch OT reproduced dense OT at 8,000 cells per day (median velocity cosine 0.9998; λ₁ Spearman 0.9999) with bounded memory (Fig. 1e). On all 130,887 LARRY cells, transport took 291 s (2.5 GB) and drift fitting with Jacobians 21 min (1.3 GB); dense OT was not run at this size and would need an estimated ≈ 21 GB cost matrix (Fig. 1e shows measured peak memory only). In the tested low-rank configurations (moscot rank 50 and 200), λ₁ did not agree with dense OT (Spearman ≈ 0).

## 4 Usage notes and limitations

The flow operator is not a measured regulatory Jacobian, and λ₁ is evaluated as an empirical branch-point ranking in the stated transport space: compared with the Jacobian of the generator's true velocity, the per-cell relative error was ≈ 1.0 and the λ₁ Spearman correlation at most 0.11 in magnitude. Consistent with eigenvalue-level benchmarks of Jacobian methods (Balubaid *et al.*, 2025), we recommend using λ₁ as a within-dataset branch-point score for groups of cells under the documented preprocessing, checking per-cell values across refits; its sign should not be read as an identified stability classification. The operator depends on the transport metric, because transport and the drift are refitted in each representation: in a separate representation benchmark of the companion preprint, the fork AUC was 0.04–0.53 in Factor Analysis coordinates, against 0.88–1.00 in PCA-10; an unwhitened PCA with 10–20 components is the documented default. Three sampling times were too coarse on dyngen; on real data, per-cell operators from independent velocity sources agreed only at null level (cell-type orderings agreed). Balanced OT treats changes in composition (proliferation, death) as movement of cells. Longer training than the default 1,500 steps lowered the minimum fork AUC (0.78 at 50,000 steps). In simulated steady-state populations, flow operators and potentials fitted to single-window metabolic-labeling velocity did not recover per-cell stability, so we do not recommend these configurations for inferring per-cell stability from single-window labeling data (details in `docs/flow.md`). Detecting the timing of critical transitions and estimating diffusion are not supported. Per-run OT and fit diagnostics are stored in `uns["flow"]`; the benchmark results above are documented in `docs/flow.md` and the companion preprint.

## 5 Conclusion

scJDO2 turns a sampled single-cell time course into per-cell flow operators with one call, at full-dataset scale, with defaults that were fixed before evaluation and an explicit account of what the operator identifies.

## Funding

This work was supported by a New Faculty Start-up award from the College of Life Sciences, Brigham Young University (to T.W.T.).

*Conflict of Interest:* none declared.

## Data availability

LARRY: GEO GSE140802 (Weinreb *et al.*, 2020). scNT-seq hematopoiesis: GEO GSE193517 (Qiu *et al.*, 2022). Synthetic benchmarks are regenerated by scripts in the repository; dyngen settings are listed in the companion preprint. Result tables underlying Fig. 1 are in the repository (`benchmarks/`).

## Figure legend

**Fig. 1. scjdo2.flow and its evaluation.** (**a**) Pipeline and call. (**b**) Fork AUC of λ₁ (branch-point vs committed cells) on 11 held-out synthetic datasets at 10, 15 and 20 principal components with the released defaults; dotted line, chance. (**c**) LARRY: Spearman correlation, pooled over 15 PCA coordinates of 554 barcoded cell–clone pairs, between drift-predicted and observed clonal displacement, before and after centring each coordinate within a sampling interval (black line, maximum of 1,000 permutations), for a population mean-shift predictor, and between λ₁ and displacement magnitude; nulls permute cell rows (clone-level checks in the text). (**d**) Energy distance between the predicted and observed held-out day-6 distributions for the drift (fitted on days 2 and 4) and two baselines; error bars, 95 % bootstrap intervals (200 resamples of cells). (**e**) Measured peak memory of the OT stage for dense and mini-batch OT; star, all 130,887 cells (total run time 26 min). Source data: `benchmarks/synthetic_multimodal/s14/` and `benchmarks/scaling/`; figure script `notebooks/appnote_fig1.py`.

## References

Balubaid, A. *et al.* (2025) Fundamental limits of inferring dynamical gene regulatory models from single-cell data. *bioRxiv*, doi:10.1101/2025.09.12.674509.

Barcenas, M., Bocci, F. and Nie, Q. (2024) Tipping points in epithelial-mesenchymal lineages from single-cell transcriptomics data. *Biophys. J.*, **123**, 2849–2859.

Bergen, V. *et al.* (2020) Generalizing RNA velocity to transient cell states through dynamical modeling. *Nat. Biotechnol.*, **38**, 1408–1414.

Bocci, F., Zhou, P. and Nie, Q. (2022) spliceJAC: transition genes and state-specific gene regulation from single-cell transcriptome data. *Mol. Syst. Biol.*, **18**, e11176.

Cannoodt, R. *et al.* (2021) Spearheading future omics analyses using dyngen, a multi-modal simulator of single cells. *Nat. Commun.*, **12**, 3942.

Cuturi, M. (2013) Sinkhorn distances: lightspeed computation of optimal transport. *Adv. Neural Inf. Process. Syst.*, **26**, 2292–2300.

Fatras, K. *et al.* (2020) Learning with minibatch Wasserstein: asymptotic and gradient properties. *Proc. AISTATS*, PMLR **108**, 2131–2141.

Flamary, R. *et al.* (2021) POT: Python Optimal Transport. *J. Mach. Learn. Res.*, **22**, 1–8.

Klein, D. *et al.* (2025) Mapping cells through time and space with moscot. *Nature*, **638**, 1065–1075.

La Manno, G. *et al.* (2018) RNA velocity of single cells. *Nature*, **560**, 494–498.

Léonard, C. (2014) A survey of the Schrödinger problem and some of its connections with optimal transport. *Discrete Contin. Dyn. Syst.*, **34**, 1533–1574.

Liu, J. *et al.* (2026) Learning stochastic dynamics and cell-fate landscapes from single-cell snapshots via optimal transport. *Sci. Adv.*, **12**, eaeb4205.

Paszke, A. *et al.* (2019) PyTorch: an imperative style, high-performance deep learning library. *Adv. Neural Inf. Process. Syst.*, **32**, 8024–8035.

Qiu, X. *et al.* (2022) Mapping transcriptomic vector fields of single cells. *Cell*, **185**, 690–711.

Redd, D., Green, S. and Terooatea, T.W. (2026) Label-free flow operators and stability operators from single-cell time courses: what local dynamical operators can and cannot identify. *bioRxiv*, doi:10.64898/2026.08.21.745996 (version 57).

Schiebinger, G. *et al.* (2019) Optimal-transport analysis of single-cell gene expression identifies developmental trajectories in reprogramming. *Cell*, **176**, 928–943.

Sha, Y. *et al.* (2024) Reconstructing growth and dynamic trajectories from single-cell transcriptomics data. *Nat. Mach. Intell.*, **6**, 25–39.

Székely, G.J. and Rizzo, M.L. (2013) Energy statistics: a class of statistics based on distances. *J. Stat. Plan. Inference*, **143**, 1249–1272.

Virshup, I. *et al.* (2024) anndata: access and store annotated data matrices. *J. Open Source Softw.*, **9**, 4371.

Weinreb, C. *et al.* (2020) Lineage tracing on transcriptional landscapes links state to fate during differentiation. *Science*, **367**, eaaw3381.

Wolf, F.A., Angerer, P. and Theis, F.J. (2018) SCANPY: large-scale single-cell gene expression data analysis. *Genome Biol.*, **19**, 15.

Zhao, W. *et al.* (2025) Optimal transport reveals dynamic gene regulatory networks via gene velocity estimation. *PLoS Comput. Biol.*, **21**, e1012476.
