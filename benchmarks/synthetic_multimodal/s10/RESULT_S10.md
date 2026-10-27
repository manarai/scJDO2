# orient/S10 — result (executor: local Claude agent)

**v3 caveats (S09; they apply to every splicing- or lineage-based number below):** unspliced share 0.40–0.52 (above the 0.25–0.35 band, inside real 0.08–0.50); splicing oracle 0.21–0.28; steady genes ≈ 1 %; scVelo recovers more genes than on real data (easier than real); multi-time clones 0.44–0.49; genes detected ≈ 830 / cell. Splicing-based numbers are marked *[splicing]*, lineage-based numbers *[lineage]*.

Machine-readable gate values: `transition_gates.json`, `orientation_gates.json`, `diffusion_summary.json`, `metrics.json`. Tables: `operators.csv`, `transitions.csv`, `transition_bins.csv`, `results.csv`, `palantir_runs.csv`, `lineage_sensitivity.csv`, `diffusion.csv`, `real_transitions.csv`, `real_transitions_agreement.csv`, `s08_transition_vs_default_*.csv`. CPU per unit: `cpu_ledger.csv`.

## Gates (verbatim from the task) and results

> **Orientation:** `{labelling}` agreement ≥ 0.80; `{lineage, pseudotime}` agreement ≥ `{pseudotime}` − 0.01; independence exact; J unchanged.

**Result: NOT EVALUABLE AS WRITTEN (3 of 4 clauses pass; the `{lineage, pseudotime}` clause cannot be computed).**

| clause | value (v3_default, PCA, raw labelling, check on; s0 / s1 / s2) | verdict |
|---|---|---|
| `{labelling}` agreement ≥ 0.80 (expected-count flow truth) | 0.957 / 0.978 / 0.978 | pass |
| `{lineage, pseudotime}` ≥ `{pseudotime}` − 0.01 | needs the forward developmental truth; the protected `forward_developmental_truth` supports v1/v2 only and raises for v3 | not evaluable |
| independence exact (all-four call = flow and development calls) | true on every unit | pass |
| J unchanged (operator hash) | true on every call | pass |

Report only: primary variant (`smoothed_shared`) `{labelling}` agreement 0.644 / 0.634 / 0.668. Substitute development truth (clone chord, *[lineage]*), strict-mode day-2 runs (no default-mode day-2 run was accepted): `{lineage, pseudotime}` 0.731 / 0.610 / 0.576 vs `{pseudotime}` 0.712 / 0.554 / 0.517.

> **Guard (fresh runs):** every latest-day-root run refused; ≥ 90 % of day-2-root runs accepted, each mode.

**Result: FAIL.**

| clause | default mode | strict mode | verdict |
|---|---|---|---|
| every latest-day (day-6) root refused | 30 / 30 refused | 30 / 30 refused | pass |
| ≥ 90 % of day-2-root runs accepted | **0 / 30 (0 %)** | 30 / 30 (100 %) | fail (default) |

Counts are 5 roots × 3 seeds × 2 embeddings, v3_default. In default mode Palantir moved the start cell of every day-2 run to a day-4 or day-6 cell, and the §16 rule refused each of them. Other presets: `palantir_runs.csv` and Step 2 below.

> **Flow operator:** `J_flow` AUC (saddle vs committed) ≥ 0.85 on every seed.

**Result: FAIL.** `J_flow` (raw labelling, shared γ), v3_default, PCA: AUC **0.840 / 0.533 / 0.747** (saddle n = 22 / 28 / 29; committed n = 12 594 / 12 550 / 12 781). Report only: primary variant `smoothed_shared` 0.732 / 0.796 / 0.887; `raw_perpulse` 0.846 / 0.544 / 0.749; `J_flow_scvelo` 0.269 / 0.507 / 0.333 *[splicing]*; `J_flow_ot` 0.974 / 0.990 / 0.999; score `J` 0.311 / 0.226 / 0.165.

> **Critical transition** (`v3_transition`; every seed informative): with `J_flow` **and separately with `J_flow_ot`**, |detected − true bin| ≤ 1 on every seed, and the null peak-to-median ratio of `λ₁` on `v3_stationary_cycle` is below its value at the detected transition on every seed. Also report the signed offset (detected − true bin): a consistent negative offset is an early warning only if the signal rises over ≥ 2 bins before it.

**Result: FAIL — untestable (`J_flow` and `J_flow_ot`).** True transition bin 1 / 5 / 0: seeds 0 and 2 are uninformative (true bin < 2).

| | s0 | s1 | s2 |
|---|---|---|---|
| true bin (informative?) | 1 (no) | 5 (yes) | 0 (no) |
| `J_flow` λ₁ detected bin; signed offset | 3; +2 | 5; 0 | 0; 0 |
| `J_flow` peak/median at transition vs null (cycle, `J_flow` λ₁) | 1.049 vs 1.156 | 1.707 vs 1.093 | 1.238 vs 1.057 |
| `J_flow` null below transition | no | yes | yes |
| `J_flow` bins rising before detected bin | 3 | 4 | 0 |
| `J_flow_ot` λ₁ detected bin; signed offset | 0; −1 | 5; 0 | 0; 0 |
| `J_flow_ot` peak/median at transition | 4.23 | 6.86 | 8.81 |
| `J_flow_ot` null | not computable: `v3_stationary_cycle` has one sampling time, so no OT flow | | |
| `J_flow_ot` bins rising before detected bin | 0 | 2 | 0 |

Signed offsets: `J_flow` +2 / 0 / 0, `J_flow_ot` −1 / 0 / 0, so no consistent negative offset. Seed 1, the only informative seed, comes from a default-mode run that started on day 6 (see Step 4). Report only, primary variant `smoothed_shared`: detected 1 / 6 / 0 (offsets 0 / +1 / 0); null 1.26 / 1.95 / 1.66 vs transition 1.10 / 1.18 / 1.53, so the null is below the transition value on no seed.

> **Diffusion** (`v3_default`, PCA, every seed): clone-dispersion and OT estimates each within a factor 2 of the truth's median effective diffusion (Δt = 48 h); score–flow estimate Spearman ≥ 0.5 with the truth over cells whose true state is a stable basin (`committed`).

**Result: FAIL.** v3_default, PCA; truth median `trace(D_eff(48 h))/d` = 0.105 / 0.104 / 0.088.

| clause | s0 / s1 / s2 | verdict |
|---|---|---|
| clone dispersion within ×2 (Σ_meas subtracted as written) *[lineage]* | ratio −1.83 / −1.59 / −1.88 | fail |
| OT noise within ×2 (minus trace(Σ_meas)/d as written) | ratio −1.16 / −1.09 / −1.25 | fail |
| score–flow Spearman ≥ 0.5 over committed cells (Δt = 6 h) | 0.115 / 0.052 / 0.090 | fail |

Report only: clone dispersion uncorrected −0.007 / 0.181 / 0.169; OT noise uncorrected 0.468 / 0.498 / 0.470 (the held-out selection hit the smallest grid value on every pair; the selection rule is the executor's construction, see Deviations); `Σ_meas` (median trace/d 0.15–0.17) exceeds the 48-h effective diffusion.

Gate scope as fixed in the task: `v3_default`, PCA, raw labelling (Step 1's `J_flow` = MeasuredDrift on the raw labelling velocity, shared γ = variant `raw_shared`), seeds 0–2; the transition gates use `v3_transition`. Step 0b names smoothed + shared γ (`smoothed_shared`) the primary labelling variant; it is reported next to each gate, not substituted for it.

## What ran

All units ran as detached `/usr/bin/time -l` processes with a checkpoint each (`benchmarks/synthetic_multimodal/s10/checkpoints/`, logs in `.../s10/logs/`), at most 12 threads in total, one scVelo at a time, `LOKY_MAX_CPU_COUNT=2`.

1. `python -m pytest -q tests/test_orient.py tests/test_smoke.py tests/test_synthetic_multimodal.py`: 74 passed (before).
2. Generator edits (Step 0/0b/6) plus tests; fingerprint check: v1/v2/v3 outputs unchanged.
3. `python scripts/s10_fit_and_runs.py generate --preset v3_transition --seed {0,1,2}`, then `s08 --preset v3_transition --seed {0,1,2}`, then `s08compare`.
4. `python scripts/s10_fit_and_runs.py fit --preset P --seed N --method {pca,fa}` (P ∈ v3_default, v3_stress, v3_batch, v3_stationary_cycle, v3_transition; 30 units).
5. `python scripts/s10_fit_and_runs.py palantir --preset P --seed N` (15 units) and `scvelo --preset P --seed N` (15 units).
6. `python scripts/s10_fit_and_runs.py flow --preset P --seed N --method M` (30 units: J_flow_scvelo, J_flow_ot, growth-weighted OT).
7. `python scripts/s10_operators.py` (Step 3); `python scripts/s10_transitions.py` (Step 4).
8. `python scripts/s10_diffusion.py meas --preset v3_default --seed N`; `truth --preset v3_default --seed N --chunk k` (k = 0–11; 36 units); `python scripts/s10_real_transitions.py diffusion`; `python scripts/s10_diffusion.py score` (Step 6).
9. `python scripts/s10_real_transitions.py prepare | fit --dataset D --embedding E --seed S | palantir --dataset D | score` (Step 5).
10. `python scripts/s10_score.py --preset P` (4 units), then `--combine` (Step 2).
11. `python -m pytest -q tests/test_orient.py tests/test_smoke.py tests/test_synthetic_multimodal.py`: 77 passed (after; 3 new S10 tests).

## Step 0 — v3_transition (z1_crit = 2.0, pulse-group batch effect) and realism

* `z1_crit` (config field of `_V3Config`, default 1.0) now sets the pitchfork `μ = z1 − z1_crit`, the cycle-exit gate, the fate label (`z1 > z1_crit + 0.2`) and the fate genes' commitment gate (`z1 > z1_crit + 0.2`). With the default, v1/v2/v3 arrays and the stored v3 metadata are byte-identical (fingerprint of small v1/v2/v3/v3_batch/v3_stationary_cycle datasets before and after; tests added). Protected functions unchanged.
* `v3_transition` = `v3_default` knobs, `z1_crit = 2.0`, `pulse_batch_effect = True` (one `LogNormal(0, 0.3)` per-gene factor per (day, pulse) group, own seeded stream `SeedSequence([seed, 0x50554C53])`, stored as `uns["pulse_batch_gene_effects"]`; the effective factor is stored as `batch_gene_effects` with `batch_day = 2·collection + (pulse == 5) + 1`, so every truth function maps expected counts with the factor each cell received).
* Generated seeds 0–2 (MANIFEST_v3 rows): 12 701 / 12 654 / 12 884 cells, cells by day [187, 1448, 11066] / [186, 1440, 11028] / [190, 1457, 11237] (identical counts to v3_default: the genealogy RNG stream is unchanged). Mean true z1 (s0 / s1 / s2): day 2 1.42 / 1.39 / 1.43 (μ < 0), day 4 2.17 / 2.16 / 2.18 (just past), day 6 2.57 / 2.56 / 2.57.
* S08 statistics (imported S08 functions; `s08_transition_vs_default_targets.csv`, `s08_v3_transition_stats.csv`): 13-target realism count **6 / 6 / 6** for v3_transition vs 6 / 6 / 6 for v3_default (same targets on seeds 1 and 2; seed 0: C7 met by v3_default only, L2 by v3_transition only). Largest moves: C7 participation ratio 8.03–8.86 → 9.13–9.27; C4 dispersion 1.85–2.45 → 1.97–2.76; Palantir terminal states 1 → 2 and Palantir entropy median 0 → 0.69; clone label purity (G5) median 0.92 → 0.60. S1 unspliced 0.398 / 0.510 / 0.437 *[splicing]*; G3 multi-time clones 0.460 / 0.501 / 0.452 *[lineage]*.

## Step 1 — fits and user runs (all v3 presets incl. v3_transition; PCA and FA)

* Energy drift + `J` (S06 configuration; symmetry max ≤ 3e-7, gate 1e-4), labelling velocity in the four Step 0b variants (`raw_shared` = S02/S06 `LabelledRateModel`, `raw_perpulse` = one `LabelledRateModel` per pulse group, `smoothed_*` = kNN-30 normalised moments as `labelling_diag_2026-10-19/diag.py`, mapped back to each cell's count units by its size factor before the S02 chain rule), `J_flow_<variant>` = `jacobian_autodiff` of `MeasuredDrift(d, seed=0)` per variant.
* scVelo: exact S03 block (`s06_user_runs.run_scvelo`), 2 512 genes kept on v3_default s0; stochastic mode raised `ValueError` (optional in S03/S06; recorded) *[splicing]*. `J_flow_scvelo` on scVelo's embedded velocity.
* Palantir: 5 day-2 roots (`rng(1000+seed)`) and 5 day-6 roots (`rng(2000+seed)`), default and strict; actual start cell recorded (`palantir_runs.csv`). In **default** mode Palantir moved the start cell off day 2 in 30 / 30 v3_default runs, 30 / 30 v3_stress, 22 / 30 v3_batch and 10 / 30 v3_transition runs (to a day-4 or day-6 cell).
* `J_flow_ot`: POT `ot.sinkhorn` (moscot not installed), balanced, ε = 0.05 × median squared distance per day pair (v3_default s0 PCA: ε ≈ 38.0 and 37.3), barycentric velocity / 48 h, MeasuredDrift on the day-2 and day-4 cells (1 635 cells), Jacobian on all cells. Growth-weighted sensitivity: `ot.unbalanced.sinkhorn_unbalanced`, source masses `exp(Δt·log 2 / 16 h)·(1 − f_sample) / n_target`, target `1 / n_target`, `reg_m = c·ε`, c ∈ {0.1, 1, 10} (`J_flow_ot_ub<c>`). The generator's division rate is the same for every cell, so the growth weights are uniform within a day; they set only the total source mass (≈ 1.00–1.02).

## Step 3 — operators for forks and basins (`operators.csv`)

Truth classes (S06 Step 4 definitions on the core dims, μ = z1 − z1_crit). v3_default has few saddle cells (22 / 28 / 29 per seed; committed 12 594 / 12 550 / 12 781; progenitor 5 / 2 / 1); v3_transition 350 / 368 / 364 saddle and 8 998 / 8 859 / 9 186 committed cells.

**v3_default, PCA** (median over seeds of the per-class median leading real eigenvalue; AUC saddle vs committed per seed; median |cos| of the leading eigenvector with the fate axis at saddle cells):

| operator | saddle λ₁ | committed λ₁ | AUC s0 / s1 / s2 | |cos| fate axis s0 / s1 / s2 |
|---|---|---|---|---|
| truth (`jacobian_state`) | 0.136 | −0.013 | 1.000 | — |
| J_score | 0.0531 | 0.0948 | 0.311 / 0.226 / 0.165 | 0.26 / 0.08 / 0.13 |
| **J_flow_raw_shared (gate)** | 0.0309 | 0.0182 | **0.840 / 0.533 / 0.747** | 0.15 / 0.09 / 0.07 |
| J_flow_raw_perpulse | 0.0325 | 0.0178 | 0.846 / 0.544 / 0.749 | 0.15 / 0.12 / 0.06 |
| J_flow_smoothed_shared (primary variant) | 0.0148 | 0.0073 | 0.732 / 0.796 / 0.887 | 0.20 / 0.27 / 0.16 |
| J_flow_smoothed_perpulse | 0.0156 | 0.0073 | 0.735 / 0.821 / 0.881 | 0.27 / 0.12 / 0.16 |
| J_flow_scvelo *[splicing]* | 0.0059 | 0.0095 | 0.269 / 0.507 / 0.333 | 0.12 / 0.09 / 0.20 |
| J_flow_ot | 0.0165 | 0.0001 | 0.974 / 0.990 / 0.999 | 0.98 / 0.97 / 0.99 |
| J_flow_ot_ub0.1 / ub1 / ub10 | 0.010 / 0.015 / 0.020 | −0.0002 / −0.0003 / −0.0008 | 0.998–1.000 (all seeds) | 0.93–0.99 |

Fraction λ₁ > 0 (v3_default PCA, median over seeds): saddle 1.00 for every flow operator, 0.89 for J_score; committed 1.00 (labelling raw), 0.99 (labelling smoothed), 1.00 (scVelo), 0.51 (OT), 0.96 (J_score).

**AUC saddle vs committed, PCA, all developmental presets** (s0 / s1 / s2; `operator_auc_by_preset.png`):

| operator | v3_default | v3_stress | v3_batch | v3_transition |
|---|---|---|---|---|
| J_score | 0.31 / 0.23 / 0.16 | 0.15 / 0.50 / 0.15 | 0.17 / 0.21 / 0.29 | 0.43 / 0.48 / 0.52 |
| J_flow_raw_shared | 0.84 / 0.53 / 0.75 | 0.67 / 0.69 / 0.69 | 0.54 / 0.72 / 0.63 | 0.50 / 0.68 / 0.52 |
| J_flow_raw_perpulse | 0.85 / 0.54 / 0.75 | 0.66 / 0.68 / 0.70 | 0.57 / 0.73 / 0.64 | 0.50 / 0.70 / 0.52 |
| J_flow_smoothed_shared | 0.73 / 0.80 / 0.89 | 0.70 / 0.83 / 0.79 | 0.92 / 0.78 / 0.92 | 0.52 / 0.55 / 0.61 |
| J_flow_smoothed_perpulse | 0.73 / 0.82 / 0.88 | 0.75 / 0.77 / 0.83 | 0.87 / 0.80 / 0.93 | 0.53 / 0.54 / 0.61 |
| J_flow_scvelo *[splicing]* | 0.27 / 0.51 / 0.33 | 0.27 / 0.34 / 0.45 | 0.57 / 0.36 / 0.50 | 0.38 / 0.48 / 0.52 |
| J_flow_ot | 0.97 / 0.99 / 1.00 | 1.00 / 0.98 / 0.98 | 1.00 / 0.98 / 0.99 | 0.82 / 0.91 / 0.97 |
| J_flow_ot_ub1 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 0.97 | 1.00 / 0.99 / 1.00 | 0.75 / 0.71 / 0.77 |

FA (report only): J_flow_ot 0.04–0.25 on v3_default/stress/batch; labelling raw 0.64–0.95; smoothed 0.58–0.99 (full table in `operators.csv`).

**v3_stationary_cycle** (median max |imag| of the eigenvalues; truth `jacobian_state` 0.296 in the latent space on every seed): PCA J_flow_raw_shared 0.145 / 0.166 / 0.165, J_flow_smoothed_shared 0.068 / 0.071 / 0.068, J_flow_scvelo 0.010 / 0.011 / 0.009 *[splicing]*, J_score 0 (symmetric); FA raw 0.142 / 0.177 / 0.175. No `J_flow_ot` on the cycle (one sampling time).

## Step 4 — critical transitions (`transitions.csv`, `transition_bins.csv`, `transition_bins_v3_transition.png`)

True transition bin (argmax of the per-bin median κ(μ − 3z0²), 12 equal-count bins of the day-2 default-mode r0 Palantir pseudotime): **v3_transition 1 / 5 / 0** → seed 1 informative, seeds 0 and 2 **uninformative** (true bin < 2). Per-bin median μ, seed 0: −0.05, 0.15, 0.29, …, 0.73; seed 2: −0.01, 0.23, 0.42, …, 0.72: μ crosses 0 inside bin 0 because day 2 (≈ 1.5 % of cells) and day 4 (≈ 11 %) fill about the first 1.5 of 12 equal-count bins. Seed 1: the default-mode r0 run started on a **day-6** cell (Palantir moved the start); μ per bin falls from 0.68 to 0.07 (bin 5) and rises again to 0.74, so its true bin 5 is the low-μ middle of a folded ordering. Strict-root sensitivity (report only): true bins 1 / 1 / 0 (all uninformative). Other presets (report only): v3_default 6 / 6 / 6, v3_stress 6 / 6 / 6, v3_batch 6 / 6 / 0 (every cell past the fork; the truth's maximum is a dip of μ in mixed-day bins).

**v3_transition, PCA, detected bin per signal** (true bins 1 / 5 / 0):

| operator | signal | detected bin s0 / s1 / s2 | peak/median s0 / s1 / s2 | ρ before true bin (s1) |
|---|---|---|---|---|
| J_flow_ot | lam1 | 0 / 5 / 0 | 4.23 / 6.86 / 8.81 | 0.90 |
| J_flow_ot | soft_curv | 0 / 5 / 0 | 4.12 / 3.26 / 5.50 | 1.00 |
| J_flow_ot | var_soft | 1 / 6 / 1 | 6.64 / 6.21 / 4.94 | 1.00 |
| J_flow_ot_ub0.1 | lam1 | 11 / 10 / 0 | 1.74 / 1.43 / 1.13 | -1.00 |
| J_flow_ot_ub0.1 | soft_curv | 11 / 11 / 7 | 1.90 / 1.44 / 1.21 | -1.00 |
| J_flow_ot_ub0.1 | var_soft | 1 / 6 / 1 | 4.48 / 2.06 / 3.98 | 1.00 |
| J_flow_ot_ub1 | lam1 | 0 / 6 / 0 | 1.70 / 1.16 / 1.46 | 0.30 |
| J_flow_ot_ub1 | soft_curv | 11 / 10 / 0 | 2.17 / 2.19 / 1.31 | 1.00 |
| J_flow_ot_ub1 | var_soft | 1 / 6 / 1 | 6.57 / 5.31 / 4.97 | 1.00 |
| J_flow_ot_ub10 | lam1 | 0 / 5 / 0 | 7.81 / 5.84 / 4.71 | 1.00 |
| J_flow_ot_ub10 | soft_curv | 0 / 5 / 0 | 7.85 / 3.17 / 3.39 | 1.00 |
| J_flow_ot_ub10 | var_soft | 1 / 6 / 1 | 6.79 / 6.25 / 4.94 | 1.00 |
| J_flow_raw_perpulse | lam1 | 3 / 5 / 0 | 1.07 / 1.68 / 1.23 | 0.70 |
| J_flow_raw_perpulse | soft_curv | 9 / 10 / 11 | -0.75 / 0.10 / 4.38 | 0.00 |
| J_flow_raw_perpulse | var_soft | 1 / 6 / 2 | 1.18 / 1.28 / 1.78 | 1.00 |
| J_flow_raw_shared | lam1 | 3 / 5 / 0 | 1.05 / 1.71 / 1.24 | 0.90 |
| J_flow_raw_shared | soft_curv | 9 / 10 / 11 | -0.74 / 0.55 / 5.81 | 0.10 |
| J_flow_raw_shared | var_soft | 1 / 6 / 2 | 1.17 / 1.29 / 1.78 | 1.00 |
| J_flow_scvelo | lam1 | 9 / 7 / 10 | 1.21 / 1.26 / 1.05 | -0.50 |
| J_flow_scvelo | soft_curv | 9 / 10 / 3 | 1.85 / 1.60 / -0.34 | -0.40 |
| J_flow_scvelo | var_soft | 0 / 5 / 1 | 1.45 / 1.32 / 1.37 | 1.00 |
| J_flow_smoothed_perpulse | lam1 | 0 / 11 / 0 | 1.16 / 1.13 / 1.60 | 1.00 |
| J_flow_smoothed_perpulse | soft_curv | 11 / 8 / 2 | -0.48 / -0.85 / -0.49 | -0.50 |
| J_flow_smoothed_perpulse | var_soft | 0 / 5 / 0 | 1.63 / 1.37 / 1.51 | 1.00 |
| J_flow_smoothed_shared | lam1 | 1 / 6 / 0 | 1.10 / 1.18 / 1.53 | 1.00 |
| J_flow_smoothed_shared | soft_curv | 11 / 11 / 3 | -0.53 / 0.02 / -0.56 | -0.70 |
| J_flow_smoothed_shared | var_soft | 0 / 4 / 0 | 1.59 / 1.35 / 1.47 | 0.90 |
| J_score | lam1 | 8 / 6 / 1 | 1.05 / 1.13 / 1.36 | 0.00 |
| J_score | soft_curv | 0 / 5 / 0 | -0.25 / -0.18 / -0.22 | 0.00 |
| J_score | var_soft | 1 / 5 / 1 | 1.77 / 2.01 / 2.55 | 1.00 |
| baseline | critical_state_index | 11 / 11 / 3 | 1.60 / 1.49 / 1.20 | 1.00 |

**Null, v3_stationary_cycle** (random-cell Palantir root `rng(1000+seed)`, default mode; peak-to-median ratio of the bin medians):

| operator | signal | null peak/median s0 / s1 / s2 |
|---|---|---|
| J_flow_raw_perpulse | lam1 | 1.17 / 1.10 / 1.04 |
| J_flow_raw_perpulse | soft_curv | 1.71 / 0.81 / 3.85 |
| J_flow_raw_perpulse | var_soft | 1.55 / 2.42 / 2.23 |
| J_flow_raw_shared | lam1 | 1.16 / 1.09 / 1.06 |
| J_flow_raw_shared | soft_curv | 1.59 / 0.77 / 3.90 |
| J_flow_raw_shared | var_soft | 1.54 / 2.44 / 2.24 |
| J_flow_scvelo | lam1 | 1.24 / 1.35 / 1.27 |
| J_flow_scvelo | soft_curv | 1.89 / 2.18 / -0.06 |
| J_flow_scvelo | var_soft | 1.28 / 1.21 / 1.88 |
| J_flow_smoothed_perpulse | lam1 | 1.47 / 2.08 / 1.56 |
| J_flow_smoothed_perpulse | soft_curv | -0.26 / 0.20 / -0.73 |
| J_flow_smoothed_perpulse | var_soft | 1.61 / 1.39 / 1.23 |
| J_flow_smoothed_shared | lam1 | 1.26 / 1.95 / 1.66 |
| J_flow_smoothed_shared | soft_curv | 0.14 / 0.19 / -0.64 |
| J_flow_smoothed_shared | var_soft | 1.89 / 1.16 / 1.28 |
| J_score | lam1 | 1.06 / 1.06 / 1.14 |
| J_score | soft_curv | -0.69 / -0.23 / -0.55 |
| J_score | var_soft | 1.12 / 1.40 / 1.66 |
| baseline | critical_state_index | 1.32 / 1.10 / 1.63 |

## Step 6 — diffusion (`diffusion.csv`, `diffusion_summary.json`, `diffusion_ratio_by_estimator.png`)

Truth: `forward_displacement_covariance` (32 paths, all latent dims incl. the 5 nuisance programmes, RNA kinetics, no division), Δt ∈ {6, 24, 48} h nested on the same paths, `D_eff = Cov / (2Δt)`, computed in 12 cell chunks per seed (seed = chunk index). Median truth `trace(D_eff)/d`, v3_default PCA: Δt = 6 h 0.0258 / 0.0175 / 0.0331; 24 h 0.118 / 0.111 / 0.112; **48 h 0.105 / 0.104 / 0.088** (FA 48 h 0.0062 / 0.0059 / 0.0066). Measurement-noise reference (counts redrawn from the true pools, noise seed 1): median `trace(Σ_meas)/d` PCA 0.171 / 0.164 / 0.152, FA 0.033 / 0.037 / 0.037 — larger than the 48-h effective diffusion.

**v3_default, PCA — median estimate / median truth (all cells; Δt matched)** and Spearman with the truth:

| estimator | variant | ratio s0 / s1 / s2 | Spearman, committed cells s0 / s1 / s2 | Spearman, all cells |
|---|---|---|---|---|
| clone dispersion (48 h) *[lineage]* | literal (− Σ_meas, task text; gate) | **−1.83 / −1.59 / −1.88** | −0.035 / 0.032 / −0.056 | −0.035 / 0.034 / −0.055 |
| clone dispersion *[lineage]* | uncorrected | −0.007 / 0.181 / 0.169 | −0.007 / 0.042 / −0.049 | |
| clone dispersion *[lineage]* | − (Σ_meas,t2 − Σ_meas,t1)/(2Δt) | −0.011 / 0.181 / 0.183 | −0.007 / 0.041 / −0.050 | |
| OT noise (48 h) | − trace(Σ_meas)/d (task text; gate) | **−1.16 / −1.09 / −1.25** | — (Spearman over 2 day pairs) | |
| OT noise | uncorrected | 0.468 / 0.498 / 0.470 | | |
| score–flow (6 h), trace/d | raw_shared (gate) | 0.166 / 0.144 / 0.203 | **0.115 / 0.052 / 0.090** | 0.119 / 0.059 / 0.091 |
| score–flow, trace/d | raw_perpulse | 0.173 / 0.161 / 0.202 | 0.116 / 0.056 / 0.090 | |
| score–flow, trace/d | smoothed_shared (primary) | 0.458 / 0.823 / 0.640 | 0.295 / 0.142 / 0.015 | |
| score–flow, trace/d | smoothed_perpulse | 0.458 / 0.816 / 0.638 | 0.293 / 0.147 / 0.017 | |
| score–flow, soft axis | raw_shared | 2.02 / −0.62 / 0.16 | 0.058 / −0.105 / 0.015 | |
| score–flow, soft axis | smoothed_shared | 1.73 / 1.21 / 1.00 | 0.018 / −0.013 / 0.022 | |

Clone dispersion uses 51 / 45 / 55 clone pairs (all day 4 → 6; no day-2 clone has ≥ 3 cells). OT noise: the held-out selection chose the smallest grid value (0.25 × ε) on every seed and day pair (boundary); D̂ = 0.049 / 0.052 / 0.041 uncorrected. Score–flow: 30 % of cells have D̂ < 0 (v3_default s0 PCA, raw).

**Payoff** (`D̂ · J_score` vs `J_flow_raw_shared`, v3_default PCA; median relative Frobenius distance, Spearman of λ₁): unscaled 18.5 / 15.0 / 12.4, ρ 0.087 / 0.060 / 0.082; per-cell score–flow D̂ 1.006 / 1.003 / 1.003, ρ 0.074 / 0.001 / 0.070; median score–flow D̂ 1.002 / 1.001 / 1.001; clone (uncorrected) 1.000 / 1.041 / 1.012, ρ 0.21 / 0.06 / 0.08; OT (uncorrected) 1.34 / 1.27 / 1.11. A scalar scale leaves the λ₁ ranking unchanged except for its sign.

**Real data, report only:** LARRY clone dispersion (uncorrected; all cells with exactly one barcode; 1 254 clone pairs) median 0.00264 *[lineage]*; LARRY OT noise 0.0050 (2 → 4), 0.0089 (4 → 6), both at the grid boundary 0.25 × ε; scNT-seq score–flow median D̂ −0.00054 (raw_shared; 77 % of cells negative), −0.0027 (smoothed_shared; 100 % negative); per-pulse variants −0.00064 / −0.0032.

## Step 2 — orientation scoring (secondary; `results.csv`, `palantir_runs.csv`, `lineage_sensitivity.csv`, `diagnostics.csv`, `orientation_gates.json`)

* `{labelling}` flow agreement (expected-count flow truth, check on), PCA, s0 / s1 / s2: raw_shared 0.957 / 0.978 / 0.978 (v3_default), 0.970 / 0.970 / 0.971 (stress), 0.961 / 0.978 / 0.977 (batch), 0.955 / 0.976 / 0.975 (transition); raw_perpulse within 0.002 of raw_shared; S06 MeasuredDrift-smoothed 0.735–0.805; **kNN-smoothed (primary variant) 0.615–0.668**.
* `{velocity}` (scVelo) 0.46 / 0.53 / 0.57 (v3_default) *[splicing]*; `{labelling, velocity}` agreement equals `{labelling}` on every unit and variant (scVelo demoted by the reliability check, e.g. v3_default s0 PCA velocity–labelling agreement 0.49 < 0.6).
* Guard (fresh runs, both embeddings): latest-day roots refused 30 / 30 in each mode on every preset; day-2 roots accepted: v3_default default 0 / 30, strict 30 / 30; v3_stress 0 / 30, 30 / 30; v3_batch 8 / 30, 28 / 30; v3_transition 20 / 30, 26 / 30. Every refused day-2 default-mode run had its actual start cell on day 4 or 6 (moved / refused: v3_default 30 / 30, v3_stress 30 / 30, v3_batch 22 / 22, v3_transition 10 / 10); strict mode: v3_transition 4 runs moved and refused, v3_batch 2 runs refused with the start on day 2.
* Development truth: the protected `forward_developmental_truth` raises for generator v3, so the forward truth could not be computed; development agreement is reported against the clone chord truth (S06's diagnostic) as a substitute *[lineage]*: v3_default PCA strict day-2 runs, median per seed: `{pseudotime}` 0.712 / 0.554 / 0.517, `{lineage, pseudotime}` 0.731 / 0.610 / 0.576, `{lineage}` 0.899 / 0.880 / 0.871 (the chord truth is built from the same clones as the lineage source).
* §17 sensitivity (`lineage_min_cells_per_end`, chord substitute, PCA) *[lineage]*: v3_default agreement 0.886 / 0.883 / 0.875 (k = 1), 0.899 / 0.880 / 0.871 (k = 2), 0.896 / 0.875 / 0.874 (k = 3); oriented fraction 0.13–0.15, 0.068, 0.029–0.035. Clones with cells of one fate vs both fates at their latest day (median over seeds, k = 2): 0.909 vs 0.878 (v3_default).
* Independence exact on every all-four call (raw_shared and smoothed_shared variants); J hash unchanged on every call.

## Step 5 — real data, report only (`real_transitions.csv`, `real_transitions_agreement.csv`)

S11's embeddings and velocities were rebuilt with S11's own functions (outputs redirected; LARRY OT ε 3.83588 / 6.86039 and the OT velocity reproduce S11 exactly; max |Δv| = 0). Palantir roots: LARRY `rng(1000)` day-2 cell (Undifferentiated) — default mode moved the start to a **day-6 Neutrophil**, so bins 0–1 are Neutrophil-dominated; scNT-seq `rng(1000)` HSC cell — start stayed an HSC.

* **LARRY** (`J_score`, `J_flow_ot`; fit seeds 0–2 × PCA/FA): λ₁ of `J_flow_ot` peaks in bin 7 (96 % Undifferentiated) for all three PCA fit seeds, and in bins 10–11 (Monocyte / Baso) for FA; `J_score` λ₁: PCA bins 8, 1, 7; FA 8, 7, 0. Critical-state index: bin 10 (Monocyte 49 %, Mast 19 %).
* **scNT-seq** (`J_score`, `J_flow` per labelling variant): λ₁ of `J_flow_raw_shared` and `J_flow_raw_perpulse` peak in bin 0 (HSC 53 %, MEP-like 42 %) for every embedding and fit seed; `J_flow_smoothed_shared` in bin 10 (Meg 53 %, Ery 26 %) for every embedding and seed; `J_flow_smoothed_perpulse` bin 10 for the three PCA fits and 10, 0, 0 for the FA fits; `J_score` λ₁ bins 7–9 (Mon-dominated). Critical-state index: bin 0 (HSC).
* Agreement of the detected λ₁ bin across 2 embeddings × 3 fit seeds (`real_transitions_agreement.csv`): one distinct bin for scNT `J_flow_raw_*` and `J_flow_smoothed_shared`; LARRY `J_flow_ot` 3 distinct bins (range 4), `J_score` 4 distinct (range 8).

## Labelling variants side by side (Step 0b)

| quantity | raw_shared | raw_perpulse | smoothed_shared (primary) | smoothed_perpulse |
|---|---|---|---|---|
| J_flow AUC, v3_default PCA | 0.840 / 0.533 / 0.747 | 0.846 / 0.544 / 0.749 | 0.732 / 0.796 / 0.887 | 0.735 / 0.821 / 0.881 |
| J_flow AUC, v3_transition PCA | 0.500 / 0.681 / 0.516 | 0.501 / 0.701 / 0.517 | 0.518 / 0.553 / 0.605 | 0.535 / 0.544 / 0.610 |
| `{labelling}` agreement, v3_default PCA | 0.957 / 0.978 / 0.978 | 0.956 / 0.977 / 0.976 | 0.644 / 0.634 / 0.668 | 0.649 / 0.632 / 0.668 |
| v3_transition λ₁ detected bin (true 1 / 5 / 0) | 3 / 5 / 0 | 3 / 5 / 0 | 1 / 6 / 0 | 0 / 11 / 0 |
| cycle null λ₁ peak/median | 1.16 / 1.09 / 1.06 | 1.17 / 1.10 / 1.04 | 1.26 / 1.95 / 1.66 | 1.47 / 2.08 / 1.56 |
| score–flow D̂ / truth (6 h) | 0.166 / 0.144 / 0.203 | 0.173 / 0.161 / 0.202 | 0.458 / 0.823 / 0.640 | 0.458 / 0.816 / 0.638 |
| scNT λ₁ detected bin (6 fits) | 0 (all) | 0 (all) | 10 (all) | 10 ×4, 0 ×2 |

## Compute

* CPU **≈ 38.5 h of 60**: 35.1 h measured with `/usr/bin/time -l` over 175 timed process trees (`cpu_ledger.csv`), plus ≈ 3.4 h estimated for processes killed together with their timer (4 contention kills, 1 kill to change the unit design, 1 Step 2 split, 1 scVelo restart, timing probes and tests). Of the measured total, 8.8 h went to truth units that were killed at the 60-min limit or superseded (4-chunk design).
* Breakdown (measured): diffusion truth 16.8 h (incl. superseded), scVelo 9.6 h, generation 2.2 h, Step 2 scoring 1.6 h, fits 1.2 h, S08 1.0 h, flow/OT 0.8 h, real data 0.6 h, Palantir 0.5 h, tests 0.14 h.
* Wall ≈ 8.4 h (09:20–17:45 UTC). Peak RSS 17.6 GB (v3_transition generation). Longest unit 59.2 min (generation s2, run concurrently with others).

