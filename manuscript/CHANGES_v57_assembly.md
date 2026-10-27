# CHANGES — assembly of scJDO_manuscript_v57.md (2026-10-26)

Assembled from: `scJDO_manuscript_v57_draft.md` (new results), v56 manuscript (`.../scJDO-v56-submission-20260922-r1/final_manuscript/manuscript/scJDO_manuscript_v56.md`) and v56 `FIGURE_LEGENDS.md`, `LITERATURE_REVIEW_v57.md`, `FIGURE_LEGENDS_v57.md`, `NUMBERS_v57.md`, `coord/orient/FINDINGS.md`, and the S14/S14b/S14c reports and reviews and the S14b, S15, S18d and S22 task files in `coord/orient/` (read for method details and the S14c outcome). No other file was edited; nothing was committed; figures untouched.

Reference details for new citations (titles, volumes, pages) were taken from the Crossref and arXiv records of the DOIs/IDs verified in `LITERATURE_REVIEW_v57.md`; no reference outside v56 or the review was added.

## 1. Word counts (whitespace tokens, rendered file)

| Section | Words |
|---|---|
| Abstract | 298 |
| Changes from v56 | 312 |
| Introduction | 827 |
| Results 1 (incl. Box 1) | 595 |
| Results 2 | 220 |
| Results 3 | 758 |
| Results 4 | 209 |
| Results 5 | 401 |
| Results 6 | 474 |
| Results 7 | 298 |
| Results 8 (v1, carried over) | 910 |
| Discussion (all subsections, incl. Table 1) | 1,607 |
| **Main text, Abstract → Limitations** | **≈ 6,910** |
| Methods, new (scJDO2) | ≈ 1,556 |
| Methods, carried over from v56 | 1,640 |
| Data / Code availability, contributions, interests | 320 |
| Figure legends (main 2,450; supplementary 1,391; tables 75) | 3,942 |
| References (52) | 1,243 |

The main text is below the 8,000–10,000-word target (≈ 10,100 including Methods). I did not pad it; new numbers would have been needed to lengthen it.

## 2. TODO markers in the manuscript (30)

1. Results 1 — which tasks test the *clock* and *composition* identifiability components (draft: "Each is tested below"; no source tests them).
2. Results 2 — citation for dyngen (not in v56 list or review).
3. Results 5 — citation/source for the scNT-seq hematopoiesis dataset.
4. Results 5 — the labeling-γ agreement value 0.41 exists only in a review diagnostic (`coord/orient/reviews/labelling_diag_2026-10-19/`); commit as a result table (NUMBERS_v57 row 67: NOT-RECOMPUTABLE).
5. Results 6 — the draft's "[v56 → PBMC basin and landscape figures]" steady-state score-operator results: v56 contains no PBMC analysis; no numbers in the sources. Include with source files or delete.
6. Results 6 — Supplementary Table S4: import S18d result tables.
7. Results 6 — Supplementary Table S3: import S22 result tables.
8. Results 6 — Fig. 6d panel pending S22/S18d tables.
9. Results 7 — citation for Leiden.
10. Results 7 — S15 values come from the interim folder; confirm after the S15 merge.
11. Methods — citation for POT.
12. Methods (S18d) — citation for analytic Pearson residuals (Lause et al. 2021, named in the S18d task).
13. Methods (Pre-registration) — whether to disclose that tasks were executed and reviewed by AI coding agents (reports and reviews name Codex and Manus).
14. Methods (Statistics) — exact energy-distance definition and citation.
15. Methods (v1 tensor) — v56 cites K cross-validation curves as "Supplementary Fig. S2"; v56's actual Fig. S2 is multiome QC and no CV-curve figure is in the v56 package.
16. Data Availability — exact repository record for Replogle K562/RPE1 (draft: figshare).
17. Data Availability — accessions and citations for scNT-seq hematopoiesis/neurons, pancreas, dentate gyrus.
18. Data Availability — commit dyngen settings files (NUMBERS_v57: not committed).
19. Data Availability — v56 K562 source conflict (10x `84K_K562_5pv3_GEMX_CRISPR_aggregate` in v56 legend vs Replogle lncRNA sublibrary in v56 text).
20. Code Availability — scJDO2 GitHub URL.
21. Code Availability — release DOI (Zenodo).
22. Author Contributions — confirm (roles copied from v56).
23. Competing Interests — confirm.
24. Fig. 3 legend — panel 3e shows S14b only; update to include S14c.
25. Fig. 6 legend — panel d pending S22/S18d bundles.
26. Fig. 7 legend — regenerate after S15 merge.
27. Fig. 9 legend — erythroid top genes differ between v56 text (HBB, AHSP, CA1, HBD) and v56 legend (HBB, AHSP, CA1, FAM178B, APOC1).
28. Supplementary Fig. S9 legend — v56 legend (uncorrected run overweights Cdc20/Ccnb1) contradicts v56 Supplementary Note text (uncorrected top-9 are neurogenic).
29. Supplementary Table S2 — extend with v57 datasets.
30. Supplementary Tables S3/S4 — import S22/S18d tables (listed twice in the file: in Results 6 and in the table list).

## 3. Structure decisions

- Title, authors, affiliations: from the draft (authors as v56). The draft's DRAFT banner was replaced by a one-line version statement.
- Draft Results 8 "Comparison with existing Jacobian methods [PENDING S16]" was **omitted** (not in the figure plan). The Discussion states that a head-to-head on a common OT velocity "has been run but not scored, and is not reported here" (draft: "outputs exist (S16a), scoring pending").
- Draft Results 9 became Results 8 "Operator archetypes on real time courses (scJDO v1)".
- New **Box 1** ("What scJDO2 estimates, and what it does not") replaces v56 Box 1; every item is a Results statement.
- **Table 1** (decision guide) updated from v56.
- Supplementary Notes: Note 1 = labeling-γ diagnostic (TODO 4); Note 2 = v56 cell-cycle note (condensed into Methods); Note 3 = v56 "Constructed Obstruction" note (referenced, not reproduced). v56's "τ-collapse at N = 100 000" note is cited in one Discussion sentence without a note number.
- Supplementary tables renumbered: S1 = v56 Supp. Table 1; S2 = v56 Supp. Table 2 (to extend); S3 = S22 (TODO); S4 = S18d (TODO).
- Spelling: v56 is mostly American (hematopoiesis, characterize, analyzed, labeling) with some British forms. The manuscript uses American throughout; the draft and FIGURE_LEGENDS_v57 British forms (haematopoiesis, normalised, labelling, neighbour, colour, artefact, standardised, residualised) were converted. Kept: "Characterising" in a reference title, the path `labelling_diag_2026-10-19`.

## 4. Figure renumbering

| Source | v57 |
|---|---|
| v56 Fig. 1 (four-stage schematic) | dropped; replaced by new Fig. 1 |
| v56 Fig. 2 (synthetic benchmark composite) | Supplementary Fig. S5 |
| v56 Fig. 3 (hematopoiesis) | **Fig. 9** |
| v56 Fig. 4 (Dynamo/SpliceJAC concordance) | Supplementary Fig. S6 |
| v56 Fig. 5 TV = 0 (reprogramming) | **Fig. 8** |
| v56 Fig. 5 TV = 0.1 companion | Supplementary Fig. S8 |
| v56 Fig. 6 + Fig. 6 validation (K562 bridges) | Supplementary Fig. S7 (panels a–e, f–g) |
| v56 Supp. Figs S1–S5 (multiome) | Supplementary Fig. S9a–e |
| v56 Supp. Fig. S6 (bandwidth sweep, simplified MET) | Supplementary Fig. S10a |
| v56 "Fig. S-BW-MET" (exact-pipeline MET sweep, unnumbered) | Supplementary Fig. S10b |
| v56 Supp. Fig. S7 (saddle localization) | Supplementary Fig. S11 |
| v56 constructed-obstruction panel ("placement TBD") | Supplementary Fig. S12 (added beyond the stated plan) |
| v56 "Supplementary Fig. S2" (CV curves) | TODO 15 |
| v57 Figs 1–7, Supp. S1–S4 | as in FIGURE_LEGENDS_v57 |

## 5. v56 text cut or changed in meaning

**Front matter and Introduction**
- Title and Abstract replaced.
- Introduction rewritten. v56's first paragraph kept in condensed form. Removed: the "two advances" framing, the statement that scJDO does not compete with fate-mapping methods, and v56's claim that the archetype decomposition "is, to our knowledge, a lens not previously available". Archetypes are now credited to scJDO v1 [1].

**Results carried over from v56**
- The Results section "scJDO infers time-resolved operator representations" (Stages 1–4, SDE and semi-NMF equations) was removed from Results. Its content survives in the carried-over Methods.
- v56 Box 1 was removed (replaced). Its Kendall τ and concurrent-activation numbers moved to Results 8, Failure modes and the Fig. S5 legend.
- Reprogramming was condensed from about 900 to about 400 words. Cut:
  - the code and figure paths of the bandwidth sweep;
  - the TV comparison table (crossover 0.69, MET order preserved and Kendall τ 1.00 retained; the IPS late-peak 0.72 → 0.75 moved to the Fig. S8 legend);
  - the IPS-early and IPS_late gene sentences (moved to the Fig. 8 legend);
  - the sentence that the operator divergence "is present at earlier experimental time than the point of visible branch separation".
  - Added: these are DSM-drift, score-type, descriptive operators not validated by S14–S22.
- Hematopoiesis was condensed. Cut from the text: the Palantir-vs-DPT ρ range (kept in the Fig. 9 legend) and "signatures are correlational rather than causal" (replaced by "model-conditioned"). Added: the descriptive/score-type sentence.
- Concordance was condensed to one sentence plus the Fig. S6 legend. Cut:
  - the scMomentum exclusion;
  - the 32×/2.2× random-gene null (kept in the legend);
  - **v56's interpretation of the ρ = −0.818 inversion as a plausible instance of density dominance**.
- Synthetic benchmarks condensed. The FA statement was revised: FA remains acceptable for the DSM (score-type) drift, not for flow operators. Cut: the basin-of-attraction and init-at-ground-truth explanation (partly kept in the Fig. S5g legend).
- Constructed obstruction condensed to one sentence. Cut: the parameter sweeps, KE-matched numbers, `prior_sensitivity_report` knob names, and the citation "[Caylor & Terooatea, in revision]" (unverified, not in any reference list).
- Bifurcation-saddle limit condensed to one sentence plus the Fig. S11 legend. Cut:
  - the Balubaid MET-system corroboration sentence;
  - **v56's scoping argument that Re(λ_max) "surfaces genuine gene-level biology" along committed branches (Ery, MET)**. v57 makes no such claim.
- K562 bridges condensed. The finding (feasibility; PVT1 p = 0.143) is unchanged.

**Discussion**
- v56 Discussion replaced. Kept in condensed form: Weinreb identifiability and the τ-collapse at N = 100,000 (one sentence). Cut:
  - the paragraph on QC removing transitional cells;
  - the claim that "the operator machinery … recovers transient instability faithfully when given a sufficient ordering along a committed branch";
  - the closing claim that local sensitivity "is recoverable at the level of structure and direction from snapshot data";
  - the DMD/Koopman comparison.
- Table 1: rows for VeloVAE/neural ODE, DMD/Koopman and ArchR/Signac removed; rows for OT, flow operator, score operator and v1 archetypes added.
- Failure modes. Kept: saddle (now scoped to snapshot-fitted drifts), static perturbation screens, concurrent activations, cyclic manifolds (shortened) and cell–cell interactions. Removed: sparse pseudotime coverage (replaced by "coarse sampling"), unreliable ordering, and strong nonlinear coordinate distortions (subsumed by the representation failure mode).
- Limitations rewritten. Dropped v56's statement that "the dominant instability magnitude is robust across temporal-aggregation bandwidths", along with its Supplementary Fig. S2 reference.

**Methods, availability and legends**
- Methods:
  - "Latent representation selection" now applies to v1 DSM-drift analyses only; v56's "FA is therefore the default representation for scJDO" is restricted accordingly.
  - "Dimensionality reduction used FA … for all datasets" now reads "for v1 analyses".
  - "These are the values used in every figure of this paper" is now "every v1 analysis".
  - The bifurcation, SB, concordance and cell-cycle sections are condensed; numeric settings are kept.
  - The cell-cycle note's gene lists were dropped.
- Code Availability: v56 text on the canonical manuscript file and the `final_manuscript/` snapshot replaced. The v56 tag is retained.
- v56 figure-legend wording changed to match v56 text:
  - Fig. 4 → S6: "concordant with Dynamo" became "partial concordance".
  - Fig. 3 → 9: "consistent with a comparatively buffered operator trajectory" became "not evidence of a biologically buffered lineage". The Ery and DC peak τ values and the archetype peak orders were dropped; peak locations are marked "shown for completeness; no peak-timing claim".
  - Fig. 5 → 8: "bandwidth-robust" and archetype peak τ values dropped. "Not a decomposition artifact" softened in Fig. S8 to "not explained solely by the unregularized default".
  - Source-notebook paths removed from legends.

## 6. Changes to the draft's new text (beyond polishing)

- Results 3, transition detection: PENDING text replaced by the S14b/S14c outcome (from `reports/S14c.md`, `reviews/S14c.md`, `reports/S14b.md`, task S14b eligibility criteria).
- Results 3, held-out test: added disclosures from `reports/S14.md`. All four gates (G1–G4) failed and applied to the tuned configuration. The k rule was amended after development AUCs had been seen (Revision 1); test seeds were untouched.
- Results 3: "whitening only the OT cost destroys it" became "degraded it (PCA-10 0.53–0.79; PCA-20 0.28–0.43)", per Fig. 3c legend values.
- Results 2: the draft listed three calibration misses. The Fig. 2 legend lists five targets missed on all three seeds; all five are given.
- Results 4: the draft's "OT could not resolve rotation" at T = 3 was reworded as "the rotational index of cycles stayed at the level of linear trajectories" after checking the source CSV (see inconsistency 9). Added: early-cell fraction at T = 8 (0.00–0.06 on two datasets, 1.00 on s16).
- Results 5: added the tuned configuration's G3 failure (0.94), S11 "6 of 14 criteria", null maxima for the per-cell comparison, and the scNT-seq pairwise cosines (all from FIGURE_LEGENDS_v57).
- Results 6: added the S18/S18b details from the Fig. 6 legend and the S22 partial ρ −0.08 and |ρ| 0.10 (FINDINGS S22).
- Results 7: added the descriptor-view numbers from the Fig. 7 legend. Archetypes are credited to scJDO v1.
- Abstract revised to the review's framing, with the archetype and transition-test results added; 298 words.
- Intro: novelty statement uses the review's defensible wording verbatim in substance, plus an explicit list of non-claims.

## 7. References

**v56 → v57 numbering:** 1→44, 2→26, 3→16, 4→19, 5→17, 6→18, 7→48, 8→51, 9→52, 10→50, 11→45, 12→39, 13→41, 14→36, 15→10. All 15 v56 references are retained. v56 refs 5, 8, 9, 10, 11 and 12 were listed but never cited in v56's text; they are now cited.

**Removed:** none from the v56 list. The "Niemiec et al. 2025" DOI (10.1101/2025.02.05.636568) is not in the v56 reference list (only in a v56 notebook), is not cited, and is not needed. The v56 in-text citation "[Caylor & Terooatea, in revision]" was dropped.

**Added (37):**
- scJDO v1 (Redd 2026, 10.64898/2026.08.21.745996).
- Landscapes and potentials: Wang 2008; Wang 2011; Zhu 2024; scEpath; Topographer; MuTrans; PRESCIENT; ddHodge; Cislo 2025.
- Transitions: Barcenas 2024; Chen 2012 (DNB); Mojtahedi 2016; Freedman 2023; CellTran.
- OT and learned dynamics: moscot; OTVelo; TrajectoryNet (arXiv:2002.04461); TIGON; DiffusionOT; DeepRUOT (arXiv:2410.00844); SF2M (arXiv:2307.03672); scDiffEq.
- Score-based decomposition and theory: Chardès 2023 (arXiv:2310.02366); PFI (arXiv:2410.07501); Boffi 2024; Coomer 2022; Lavenant 2024; Fang 2019; Godrèche & Luck.
- Other: Cell-MNN (arXiv:2510.02903); Weinreb 2020 (LARRY); CellRank 2.
- Methods: Hyvärinen 2005; Vincent 2011; Ding 2010; Cutler 1994.

Notes on the added references:
- Hyvärinen 2005 has no DOI (JMLR). The review verified it via the JMLR landing page; pages 695–709 are added from the journal record.
- TIGON: the review says 2024; the Crossref published-print field gave 2023 for vol. 6(1). 2024 is used, consistent with vol. 6.
- Godrèche & Luck: the review says 2018 (online); the print record is 2019;52(3):035002, which is used.
- scJDO v1 author initials: the review gives "Redd DM, Green SG"; the Crossref record gives "Redd D, Green S"; the Crossref form is used.

**Needed but not added (unverified; TODO-ref):** dyngen, POT, Leiden, Lause et al. (Pearson residuals), the scNT-seq dataset, the energy-distance definition, and the pancreas and dentate gyrus datasets.

**Verified in the review but not cited** (not needed for the text): Shi 2018/2019/2022, SCENT, CytoTRACE 1/2, energy indicator 2023, Hopfield/HopLand, PI-SDE, STORIES, Su 2024, Sáez 2022, Rand 2021, Mochulska 2025, Fontaine 2025, Zhou 2012, Brackston 2018, Han 2025, scMomentum, RegVelo, GraphVelo, DynaVelo, scKINETICS, PHOENIX, DynNet, Sritharan 2021, DeepVelo, LineageOT, MIOFlow, JKOnet, UPFI, PFM, Taylor-King 2020, Zhang reviews, BioTIP, Barter, Wang 2026 sloppiness, Yu 2025, Battle 2016, Gnesotto 2018, Holehouse 2026, Dieball & Godec, Alain & Bengio, Yakovlev 2025, Kinger 2026.

## 8. Inconsistencies found between sources

1. FINDINGS S11 says median cosine 0.70; NUMBERS recomputes 0.69. 0.69 is used, as in the draft.
2. FINDINGS S11 says seed/estimator λ₁ 0.93–0.95; recomputed 0.92–0.95. 0.92–0.95 is used.
3. FINDINGS S10 says score-J AUC 0.17–0.31; recomputed 0.16–0.31. 0.16–0.31 is used.
4. FINDINGS S13 says whitened drift PCA-10 0.80–0.96; recomputed 0.79–0.96. 0.79–0.96 is used.
5. FINDINGS S15 says NMI flow 0.35–0.40; recomputed 0.35–0.39. 0.35–0.39 is used.
6. FINDINGS S20 says low-rank cosine 0.79; recomputed 0.79–0.80. 0.79–0.80 is used.
7. FINDINGS S14 reports LARRY lineage 0.55 (null 0.14) with held-out prediction "below baselines". This is the tuned configuration. The S20 correction and the S14 review addendum give reference 0.56 (0.15) and energy 0.687 < 0.867/0.882. Reference numbers are used.
8. FINDINGS S10 says FA inversion 0.04–0.25; S12 says FA-20 0.04–0.53 on a different dataset set. The S12 range is used.
9. The draft says that at three time points "OT could not resolve rotation", and FINDINGS S21 says "Rotation not detected (F2 fail)". The Fig. S2b legend reports cycle > linear on all drift seeds at T = 3 for 2 of 3 cycles (True, False, True), which reads as if rotation were detected. I checked `benchmarks/dyngen/s21/scores.csv`. At T = 3, the flow median_rotation is 0.03–0.04 for cycles and 0.03–0.05 for linear backbones, so the "True" flags are marginal differences at linear level. The text says "the rotational index of cycles stayed at the level of linear trajectories", without new numbers. Consider rewording the S2b legend.
10. The draft says the S21 "pre-branch λ₁ > 0" reading was withdrawn as a sampling artifact. The Fig. S2c legend shows s16 still at 1.00 at T = 8. Both are reported.
11. The draft's "score operator … only its eigenvalue signs are coordinate-invariant" is in tension with the tests that rank cells by score-J λ₁ (fork AUC; S10, S21b, S22). This is resolved in the text as "compared across representations only through inertia; within one fixed embedding, cells are also ranked by λ₁".
12. The draft says "Each [identifiability component] is tested below", but no source tests clock or composition (TODO 1).
13. The draft's Results 6 refers to "[v56 → PBMC basin and landscape figures]", but v56 has no PBMC figure (TODO 5).
14. The draft's Data Availability lists Replogle "figshare". The v56 K562 source is the 10x aggregate in the v56 legend and the Replogle lncRNA sublibrary in the v56 text (TODO 16, 19).
15. v56 internal inconsistencies:
    - Fig. 3 legend genes vs text (TODO 27).
    - Fig. 4 legend "concordant" vs text "partial concordance".
    - Fig. 3 legend "buffered" vs text.
    - Fig. 5 legend "not a decomposition artifact" vs text.
    - Multiome S5 legend vs Supplementary Note (TODO 28).
    - "Supplementary Fig. S2" cited for CV curves and support diagnostics vs the actual S2, multiome QC (TODO 15).
    - Bandwidth sweeps: the simplified pipeline is positive only at h = 0.01, while the exact pipeline is positive at 7/7. These are different pipelines and are labeled separately in S10a/b.
16. FIGURE_LEGENDS_v57, Fig. 3e, says "S14c pending". FINDINGS (2026-10-26) and `reports/S14c.md` give the construction failure. The legend text was updated; the figure panel needs updating (TODO 24).
17. S15 numbers come from the interim external folder `/Users/terooatt/Documents/scJDO2-manus/benchmarks/synthetic_multimodal/s15/` (NUMBERS_v57). FINDINGS still labels S15 interim. The user instruction treats them as final; this is flagged (TODO 10, 26).
18. The S14c review gives minimum bins of 42–46 cells for s8, s29 and s36. The S14c table shows s29 = 46 and s36 = 46; the S14b report does not give s8's minimum bin. The range 42–46 is quoted from the review.
19. Score-operator β: the draft does not state it. The S22 task gives β = 1.0 (S11 settings); S10's β was not checked. The Methods state "β = 1 in the S11/S22 fits".
20. NUMBERS_v57 marks dyngen 1.1.1 and "64 simulations" as not recomputable (settings files not committed). They are kept from the draft (TODO 18).
21. The draft's version banner said S15, S14c and S16 were pending. S14c is now resolved; S16 is omitted; S15 is still interim.
