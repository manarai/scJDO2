# Visual QA — v57 figures

Every PNG in `manuscript/figures/{main,supplementary}/` was opened and inspected after regeneration (2026-10-26).
Global changes applied to all new v57 figures through `notebooks/_style.py`: PDF (TrueType fonts) + PNG 600 dpi + TIFF 600 dpi (LZW);
double-column width 183 mm; text clamped to ≥ 6 pt and non-zero line widths to ≥ 0.5 pt at save time; bold panel letters placed
top-left outside each panel's tick labels and title and aligned per row; fixed operator colours (flow = blue `#0072B2`,
score = vermillion `#D55E00`, Leiden / baselines / nulls = grey `#7F7F7F`) checked identical in Fig2–Fig7 and FigS1–FigS4.

## New v57 figures (redrawn from committed result files)

| figure | issues found | fixed |
|---|---|---|
| Fig1 | schematic occupied 143 of 183 mm; text touching box edges (score operator, identifiability boxes) | axes fill the double-column width; box text fits with margin |
| Fig2 | figure legend floating far above panels (empty band); panel-b colour bar overlapping panel-a tick labels; panel-d legend covering ECDF curves; dyngen per-dataset dots touching the 0.51 label; large gap between b and d | legend placed directly above a; colour bar moved to the right of b (vertical); d legend moved to the empty lower right of the dispersion ECDF; label offset; spacer row reduced |
| Fig3 | large empty band at the top (INTERIM stamp); letter "a" overlapping its title; large gaps between rows; S13 legend covering points; S14b seed labels crowded; e (left) labels running into the eligibility panel; S14c results missing | stamp removed (S15 final); letters placed outside titles; explicit row spacing; S13 legend in a dedicated band above the data; e rebuilt with its own column widths, eligibility panel now s5–s40 (S14b + S14c) with eligible, fails-only-≥50-cells, memory-guard and other-failure markers, every seed ticked and every fifth labelled |
| Fig4 | letters "b" and "d" colliding with neighbouring panels/legend; panel-a legend running into panel b; panel-c legend covering the s16 bars; "0 in every region" text over a bar | wider spacing; two-line legend label; c legend moved over the empty linear-trajectory area; annotation moved |
| Fig5 | letter "b" on the a-panel title; cell-type tick labels running into the histogram; x labels "mean displ."/"no move" overlapping; histogram legend leaving an empty area under a; runtime/memory legends covering lines; agreement-panel legend running into panel d's y label; cell-type text labels overlapping in d (Monocyte, Mast, Baso; Ccr7_DC, Erythroid) | nested grids with a spacer column; rotated tick labels; legends moved below / into empty log-decades; shortened legend labels; deterministic label repulsion with leader lines (`repel_labels`) in d; y label cell count now formatted from the data |
| Fig6 | large empty band between rows; placeholder text overflowing its boxes; "chance 0.15" over a violin; after the S22/S18d import (panel d replaces the placeholders): letter "d" on the c-right title, d legends covering the centroid-distance points and the S18d per-target points, "RPE1 underpowered" and "K562 report only" labels running into each other, d titles touching | row spacing reduced; label moved below the chance line; bottom row on its own sub-grid with a spacer column before d; d legends moved below the axes; RPE1/K562 labels wrapped into three lines above the data (y-range extended to 1.36); shortened d titles |
| Fig7 | letter "b" on the a-panel title; panel-c legend overlapping panel b and data; b legend on Leiden markers; INTERIM stamp | two-line titles, wider spacing; redundant c legend removed (top legend covers both views); b y-range extended above the data for the legend; S15 final, stamp removed, gate pass/fail words computed |
| FigS1 | a legend hiding FA points at −6.6 to −7.7; "within ×2" over a point; b legend over the Σ_meas marker; c tick labels overlapping | legends moved to empty regions; label repositioned; c tick labels rotated |
| FigS2 | light T = 3 bar clipped under the legend; panel-b title running into letter "c" | y-range extended; titles wrapped; wider spacing |
| FigS3 | none (139.5 mm, all labels readable) | — |
| FigS4 | 209 mm wide (exceeded 183 mm); legend over data; no panel letters | fixed left margin, legend below in two columns (181.5 mm); letters a, b added and referenced in the legend |

## Carried-over v56 figures (vector PDF unchanged; PNG/TIFF via `pdftoppm`)

Not redrawn by instruction, so layout issues inside the v56 artwork remain. Checked: rendering complete, no clipping by the
rasteriser, page scaled to fit 183 × 240 mm. Estimated smallest text at that print size from `pdftotext -bbox` word heights
(5th percentile; values for FigS14 and FigS16, which use Type 3 fonts, are less reliable).

| figure (v56 file) | print size (mm) | est. smallest text (pt) | issues found (remaining, inherent to the v56 PDF) |
|---|---|---|---|
| Fig8 (main/Figure5_TV0.pdf) | 183.0 × 214.2 | 3.4 | internal title reads "Figure 4"; panel-c/c′ legends overlap the peak-order boxes and curves; b legend overlaps the MET curve label; small text below 6 pt |
| Fig9 (main/Figure3.pdf) | 183.0 × 218.1 | 3.5 | internal title reads "Figure 3 (FA)"; b legend overlaps the t = 0.72/0.98 annotations; empty Mono gene rows in e (by design); small text |
| FigS5 (main/Figure2.pdf) | 183.0 × 137.4 | 3.2 | internal title "Figure 2"; panel-b title overlaps the suptitle; d p-value text small |
| FigS6 (main/Figure4.pdf) | 183.0 × 65.2 | 1.3 | b x tick labels overlap each other; c "×32.5" overlaps the legend; cluster labels in a overlap |
| FigS7 (main/Figure6.pdf) | 183.0 × 221.4 | 3.5 | internal title "Figure 5"; gene-group labels in e drawn over bars |
| FigS8 (main/Figure6_validation.pdf) | 183.0 × 63.0 | 3.0 | internal title "Figure 5 — Validation"; panel-b title overlaps the suptitle |
| FigS9 (main/Figure5_TV01.pdf) | 183.0 × 214.2 | 3.4 | as Fig8 (internal title "Figure 4"; legend overlaps in c/c′) |
| FigS10 (supplementary/FigureS_multiome_FA_integration.pdf) | 183.0 × 123.8 | 3.2 | internal title "Figure 6"; factor labels in f overlap |
| FigS11 (supplementary/FigureS_multiome_integration_QC.pdf) | 91.6 × 240.0 | 1.7 | very tall page (aspect 2.6): at 240 mm height text is ~1.7 pt; per-cluster numbers in i overlap; internal title "Figure 6 — Integration QC" |
| FigS12 (supplementary/FigureS_multiome_marker_ordering_pre_vs_post_cc.pdf) | 183.0 × 59.9 | 4.6 | legends cover part of the curves |
| FigS13 (supplementary/FigureS_multiome_instab_ExcNeuron_cc_regressed.pdf) | 183.0 × 199.8 | 5.6 | gene-score panel empty and heat map uniform (no sensitive window); no panel letters |
| FigS14 (supplementary/FigureS_multiome_instab_ExcNeuron_uncorrected.pdf) | 183.0 × 196.4 | 8.9 | as FigS13; Type 3 fonts |
| FigS15 (supplementary/FigureS_bandwidth_sweep_DC_MET.pdf) | 183.0 × 47.7 | 2.9 | right third of the page empty; n_eff labels overlap; title "Supplementary Fig. Sxxx-BW" |
| FigS16 (supplementary/fig5_met_exact_pipeline_bandwidth_sweep.pdf) | 183.0 × 71.2 | 9.2 | panel titles overlap each other and the right title is clipped ("negative…"); no panel letters; Type 3 fonts |
| FigS17 (supplementary/FigureS_saddle_localization_label_blind.pdf) | 183.0 × 49.2 | 3.9 | none beyond small text |

## Carried-over title crop (2026-10-26)
Built-in v56 title lines carrying a v56 figure number were cropped (pypdf crop/media box; originals kept in `figures/carryover_v56_originals/`): Fig8, Fig9, FigS5, FigS7, FigS9, FigS10, FigS11, FigS15. Not cropped: FigS8 and FigS16 (title shares a line with panel titles). FigS5 header overlap is in the v56 artwork; regenerate for journal submission.

## v56 figures removed (2026-10-26)
The carried-over scJDO v1 (v56) figures (Fig8, Fig9, FigS5–FigS17), their originals in `figures/carryover_v56_originals/`, their legends and values, and `notebooks/carryover_v56.ipynb` were removed from v57 (the manuscript now points to version 56 for these results); the two sections above on carried-over figures are historical. v57 figures: Fig1–Fig7 and FigS1–FigS4.
