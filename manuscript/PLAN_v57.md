# scJDO preprint v2 (manuscript v57) — plan

Decisions (user, 2026-10-21): **new version of the same bioRxiv preprint**
(v56 → v57), written **after S14–S16**, **Markdown like v56**; figures from
reproducible scripts. Posting to bioRxiv is done by the authors.

## Framing (changes from v56)
* v56: one pseudotime-conditioned drift (DSM), temporal Jacobian tensor,
  semi-NMF archetypes; FA as default representation.
* v57: **two operators matched to the time information measured** —
  score J (snapshot → stability archetypes, valid at steady state) and flow J
  (time course / lineage / labelling / velocity → dynamics archetypes) — in a
  shared **operator space** with archetypes; their disagreement as a
  non-equilibrium measure; identifiability analysis (representation, clock,
  rotation, stationarity, diffusion, composition).
* **Correction to v56:** FA is not a safe default for flow operators — OT
  flow J needs a variance-weighted transport space (S12, S13). State plainly.

## Claim → evidence → figure (status)
| # | claim | evidence | status |
|---|---|---|---|
| 1 | framework and what each operator needs | design + S06/S10 | ready |
| 2 | calibrated synthetic benchmark matches real data | S08, S09 | ready (documented misses) |
| 3 | score J = stability archetypes at steady state; fails out of equilibrium | PBMC, pitchfork, S06, S10 | ready |
| 4 | flow J from time course detects forks (label-free) in a defined transport space | S10, S12–S14 | **pending S14** |
| 5 | dynamics archetypes recover true regimes | S15 | **pending** |
| 6 | vs dynamo / spliceJAC / scJDO v1 | S16 | **pending** |
| 7 | real data: LARRY, scNT-seq (and K562) | S11, S14–S15 | partial |
| 8 | limits: diffusion not identifiable; labelling γ dependence; Palantir default unreliable | S10, labelling diag, S10 guard | ready |
| 9 | orientation layer (signs, reliability check, guards) | S02–S06 | ready (supplement) |

## Figure set (draft)
1. Framework: time information → operator → operator space → archetypes.
2. Calibrated synthetic data vs real (S08/S09 panels).
3. Score J: stability landscape on steady-state data (PBMC; ternary,
   landscape, energy); failure on developmental data.
4. Flow J: fork detection across transport spaces (S12–S14), held-out test.
5. Dynamics archetypes vs true regimes (S15).
6. Head-to-head (S16).
7. Real data: LARRY / scNT-seq operator spaces and archetypes.
S. Identifiability panel; orientation layer; labelling γ diagnosis.

## Not to claim
Critical-transition detection (untested formally), diffusion estimation,
per-cell real-data reproducibility of flow J (not shown), uniqueness
without a systematic literature review.
