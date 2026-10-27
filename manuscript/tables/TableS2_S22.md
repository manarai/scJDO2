**Supplementary Table S2 | S22: score-J stiffness vs LARRY sister-cell fate coherence.** Spearman ρ between a clone's median score-J λ₁ on day *t* and the Shannon entropy H of its cell types on day *t* + 2, for 1,818 clone/time pairs (1,694 distinct clones; 67 pairs with tied modal types). Score operator refitted with seeds 0–2 (energy network, exact-Laplacian score matching, LARRY PCA-30, 4,000 cells per day). p: one-sided (ρ > 0) permutation p-value, 1,000 within-stratum permutations. Baselines: ρ of local density and of distance to the day-2 Undifferentiated centroid with H; partial ρ: λ₁ vs H given density. **Pre-registered** gates use seed 0 only (rows marked L1–L3); seeds 1–2, the tie-free pooled cohort and the per-day cohorts are **report only**. Source: `benchmarks/real_flow/s22/scores.csv`.

| seed | cohort | gate | n pairs | n tied | ρ(λ₁, H) | p (perm.) | ρ density | ρ centroid distance | partial ρ given density |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | pooled | **L1** (fail) | 1,818 | 67 | −0.003 | 1 | 0.078 | 0.058 | 0.011 |
| 0 | Undifferentiated | **L2, L3** (fail) | 1,401 | 0 | −0.101 | 1 | 0.076 | 0.221 | −0.078 |
| 0 | pooled, tied pairs removed | report only | 1,751 | 0 | 0.001 | 0.999 | 0.082 | 0.056 | 0.017 |
| 0 | day 2 → 4 | report only | 232 | 3 | −0.099 | 0.955 | 0.195 | 0.114 | 0.033 |
| 0 | day 4 → 6 | report only | 1,586 | 64 | 0.012 | 0.999 | 0.082 | 0.032 | 0.022 |
| 1 | pooled | report only | 1,818 | 67 | −0.018 | 0.999 | 0.078 | 0.058 | −0.009 |
| 1 | Undifferentiated | report only | 1,401 | 0 | −0.111 | 1 | 0.076 | 0.221 | −0.091 |
| 1 | pooled, tied pairs removed | report only | 1,751 | 0 | −0.009 | 1 | 0.082 | 0.056 | 0.001 |
| 1 | day 2 → 4 | report only | 232 | 3 | −0.166 | 0.995 | 0.195 | 0.114 | −0.048 |
| 1 | day 4 → 6 | report only | 1,586 | 64 | 0.005 | 1 | 0.082 | 0.032 | 0.008 |
| 2 | pooled | report only | 1,818 | 67 | −0.035 | 1 | 0.078 | 0.058 | −0.025 |
| 2 | Undifferentiated | report only | 1,401 | 0 | −0.130 | 1 | 0.076 | 0.221 | −0.111 |
| 2 | pooled, tied pairs removed | report only | 1,751 | 0 | −0.027 | 1 | 0.082 | 0.056 | −0.016 |
| 2 | day 2 → 4 | report only | 232 | 3 | −0.104 | 0.954 | 0.195 | 0.114 | 0.016 |
| 2 | day 4 → 6 | report only | 1,586 | 64 | −0.029 | 1 | 0.082 | 0.032 | −0.024 |

Gate outcomes (seed 0; definitions as fixed in the task, from `RESULT_S22.md`):

| gate | condition | outcome |
|---|---|---|
| L1 | pooled ρ > 0; p < 0.01 | fail |
| L2 | Undifferentiated ρ > 0; p < 0.01 | fail |
| L3 | abs(ρλ) > both abs(ρbaseline); partial ρ > 0 | fail |
