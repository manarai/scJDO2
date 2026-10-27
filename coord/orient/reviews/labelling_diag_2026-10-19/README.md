# Labelling-velocity discrepancy on scNT-seq haematopoiesis (Claude, 2026-10-19)

Why our labelling velocity and dynamo's (same new/total counts) disagreed in
S11 (embedding median cosine −0.05). `diag.py`, `diag2.py` (read-only).

* dynamo's `velocity_alpha_minus_gamma_s` = the same formula as ours,
  `α̂ − γ·total` with `α̂ = γ·new/(1−e^{−γt})` (`asspt_mRNA: ss`,
  one-shot), but on **kNN-smoothed moments** (`M_n`, `M_t`) and with **one γ
  per labelling time** (3 h, 5 h); ours: raw per-cell counts, **one γ per
  gene** shared across times (LabelledRateModel by design).
* γ: ours vs dynamo's 3 h Spearman 0.91 (ratio 0.93); 5 h 0.78 (dynamo's
  5 h γ ≈ 2.3× smaller). The 3 h and 5 h cells are different batches.

| velocity | gene sign agreement with dynamo | embedding median cos |
|---|---|---|
| ours, raw counts, our γ (S11) | 0.44 | 0.01 |
| ours, kNN-30 smoothed, our γ | 0.64 | −0.01 |
| ours, smoothed, dynamo's γ | 0.67 | 0.41 |
| dynamo moments + dynamo γ (reconstruction) | 0.78 | 0.56 |

Velocity is a small difference of large terms (|α̂ − γ·total| / |γ·total|
median 0.37), so its direction is set mainly by the γ assumption; smoothing
reduces noise but does not fix the direction. A labelling-duration
dependence of γ is biologically implausible — likely batch / model misfit —
so neither choice is clearly right.
