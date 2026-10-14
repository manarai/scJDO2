# K562 Perturb-seq — RESULT (final, post-LOO-null)

**Tag:** `v0.5-basin-lock` (unchanged).
**v0.6-perturb NOT applied** — both the generic-response control
and the LOO-with-null driver check fail. The paper's K562 arm is
written as measured: a small direction-specific effect that is
dominated by the generic KD response, and a driver claim that
does not survive a random-eigenvector null.

---

## 1. The direction result — as measured

In full gene space, operator vs DE-only vs random (1 661 KDs,
scVI(d=15) + ESM energy drift + NB rate model, control cells only):

| arm | n | op mean | DE mean | random mean | frac(op > DE) | Wilcoxon p |
|---|---|---|---|---|---|---|
| **full-space, all** | 1 661 | **+0.049** | +0.002 | ≈ 1 × 10⁻⁴ | 0.59 | 2.1 × 10⁻¹⁶ |
| full-space, linear | 831 | +0.028 | +0.001 |            | 0.56 | 3.8 × 10⁻⁴ |
| full-space, nonlinear | 830 | +0.070 | +0.003 |          | 0.63 | 3.8 × 10⁻¹⁶ |

### Generic-response subtraction

* `cos(δx_pred, mean δx_obs)` = +0.09
* `cos(δx_obs,  mean δx_obs)` = +0.45  (observations have a strong
  generic component)

After subtracting the mean `δx_obs` from both prediction and
observation:

| arm | n | op mean | DE mean | frac(op > DE) | Wilcoxon p |
|---|---|---|---|---|---|
| **residual, all** | 1 661 | **+0.014** | −0.007 | 0.55 | 4.4 × 10⁻⁵ |
| residual, linear | 831 | +0.000 | −0.012 | 0.52 | 4.9 × 10⁻² |
| residual, nonlinear | 830 | +0.027 | −0.001 | 0.57 | 8.8 × 10⁻⁵ |

**Generic-response control: FAILS** (pre-stated gate was
`frac(op > DE) ≥ 0.60 AND op_mean > 0.05` in residual space).

**The paper should quote +0.049 full-space and +0.014 residual** as
the operator's effect size, not the Wilcoxon p-value. The operator
beats DE-only mostly by aligning with the generic KD response more
faithfully than a pointer at just the perturbed gene; only +0.014
of the signal is KD-specific direction.

---

## 2. The driver claim — withdrawn

Per the user's instruction, the driver check is reported as a
LOO partial Spearman projected onto the lead eigvec of `J_mean`
at the control basin, with a random-eigvec null:

* Full-projection partial Spearman (sens, |δ_proj| | expression) = **+0.620**
* **LOO partial Spearman** (gene g's own coordinate zeroed in `δx_obs`
  before projecting) = **+0.620**
* **Random-eigvec null** partial Spearman distribution (20 draws,
  LOO-style per gene):
  * mean +0.606, std 0.027, **95th percentile +0.637**

**The LOO partial Spearman (+0.620) is BELOW the random-null 95pct
(+0.637).** The pre-stated gate required
`LOO ≥ 0.30 AND LOO > null_q95`; LOO holds the absolute threshold
but fails the null comparison. **The driver claim is withdrawn.**

### Why — mathematically

The sensitivity score is `sens[g] = |v_lead_g| · Σ|v_lead_h|`, so
`sens[g] ∝ |v_lead_g|`. The projected observed shift at gene g is
`obs_proj[g] = |scalar · v_lead_g|`, so `obs_proj[g] ∝ |v_lead_g|`
up to a scalar that varies per KD. The correlation
`spearman(sens, obs_proj)` therefore picks up the common factor
`|v_lead_g|` whether `v_lead` is the operator's eigenvector OR a
random unit vector in gene space. The random-null test the user
asked for exposes this construction artefact. The sensitivity
measure as currently defined cannot be a specific-direction claim.

A real driver claim needs either a different sensitivity formula
(one that is NOT just a gene-wise amplitude of some direction) or
a different experimental test (e.g., does the operator predict
which KDs produce the LARGEST shifts, not which gene-indices move
within each KD).

---

## 3. §7c (Dynamo scNT-seq, labelled rates) — STOP

Under the LabelledRateModel with per-cell `t_c` + closed-form γ init
+ separate `log_γ` learning rate + batch-conditioned α(z, batch):

* γ pooled median = **0.299 /h**, IQR [0.154, 0.566], range
  [0.002, 5.71]. Biologically realistic.
* `log_γ` grad norm non-zero throughout training (2.1 × 10⁻² → 3.8 × 10⁻⁴).
* **3h-only vs 5h-only per-gene γ Spearman: +0.081**
  (Pearson +0.046).

Adding the batch covariate did NOT make γ batch-invariant
(Spearman +0.08, essentially unchanged from the no-batch run's
+0.07). Per user directive — **labelled rates on this dataset are
batch-confounded; §7c stops here.** The batch-flip pattern in the
behaviour table (`benchmarks/v2/dynamo_s7c/behaviour_by_batch.csv`,
HSC/MEP/GMP/Mon more basin at 5h than 3h) is therefore not
interpretable against §7b until the batch confound is resolved at
source (e.g., with a dataset where every cell shares the same
pulse length, like the single-pulse scEU-seq organoid).

---

## 4. Summary — what we claim, what we don't

**Claim (direction, honest size):** The scJDO2 operator predicts
per-KD direction with mean cosine **+0.049** in full space and
**+0.014** after subtracting the generic KD response, both
significantly above DE-only on 1 661 KDs. The signal is modest and
dominated by generic-response alignment.

**Claim (driver, withdrawn):** The sensitivity-score ↔ projected-
observation partial Spearman (+0.62) does not exceed a random-
eigvec null (95pct +0.637). The lead-eigenvector of `J_mean` does
not out-rank random gene-space directions on this measure. The
paper should not state a driver recovery based on this score.

**Claim (basin, held at tag v0.5-basin-lock):** 92.6 % of resting
PBMC cells read as basins at d_latent = 15 under the lead-eig rule,
with the continuous basin_index as the primary readout. Unchanged.

**Figures a reader should look at first:**

1. `figures/scatter_cos_vs_fisher_residual.pdf` — the honest K562
   effect size after generic-response removal.
2. `../v0.5_lock/pbmc_operator_space.pdf` — the §5 basin result.
3. `../v0.5_lock/pitchfork_rule_auc.pdf` — the lead-eig rule's
   methodological justification.
