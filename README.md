# scJDO v2

A lean successor to [scJDO](https://github.com/manarai/scJDO). Local Jacobians are
estimated from **latent-space neighbourhoods around every cell** (an operator field),
not from pseudotime slices. Operators are fit in the full gene space from **raw
counts under a count likelihood**, and every entry carries a **Fisher confidence**.

v1 remains frozen and is installed as a pinned dependency for side-by-side
comparison (`pip install -e .[compare]`). scJDO2 imports exactly two things from
v1 — the neural-ODE drift field (Step 2) and the semi-NMF with column-wise NNLS
(Step 8) — and nothing else. The Schrödinger-bridge, regulatory-inference, and
Koopman-backend workflows and the v1 CLI are out of scope.

## Pipeline

1. **Positional layer** (`scjdo2.embed`) — latent `z_c` via scVI (default), factor
   analysis, or PCA. Encoder frozen after fit so new cells can be projected in.
2. **Drift field** (`scjdo2.drift`) — v1 neural-ODE drift reused via adapter;
   gives `f(z)` and per-cell `v_c = f(z_c)`. Timescale relative unless a clock
   is supplied.
3. **Gene rate model** (`scjdo2.rates`) — scvi-tools module with fixed latent
   positions, smooth decoder (gene-specific heads over a shared trunk), negative
   binomial likelihood with gene-specific dispersion, size factors as offsets.
   Differentiable w.r.t. `z`.
4. **Rates along drift** (`scjdo2.rates`) — `dμ_g/dt = ∇_z μ_g(z_c) · f(z_c)`
   via autodiff of the decoder, never finite differences.
5. **Neighbourhood Jacobian field** (`scjdo2.field`) — per-cell `J_c` from a
   likelihood-weighted regression on `k` latent-space neighbours, with a
   smoothness penalty across neighbours. Sparse (columns restricted to genes
   that vary within the ball). `k` by held-out likelihood.
6. **Fisher ledger** (`scjdo2.fisher`) — diagonal Fisher per entry from the
   same NB log-likelihood at the fitted `J`; per-entry reason codes
   (`no_counts`, `no_variation`, `not_in_ball`).
7. **Operator space** (`scjdo2.opspace`) — invariant descriptors (eigenvalues,
   stable/unstable/near-zero/complex-pair counts, spectral gap, Henrici index,
   transient gain) and coordinate-dependent leading eigenvectors; Fisher-weighted.
8. **Archetypes** (`scjdo2.archetypes`) — v1 semi-NMF reused via adapter.
   Fisher-weighted. Number of archetypes by held-out likelihood. Auto-named
   from eigenvalue signatures (holder / fork / drift / damped oscillator /
   growing oscillator) with a non-normal "sensitive" flag.
9. **Grammar scaffold** (`scjdo2.grammar`) — archetype sequences along drift;
   pairwise commutator norms between consecutive archetype operators.

## Install

```bash
cd scJDO2
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev,compare]
```

The `[compare]` extra installs v1 from the preprint tag
`scjdo-v56-submission-20260922-r25` for comparison runs. v1 itself is not
imported inside the scjdo2 package; only `compare/run_v1.py` touches it.

## CLI

```bash
scjdo2 run     --input <h5ad> --embed scvi|fa|pca --k auto --out <dir>
scjdo2 compare --v1 <dir> --v2 <dir>
```

## Status

Phase A (audit + scaffold) in progress — see `V2_PLAN.md` and `CHANGELOG.md`.
All module files are typed stubs; no scientific code yet. Phase B/C acceptance
tests gate the implementation work.

## What this is not

- Not a port of v1's Schrödinger-bridge or regulatory-inference pipelines.
- Not a reimplementation of v1's Koopman backend. v2 reimplements only the
  Koopman non-normality descriptors (Henrici index, transient gain) directly
  as formulas in `opspace.py`.
- Not a solver for rotational non-identifiability or absolute timescale.
  Hooks exist for a per-cell clock scale and a known rotation, but when
  absent they are documented as "not calibrated".
