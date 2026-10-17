"""Build notebooks/06_behaviour_landscape.ipynb programmatically.

Run: python notebooks/make_tutorial_06.py && jupyter nbconvert --execute
     --to notebook --inplace notebooks/06_behaviour_landscape.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(r"""# 06 — Behaviour landscape, energy landscape and recurrent motifs

End-to-end tutorial for the landscape API (`scjdo2.landscape`) on the
equilibrated toggle-switch pitchfork, where the true dynamics are known.

What this notebook does:

1. generate the equilibrated pitchfork snapshot (`scjdo2.data.synthetic_pitchfork`)
2. fit the energy-parameterised score drift and read the per-cell latent operator `J = −β ∇²E`
3. run `scjdo2.landscape.analyse` — oracle tolerance, operator-space invariants, lead-eig basin classes,
   eigenvalue-sign fractions, well stiffness, energy, recurrent motifs
4. build a scrambled null (`scjdo2.opspace.flow_scrambled_null`)
5. render every figure with `scjdo2.landscape.figures`

Everything is computed from scratch here, so the numbers differ slightly from the
locked benchmark tables in `SUMMARY.md` (fewer epochs, smaller latent); the point
is the workflow, not the values.

**Caveat carried by every figure.** On a snapshot the operator is the Hessian of
the learned log density, so basins are where cells pile up and are held, forks are
density saddles; it is not the dynamics Jacobian except near stable states. The
pitchfork is used precisely because the analytic Jacobian is available to check that."""))

cells.append(nbf.v4.new_code_cell(r"""import warnings, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import Image, display

import scjdo2
from scjdo2 import opspace, landscape
from scjdo2.data.synthetic_pitchfork import generate_pitchfork_equilibrated, analytic_jacobian
from scjdo2.energy_drift import EnergyDriftAdapter, EnergyDriftConfig

warnings.filterwarnings("ignore", category=FutureWarning)
OUT = Path("../benchmarks/v2/tutorial_06")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 0
np.random.seed(SEED)
print(scjdo2.__version__ if hasattr(scjdo2, "__version__") else "scjdo2")"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 1. Data — equilibrated pitchfork

A mixture of stationary densities across an α-ramp through the bifurcation at
`τ_crit = 0.3`. Each cell carries its true 2-D state (`obsm["X_true"]`), the
true drift (`obsm["V_true"]`) and the analytic Jacobian, which we use below as
ground truth for the behaviour classes.

We add three labels the figures use: a `cluster` (upper branch / lower branch /
saddle region, from the true state), a `condition` (early vs late along the ramp,
standing in for young vs aged), and a `donor` (random thirds, the replicate unit
for the motif null)."""))

cells.append(nbf.v4.new_code_cell(r"""adata = generate_pitchfork_equilibrated(tau_crit=0.3, n_cells=1200, n_genes=200,
                                        n_latent=20, seed=SEED)
Xt = adata.obsm["X_true"]
pt = adata.obs["pseudotime"].to_numpy()

# ground truth from the analytic Jacobian
Jt = analytic_jacobian(Xt, adata.obs["alpha"].to_numpy())
lead_true = np.linalg.eigvals(Jt).real.max(axis=1)
adata.obs["lead_re_true"] = lead_true

# labels
diff = Xt[:, 0] - Xt[:, 1]
cluster = np.where(np.abs(diff) < 0.35, "saddle", np.where(diff > 0, "upper", "lower"))
adata.obs["cluster"] = pd.Categorical(cluster)
adata.obs["condition"] = pd.Categorical(np.where(pt < 0.5, "early", "late"))
rng = np.random.default_rng(SEED)
adata.obs["donor"] = pd.Categorical(rng.choice(["d1", "d2", "d3"], size=adata.n_obs))
adata.uns["scjdo2"] = {"meta": {"use_rep": "X_pca", "clock_calibrated": False}}
adata"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 2. Energy drift and the latent operator

`EnergyDriftAdapter` learns a scalar `E(z)` by exact (Hyvärinen) score matching;
the score is `−∇E`, the drift `β·score`, and the per-cell operator is
`J = −β ∇²E`, symmetric by construction. A few hundred epochs are enough for the
tutorial; the benchmark runs use 2000."""))

cells.append(nbf.v4.new_code_cell(r"""z = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
t0 = time.time()
drift = EnergyDriftAdapter(dim=z.shape[1], config=EnergyDriftConfig(
    n_epochs=2000, hidden=96, depth=3, seed=SEED, device="cpu")).fit(z)
J = drift.jacobian_autodiff(z)
adata.obsm["X_drift_v"] = drift.velocity(z)
adata.uns["scjdo2"]["J"] = J
sym = drift.symmetry_check(z)
print(f"fit {time.time()-t0:.0f}s; J shape {J.shape}; median asymmetry {sym['median_asym']:.1e}")"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 3. One-call analysis

`landscape.analyse` fixes the order: oracle tolerance → operator-space
invariants → lead-eig basin classes and `basin_index` → eigenvalue-sign
fractions → well stiffness → energy → motifs. Every step is also a public
function you can call on its own (`opspace.oracle_tol`,
`opspace.behaviour_fractions`, `opspace.well_stiffness`, `opspace.store_energy`,
`motifs.fit_motifs`, …)."""))

cells.append(nbf.v4.new_code_cell(r"""info = landscape.analyse(
    adata, cluster_key="cluster", resting_types=("upper", "lower"),
    drift=drift, motifs=True, replicate_key="donor", n_perm=100, seed=SEED, verbose=True,
)
info"""))

cells.append(nbf.v4.new_code_cell(r"""# what got written
cols = ["lead_re", "spectral_gap", "henrici", "op_behaviour_basin", "op_basin_index",
        "op_frac_neg", "op_frac_zero", "op_frac_pos", "op_stiffness", "energy", "op_motif"]
adata.obs[cols].head()"""))

cells.append(nbf.v4.new_markdown_cell(r"""### Check against the analytic truth

The learned `lead_re` should track the analytic leading eigenvalue, and the
saddle region should carry the fork calls. This is the "eigenvalue check" that
tells you whether the density Hessian agrees with the dynamics Jacobian on this
system — it does near the basins, and only partly at the saddle (the caveat in
the introduction)."""))

cells.append(nbf.v4.new_code_cell(r"""from scipy.stats import spearmanr
rho = spearmanr(adata.obs["lead_re"], adata.obs["lead_re_true"]).correlation
print(f"Spearman(lead_re learned, analytic) = {rho:.3f}")
pd.crosstab(adata.obs["cluster"], adata.obs["op_behaviour_basin"])"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 4. A scrambled null

The landscape figures overlay a null so that only density above it reads as
structure. On the pitchfork the measured flow is `V_true`, so `mode="flow"`
would permute it across cells and refit; here, to keep the notebook fast, we
use `mode="marginal"` (shuffle each latent dimension, refit the energy drift)
with two permutations and a short fit. For a paper figure use `n_perm ≥ 10`
and the full epoch count."""))

cells.append(nbf.v4.new_code_cell(r"""def fit_marginal(z_perm, seed):
    return EnergyDriftAdapter(dim=z_perm.shape[1], config=EnergyDriftConfig(
        n_epochs=600, hidden=96, depth=3, seed=seed, device="cpu")).fit(z_perm)

t0 = time.time()
null = opspace.flow_scrambled_null(z, None, fit_marginal, n_perm=2, mode="marginal", seed=SEED,
                                   fisher_weights=opspace.fisher_mass(adata))
print(f"null: {null['n_perm']} perms in {time.time()-t0:.0f}s; "
      f"null lead_re median {np.nanmedian(null['lead_re']):.3f} vs real {adata.obs['lead_re'].median():.3f}")"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 5. All figures in one call

`landscape.figures` renders whatever the stored state supports. For the energy
landscape we use the true 2-D state as the floor (`energy_rep="X_true"`); on real
data this would be `X_umap`, `X_diffmap` or `X_draw_graph_fa`."""))

cells.append(nbf.v4.new_code_cell(r"""paths = landscape.figures(
    adata, OUT, color_key="cluster", condition_key="condition", conditions=("early", "late"),
    null=null, y="spectral_gap", height="stiffness", energy_rep="X_true", verbose=True,
)
paths"""))

for name, blurb in [
    ("operator_space", "Operator-space overview (existing figure): embedding coloured by basin index, spectrum ladder, drivers."),
    ("ternary", "Panel A: each cell by the fraction of its eigenvalues below −tol / within ±tol / above +tol. Panel B: the same three fractions per cluster, early vs late.\n\n"
                "Note the pile-up at the holder corner: the pitchfork is a 2-D system embedded in 20 PCA dimensions, so 18 axes are padding, all weakly contracting, and at most 1/20 of a cell's eigenvalues can be positive. "
                "This is the PCA tangent-rank ceiling (SUMMARY §5.6b) made visible; on real data with an anisotropic embedding the cloud spreads. The late saddle cells still carry the largest fork share, as they should."),
    ("landscape", "Fisher-weighted density on the (lead_re, spectral gap) plane; dashed grey = scrambled null."),
    ("landscape_3d", "Same floor; bars = well stiffness per bin (depth along the holding directions), shade = occupancy. Tall-but-sparse and crowded-but-shallow regions are the point."),
    ("energy", "The Waddington view: E(z) over the true 2-D state. Valleys = where cells pile up."),
    ("motifs", "Motif × context occurrence matrix: rows are spectral motifs, columns are cluster contexts, filled where enriched above the shuffled-context null."),
]:
    cells.append(nbf.v4.new_markdown_cell(f"### `{name}`\n\n{blurb}"))
    cells.append(nbf.v4.new_code_cell(
        f'display(Image(filename=str(OUT / "{name}.png"), width=760)) if "{name}" in paths else print("{name}: skipped")'))

cells.append(nbf.v4.new_markdown_cell(r"""## 6. The same thing step by step

If you want one figure with different options, call the pieces directly."""))

cells.append(nbf.v4.new_code_cell(r"""from scjdo2.plot import figure_behaviour_landscape_3d, figure_behaviour_ternary

fig = figure_behaviour_landscape_3d(adata, y="spectral_gap", height="lambda_min", null=null)
plt.show()
fig = figure_behaviour_ternary(adata, color_key="cluster", condition_key="condition")
plt.show()"""))

cells.append(nbf.v4.new_code_cell(r"""adata.uns["scjdo2"]["motifs"]["summary"]"""))

cells.append(nbf.v4.new_code_cell(r"""occ = adata.uns["scjdo2"]["motifs"]["occurrence"]
occ[occ["enriched"]].sort_values(["motif", "context"])"""))

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
Path(__file__).with_name("06_behaviour_landscape.ipynb").write_text(nbf.writes(nb))
print("wrote 06_behaviour_landscape.ipynb")
