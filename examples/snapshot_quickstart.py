"""scJDO2 snapshot quickstart: the energy operator on one time point.

Test data: a 2-D double well sampled once (no time information). Two dense
clusters ("left", "right") are joined by a thin bridge. The energy operator
J = -Hess E describes the curvature of the cell density, so cells in the dense
clusters should get a negative leading eigenvalue ("holder") and bridge cells,
where the density is lowest, a positive one ("fork").

    python examples/snapshot_quickstart.py            # writes runs/snapshot_quickstart/

This reads the density, not the dynamics. Holder / fork labels are not
attractors or relaxation rates; see docs/math.md, section 3.
"""

from __future__ import annotations

import numpy as np
import anndata as ad

import scjdo2 as sj


def double_well(n: int = 3000, seed: int = 0) -> ad.AnnData:
    """Samples from p(z) ~ exp(-U(z)), U = (x^2 - 4)^2 / 4 + y^2 / 2, by rejection."""
    rng = np.random.default_rng(seed)
    out = []
    while sum(len(o) for o in out) < n:
        x = rng.uniform(-3.5, 3.5, 20 * n)
        keep = rng.random(x.size) < np.exp(-((x**2 - 4) ** 2) / 4)
        out.append(x[keep])
    x = np.concatenate(out)[:n]
    y = rng.standard_normal(n)
    z = np.column_stack([x, y]).astype(np.float32)
    a = ad.AnnData(X=np.zeros((n, 1), np.float32))
    a.obsm["X_pca"] = z                       # stands in for a 2-component PCA
    a.obsm["X_umap"] = z                      # 2-D basis for the energy figure
    a.obs["region"] = np.where(x < -1, "left", np.where(x > 1, "right", "bridge"))
    return a


if __name__ == "__main__":
    adata = double_well()
    summary = sj.run(adata, "runs/snapshot_quickstart", cluster_key="region",
                     use_rep="X_pca", fit_kwargs={"n_epochs": 1500})

    lead = -adata.obs["op_basin_index"].to_numpy()          # leading eigenvalue of J
    print(f"cells: {adata.n_obs}; J: {adata.uns['scjdo2']['J'].shape}")
    for g in ("left", "bridge", "right"):
        m = (adata.obs["region"] == g).to_numpy()
        frac = adata.obs.loc[m, "op_behaviour_basin"].value_counts(normalize=True)
        print(f"{g:>6}: median lead eigenvalue {np.median(lead[m]):+.3f}   "
              + "  ".join(f"{k} {frac.get(k, 0):.2f}" for k in ("holder", "fork", "drift")))
    print("figures:", ", ".join(summary["figures"]))
    print("saved to runs/snapshot_quickstart/ (reload with scjdo2.io.load)")
