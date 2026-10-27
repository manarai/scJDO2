"""scJDO2 flow-operator quickstart (runs in about a minute on a laptop CPU).

Test data: a simulated pitchfork time course. Five sampling times; cells move along
x and split into two branches in y after t = 2. Branch-point cells (t = 2) should
receive a larger leading eigenvalue lambda_1 of the flow operator than committed
cells (t = 4).

    python examples/flow_quickstart.py
"""

from __future__ import annotations

import numpy as np
import anndata as ad
from sklearn.metrics import roc_auc_score

import scjdo2 as sj


def pitchfork(n_per_time: int = 400, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    z, t = [], []
    for k in range(5):
        x = k + 0.15 * rng.standard_normal(n_per_time)
        arm = rng.choice([-1.0, 1.0], n_per_time)
        y = arm * max(k - 2, 0) * 0.8 + 0.15 * rng.standard_normal(n_per_time)
        noise = 0.1 * rng.standard_normal((n_per_time, 3))
        z.append(np.column_stack([x, y, noise]))
        t.append(np.full(n_per_time, float(k)))
    a = ad.AnnData(X=np.zeros((5 * n_per_time, 1), np.float32))
    a.obsm["X_pca"] = np.vstack(z).astype(np.float32)   # stands in for a PCA of log-normalised counts
    a.obs["day"] = np.concatenate(t)
    return a


if __name__ == "__main__":
    adata = pitchfork()
    sj.flow(adata, time_key="day", n_comps=5, device="cpu")

    lam = adata.obs["flow_lambda1"].to_numpy()
    day = adata.obs["day"].to_numpy()
    branch, committed = day == 2, day == 4
    auc = roc_auc_score(np.r_[np.ones(branch.sum()), np.zeros(committed.sum())],
                        np.r_[lam[branch], lam[committed]])
    print(f"cells: {adata.n_obs}; J_flow: {adata.uns['scjdo2']['J'].shape}")
    print(f"median lambda_1  branch point (t=2): {np.median(lam[branch]):+.3f}   "
          f"committed (t=4): {np.median(lam[committed]):+.3f}")
    print(f"branch vs committed AUC of lambda_1: {auc:.2f}")
    print("descriptors:", adata.uns["flow"]["descriptor_names"])
    print(adata.uns["flow"]["ot_pairs"][["from_time", "to_time", "solver", "epsilon", "n_iter"]])
