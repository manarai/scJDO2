"""Operator motifs along a time course: a step-by-step tutorial (about a minute on a laptop CPU).

Test data: the simulated pitchfork of ``flow_quickstart.py``. Five sampling times; cells
move along x and split into two arms in y after t = 2. So the true story is
"approach (t = 0, 1) -> branch point (t = 2) -> committed arms (t = 3, 4)", and the arm
a cell takes is fixed once it has split.

Every function of the motif layer is called once, with what it writes:

1. ``sj.flow``              operator per cell; now also keeps the OT plan and the drift
2. ``sj.flow_archetypes``   operator shapes and a label per cell
3. ``sj.flow_transitions``  label -> label along the plan, per time pair (main readout)
4. ``sj.flow_paths``        chains through the plan vs a Markov chain (robustness)
5. ``sj.flow_integrate``    drift paths with the operator profile (pictures)
6. figures                  archetype shapes, transition heatmaps, drift paths

    python examples/flow_motifs_tutorial.py            # figures in runs/flow_motifs_tutorial/
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

import scjdo2 as sj
from scjdo2.plot import figure_flow_archetypes, figure_flow_transitions

OUT = Path("runs/flow_motifs_tutorial")


def pitchfork(n_per_time: int = 400, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    z, t, arm_lab = [], [], []
    for k in range(5):
        x = k + 0.15 * rng.standard_normal(n_per_time)
        arm = rng.permutation(np.repeat([-1.0, 1.0], n_per_time // 2))   # equal arms: balanced transport
        y = arm * max(k - 2, 0) * 0.8 + 0.15 * rng.standard_normal(n_per_time)
        z.append(np.column_stack([x, y, 0.1 * rng.standard_normal((n_per_time, 3))]))
        t.append(np.full(n_per_time, float(k)))
        arm_lab.append(np.where(k < 2, "approach", np.where(k == 2, "branch",
                                                            np.where(arm > 0, "arm_up", "arm_down"))))
    a = ad.AnnData(X=np.zeros((5 * n_per_time, 1), np.float32))
    a.obsm["X_pca"] = np.vstack(z).astype(np.float32)   # stands in for a PCA of log-normalised counts
    a.obs["day"] = np.concatenate(t)
    a.obs["stage"] = pd.Categorical(np.concatenate(arm_lab),
                                    categories=["approach", "branch", "arm_up", "arm_down"])
    return a


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 120)
    adata = pitchfork()

    # 1. The flow operator. keep_plan (default 100) keeps each cell's most likely
    #    destinations at the next time in obsp["flow_plan"]; the drift weights go to
    #    uns["flow"]["drift_weights"].
    sj.flow(adata, time_key="day", n_comps=5, device="cpu")
    print("1. flow:", adata.uns["scjdo2"]["J"].shape, "operators;",
          "plan", adata.obsp["flow_plan"].shape, f"with {adata.obsp['flow_plan'].nnz:,} entries")
    print(adata.uns["flow"]["ot_pairs"][["from_time", "to_time", "plan_kept_mass_median"]].to_string(index=False))
    print(sj.flow_groups(adata, "stage")[["group", "median_lambda1", "ci_low", "ci_high"]].to_string(index=False))

    # 2. Archetypes: operator shapes (each J divided by its Frobenius norm), semi-NMF.
    patterns = sj.flow_archetypes(adata, n_archetypes=2)
    arch = adata.uns["flow"]["archetypes"]
    print("\n2. archetypes:", patterns.shape, "| leading Re(lambda) per archetype:",
          dict(zip(arch["names"], np.round(arch["eig_real"][:, 0], 3))))
    print(pd.crosstab(adata.obs["stage"], adata.obs["flow_archetype"]))

    # 3. Transitions, the main readout: P(to | from) per time pair against a
    #    source-label shuffle null. Exact: re-runs the transport and uses the full plan.
    tab = sj.flow_transitions(adata, labels="stage", n_perm=500)
    cols = ["from_time", "to_time", "from_label", "to_label", "prob", "expected_prob", "z", "p_enriched"]
    print("\n3. transitions by stage (prob > 0.05):")
    print(tab[tab["prob"] > 0.05][cols].round(3).to_string(index=False))
    figure_flow_transitions(adata, out=OUT / "transitions_stage")
    tab_a = sj.flow_transitions(adata, n_perm=500)                     # by archetype
    print("   transitions by archetype (p_enriched <= 0.01):")
    print(tab_a[tab_a["p_enriched"] <= 0.01][cols].round(3).to_string(index=False))
    figure_flow_transitions(adata, out=OUT / "transitions_archetype")
    figure_flow_archetypes(adata, out=OUT / "archetypes")

    # 4. Paths through the stored plan vs a Markov chain on the same plan.
    paths = sj.flow_paths(adata, labels="stage", n_paths=5000, collapse=True)
    print("\n4. paths (collapsed):")
    print(paths.head(6).round(3).to_string(index=False))

    # 5. Drift paths from ten day-0 cells, with lambda_1 along them.
    start = np.flatnonzero(adata.obs["day"].to_numpy() == 0)[:10]
    res = sj.flow_integrate(adata, start, n_steps=80)
    peak = res["lambda1"].max(axis=1)
    print("\n5. drift paths:", res["z"].shape, "| peak lambda_1 along each path:", np.round(peak, 2),
          "at t =", np.round(res["t"][res["lambda1"].argmax(axis=1)], 2))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3))
    z = np.asarray(adata.obsm["X_pca"])
    ax1.scatter(z[:, 0], z[:, 1], s=2, c="0.8", rasterized=True)
    for p in res["z"]:
        ax1.plot(p[:, 0], p[:, 1], lw=1)
    ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2"); ax1.set_title("drift paths from day-0 cells", fontsize=9)
    ax2.plot(res["t"], res["lambda1"].T, lw=0.8, color="#0072b2", alpha=0.6)
    ax2.axhline(0, color="0.5", lw=0.6)
    ax2.set_xlabel("time (days)"); ax2.set_ylabel(r"$\lambda_1$ along the path")
    ax2.set_title("operator profile (a path cannot branch)", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "drift_paths.png", dpi=150); plt.close(fig)

    print(f"\nfigures: {sorted(p.name for p in OUT.glob('*.png'))} in {OUT}/")


if __name__ == "__main__":
    main()
