"""S14 addendum 3 (post hoc, App Note review): fit-to-fit variation of the headline results.

Refits the a-priori reference configuration with drift seeds 0, 1 and 2 (seed 0 is the reported fit) on
the 11 held-out synthetic datasets with a fork (k = 10, 15, 20) and on LARRY (PCA-15), and reports the
fork AUC of lambda_1, the per-cell lambda_1 Spearman between seeds, the LARRY lineage statistic and the
held-out day-6 energy distance. Exploratory, not pre-registered.

    python scripts/s14_seed_check.py
"""
from __future__ import annotations
import itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s14_choose_k as core, s14_optimise as opt, s14_test as st  # noqa: E402

SEEDS = (0, 1, 2)
DATASETS = st.TRANSITION_TEST + st.DEFAULT_TEST + st.REPORT_ONLY
OUT = core.ROOT / "benchmarks/synthetic_multimodal/s14"
import os
CACHE = Path(os.environ.get("SEED_CHECK_CACHE", str(OUT / "arrays/seed_check"))); CACHE.mkdir(parents=True, exist_ok=True)


def lam1(J):
    return np.linalg.eigvals(np.asarray(J, np.float64)).real.max(axis=1)


def main() -> None:
    core.set_threads(); config = st.config_for("reference"); rows = []; rho = []
    for ds in DATASETS:
        for k in (10, 15, 20):
            cache = CACHE / f"{ds}_k{k}.json"
            if cache.exists():
                c = json.loads(cache.read_text()); rows += c["rows"]; rho += c["rho"]; continue
            d = core.load_prepared(ds, k)
            flow = core.fit_flow(d, config, seeds=SEEDS)
            lams, new_rows, new_rho = [], [], []
            for s, J in zip(SEEDS, flow["Js"]):
                one = dict(flow, J=J)
                score, arr = opt.score_after_fit(ds, d, one)
                new_rows.append({"dataset": ds, "k": k, "seed": s, "auc_lam1": score.get("auc_lam1")})
                lams.append(lam1(J))
            for a, b in itertools.combinations(range(len(SEEDS)), 2):
                new_rho.append({"dataset": ds, "k": k, "pair": f"{SEEDS[a]}-{SEEDS[b]}",
                                "lambda1_spearman": float(spearmanr(lams[a], lams[b]).statistic)})
            cache.write_text(json.dumps({"rows": new_rows, "rho": new_rho})); rows += new_rows; rho += new_rho
            print(ds, k, [round(r["auc_lam1"], 3) for r in rows[-3:]], flush=True)
    syn = pd.DataFrame(rows); rh = pd.DataFrame(rho)
    syn.to_csv(OUT / "addendum_seeds_synthetic.csv", index=False); rh.to_csv(OUT / "addendum_seeds_lambda1.csv", index=False)
    # LARRY
    d = core.load_prepared("larry", 15)
    z = np.asarray(d["z"], np.float64); days = np.asarray(d["day"], int)
    prs, target, gaps = core._lineage_pairs(d); actual = target - z[prs]
    larry = []
    for s in SEEDS:
        cfg = dict(config, ensemble=1)
        flow = core.fit_flow(d, cfg, seeds=(s,))
        pred = np.empty_like(actual)
        for g in np.unique(gaps):
            k = np.flatnonzero(gaps == g); pred[k] = core.integrate_displacement(flow, z[prs[k]], float(g))
        lin = float(spearmanr(pred.ravel(), actual.ravel()).statistic)
        # held-out day with this seed (same construction as s14_test.larry_heldout)
        orig = core.fit_flow
        core.fit_flow = lambda dd, c, seeds=None, _s=s: orig(dd, c, seeds=(_s,))
        try:
            held, _ = st.larry_heldout(d, cfg)
        finally:
            core.fit_flow = orig
        e = {r["method"]: r["energy_distance"] for r in held}
        larry.append({"seed": s, "lineage_spearman": lin, "heldout_flow_energy": e["flow"],
                      "heldout_no_movement_energy": e["no_movement"], "heldout_mean_displacement_energy": e["mean_displacement"]})
        print("larry", larry[-1], flush=True)
    summary = {
        "note": "post hoc, exploratory; reference configuration; seed 0 = reported fit",
        "n_synthetic_datasets": len(DATASETS), "ks": [10, 15, 20], "seeds": list(SEEDS),
        "fork_auc_range_by_seed": {int(s): [float(g.auc_lam1.min()), float(g.auc_lam1.max())] for s, g in syn.groupby("seed")},
        "fork_auc_range_all": [float(syn.auc_lam1.min()), float(syn.auc_lam1.max())],
        "fork_auc_max_abs_change_vs_seed0": float(max(abs(syn[syn.seed == s].auc_lam1.values - syn[syn.seed == 0].auc_lam1.values).max() for s in SEEDS[1:])),
        "lambda1_spearman_between_seeds": {"min": float(rh.lambda1_spearman.min()), "median": float(rh.lambda1_spearman.median())},
        "larry": larry,
    }
    (OUT / "addendum_seeds.json").write_text(json.dumps(summary, indent=2)); print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
