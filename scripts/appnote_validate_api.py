"""Application Note check: the public ``scjdo2.flow`` API reproduces the S14 results.

For the three held-out v3_transition test seeds (PCA-15) this runs ``sj.flow`` on
observed inputs only (PCA coordinates and sampling hours), then scores the fork AUC
with the S14 scorer (truth used for scoring only), and compares it with the stored
S14 reference row. It also times ``sj.flow`` on the fixed 12,000-cell LARRY sample.

    python scripts/appnote_validate_api.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import s14_choose_k as core  # noqa: E402
import s14_optimise as opt  # noqa: E402

import anndata as ad  # noqa: E402
import scjdo2 as sj  # noqa: E402

OUT = core.ROOT / "benchmarks/appnote/api_validation.json"
SEEDS = ("v3_transition_s2", "v3_transition_s3", "v3_transition_s4")


def adata_from(d: dict) -> ad.AnnData:
    a = ad.AnnData(X=np.zeros((len(d["z"]), 1), np.float32))
    a.obsm["X_pca"] = np.asarray(d["z"], np.float32)
    a.obs["hours"] = np.asarray(d["hours"], float)
    return a


def main() -> None:
    core.set_threads()
    test = pd.read_csv(core.ROOT / "benchmarks/synthetic_multimodal/s14/test.csv")
    rows = []
    for ds in SEEDS:
        d = core.load_prepared(ds, 15)
        a = adata_from(d)
        t0 = time.time()
        sj.flow(a, time_key="hours", n_comps=15)
        secs = time.time() - t0
        J = a.uns["scjdo2"]["J"]
        score, arrays = opt.score_after_fit(ds, d, {"J": J, "config": {"space": "raw"}, "WI": None})
        ref = test[(test.dataset == ds) & (test.k == 15) & (test.configuration_id == "reference")].iloc[0]
        stored = np.load(core.ARR / "test" / f"{ds}_k15_reference.npz")["lambda1"]
        rows.append({"dataset": ds, "n_cells": int(a.n_obs), "auc_api": float(score["auc_lam1"]),
                     "auc_s14_reference": float(ref["auc_lam1"]),
                     "lambda1_spearman_api_vs_s14": float(spearmanr(a.obs["flow_lambda1"], stored).statistic),
                     "seconds": round(secs, 1)})
        print(rows[-1], flush=True)
    d = core.load_prepared("larry", 15)
    a = adata_from(d)
    t0 = time.time()
    sj.flow(a, time_key="hours", n_comps=15)
    larry = {"n_cells": int(a.n_obs), "seconds": round(time.time() - t0, 1),
             "ot_pairs": a.uns["flow"]["ot_pairs"][["from_time", "to_time", "solver", "n_iter"]].to_dict("records")}
    print(larry, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"synthetic": rows, "larry_12000": larry,
                               "note": "sj.flow defaults (PCA-15, eps 0.05, MLP 96x3, wd 1e-4, seed 0); "
                                       "fork AUC scored with the S14 scorer after fitting"}, indent=2))


if __name__ == "__main__":
    main()
