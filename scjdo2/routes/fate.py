"""Held-out clone fate test: does the drift-guided walk predict clonal fate?

Needs lineage barcodes and time points. Clones are split in half
(``n_splits`` seeds). Per split, the drift is refitted on training clones
only when it comes from clone velocity (otherwise it never saw clones);
free settings (kernel σ, proximity temperature) are picked on training
clones' earliest-time cells and frozen for test clones.

Clonal fate = the most common fate state among a clone's later cells.
Prediction = absorption probability of the walk into each fate state
(cells in fate states absorb). Baselines on the same cells: the same walk
without direction (σ = ∞), closeness to each fate's centre, and a 30-NN
classifier trained on training clones' fates (a ceiling: it sees labels).
Scored by per-fate AUROC (fates with ≥ 10 test cells), macro AUROC with a
95 % clone-bootstrap interval, and top-fate accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

from .walk import absorption


def _macro(prob, truth, fates, min_n=10):
    out = {}
    for j, f in enumerate(fates):
        y = truth == f
        if y.sum() >= min_n and (~y).sum() >= min_n:
            out[f] = roc_auc_score(y, prob[:, j])
    return out


def _boot(prob, truth, clones, fates, rng, n=500):
    uc = np.unique(clones)
    by = {c: np.flatnonzero(clones == c) for c in uc}
    vals = []
    for _ in range(n):
        idx = np.concatenate([by[c] for c in rng.choice(uc, len(uc))])
        m = _macro(prob[idx], truth[idx], fates)
        if m:
            vals.append(np.mean(list(m.values())))
    return np.quantile(vals, [0.025, 0.975]) if vals else (np.nan, np.nan)


def clone_fate_test(Z, time, clone, state, fate_states, velocity_fn, refit_on_train=True,
                    n_splits=3, sigmas=(0.1, 0.25, 0.5), k=30, seed=0):
    """Returns (per-split table, mean-over-splits summary).

    ``velocity_fn(train_clones)`` returns the drift at every row of ``Z``;
    ``train_clones`` is None when ``refit_on_train`` is False.
    """
    state = np.asarray(state).astype(str)
    fate_states = list(fate_states)
    t0 = np.min(time)
    late = (clone >= 0) & (time > t0) & np.isin(state, fate_states)
    maj = pd.Series(state[late]).groupby(clone[late]).agg(lambda s: s.value_counts().idxmax())
    fate = pd.Series(clone).map(maj).to_numpy()
    cand = np.flatnonzero((time == t0) & pd.notna(fate))
    fates = sorted(set(fate[cand]))
    if len(cand) < 50 or len(fates) < 2:
        raise ValueError(f"too few earliest-time cells with a clonal fate ({len(cand)})")
    absorb = np.array([fates.index(s) if s in fates else -1 for s in state])
    cent = np.stack([np.median(Z[state == f], 0) for f in fates])
    rows = []
    V_shared = None if refit_on_train else velocity_fn(None)
    for sp_seed in range(n_splits):
        rng = np.random.default_rng(100 + sp_seed)
        allc = np.unique(clone[clone >= 0])
        train = set(rng.choice(allc, len(allc) // 2, replace=False))
        is_tr = np.isin(clone[cand], list(train))
        tr, te = cand[is_tr], cand[~is_tr]
        V = velocity_fn(train) if refit_on_train else V_shared
        res = {}
        best = None
        for sg in sigmas:
            Y = absorption(Z, V, absorb, k=k, sigma=sg)
            s_tr = np.mean(list(_macro(Y[tr], fate[tr], fates).values()))
            if best is None or s_tr > best[0]:
                best = (s_tr, sg, Y)
        res["walk_drift"] = (best[2][te], f"sigma={best[1]}", best[0])
        Yu = absorption(Z, V, absorb, k=k, sigma=np.inf)
        res["walk_undirected"] = (Yu[te], "sigma=inf", np.mean(list(_macro(Yu[tr], fate[tr], fates).values())))
        d_all = np.linalg.norm(Z[cand][:, None] - cent[None], axis=2)
        md = float(np.median(d_all))

        def prox(idx, tau):
            d = np.linalg.norm(Z[idx][:, None] - cent[None], axis=2)
            e = np.exp(-(d - d.min(1, keepdims=True)) / tau)
            return e / e.sum(1, keepdims=True)
        taus = [md * f for f in (0.05, 0.1, 0.25, 0.5, 1.0)]
        sc = [np.mean(list(_macro(prox(tr, t), fate[tr], fates).values())) for t in taus]
        j = int(np.argmax(sc))
        res["proximity"] = (prox(te, taus[j]), f"tau={taus[j] / md:.2f}×median", sc[j])
        clf = KNeighborsClassifier(n_neighbors=30).fit(Z[tr], fate[tr])
        P = np.zeros((len(te), len(fates)))
        for c_i, c in enumerate(clf.classes_):
            P[:, fates.index(c)] = clf.predict_proba(Z[te])[:, c_i]
        res["supervised_ceiling"] = (P, "k=30, sees fates", np.nan)
        brng = np.random.default_rng(7)
        for name, (prob, setting, s_tr) in res.items():
            per = _macro(prob, fate[te], fates)
            lo, hi = _boot(prob, fate[te], clone[te], fates, brng)
            row = {"split": sp_seed, "method": name, "setting": setting,
                   "test_macro_auroc": float(np.mean(list(per.values()))), "ci_lo": float(lo), "ci_hi": float(hi),
                   "top_fate_acc": float((np.array(fates)[prob.argmax(1)] == fate[te]).mean()),
                   "n_test": int(len(te))}
            row.update({f"auroc_{f}": float(x) for f, x in per.items()})
            rows.append(row)
    tbl = pd.DataFrame(rows)
    summary = (tbl.groupby("method")[["test_macro_auroc", "ci_lo", "ci_hi", "top_fate_acc"]].mean()
               .sort_values("test_macro_auroc", ascending=False))
    return tbl, summary
