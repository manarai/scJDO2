"""S14 addendum 2 (post hoc, App Note review): clone-aware uncertainty for the LARRY lineage statistic.

Cells of one clone share a target centroid. This reports a clone-cluster bootstrap 95% interval of the
cell-level pooled Spearman and the same statistic on clone means with clone-level permutations.
Exploratory, not pre-registered.

    python scripts/s14_clone_null.py
"""
import sys, json, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s14_choose_k as core, s14_test as st
core.set_threads()
d = core.load_prepared("larry", 15); flow = core.fit_flow(d, st.config_for("reference"))
z = np.asarray(d["z"], float); days = np.asarray(d["day"], int); hours = np.asarray(d["hours"], float)
rows, target, gaps = core._lineage_pairs(d)
# clone label per pair, same loop order as core._lineage_pairs
lab = []; u = np.unique(days)
if "clone" in d:
    clone = np.asarray(d["clone"], str)
    for d0, d1 in zip(u[:-1], u[1:]):
        later = set(clone[(days == d1) & (clone != "")])
        for i in np.flatnonzero((days == d0) & (clone != "")):
            if clone[i] in later: lab.append(f"{clone[i]}|{d0}")
else:
    ip, ix = np.asarray(d["clone_indptr"], int), np.asarray(d["clone_indices"], int)
    for d0, d1 in zip(u[:-1], u[1:]):
        later = {int(b) for i in np.flatnonzero(days == d1) for b in ix[ip[i]:ip[i+1]]}
        for i in np.flatnonzero(days == d0):
            for b in ix[ip[i]:ip[i+1]]:
                if int(b) in later: lab.append(f"{b}|{d0}")
lab = np.array(lab); assert len(lab) == len(rows)
actual = target - z[rows]; pred = np.empty_like(actual)
for g in np.unique(gaps):
    k = np.flatnonzero(gaps == g); pred[k] = core.integrate_displacement(flow, z[rows[k]], float(g))
def cen(x):
    o = x.copy()
    for g in np.unique(gaps): m = gaps == g; o[m] -= o[m].mean(0)
    return o
pooled = lambda a, b: float(spearmanr(a.ravel(), b.ravel()).statistic)
rng = np.random.default_rng(0)
units = np.unique(lab); idx = {c: np.flatnonzero(lab == c) for c in units}           # clone x interval
bio = np.array([c.split("|")[0] for c in lab]); clones = np.unique(bio)
bidx = {c: np.flatnonzero(bio == c) for c in clones}                                  # biological clone, both intervals
ugap = np.array([c.split("|")[1] for c in units])
out = {"n_pairs": len(rows), "n_clone_interval_units": len(units), "n_biological_clones": len(clones),
       "clones_in_both_intervals": int(sum(len(np.unique(lab[v])) > 1 for v in bidx.values())),
       "pairs_per_unit_median": float(np.median([len(v) for v in idx.values()])),
       "max_pairs_one_unit": int(max(len(v) for v in idx.values())),
       "protocol": "bootstrap resamples biological clones (all their pairs in both intervals); clone-interval means are permuted within sampling interval (unit label clone|start day); conditional on one fitted drift (seed 0)"}
def perm_within(P):
    Q = P.copy()
    for g in np.unique(ugap):
        m = np.flatnonzero(ugap == g); Q[m] = P[rng.permutation(m)]
    return Q
for name, (p, a) in {"pooled": (pred, actual), "centred": (cen(pred), cen(actual))}.items():
    obs = pooled(p, a)
    boot = []
    for _ in range(1000):
        pick = np.concatenate([bidx[c] for c in rng.choice(clones, len(clones))]); boot.append(pooled(p[pick], a[pick]))
    P = np.array([p[idx[c]].mean(0) for c in units]); A = np.array([a[idx[c]].mean(0) for c in units])
    cobs = pooled(P, A); cnull = np.array([pooled(perm_within(P), A) for _ in range(1000)])
    out[name] = {"cell_level_spearman": obs, "clone_bootstrap_ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                 "clone_level_spearman": cobs, "clone_perm_null_max": float(cnull.max()), "clone_perm_null_q99": float(np.quantile(cnull, .99)),
                 "clone_perm_p": float((1 + (cnull >= cobs).sum()) / 1001)}
(core.ROOT / "benchmarks/synthetic_multimodal/s14/addendum_clone_null.json").write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2))
