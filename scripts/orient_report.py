"""Gated numbers for the orientation layer (docs/ORIENTATION_SPEC.md §8)
→ benchmarks/orient/report.json. Synthetic tests 1–6 only; Paul15 and
LARRY (tests 7–8) run separately on the user's Palantir/scVelo outputs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scjdo2.data.synthetic_oriented import make_oriented_pitchfork, make_rotating_two_well  # noqa: E402
from scjdo2.orient import orient  # noqa: E402

SEEDS = (0, 1, 2)


def _chord(a):
    z = a.obsm["X_lat"]; cl = a.obs["clone_true"].to_numpy(); day = a.obs["day"].to_numpy()
    out = np.full_like(z, np.nan)
    for c in np.unique(cl):
        i = np.where(cl == c)[0]; ds = np.unique(day[i])
        out[i] = z[i][day[i] == ds.max()].mean(axis=0) - z[i][day[i] == ds.min()].mean(axis=0)
    return out


def agreement(a, out, group):
    truth = a.obsm["velocity_true"] if group == "flow" else _chord(a)
    t = np.sign(np.einsum("nd,ndk->nk", np.nan_to_num(truth), out["axes"].U))
    m = out[group]; sup = m["source"] > 0
    return float(np.mean(m["sign"][sup] == t[sup])), float(np.mean(sup.any(axis=1)))


def main() -> int:
    rep = {"spec": "docs/ORIENTATION_SPEC.md", "seeds": list(SEEDS), "tests": {}}
    hashes = []
    t1 = {}
    for srcs, group, gate in ((["labelling"], "flow", 0.95), (["labelling", "velocity"], "flow", 0.95),
                              (["lineage"], "dev", 0.97), (["pseudotime"], "dev", 0.65)):
        vals = []
        for s in SEEDS:
            a = make_oriented_pitchfork(seed=s); out = orient(a, srcs, "X_lat")
            hashes.append(out["meta"]["hash_unchanged"]); vals.append(agreement(a, out, group))
        ag = [v[0] for v in vals]; cov = [v[1] for v in vals]
        t1[f"{group}:{'+'.join(srcs)}"] = {"gate": gate, "agreement": ag, "oriented_fraction": cov,
                                          "pass": bool(min(ag) >= gate and min(cov) >= 0.5)}
    rep["tests"]["1_pitchfork_sign_agreement"] = t1
    errs = []
    for s in SEEDS:
        a = make_rotating_two_well(seed=s); out = orient(a, ["labelling"], "X_lat"); hashes.append(out["meta"]["hash_unchanged"])
        t = a.uns["synthetic"]["omega_true"]
        errs.append(float(np.nanmedian(np.linalg.norm(out["flow"]["omega"] - t, axis=(1, 2)) / np.linalg.norm(t))))
    p = make_oriented_pitchfork(seed=0); op = orient(p, ["lineage", "pseudotime"], "X_lat"); hashes.append(op["meta"]["hash_unchanged"])
    rep["tests"]["2_omega"] = {"gate": 0.3, "median_rel_frobenius_error": errs, "pass": bool(max(errs) <= 0.3),
                               "no_rotation_from_dev_sources": bool(np.all(op["flow"]["omega_source"] == "none"))}
    t3 = []
    for s in SEEDS:
        a = make_oriented_pitchfork(seed=s); inj = np.random.default_rng(100 + s).random(a.n_obs) < 0.10
        a.obsm["velocity_lat"] = np.where(inj[:, None], -a.obsm["velocity_lat"], a.obsm["velocity_lat"])
        out = orient(a, ["labelling", "velocity"], "X_lat"); hashes.append(out["meta"]["hash_unchanged"])
        L, V = out["results"]["labelling"], out["results"]["velocity"]
        both = inj & (L.support & V.support).any(axis=1)
        t3.append({"labelling_wins_all": bool(np.all((out["flow"]["sign"] == L.sign)[L.support & inj[:, None]])),
                   "conflict_on_injected_with_both_supported": float(out["flow"]["conflict_flag"][both].mean()),
                   "fraction_injected_with_both_supported": float(both.sum() / inj.sum()),
                   "false_conflict_on_clean": float(out["flow"]["conflict_flag"][~inj].mean())})
    rep["tests"]["3_conflict"] = {"gate": 0.95, "per_seed": t3,
                                  "pass": all(x["labelling_wins_all"] and x["conflict_on_injected_with_both_supported"] >= 0.95 for x in t3)}
    t4 = []
    for s in SEEDS:
        a = make_oriented_pitchfork(seed=s, clone_coverage=0.4); out = orient(a, ["lineage", "pseudotime"], "X_lat")
        hashes.append(out["meta"]["hash_unchanged"]); cov = a.obs["clone_covered"].to_numpy(); cls = out["dev"]["orientation_class"]
        lin = out["results"]["lineage"].support.any(axis=1)
        t4.append({"covered_fraction": float(cov.mean()), "measured_equals_covered_with_lineage_sign": bool(np.array_equal(cls == "measured", cov & lin)),
                   "measured_on_uncovered": float(np.mean(cls[~cov] == "measured")),
                   "measured_on_covered": float(np.mean(cls[cov] == "measured")),
                   "imputed_on_uncovered": float(np.mean(cls[~cov] == "imputed")),
                   "oriented_fraction": out["meta"]["dev_coverage"]})
    rep["tests"]["4_coverage"] = {"per_seed": t4, "pass": all(x["measured_equals_covered_with_lineage_sign"] and x["measured_on_uncovered"] == 0 for x in t4)}
    g = make_rotating_two_well(seed=0, omega=0.0); og = orient(g, ["labelling"], "X_lat")
    r = make_rotating_two_well(seed=0); orr = orient(r, ["labelling"], "X_lat")
    n = make_oriented_pitchfork(seed=0); on = orient(n, ["lineage", "pseudotime"], "X_lat")
    hashes += [og["meta"]["hash_unchanged"], orr["meta"]["hash_unchanged"], on["meta"]["hash_unchanged"]]
    rep["tests"]["5_flags"] = {
        "gradient_equilibrium_fraction": float(np.mean(og["equilibrium_flag"] == "equilibrium")),
        "gradient_stationary_fraction": float(np.mean(og["stationarity_flag"] == "stationary")),
        "rotating_non_equilibrium_fraction": float(np.mean(orr["equilibrium_flag"] == "non_equilibrium")),
        "no_velocity_not_identified_fraction": float(np.mean(on["equilibrium_flag"] == "not_identified"))}
    f = rep["tests"]["5_flags"]
    f["pass"] = bool(f["gradient_equilibrium_fraction"] >= 0.9 and f["rotating_non_equilibrium_fraction"] >= 0.9
                     and f["no_velocity_not_identified_fraction"] == 1.0)
    rep["tests"]["6_hash_unchanged"] = {"runs": len(hashes), "all_unchanged": bool(all(hashes))}
    pyproj = (ROOT / "pyproject.toml").read_text()
    v1_imports = subprocess.run(["grep", "-rn", "from scjdo\\.", "scjdo2"], cwd=ROOT, capture_output=True, text=True).stdout.strip().splitlines()
    rep["v1"] = {"scjdo_listed_as_dependency": "scJDO" in pyproj.split("[project.optional-dependencies]")[0],
                 "v1_imports_remaining_in_library": v1_imports}
    out = ROOT / "benchmarks" / "orient" / "report.json"
    out.write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps({k: v.get("pass", v) for k, v in rep["tests"].items()}, indent=1))
    print("v1 imports remaining:", len(v1_imports))
    return 0


if __name__ == "__main__":
    sys.exit(main())
