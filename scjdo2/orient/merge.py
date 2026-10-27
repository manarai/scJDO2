"""§5/§12 Merge: within one orientation group, by component and trust
ranking, never blended — and never across groups.

Sign, per (cell, axis): the highest-ranked source of the group with support
wins; a lower-ranked supporting source of the same group that disagrees
sets ``conflict`` there and is recorded. Magnitude and omega belong to the
flow group: magnitude from labelling only; omega from labelling, else
velocity (clock unidentified).

A source the reliability check demoted (§15) is a cross-check only: it
decides no sign and gives no omega, but its disagreements with the deciding
source are still recorded as conflicts.
"""

from __future__ import annotations

import numpy as np

from .base import GROUPS, MEASURED_SOURCES


def reliability(group: str, results: dict, min_agree: dict, min_pairs: int) -> dict:
    """§15 Dataset-level check of each listed source against the
    higher-ranked sources of its group, on (cell, axis) pairs both sign."""
    out = {}
    ranked = [s for s in GROUPS[group] if s in results]
    for i, s in enumerate(ranked):
        if s not in min_agree:
            continue
        above = ranked[:i]
        if not above:
            out[s] = {"status": "not_checked", "reason": "no higher-ranked source", "demoted": False}
            continue
        r = results[s]
        both = np.zeros_like(r.support)
        ref = np.zeros(r.sign.shape, dtype=np.int8)
        for a in above:                      # the sign the higher sources decide
            take = results[a].support & ~both
            ref[take] = results[a].sign[take]
            both |= take
        both &= r.support
        n_pairs = int(both.sum())
        if n_pairs < min_pairs:
            out[s] = {"status": "not_checked", "reason": f"{n_pairs} shared pairs < {min_pairs}",
                      "n_pairs": n_pairs, "demoted": False}
            continue
        agree = float(np.mean(r.sign[both] == ref[both]))
        demoted = agree < min_agree[s]
        out[s] = {"status": "demoted" if demoted else "passed", "agreement": agree,
                  "n_pairs": n_pairs, "threshold": float(min_agree[s]),
                  "against": above, "demoted": bool(demoted)}
    return out


def merge_group(group: str, results: dict, n: int, k: int, d: int,
                demoted: frozenset = frozenset()) -> dict:
    order = [s for s in GROUPS[group] if s in results]
    code = {s: i + 1 for i, s in enumerate(GROUPS[group])}            # 0 = none
    sign = np.zeros((n, k), dtype=np.int8)
    source = np.zeros((n, k), dtype=np.int8)
    conflict = np.zeros((n, k), dtype=bool)
    conflict_sources = [set() for _ in range(n)]
    for s in order:
        r = results[s]
        take = r.support & (source == 0) & (s not in demoted)
        sign[take] = r.sign[take]
        source[take] = code[s]
        disagree = r.support & (source != 0) & (source != code[s]) & (r.sign != sign)
        conflict |= disagree
        for i in np.where(disagree.any(axis=1))[0]:
            conflict_sources[i].add(s)

    name = {v: s for s, v in code.items()}
    any_sup = source > 0
    first = np.argmax(any_sup, axis=1)
    sign_source = np.where(any_sup.any(axis=1),
                           [name.get(int(source[i, first[i]]), "none") for i in range(n)], "none")
    measured = np.isin(source, [code[s] for s in MEASURED_SOURCES[group]]).any(axis=1)
    imputed = any_sup.any(axis=1) & ~measured
    orientation_class = np.where(measured, "measured", np.where(imputed, "imputed", "none"))
    n_supporting = (np.sum([results[s].support.any(axis=1) for s in order if s not in demoted], axis=0)
                    if set(order) - set(demoted) else np.zeros(n, int))
    out = {
        "sources": order, "demoted": sorted(demoted), "sign": sign, "source": source, "conflict": conflict,
        "sign_source": np.asarray(sign_source, dtype=str),
        "orientation_class": orientation_class,
        "n_sources_supporting": np.asarray(n_supporting, int),
        "conflict_flag": conflict.any(axis=1),
        "conflict_sources": np.array([";".join(sorted(c)) for c in conflict_sources]),
        "source_codes": {str(v): s for s, v in code.items()} | {"0": "none"},
    }
    if group == "flow":
        magnitude = np.full((n, k), np.nan)
        if "labelling" in results and results["labelling"].magnitude is not None:
            magnitude = results["labelling"].magnitude.copy()
        omega = np.full((n, d, d), np.nan)
        omega_source = np.full(n, "none", dtype=object)
        for s in ("labelling", "velocity"):
            r = results.get(s)
            if r is None or r.omega is None or s in demoted:
                continue
            take = r.omega_support & (omega_source == "none")
            omega[take] = r.omega[take]
            omega_source[take] = s
        out.update(magnitude=magnitude, omega=omega, omega_source=omega_source.astype(str),
                   magnitude_source=np.where(np.isfinite(magnitude).any(axis=1), "labelling", "none"))
    return out


def cross_group(flow: dict, dev: dict) -> np.ndarray:
    """(n, k) True where both groups sign an axis and disagree. Reported,
    never resolved: the two groups answer different questions."""
    return (flow["source"] > 0) & (dev["source"] > 0) & (flow["sign"] != dev["sign"])
