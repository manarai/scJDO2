"""Archetypes, genes and gene coupling along simulated routes.

Given walkers' cell indices over time (``walk.cell_walk``) on cells with a
fitted scJDO2 operator (Phase C + ``opspace.compute``):

1. routes: walkers grouped by the state they end in (other than the start
   state; at least ``min_route`` walkers); the label adds states at least
   half of them hold for ``min_hold`` frames on the way;
2. relative time per walker: 0 = start, 1 = arrival, the first frame after
   which it spends ≥ 80 % of the remaining frames in its final state;
3. archetype activity: at every frame, the share of the walker's cell and
   its 14 nearest cells (latent space) that are members of each archetype
   (``archetype_weights`` ≥ 0.5 × that archetype's maximum, as
   ``opspace.archetype_operators``), averaged in ``n_bins`` relative-time bins;
4. for each route's top archetypes, genes from ``drivers.influence``: cell-
   cycle genes dropped, the ``pool`` most influential measured genes ranked
   by polarity log2(out/in), ``n_genes`` at each end plus ``n_hubs`` hubs,
   with their expression along the route (log-normalised, min-max scaled);
5. gene coupling among those genes from M(t*) = Pᵀ(e^{Bt*} − I)P: each gene
   keeps its ``net_targets`` strongest targets in the set if ≥ ``min_rel`` of
   its largest effect; tagged when a reference network has the pair.
   **Coupling, not regulation**: on every dataset tested, operator edges
   were not TF → target regulation.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree

from .. import drivers

# Tirosh et al. 2016 S and G2/M genes (Seurat ``cc.genes``) plus CCNB1, ASPM,
# PLK1 and current names of MLF1IP, FAM64A, HN1; matched case-insensitively.
CC_S = """MCM5 PCNA TYMS FEN1 MCM2 MCM4 RRM1 UNG GINS2 MCM6 CDCA7 DTL PRIM1 UHRF1 MLF1IP
HELLS RFC2 RPA2 NASP RAD51AP1 GMNN WDR76 SLBP CCNE2 UBR7 POLD3 MSH2 ATAD2 RAD51 RRM2
CDC45 CDC6 EXO1 TIPIN DSCC1 BLM CASP8AP2 USP1 CLSPN POLA1 CHAF1B BRIP1 E2F8""".split()
CC_G2M = """HMGB2 CDK1 NUSAP1 UBE2C BIRC5 TPX2 TOP2A NDC80 CKS2 NUF2 CKS1B MKI67 TMPO CENPF
TACC3 FAM64A SMC4 CCNB2 CKAP2L CKAP2 AURKB BUB1 KIF11 ANP32E TUBB4B GTSE1 KIF20B HJURP
CDCA3 HN1 CDC20 TTK CDC25C KIF2C RANGAP1 NCAPD2 DLGAP5 CDCA2 CDCA8 ECT2 KIF23 HMMR
AURKA PSRC1 ANLN LBR CKAP5 CENPE CTCF NEK2 G2E3 GAS2L3 CBX5 CENPA""".split()
CELL_CYCLE = {g.upper() for g in CC_S + CC_G2M} | {"CCNB1", "ASPM", "PLK1", "CENPU", "PIMREG", "JPT1"}


def operator_tables(adata, reference=None, n_perm=2000, seed=0) -> dict:
    """Everything the routes need from a fitted run, in plain arrays and tables."""
    drivers.influence(adata)
    roles = drivers.roles_summary(adata)
    ref_summary = None
    if reference is not None:
        ref_summary = drivers.reference_check(adata, reference, n_perm=n_perm, seed=seed)["summary"]
    X = adata.layers["counts"] if "counts" in adata.layers else adata.X
    X = sp.csr_matrix(X).astype(float)
    lib = np.asarray(X.sum(1)).ravel()
    X = sp.diags(1e4 / np.maximum(lib, 1)) @ X
    X.data = np.log1p(X.data)
    arch = {}
    for k, v in drivers._arch(adata).items():
        B = np.asarray(v["block"], float)
        S, A = B + B.T, B - B.T
        arch[int(k)] = {"influence": v["influence"], "behaviour": v.get("behaviour"), "block": B,
                        "projector": np.asarray(v["projector"], float), "t": float(v["influence_t"]),
                        "asym": float(np.linalg.norm(A) / (np.linalg.norm(S) + 1e-12))}
    return {"W": np.asarray(adata.obsm["archetype_weights"], float), "arch": arch, "roles": roles,
            "ref_summary": ref_summary, "X": X.tocsc(), "var_names": list(adata.var_names)}


def _collapse(seq, min_hold):
    out, run_s, run_n = [], seq[0], 0
    for s in list(seq) + [None]:
        if s == run_s:
            run_n += 1
            continue
        if run_n >= min_hold and (not out or out[-1] != run_s):
            out.append(run_s)
        run_s, run_n = s, 1
    return out


def _coupling(info, genes, gidx, ref_pairs, net_targets, min_rel):
    from scipy.linalg import expm
    B, P = info["block"], info["projector"]
    D = expm(B * info["t"]) - np.eye(B.shape[0])
    ids = [gidx[g] for g in genes]
    Msub = P[:, ids].T @ (D @ P[:, ids])          # rows target, cols source
    col_max = np.abs(P.T @ (D @ P[:, ids]))
    for c, j in enumerate(ids):
        col_max[j, c] = 0.0
    col_max = col_max.max(0)
    rows = []
    for c, s in enumerate(genes):
        w = Msub[:, c].copy()
        w[c] = 0.0
        for r in np.argsort(-np.abs(w))[:net_targets]:
            if abs(w[r]) <= 0 or abs(w[r]) < min_rel * col_max[c]:
                continue
            rows.append({"source": s, "target": genes[r], "abs": float(abs(w[r])),
                         "sign": int(np.sign(w[r])), "trrust": (s, genes[r]) in ref_pairs})
    top = max((e["abs"] for e in rows), default=1.0)
    for e in rows:
        e["w"] = round(e["abs"] / top, 3)
        e["abs"] = float(f"{e['abs']:.4g}")
    return rows


def route_dynamics(cells, states, names, start_state, latent, ops, reference=None,
                   min_hold=8, min_route=8, n_bins=20, n_top_arch=3, n_genes=5, n_hubs=4,
                   net_targets=2, min_rel=0.05, pool=20, keep_cycle=False, k=15) -> dict:
    """Route panels for the viewer. ``cells``/``states``: frames × walkers (cell index, state index)."""
    F, N = states.shape
    W = (ops["W"] >= 0.5 * np.nanmax(ops["W"], axis=0)).astype(float)
    K = W.shape[1]
    X, gidx = ops["X"], {g: i for i, g in enumerate(ops["var_names"])}
    ref_pairs = set(zip(reference["source"], reference["target"])) if reference is not None else set()
    nbl = cKDTree(latent).query(latent, k=k)[1]
    cycle = set() if keep_cycle else CELL_CYCLE
    s0 = names.index(start_state)
    final = states[-1]

    def expr_curve(g, rows_by_bin):
        col = X[:, gidx[g]].toarray().ravel()
        curve = np.array([col[np.concatenate(r)].mean() if r else np.nan for r in rows_by_bin])
        lo, hi = np.nanmin(curve), np.nanmax(curve)
        return ((curve - lo) / (hi - lo) if hi > lo else curve * 0), curve

    routes = []
    for fate in [s for s, c in Counter(final.tolist()).most_common() if s != s0 and c >= min_route]:
        members = np.flatnonzero(final == fate).tolist()
        arrive = []
        for i in members:
            hit = (states[:, i] == fate).astype(float)
            tail = np.cumsum(hit[::-1])[::-1] / np.arange(F, 0, -1)
            arrive.append(int(np.flatnonzero((tail >= 0.8) & (hit > 0))[0]))
        via = Counter()
        for i, t1 in zip(members, arrive):
            via.update(set(_collapse(states[: t1 + 1, i].tolist(), min_hold)) - {fate, s0})
        path = [s0] + [s for s, c in via.most_common() if c >= 0.5 * len(members)] + [fate]
        bins = np.linspace(0, 1, n_bins + 1)
        curve = np.zeros((n_bins, K))
        cnt = np.zeros(n_bins)
        rows_by_bin = [[] for _ in range(n_bins)]
        for i, t1 in zip(members, arrive):
            fr = np.arange(t1 + 1)
            b = np.minimum((fr / max(t1, 1) * n_bins).astype(int), n_bins - 1)
            for f, bi in zip(fr, b):
                rows = nbl[cells[f, i]]
                curve[bi] += W[rows].mean(0)
                cnt[bi] += 1
                rows_by_bin[bi].append(rows)
        curve /= np.maximum(cnt, 1)[:, None]
        top = np.argsort(-curve.mean(0))[:n_top_arch]
        archs = []
        for ak in map(int, top):
            info = ops["arch"].get(ak)
            if info is None:
                continue
            tbl = info["influence"]
            pl = tbl[(tbl["role"] != "unmeasured") & ~tbl["gene"].astype(str).str.upper().isin(cycle)]
            pl = pl.assign(total=pl["out_influence"] + pl["in_influence"])
            pl = pl.sort_values("total", ascending=False).head(pool).sort_values("polarity", ascending=False)
            role_of, pol_of = dict(zip(pl["gene"], pl["role"])), dict(zip(pl["gene"], pl["polarity"]))
            drv = pl["gene"].head(n_genes).tolist()
            rsp = [g for g in pl["gene"].iloc[::-1].head(n_genes).tolist() if g not in drv]
            hubs = [g for g in pl.sort_values("total", ascending=False)["gene"]
                    if g not in drv and g not in rsp][:n_hubs]
            genes = []
            for side, gl in (("driver", drv), ("responder", rsp), ("hub", hubs)):
                for g in gl:
                    scaled, raw = expr_curve(g, rows_by_bin)
                    genes.append({"gene": g, "side": side, "role": role_of[g],
                                  "polarity": round(float(pol_of[g]), 3),
                                  "curve": np.round(scaled, 3).tolist(),
                                  "mean_log": round(float(np.nanmean(raw)), 3)})
            archs.append({"id": ak, "behaviour": info.get("behaviour"),
                          "edges": _coupling(info, [g["gene"] for g in genes], gidx, ref_pairs,
                                             net_targets, min_rel),
                          "n_measured": int((tbl["role"] != "unmeasured").sum()),
                          "polarity_range": [round(float(pl["polarity"].min()), 3),
                                             round(float(pl["polarity"].max()), 3)] if len(pl) else [0, 0],
                          "share": round(float(curve[:, ak].mean()), 3), "genes": genes})
        routes.append({"states": [names[s] for s in path], "n": len(members), "members": members,
                       "arrive_frame_median": int(np.median(arrive)),
                       "rel_bins": np.round((bins[:-1] + bins[1:]) / 2, 3).tolist(),
                       "arch_curve": np.round(curve, 4).tolist(),
                       "top": [int(x) for x in top], "archetypes": archs})
    roles = ops["roles"]
    ref = ops["ref_summary"]
    return {
        "n_archetypes": K,
        "archetypes": [{"id": int(r.archetype), "behaviour": r.behaviour, "n_cells": int(r.n_cells),
                        "n_driver": int(r.n_driver), "n_hub": int(r.n_hub), "n_responder": int(r.n_responder),
                        "top_drivers": r.top_drivers, "top_responders": r.top_responders,
                        "asym": round(ops["arch"][int(r.archetype)]["asym"], 3)} for r in roles.itertuples()],
        "routes": routes,
        "reference": [] if ref is None else json.loads(ref.to_json(orient="records")),
        "meta": {"min_hold": min_hold, "min_route": min_route, "k": k, "pool": pool,
                 "cycle_dropped": not keep_cycle},
    }
