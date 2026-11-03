"""One-call API for the driver / responder analyses and their figures.

Analyses (each a public function, usable alone):

* :func:`influence`       — finite-time influence + driver / hub /
  responder roles per archetype (:mod:`scjdo2.influence`).
* :func:`roles_summary`   — one row per archetype: role counts, top genes.
* :func:`edges`           — signed driver → target edges at ``t*``,
  optionally tagged against a reference network.
* :func:`load_reference`  — read a TF→target network (TRRUST TSV, a
  DataFrame, or v1 ``load_network``).
* :func:`reference_check` — do Jacobian edges recover the reference
  targets? AUROC vs a random-target null and vs co-expression.
* :func:`timing`          — expression-timing QC: do drivers change
  before responders along an existing pseudotime (:mod:`scjdo2.timing`)?
* :func:`branch_timing`   — the same edge test along each fate branch
  separately (fork archetypes act on the split).
* :func:`network_expression` — optional QC for one archetype: its
  network genes' expression across ordered cell states, and whether
  each driver switches on before its targets.

Figures:

* :func:`plot_network`    — driver/responder map, influence over time, network.
* :func:`plot_cascade`    — pseudotime cascade (timing QC).
* :func:`plot_reference`  — reference-recovery AUROC per archetype.
* :func:`plot_branch_timing` — archetype × branch source-first matrix + lags.
* :func:`plot_network_expression` — one archetype: network genes across states.

Wrappers: :func:`analyse` runs every analysis in order and returns the
tables; :func:`figures` renders every figure into one folder;
:func:`save_tables` writes the tables as CSVs.

Typical use::

    import scjdo2.drivers as D
    ref = D.load_reference("trrust_rawdata.mouse.tsv")
    res = D.analyse(adata, ref_network=ref, time_key="pseudotime")
    D.save_tables(res, "out/drivers")
    D.figures(adata, "out/drivers", ref_network=ref, results=res)

Needs a gene-space run with archetypes (``obsm['archetype_weights']``,
the Phase C Fisher ledger). :func:`influence` re-runs
:func:`scjdo2.opspace.archetype_operators` when the stored archetypes
lack the operator block (runs saved before this module existed).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import rankdata, spearmanr

from . import influence as _inf
from . import opspace as _opspace
from . import timing as _timing

KEY_GENES_DEFAULT = ("Gata1", "Sfpi1", "Spi1", "Klf1", "Cebpa", "Cebpe", "Gata2",
                     "Fli1", "Irf8", "Gfi1", "Zfpm1", "Mpo")


# ------------------------------------------------------------------ helpers


def _arch(adata: AnnData) -> dict:
    return adata.uns.get("scjdo2", {}).get("archetypes") or {}


def _ensure_operators(adata: AnnData) -> None:
    arch = _arch(adata)
    if not arch or any("block" not in v for v in arch.values()):
        _opspace.archetype_operators(adata)


def _ensure_influence(adata: AnnData) -> None:
    arch = _arch(adata)
    if not arch or any("influence" not in v for v in arch.values()):
        influence(adata)


def _ids(adata: AnnData, archetype_ids: Optional[Iterable[int]]) -> list[int]:
    return sorted(_arch(adata)) if archetype_ids is None else [int(a) for a in archetype_ids]


# ---------------------------------------------------------------- analyses


def influence(
    adata: AnnData,
    t: float | str = "auto",
    quantile: float = 0.9,
    n_sd: float = 1.0,
    min_coverage: float = 0.1,
    archetype_ids: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """Influence scores and roles for every archetype (long format).

    Wraps :func:`scjdo2.influence.archetype_influence` after making sure
    the archetype operators carry their block and projector.
    """
    _ensure_operators(adata)
    return _inf.archetype_influence(adata, t=t, quantile=quantile, n_sd=n_sd,
                                    min_coverage=min_coverage, archetype_ids=archetype_ids)


def roles_summary(adata: AnnData, n_top: int = 5) -> pd.DataFrame:
    """One row per archetype: behaviour, ``t*``, role counts and top genes."""
    _ensure_influence(adata)
    rows = []
    for ak, info in sorted(_arch(adata).items()):
        tbl = info["influence"]
        counts = tbl["role"].value_counts()
        top = lambda role, col, asc=False: ", ".join(
            tbl[tbl["role"] == role].sort_values(col, ascending=asc)["gene"].head(n_top))
        rows.append({
            "archetype": ak, "behaviour": info.get("behaviour"),
            "n_cells": info.get("n_cells"),
            "lead_re": float(np.real(np.asarray(info["eigenvalues"])[0])),
            "t_star": info["influence_t"], "t_rule": info["influence_t_rule"],
            **{f"n_{r}": int(counts.get(r, 0)) for r in _inf.ROLES},
            "top_drivers": top("driver", "polarity"),
            "top_hubs": top("hub", "out_influence"),
            "top_responders": top("responder", "polarity", asc=True),
        })
    return pd.DataFrame(rows)


def key_genes(adata: AnnData, genes: Sequence[str] = KEY_GENES_DEFAULT) -> pd.DataFrame:
    """Role and ranks of chosen genes in every archetype (long format)."""
    _ensure_influence(adata)
    keep = [g for g in genes if g in set(adata.var_names)]
    frames = [info["influence"][info["influence"]["gene"].isin(keep)].assign(archetype=ak)
              for ak, info in sorted(_arch(adata).items())]
    cols = ["archetype", "gene", "role", "out_rank", "in_rank", "out_influence",
            "in_influence", "polarity", "coverage", "reason"]
    return pd.concat(frames, ignore_index=True)[cols].sort_values(["gene", "archetype"])


def edges(
    adata: AnnData,
    archetype_ids: Optional[Iterable[int]] = None,
    ref_network: Optional[pd.DataFrame] = None,
    n_sources: int = 5,
    n_targets: int = 4,
    n_hubs: int = 2,
) -> pd.DataFrame:
    """Signed ``source → target`` edges at each archetype's ``t*``.

    Adds ``in_reference`` when ``ref_network`` is given.
    """
    _ensure_influence(adata)
    out = []
    for ak in _ids(adata, archetype_ids):
        e = _inf.influence_edges(adata, ak, n_sources=n_sources, n_targets=n_targets,
                                 n_hubs=n_hubs)
        out.append(_inf.annotate_reference(e, ref_network).assign(archetype=ak))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def load_reference(source: str | Path | pd.DataFrame = "builtin",
                   organism: str = "mouse") -> pd.DataFrame:
    """TF → target network with columns ``source, target, mode, weight``.

    ``source`` may be a DataFrame (``source``/``target`` columns), a path
    to a TRRUST-style TSV (``TF  target  mode  pmid``, no header), or a
    v1 network name (``builtin`` | ``trrust`` | ``collectri``; needs v1
    ``scjdo`` installed — note v1 silently falls back to ``builtin`` if a
    download fails, so prefer a local TSV for anything you report).
    """
    if isinstance(source, pd.DataFrame):
        ref = source.copy()
    elif Path(str(source)).exists():
        ref = pd.read_csv(source, sep="\t", header=None).iloc[:, :3]
        ref.columns = ["source", "target", "mode"]
    else:
        from scjdo.tl import load_network
        ref = load_network(organism=organism, source=str(source))
    if "mode" not in ref.columns:
        w = ref.get("weight", pd.Series(0.0, index=ref.index))
        ref["mode"] = np.where(w > 0, "Activation", np.where(w < 0, "Repression", "Unknown"))
    if "weight" not in ref.columns:
        ref["weight"] = ref["mode"].map({"Activation": 1.0, "Repression": -1.0}).fillna(0.0)
    return ref[["source", "target", "mode", "weight"]]


def _auroc(scores: np.ndarray, pos: np.ndarray) -> float:
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def reference_check(
    adata: AnnData,
    ref_network: pd.DataFrame,
    n_perm: int = 2000,
    seed: int = 0,
    archetype_ids: Optional[Iterable[int]] = None,
    membership_cutoff: float = 0.5,
) -> dict[str, pd.DataFrame]:
    """Do Jacobian influence edges recover reference TF → target pairs?

    For every reference TF in the gene set, other genes are ranked by
    ``|M[i, TF](t*)|`` and the TF's reference targets are scored by
    AUROC, per archetype. Compared with ``|Spearman|`` co-expression over
    the archetype's member cells, and with a null of random target sets
    of the same size (one-sided p on the mean AUROC across TFs). Also
    reports sign agreement on Activation / Repression edges next to the
    base rate of positive ``M`` entries.

    Returns ``{"summary", "per_tf", "edges"}`` DataFrames. The archetypes
    share the same reference edges, so their p-values are not independent.
    """
    _ensure_influence(adata)
    rng = np.random.default_rng(seed)
    genes = np.asarray(adata.var_names)
    gidx = {g: i for i, g in enumerate(genes)}
    G = genes.size
    ref = ref_network[ref_network["source"].isin(gidx) & ref_network["target"].isin(gidx)
                      & (ref_network["source"] != ref_network["target"])]
    ref = ref.drop_duplicates(["source", "target"])
    tfs = sorted(ref["source"].unique())
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    W = np.asarray(adata.obsm["archetype_weights"])
    arch = _arch(adata)

    per_tf, per_edge, summary = [], [], []
    for ak in _ids(adata, archetype_ids):
        info = arch[ak]
        B, P = np.asarray(info["block"]), np.asarray(info["projector"])
        M = P.T @ (_inf._change_operator(B, float(info["influence_t"])) @ P)
        base_pos = float(np.mean(M[~np.eye(G, dtype=bool)] > 0))
        wk = W[:, ak]
        Xm = X[wk >= membership_cutoff * wk.max()]
        roles = info["influence"].set_index("gene")["role"]
        auc_j, auc_c, sizes, cols_j, cols_c = [], [], [], [], []
        keep_all = np.ones(G, bool)
        for tf in tfs:
            j = gidx[tf]
            pos = np.zeros(G, bool)
            pos[[gidx[x] for x in ref.loc[ref["source"] == tf, "target"]]] = True
            keep = keep_all.copy()
            keep[j] = False
            s_j = np.abs(M[:, j])
            if Xm[:, j].std() > 0:
                with np.errstate(invalid="ignore", divide="ignore"):
                    rho = np.asarray(spearmanr(Xm[:, j], Xm).statistic)[0, 1:]
            else:
                rho = np.zeros(G)
            s_c = np.abs(np.nan_to_num(rho))
            a_j, a_c = _auroc(s_j[keep], pos[keep]), _auroc(s_c[keep], pos[keep])
            auc_j.append(a_j)
            auc_c.append(a_c)
            sizes.append(int(pos.sum()))
            cols_j.append(s_j[keep])
            cols_c.append(s_c[keep])
            per_tf.append({"archetype": ak, "tf": tf, "n_targets": int(pos.sum()),
                           "tf_role": roles.get(tf, "?"), "auroc_jacobian": a_j,
                           "auroc_coexpr": a_c})
            for x, mode in ref.loc[ref["source"] == tf, ["target", "mode"]].itertuples(index=False):
                v = float(M[gidx[x], j])
                agree = (float(np.sign(v) == (1 if mode == "Activation" else -1))
                         if mode in ("Activation", "Repression") else np.nan)
                per_edge.append({"archetype": ak, "tf": tf, "target": x, "mode": mode, "M": v,
                                 "rank_jacobian": int((s_j[keep] > abs(v)).sum()) + 1,
                                 "sign_agrees": agree})
        null_j, null_c = np.empty(int(n_perm)), np.empty(int(n_perm))
        for b in range(int(n_perm)):
            vj, vc = [], []
            for sj, sc, n1 in zip(cols_j, cols_c, sizes):
                pos = np.zeros(G - 1, bool)
                pos[rng.choice(G - 1, n1, replace=False)] = True
                vj.append(_auroc(sj, pos))
                vc.append(_auroc(sc, pos))
            null_j[b], null_c[b] = np.nanmean(vj), np.nanmean(vc)
        mj, mc = float(np.nanmean(auc_j)), float(np.nanmean(auc_c))
        signed = [e["sign_agrees"] for e in per_edge if e["archetype"] == ak
                  and np.isfinite(e["sign_agrees"])]
        summary.append({
            "archetype": ak, "behaviour": info.get("behaviour"), "n_tfs": len(tfs),
            "n_edges": int(sum(sizes)),
            "mean_auroc_jacobian": mj,
            "p_jacobian": float((1 + (null_j >= mj).sum()) / (1 + n_perm)),
            "null_mean_jacobian": float(np.mean(null_j)),
            "null_q95_jacobian": float(np.quantile(null_j, 0.95)),
            "mean_auroc_coexpr": mc,
            "p_coexpr": float((1 + (null_c >= mc).sum()) / (1 + n_perm)),
            "sign_agreement": float(np.mean(signed)) if signed else np.nan,
            "base_rate_M_pos": base_pos,
        })
    return {"summary": pd.DataFrame(summary), "per_tf": pd.DataFrame(per_tf),
            "edges": pd.DataFrame(per_edge)}


def timing(
    adata: AnnData,
    time_key: str = "pseudotime",
    archetype_ids: Optional[Iterable[int]] = None,
    layer: Optional[str] = None,
    n_bins: int = 25,
    n_perm: int = 2000,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Expression-timing QC for every archetype.

    Returns ``{"summary"`` (one row per archetype: role test + edge
    test), ``"onsets"`` (gene × archetype: role, onset, direction),
    ``"edges"`` (edges with ``lag``)``}``. See :mod:`scjdo2.timing`.
    """
    _ensure_influence(adata)
    summ, ons, eds = [], [], []
    for ak in _ids(adata, archetype_ids):
        r = _timing.expression_timing(adata, ak, time_key=time_key, layer=layer,
                                      n_bins=n_bins, n_perm=n_perm, seed=seed)
        summ.append({"archetype": ak, "behaviour": _arch(adata)[ak].get("behaviour"),
                     "n_cells": r["n_cells"], **r["role_test"],
                     **{f"edge_{k}": v for k, v in r["edge_test"].items()}})
        ons.append(r["genes"].assign(archetype=ak))
        eds.append(r["edges"].assign(archetype=ak))
    return {"summary": pd.DataFrame(summ),
            "onsets": pd.concat(ons, ignore_index=True) if ons else pd.DataFrame(),
            "edges": pd.concat(eds, ignore_index=True) if eds else pd.DataFrame()}


def branch_timing(
    adata: AnnData,
    branches: dict[str, Sequence[str]],
    cluster_key: str,
    root: Sequence[str] = (),
    time_key: str = "pseudotime",
    cells: str = "path",
    archetype_ids: Optional[Iterable[int]] = None,
    n_bins: int = 20,
    n_perm: int = 2000,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Branch-aware edge timing for every archetype.

    ``branches`` maps a fate name to its clusters in ``obs[cluster_key]``;
    ``root`` clusters are prepended to every branch path. See
    :func:`scjdo2.timing.branch_timing` (``cells="path"`` uses all cells
    on the branch, ``"members"`` only the archetype's members).

    Returns ``{"summary"`` (archetype × branch rows), ``"pooled"`` (one
    row per archetype, edge × branch pairs pooled), ``"lags"``,
    ``"onsets"``, ``"overall"`` (one row: all archetypes pooled)``}``.
    """
    _ensure_influence(adata)
    summ, pooled, lags, ons = [], [], [], []
    for ak in _ids(adata, archetype_ids):
        r = _timing.branch_timing(adata, ak, branches, cluster_key, root=root,
                                  time_key=time_key, cells=cells, n_bins=n_bins,
                                  n_perm=n_perm, seed=seed + ak)
        beh = _arch(adata)[ak].get("behaviour")
        summ.append(r["per_branch"].assign(archetype=ak, behaviour=beh))
        pooled.append({"archetype": ak, "behaviour": beh, **r["pooled"]})
        if len(r["lags"]):
            lags.append(r["lags"].assign(archetype=ak))
        if len(r["onsets"]):
            ons.append(r["onsets"].assign(archetype=ak))
    summary = pd.concat(summ, ignore_index=True) if summ else pd.DataFrame()
    pooled_df = pd.DataFrame(pooled)
    # Overall: pair-weighted fractions. Archetypes share genes and cells,
    # so no pooled p-value; inference stays with the per-archetype and
    # per-branch permutation p-values.
    ok = summary.dropna(subset=["frac_source_first"]) if len(summary) else summary
    n = int(ok["n_edges"].sum()) if len(ok) else 0
    overall = pd.DataFrame([{
        "n_pairs": n,
        "frac_source_first": float((ok["frac_source_first"] * ok["n_edges"]).sum() / n) if n else np.nan,
        "null_frac": float((ok["null_frac"] * ok["n_edges"]).sum() / n) if n else np.nan,
        "n_archetype_branch_p05": int((ok["p_perm"] < 0.05).sum()) if len(ok) else 0,
        "n_archetype_branch_tested": int(len(ok)),
        "cells": cells,
    }])
    return {"summary": summary, "pooled": pooled_df,
            "lags": pd.concat(lags, ignore_index=True) if lags else pd.DataFrame(),
            "onsets": pd.concat(ons, ignore_index=True) if ons else pd.DataFrame(),
            "overall": overall}


def network_expression(
    adata: AnnData,
    archetype_id: int,
    cluster_key: str,
    states: Optional[Sequence[str]] = None,
    time_key: str = "pseudotime",
    n_sources: int = 5,
    n_targets: int = 4,
    n_hubs: int = 2,
) -> dict:
    """Optional QC: does each driver switch on before its targets?

    One archetype; genes = its influence-network sources and targets;
    columns = cell states (``obs[cluster_key]``) in the order given by
    ``states``, else clusters holding ≥ 5% of the archetype's members
    ordered by median ``time_key``. See
    :func:`scjdo2.timing.network_state_expression` for the returned dict.
    """
    _ensure_influence(adata)
    e = _inf.influence_edges(adata, archetype_id, n_sources=n_sources,
                             n_targets=n_targets, n_hubs=n_hubs)
    return _timing.network_state_expression(adata, archetype_id, cluster_key,
                                            states=states, time_key=time_key, edges=e)


# ---------------------------------------------------------------- wrappers


def analyse(
    adata: AnnData,
    ref_network: Optional[pd.DataFrame] = None,
    time_key: Optional[str] = "pseudotime",
    t: float | str = "auto",
    quantile: float = 0.9,
    n_sd: float = 1.0,
    key_gene_list: Sequence[str] = KEY_GENES_DEFAULT,
    branches: Optional[dict[str, Sequence[str]]] = None,
    cluster_key: Optional[str] = None,
    root: Sequence[str] = (),
    n_perm: int = 2000,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Run every driver / responder analysis in order.

    Steps: :func:`influence` → :func:`roles_summary` → :func:`key_genes`
    → :func:`edges` → :func:`reference_check` (if ``ref_network``) →
    :func:`timing` (if ``time_key`` is in ``obs``) → :func:`branch_timing`
    (if ``branches`` and ``cluster_key`` are given; adds ``branch_*``
    tables for ``cells="path"`` and ``branch_members_*`` for
    ``cells="members"``). Returns a dict of
    DataFrames keyed ``influence, roles, key_genes, edges``, plus
    ``reference_summary / reference_per_tf / reference_edges`` and
    ``timing_summary / timing_onsets / timing_edges`` when run.
    """
    out: dict[str, pd.DataFrame] = {}
    out["influence"] = influence(adata, t=t, quantile=quantile, n_sd=n_sd)
    out["roles"] = roles_summary(adata)
    out["key_genes"] = key_genes(adata, key_gene_list)
    out["edges"] = edges(adata, ref_network=ref_network)
    if ref_network is not None and not ref_network.empty:
        rc = reference_check(adata, ref_network, n_perm=n_perm, seed=seed)
        out.update({f"reference_{k}": v for k, v in rc.items()})
    if time_key is not None and time_key in adata.obs:
        tm = timing(adata, time_key=time_key, n_perm=n_perm, seed=seed)
        out.update({f"timing_{k}": v for k, v in tm.items()})
        if branches and cluster_key:
            for cells, pre in (("path", "branch"), ("members", "branch_members")):
                bt = branch_timing(adata, branches, cluster_key, root=root,
                                   time_key=time_key, cells=cells, n_perm=n_perm, seed=seed)
                out.update({f"{pre}_{k}": v for k, v in bt.items()})
    return out


def save_tables(results: dict[str, pd.DataFrame], out_dir: str | Path) -> dict[str, Path]:
    """Write every table from :func:`analyse` as ``<name>.csv``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in results.items():
        p = out / f"{name}.csv"
        df.to_csv(p, index=False)
        paths[name] = p
    return paths


# ----------------------------------------------------------------- figures


def plot_network(adata: AnnData, archetype_ids: Optional[Sequence[int]] = None,
                 ref_network: Optional[pd.DataFrame] = None,
                 save: Optional[str | Path] = None, **kwargs):
    """Driver/responder map, influence over time and network per archetype."""
    from .plot.influence import figure_influence_network
    _ensure_influence(adata)
    return figure_influence_network(adata, archetype_ids=archetype_ids,
                                    ref_network=ref_network, save=save, **kwargs)


def plot_cascade(adata: AnnData, archetype_ids: Optional[Sequence[int]] = None,
                 time_key: str = "pseudotime", save: Optional[str | Path] = None, **kwargs):
    """Pseudotime cascade (timing QC). Returns ``(figure, summary)``."""
    from .plot.influence import figure_expression_cascade
    _ensure_influence(adata)
    return figure_expression_cascade(adata, archetype_ids=archetype_ids,
                                     time_key=time_key, save=save, **kwargs)


def plot_reference(check: dict[str, pd.DataFrame] | pd.DataFrame,
                   save: Optional[str | Path] = None):
    """Reference-recovery AUROC per archetype (from :func:`reference_check`)."""
    from .plot.influence import figure_reference_auroc
    return figure_reference_auroc(check, save=save)


def plot_branch_timing(branch_result: dict[str, pd.DataFrame],
                       save: Optional[str | Path] = None):
    """Archetype × branch source-first matrix and lag distributions."""
    from .plot.influence import figure_branch_timing
    return figure_branch_timing(branch_result, save=save)


def plot_network_expression(adata: AnnData, archetype_id: int, cluster_key: str,
                            states: Optional[Sequence[str]] = None,
                            time_key: str = "pseudotime", result: Optional[dict] = None,
                            save: Optional[str | Path] = None):
    """One archetype: network genes across ordered cell states (optional QC)."""
    from .plot.influence import figure_network_expression
    _ensure_influence(adata)
    if result is None:
        result = network_expression(adata, archetype_id, cluster_key, states=states,
                                    time_key=time_key)
    return figure_network_expression(adata, archetype_id, cluster_key, result=result,
                                     save=save)


def figures(
    adata: AnnData,
    out_dir: str | Path,
    ref_network: Optional[pd.DataFrame] = None,
    time_key: Optional[str] = "pseudotime",
    results: Optional[dict[str, pd.DataFrame]] = None,
    archetype_ids: Optional[Sequence[int]] = None,
    formats: Sequence[str] = ("pdf", "png"),
    cluster_key: Optional[str] = None,
    states: Optional[dict[int, Sequence[str]]] = None,
) -> dict[str, Path]:
    """Render every figure the stored state supports into ``out_dir``.

    ``results`` (from :func:`analyse`) is reused for the reference figure
    so the permutation test is not run twice. With ``cluster_key``, also
    renders the optional per-archetype ``network_expression_arch<k>``
    QC (``states``: optional ``{archetype: ordered states}``). Returns
    ``{name: Path}`` (first format).
    """
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    def _save(fig, name):
        for ext in formats:
            fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths[name] = out / f"{name}.{formats[0]}"

    _save(plot_network(adata, archetype_ids=archetype_ids, ref_network=ref_network),
          "influence_network")
    if time_key is not None and time_key in adata.obs:
        fig, _ = plot_cascade(adata, archetype_ids=archetype_ids, time_key=time_key)
        _save(fig, "expression_cascade")
    if ref_network is not None and not ref_network.empty:
        if results is not None and "reference_summary" in results:
            check = {"summary": results["reference_summary"],
                     "per_tf": results["reference_per_tf"]}
        else:
            check = reference_check(adata, ref_network)
        _save(plot_reference(check), "reference_auroc")
    if results is not None:
        for pre in ("branch", "branch_members"):
            if f"{pre}_summary" in results:
                br = {k: results[f"{pre}_{k}"] for k in ("summary", "pooled", "lags", "overall")
                      if f"{pre}_{k}" in results}
                _save(plot_branch_timing(br), f"{pre}_timing")
    if cluster_key is not None:
        for ak in _ids(adata, archetype_ids):
            st = (states or {}).get(ak)
            _save(plot_network_expression(adata, ak, cluster_key, states=st,
                                          time_key=time_key or "pseudotime"),
                  f"network_expression_arch{ak}")
    return paths
