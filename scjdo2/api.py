"""One-stop API: every scJDO2 analysis and figure from a few calls.

    import scjdo2 as sj

    drift = sj.fit(adata, use_rep="X_pca")                   # energy drift + J + energy
    sj.analyse(adata, cluster_key="cell_type", drift=drift)  # behaviour landscape
    sj.orient(adata, ["labelling", "lineage"])               # flow + developmental signs
    sj.figures(adata, "figures/", cluster_key="cell_type")   # every figure the state supports

For a sampled time course, the flow operator (``sj.flow``) replaces ``fit``:

    sj.flow(adata, time_key="day")                           # J_flow, lambda_1
    sj.flow_groups(adata, "cell_type")                       # lambda_1 per group
    sj.flow_stability(adata, seeds=(0, 1, 2))                # agreement across refits

or all at once, with the AnnData, figures and a summary saved to a run folder:

    summary = sj.run(adata, "runs/x", cluster_key="cell_type",
                     sources=["labelling", "lineage"])
    summary = sj.run(adata, "runs/x", cluster_key="cell_type", time_key="day")

The two operators share ``uns['scjdo2']['J']`` and are kept apart: ``fit``
and ``flow`` refuse to overwrite each other's operator (``overwrite=True``
forces it), and ``analyse`` refuses a flow operator.

Each call is a thin wrapper that fixes the order and defaults of public
functions elsewhere (:mod:`scjdo2.pipeline`, :mod:`scjdo2.landscape`,
:mod:`scjdo2.opspace`, :mod:`scjdo2.compare`, :mod:`scjdo2.orient`,
:mod:`scjdo2.plot`); each of those can still be called on its own. Results
go into the AnnData (``obs``, ``obsm``, ``uns['scjdo2']``) as before.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
from anndata import AnnData

__all__ = ["fit", "analyse", "compare_conditions", "orient", "figures", "run", "FIGURES",
           "flow_groups", "flow_stability"]


# --------------------------------------------------------------------- fit
def fit(adata: AnnData, use_rep: str = "X_pca", *, n_epochs: int = 1500, seed: int = 0,
        hidden: int = 96, depth: int = 3, beta: float = 1.0, lr: float = 5e-4,
        loss: str = "esm", with_rate_model: bool = False, rate_kwargs: Optional[dict] = None,
        overwrite: bool = False, verbose: bool = False):
    """Fit the energy drift on ``obsm[use_rep]`` and store the operator field.

    Writes ``uns['scjdo2']['J']`` (per-cell latent Jacobian, symmetric),
    ``obsm['X_drift_v']`` (the score) and ``obs['energy']``. With
    ``with_rate_model=True`` also fits the NB rate model, the gene-space
    Jacobian and the Fisher ledger (for driver genes); ``rate_kwargs`` go to
    :func:`scjdo2.pipeline.fit_rate_and_gene_field`.

    Returns the fitted :class:`scjdo2.energy_drift.EnergyDriftAdapter`.
    Defaults are the settings used throughout the orientation benchmarks.

    Raises if ``uns['scjdo2']['J']`` holds a flow operator (:func:`scjdo2.flow`),
    unless ``overwrite=True``.
    """
    from .flow_operator import operator_kind

    if operator_kind(adata) == "flow" and not overwrite:
        raise ValueError(
            'adata.uns["scjdo2"]["J"] holds a flow operator (scjdo2.flow). The energy operator '
            "uses the same key: fit it on adata.copy(), or pass overwrite=True.")
    from . import opspace
    from .energy_drift import EnergyDriftConfig
    from .pipeline import fit_latent_field, fit_rate_and_gene_field

    cfg = EnergyDriftConfig(hidden=hidden, depth=depth, n_epochs=n_epochs, seed=seed,
                            beta=beta, lr=lr, loss=loss)
    _, drift = fit_latent_field(adata, use_rep=use_rep, drift_config=cfg, verbose=verbose,
                                return_drift=True)
    opspace.store_energy(adata, drift, use_rep=use_rep)
    if with_rate_model:
        fit_rate_and_gene_field(adata, use_rep=use_rep, seed=seed, verbose=verbose,
                                **(rate_kwargs or {}))
    return drift


# ----------------------------------------------------------------- analyse
def analyse(adata: AnnData, cluster_key: str = "cluster", *, drift: Any = None,
            condition_key: Optional[str] = None, conditions: Optional[Sequence[str]] = None,
            replicate_key: Optional[str] = None, motifs: bool = False,
            tissue_key: Optional[str] = None, resting_types: Optional[Sequence[str]] = None,
            tol: Optional[float] = None, use_rep: Optional[str] = None, seed: int = 0,
            verbose: bool = False) -> dict[str, Any]:
    """Behaviour landscape on a fitted AnnData (:func:`scjdo2.landscape.analyse`).

    Writes the oracle tolerance, operator-space invariants (``lead_re``,
    ``spectral_gap``, ``henrici`` ...), basin classes and index
    (``op_behaviour_basin``, ``op_basin_index``), eigenvalue-sign fractions,
    well stiffness, ``energy`` (when ``drift`` is given) and, with
    ``motifs=True``, recurrent operator motifs.

    With ``condition_key`` and two ``conditions`` also runs :func:`compare_conditions` and returns
    its table under ``"conditions"``.
    """
    from . import landscape

    info = landscape.analyse(adata, cluster_key=cluster_key, use_rep=use_rep,
                             resting_types=resting_types, tol=tol, drift=drift, motifs=motifs,
                             tissue_key=tissue_key, replicate_key=replicate_key, seed=seed,
                             verbose=verbose)
    if condition_key is not None:
        info["conditions"] = compare_conditions(
            adata, cluster_key=cluster_key, condition_key=condition_key,
            conditions=conditions, use_rep=info["use_rep"], seed=seed)
    return info


def compare_conditions(adata: AnnData, cluster_key: str = "cluster",
                       condition_key: str = "condition",
                       conditions: Optional[Sequence[str]] = None, *,
                       use_rep: Optional[str] = None, min_cells: int = 200,
                       n_null: int = 1000, seed: int = 0):
    """Per cell type, compare the operator between two conditions
    (:mod:`scjdo2.compare`, not included in this release; calling this raises
    ``ImportError``)."""
    try:
        from .compare import compare_types
    except ImportError as e:   # not shipped in every release
        raise ImportError("condition comparison (scjdo2.compare) is not included in this "
                          "release of scjdo2") from e

    if conditions is None:
        levels = list(dict.fromkeys(adata.obs[condition_key].astype(str)))
        if len(levels) != 2:
            raise ValueError(f"obs[{condition_key!r}] has {len(levels)} levels; pass conditions=(a, b)")
        conditions = levels
    use_rep = use_rep or adata.uns.get("scjdo2", {}).get("meta", {}).get("use_rep", "X_pca")
    table = compare_types(adata, by=cluster_key, groups=tuple(conditions), group_key=condition_key,
                          use_rep=use_rep, min_cells=min_cells, n_null=n_null, seed=seed)
    adata.uns.setdefault("scjdo2", {})["conditions"] = {
        "key": condition_key, "groups": list(conditions), "table": table.to_dict("list")}
    return table


# ------------------------------------------------------------------ orient
def orient(adata: AnnData, sources: Iterable[str], use_rep: Optional[str] = None, **config):
    """Flow and developmental orientation (:func:`scjdo2.orient.orient`).

    ``sources`` is any subset of ``"labelling"``, ``"velocity"`` (your
    scVelo run), ``"lineage"``, ``"pseudotime"`` (your Palantir run).
    Keyword arguments set :class:`scjdo2.orient.OrientConfig` fields, e.g.
    ``labelling_key="velocity_labelling"``, ``velocity_key="velocity_pca"``,
    ``clone_key="clone"``, ``day_key="day"``. ``use_rep`` defaults to the
    embedding the operator was fitted on.
    """
    from .orient import OrientConfig
    from .orient import orient as _orient

    use_rep = use_rep or adata.uns.get("scjdo2", {}).get("meta", {}).get("use_rep", "X_pca")
    return _orient(adata, list(sources), use_rep, OrientConfig(**config))


# -------------------------------------------------------------------- flow
def flow_groups(adata: AnnData, groupby: str, *, key: str = "flow", n_boot: int = 1000,
                ci: float = 0.95, min_cells: int = 20, seed: int = 0):
    """``lambda_1`` of the flow operator summarised per group of cells.

    The App Note recommends ``lambda_1`` as a within-dataset branch-point score for
    groups of cells; this is that summary. Per level of ``obs[groupby]``: number of
    cells, the fraction whose operator is extrapolated (cells of the last sampling
    time, which have no OT velocity), the median ``lambda_1`` with a percentile
    bootstrap interval over cells (``n_boot`` resamples; none below ``min_cells``),
    and the rank of the median (1 = highest). Cells are resampled as independent
    units, so the interval is conditional on the fitted operator; use
    :func:`flow_stability` for refit variation.

    Stores the table in ``uns[key]['groups']`` and returns it as a DataFrame sorted
    by the median, highest first.
    """
    import pandas as pd

    col = f"{key}_lambda1"
    if col not in adata.obs:
        raise KeyError(f"obs[{col!r}] missing: run scjdo2.flow first")
    if groupby not in adata.obs:
        raise KeyError(f"obs[{groupby!r}] missing")
    lam = adata.obs[col].to_numpy(np.float64)
    grp = adata.obs[groupby].astype(str).to_numpy()
    hv = adata.obs.get(f"{key}_has_velocity")
    hv = None if hv is None else np.asarray(hv, bool)
    rng = np.random.default_rng(seed)
    alpha = (1.0 - ci) / 2.0
    rows = []
    for level in pd.unique(grp):
        m = grp == level
        x = lam[m][np.isfinite(lam[m])]
        lo = hi = np.nan
        if len(x) >= min_cells:
            meds = np.concatenate([np.median(x[rng.integers(0, len(x), (min(100, n_boot - b), len(x)))],
                                             axis=1) for b in range(0, n_boot, 100)])
            lo, hi = np.quantile(meds, [alpha, 1.0 - alpha])
        rows.append({"group": level, "n_cells": int(len(x)),
                     "frac_extrapolated": float(1.0 - hv[m].mean()) if hv is not None else np.nan,
                     "median_lambda1": float(np.median(x)) if len(x) else np.nan,
                     "ci_low": float(lo), "ci_high": float(hi)})
    table = pd.DataFrame(rows).sort_values("median_lambda1", ascending=False, ignore_index=True)
    table["rank"] = np.arange(1, len(table) + 1)
    adata.uns.setdefault(key, {})["groups"] = {
        "groupby": groupby, "ci": ci, "n_boot": n_boot, "min_cells": min_cells, "seed": seed,
        "table": {c: table[c].tolist() for c in table.columns}}
    return table


def flow_stability(adata: AnnData, time_key: Optional[str] = None, *,
                   seeds: Sequence[int] = (0, 1, 2), groupby: Optional[str] = None,
                   key: str = "flow", min_cells: int = 20, **flow_kwargs) -> dict[str, Any]:
    """Refit the flow operator with several fit seeds and report how well
    ``lambda_1`` agrees (the refit check the App Note recommends).

    Each seed reruns :func:`scjdo2.flow` (transport and drift) on a light copy that
    holds only ``obs[time_key]`` and the transport space; the operator stored in
    ``adata`` is not changed. Parameters default to those of the stored run
    (``uns[key]['params']``); ``flow_kwargs`` override them (e.g. ``device="cpu"``).

    Returns, and stores in ``uns[key]['stability']``:

    * ``per_cell_spearman``: Spearman correlation of per-cell ``lambda_1`` for each
      pair of seeds, with its median and minimum;
    * with ``groupby``: the same for group medians (groups with at least
      ``min_cells`` cells), and whether the top-ranked group is the same in every
      refit.

    Also writes ``obs[f"{key}_lambda1_rank_sd"]``: the standard deviation of each
    cell's within-dataset rank quantile across the refits (0 = same rank every time).
    """
    import itertools

    from anndata import AnnData as _AnnData
    from scipy.stats import rankdata, spearmanr

    from .flow_operator import _rank_quantiles, flow

    params = dict(adata.uns.get(key, {}).get("params", {}))
    time_key = time_key or params.get("time_key")
    if time_key is None:
        raise ValueError("pass time_key (no stored scjdo2.flow run to take it from)")
    kw = {k: params[k] for k in ("use_rep", "n_comps", "eps", "solver", "hidden", "weight_decay",
                                 "n_epochs", "n_seeds") if k in params}
    kw.update(flow_kwargs)
    use_rep = kw.setdefault("use_rep", "X_pca")
    if use_rep not in adata.obsm:
        raise KeyError(f"obsm[{use_rep!r}] missing: run scjdo2.flow (or PCA) first")
    if len(seeds) < 2:
        raise ValueError("need at least two seeds")

    light = _AnnData(obs=adata.obs[[time_key]].copy())
    light.obsm[use_rep] = np.asarray(adata.obsm[use_rep])
    lams = []
    for s in seeds:
        b = light.copy()
        flow(b, time_key, seed=int(s), descriptors=False, key_added="_refit", **kw)
        lams.append(b.obs["_refit_lambda1"].to_numpy(np.float64))
    lams = np.vstack(lams)

    pairs = list(itertools.combinations(range(len(seeds)), 2))
    cell_rho = {f"{seeds[i]}-{seeds[j]}": float(spearmanr(lams[i], lams[j]).statistic) for i, j in pairs}
    out: dict[str, Any] = {
        "seeds": [int(s) for s in seeds], "params": {k: v for k, v in kw.items()},
        "per_cell_spearman": cell_rho,
        "per_cell_spearman_median": float(np.median(list(cell_rho.values()))),
        "per_cell_spearman_min": float(np.min(list(cell_rho.values())))}
    ranks = np.vstack([_rank_quantiles(l) for l in lams])
    adata.obs[f"{key}_lambda1_rank_sd"] = ranks.std(axis=0).astype(np.float32)

    if groupby is not None:
        if groupby not in adata.obs:
            raise KeyError(f"obs[{groupby!r}] missing")
        grp = adata.obs[groupby].astype(str).to_numpy()
        levels = [g for g in dict.fromkeys(grp) if (grp == g).sum() >= min_cells]
        if len(levels) < 3:
            raise ValueError(f"fewer than 3 groups with >= {min_cells} cells in obs[{groupby!r}]")
        med = np.array([[np.median(l[grp == g]) for g in levels] for l in lams])
        grho = {f"{seeds[i]}-{seeds[j]}": float(spearmanr(med[i], med[j]).statistic) for i, j in pairs}
        top = [levels[int(np.argmax(m))] for m in med]
        out.update({
            "groupby": groupby, "groups": levels,
            "group_median_lambda1": {str(s): m.tolist() for s, m in zip(seeds, med)},
            "group_spearman": grho,
            "group_spearman_median": float(np.median(list(grho.values()))),
            "group_spearman_min": float(np.min(list(grho.values()))),
            "top_group": top, "top_group_same": bool(len(set(top)) == 1),
            "group_rank_per_seed": {str(s): (len(levels) + 1 - rankdata(m)).tolist()
                                    for s, m in zip(seeds, med)}})
    adata.uns.setdefault(key, {})["stability"] = out
    return out


# ----------------------------------------------------------------- figures
#: Figure names :func:`figures` can produce, with what each needs.
FIGURES = {
    "operator_space": "fit + analyse",
    "landscape": "fit + analyse",
    "landscape_3d": "fit + analyse",
    "ternary": "fit + analyse",
    "energy": "fit (energy) + a 2-D basis (X_umap by default)",
    "motifs": "analyse(motifs=True)",
    "basin_index_by_type": "analyse",
    "spectrum_ladder": "fit (+ condition_key for paired ladders)",
    "soft_axis_map": "fit (+ condition_key for displacement arrows)",
    "orientation": "orient",
    "flow_groups": "flow (time course)",
}

#: Figures that read the energy operator; skipped when the stored operator is a flow one.
_ENERGY_FIGURES = ("operator_space", "landscape", "landscape_3d", "ternary", "energy", "motifs",
                   "basin_index_by_type", "spectrum_ladder", "soft_axis_map", "orientation")


def figures(adata: AnnData, out_dir: str | Path, cluster_key: str = "cluster", *,
            condition_key: Optional[str] = None, conditions: Optional[Sequence[str]] = None,
            energy_rep: str = "X_umap", include: Optional[Sequence[str]] = None,
            null: Optional[dict] = None, verbose: bool = False) -> dict[str, Path]:
    """Render every figure the stored state supports into ``out_dir``.

    Each figure is written as ``<out_dir>/<name>.pdf`` and ``.png``; figures
    whose inputs are missing are skipped (with a note when ``verbose``).
    ``include`` restricts to a subset of :data:`FIGURES`. Returns
    ``{name: path_without_suffix}``.
    """
    import matplotlib.pyplot as plt

    from . import landscape
    from . import plot as P
    from .plot.orientation import figure_orientation

    from .flow_operator import operator_kind
    from .plot.flow import figure_flow_groups

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    want = set(include) if include else set(FIGURES)
    unknown = want - set(FIGURES)
    if unknown:
        raise ValueError(f"unknown figure names {sorted(unknown)}; choose from {sorted(FIGURES)}")
    if operator_kind(adata) == "flow":
        skipped = sorted(want & set(_ENERGY_FIGURES))
        if skipped and verbose and include:
            print(f"[figures] skip {skipped}: they need the energy operator (scjdo2.fit)")
        want -= set(_ENERGY_FIGURES)

    lands = [n for n in ("operator_space", "landscape", "landscape_3d", "ternary", "energy", "motifs")
             if n in want]
    paths: dict[str, Path] = {}
    if lands and "lead_re" in adata.obs:
        paths.update(landscape.figures(adata, out_dir, color_key=cluster_key,
                                       condition_key=condition_key, conditions=conditions,
                                       null=null, energy_rep=energy_rep, include=lands,
                                       verbose=verbose))
    elif lands and verbose:
        print("[figures] skip landscape figures: run sj.analyse first")

    rep = adata.uns.get("scjdo2", {}).get("meta", {}).get("use_rep")
    group_kw = {"use_rep": rep} if rep else {}
    if condition_key is not None:
        group_kw.update(group_key=condition_key,
                        groups=list(conditions) if conditions else None)

    def _do(name, fn):
        if name not in want:
            return
        try:
            fig = fn(out_dir / name)
        except (RuntimeError, KeyError, ValueError) as e:
            if verbose:
                print(f"[figures] skip {name}: {e}")
            return
        plt.close(fig)
        paths[name] = out_dir / name
        if verbose:
            print(f"[figures] wrote {name}.pdf/.png")

    _do("basin_index_by_type", lambda p: P.basin_index_by_type(adata, by=cluster_key, out=p))
    _do("spectrum_ladder", lambda p: P.spectrum_ladder(adata, by=cluster_key, out=p, **group_kw))
    _do("soft_axis_map", lambda p: P.soft_axis_map(adata, by=cluster_key, out=p, **group_kw))
    _do("orientation", lambda p: figure_orientation(adata, groupby=cluster_key if cluster_key in adata.obs
                                                    else None, out=p))
    if "flow_lambda1" in adata.obs and cluster_key in adata.obs:
        _do("flow_groups", lambda p: figure_flow_groups(adata, cluster_key, out=p))
    elif "flow_groups" in want and include and verbose:
        print("[figures] skip flow_groups: run sj.flow and give a cluster_key in obs")
    return paths


# --------------------------------------------------------------------- run
def run(adata: AnnData, out_dir: str | Path, cluster_key: str = "cluster", *,
        use_rep: str = "X_pca", sources: Optional[Iterable[str]] = None,
        orient_config: Optional[dict] = None, condition_key: Optional[str] = None,
        conditions: Optional[Sequence[str]] = None, replicate_key: Optional[str] = None,
        motifs: bool = False, with_rate_model: bool = False, fit_kwargs: Optional[dict] = None,
        figures_kwargs: Optional[dict] = None, refit: bool = False,
        time_key: Optional[str] = None, flow_kwargs: Optional[dict] = None,
        stability_seeds: Optional[Sequence[int]] = None,
        verbose: bool = False) -> dict[str, Any]:
    """Everything in one call: fit → analyse → (orient) → figures → save.

    Skips the fit when ``uns['scjdo2']['J']`` already exists, unless
    ``refit``.

    **Time course.** With ``time_key`` (a numeric sampling-time column) the flow
    operator replaces the energy fit: :func:`scjdo2.flow` (``flow_kwargs`` go to it)
    → :func:`flow_groups` by ``cluster_key`` (``groups.csv``) → optionally
    :func:`flow_stability` over ``stability_seeds`` → the ``flow_groups`` figure →
    save. ``sources``, ``condition_key``, ``motifs`` and ``with_rate_model`` belong to
    the energy route and are refused here. Writes to ``out_dir``: the run (``adata.h5ad`` + sidecar, via
    :func:`scjdo2.io.save`), ``figures/``, ``summary.json`` and
    ``per_cell.csv``. Returns the summary dict.
    """
    from . import io

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"use_rep": use_rep, "cluster_key": cluster_key}

    if time_key is not None:
        return _run_flow(adata, out_dir, cluster_key, summary, time_key=time_key, use_rep=use_rep,
                         flow_kwargs=flow_kwargs, stability_seeds=stability_seeds, refit=refit,
                         figures_kwargs=figures_kwargs, verbose=verbose,
                         refused={"sources": sources, "condition_key": condition_key,
                                  "motifs": motifs, "with_rate_model": with_rate_model})

    drift = None
    if refit or "J" not in adata.uns.get("scjdo2", {}):
        drift = fit(adata, use_rep=use_rep, with_rate_model=with_rate_model, verbose=verbose,
                    **(fit_kwargs or {}))
        summary["fit"] = adata.uns["scjdo2"]["meta"]["drift"]
    else:
        summary["fit"] = "reused stored J"

    info = analyse(adata, cluster_key=cluster_key, drift=drift, condition_key=condition_key,
                   conditions=conditions, replicate_key=replicate_key, motifs=motifs,
                   use_rep=use_rep, verbose=verbose)
    beh = adata.obs.get("op_behaviour_basin")
    summary["analyse"] = {
        "tol": info["tol"], "motifs_K": info.get("K"),
        "behaviour_fractions": (beh.astype(str).value_counts(normalize=True).round(4).to_dict()
                                if beh is not None else None),
    }
    if "conditions" in info:
        info["conditions"].to_csv(out_dir / "conditions.csv", index=False)
        summary["analyse"]["conditions_csv"] = "conditions.csv"

    if sources:
        o = orient(adata, sources, use_rep=use_rep, **(orient_config or {}))
        m = o["meta"]
        summary["orient"] = {"sources": m["sources"], "flow_coverage": m["flow_coverage"],
                             "dev_coverage": m["dev_coverage"],
                             "flow_vs_dev_disagreement_rate": m["flow_vs_dev_disagreement_rate"],
                             "demoted": m["demoted"], "J_unchanged": m["hash_unchanged"]}

    figs = figures(adata, out_dir / "figures", cluster_key=cluster_key,
                   condition_key=condition_key, conditions=conditions, verbose=verbose,
                   **(figures_kwargs or {}))
    summary["figures"] = sorted(figs)

    cols = [c for c in ("lead_re", "spectral_gap", "henrici", "op_behaviour_basin", "op_basin_index",
                        "op_stiffness", "energy", "op_motif", "flow_orientation_class",
                        "dev_orientation_class", "flow_conflict_flag", "dev_conflict_flag",
                        "flow_vs_dev_disagree", "equilibrium_flag", "stationarity_flag")
            if c in adata.obs]
    keep = [cluster_key] if cluster_key in adata.obs else []
    adata.obs[keep + cols].to_csv(out_dir / "per_cell.csv")
    io.save(adata, out_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json))
    return summary


def _run_flow(adata, out_dir: Path, cluster_key: str, summary: dict, *, time_key: str,
              use_rep: str, flow_kwargs, stability_seeds, refit: bool, figures_kwargs,
              verbose: bool, refused: dict) -> dict[str, Any]:
    from . import io
    from .flow_operator import flow, operator_kind

    bad = [k for k, v in refused.items() if v]
    if bad:
        raise ValueError(f"{bad} apply to the energy route (no time_key); the flow route does not "
                         "use them")
    summary["route"] = "flow"
    if refit or operator_kind(adata) != "flow":
        flow(adata, time_key, use_rep=use_rep, **(flow_kwargs or {}))
        summary["flow"] = "fitted"
    else:
        summary["flow"] = "reused stored flow operator"
    fl = adata.uns["flow"]
    summary["flow_params"] = fl["params"]
    summary["n_cells"] = int(adata.n_obs)
    summary["frac_with_velocity"] = float(np.mean(adata.obs["flow_has_velocity"]))
    if cluster_key in adata.obs:
        tab = flow_groups(adata, cluster_key)
        tab.to_csv(out_dir / "groups.csv", index=False)
        summary["groups_csv"] = "groups.csv"
        summary["top_groups"] = tab["group"].head(3).tolist()
    elif verbose:
        print(f"[run] obs[{cluster_key!r}] missing: no group summary")
    if stability_seeds:
        st = flow_stability(adata, time_key, seeds=stability_seeds,
                            groupby=cluster_key if cluster_key in adata.obs else None,
                            **{k: v for k, v in (flow_kwargs or {}).items() if k == "device"})
        summary["stability"] = {k: st[k] for k in ("seeds", "per_cell_spearman_median",
                                                   "per_cell_spearman_min", "group_spearman_median",
                                                   "group_spearman_min", "top_group_same") if k in st}
    figs = figures(adata, out_dir / "figures", cluster_key=cluster_key, verbose=verbose,
                   **(figures_kwargs or {}))
    summary["figures"] = sorted(figs)
    cols = [c for c in ("flow_lambda1", "flow_has_velocity", "flow_lambda1_rank_sd") if c in adata.obs]
    keep = [cluster_key] if cluster_key in adata.obs else []
    adata.obs[keep + [time_key] + cols].to_csv(out_dir / "per_cell.csv")
    io.save(adata, out_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json))
    return summary


def _json(x):
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)
