"""Orientation layer (docs/ORIENTATION_SPEC.md).

``orient(adata, sources=[...], embedding_key=...)`` gives the score-matched
operator two directions at every cell, from any subset of four sources:

* **flow** — how the cell moves now: labelling > velocity (the user's
  scVelo run); with magnitude (labelling clock) and rotation;
* **developmental** — where development goes over days: lineage >
  pseudotime (the user's Palantir run).

Each group is merged by component and trust ranking, never blended; the two
groups are never merged with each other, and where both sign an axis and
disagree that is reported (``obsm["orient_flow_vs_dev"]``). The energy
drift ``f = ∇log p`` points toward higher density, not along development,
which is why direction needs separate evidence.

A dataset-level reliability check (§15) demotes velocity to a cross-check
when it agrees with labelling below ``cfg.reliability_min`` on the pairs
both sign; the outcome is in ``orient_meta["reliability"]``.

The operator ``uns["scjdo2"]["J"]`` is read, never modified; its byte hash
before and after is stored in ``uns["scjdo2"]["orient_meta"]``.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from . import labelling, lineage, pseudotime, velocity
from .axes import compute_axes
from .base import GROUPS, RANKING, OrientConfig, get_operator, operator_hash
from .flags import flags as _flags
from .merge import cross_group, merge_group, reliability

SOURCES = {"lineage": lineage.orient, "labelling": labelling.orient,
           "velocity": velocity.orient, "pseudotime": pseudotime.orient}

__all__ = ["orient", "OrientConfig", "SOURCES", "RANKING", "GROUPS"]


def orient(adata, sources: Iterable[str], embedding_key: str,
           config: Optional[OrientConfig] = None) -> dict:
    """Run the requested sources, merge each group, write outputs to
    ``adata``; return ``{"flow": …, "dev": …, "flow_vs_dev": …}`` plus the
    per-source results, axes, flags and metadata."""
    cfg = config or OrientConfig()
    sources = list(dict.fromkeys(sources))
    bad = set(sources) - set(SOURCES)
    if bad or not sources:
        raise ValueError(f"sources must be a non-empty subset of {sorted(SOURCES)}; got {sources}")
    J = get_operator(adata)
    h_before = operator_hash(J)
    axes = compute_axes(J, cfg.n_axes, cfg.gap_min)
    results = {s: SOURCES[s](adata, J, embedding_key, axes, cfg) for s in sources}
    n, d = adata.n_obs, J.shape[1]
    k = axes.U.shape[2]
    rel = {g: reliability(g, results, cfg.reliability_min, cfg.reliability_min_pairs) for g in GROUPS}
    demoted = {g: frozenset(s for s, v in rel[g].items() if v["demoted"]) for g in GROUPS}
    flow = merge_group("flow", results, n, k, d, demoted["flow"])
    dev = merge_group("dev", results, n, k, d, demoted["dev"])
    disagree = cross_group(flow, dev)
    fl = _flags(adata, embedding_key, results, cfg)

    adata.obsm["orient_axis_usable"] = axes.usable
    for g, m in (("flow", flow), ("dev", dev)):
        adata.obsm[f"orient_{g}_sign"] = m["sign"]
        adata.obsm[f"orient_{g}_source"] = m["source"]
        adata.obsm[f"orient_{g}_conflict"] = m["conflict"]
        for col in ("sign_source", "orientation_class", "n_sources_supporting",
                    "conflict_flag", "conflict_sources"):
            adata.obs[f"{g}_{col}"] = m[col]
    adata.obsm["orient_flow_magnitude"] = flow["magnitude"]
    adata.obsm["orient_flow_vs_dev"] = disagree
    adata.obs["magnitude_source"] = flow["magnitude_source"]
    adata.obs["omega_source"] = flow["omega_source"]
    adata.obs["flow_vs_dev_disagree"] = disagree.any(axis=1)
    adata.obs["equilibrium_flag"] = fl["equilibrium_flag"]
    adata.obs["stationarity_flag"] = fl["stationarity_flag"]
    scj = adata.uns.setdefault("scjdo2", {})
    scj["omega"] = flow["omega"]
    h_after = operator_hash(get_operator(adata))
    oriented = lambda m: float(np.mean(m["orientation_class"] != "none"))  # noqa: E731
    scj["orient_meta"] = {
        "sources": sources, "groups": {g: [s for s in GROUPS[g] if s in sources] for g in GROUPS},
        "n_axes": int(k), "gap_min": cfg.gap_min, "align_min": cfg.align_min,
        "noise_quantile": dict(cfg.noise_quantile),
        "source_codes": {"flow": flow["source_codes"], "dev": dev["source_codes"]},
        "J_hash_before": h_before, "J_hash_after": h_after,
        "hash_unchanged": bool(h_before == h_after),
        "flow_coverage": oriented(flow), "dev_coverage": oriented(dev),
        "flow_vs_dev_disagreement_rate": float(disagree.sum() / max(1, ((flow["source"] > 0) & (dev["source"] > 0)).sum())),
        "reliability": rel, "demoted": {g: sorted(v) for g, v in demoted.items()},
        "per_source": {s: r.meta for s, r in results.items()}, "flags": fl["meta"],
    }
    return {"flow": flow, "dev": dev, "flow_vs_dev": disagree, **fl,
            "results": results, "axes": axes, "meta": scj["orient_meta"]}
