"""scJDO v2 — per-cell operator fields, behaviour landscapes and orientation.

Time course (label-free flow operator, App Note)::

    import scjdo2 as sj
    sj.flow(adata, time_key="day")              # J_flow, lambda_1, descriptors
    sj.flow_groups(adata, "cell_type")          # lambda_1 per group, bootstrap CI
    sj.flow_stability(adata, time_key="day")    # per-cell agreement across refits

Snapshot (energy landscape; see API.md)::

    drift = sj.fit(adata, use_rep="X_pca")
    sj.analyse(adata, cluster_key="cell_type", drift=drift)
    sj.orient(adata, ["labelling", "lineage"])
    sj.figures(adata, "figures/", cluster_key="cell_type")
    # or: sj.run(adata, "runs/x", cluster_key="cell_type", sources=[...])
    #     sj.run(adata, "runs/x", cluster_key="cell_type", time_key="day")

The top-level functions come from :mod:`scjdo2.api` and
:mod:`scjdo2.flow_operator`; every step is also a public function in its own
module.
"""

__version__ = "0.6.0"

from .flow_operator import flow, flow_descriptors, ot_velocity  # noqa: E402

_API = ("fit", "analyse", "compare_conditions", "orient", "figures", "run", "FIGURES",
        "flow_groups", "flow_stability")

__all__ = [
    *_API,
    "flow", "ot_velocity", "flow_descriptors", "flow_operator",
    "api", "embed", "energy_drift", "rates", "field", "fisher", "opspace", "landscape",
    "motifs", "compare", "archetypes", "clone", "grammar", "labelled_rates", "measured_drift",
    "orient", "io", "cli", "pipeline", "plot", "data",
]
import importlib.util as _ilu  # noqa: E402

# Optional modules are not shipped in every release.
__all__ = [n for n in __all__ if n not in ("motifs", "compare")
           or _ilu.find_spec(f"{__name__}.{n}") is not None]
del _ilu


def __getattr__(name):
    # Lazy: ``import scjdo2`` stays light; torch loads on first use.
    # (``scjdo2.orient`` is also the orientation subpackage; once imported it
    # is bound here as a callable module that forwards to ``api.orient``.)
    if name in _API:
        from . import api
        return getattr(api, name)
    raise AttributeError(f"module 'scjdo2' has no attribute {name!r}")
