"""scJDO v2 — latent-space neighbourhood Jacobian operator fields.

Flow operator from a time course: ``scjdo2.flow(adata, time_key="day")``
(see :mod:`scjdo2.flow_operator`).

Phase B surface: :func:`scjdo2.pipeline.fit_latent_field` to produce the
AnnData state, :mod:`scjdo2.opspace` for analysis, :mod:`scjdo2.plot` for
publication figures. Phase C adds the NB rate model and the Fisher ledger.
See `V2_PLAN.md` and `CHANGELOG.md`.
"""

__version__ = "0.1.0.dev1"

from .flow_operator import flow, flow_descriptors, ot_velocity  # noqa: E402

__all__ = [
    "flow",
    "ot_velocity",
    "flow_descriptors",
    "flow_operator",
    "embed",
    "drift",
    "rates",
    "field",
    "fisher",
    "opspace",
    "archetypes",
    "clone",
    "grammar",
    "io",
    "cli",
    "pipeline",
    "plot",
    "data",
]
