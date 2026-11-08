"""Route simulation: walkers that hop between measured cells along a scJDO2 drift.

Entry point :func:`simulate` (also ``scjdo2 simulate`` on the command line)
runs everything from an AnnData with raw counts to an interactive page and a
report. The pieces are public:

* :mod:`.direction` — direction sources (clone, velocity, OT, pseudotime);
* :mod:`.walk` — drift kernel, cell walk, absorption probabilities;
* :mod:`.dynamics` — archetypes, genes and gene coupling along routes;
* :mod:`.fate` — held-out clone fate test;
* :mod:`.backdrop` — the 2-D landscape drawn behind the walkers;
* :mod:`.report` — page, report and tables with generated caveats.
"""

from .direction import choose_source
from .walk import absorption, cell_walk, drift_kernel


def simulate(*args, **kwargs):
    """See :func:`scjdo2.routes.run.simulate`."""
    from .run import simulate as _simulate
    return _simulate(*args, **kwargs)


__all__ = ["simulate", "choose_source", "drift_kernel", "cell_walk", "absorption"]
