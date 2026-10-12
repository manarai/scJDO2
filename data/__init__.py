"""Dataset loaders for the five v1 tutorial examples.

Datasets live under ``/Users/terooatt/Documents/scJDO/data/`` and
``/Users/terooatt/Documents/scJDO/examples/data/``. We **do not duplicate**
them here; the loaders read from those paths directly (and fail with a clear
message if the path is missing).

Supported dataset keys and the v1 notebooks they correspond to:

    paul15          — examples/01_paul15_hybrid_drift_tutorial.ipynb
    fourier_paul15  — examples/02_paul15_fourier_tutorial.ipynb
    sb_synthetic    — examples/03_schrodinger_bridge_tutorial.ipynb
    paul15_bridge   — examples/04_paul15_schrodinger_tutorial.ipynb
    scopatlas       — examples/05_scopatlas_complete_workflow.ipynb

Phase A audit will confirm these mappings; adjust in Phase A if the audit
finds different notebook↔dataset relationships.
"""

from . import loaders  # re-export for ``from data import loaders``

__all__ = ["loaders"]
