"""Dataset loaders for the v1 tutorials, matched to the audit (V2_PLAN §7).

Keys:
    paul15          — ``sc.datasets.paul15()``, used by notebooks 01, 02, 04, 05.
    fourier_paul15  — alias for paul15 (notebook 02 is Fourier-only on Paul15).
    paul15_bridge   — alias for paul15 (notebook 04 is bridge on Paul15).
    scopatlas       — Paul15 again (notebook 05 loader stub falls back to it).
    sb_synthetic    — generated in-notebook; see :func:`load_sb_synthetic`.

v1 is not needed to call ``load(key)`` — scanpy is enough.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from anndata import AnnData

V1_REPO = Path("/Users/terooatt/Documents/scJDO")

# Groupby / root defaults match v1 notebook calls to prepare_trajectory.
# For the 2D synthetic bridge dataset there is no cluster/root — the bridge
# workflow does not call prepare_trajectory. Marked None.
_GROUPBY: dict[str, str | None] = {
    "paul15": "paul15_clusters",
    "fourier_paul15": "paul15_clusters",
    "paul15_bridge": "paul15_clusters",
    "scopatlas": "paul15_clusters",
    "sb_synthetic": None,
}

_ROOT: dict[str, str | None] = {
    "paul15": "7MEP",
    "fourier_paul15": "7MEP",
    "paul15_bridge": "7MEP",
    "scopatlas": "7MEP",
    "sb_synthetic": None,
}


def _load_paul15() -> AnnData:
    """Paul 2015 haematopoiesis — scanpy's bundled version, as v1 uses."""
    import scanpy as sc

    return sc.datasets.paul15()


def _load_sb_synthetic() -> AnnData:
    """Notebook 03's 2D synthetic bridge data is generated in-notebook and not
    reused via a loader. v1's `fit_drift` is not called on it. Phase A simply
    raises here with a pointer to the notebook; `compare/run_v1.py` skips this
    dataset when the workflow is bridge-only.
    """
    raise NotImplementedError(
        "sb_synthetic is generated inside examples/03_schrodinger_bridge_tutorial.ipynb; "
        "there is no stand-alone loader. compare/run_v1.py skips this dataset."
    )


_LOADERS: dict[str, Callable[[], AnnData]] = {
    "paul15": _load_paul15,
    "fourier_paul15": _load_paul15,
    "paul15_bridge": _load_paul15,
    "scopatlas": _load_paul15,
    "sb_synthetic": _load_sb_synthetic,
}

# Which v1 workflow each dataset exercises. The comparison harness switches on
# this: only `fit_drift` datasets map to v2's pipeline. See V2_PLAN §7.
WORKFLOW: dict[str, str] = {
    "paul15": "fit_drift",
    "fourier_paul15": "fourier",
    "paul15_bridge": "fit_bridge",
    "scopatlas": "atlas",
    "sb_synthetic": "fit_bridge",
}


def load(key: str) -> AnnData:
    try:
        fn = _LOADERS[key]
    except KeyError:
        raise KeyError(f"Unknown dataset {key!r}. Options: {sorted(_LOADERS)}")
    return fn()


def groupby_for(key: str) -> str | None:
    return _GROUPBY[key]


def root_for(key: str) -> str | None:
    return _ROOT[key]


def workflow_for(key: str) -> str:
    return WORKFLOW[key]


def available() -> list[str]:
    return sorted(_LOADERS)
