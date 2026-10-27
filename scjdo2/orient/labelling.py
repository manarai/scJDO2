"""Labelling source: measured latent velocity with a known clock.

The velocity comes from the labelled-rate path (``scjdo2.labelled_rates``
→ ``scjdo2.measured_drift``), stored as ``obsm[cfg.labelling_key]`` in the
embedding's coordinates, per hour. Provides sign, magnitude (``|v·u|`` per
hour) and rotation: ``omega`` = antisymmetric part of a local ridge
regression of the measured velocity on position. ``J`` itself is symmetric
and carries no rotation.
"""

from __future__ import annotations

import numpy as np

from .base import (OrientConfig, SourceResult, antisymmetric, local_jacobian,
                   neighbours, signs_from_direction)


def velocity_from_model(model, z: np.ndarray) -> np.ndarray:
    """Helper: latent velocity from a fitted ``MeasuredDrift`` (or any object
    with ``.velocity(z)``), for writing into ``obsm[cfg.labelling_key]``."""
    return np.asarray(model.velocity(z), dtype=np.float64)


def orient(adata, J, embedding_key, axes, cfg: OrientConfig) -> SourceResult:
    if cfg.labelling_key not in adata.obsm:
        raise KeyError(f"labelling velocity obsm[{cfg.labelling_key!r}] missing")
    z = np.asarray(adata.obsm[embedding_key], dtype=np.float64)
    v = np.asarray(adata.obsm[cfg.labelling_key], dtype=np.float64)
    sign, support, proj, meta = signs_from_direction(v, axes, cfg, "labelling")
    magnitude = np.where(support, np.abs(proj), np.nan)
    finite = np.all(np.isfinite(v), axis=1)
    idx = neighbours(z, cfg.k_neighbours)
    vf = np.where(finite[:, None], v, 0.0)
    omega = antisymmetric(local_jacobian(z, vf, idx, cfg.ridge))
    omega_ok = finite & np.all(finite[idx], axis=1)
    omega[~omega_ok] = np.nan
    meta.update(clock_identified=True, unit="per hour")
    return SourceResult("labelling", sign, support, magnitude=magnitude,
                        omega=omega, omega_support=omega_ok, direction=v, meta=meta)
