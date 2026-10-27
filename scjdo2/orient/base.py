"""Shared pieces of the orientation layer (docs/ORIENTATION_SPEC.md).

The layer is post-hoc: it reads the stored operator
``adata.uns["scjdo2"]["J"]`` (or ``"J_latent"``) and never modifies it or
the energy model. Each source returns a :class:`SourceResult`; ``merge``
combines them without blending.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors

# Two orientation outputs (docs/ORIENTATION_SPEC.md §12): each is merged on its
# own and never with the other.
#   flow           — how a cell moves now:          labelling > velocity
#   developmental  — where development goes (days): lineage > pseudotime
GROUPS = {
    "flow": ("labelling", "velocity"),
    "dev": ("lineage", "pseudotime"),
}
MEASURED_SOURCES = {"flow": ("labelling", "velocity"), "dev": ("lineage",)}
RANKING = GROUPS["flow"] + GROUPS["dev"]          # every source, for validation


@dataclass
class OrientConfig:
    """Thresholds and input keys. Defaults are the spec's (§2–§4, §7)."""

    n_axes: int = 5
    gap_min: float = 0.1            # relative eigen-gap / median |λ|
    align_min: float = 0.2          # |v·u| / ‖v‖ for a sign to count
    # Noise floor: a cell's direction vector must exceed this quantile of
    # the source's norms. Velocity-like sources use 0.1 so basin centres
    # (v ≈ 0) drop out; lineage and pseudotime directions are not noisy
    # velocities and use 0.
    noise_quantile: dict = field(default_factory=lambda: {
        "lineage": 0.0, "labelling": 0.1, "velocity": 0.1, "pseudotime": 0.0})
    k_neighbours: int = 30          # local regressions (pseudotime gradient, omega)
    ridge: float = 1e-3             # relative to trace(AᵀA)/d
    # input keys
    clone_key: str = "clone"        # obs column, or obsm clone matrix
    day_key: str = "day"
    labelling_key: str = "velocity_labelling"   # obsm, latent velocity per hour
    velocity_key: Optional[str] = None          # obsm; default velocity_<basis>
    gene_velocity_layer: Optional[str] = None   # layers, gene-space velocity
    linear_map: Optional[np.ndarray] = None     # (d, G) map, gene → latent
    chain_rule_layer: Optional[str] = None      # layers, expression x for v/(1+x)
    pseudotime_key: str = "palantir_pseudotime"
    root_cell: Optional[str] = None             # obs name; else uns["palantir_start_cell"]
    # The start cell must lie in this lowest fraction of pseudotime. It guards
    # against a wrong or reversed root; Palantir places its start cell near,
    # not exactly at, the minimum (orient/S03: 0.6–3.7th percentile).
    root_quantile_max: float = 0.10
    # Direction guard from real time (§16): where sampling days are present,
    # pseudotime must increase with day (Spearman > this), else the source is
    # refused. The start-cell check alone cannot catch a reversed run once
    # the start cell is the pseudotime minimum.
    pseudotime_day_min_spearman: float = 0.0
    # §16 (amended 2026-10-18, orient/S06): the pseudotime minimum must lie on
    # the earliest sampling day; correlation with day alone let reversed
    # runs through (104 of 180 day-4 roots accepted).
    pseudotime_start_on_earliest_day: bool = True
    # §17 (2026-10-18, orient/S06): a clone gives a lineage direction only if
    # it has at least this many sampled cells on both its earliest and its
    # latest day; with one cell per end the direction is one cell's state
    # minus another's (lineage 0.72 vs pseudotime 0.96 under sparse sampling).
    lineage_min_cells_per_end: int = 2
    # Dataset-level reliability check (§15): where a source and the
    # higher-ranked source of its group both sign (cell, axis) pairs, their
    # sign agreement must reach this level, else the source is demoted to a
    # cross-check for this dataset (it records conflicts, decides no sign and
    # gives no omega). Sources not listed are not checked. orient/S03: scVelo
    # agreed with labelling at chance on realistic kinetics.
    reliability_min: dict = field(default_factory=lambda: {"velocity": 0.6})
    reliability_min_pairs: int = 200            # fewer shared pairs: not checked
    score_key: str = "X_drift_v"                # obsm, score f = −∇E (for flags)
    energy_key: str = "energy"                  # obs, E (for stationarity)
    eq_cos_min: float = 0.9
    eq_resid_max: float = 0.3
    stat_ratio_max: float = 0.3


@dataclass
class SourceResult:
    """What one source measured. ``sign`` is 0 where ``support`` is False."""

    name: str
    sign: np.ndarray                       # (n, k) int8
    support: np.ndarray                    # (n, k) bool
    magnitude: Optional[np.ndarray] = None # (n, k) float, labelling only
    omega: Optional[np.ndarray] = None     # (n, d, d)
    omega_support: Optional[np.ndarray] = None  # (n,) bool
    direction: Optional[np.ndarray] = None # (n, d), for flags
    meta: dict = field(default_factory=dict)


def get_operator(adata) -> np.ndarray:
    scj = adata.uns.get("scjdo2", {})
    key = "J_latent" if "J_latent" in scj else "J"
    if key not in scj:
        raise KeyError('adata.uns["scjdo2"] has no "J" or "J_latent"')
    return np.asarray(scj[key])


def operator_hash(J: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(J).tobytes()).hexdigest()


def basis_name(embedding_key: str) -> str:
    """scVelo's naming: ``X_pca`` → ``pca`` (velocity stored as ``velocity_pca``)."""
    return embedding_key[2:] if embedding_key.startswith("X_") else embedding_key


def neighbours(z: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(z) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(z)
    return nn.kneighbors(z, return_distance=False)[:, 1:]


def _normal_equations(z: np.ndarray, idx: np.ndarray, ridge: float):
    A = z[idx] - z[:, None, :]                       # (n, k, d)
    AtA = np.einsum("nkd,nke->nde", A, A)
    d = z.shape[1]
    lam = ridge * np.trace(AtA, axis1=1, axis2=2)[:, None, None] / d
    return A, AtA + lam * np.eye(d)[None]


def local_gradient(z: np.ndarray, t: np.ndarray, idx: np.ndarray, ridge: float) -> np.ndarray:
    """Ridge least-squares gradient of a scalar field over each cell's neighbours."""
    A, M = _normal_equations(z, idx, ridge)
    b = t[idx] - t[:, None]
    return np.linalg.solve(M, np.einsum("nkd,nk->nd", A, b)[..., None])[..., 0]


def local_jacobian(z: np.ndarray, v: np.ndarray, idx: np.ndarray, ridge: float) -> np.ndarray:
    """Ridge least-squares Jacobian of a vector field: v_n − v_c ≈ Jv (z_n − z_c)."""
    A, M = _normal_equations(z, idx, ridge)
    B = v[idx] - v[:, None, :]
    JvT = np.linalg.solve(M, np.einsum("nkd,nke->nde", A, B))
    return np.transpose(JvT, (0, 2, 1))


def antisymmetric(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M - np.transpose(M, (0, 2, 1)))


def signs_from_direction(v: np.ndarray, axes, cfg: OrientConfig, source: str):
    """§3: sign_ij = sign(v_i · u_ij), supported where aligned, above the
    noise floor, and the axis is usable (gap rule)."""
    v = np.asarray(v, dtype=np.float64)
    finite = np.all(np.isfinite(v), axis=1)
    vv = np.where(finite[:, None], v, 0.0)
    norm = np.linalg.norm(vv, axis=1)
    q = cfg.noise_quantile.get(source, 0.0)
    floor = np.quantile(norm[finite], q) if finite.any() and q > 0 else 0.0
    proj = np.einsum("nd,ndk->nk", vv, axes.U)
    align = np.abs(proj) / np.maximum(norm, 1e-300)[:, None]
    support = (finite & (norm > floor) & (norm > 0))[:, None] & (align >= cfg.align_min) & axes.usable
    sign = np.where(support, np.sign(proj), 0).astype(np.int8)
    return sign, support, proj, {"noise_floor": float(floor), "n_finite": int(finite.sum())}
