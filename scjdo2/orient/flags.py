"""§7 Flags. ``equilibrium_flag`` asks whether measured velocity is a scaled
score (detailed balance); ``stationarity_flag`` asks whether the density is
locally balanced, ∇·(p v) − D Δp ≈ 0. A rotating steady state is stationary
but non-equilibrium. Where the evidence is absent: ``not_identified``.
"""

from __future__ import annotations

import numpy as np

from .base import OrientConfig, local_jacobian, neighbours


def flags(adata, embedding_key: str, results: dict, cfg: OrientConfig) -> dict:
    n = adata.n_obs
    eq = np.full(n, "not_identified", dtype=object)
    st = np.full(n, "not_identified", dtype=object)
    info = {}
    src = "labelling" if "labelling" in results else ("velocity" if "velocity" in results else None)
    if src is None or cfg.score_key not in adata.obsm:
        return {"equilibrium_flag": eq.astype(str), "stationarity_flag": st.astype(str), "meta": info}
    v = results[src].direction
    f = np.asarray(adata.obsm[cfg.score_key], dtype=np.float64)
    ok = np.all(np.isfinite(v), axis=1) & np.all(np.isfinite(f), axis=1) & \
         (np.linalg.norm(np.nan_to_num(v), axis=1) > 0) & (np.linalg.norm(np.nan_to_num(f), axis=1) > 0)
    vo, fo = v[ok], f[ok]
    D = float(np.sum(vo * fo) / np.sum(fo * fo))
    cos = np.sum(vo * fo, axis=1) / (np.linalg.norm(vo, axis=1) * np.linalg.norm(fo, axis=1))
    resid = np.linalg.norm(vo - D * fo, axis=1) / np.linalg.norm(vo, axis=1)
    is_eq = (cos >= cfg.eq_cos_min) & (resid <= cfg.eq_resid_max)
    eq[ok] = np.where(is_eq, "equilibrium", "non_equilibrium")
    info.update(velocity_source=src, D=D)
    if src == "labelling" and cfg.energy_key in adata.obs:
        z = np.asarray(adata.obsm[embedding_key], dtype=np.float64)
        E = np.asarray(adata.obs[cfg.energy_key], dtype=np.float64)
        p = np.exp(-(E - np.nanmin(E)))
        q = p[:, None] * (np.nan_to_num(v) - D * np.nan_to_num(f))   # p · flux velocity
        Jq = local_jacobian(z, q, neighbours(z, cfg.k_neighbours), cfg.ridge)
        ratio = np.abs(np.trace(Jq, axis1=1, axis2=2)) / (np.linalg.norm(Jq, axis=(1, 2)) + 1e-300)
        ok_s = ok & np.isfinite(E)
        stationary = (eq == "equilibrium") | (ratio <= cfg.stat_ratio_max)
        st[ok_s] = np.where(stationary[ok_s], "stationary", "non_stationary")
        info.update(stationarity="labelling clock + energy")
    return {"equilibrium_flag": eq.astype(str), "stationarity_flag": st.astype(str), "meta": info}
