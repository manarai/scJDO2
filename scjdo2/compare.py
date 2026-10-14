"""Aging-basins comparison tools (PREREG_aging_basins.md).

Four public entry points:

  * :func:`match_directions` — pair eigenvectors of two symmetric
    Jacobians by cosine; return matched eigenvalue pairs.
  * :func:`isotropy_test`    — test whether old/young eigenvalue
    ratios are anisotropic across matched directions. CV statistic
    vs shuffled null.
  * :func:`anticipation_test` — test whether the shared-space
    aging-displacement `d` points along the young soft axis
    `u_young`. |cos| statistic vs random-half-split null.
  * :func:`compare_types`     — iterate every eligible type in an
    AnnData with young/old labels and return a results DataFrame.

All operate on dense symmetric Jacobians (energy-drift) and respect
the v0.5-basin-lock classification rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------
# match_directions
# -----------------------------------------------------------------


@dataclass
class MatchedDirections:
    """Result of :func:`match_directions`.

    * ``pairs`` — list of ``(i_a, i_b, cos)`` giving the matched
      eigenvector indices in ``J_a`` and ``J_b`` and their cosine
      similarity.
    * ``eig_a``, ``eig_b`` — matched eigenvalues in the same order.
    """

    pairs: list[tuple[int, int, float]]
    eig_a: np.ndarray
    eig_b: np.ndarray

    def ratios(self) -> np.ndarray:
        """``eig_b / eig_a`` per matched direction (``old / young``
        in the aging context). Both eigenvalues are negative at a
        basin; division of two negatives gives a positive ratio."""
        with np.errstate(divide="ignore", invalid="ignore"):
            r = self.eig_b / self.eig_a
        return r


def match_directions(J_a: np.ndarray, J_b: np.ndarray) -> MatchedDirections:
    """Pair eigenvectors of ``J_a`` and ``J_b`` by cosine.

    Both inputs are symmetric ``(d, d)`` matrices (energy drift). We
    symmetrise and ``eigh``. Pairing: greedy, largest-|cos| first,
    without replacement. Returns matched (eigenvalue, eigenvalue)
    pairs in the order of ``J_a``'s descending eigenvalue index.
    """
    if J_a.shape != J_b.shape or J_a.ndim != 2:
        raise ValueError("J_a and J_b must be square matrices of the same shape")
    d = J_a.shape[0]
    ev_a, V_a = np.linalg.eigh(0.5 * (J_a + J_a.T))
    ev_b, V_b = np.linalg.eigh(0.5 * (J_b + J_b.T))

    # Cosine matrix between a-eigvecs and b-eigvecs (sign-free).
    C = np.abs(V_a.T @ V_b)                                        # (d, d)
    pairs: list[tuple[int, int, float]] = []
    used_b = np.zeros(d, dtype=bool)
    for i_a in range(d):
        row = C[i_a].copy()
        row[used_b] = -1.0
        i_b = int(np.argmax(row))
        pairs.append((i_a, i_b, float(C[i_a, i_b])))
        used_b[i_b] = True

    eig_a = np.array([ev_a[i_a] for i_a, _, _ in pairs])
    eig_b = np.array([ev_b[i_b] for _, i_b, _ in pairs])
    return MatchedDirections(pairs=pairs, eig_a=eig_a, eig_b=eig_b)


# -----------------------------------------------------------------
# isotropy_test (PREREG §4)
# -----------------------------------------------------------------


@dataclass
class IsotropyResult:
    ratios: np.ndarray            # per matched direction (reported, not the stat)
    rho_lambda: float             # Spearman(|λ_a|, |λ_b|) across cosine-matched pairs
    null_rho: np.ndarray          # (n_null,)
    null_q95: float
    passes: bool                  # observed ρ > null_q95
    n_valid: int                  # number of pairs used (finite |eig| both sides)
    cv_ratio: float               # legacy field — reported for provenance only


def isotropy_test(J_a: np.ndarray, J_b: np.ndarray,
                   n_null: int = 1000, seed: int = 0) -> IsotropyResult:
    """Direction-coherence of eigenvalue orderings across age
    (PREREG §4a, amended 2026-10-14 before any aging data run).

    **Statistic** — Spearman correlation
    ``ρ_λ = ρ(|λ_a_i|, |λ_b_j(i)|)`` across cosine-matched pairs
    ``(i, j(i))``. High ``ρ_λ`` means directions soft in young stay
    soft in old; stiff stays stiff.

    **Null** — ``n_null`` random pairings of a-eigenvalues to
    b-eigenvalues (eigenvectors ignored). Compute ``ρ_λ`` per draw.

    **Pass** — ``ρ_λ > q95(null)``.

    The CV-of-ratios statistic used in the original prereg is
    permutation-invariant, so its "shuffle ratios" null is
    degenerate; the amendment replaces it with the Spearman
    coherence test above. ``cv_ratio`` is still reported in the
    result for provenance only — it does NOT drive ``passes``.
    """
    from scipy.stats import spearmanr

    md = match_directions(J_a, J_b)
    abs_a = np.abs(md.eig_a)
    abs_b = np.abs(md.eig_b)
    mask = np.isfinite(abs_a) & np.isfinite(abs_b) & (abs_a > 0) & (abs_b > 0)
    a_v = abs_a[mask]
    b_v = abs_b[mask]
    if a_v.size < 3:
        return IsotropyResult(
            ratios=md.ratios(), rho_lambda=float("nan"),
            null_rho=np.array([]), null_q95=float("nan"),
            passes=False, n_valid=int(mask.sum()),
            cv_ratio=float("nan"),
        )
    import warnings
    from scipy.stats import ConstantInputWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        rho_obs = float(spearmanr(a_v, b_v).statistic)
        rng = np.random.default_rng(seed)
        nulls = np.empty(int(n_null), dtype=np.float64)
        for k in range(int(n_null)):
            perm = rng.permutation(b_v.size)
            nulls[k] = float(spearmanr(a_v, b_v[perm]).statistic)
    valid_null = nulls[np.isfinite(nulls)]
    q95 = (float(np.percentile(valid_null, 95))
           if valid_null.size else float("nan"))
    # Legacy CV, reported but no gate action.
    r = md.ratios()
    r_valid = r[np.isfinite(r) & (r > 0)]
    cv_legacy = (float(np.std(r_valid) / (abs(np.mean(r_valid)) + 1e-12))
                 if r_valid.size >= 2 else float("nan"))
    return IsotropyResult(
        ratios=r, rho_lambda=rho_obs, null_rho=nulls, null_q95=q95,
        passes=bool(np.isfinite(q95) and rho_obs > q95),
        n_valid=int(mask.sum()),
        cv_ratio=cv_legacy,
    )


# -----------------------------------------------------------------
# anticipation_test (PREREG §5)
# -----------------------------------------------------------------


@dataclass
class AnticipationResult:
    u_a: np.ndarray               # young soft axis (unit)
    d: np.ndarray                 # old − young centroid
    cos_abs: float                # |cos(u_a, d)|
    null_cos: np.ndarray          # (n_null,) half-split cos
    null_q95: float
    passes: bool
    n_a: int
    n_b: int


def anticipation_test(z_a: np.ndarray, z_b: np.ndarray,
                       J_a: np.ndarray,
                       n_null: int = 1000, seed: int = 0,
                       match_size: bool = True) -> AnticipationResult:
    """Does old − young displacement align with the young soft axis?

    ``z_a`` young latent, ``z_b`` old latent (shared space).
    ``J_a`` is the young centroid Jacobian (symmetric).

    * ``u_a`` = eigenvector of the LEAST-NEGATIVE eigenvalue of
      ``J_a`` (soft axis of the young basin).
    * ``d`` = ``mean(z_b) − mean(z_a)``.
    * Statistic: ``|cos(u_a, d)|``.
    * Null: ``n_null`` random half-splits of ``z_a``; each split
      size matches ``len(z_b)`` iff ``match_size=True``. For every
      split, compute the displacement between the halves and its
      ``|cos|`` with ``u_a``.
    * Pass = observed |cos| > null_q95.
    """
    z_a = np.asarray(z_a, dtype=np.float64)
    z_b = np.asarray(z_b, dtype=np.float64)
    if z_a.ndim != 2 or z_b.ndim != 2 or z_a.shape[1] != z_b.shape[1]:
        raise ValueError("z_a and z_b must be (n, d) with same d")
    ev, V = np.linalg.eigh(0.5 * (J_a + J_a.T))
    lead = int(np.argmax(ev))                                      # least-neg
    u = V[:, lead]
    u = u / (np.linalg.norm(u) + 1e-12)

    mu_a = z_a.mean(axis=0)
    mu_b = z_b.mean(axis=0)
    d = mu_b - mu_a
    cos_abs = abs(float(np.dot(u, d) / (np.linalg.norm(d) + 1e-12)))

    rng = np.random.default_rng(seed)
    n_a = int(z_a.shape[0])
    n_b = int(z_b.shape[0])
    half = n_b if match_size else n_a // 2
    half = max(2, min(half, n_a // 2))
    nulls = np.empty(int(n_null), dtype=np.float64)
    for k in range(int(n_null)):
        idx = rng.permutation(n_a)
        g1 = idx[:half]; g2 = idx[half:2 * half]
        d_k = z_a[g2].mean(axis=0) - z_a[g1].mean(axis=0)
        nulls[k] = abs(float(np.dot(u, d_k) / (np.linalg.norm(d_k) + 1e-12)))
    q95 = float(np.percentile(nulls, 95))
    return AnticipationResult(
        u_a=u, d=d, cos_abs=cos_abs,
        null_cos=nulls, null_q95=q95, passes=bool(cos_abs > q95),
        n_a=n_a, n_b=n_b,
    )


# -----------------------------------------------------------------
# compare_types
# -----------------------------------------------------------------


def compare_types(adata, by: str = "cell_type",
                   groups: Tuple[str, str] = ("young", "old"),
                   group_key: str = "age_group",
                   use_rep: str = "X_scVI",
                   n_cells_centroid: int = 30,
                   min_cells: int = 200,
                   n_null: int = 1000,
                   seed: int = 0) -> pd.DataFrame:
    """Per-type isotropy + anticipation across ``groups`` on a shared
    AnnData.

    Expects ``adata.obsm[use_rep]`` and ``adata.uns["scjdo2"]["J_latent"]``
    (or ``J``) populated for ALL cells (young + old mapped into the
    shared space). ``adata.obs[by]`` holds cell-type labels,
    ``adata.obs[group_key]`` holds the two group labels in ``groups``.

    Returns a DataFrame with one row per eligible type.
    """
    import numpy as _np
    g_young, g_old = groups
    types = adata.obs[by].astype(str).to_numpy()
    grp = adata.obs[group_key].astype(str).to_numpy()
    z = _np.asarray(adata.obsm[use_rep], dtype=_np.float64)
    J = _np.asarray(adata.uns["scjdo2"].get("J_latent",
                                               adata.uns["scjdo2"]["J"]))

    rows = []
    for ct in sorted(set(types)):
        m_y = (types == ct) & (grp == g_young)
        m_o = (types == ct) & (grp == g_old)
        n_y, n_o = int(m_y.sum()), int(m_o.sum())
        if n_y < min_cells or n_o < min_cells:
            continue
        # Centroid operators.
        mu_y = z[m_y].mean(axis=0); mu_o = z[m_o].mean(axis=0)
        d_mu_y = _np.linalg.norm(z - mu_y, axis=1)
        d_mu_o = _np.linalg.norm(z - mu_o, axis=1)
        idx_y = _np.argpartition(d_mu_y, min(n_cells_centroid, len(z) - 1))[
            : n_cells_centroid
        ]
        idx_o = _np.argpartition(d_mu_o, min(n_cells_centroid, len(z) - 1))[
            : n_cells_centroid
        ]
        J_y = J[idx_y].mean(axis=0); J_o = J[idx_o].mean(axis=0)
        iso = isotropy_test(J_y, J_o, n_null=n_null, seed=seed + hash(ct) % 2**31)
        ant = anticipation_test(z[m_y], z[m_o], J_y,
                                 n_null=n_null, seed=seed + hash(ct) % 2**31,
                                 match_size=True)
        rows.append({
            "cell_type": ct, "n_young": n_y, "n_old": n_o,
            "rho_lambda": iso.rho_lambda,
            "iso_null_q95": iso.null_q95,
            "iso_pass": iso.passes,
            "n_valid_ratios": iso.n_valid,
            "CV_ratio_legacy": iso.cv_ratio,
            "cos_abs_u_young_d": ant.cos_abs,
            "ant_null_q95": ant.null_q95,
            "ant_pass": ant.passes,
        })
    return pd.DataFrame(rows)
