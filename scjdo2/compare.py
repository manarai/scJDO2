"""Aging-basins comparison tools (PREREG_aging_basins.md).

Four public entry points:

  * :func:`match_directions` — pair eigenvectors of two symmetric
    Jacobians by cosine; return matched eigenvalue pairs.
  * :func:`isotropy_test`    — test whether the old/young operator
    change departs from uniform across matched directions
    (``S = max |r_k − median r|``, split-half-within-young null;
    PREREG §4b).
  * :func:`anticipation_test` — test whether the shared-space
    aging-displacement `d` points along the young soft axis
    `u_young`. |cos| statistic vs random-half-split null.
  * :func:`compare_types`     — iterate every eligible type in an
    AnnData with young/old labels and return a results DataFrame.

All operate on dense symmetric Jacobians (energy-drift) and respect
the v0.5-basin-lock classification rule.
"""

from __future__ import annotations

import zlib
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
# isotropy_test (PREREG §4, §4b)
# -----------------------------------------------------------------


def centroid_operator(z: np.ndarray, J_cells: np.ndarray,
                      n_cells_centroid: int = 30) -> np.ndarray:
    """Mean per-cell Jacobian over the ``n_cells_centroid`` cells of
    this set nearest the set's centroid (PREREG §3, §4b)."""
    z = np.asarray(z, dtype=np.float64)
    mu = z.mean(axis=0)
    dist = np.linalg.norm(z - mu, axis=1)
    k = min(int(n_cells_centroid), len(z))
    idx = np.argsort(dist, kind="stable")[:k]
    return np.asarray(J_cells, dtype=np.float64)[idx].mean(axis=0)


def isotropy_statistic(J_a: np.ndarray, J_b: np.ndarray
                       ) -> tuple[float, np.ndarray, MatchedDirections]:
    """``S = max_k |r_k − median(r)|`` with
    ``r_k = log(|λ_b,k| / |λ_a,k|)`` over cosine-matched pairs.

    ``S ≈ 0`` when every direction changes by the same factor
    (isotropic); large when one direction departs from the typical
    change. Returns ``(S, r, matched)``; ``S`` is NaN with fewer than
    3 valid pairs.
    """
    md = match_directions(J_a, J_b)
    abs_a = np.abs(md.eig_a)
    abs_b = np.abs(md.eig_b)
    mask = np.isfinite(abs_a) & np.isfinite(abs_b) & (abs_a > 0) & (abs_b > 0)
    r = np.full(abs_a.shape, np.nan)
    r[mask] = np.log(abs_b[mask] / abs_a[mask])
    r_v = r[mask]
    if r_v.size < 3:
        return float("nan"), r, md
    return float(np.max(np.abs(r_v - np.median(r_v)))), r, md


@dataclass
class IsotropyResult:
    S: float                      # max_k |r_k − median(r)|  (the §4b statistic)
    log_ratios: np.ndarray        # r_k = log(|λ_old| / |λ_young|) per matched pair
    null_S: np.ndarray            # (n_null,) split-half-within-young
    null_q95: float
    passes: bool                  # S > null_q95
    n_valid: int                  # matched pairs with finite, non-zero |λ| both sides
    n_half: int                   # cells per half in the null
    ratios: np.ndarray            # λ_old / λ_young per matched pair (provenance)
    rho_lambda: float             # §4a Spearman — provenance only, not a gate
    cv_ratio: float               # original §4 CV — provenance only, not a gate


def isotropy_test(z_a: np.ndarray, J_cells_a: np.ndarray,
                  z_b: np.ndarray, J_cells_b: np.ndarray,
                  n_cells_centroid: int = 30,
                  n_null: int = 1000, seed: int = 0) -> IsotropyResult:
    """Is the young → old operator change concentrated in specific
    directions rather than uniform? (PREREG §4b, amended 2026-10-14
    before any aging data run; replaces §4a.)

    ``z_a``/``J_cells_a`` are young cells of one type (latent
    positions and per-cell Jacobians from the fixed shared-space
    drift), ``z_b``/``J_cells_b`` the old cells.

    * Operators: :func:`centroid_operator` of each group.
    * Statistic: :func:`isotropy_statistic` of (young, old).
    * Null: ``n_null`` random splits of the young cells into two
      disjoint halves of ``min(n_old, n_young // 2)`` cells; ``S``
      between the two halves' centroid operators. Sampling noise
      under no aging.
    * Pass: ``S > q95(null)``.

    Uniform (isotropic) softening scales every eigenvalue by the same
    factor, so ``S`` stays at the null and the test fails, which is
    the outcome §4 must reject. The §4a Spearman ``ρ`` and the §4 CV
    are returned for provenance only.
    """
    z_a = np.asarray(z_a, dtype=np.float64)
    z_b = np.asarray(z_b, dtype=np.float64)
    J_cells_a = np.asarray(J_cells_a, dtype=np.float64)
    J_cells_b = np.asarray(J_cells_b, dtype=np.float64)
    if len(z_a) != len(J_cells_a) or len(z_b) != len(J_cells_b):
        raise ValueError("z and J_cells must have the same number of cells")

    J_a = centroid_operator(z_a, J_cells_a, n_cells_centroid)
    J_b = centroid_operator(z_b, J_cells_b, n_cells_centroid)
    S_obs, r, md = isotropy_statistic(J_a, J_b)

    n_a = len(z_a)
    n_half = max(2, min(len(z_b), n_a // 2))
    rng = np.random.default_rng(seed)
    nulls = np.empty(int(n_null), dtype=np.float64)
    for k in range(int(n_null)):
        idx = rng.permutation(n_a)
        h1, h2 = idx[:n_half], idx[n_half:2 * n_half]
        nulls[k] = isotropy_statistic(
            centroid_operator(z_a[h1], J_cells_a[h1], n_cells_centroid),
            centroid_operator(z_a[h2], J_cells_a[h2], n_cells_centroid),
        )[0]
    valid_null = nulls[np.isfinite(nulls)]
    q95 = (float(np.percentile(valid_null, 95))
           if valid_null.size else float("nan"))

    # Provenance: §4a Spearman and §4 CV, no gate action.
    import warnings
    from scipy.stats import ConstantInputWarning, spearmanr
    abs_a, abs_b = np.abs(md.eig_a), np.abs(md.eig_b)
    mask = np.isfinite(r)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        rho = (float(spearmanr(abs_a[mask], abs_b[mask]).statistic)
               if mask.sum() >= 3 else float("nan"))
    ratios = md.ratios()
    r_pos = ratios[np.isfinite(ratios) & (ratios > 0)]
    cv = (float(np.std(r_pos) / (abs(np.mean(r_pos)) + 1e-12))
          if r_pos.size >= 2 else float("nan"))

    return IsotropyResult(
        S=S_obs, log_ratios=r, null_S=nulls, null_q95=q95,
        passes=bool(np.isfinite(S_obs) and np.isfinite(q95) and S_obs > q95),
        n_valid=int(mask.sum()), n_half=int(n_half),
        ratios=ratios, rho_lambda=rho, cv_ratio=cv,
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

    Centroid operators are taken within each (type, group) cell set
    (:func:`centroid_operator`). ``rho`` and ``cv_ratio`` are
    provenance columns; ``iso_pass`` is the §4b gate.

    Returns a DataFrame with one row per eligible type.
    """
    import numpy as _np
    g_young, g_old = groups
    types = adata.obs[by].astype(str).to_numpy()
    grp = adata.obs[group_key].astype(str).to_numpy()
    z = _np.asarray(adata.obsm[use_rep], dtype=_np.float64)
    uns = adata.uns["scjdo2"]
    J = _np.asarray(uns["J_latent"] if "J_latent" in uns else uns["J"])

    rows = []
    for ct in sorted(set(types)):
        m_y = (types == ct) & (grp == g_young)
        m_o = (types == ct) & (grp == g_old)
        n_y, n_o = int(m_y.sum()), int(m_o.sum())
        if n_y < min_cells or n_o < min_cells:
            continue
        # Stable per-type seed (builtin hash() of str is salted per process).
        ct_seed = seed + zlib.crc32(ct.encode("utf-8")) % 2**31
        iso = isotropy_test(z[m_y], J[m_y], z[m_o], J[m_o],
                            n_cells_centroid=n_cells_centroid,
                            n_null=n_null, seed=ct_seed)
        J_y = centroid_operator(z[m_y], J[m_y], n_cells_centroid)
        ant = anticipation_test(z[m_y], z[m_o], J_y,
                                 n_null=n_null, seed=ct_seed,
                                 match_size=True)
        rows.append({
            "cell_type": ct, "n_young": n_y, "n_old": n_o,
            "S": iso.S,
            "iso_null_q95": iso.null_q95,
            "iso_pass": iso.passes,
            "n_valid_ratios": iso.n_valid,
            "rho": iso.rho_lambda,
            "cv_ratio": iso.cv_ratio,
            "cos_abs_u_young_d": ant.cos_abs,
            "ant_null_q95": ant.null_q95,
            "ant_pass": ant.passes,
        })
    return pd.DataFrame(rows)

# -----------------------------------------------------------------
# aging_modes (spec §5 descriptive clustering)
# -----------------------------------------------------------------

def _mode_spectrum(value, width: int = 10) -> np.ndarray:
    """Return the ten softest sorted eigenvalues, padded deterministically."""
    if isinstance(value, str):
        values = [float(x) for x in value.split(";") if x]
    else:
        values = np.asarray(value, dtype=float).ravel().tolist()
    values = np.sort(np.asarray(values, dtype=float))
    values = values[-width:]
    out = np.zeros(width, dtype=float)
    out[-len(values):] = values
    return out


def aging_modes(per_type: pd.DataFrame, *, seed: int = 0) -> pd.DataFrame:
    """Cluster aging operator-coordinate changes and label each mode.

    ``per_type`` has one shared-space row per cell type and must contain the
    paired operator coordinates used in the frozen §5 description:
    ``spectrum_{young,old}``, ``median_abs_lambda_{young,old}``,
    ``anisotropy_{young,old}``, ``n_soft_{young,old}``,
    ``basin_index_{young,old}``, plus ``displacement`` and
    ``split_half_noise``. Spectra are sorted, reduced to their ten softest
    directions, and zero-padded where needed. K-means chooses ``k`` from 2–5
    by silhouette score; if fewer than three rows are supplied, the sole
    deterministic mode is labelled ``mixed``.

    Labels are descriptive, never outcome gates. They are assigned from the
    unscaled centroid of each selected mode: ``unchanged`` has centroid
    displacement at or below split-half noise; ``single-axis loss`` has a
    >25% anisotropy rise while median stiffness changes by at most 20%;
    ``uniform softening`` has all ten spectral coordinates shrinking with an
    absolute anisotropy change below 10%; all other modes are ``mixed``.
    """
    required = {
        "spectrum_young", "spectrum_old",
        "median_abs_lambda_young", "median_abs_lambda_old",
        "anisotropy_young", "anisotropy_old",
        "n_soft_young", "n_soft_old",
        "basin_index_young", "basin_index_old",
        "displacement", "split_half_noise",
    }
    missing = sorted(required - set(per_type.columns))
    if missing:
        raise KeyError(f"aging_modes missing columns: {missing}")
    n = len(per_type)
    if n == 0:
        return pd.DataFrame(columns=["mode_id", "mode_label", "silhouette_k"])

    eps = 1e-12
    specs_y = np.vstack([_mode_spectrum(x) for x in per_type["spectrum_young"]])
    specs_o = np.vstack([_mode_spectrum(x) for x in per_type["spectrum_old"]])
    spec_log_ratio = np.log(np.maximum(np.abs(specs_o), eps) /
                            np.maximum(np.abs(specs_y), eps))
    med_y = per_type["median_abs_lambda_young"].to_numpy(dtype=float)
    med_o = per_type["median_abs_lambda_old"].to_numpy(dtype=float)
    ani_y = per_type["anisotropy_young"].to_numpy(dtype=float)
    ani_o = per_type["anisotropy_old"].to_numpy(dtype=float)
    nsoft_y = per_type["n_soft_young"].to_numpy(dtype=float)
    nsoft_o = per_type["n_soft_old"].to_numpy(dtype=float)
    basin_y = per_type["basin_index_young"].to_numpy(dtype=float)
    basin_o = per_type["basin_index_old"].to_numpy(dtype=float)
    displacement = per_type["displacement"].to_numpy(dtype=float)
    split_noise = per_type["split_half_noise"].to_numpy(dtype=float)

    med_log_ratio = np.log(np.maximum(np.abs(med_o), eps) /
                           np.maximum(np.abs(med_y), eps))
    anisotropy_rel = (ani_o - ani_y) / np.maximum(np.abs(ani_y), eps)
    raw = np.column_stack([
        spec_log_ratio,
        med_log_ratio,
        anisotropy_rel,
        nsoft_o - nsoft_y,
        basin_o - basin_y,
    ])
    # The energy field should yield finite operator coordinates. Retain a
    # deterministic finite fallback for descriptive clustering only.
    col_medians = np.nanmedian(raw, axis=0)
    col_medians = np.where(np.isfinite(col_medians), col_medians, 0.0)
    raw = np.where(np.isfinite(raw), raw, col_medians[None, :])

    if n < 3:
        labels = np.zeros(n, dtype=int)
        silhouette_k = 1
    else:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler

        scaled = StandardScaler().fit_transform(raw)
        best: tuple[float, int, np.ndarray] | None = None
        for k in range(2, min(5, n - 1) + 1):
            model = KMeans(n_clusters=k, random_state=seed, n_init=50)
            candidate = model.fit_predict(scaled)
            if len(set(candidate)) < 2:
                continue
            score = float(silhouette_score(scaled, candidate))
            # `>` deliberately preserves the smaller k on an exact tie.
            if best is None or score > best[0]:
                best = (score, k, candidate)
        if best is None:
            labels = np.zeros(n, dtype=int)
            silhouette_k = 1
        else:
            _, silhouette_k, labels = best

    def label_mode(member: np.ndarray) -> str:
        center_spec = np.mean(spec_log_ratio[member], axis=0)
        center_med_rel = float(np.expm1(np.mean(med_log_ratio[member])))
        center_aniso_rel = float(np.mean(anisotropy_rel[member]))
        center_displacement = float(np.mean(displacement[member]))
        center_noise = float(np.mean(split_noise[member]))
        if np.isfinite(center_displacement) and np.isfinite(center_noise) and center_displacement <= center_noise:
            return "unchanged"
        if center_aniso_rel > 0.25 and abs(center_med_rel) <= 0.20:
            return "single-axis loss"
        active_spectrum = np.any(
            (np.abs(specs_y[member]) > eps) & (np.abs(specs_o[member]) > eps),
            axis=0,
        )
        if active_spectrum.any() and np.all(center_spec[active_spectrum] < 0.0) and abs(center_aniso_rel) < 0.10:
            return "uniform softening"
        return "mixed"

    unique = sorted(set(int(x) for x in labels))
    # KMeans labels are arbitrary. Reindex by the raw centroid for stable output.
    centroid_key = {
        c: tuple(np.mean(raw[labels == c], axis=0).round(12))
        for c in unique
    }
    relabel = {old: new for new, old in enumerate(sorted(unique, key=centroid_key.get))}
    mode_id = np.array([relabel[int(x)] for x in labels], dtype=int)
    mode_label = {}
    for old in unique:
        mode_label[relabel[old]] = label_mode(labels == old)
    return pd.DataFrame(
        {
            "mode_id": mode_id,
            "mode_label": [mode_label[int(x)] for x in mode_id],
            "silhouette_k": int(silhouette_k),
        },
        index=per_type.index,
    )


# -----------------------------------------------------------------
# Continuous-age, donor-level tests (PREREG_aging_hlca.md §4–§6)
# -----------------------------------------------------------------
#
# The two-group tests above use a cell-level split-half null, which does
# not see donor-to-donor variation (aging/X05). The human arm treats age
# as continuous and makes the donor the unit: one equal-count operator
# per donor, statistics are slopes against age, and the null permutes
# ages across donors.


def donor_operator(z: np.ndarray, J_cells: np.ndarray, n_common: int,
                   n_draws: int = 100, n_cells_centroid: int = 30,
                   seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Equal-count operator and centroid for one donor × type.

    Draws ``n_draws`` subsamples of ``n_common`` cells without
    replacement; for each, :func:`centroid_operator` (``n_cells_centroid``
    cells). Returns the mean operator over draws and the mean centroid
    of the draws. Equal counts across donors keep the 30-nearest-cell
    operator comparable (aging/X05).
    """
    z = np.asarray(z, dtype=np.float64)
    J_cells = np.asarray(J_cells, dtype=np.float64)
    if len(z) < n_common:
        raise ValueError(f"donor has {len(z)} cells < n_common={n_common}")
    rng = np.random.default_rng(seed)
    ops, cents = [], []
    for _ in range(int(n_draws)):
        s = rng.choice(len(z), int(n_common), replace=False)
        ops.append(centroid_operator(z[s], J_cells[s], n_cells_centroid))
        cents.append(z[s].mean(axis=0))
    op = np.mean(ops, axis=0)
    return 0.5 * (op + op.T), np.mean(cents, axis=0)


def _slopes(Y: np.ndarray, ages: np.ndarray) -> np.ndarray:
    """OLS slope of each column of ``Y`` (donors × k) on ``ages``."""
    a = ages - ages.mean()
    return (a @ (Y - Y.mean(axis=0))) / (a @ a)


def _perm_ages(ages: np.ndarray, n_perm: int, seed: int,
               strata: Optional[Sequence] = None) -> np.ndarray:
    """``n_perm`` permutations of ``ages`` across donors; with ``strata``
    (e.g. study per donor), ages are permuted only within each stratum
    (PREREG_aging_hlca §7a). ``strata=None`` is the primary test."""
    rng = np.random.default_rng(seed)
    if strata is None:
        return np.stack([rng.permutation(ages) for _ in range(int(n_perm))])
    strata = np.asarray(strata)
    groups = [np.where(strata == g)[0] for g in np.unique(strata)]
    out = np.empty((int(n_perm), len(ages)))
    for k in range(int(n_perm)):
        a = ages.copy()
        for idx in groups:
            a[idx] = ages[idx][rng.permutation(len(idx))]
        out[k] = a
    return out


@dataclass
class TrendResult:
    statistic: float
    null: np.ndarray
    null_q95: float
    passes: bool
    p_value: float
    detail: dict


def isotropy_trend(donor_ops: Sequence[np.ndarray], ages: Sequence[float],
                   ref_op: np.ndarray, n_perm: int = 1000,
                   seed: int = 0,
                    strata: Optional[Sequence] = None) -> TrendResult:
    """Continuous analogue of §4b: does aging change some directions more
    than the typical direction?

    For each donor ``d`` and each reference direction ``k`` (cosine
    matching to ``ref_op``, the pooled-young operator),
    ``r_dk = log(|λ_dk| / |λ_ref,k|)``. ``β_k`` = slope of ``r_dk`` on
    age. Statistic ``S = max_k |β_k − median(β)|``: ≈ 0 when every
    direction changes at the same rate with age (isotropic), large when
    one direction changes faster. Null: ``n_perm`` permutations of age
    across donors. Pass: ``S > q95(null)``.
    """
    ages = np.asarray(ages, dtype=np.float64)
    R = []
    for op in donor_ops:
        md = match_directions(ref_op, op)
        with np.errstate(divide="ignore", invalid="ignore"):
            R.append(np.log(np.abs(md.eig_b) / np.abs(md.eig_a)))
    R = np.asarray(R)
    keep = np.all(np.isfinite(R), axis=0)
    R = R[:, keep]
    if R.shape[1] < 3 or len(ages) < 4:
        return TrendResult(float("nan"), np.array([]), float("nan"), False,
                           float("nan"), {"n_directions": int(R.shape[1])})

    def stat(a):
        b = _slopes(R, a)
        return float(np.max(np.abs(b - np.median(b))))

    s_obs = stat(ages)
    null = np.array([stat(a) for a in _perm_ages(ages, n_perm, seed, strata)])
    q95 = float(np.percentile(null, 95))
    beta = _slopes(R, ages)
    return TrendResult(s_obs, null, q95, bool(s_obs > q95),
                       float((1 + np.sum(null >= s_obs)) / (1 + len(null))),
                       {"beta": beta, "median_beta": float(np.median(beta)),
                        "n_directions": int(R.shape[1])})


def anticipation_trend(donor_centroids: Sequence[np.ndarray],
                       ages: Sequence[float], ref_op: np.ndarray,
                       n_perm: int = 1000, seed: int = 0,
                    strata: Optional[Sequence] = None) -> TrendResult:
    """Continuous analogue of §5: does the donor centroid move with age
    along the young soft axis?

    ``u`` = eigenvector of the largest (least negative) eigenvalue of
    ``ref_op``. ``b`` = per-dimension slope of donor centroids on age.
    Statistic ``|cos(u, b)|``. Null: permutations of age across donors.
    Pass: statistic > q95(null).
    """
    ages = np.asarray(ages, dtype=np.float64)
    C = np.asarray(donor_centroids, dtype=np.float64)
    ev, V = np.linalg.eigh(0.5 * (ref_op + ref_op.T))
    u = V[:, int(np.argmax(ev))]

    def stat(a):
        b = _slopes(C, a)
        return abs(float(u @ b) / (np.linalg.norm(b) + 1e-12))

    s_obs = stat(ages)
    null = np.array([stat(a) for a in _perm_ages(ages, n_perm, seed, strata)])
    q95 = float(np.percentile(null, 95))
    return TrendResult(s_obs, null, q95, bool(s_obs > q95),
                       float((1 + np.sum(null >= s_obs)) / (1 + len(null))),
                       {"u_ref": u, "slope_vector": _slopes(C, ages)})


def stiffness_trend(donor_ops: Sequence[np.ndarray], ages: Sequence[float],
                    n_perm: int = 1000, seed: int = 0,
                    strata: Optional[Sequence] = None) -> TrendResult:
    """Slope of ``log median |λ|`` of the donor operators on age, with a
    two-sided age-permutation p. Negative slope = softening with age.
    ``passes`` = two-sided p ≤ 0.05. Reported beside the outcome header
    (PREREG_aging_hlca §6), not a header gate.
    """
    ages = np.asarray(ages, dtype=np.float64)
    y = np.array([np.log(np.median(np.abs(np.linalg.eigvalsh(op))))
                  for op in donor_ops])[:, None]
    s_obs = float(_slopes(y, ages)[0])
    null = np.array([float(_slopes(y, a)[0]) for a in _perm_ages(ages, n_perm, seed, strata)])
    p = float((1 + np.sum(np.abs(null) >= abs(s_obs))) / (1 + len(null)))
    return TrendResult(s_obs, null, float(np.percentile(np.abs(null), 95)),
                       bool(p <= 0.05), p, {"log_median_abs_lambda": y[:, 0]})
