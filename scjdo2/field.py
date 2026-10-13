"""Step 5 — neighbourhood Jacobian operator field (Phase B minimum).

For each cell take its ``k`` nearest neighbours in latent space; fit the local
Jacobian by least-squares such that

    v_i − v_c  ≈  J_c (z_i − z_c)    for each neighbour i of cell c

Phase B uses equal weights across the ball (no NB likelihood yet); Phase C
switches to a likelihood weight derived from :mod:`scjdo2.fisher`. The
smoothness penalty between neighbouring ``J_c`` and the gene-space variant are
Phase C work; here we fit each cell independently in latent space.

Cells with fewer than ``k`` neighbours within the configured radius are
**marked** (``reason == 'not_in_ball'``), never silently filled.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from typing import Any, Literal, Optional, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import gammaln as _lgamma
from sklearn.neighbors import NearestNeighbors


def smooth_jacobians(
    J: np.ndarray,
    z: np.ndarray,
    k: int = 30,
    bandwidth: Optional[float] = None,
    include_self: bool = True,
) -> np.ndarray:
    """Post-hoc Gaussian-kernel-weighted smoothing of a per-cell Jacobian field.

    For each cell ``c`` the smoothed Jacobian is::

        J̃_c = Σ_{i ∈ N_k(c)} w_{ci} J_i / Σ w_{ci},
        w_{ci} = exp( - ||z_c - z_i||² / (2 h²) )

    where ``N_k(c)`` is the k nearest neighbours of ``z_c`` in ``z`` (plus the
    cell itself when ``include_self=True``). The bandwidth ``h`` defaults to
    the median neighbour distance across all balls (a conservative Silverman-
    style choice that scales with the latent geometry).

    Parameters
    ----------
    J : (n, d, d) ndarray
        Per-cell Jacobians to smooth. NaN rows are kept as NaN in the output
        and are excluded from any neighbour's weighted sum.
    z : (n, d) ndarray
        Latent positions used to define neighbourhoods.
    k : int, default 30
        Number of neighbours (excluding self) per cell. Clamped to ``n - 1``.
    bandwidth : float or None, default None
        Gaussian kernel bandwidth. ``None`` → median neighbour distance over
        all balls; non-positive values fall back to the median too.
    include_self : bool, default True
        Include the cell itself (weight 1.0 at zero distance) in the sum.

    Returns
    -------
    J_smooth : (n, d, d) ndarray
        Smoothed Jacobians, same shape as ``J``.
    """
    J = np.asarray(J, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    n = z.shape[0]
    if J.shape[0] != n:
        raise ValueError(f"J has n={J.shape[0]} rows but z has n={n}")
    if J.ndim != 3 or J.shape[1] != J.shape[2]:
        raise ValueError(f"J must be (n, d, d); got {J.shape}")
    k = max(1, min(int(k), n - 1))

    nn = NearestNeighbors(n_neighbors=k + 1).fit(z)
    dists, idxs = nn.kneighbors(z, return_distance=True)
    # drop self (index 0)
    nbr_d = dists[:, 1:]
    nbr_i = idxs[:, 1:]

    if bandwidth is None or not np.isfinite(bandwidth) or bandwidth <= 0:
        h = float(np.median(nbr_d))
        if not np.isfinite(h) or h <= 0:
            h = 1.0
    else:
        h = float(bandwidth)
    inv2h2 = 1.0 / (2.0 * h * h)

    row_finite = np.all(np.isfinite(J.reshape(n, -1)), axis=1)
    out = np.full_like(J, np.nan)
    for c in range(n):
        nbrs = nbr_i[c]
        ds = nbr_d[c]
        mask = row_finite[nbrs]
        nbrs = nbrs[mask]
        ds = ds[mask]
        if include_self and row_finite[c]:
            nbrs = np.concatenate([[c], nbrs])
            ds = np.concatenate([[0.0], ds])
        if nbrs.size == 0:
            continue
        w = np.exp(-ds * ds * inv2h2)
        s = float(w.sum())
        if not np.isfinite(s) or s <= 0:
            continue
        out[c] = np.tensordot(w / s, J[nbrs], axes=(0, 0))
    return out


def fit_jacobian_tangent_projectors(
    z: np.ndarray,
    rate_model,
    size_factor: Optional[np.ndarray] = None,
    n_perm: int = 30,
    quantile: float = 0.95,
    max_rank: Optional[int] = None,
    seed: int = 0,
) -> dict:
    """Per-cell η-tangent projector from the Fisher-weighted decoder Jacobian,
    with a permutation-null rank truncation.

    For cell ``c`` the Fisher-weighted decoder Jacobian is::

        A_c  =  diag(sqrt( w_c ))  ·  ∇η(z_c)         (G × d)

    where ``w_cg = μ_cg · θ_g / (μ_cg + θ_g)`` is the per-gene NB precision
    at that cell. Under the NB precision metric, each entry of ``A_c`` has
    unit noise variance when the direction is signal-free, so the singular
    values of ``A_c`` live on the SNR scale.

    Truncation (docs/opspace_figure_spec.md §9 fix (c), updated). For each
    cell we build a **permutation null** by shuffling the ``G·d`` entries
    of ``A_c`` ``n_perm`` times and recording the largest singular value
    of each shuffle. The noise floor is the ``quantile`` (default 0.95)
    of that null distribution; the kept rank is::

        r_c  =  |{ j : σ_j(A_c)  >  floor_c }|   (clamped to [1, min(d, G, max_rank)])

    The projector ``P_c`` is the ``r_c × G`` matrix whose rows are the
    top-``r_c`` **left** singular vectors of ``A_c``.

    The permutation null is the recommendation in the latest spec iteration;
    it is scale-aware (uses the actual ``A_c`` entries) and does not require
    an analytical Marchenko-Pastur edge. ``n_perm=30`` is a bias/variance
    sweet spot and the per-cell cost is dominated by the main SVD anyway.

    Rationale. ``image(∇η(z_c))`` is at most ``d``-dim, but on a decoder
    trained on PCA(d) of an intrinsically-r-dim process (r ≪ d) only ``r``
    of those directions carry decoder-visible signal above noise. The
    remaining ``d - r`` are learned-in noise — Phase C's per-gene weighted
    ridge is dominated by its regulariser there, which inflates the lead
    eigenvalue of the ``(d, d)`` block. The permutation null picks exactly
    the ``r`` significant singular directions, cell by cell, from the
    model itself (no external rank input).

    Returns a dict (shape-compatible with the previous version):
        * ``rank`` : ``(n,)`` int — per-cell retained rank ``r_c`` (variable).
        * ``projector`` : list of ``(r_c, G)`` float64 arrays — top-``r_c``
          left singular directions of the Fisher-weighted ``∇η(z_c)`` as
          rows.
        * ``eigvals`` : list of ``(min(G, d),)`` singular-value squares
          (descending) of ``A_c``.
        * ``noise_floor`` : ``(n,)`` float — the per-cell σ threshold.
        * ``method`` : ``"jacobian_svd_perm"``.
        * ``n_perm``, ``quantile`` : config echo.
    """
    z = np.asarray(z, dtype=np.float64)
    n, d = z.shape
    g_eta = rate_model.grad_eta(z).astype(np.float64)          # (n, G, d)
    w = rate_model.nb_weights(z, size_factor=size_factor).astype(np.float64)  # (n, G)
    G = g_eta.shape[1]
    cap = int(max_rank) if max_rank is not None else min(d, G)
    cap = max(1, min(cap, min(d, G)))
    rng = np.random.default_rng(int(seed))

    rank_arr = np.ones(n, dtype=np.int32)
    projectors: list[np.ndarray] = []
    sv_list: list[np.ndarray] = []
    thr_arr = np.zeros(n, dtype=np.float64)
    for c in range(n):
        A = (np.sqrt(np.clip(w[c], 0.0, None))[:, None] * g_eta[c])  # (G, d)
        try:
            U, S, _ = np.linalg.svd(A, full_matrices=False)    # U (G, d), S (d,)
        except np.linalg.LinAlgError:
            projectors.append(np.eye(G)[:cap])
            sv_list.append(np.zeros(min(d, G)))
            rank_arr[c] = cap
            continue
        # Permutation null on the entries of A_c.
        flat = A.ravel()
        null_max = np.empty(int(n_perm), dtype=np.float64)
        for p in range(int(n_perm)):
            perm = rng.permutation(flat.size)
            Ap = flat[perm].reshape(G, d)
            Sp = np.linalg.svd(Ap, compute_uv=False)
            null_max[p] = float(Sp[0])
        thr = float(np.quantile(null_max, float(quantile)))
        r_c = int(np.sum(S > thr))
        r_c = max(1, min(r_c, cap))
        P_c = U[:, :r_c].T.astype(np.float64)                  # (r_c, G)
        projectors.append(P_c)
        sv_list.append((S ** 2).astype(np.float64))
        rank_arr[c] = r_c
        thr_arr[c] = thr
    return {
        "rank": rank_arr,
        "projector": projectors,
        "eigvals": sv_list,
        "noise_floor": thr_arr,
        "method": "jacobian_svd_perm",
        "n_perm": int(n_perm),
        "quantile": float(quantile),
    }


def smooth_jacobians_sparse(
    J_list: Sequence[csr_matrix],
    z: np.ndarray,
    k: int = 30,
    bandwidth: Optional[float] = None,
    include_self: bool = True,
) -> list[csr_matrix]:
    """Gaussian-kernel-weighted smoothing of per-cell CSR Jacobians.

    Sparse analogue of :func:`smooth_jacobians`. For each cell ``c``::

        J̃_c = Σ_{i ∈ N_k(c)∪{c}} w_{ci} J_i / Σ w_{ci}

    with ``w_{ci} = exp(-||z_c - z_i||² / (2 h²))`` and ``h`` the median
    neighbour distance (or ``bandwidth`` if explicit). The resulting
    CSR has support equal to the union of neighbour live-column indices.

    Empty cells (``nnz == 0``) are skipped — the output keeps an empty
    CSR at that index. All ``J_i`` must share the same shape.
    """
    if not J_list:
        return []
    z = np.asarray(z, dtype=np.float64)
    n = len(J_list)
    if z.shape[0] != n:
        raise ValueError(f"J_list has {n} entries but z has {z.shape[0]}")
    shape = J_list[0].shape
    for i, Jc in enumerate(J_list):
        if Jc.shape != shape:
            raise ValueError(f"J_list[{i}].shape {Jc.shape} != {shape}")
    k = max(1, min(int(k), n - 1))

    nn_ = NearestNeighbors(n_neighbors=k + 1).fit(z)
    dists, idxs = nn_.kneighbors(z, return_distance=True)
    nbr_d = dists[:, 1:]
    nbr_i = idxs[:, 1:]

    if bandwidth is None or not np.isfinite(bandwidth) or bandwidth <= 0:
        h = float(np.median(nbr_d))
        if not np.isfinite(h) or h <= 0:
            h = 1.0
    else:
        h = float(bandwidth)
    inv2h2 = 1.0 / (2.0 * h * h)

    out: list[csr_matrix] = []
    for c in range(n):
        nbrs = list(nbr_i[c])
        ds = list(nbr_d[c])
        if include_self:
            nbrs = [c] + nbrs
            ds = [0.0] + ds
        total = csr_matrix(shape, dtype=np.float64)
        wsum = 0.0
        for nb, dist in zip(nbrs, ds):
            Jn = J_list[nb]
            if Jn.nnz == 0:
                continue
            w = float(np.exp(-dist * dist * inv2h2))
            total = total + Jn * w
            wsum += w
        if wsum > 0 and total.nnz > 0:
            total = total * (1.0 / wsum)
            total.eliminate_zeros()
        out.append(total.tocsr())
    return out


REASON_OK = 0
REASON_NO_COUNTS = 1
REASON_NO_VARIATION = 2
REASON_NOT_IN_BALL = 3
REASON_CAPPED = 4

_REASON_NAMES = {
    REASON_OK: "ok",
    REASON_NO_COUNTS: "no_counts",
    REASON_NO_VARIATION: "no_variation",
    REASON_NOT_IN_BALL: "not_in_ball",
    REASON_CAPPED: "capped",
}


@dataclass
class FieldConfig:
    k: int | Literal["auto"] = 50
    radius: Optional[float] = None
    variation_threshold: float = 1e-8
    ridge: float = 1e-4
    seed: int = 0


class JacobianField:
    """Per-cell local Jacobian by ridge regression on the kNN ball."""

    def __init__(self, config: Optional[FieldConfig] = None, **overrides):
        self.config = config or FieldConfig()
        for k, v in overrides.items():
            if not hasattr(self.config, k):
                raise AttributeError(f"FieldConfig has no field {k!r}")
            setattr(self.config, k, v)
        self._J: Optional[list[np.ndarray]] = None
        self._reason: Optional[np.ndarray] = None  # (n_cells,) int code
        self._z: Optional[np.ndarray] = None

    def fit(self, z: np.ndarray, v: np.ndarray) -> "JacobianField":
        """Fit Jacobians from positions ``z`` and velocities ``v`` (both n×d)."""
        z = np.asarray(z, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        n, d = z.shape
        if v.shape != z.shape:
            raise ValueError(f"v {v.shape} must match z {z.shape}")

        k = int(self.config.k) if self.config.k != "auto" else min(50, max(5, int(np.sqrt(n))))
        if k >= n:
            k = max(5, n - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(z)
        dists, idxs = nn.kneighbors(z, return_distance=True)
        # drop self (index 0 is the cell itself after fit on same data)
        dists = dists[:, 1:]
        idxs = idxs[:, 1:]

        J_list: list[np.ndarray] = []
        reason = np.full(n, REASON_OK, dtype=np.int8)
        ridge = float(self.config.ridge)
        max_r = float("inf") if self.config.radius is None else float(self.config.radius)

        var_tol = float(self.config.variation_threshold)
        for c in range(n):
            nbrs = idxs[c]
            if self.config.radius is not None:
                within = dists[c] <= max_r
                nbrs = nbrs[within]
            if nbrs.size < max(3, min(k, d + 1)):
                J_list.append(np.full((d, d), np.nan))
                reason[c] = REASON_NOT_IN_BALL
                continue

            dz = z[nbrs] - z[c]        # (k, d)
            dv = v[nbrs] - v[c]        # (k, d)

            # Drop ball if latent variation is below threshold.
            if np.max(np.linalg.norm(dz, axis=1)) < var_tol:
                J_list.append(np.full((d, d), np.nan))
                reason[c] = REASON_NO_VARIATION
                continue

            # Ridge-regularized least squares: J @ dz.T ≈ dv.T  →  J = dv.T @ dz @ (dz.T @ dz + ridge I)^{-1}
            G = dz.T @ dz + ridge * np.eye(d)
            try:
                A = np.linalg.solve(G, dz.T @ dv)      # (d, d) — solves dz' dz J' = dz' dv
            except np.linalg.LinAlgError:
                J_list.append(np.full((d, d), np.nan))
                reason[c] = REASON_NO_VARIATION
                continue
            J_list.append(A.T)       # since J @ dz_i ≈ dv_i  (column convention)

        self._J = J_list
        self._reason = reason
        self._z = z
        self._k = k
        return self

    # ------------------------------------------------------------ accessors
    def J_dense(self) -> np.ndarray:
        """Stack per-cell Jacobians into a dense ``(n, d, d)`` tensor."""
        if self._J is None:
            raise RuntimeError("fit() must be called first")
        return np.stack(self._J, axis=0)

    def J_sparse(self) -> list[csr_matrix]:
        """Return per-cell CSR Jacobians (dense for now; sparsification is Phase C)."""
        if self._J is None:
            raise RuntimeError("fit() must be called first")
        return [csr_matrix(j) for j in self._J]

    def reasons(self) -> np.ndarray:
        if self._reason is None:
            raise RuntimeError("fit() must be called first")
        return self._reason

    def reason_summary(self) -> dict[str, float]:
        if self._reason is None:
            raise RuntimeError("fit() must be called first")
        n = self._reason.size
        return {
            _REASON_NAMES[c]: float(np.mean(self._reason == c))
            for c in sorted(_REASON_NAMES)
        }

    # ------------------------------------------------------------- null test
    def smooth_field(
        self,
        z: Optional[np.ndarray] = None,
        k: int = 30,
        bandwidth: Optional[float] = None,
        include_self: bool = True,
    ) -> np.ndarray:
        """Return the Gaussian-kernel-smoothed Jacobian field over the kNN ball
        in latent space.

        Delegates to the module-level ``smooth_jacobians`` on this field's
        fitted ``J_dense()`` and the latent positions used in ``fit``.
        """
        if self._J is None or self._z is None:
            raise RuntimeError("fit() must be called first")
        z_use = self._z if z is None else np.asarray(z, dtype=np.float64)
        return smooth_jacobians(
            self.J_dense(), z_use, k=k, bandwidth=bandwidth, include_self=include_self,
        )

    def permutation_null(
        self,
        z: np.ndarray,
        v: np.ndarray,
        n_perm: int = 50,
        subset_idx: Optional[np.ndarray] = None,
        seed: int = 0,
    ) -> np.ndarray:
        """Per-cell permutation null for ``Re(λ_max)``.

        For each cell ``c`` in ``subset_idx`` (default: all cells), rebuild its
        kNN ball with the same ``k`` as ``fit``, then draw ``n_perm`` shuffles
        of which neighbour's ``v`` pairs with which neighbour's ``z``, refit
        the ridge regression and record ``Re`` of the dominant eigenvalue.

        Returns an array of shape ``(len(subset_idx), n_perm)``; cells that
        are not in the ball (or whose ball is degenerate) get a row of NaNs.
        """
        z = np.asarray(z, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        n, d = z.shape
        rng = np.random.default_rng(int(seed))
        k = int(self.config.k) if self.config.k != "auto" else min(50, max(5, int(np.sqrt(n))))
        if k >= n:
            k = max(5, n - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(z)
        dists, idxs = nn.kneighbors(z, return_distance=True)
        idxs = idxs[:, 1:]
        ridge = float(self.config.ridge)

        if subset_idx is None:
            subset_idx = np.arange(n)
        else:
            subset_idx = np.asarray(subset_idx, dtype=np.int64)

        out = np.full((subset_idx.size, int(n_perm)), np.nan, dtype=np.float64)
        for row, c in enumerate(subset_idx):
            nbrs = idxs[c]
            if nbrs.size < max(3, min(k, d + 1)):
                continue
            dz = z[nbrs] - z[c]
            dv = v[nbrs] - v[c]
            if np.max(np.linalg.norm(dz, axis=1)) < float(self.config.variation_threshold):
                continue
            G = dz.T @ dz + ridge * np.eye(d)
            # Precompute the (G^{-1} dz.T) factor; the permutation only affects
            # the right-hand side dv.T so (G^{-1} dz.T @ dv_perm.T) is a cheap
            # matmul.
            try:
                dz_proj = np.linalg.solve(G, dz.T)      # (d, k)
            except np.linalg.LinAlgError:
                continue
            for p in range(int(n_perm)):
                perm = rng.permutation(dz.shape[0])
                dv_p = dv[perm]
                A = dz_proj @ dv_p                      # (d, d) — solves G J' = dz' dv
                Jc = A.T
                try:
                    evals = np.linalg.eigvals(Jc)
                except np.linalg.LinAlgError:
                    continue
                out[row, p] = float(np.max(evals.real))
        return out


# ============================================================ Step 5 gene-space
#
# Per-cell gene-space Jacobian by per-gene weighted ridge regression in
# η-coordinates (docs/opspace_figure_spec.md §8.3).
#
# For cell c with k latent neighbours ``N_k(c)`` and η-coord drift
# ``dη/dt = ∇_z η(z_c) · f(z_c)`` from the NB rate model::
#
#     Δη_i = η_i - η_c                                             (k × G)
#     y_i  = (dη/dt)_i - (dη/dt)_c                                 (k × G)
#     w_ig = μ_ig · θ_g / (μ_ig + θ_g)      (NB precision of η)    (k × G)
#     live_c = { g : var_i(Δη_ig) > tol }                          (⊆ 1..G)
#     X_c  = Δη[:, live_c]                                         (k × L)
#     For each gene row g:
#         a_g = (X^T diag(w[:,g]) X + λ I_L)^{-1} X^T diag(w[:,g]) y[:,g]
#     J_c = CSR with rows span 1..G, columns live_c, entries a_g[j].
#
# Ridge λ is chosen globally by 5-fold neighbour cross-validation. The
# default criterion is the **held-out NB log-likelihood of predicted
# Δη** on neighbour counts (Option A in R14): for a held-out neighbour i,
# the ridge-predicted velocity shift ``v̂_i = a · (η_i - η_c)`` is used as a
# log-rate shift from the base cell c, so the predicted log-rate at i is
# ``log μ̂_i,g = η_c,g + v̂_i,g + log s_i``, and NB NLL is evaluated on the
# observed counts ``c_i,g`` with the rate model's per-gene θ_g. We also
# record the weighted-MSE score for comparison. Weighted MSE is the local
# Gaussian approximation of the NB NLL — on weak equilibrated signal it
# pushes λ to the grid maximum regardless of fit quality, so NB NLL on
# counts is the primary selector. See docs/opspace_figure_spec.md §8.3,
# V2_PLAN R14, and the Phase C gate report for the comparison table.
#
# Storage per spec §1 and §8.3:
#     J_c as CSR, rows span all G, columns restricted to live_c with a
#     live-column index array per cell. Latent J moves to adata.obsm.


@dataclass
class GeneFieldConfig:
    k: int = 30
    variation_threshold: float = 1e-6     # η-variance threshold for live genes
    ridge_lambdas: tuple = (1e-3, 1e-2, 1e-1, 1.0)
    cv_folds: int = 5
    cv_sample_cells: int = 200            # sample for λ selection; -1 = all
    cv_metric: Literal["nb_nll", "weighted_mse"] = "nb_nll"
    seed: int = 0
    # Anchor-based fitting knobs (see fit_gene_space_jacobians_anchored).
    n_anchors: int = 300                   # k-means centroids in latent z
    anchor_top_k: int = 10                 # anchors used in each cell's lift
    anchor_bandwidth: Optional[float] = None  # Gaussian kernel σ (None → auto)
    anchor_batch: int = 25                 # anchors solved per batched call
    anchor_max_live: int = 300             # cap per-anchor live-cols to the top-N
                                           # genes by NB-precision-weighted η
                                           # variance (signal-to-noise on the
                                           # log-rate scale). Genes above the
                                           # variation threshold but below this
                                           # cap are tagged REASON_CAPPED, not
                                           # REASON_NO_VARIATION — the ledger
                                           # distinguishes "no SNR here" from
                                           # "SNR there but the cap wouldn't
                                           # fit it". No must-keep gene lists.


def fit_gene_space_jacobians(
    z: np.ndarray,                        # (n, d) latent positions
    f: np.ndarray,                        # (n, d) latent drift f(z)
    rate_model,                           # scjdo2.rates.GeneRateModel (fitted)
    counts: Optional[np.ndarray] = None,  # (n, G) raw counts, for NB-NLL CV
    size_factor: Optional[np.ndarray] = None,
    config: Optional[GeneFieldConfig] = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Fit per-cell gene-space Jacobians by per-gene weighted ridge in η.

    Parameters
    ----------
    z, f
        Latent positions and drift velocities, both ``(n, d)``.
    rate_model
        Fitted :class:`scjdo2.rates.GeneRateModel`. Supplies ``η(z)``,
        ``dη/dt`` by autodiff, and NB precision weights ``w_cg``.
    size_factor
        Per-cell library-size factor; defaults to the one the rate model
        was fit on (``s_c = lib / median_lib``). Passed through to
        ``rate_model.nb_weights`` so weights are in the fitted units.

    Returns
    -------
    dict with keys
        * ``J`` : list of ``(G, G)`` ``csr_matrix``, one per cell.
          Columns are restricted to live genes; dense Gs stay sparse.
        * ``live_cols`` : list of ``(L_c,)`` int arrays — the live-gene
          index per cell (one array per cell).
        * ``reason`` : ``(n,)`` int8, one of ``REASON_OK``,
          ``REASON_NO_VARIATION``, ``REASON_NOT_IN_BALL``.
        * ``ridge_lambda`` : the single λ chosen by the primary CV
          metric (``cv_metric``). Fit at this λ, same for every cell.
        * ``ridge_lambda_nll`` / ``ridge_lambda_mse`` : the two λ picks
          side by side (either may be ``nan`` if the corresponding
          score was non-finite throughout).
        * ``lambda_scores`` : dict ``{λ: score}`` from the primary CV
          metric; ``lambda_scores_nll`` and ``lambda_scores_mse`` carry
          the two score curves separately.
        * ``cv_metric`` : the selector used for ``ridge_lambda``.
        * ``k``, ``variation_threshold`` : config echo.

    Notes
    -----
    Memory. The per-cell batched solve builds a ``(G, L_c, L_c)`` normal-
    equation stack; for ``G=2000`` and ``L_c=500`` that is 4 GB per cell.
    The ``variation_threshold`` + the η-neighbourhood both bound ``L_c``
    much lower than ``G`` in practice (on the pitchfork synthetic
    ``G=200``, ``L_c~50``). Real-data runs with large ``G`` must enable
    column chunking (future knob).
    """
    cfg = config or GeneFieldConfig()
    z = np.asarray(z, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    n, d = z.shape
    if f.shape != z.shape:
        raise ValueError(f"f {f.shape} must match z {z.shape}")
    k = max(3, min(int(cfg.k), n - 1))
    metric = str(getattr(cfg, "cv_metric", "nb_nll"))
    if metric not in ("nb_nll", "weighted_mse"):
        raise ValueError(f"cv_metric must be 'nb_nll' or 'weighted_mse', got {metric!r}")

    eta = rate_model.eta(z).astype(np.float64)              # (n, G)
    d_eta = rate_model.rate_along_eta(z, f).astype(np.float64)  # (n, G)
    w = rate_model.nb_weights(z, size_factor=size_factor).astype(np.float64)  # (n, G)
    G = eta.shape[1]

    # NB-NLL path needs counts and θ_g. Default size_factor matches the
    # rate model's fit convention (lib / median_lib) when counts are
    # provided; else size_factor=1 everywhere, consistent with
    # nb_weights(size_factor=None).
    counts_arr: Optional[np.ndarray] = None
    log_s: Optional[np.ndarray] = None
    log_theta: Optional[np.ndarray] = None
    if metric == "nb_nll":
        if counts is None:
            raise ValueError("cv_metric='nb_nll' requires counts (n, G)")
        counts_arr = np.asarray(counts, dtype=np.float64)
        if counts_arr.shape != (n, G):
            raise ValueError(f"counts {counts_arr.shape} must match (n, G)=({n},{G})")
        if size_factor is None:
            lib = counts_arr.sum(axis=1)
            med = float(np.median(lib[lib > 0])) if np.any(lib > 0) else 1.0
            s_vec = (lib / med) if med > 0 else np.ones(n, dtype=np.float64)
        else:
            s_vec = np.asarray(size_factor, dtype=np.float64)
        log_s = np.log(np.clip(s_vec, 1e-8, None))
        theta = np.asarray(rate_model.theta(), dtype=np.float64)
        log_theta = np.log(np.clip(theta, 1e-6, 1e8))

    nn = NearestNeighbors(n_neighbors=k + 1).fit(z)
    _, idxs = nn.kneighbors(z, return_distance=True)
    idxs = idxs[:, 1:]                                      # drop self

    # --------- global λ selection by k-fold neighbour CV on sampled cells
    rng = np.random.default_rng(int(cfg.seed))
    sample_n = n if cfg.cv_sample_cells < 0 else min(int(cfg.cv_sample_cells), n)
    sample_idx = rng.choice(n, size=sample_n, replace=False)
    lambda_scores_mse: dict[float, float] = {}
    lambda_scores_nll: dict[float, float] = {}

    def _ridge_batched(X_tr, W_tr, y_tr, lam, L):
        """Solve the per-gene weighted ridge for every gene row in one shot.

        * Primal ``(L, L)`` solve when ``L <= kt`` (overdetermined).
        * Dual ``(kt, kt)`` solve via Woodbury when ``L > kt``
          (underdetermined). Much faster on pitchfork-like data where
          every gene is live (L ≈ G).
        """
        kt = X_tr.shape[0]
        if L <= kt:
            A = np.einsum("ig,ip,iq->gpq", W_tr, X_tr, X_tr)
            A = A + lam * np.eye(L)[None, :, :]
            b = np.einsum("ig,ip,ig->gp", W_tr, X_tr, y_tr)
            return np.linalg.solve(A, b[..., None]).squeeze(-1)       # (G, L)
        # Dual form: a_g = X^T diag(w_g) (X X^T diag(w_g) + λ I_kt)^{-1} y_g
        # Build the (G, kt, kt) kernel K_g = X diag(w_g) X^T.
        K = np.einsum("ip,ig,jp->gij", X_tr, W_tr, X_tr)              # (G, kt, kt)
        K = K + lam * np.eye(kt)[None, :, :]
        try:
            u = np.linalg.solve(K, y_tr.T[..., None]).squeeze(-1)      # (G, kt)
        except np.linalg.LinAlgError:
            raise
        # a_g = (X^T diag(w_g)) u_g
        a = np.einsum("ip,ig,gi->gp", X_tr, W_tr, u)                   # (G, L)
        return a

    # Precompute per-cell neighbour slices used by both metrics.
    for lam in cfg.ridge_lambdas:
        tot_mse = 0.0
        cnt_mse = 0
        tot_nll = 0.0
        cnt_nll = 0
        for c in sample_idx:
            nbrs = idxs[c]
            X_full = eta[nbrs] - eta[c]                     # (k, G)
            y_full = d_eta[nbrs] - d_eta[c]                 # (k, G)
            live = np.where(np.var(X_full, axis=0) > cfg.variation_threshold)[0]
            if live.size < 2 or k < max(5, cfg.cv_folds + 1):
                continue
            X = X_full[:, live]                             # (k, L)
            W = w[nbrs]                                     # (k, G)
            fold = rng.integers(0, cfg.cv_folds, size=k)
            fold_mse = 0.0
            fold_mse_cnt = 0
            fold_nll = 0.0
            fold_nll_cnt = 0
            for fo in range(cfg.cv_folds):
                tr = fold != fo
                te = ~tr
                if tr.sum() < 2 or te.sum() == 0:
                    continue
                Xt = X[tr]
                Wt = W[tr]
                yt = y_full[tr]
                try:
                    a = _ridge_batched(Xt, Wt, yt, lam, live.size)
                except np.linalg.LinAlgError:
                    continue
                Xe = X[te]
                We = W[te]
                ye = y_full[te]
                yhat = Xe @ a.T                             # (ke, G)
                # --- weighted MSE
                resid2 = (yhat - ye) ** 2
                w_sum = float(We.sum())
                if w_sum > 0 and np.isfinite(w_sum):
                    fold_mse += float((We * resid2).sum()) / w_sum
                    fold_mse_cnt += 1
                # --- NB NLL on held-out neighbour counts
                if metric == "nb_nll":
                    te_nbrs = nbrs[te]                      # (ke,)
                    log_mu = eta[c][None, :] + yhat + log_s[te_nbrs][:, None]  # (ke, G)
                    # Clip log_mu for numerical safety (exp underflow/overflow).
                    log_mu = np.clip(log_mu, -30.0, 30.0)
                    mu_hat = np.exp(log_mu)                 # (ke, G)
                    # NB log-pmf (drop lgamma(c+1) const in c):
                    #   log p = lgamma(c+θ) - lgamma(θ) + θ·(log θ - log(θ+μ))
                    #           + c·(log μ - log(θ+μ))
                    c_i = counts_arr[te_nbrs]               # (ke, G)
                    theta_e = np.exp(log_theta)[None, :]    # (1, G)
                    log_tpm = np.log(theta_e + mu_hat)
                    lp = (
                        _lgamma(c_i + theta_e) - _lgamma(theta_e)
                        + theta_e * (log_theta[None, :] - log_tpm)
                        + c_i * (log_mu - log_tpm)
                    )
                    nll = -float(lp.mean())
                    if np.isfinite(nll):
                        fold_nll += nll
                        fold_nll_cnt += 1
            if fold_mse_cnt > 0 and np.isfinite(fold_mse):
                tot_mse += fold_mse / fold_mse_cnt
                cnt_mse += 1
            if fold_nll_cnt > 0 and np.isfinite(fold_nll):
                tot_nll += fold_nll / fold_nll_cnt
                cnt_nll += 1
        lambda_scores_mse[float(lam)] = (tot_mse / cnt_mse) if cnt_mse > 0 else float("inf")
        lambda_scores_nll[float(lam)] = (tot_nll / cnt_nll) if cnt_nll > 0 else float("inf")
        if verbose:
            msg = f"  [gene-field] λ={lam:g}  weighted MSE={lambda_scores_mse[float(lam)]:.4f}"
            if metric == "nb_nll":
                msg += f"  NB NLL={lambda_scores_nll[float(lam)]:.4f}"
            print(msg)
    lambda_mse = min(lambda_scores_mse, key=lambda l: lambda_scores_mse[l])
    if metric == "nb_nll" and any(np.isfinite(v) for v in lambda_scores_nll.values()):
        best_lambda = min(lambda_scores_nll, key=lambda l: lambda_scores_nll[l])
    else:
        best_lambda = lambda_mse
    lambda_scores = lambda_scores_nll if metric == "nb_nll" else lambda_scores_mse

    # --------- fit all cells at best λ, and compute the Fisher ledger
    # The Fisher info for the (g, j_live)-th entry of a_g is the (j, j)
    # diagonal of the ridge Hessian: H_{g, j, j} = Σ_i w_{i,g} X_{i,j}² + λ
    # (per docs/opspace_figure_spec.md §8.4 — "diagonal curvature of the
    # same ridge likelihood").
    J_list: list[csr_matrix] = []
    F_list: list[csr_matrix] = []           # same sparse layout as J
    reason_list: list[np.ndarray] = []      # per-cell (G,) int8
    live_cols: list[np.ndarray] = []
    reason = np.full(n, REASON_OK, dtype=np.int8)
    for c in range(n):
        nbrs = idxs[c]
        X_full = eta[nbrs] - eta[c]                         # (k, G)
        y_full = d_eta[nbrs] - d_eta[c]                     # (k, G)
        live = np.where(np.var(X_full, axis=0) > cfg.variation_threshold)[0]
        rsn = np.full(G, REASON_NO_VARIATION, dtype=np.int8)
        if live.size < 2:
            J_list.append(csr_matrix((G, G)))
            F_list.append(csr_matrix((G, G)))
            live_cols.append(np.zeros(0, dtype=np.int64))
            reason_list.append(rsn)
            reason[c] = REASON_NO_VARIATION
            continue
        X = X_full[:, live]                                 # (k, L)
        W = w[nbrs]                                         # (k, G)
        try:
            a = _ridge_batched(X, W, y_full, best_lambda, live.size)
        except np.linalg.LinAlgError:
            J_list.append(csr_matrix((G, G)))
            F_list.append(csr_matrix((G, G)))
            live_cols.append(np.zeros(0, dtype=np.int64))
            reason_list.append(rsn)
            reason[c] = REASON_NO_VARIATION
            continue
        # Fisher diagonal per entry: F[g, j_live] = Σ_i w[i,g] · X[i, j]² + λ
        F = (W.T @ (X ** 2)) + float(best_lambda)           # (G, L)

        # Build CSR with rows = all G, columns = live_cols[c]
        rows = np.repeat(np.arange(G, dtype=np.int64), live.size)
        cols = np.tile(live.astype(np.int64), G)
        J_c = csr_matrix((a.reshape(-1), (rows, cols)), shape=(G, G))
        F_c = csr_matrix((F.reshape(-1), (rows, cols)), shape=(G, G))
        J_list.append(J_c)
        F_list.append(F_c)
        live_cols.append(live.astype(np.int64))
        # Per-cell reason: ok on live genes, no_variation elsewhere.
        rsn[live] = REASON_OK
        reason_list.append(rsn)

    return {
        "J": J_list,
        "F": F_list,                                        # Fisher ledger
        "reason_per_cell": reason_list,                     # list of (G,) int8
        "live_cols": live_cols,
        "reason": reason,
        "ridge_lambda": float(best_lambda),
        "lambda_scores": lambda_scores,                     # primary metric
        "lambda_scores_mse": lambda_scores_mse,
        "lambda_scores_nll": lambda_scores_nll,
        "ridge_lambda_mse": float(lambda_mse),
        "ridge_lambda_nll": float(
            min(lambda_scores_nll, key=lambda l: lambda_scores_nll[l])
            if any(np.isfinite(v) for v in lambda_scores_nll.values())
            else float("nan")
        ),
        "cv_metric": metric,
        "k": int(k),
        "variation_threshold": float(cfg.variation_threshold),
    }


# ============================================================ Step 5 (anchor variant)
#
# Scale fix for real data: fitting the per-cell weighted ridge at every
# cell is O(n_cells · G · L²) with large constants, and on Paul15
# (n ≈ 2700, G = 2000) one-shot solves take over an hour. The anchor
# variant does three things:
#
#   1. Pick ``n_anchors`` cells by k-means on the latent embedding (each
#      centroid is replaced by its nearest actual cell so the anchor set
#      is cells, not synthetic points).
#   2. Fit the per-gene weighted ridge at anchors only, batched in
#      ``anchor_batch`` chunks (each chunk solves a stacked
#      ``(batch, G, L_max, L_max)`` tensor via ``np.linalg.solve``).
#   3. Interpolate ``J`` and the Fisher ledger ``F`` to every cell by a
#      Gaussian kernel over the anchor distances in latent space, keeping
#      the ``anchor_top_k`` nearest anchors per cell.
#
# ``obs["op_anchor"]`` stores the index of each cell's dominant anchor
# (nearest in z); ``obsm["anchor_weights"]`` is the sparse ``(n,
# n_anchors)`` kernel matrix, both for downstream auditing. The
# λ-selection still runs by held-out NB NLL / weighted MSE on sampled
# anchor neighbourhoods.


def _ridge_batched_live(
    X_tr: np.ndarray, W_tr: np.ndarray, y_tr: np.ndarray, lam: float, L: int,
) -> np.ndarray:
    """Per-gene weighted ridge over a single ball's live columns.

    Same math as the ``_ridge_batched`` closure inside
    :func:`fit_gene_space_jacobians`; lifted to module scope so the
    anchored variant can reuse it without duplicating the primal/dual
    switch.
    """
    kt = X_tr.shape[0]
    if L <= kt:
        A = np.einsum("ig,ip,iq->gpq", W_tr, X_tr, X_tr)
        A = A + lam * np.eye(L)[None, :, :]
        b = np.einsum("ig,ip,ig->gp", W_tr, X_tr, y_tr)
        return np.linalg.solve(A, b[..., None]).squeeze(-1)       # (G, L)
    K = np.einsum("ip,ig,jp->gij", X_tr, W_tr, X_tr) + lam * np.eye(kt)[None]
    u = np.linalg.solve(K, y_tr.T[..., None]).squeeze(-1)         # (G, kt)
    return np.einsum("ip,ig,gi->gp", X_tr, W_tr, u)               # (G, L)


def _kmeans_anchors(z: np.ndarray, n_anchors: int, seed: int = 0) -> np.ndarray:
    """Return cell indices (one per k-means centroid) spanning the latent
    manifold. Each centroid is replaced by its nearest actual cell so the
    anchors are rows of ``z``, not synthetic points.
    """
    from sklearn.cluster import KMeans

    n = z.shape[0]
    K = int(min(max(1, n_anchors), n))
    if K == n:
        return np.arange(n, dtype=np.int64)
    km = KMeans(n_clusters=K, random_state=int(seed), n_init=4).fit(z)
    nn = NearestNeighbors(n_neighbors=1).fit(z)
    _, idx = nn.kneighbors(km.cluster_centers_, return_distance=True)
    anchors = np.unique(idx.ravel())
    if anchors.size < K:
        # Pad with uniformly spaced fallbacks when centroids collide.
        extra = np.setdiff1d(np.arange(n), anchors)
        rng = np.random.default_rng(int(seed))
        pad = rng.choice(extra, size=K - anchors.size, replace=False)
        anchors = np.concatenate([anchors, pad])
    return np.sort(anchors.astype(np.int64))


def _ridge_batched_stacked(
    X_stack: np.ndarray,     # (B, k, L_max) — padded with zeros past L_a
    W_stack: np.ndarray,     # (B, k, G)
    y_stack: np.ndarray,     # (B, k, G)
    live_mask: np.ndarray,   # (B, L_max) bool, True where column j is live for anchor b
    lam: float,
) -> np.ndarray:
    """Solve ``B`` weighted ridges in one batched call.

    Returns ``a`` of shape ``(B, G, L_max)``. Columns past each anchor's
    ``L_a`` are zeroed out, matching the per-anchor variable-``L`` fits.
    """
    B, kt, L = X_stack.shape
    G = W_stack.shape[2]
    # A[b, g, p, q] = Σ_i W_stack[b, i, g] · X_stack[b, i, p] · X_stack[b, i, q]
    A = np.einsum("big,bip,biq->bgpq", W_stack, X_stack, X_stack, optimize=True)
    A = A + lam * np.eye(L)[None, None, :, :]
    # b[b, g, p] = Σ_i W_stack[b, i, g] · X_stack[b, i, p] · y_stack[b, i, g]
    rhs = np.einsum("big,bip,big->bgp", W_stack, X_stack, y_stack, optimize=True)
    try:
        a = np.linalg.solve(A, rhs[..., None]).squeeze(-1)       # (B, G, L_max)
    except np.linalg.LinAlgError:
        # Fall back per-anchor so one singular ball doesn't kill the batch.
        a = np.zeros((B, G, L), dtype=np.float64)
        for bi in range(B):
            try:
                a[bi] = np.linalg.solve(A[bi], rhs[bi][..., None]).squeeze(-1)
            except np.linalg.LinAlgError:
                continue
    # Zero out padded columns.
    a = a * live_mask[:, None, :]
    return a


def fit_gene_space_jacobians_anchored(
    z: np.ndarray,
    f: np.ndarray,
    rate_model,
    counts: Optional[np.ndarray] = None,
    size_factor: Optional[np.ndarray] = None,
    config: Optional[GeneFieldConfig] = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Anchored per-cell gene-space Jacobian + Fisher ledger (fast path).

    Fits the per-gene weighted ridge only at ``cfg.n_anchors`` k-means
    anchor cells and interpolates ``J`` / ``F`` to every cell by Gaussian
    kernel weights over the anchor distances in latent space. Returns the
    same dict shape as :func:`fit_gene_space_jacobians` plus:

    * ``anchor_ids`` : ``(n_anchors,)`` cell indices of the anchors.
    * ``op_anchor`` : ``(n_cells,)`` nearest-anchor id per cell.
    * ``anchor_weights`` : ``(n_cells, n_anchors)`` CSR Gaussian kernel.
    * ``anchor_bandwidth`` : the bandwidth used.
    * ``variant`` : ``"anchor"``.

    See the module header for the design rationale and the knobs on
    :class:`GeneFieldConfig` (``n_anchors``, ``anchor_top_k``,
    ``anchor_bandwidth``, ``anchor_batch``).
    """
    cfg = config or GeneFieldConfig()
    z = np.asarray(z, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    n, d = z.shape
    if f.shape != z.shape:
        raise ValueError(f"f {f.shape} must match z {z.shape}")
    k = max(3, min(int(cfg.k), n - 1))
    metric = str(getattr(cfg, "cv_metric", "nb_nll"))
    if metric not in ("nb_nll", "weighted_mse"):
        raise ValueError(f"cv_metric must be 'nb_nll' or 'weighted_mse', got {metric!r}")

    eta = rate_model.eta(z).astype(np.float64)              # (n, G)
    d_eta = rate_model.rate_along_eta(z, f).astype(np.float64)  # (n, G)
    w = rate_model.nb_weights(z, size_factor=size_factor).astype(np.float64)
    G = eta.shape[1]

    # Rate-model counts / log size factor / log theta for the NB NLL metric.
    counts_arr: Optional[np.ndarray] = None
    log_s: Optional[np.ndarray] = None
    log_theta: Optional[np.ndarray] = None
    if metric == "nb_nll":
        if counts is None:
            raise ValueError("cv_metric='nb_nll' requires counts (n, G)")
        counts_arr = np.asarray(counts, dtype=np.float64)
        if counts_arr.shape != (n, G):
            raise ValueError(f"counts {counts_arr.shape} must match (n, G)=({n},{G})")
        if size_factor is None:
            lib = counts_arr.sum(axis=1)
            med = float(np.median(lib[lib > 0])) if np.any(lib > 0) else 1.0
            s_vec = (lib / med) if med > 0 else np.ones(n, dtype=np.float64)
        else:
            s_vec = np.asarray(size_factor, dtype=np.float64)
        log_s = np.log(np.clip(s_vec, 1e-8, None))
        theta = np.asarray(rate_model.theta(), dtype=np.float64)
        log_theta = np.log(np.clip(theta, 1e-6, 1e8))

    # ---------- anchors + neighbour balls (per anchor)
    anchors = _kmeans_anchors(z, cfg.n_anchors, seed=cfg.seed)
    n_anchors = int(anchors.size)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(z)
    _, anchor_idxs = nn.kneighbors(z[anchors], return_distance=True)
    anchor_idxs = anchor_idxs[:, 1:]                        # drop self; (n_anchors, k)
    if verbose:
        print(f"  [gene-field anchor] n_anchors={n_anchors} (k-means on z)")

    # ---------- λ-CV on sampled anchors (reuses the per-anchor ridge loop)
    rng = np.random.default_rng(int(cfg.seed))
    sample_n = min(int(cfg.cv_sample_cells), n_anchors) if cfg.cv_sample_cells > 0 else n_anchors
    sample_idx = rng.choice(n_anchors, size=sample_n, replace=False)
    lambda_scores_mse: dict[float, float] = {}
    lambda_scores_nll: dict[float, float] = {}
    for lam in cfg.ridge_lambdas:
        tot_mse, cnt_mse = 0.0, 0
        tot_nll, cnt_nll = 0.0, 0
        for sa in sample_idx:
            a_cell = int(anchors[sa])
            nbrs = anchor_idxs[sa]
            X_full = eta[nbrs] - eta[a_cell]
            y_full = d_eta[nbrs] - d_eta[a_cell]
            live = np.where(np.var(X_full, axis=0) > cfg.variation_threshold)[0]
            if live.size < 2 or k < max(5, cfg.cv_folds + 1):
                continue
            X = X_full[:, live]
            W = w[nbrs]
            fold = rng.integers(0, cfg.cv_folds, size=k)
            fold_mse, fold_mse_cnt = 0.0, 0
            fold_nll, fold_nll_cnt = 0.0, 0
            for fo in range(cfg.cv_folds):
                tr = fold != fo
                te = ~tr
                if tr.sum() < 2 or te.sum() == 0:
                    continue
                Xt, Wt, yt = X[tr], W[tr], y_full[tr]
                try:
                    kt = Xt.shape[0]
                    L = live.size
                    if L <= kt:
                        A = np.einsum("ig,ip,iq->gpq", Wt, Xt, Xt)
                        A = A + lam * np.eye(L)[None, :, :]
                        rhs = np.einsum("ig,ip,ig->gp", Wt, Xt, yt)
                        a = np.linalg.solve(A, rhs[..., None]).squeeze(-1)
                    else:
                        K = np.einsum("ip,ig,jp->gij", Xt, Wt, Xt) + lam * np.eye(kt)[None]
                        u = np.linalg.solve(K, yt.T[..., None]).squeeze(-1)
                        a = np.einsum("ip,ig,gi->gp", Xt, Wt, u)
                except np.linalg.LinAlgError:
                    continue
                Xe, We, ye = X[te], W[te], y_full[te]
                yhat = Xe @ a.T
                resid2 = (yhat - ye) ** 2
                w_sum = float(We.sum())
                if w_sum > 0 and np.isfinite(w_sum):
                    fold_mse += float((We * resid2).sum()) / w_sum
                    fold_mse_cnt += 1
                if metric == "nb_nll" and counts_arr is not None and log_theta is not None:
                    te_nbrs = nbrs[te]
                    log_mu = eta[a_cell][None, :] + yhat + log_s[te_nbrs][:, None]
                    log_mu = np.clip(log_mu, -30.0, 30.0)
                    mu_hat = np.exp(log_mu)
                    c_i = counts_arr[te_nbrs]
                    theta_e = np.exp(log_theta)[None, :]
                    log_tpm = np.log(theta_e + mu_hat)
                    lp = (
                        _lgamma(c_i + theta_e) - _lgamma(theta_e)
                        + theta_e * (log_theta[None, :] - log_tpm)
                        + c_i * (log_mu - log_tpm)
                    )
                    nll = -float(lp.mean())
                    if np.isfinite(nll):
                        fold_nll += nll
                        fold_nll_cnt += 1
            if fold_mse_cnt > 0:
                tot_mse += fold_mse / fold_mse_cnt
                cnt_mse += 1
            if fold_nll_cnt > 0:
                tot_nll += fold_nll / fold_nll_cnt
                cnt_nll += 1
        lambda_scores_mse[float(lam)] = (tot_mse / cnt_mse) if cnt_mse else float("inf")
        lambda_scores_nll[float(lam)] = (tot_nll / cnt_nll) if cnt_nll else float("inf")
        if verbose:
            msg = f"  [gene-field anchor] λ={lam:g}  weighted MSE={lambda_scores_mse[float(lam)]:.4f}"
            if metric == "nb_nll":
                msg += f"  NB NLL={lambda_scores_nll[float(lam)]:.4f}"
            print(msg)
    lambda_mse = min(lambda_scores_mse, key=lambda l: lambda_scores_mse[l])
    if metric == "nb_nll" and any(np.isfinite(v) for v in lambda_scores_nll.values()):
        best_lambda = min(lambda_scores_nll, key=lambda l: lambda_scores_nll[l])
    else:
        best_lambda = lambda_mse

    # ---------- per-anchor ridge fit — reuses `_ridge_batched` across genes.
    # We keep each anchor's dense (G, L_a) block so the kernel-interp step
    # can accumulate columns without creating a new sparse matrix per
    # (cell, anchor) pair. Batched-across-anchors einsum blew up memory
    # on G ≈ 2000 (16 GB max RSS seen before); the anchor count (default
    # 300) is already the real speedup over per-cell n ≈ 2700.
    J_anchor_dense: list[np.ndarray] = []       # (G, L_a) per anchor
    F_anchor_dense: list[np.ndarray] = []       # (G, L_a) per anchor
    live_cols_anchor: list[np.ndarray] = []
    capped_cols_anchor: list[np.ndarray] = []    # above threshold, clipped by cap
    # Per-anchor SNR of η across the ball on the log-rate scale, stored as
    # float32 so downstream (archetype aggregation + sidecar) stays cheap on
    # (n_anchors × G) tensors. snr_g = Var_i(η_i,g) · <w_i,g>_i : signal² ·
    # NB precision = the chi-square-like statistic for detectable η movement
    # at that anchor. Reporting it per-gene lets a `capped` or `no_variation`
    # finding carry a measurement, not just a label.
    snr_per_anchor = np.zeros((n_anchors, G), dtype=np.float32)
    max_live = int(getattr(cfg, "anchor_max_live", 300) or 0)
    for sa in range(n_anchors):
        a_cell = int(anchors[sa])
        nbrs = anchor_idxs[sa]
        X_full = eta[nbrs] - eta[a_cell]
        y_full = d_eta[nbrs] - d_eta[a_cell]
        var_g = np.var(X_full, axis=0)
        W = w[nbrs]
        mean_w_g = W.mean(axis=0)                           # (G,) NB precision
        snr_g = var_g * mean_w_g                            # (G,) SNR on log-rate
        snr_per_anchor[sa] = snr_g.astype(np.float32)
        live_above = np.where(var_g > cfg.variation_threshold)[0]
        # Rank live cols by NB-precision-weighted η variance (signal·precision
        # on the log-rate scale); keep the top-``max_live``. Genes above the
        # variation threshold but below the cap are "capped", not
        # "no_variation" — the former is a storage choice, the latter is a
        # measurement about the ball.
        if max_live > 0 and live_above.size > max_live:
            order = np.argsort(-snr_g[live_above])[:max_live]
            live = np.sort(live_above[order])
            capped = np.setdiff1d(live_above, live, assume_unique=True)
        else:
            live = live_above
            capped = np.zeros(0, dtype=np.int64)
        if live.size < 2:
            J_anchor_dense.append(np.zeros((G, 0)))
            F_anchor_dense.append(np.zeros((G, 0)))
            live_cols_anchor.append(np.zeros(0, dtype=np.int64))
            capped_cols_anchor.append(capped.astype(np.int64))
            continue
        X = X_full[:, live]
        try:
            a_mat = _ridge_batched_live(X, W, y_full, float(best_lambda), live.size)
        except np.linalg.LinAlgError:
            J_anchor_dense.append(np.zeros((G, 0)))
            F_anchor_dense.append(np.zeros((G, 0)))
            live_cols_anchor.append(np.zeros(0, dtype=np.int64))
            capped_cols_anchor.append(capped.astype(np.int64))
            continue
        Fv = (W.T @ (X ** 2)) + float(best_lambda)          # (G, L)
        # Dense anchor cache stored as float32: 300 anchors × (G, L=300)
        # ×2 is 2.88 GB in float64 but only 1.44 GB in float32, and the
        # kernel-interp step is a weighted sum where float32 precision
        # is plenty. The ridge solve itself ran in float64 above.
        J_anchor_dense.append(np.asarray(a_mat, dtype=np.float32))
        F_anchor_dense.append(np.asarray(Fv, dtype=np.float32))
        live_cols_anchor.append(live.astype(np.int64))
        capped_cols_anchor.append(capped.astype(np.int64))
    if verbose:
        print(f"  [gene-field anchor] fit {n_anchors} anchor ridges")

    # ---------- Gaussian kernel over anchors → per-cell interpolation
    nn_a = NearestNeighbors(n_neighbors=min(cfg.anchor_top_k, n_anchors)).fit(z[anchors])
    d_a, idx_a = nn_a.kneighbors(z, return_distance=True)      # (n, top_k)
    if cfg.anchor_bandwidth is not None and cfg.anchor_bandwidth > 0:
        h = float(cfg.anchor_bandwidth)
    else:
        h = float(np.median(d_a[:, 0]))                        # auto bandwidth
        h = h if h > 0 else 1.0
    weights = np.exp(-(d_a ** 2) / (2.0 * h ** 2))            # (n, top_k)
    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-12)

    # Build CSR anchor_weights (n, n_anchors).
    aw_rows = np.repeat(np.arange(n), weights.shape[1])
    aw_cols = idx_a.reshape(-1)
    aw_data = weights.reshape(-1)
    anchor_weights = csr_matrix(
        (aw_data, (aw_rows, aw_cols)), shape=(n, n_anchors),
    )

    # Per-cell J[c] / F[c] = Σ_a w_ca · J_anchor[a], restricted to the
    # DOMINANT anchor's live column set (≤ k columns, same storage
    # footprint as the per-cell path). Taking the union across the top-K
    # anchors blows up to ≤ k·top_K columns per cell — on Paul15 (G =
    # 2000, k = 30, top_k = 10) that produced ~24 GB of CSR data for
    # 2730 cells and OOM-killed the run. Each off-dominant anchor still
    # contributes through the Gaussian kernel, but only on the live
    # columns of the cell's nearest anchor.
    J_list: list[csr_matrix] = [csr_matrix((G, G)) for _ in range(n)]
    F_list: list[csr_matrix] = [csr_matrix((G, G)) for _ in range(n)]
    reason_list: list[np.ndarray] = [np.full(G, REASON_NOT_IN_BALL, dtype=np.int8)
                                     for _ in range(n)]
    live_cols: list[np.ndarray] = [np.zeros(0, dtype=np.int64) for _ in range(n)]
    reason = np.full(n, REASON_OK, dtype=np.int8)
    top_k = idx_a.shape[1]
    if verbose:
        print(f"  [gene-field anchor] lifting J/F to {n} cells "
              f"(top_k={top_k}, dominant-live-cols)")
    # Pre-allocate the per-cell working buffers once; reused each cell to
    # keep the macOS jetsam-triggering address-space churn in check (we
    # saw 50 GB peak VM when the loop allocated fresh float64 (G, L_c)
    # arrays per cell). L_cap is the max live-col count across anchors
    # (bounded by G); the ball's live-column count can exceed ``k`` since
    # live-cols are genes whose η-variance across the k-neighbour ball
    # exceeds ``variation_threshold``, not a count of rows.
    L_cap = max((la.size for la in live_cols_anchor), default=1)
    L_cap = max(1, int(L_cap))
    J_buf = np.zeros((G, L_cap), dtype=np.float32)
    F_buf = np.zeros((G, L_cap), dtype=np.float32)
    import gc as _gc
    for c in range(n):
        if verbose and c and c % 500 == 0:
            print(f"    [gene-field anchor] lifted {c}/{n} cells")
            _gc.collect()
        a0 = int(idx_a[c, 0])
        live_this = live_cols_anchor[a0]
        if live_this.size == 0:
            reason[c] = REASON_NO_VARIATION
            continue
        L_c = int(live_this.size)
        J_c = J_buf[:, :L_c]
        F_c = F_buf[:, :L_c]
        J_c.fill(0.0)
        F_c.fill(0.0)
        # Renormalise the kernel over the top-K anchors restricted to
        # those that overlap the dominant anchor's live set.
        for i_a in range(top_k):
            a = int(idx_a[c, i_a])
            wca = float(weights[c, i_a])
            if wca <= 0.0:
                continue
            la = live_cols_anchor[a]
            if la.size == 0:
                continue
            if i_a == 0:
                # Dominant anchor contributes to every column of J_c.
                J_c += wca * J_anchor_dense[a]
                F_c += wca * F_anchor_dense[a]
                continue
            # Other anchors contribute only where they share columns with
            # the dominant anchor's live set.
            common_mask = np.isin(la, live_this, assume_unique=False)
            if not common_mask.any():
                continue
            cols_shared = la[common_mask]
            dst = np.searchsorted(live_this, cols_shared)
            src = common_mask
            J_c[:, dst] += wca * J_anchor_dense[a][:, src]
            F_c[:, dst] += wca * F_anchor_dense[a][:, src]
        # Direct CSR construction (indices + indptr) — skips the COO sort
        # that the (data, (row, col)) constructor does. Each row has
        # exactly ``L_c`` nnz at the same column pattern, so indptr is a
        # stride and indices are tiled. .copy() the slice so the CSR
        # owns its data (the buffer is reused next iteration).
        indptr = (np.arange(G + 1, dtype=np.int32) * L_c)
        indices = np.tile(live_this.astype(np.int32), G)
        # Store CSR data as float32 — on real data (G ≈ 2000, L ≈ 300,
        # n ≈ 2700) the per-cell J + F dominate peak RSS, and halving
        # the dtype keeps the lift within the 36 GB machine budget.
        # Downstream opspace arithmetic (``F_sum + float·CSR`` etc.)
        # auto-upcasts to float64 on the aggregated side, so precision
        # at the archetype level is unaffected.
        J_list[c] = csr_matrix(
            (J_c.reshape(-1).astype(np.float32, copy=True), indices, indptr),
            shape=(G, G),
        )
        F_list[c] = csr_matrix(
            (F_c.reshape(-1).astype(np.float32, copy=True), indices, indptr),
            shape=(G, G),
        )
        live_cols[c] = live_this.astype(np.int64)
        # Per-cell gene reason: no_variation everywhere by default, then
        # mark the dominant anchor's capped-out set, then mark the kept
        # live set. ``capped`` reports "had SNR but didn't fit the cap";
        # ``no_variation`` is the measurement-level verdict (η-variance
        # below the floor across the ball). ``not_in_ball`` is reserved
        # for cells whose dominant anchor has no live columns at all,
        # which we handled with the ``REASON_NO_VARIATION`` short-circuit
        # above.
        rsn = np.full(G, REASON_NO_VARIATION, dtype=np.int8)
        capped_this = capped_cols_anchor[a0]
        if capped_this.size:
            rsn[capped_this] = REASON_CAPPED
        rsn[live_this] = REASON_OK
        reason_list[c] = rsn

    op_anchor = idx_a[:, 0].astype(np.int32)

    return {
        "J": J_list,
        "F": F_list,
        "reason_per_cell": reason_list,
        "live_cols": live_cols,
        "reason": reason,
        "ridge_lambda": float(best_lambda),
        "lambda_scores": lambda_scores_nll if metric == "nb_nll" else lambda_scores_mse,
        "lambda_scores_mse": lambda_scores_mse,
        "lambda_scores_nll": lambda_scores_nll,
        "ridge_lambda_mse": float(lambda_mse),
        "ridge_lambda_nll": float(
            min(lambda_scores_nll, key=lambda l: lambda_scores_nll[l])
            if any(np.isfinite(v) for v in lambda_scores_nll.values())
            else float("nan")
        ),
        "cv_metric": metric,
        "k": int(k),
        "variation_threshold": float(cfg.variation_threshold),
        "variant": "anchor",
        "anchor_ids": anchors.astype(np.int64),
        "op_anchor": op_anchor,
        "anchor_weights": anchor_weights,
        "anchor_bandwidth": float(h),
        "n_anchors": int(n_anchors),
        "anchor_top_k": int(cfg.anchor_top_k),
        "anchor_snr": snr_per_anchor,                 # (n_anchors, G) float32
        "anchor_capped_cols": capped_cols_anchor,     # list of (K_a,) int64
        "anchor_max_live": int(max_live),
    }
