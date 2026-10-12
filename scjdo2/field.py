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
    max_rank: Optional[int] = None,
) -> dict:
    """Per-cell η-tangent projector from the rate model's own Jacobian ∇η(z_c).

    The decoder ``η: R^d → R^G`` is differentiable; its Jacobian at cell
    ``c`` is ``∂η/∂z|_{z_c} ∈ R^{G × d}``. The image of ``∇η(z_c)`` is the
    η-tangent plane the decoder can actually move the cell along in gene
    space. Its left singular vectors are the orthonormal basis of that
    plane; by construction there are at most ``d`` of them with nonzero
    singular values, so the tangent rank is **r_c = d exactly**.

    The projector ``P_c`` is the ``d × G`` matrix of top-``d`` left
    singular vectors (``U.T`` from the thin SVD of ``∇η(z_c)``). It is
    tighter than the kNN-ball η-covariance tangent of the pre-fix: no
    neighbour sampling, no SNR threshold, no residual-tangent slack from
    the SiLU trunk's nonlinear fit sampled over the ball.

    Rationale (docs/opspace_figure_spec.md §9 fix (c)). On a ``d``-dim
    decoder the gene-space J is a lift of the latent J: ``J_gene =
    (∇η) · J_latent · (∇η)^+`` has rank at most ``d``, and its spectrum
    on the η-tangent equals the latent spectrum up to a similarity. The
    Jacobian-SVD tangent restricts eigendecomposition to that
    ``d``-dim subspace exactly, so the gene-space classifier can MATCH
    the latent one on this synthetic (it cannot exceed it — the extra
    G - d ambient directions carry no decoder-visible dynamics).

    Returns a dict with the same shape ``fit_tangent_projectors`` had,
    for drop-in use by :mod:`scjdo2.fisher` and :func:`scjdo2.opspace.spectrum`:
        * ``rank`` : ``(n,)`` int — all equal to ``r_c = min(d, max_rank, G)``.
        * ``projector`` : list of ``(r_c, G)`` float64 arrays — the top-r
          left singular directions of ``∇η(z_c)`` as rows.
        * ``eigvals`` : list of ``(d,)`` singular-value squares (descending),
          kept for diagnostics parity with the ball-covariance variant.
        * ``method`` : ``"jacobian_svd"``.
    """
    z = np.asarray(z, dtype=np.float64)
    n, d = z.shape
    g_eta = rate_model.grad_eta(z).astype(np.float64)          # (n, G, d)
    G = g_eta.shape[1]
    cap = int(max_rank) if max_rank is not None else G
    r_c = max(1, min(d, cap, G))

    rank_arr = np.full(n, r_c, dtype=np.int32)
    projectors: list[np.ndarray] = []
    sv_list: list[np.ndarray] = []
    for c in range(n):
        A = g_eta[c]                                           # (G, d)
        try:
            U, S, _ = np.linalg.svd(A, full_matrices=False)    # U (G, d), S (d,)
        except np.linalg.LinAlgError:
            projectors.append(np.eye(G)[:r_c])
            sv_list.append(np.zeros(min(d, G)))
            continue
        P_c = U[:, :r_c].T.astype(np.float64)                  # (r_c, G)
        projectors.append(P_c)
        sv_list.append((S ** 2).astype(np.float64))            # σ² (descending)
    return {
        "rank": rank_arr,
        "projector": projectors,
        "eigvals": sv_list,
        "method": "jacobian_svd",
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

_REASON_NAMES = {
    REASON_OK: "ok",
    REASON_NO_COUNTS: "no_counts",
    REASON_NO_VARIATION: "no_variation",
    REASON_NOT_IN_BALL: "not_in_ball",
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
# Ridge λ is chosen globally by 5-fold neighbour cross-validation using
# weighted MSE on held-out neighbours. (Weighted MSE is the local Gaussian
# approximation of the NB NLL — the Fisher-weighted ridge IS the
# second-order NB approximation of the local likelihood, which is why the
# user's spec says "ridge strength chosen by held-out likelihood"; here
# we use its Gaussian approximant.) See §8.3 and §8.1 of the spec.
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
    seed: int = 0


def fit_gene_space_jacobians(
    z: np.ndarray,                        # (n, d) latent positions
    f: np.ndarray,                        # (n, d) latent drift f(z)
    rate_model,                           # scjdo2.rates.GeneRateModel (fitted)
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
        * ``ridge_lambda`` : the single λ chosen by held-out MSE.
        * ``lambda_scores`` : dict ``{λ: mean_weighted_mse}`` from CV.
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

    eta = rate_model.eta(z).astype(np.float64)              # (n, G)
    d_eta = rate_model.rate_along_eta(z, f).astype(np.float64)  # (n, G)
    w = rate_model.nb_weights(z, size_factor=size_factor).astype(np.float64)  # (n, G)
    G = eta.shape[1]

    nn = NearestNeighbors(n_neighbors=k + 1).fit(z)
    _, idxs = nn.kneighbors(z, return_distance=True)
    idxs = idxs[:, 1:]                                      # drop self

    # --------- global λ selection by k-fold neighbour CV on sampled cells
    rng = np.random.default_rng(int(cfg.seed))
    sample_n = n if cfg.cv_sample_cells < 0 else min(int(cfg.cv_sample_cells), n)
    sample_idx = rng.choice(n, size=sample_n, replace=False)
    lambda_scores: dict[float, float] = {}

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

    for lam in cfg.ridge_lambdas:
        tot = 0.0
        cnt = 0
        for c in sample_idx:
            nbrs = idxs[c]
            X_full = eta[nbrs] - eta[c]                     # (k, G)
            y_full = d_eta[nbrs] - d_eta[c]                 # (k, G)
            live = np.where(np.var(X_full, axis=0) > cfg.variation_threshold)[0]
            if live.size < 2 or k < max(5, cfg.cv_folds + 1):
                continue
            X = X_full[:, live]                             # (k, L)
            W = w[nbrs]                                     # (k, G)
            # Fold assignment on neighbours
            fold = rng.integers(0, cfg.cv_folds, size=k)
            fold_mse = 0.0
            fold_cnt = 0
            for fo in range(cfg.cv_folds):
                tr = fold != fo
                te = ~tr
                if tr.sum() < 2 or te.sum() == 0:
                    continue
                Xt = X[tr]                                  # (kt, L)
                Wt = W[tr]                                  # (kt, G)
                yt = y_full[tr]                             # (kt, G)
                try:
                    a = _ridge_batched(Xt, Wt, yt, lam, live.size)
                except np.linalg.LinAlgError:
                    continue
                # Held-out weighted MSE on te neighbours
                Xe = X[te]                                  # (ke, L)
                We = W[te]                                  # (ke, G)
                ye = y_full[te]                             # (ke, G)
                yhat = Xe @ a.T                             # (ke, G)
                resid2 = (yhat - ye) ** 2
                w_sum = float(We.sum())
                if w_sum <= 0 or not np.isfinite(w_sum):
                    continue
                fold_mse += float((We * resid2).sum()) / w_sum
                fold_cnt += 1
            if fold_cnt > 0 and np.isfinite(fold_mse):
                tot += fold_mse / fold_cnt
                cnt += 1
        lambda_scores[float(lam)] = (tot / cnt) if cnt > 0 else float("inf")
        if verbose:
            print(f"  [gene-field] λ={lam:g}  mean weighted MSE = {lambda_scores[float(lam)]:.4f}")
    best_lambda = min(lambda_scores, key=lambda l: lambda_scores[l])

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
        "lambda_scores": lambda_scores,
        "k": int(k),
        "variation_threshold": float(cfg.variation_threshold),
    }
