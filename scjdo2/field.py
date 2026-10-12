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

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


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
