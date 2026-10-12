"""Step 8 — operator archetypes.

Semi-NMF decomposition of the per-cell latent Jacobian tensor::

    J_c  ≈  Σ_k W[c, k] · H_k        for every cell c,
    W ≥ 0 (activations),   H_k signed (archetype operator in R^{d×d}).

K is chosen by **held-out Frobenius reconstruction** on cells (10%
default) — the Gaussian analogue of the "K by held-out likelihood"
rule in docs/scJDO2_figure_layer_build_instructions.md. The per-cell
Jacobian is a Gaussian-noised linear operator in Phase B / R14's
truncated-tangent view, so Frobenius NLL is the native criterion;
if a downstream caller wants to pin K to an NB NLL via predicted
velocities, pass ``k=K`` directly.

Writes ``adata.obsm["archetype_weights"]`` (``(n_cells, K)``) and
``adata.uns["scjdo2"]["archetype_K_curve"]`` (list of
``(K, held_out_frobenius)``) so the K-selection curve is reproducible.
Downstream, :func:`scjdo2.opspace.archetype_operators` consumes the
weights to compute Fisher-weighted archetype operators and the
per-archetype eigenpair / behaviour / driver block.

The semi-NMF solver mirrors v1's ``scjdo.archetypes.decompose._seminmf``
(SciPy NNLS W-update, closed-form ridge H-update, scale resolution
``||H_k||_2 = 1``). We re-implement here to avoid a hard v1 import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
from scipy.optimize import nnls


# ---------------------------------------------------------------------- NMF


def _nnls_rows(H: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Row-wise ``min_{w ≥ 0} ||H.T w - m||`` for every row of ``M``."""
    n = M.shape[0]
    K = H.shape[0]
    A = H.T.astype(np.float64)
    W = np.empty((n, K), dtype=np.float32)
    for c in range(n):
        w_c, _ = nnls(A, M[c].astype(np.float64))
        W[c] = w_c.astype(np.float32)
    return W


def _rescale(W: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h_norm = np.linalg.norm(H, axis=1)
    h_norm = np.where(h_norm < 1e-12, 1.0, h_norm)
    return W * h_norm[None, :], H / h_norm[:, None]


def _seminmf(
    M: np.ndarray,
    rank: int,
    max_iter: int = 500,
    tol: float = 1e-4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Alternating NNLS / ridge update. Returns (W, H, Frobenius err)."""
    rng = np.random.default_rng(seed)
    n, D = M.shape
    W = np.abs(rng.standard_normal((n, rank)).astype(np.float32)) + 1e-4
    W /= W.sum(axis=0, keepdims=True) + 1e-8
    prev = float("inf")
    err = float("inf")
    for _ in range(max_iter):
        WtW = W.T @ W + 1e-6 * np.eye(rank)
        H = np.linalg.solve(WtW, W.T @ M).astype(np.float32)
        W = _nnls_rows(H, M)
        err = float(np.linalg.norm(M - W @ H, "fro"))
        if abs(prev - err) / (prev + 1e-8) < tol:
            break
        prev = err
    W, H = _rescale(W, H)
    return W, H, err


# -------------------------------------------------------------- K selection


def _heldout_frobenius(
    M: np.ndarray,
    rank: int,
    seed: int = 0,
    frac: float = 0.1,
    max_iter: int = 300,
    n_restarts: int = 2,
) -> float:
    """Hold out ``frac`` of rows, fit semi-NMF on the rest, score the held-
    out block by its best (NNLS) reconstruction under the trained H."""
    rng = np.random.default_rng(seed)
    n = M.shape[0]
    n_test = max(1, int(round(frac * n)))
    perm = rng.permutation(n)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    M_train = M[train_idx]
    M_test = M[test_idx]
    best_err = float("inf")
    for s in range(max(1, n_restarts)):
        try:
            _, H, _ = _seminmf(
                M_train, rank=rank, max_iter=max_iter, seed=seed + s,
            )
        except Exception:
            continue
        # Project the held-out rows onto H via NNLS.
        W_test = _nnls_rows(H, M_test)
        err = float(np.linalg.norm(M_test - W_test @ H, "fro"))
        if err < best_err:
            best_err = err
    return best_err


@dataclass
class ArchetypeConfig:
    n_archetypes: int | str = "auto"            # int or "auto" (held-out F)
    k_candidates: Sequence[int] = (2, 3, 4, 5, 6, 8)
    held_out_frac: float = 0.10
    n_restarts: int = 3
    max_iter: int = 500
    seed: int = 0


@dataclass
class ArchetypeResult:
    weights: np.ndarray                         # (n, K)
    patterns: np.ndarray                        # (K, d, d) latent archetypes
    k_curve: list                               # [(K, heldout_frobenius)]
    k_selected: int
    frobenius_error: float                      # on the full dataset


class ArchetypeDecomposer:
    """Semi-NMF archetype decomposer.

    Fits ``M (n, d*d) ≈ W (n, K) @ H (K, d*d)`` with ``W ≥ 0``, chooses
    ``K`` by held-out Frobenius reconstruction when the config says so,
    and returns both the activations and the latent archetype patterns.
    """

    def __init__(self, config: Optional[ArchetypeConfig] = None) -> None:
        self.config = config or ArchetypeConfig()

    def fit(self, J_tensor: np.ndarray) -> ArchetypeResult:
        if J_tensor.ndim != 3 or J_tensor.shape[1] != J_tensor.shape[2]:
            raise ValueError(
                f"J_tensor must be (n, d, d); got shape {J_tensor.shape}"
            )
        n, d, _ = J_tensor.shape
        M = J_tensor.reshape(n, d * d).astype(np.float32)

        if self.config.n_archetypes == "auto":
            curve: list[tuple[int, float]] = []
            for K in self.config.k_candidates:
                err = _heldout_frobenius(
                    M, rank=int(K), seed=int(self.config.seed),
                    frac=float(self.config.held_out_frac),
                    max_iter=int(self.config.max_iter),
                    n_restarts=int(self.config.n_restarts),
                )
                curve.append((int(K), float(err)))
            K_star = min(curve, key=lambda t: t[1])[0]
        else:
            K_star = int(self.config.n_archetypes)
            curve = []

        best_W, best_H, best_err = None, None, float("inf")
        for s in range(int(self.config.n_restarts)):
            W, H, err = _seminmf(
                M, rank=K_star,
                max_iter=int(self.config.max_iter),
                seed=int(self.config.seed) + s,
            )
            if err < best_err:
                best_W, best_H, best_err = W, H, err

        patterns = np.asarray(best_H).reshape(K_star, d, d)
        return ArchetypeResult(
            weights=np.asarray(best_W),
            patterns=patterns,
            k_curve=curve,
            k_selected=int(K_star),
            frobenius_error=float(best_err),
        )
