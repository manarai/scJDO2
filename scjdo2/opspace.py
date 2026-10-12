"""Step 7 — operator space (Phase B subset).

Per-cell invariants (eigenvalues, counts of stable/unstable/near-zero/complex
pairs, spectral gap, Henrici index, reactivity, transient gain, eigvec
conditioning) and the leading eigenvectors. The four non-normality descriptors
are reimplemented from v1's :mod:`scjdo.archetypes.koopman._matrix_geometry`;
see V2_PLAN.md §4 for provenance and the exact formulas.

Fisher weighting (Phase C) layers on top; Phase B passes ``ledger=None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class OpInvariants:
    eigenvalues: np.ndarray         # (n_cells, n_eigs), complex, sorted by Re desc
    n_unstable: np.ndarray          # (n_cells,) int
    n_stable: np.ndarray            # (n_cells,) int
    n_near_zero: np.ndarray         # (n_cells,) int
    n_complex_pairs: np.ndarray     # (n_cells,) int
    spectral_gap: np.ndarray        # (n_cells,) float — Re(λ_1) − Re(λ_2)
    max_real_eig: np.ndarray        # (n_cells,) float — Re(λ_max)
    henrici: np.ndarray             # (n_cells,) float ∈ [0, 1]
    reactivity: np.ndarray          # (n_cells,) float ≥ 1
    transient_gain: np.ndarray      # (n_cells,) float
    eigvec_cond: np.ndarray         # (n_cells,) float


@dataclass
class OpCoordinates:
    invariants: OpInvariants
    leading_eigvecs: np.ndarray     # (n_cells, n_lead, d), complex
    valid: np.ndarray               # (n_cells,) bool — False where J was NaN


def _sort_eigs_by_real(eigs: np.ndarray, eigvecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-eigs.real)
    return eigs[order], eigvecs[:, order]


def _transient_gain(J: np.ndarray, n_max: int = 30) -> float:
    """max_{n ∈ [1, n_max]} ||J^n||_2. Early-exit when ρ(J) < 1 and the norm decays."""
    d = J.shape[0]
    eigs = np.linalg.eigvals(J)
    rho = float(np.max(np.abs(eigs))) if eigs.size else 0.0
    P = np.eye(d, dtype=np.float64)
    gain = 0.0
    for _ in range(int(n_max)):
        P = J @ P
        gn = float(np.linalg.norm(P, ord=2))
        if gn > gain:
            gain = gn
        if rho < 1.0 and gn < 1e-12:
            break
    return gain


def compute(
    J,
    ledger: Optional[object] = None,   # Phase C: scjdo2.fisher.FisherLedger
    n_leading: int = 2,
    near_zero_tol: float = 1e-3,
    transient_n_max: int = 30,
) -> OpCoordinates:
    """Per-cell operator coordinates.

    ``J`` may be a ``(n_cells, d, d)`` ndarray or a list of ``(d, d)`` arrays
    (``np.nan`` entries flag unfit cells, forwarded as ``valid=False``).
    """
    if isinstance(J, np.ndarray) and J.ndim == 3:
        J_list = [J[i] for i in range(J.shape[0])]
    else:
        J_list = list(J)
    n = len(J_list)
    if n == 0:
        raise ValueError("empty J")
    d = J_list[0].shape[0]

    eigvals = np.full((n, d), np.nan + 0j, dtype=np.complex128)
    leading = np.full((n, n_leading, d), np.nan + 0j, dtype=np.complex128)
    n_unstable = np.zeros(n, dtype=np.int32)
    n_stable = np.zeros(n, dtype=np.int32)
    n_near_zero = np.zeros(n, dtype=np.int32)
    n_complex_pairs = np.zeros(n, dtype=np.int32)
    spectral_gap = np.full(n, np.nan)
    max_real_eig = np.full(n, np.nan)
    henrici = np.full(n, np.nan)
    reactivity = np.full(n, np.nan)
    transient_gain = np.full(n, np.nan)
    eigvec_cond = np.full(n, np.nan)
    valid = np.ones(n, dtype=bool)

    for i, Jc in enumerate(J_list):
        if not np.all(np.isfinite(Jc)):
            valid[i] = False
            continue
        eigs, V = np.linalg.eig(Jc)
        eigs, V = _sort_eigs_by_real(eigs, V)
        eigvals[i] = eigs
        n_lead = min(n_leading, V.shape[1])
        leading[i, :n_lead] = V[:, :n_lead].T

        re = eigs.real
        im = eigs.imag
        max_real_eig[i] = float(re[0])
        if len(re) >= 2:
            spectral_gap[i] = float(re[0] - re[1])
        n_unstable[i] = int(np.sum(re > near_zero_tol))
        n_stable[i] = int(np.sum(re < -near_zero_tol))
        n_near_zero[i] = int(np.sum(np.abs(re) <= near_zero_tol))
        n_complex_pairs[i] = int(np.sum(im > 0))  # count each pair once

        frob2 = float(np.sum(Jc * Jc))
        frob = float(np.sqrt(max(frob2, 0.0)))
        sum_lam2 = float(np.sum(np.abs(eigs) ** 2))
        henrici_abs = float(np.sqrt(max(0.0, frob2 - sum_lam2)))
        henrici[i] = henrici_abs / (frob + 1e-12)

        sigma_max = float(np.linalg.norm(Jc, ord=2))
        rho = float(np.max(np.abs(eigs))) if eigs.size else 0.0
        reactivity[i] = sigma_max / (rho + 1e-12)
        transient_gain[i] = _transient_gain(Jc.astype(np.float64), n_max=transient_n_max)
        try:
            eigvec_cond[i] = float(np.linalg.cond(V))
        except np.linalg.LinAlgError:
            eigvec_cond[i] = np.inf

    inv = OpInvariants(
        eigenvalues=eigvals,
        n_unstable=n_unstable,
        n_stable=n_stable,
        n_near_zero=n_near_zero,
        n_complex_pairs=n_complex_pairs,
        spectral_gap=spectral_gap,
        max_real_eig=max_real_eig,
        henrici=henrici,
        reactivity=reactivity,
        transient_gain=transient_gain,
        eigvec_cond=eigvec_cond,
    )
    return OpCoordinates(invariants=inv, leading_eigvecs=leading, valid=valid)
