"""Step 7 — operator-space analysis API.

Operator-space coordinates and summaries derived from a fitted per-cell local
Jacobian field (Step 5). All functions act on an ``AnnData`` and read the
field from ``adata.uns["scjdo2"]["J"]`` (dense ``(n_cells, d, d)`` in Phase
B's latent-space variant; see V2_PLAN R11 and §3). Results are written to
``adata.obs`` / ``adata.obsm`` / ``adata.uns`` so the plotting layer never
estimates — it only reads.

The four non-normality formulas (Henrici, reactivity, transient gain, eigvec
conditioning) are reimplemented here from v1's
``scjdo.archetypes.koopman._matrix_geometry`` per V2_PLAN §4; v1 is not
imported.

Phase B scope (latent-space, dense J, no Fisher ledger):
    - :func:`spectrum`, :func:`invariants`, :func:`behaviour`,
      :func:`field_gradient`, :func:`lineage_path`, :func:`export_tables`
      are implemented.
    - :func:`drivers` and :func:`archetype_operators` raise
      ``NotImplementedError`` until Phase C's gene-space Jacobians and the
      semi-NMF adapter land.

Clock calibration: eigenvalue units are "relative" unless
``adata.uns["scjdo2"]["meta"]["clock_calibrated"]`` is True. Exported table
headers reflect this; the plot layer uses the same flag for axis labels.

Complex eigenvalues come in conjugate pairs; we store and plot only the
member with ``Im >= 0``. The partner at ``-Im`` is dropped by construction
(see :func:`spectrum`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors


BEHAVIOURS = ("holder", "drift", "fork", "damped_oscillator", "growing_oscillator")
REASON_CODES = ("ok", "no_counts", "no_variation", "not_in_ball", "off_tangent")

_EPS_IM = 1e-10
_FISHER_MASS_TIE_TOL = 1e-6     # Re ties broken by Fisher mass within this margin


# ======================================================================
# Legacy dataclasses — kept so tests that imported them keep working.
# New code should use the AnnData-centric API below.
# ======================================================================


@dataclass
class OpInvariants:
    eigenvalues: np.ndarray
    n_unstable: np.ndarray
    n_stable: np.ndarray
    n_near_zero: np.ndarray
    n_complex_pairs: np.ndarray
    spectral_gap: np.ndarray
    max_real_eig: np.ndarray
    henrici: np.ndarray
    reactivity: np.ndarray
    transient_gain: np.ndarray
    eigvec_cond: np.ndarray


@dataclass
class OpCoordinates:
    invariants: OpInvariants
    leading_eigvecs: np.ndarray
    valid: np.ndarray


# ======================================================================
# Non-normality helpers (reimplemented from v1 _matrix_geometry)
# ======================================================================


def _henrici(J: np.ndarray, eigs: np.ndarray) -> float:
    frob2 = float(np.sum(J * J))
    frob = float(np.sqrt(max(frob2, 0.0)))
    sum_lam2 = float(np.sum(np.abs(eigs) ** 2))
    henrici_abs = float(np.sqrt(max(0.0, frob2 - sum_lam2)))
    return henrici_abs / (frob + 1e-12)


def _reactivity(J: np.ndarray, eigs: np.ndarray) -> float:
    sigma_max = float(np.linalg.norm(J, ord=2))
    rho = float(np.max(np.abs(eigs))) if eigs.size else 0.0
    return sigma_max / (rho + 1e-12)


def _transient_gain(J: np.ndarray, n_max: int = 30) -> float:
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


def _eigvec_cond(V: np.ndarray) -> float:
    try:
        return float(np.linalg.cond(V))
    except np.linalg.LinAlgError:
        return float("inf")


# ======================================================================
# Conjugate-free eigenpairs
# ======================================================================


def _conjugate_free(eigs: np.ndarray, vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep only eigenvalues with Im >= 0 (modulo ``_EPS_IM``).

    Real eigenvalues (Im ≈ 0) are kept once; complex-conjugate pairs are
    represented by their upper-half member only. The partner at ``-Im`` is
    discarded by construction, so no two returned eigenvalues are conjugates
    of each other.
    """
    mask = eigs.imag >= -_EPS_IM
    kept_eigs = eigs[mask]
    kept_vecs = vecs[:, mask]
    # Sort by Re desc, then by Im desc as a tie-break so the Im>=0 member
    # of a near-real duplicate wins.
    order = np.lexsort((-kept_eigs.imag, -kept_eigs.real))
    return kept_eigs[order], kept_vecs[:, order]


def _pad_to(arr: np.ndarray, n: int, dtype) -> np.ndarray:
    out = np.full(n, np.nan, dtype=dtype)
    k = min(n, arr.size)
    out[:k] = arr[:k]
    return out


def _get_meta(adata: AnnData) -> dict:
    return dict(adata.uns.get("scjdo2", {}).get("meta", {}))


def _is_clock_calibrated(adata: AnnData) -> bool:
    return bool(_get_meta(adata).get("clock_calibrated", False))


def _get_J(adata: AnnData) -> tuple[Any, str, int]:
    """Return ``(J, kind, dim)`` where ``kind`` is ``"dense"`` or ``"sparse"``.

    * ``"dense"``: ``J`` is an ``(n, d, d)`` ndarray (Phase B latent-space J).
      ``dim`` is the latent dimension ``d``.
    * ``"sparse"``: ``J`` is a list/tuple of ``n`` ``csr_matrix`` of shape
      ``(G, G)``, with columns restricted to each cell's live-gene index
      (Phase C gene-space J, per docs/opspace_figure_spec.md §1 and §8.3).
      ``dim`` is ``G``.
    """
    try:
        J = adata.uns["scjdo2"]["J"]
    except KeyError as e:
        raise KeyError(
            "adata.uns['scjdo2']['J'] not found. Run scjdo2.pipeline.fit_latent_field "
            "(Phase B) or the Phase C rate-model + gene-space pipeline before "
            "calling opspace functions."
        ) from e
    # Sparse / per-cell CSR path (Phase C)
    if isinstance(J, (list, tuple)) and len(J) > 0 and issparse(J[0]):
        if len(J) != adata.n_obs:
            raise ValueError(
                f"J has {len(J)} per-cell matrices but adata.n_obs={adata.n_obs}"
            )
        G = int(J[0].shape[0])
        for c, Jc in enumerate(J):
            if Jc.shape != (G, G):
                raise ValueError(
                    f"J[{c}] has shape {Jc.shape}; expected ({G}, {G})"
                )
        return list(J), "sparse", G
    # Dense latent-space path (Phase B)
    J_arr = np.asarray(J)
    if J_arr.ndim != 3 or J_arr.shape[1] != J_arr.shape[2]:
        raise ValueError(
            f"adata.uns['scjdo2']['J'] must be (n_cells, d, d) dense or a list of "
            f"per-cell CSR; got dense shape {J_arr.shape}"
        )
    if J_arr.shape[0] != adata.n_obs:
        raise ValueError(
            f"J.shape[0]={J_arr.shape[0]} != adata.n_obs={adata.n_obs}"
        )
    return J_arr, "dense", int(J_arr.shape[1])


def _get_ledger(adata: AnnData):
    """Return the Fisher ledger if present, else ``None``.

    Accepts either a :class:`scjdo2.fisher.FisherLedger` instance or a plain
    dict with keys ``fisher`` (list of CSR) and ``reason_per_cell``.
    """
    return adata.uns.get("scjdo2", {}).get("fisher_ledger")


def _ledger_fisher_list(ledger) -> Optional[list[csr_matrix]]:
    """Normalise ledger -> list of per-cell Fisher CSR, or None."""
    if ledger is None:
        return None
    if hasattr(ledger, "fisher"):
        return list(ledger.fisher)
    if isinstance(ledger, dict) and "fisher" in ledger:
        return list(ledger["fisher"])
    return None


def _ledger_tangent(ledger) -> Optional[dict]:
    """Normalise ledger -> off_tangent dict with per-cell projector, or None."""
    if ledger is None:
        return None
    if hasattr(ledger, "off_tangent"):
        return ledger.off_tangent
    if isinstance(ledger, dict):
        return ledger.get("off_tangent")
    return None


def _live_cols_of(Jc: csr_matrix) -> np.ndarray:
    """Return the live-column indices of a CSR Jacobian (unique column ids)."""
    if Jc.nnz == 0:
        return np.zeros(0, dtype=np.int64)
    return np.unique(Jc.indices).astype(np.int64)


def _dense_live_block(Jc: csr_matrix, live: np.ndarray) -> np.ndarray:
    """Dense ``(L, L)`` live-block of a CSR Jacobian — rows and cols
    restricted to ``live``. For the eigendecomposition in :func:`spectrum`.
    """
    if live.size == 0:
        return np.zeros((0, 0), dtype=np.float64)
    return np.asarray(Jc[live, :][:, live].todense(), dtype=np.float64)


def _fix_eigvec_sign(v: np.ndarray) -> np.ndarray:
    """Fix the phase of ``v`` so the largest-|magnitude| entry is real and
    positive (docs/opspace_figure_spec.md §2.4 and §8.6). Works for real
    and complex vectors.
    """
    v = np.asarray(v)
    if v.size == 0:
        return v
    k = int(np.argmax(np.abs(v)))
    if np.abs(v[k]) == 0:
        return v
    phase = v[k] / np.abs(v[k])
    return v / phase


# ======================================================================
# 2.1 spectrum
# ======================================================================


def compute(
    adata: AnnData,
    smooth_k: int = 0,
    n_eig: int = 5,
    min_fisher: Optional[float] = None,
    tol: Optional[float] = None,
    sensitive_pct: float = 90.0,
) -> AnnData:
    """§1 wrapper — run 1.1–1.6 on ``adata`` and apply V2_PLAN R15.

    Order of operations (see
    ``docs/scJDO2_figure_layer_build_instructions.md`` §1 and V2_PLAN R15):

    1. **Latent pass.** If ``adata.uns["scjdo2"]["J_latent"]`` is present
       alongside the gene-space J (``uns["scjdo2"]["J"]`` as a list of
       sparse CSR), the latent dense J is temporarily substituted and
       :func:`spectrum`, :func:`invariants`, :func:`behaviour` are run on
       it. This writes the behavioural columns (``lead_re``, ``lead_im``,
       ``n_*``, ``spectral_gap``, ``henrici``, ``transient_gain``,
       ``reactivity``, ``eigvec_cond``, ``op_confidence``, ``op_behaviour``,
       ``op_sensitive``) from the latent operator — R15. The latent lead
       eigenvector is cached in ``uns["scjdo2"]["op_eigvec_latent_lead"]``
       (shape ``(n, d)``, complex).
    2. **Gene-space pass.** The gene-space J (now restored) is run through
       :func:`spectrum` to obtain the gene-space eigenbasis stored in
       ``uns["scjdo2"]["op_eigvecs"]``. The invariants and behaviour from
       step 1 are not overwritten.
    3. **Lifted driver eigvec.** If ``uns["scjdo2"]["grad_eta"]`` (shape
       ``(n, G, d)``) is present, the chain-rule lift ``w_c = ∇η(z_c) ·
       v_latent_lead`` is computed and stored in
       ``uns["scjdo2"]["op_eigvec_lifted"]`` (``(n, G)`` complex). Panel C
       drivers consume this (see :func:`drivers`).
    4. :func:`archetype_operators` (skips cleanly if archetype weights
       are missing) and :func:`field_gradient` run last.

    Pure latent runs (no gene-space J): step 1 only; step 2 is a no-op
    (same J). Pure gene-space runs (no latent J stored): step 1 is
    skipped and behavioural columns come from the gene-space J — this
    is the pre-R15 fallback and callers get a one-line warning.
    """
    import warnings

    uns = adata.uns.setdefault("scjdo2", {})
    J_main = uns.get("J")
    J_latent = uns.get("J_latent")
    gene_space_present = (
        isinstance(J_main, (list, tuple)) and len(J_main) > 0
        and issparse(J_main[0])
    )
    latent_present = J_latent is not None

    did_latent_pass = False
    if gene_space_present and latent_present:
        # 1. Latent pass — swap in the latent J, run spectrum→invariants→behaviour.
        # The Fisher ledger is gene-space; pulling it aside for the latent
        # pass avoids a shape mismatch inside invariants' op_confidence step.
        gene_ledger = uns.pop("fisher_ledger", None)
        uns["J"] = J_latent
        spectrum(adata, n_eig=n_eig, fisher_mask=False, smooth_k=0)
        invariants(adata, tol=tol)
        behaviour(adata, tol=tol, sensitive_pct=sensitive_pct)
        # Cache the latent leading eigvec for the ∇η lift step.
        eigvecs_latent = np.asarray(uns["op_eigvecs"])
        uns["op_eigvec_latent_lead"] = eigvecs_latent[:, 0, :].copy()
        # Restore the gene-space J + ledger for step 2 and downstream calls.
        uns["J"] = J_main
        if gene_ledger is not None:
            uns["fisher_ledger"] = gene_ledger
        did_latent_pass = True
    elif latent_present and not gene_space_present:
        # Pure latent run — behavioural columns come from the latent J.
        spectrum(adata, n_eig=n_eig, fisher_mask=False, smooth_k=smooth_k)
        invariants(adata, tol=tol)
        behaviour(adata, tol=tol, sensitive_pct=sensitive_pct)
        eigvecs_latent = np.asarray(uns["op_eigvecs"])
        uns["op_eigvec_latent_lead"] = eigvecs_latent[:, 0, :].copy()
    else:
        # Pure gene-space run or legacy: no latent J to classify from.
        if gene_space_present:
            warnings.warn(
                "scjdo2.opspace.compute: no J_latent in adata.uns; behavioural "
                "columns will be derived from the gene-space J (pre-R15 "
                "fallback). Store the latent J under uns['scjdo2']['J_latent']"
                " to enable R15.",
                RuntimeWarning,
                stacklevel=2,
            )

    # 2. Gene-space (or main) pass for the eigenbasis used by drivers.
    spectrum(adata, n_eig=n_eig, fisher_mask=True,
             min_fisher=min_fisher, smooth_k=smooth_k)
    if not did_latent_pass:
        invariants(adata, tol=tol)
        behaviour(adata, tol=tol, sensitive_pct=sensitive_pct)

    # 3. Lifted driver eigvec via ∇η(z_c) · v_latent_lead.
    grad_eta = uns.get("grad_eta")
    v_latent = uns.get("op_eigvec_latent_lead")
    if grad_eta is not None and v_latent is not None:
        grad_eta = np.asarray(grad_eta)             # (n, G, d)
        v_latent = np.asarray(v_latent)             # (n, d), complex
        if (
            grad_eta.ndim == 3
            and v_latent.ndim == 2
            and grad_eta.shape[0] == v_latent.shape[0]
            and grad_eta.shape[2] == v_latent.shape[1]
        ):
            w_lifted = np.einsum("ngd,nd->ng", grad_eta.astype(np.complex128), v_latent)
            # Sign-fix each cell's lifted vector (largest-|magnitude| → phase 0).
            for c in range(w_lifted.shape[0]):
                w_lifted[c] = _fix_eigvec_sign(w_lifted[c])
            uns["op_eigvec_lifted"] = w_lifted

    # 4. Archetype operators (if weights present) and field gradient.
    try:
        archetype_operators(adata)
    except Exception as err:  # keep downstream figures functional
        warnings.warn(
            f"archetype_operators skipped: {err}", RuntimeWarning, stacklevel=2,
        )
    try:
        field_gradient(adata)
    except Exception as err:
        warnings.warn(
            f"field_gradient skipped: {err}", RuntimeWarning, stacklevel=2,
        )

    uns["compute_meta"] = {
        "smooth_k": int(smooth_k),
        "n_eig": int(n_eig),
        "min_fisher": None if min_fisher is None else float(min_fisher),
        "did_latent_pass": bool(did_latent_pass),
        "has_lifted_eigvec": bool("op_eigvec_lifted" in uns),
    }
    return adata


def spectrum(
    adata: AnnData,
    n_eig: int = 5,
    fisher_mask: bool = True,
    min_fisher: Optional[float] = None,
    smooth_k: int = 0,
) -> None:
    """Compute per-cell conjugate-free eigenpairs of the stored Jacobian.

    Stores:
        ``adata.obsm["op_eigvals"]`` — ``(n_cells, n_eig)`` complex
        ``adata.uns["scjdo2"]["op_eigvecs"]`` — ``(n_cells, n_eig, dim)`` complex
            where ``dim`` is the Jacobian space (latent ``d`` in Phase B,
            genes ``G`` in Phase C). The eigenvector tensor is dense;
            Phase D may sparsify.
        ``adata.uns["scjdo2"]["op_valid"]`` — ``(n_cells,)`` bool
        ``adata.uns["scjdo2"]["spectrum_meta"]`` — ``{n_eig, fisher_mask,
          min_fisher, J_kind, smooth_k}``
        ``adata.obs["op_rank"]`` — per-cell ``r_c`` (Phase C only; the
          tangent rank from ``scjdo2.field.fit_jacobian_tangent_projectors``).

    Behaviour:

    * **Phase B dense J** (``(n, d, d)``). The ``fisher_mask`` kwarg has no
      effect (no ledger); the call records ``fisher_mask=False`` with a
      warning and continues.
    * **Phase C sparse J** (list of per-cell CSR). For each cell the
      live-block ``J_c[live, :][:, live]`` is dense ``(L_c, L_c)`` and gets
      eigendecomposed. If ``fisher_mask`` and a ledger is present, entries
      of ``J_c.data`` whose Fisher info is below ``min_fisher`` are zeroed
      before the live block is extracted. ``min_fisher`` defaults to the
      10th percentile of each cell's live Fisher data.

    If ``smooth_k > 0`` the per-cell J field is pre-smoothed via
    :func:`scjdo2.field.smooth_jacobians` (dense) or
    :func:`scjdo2.field.smooth_jacobians_sparse` (per-cell CSR) before
    the eigendecomposition. The latent kNN is taken from
    ``adata.obsm["X_pca"]`` when present, else the first ``d`` columns
    of the dense latent array. The smoothed field is kept in-place for
    the duration of the call and the original restored on return (so
    callers chaining multiple spectrum passes stay well-behaved).
    """
    import warnings

    smooth_k = int(smooth_k)
    J, kind, dim = _get_J(adata)
    n = adata.n_obs
    n_eig = min(int(n_eig), dim)

    # Optional pre-smoothing. We stage the smoothed J inside the call;
    # callers see the original adata on return.
    orig_J = adata.uns["scjdo2"]["J"] if smooth_k > 0 else None
    if smooth_k > 0:
        from scjdo2 import field as _field_mod
        z_rep = (
            np.asarray(adata.obsm["X_pca"])
            if "X_pca" in adata.obsm
            else np.asarray(adata.uns["scjdo2"].get("z", np.zeros((n, dim))))
        )
        if kind == "dense":
            J_sm = _field_mod.smooth_jacobians(np.asarray(J), z_rep, k=smooth_k)
        else:
            J_sm = _field_mod.smooth_jacobians_sparse(J, z_rep, k=smooth_k)
        adata.uns["scjdo2"]["J"] = J_sm
        J, kind, dim = _get_J(adata)
        adata.uns["scjdo2"]["smooth_k"] = smooth_k

    ledger = _get_ledger(adata)
    F_list = _ledger_fisher_list(ledger)
    applied_mask = False
    threshold_used: Optional[float] = None
    if fisher_mask and F_list is None:
        warnings.warn(
            "spectrum(fisher_mask=True) called without a Fisher ledger; "
            "falling back to no mask. Phase B latent runs have no ledger; "
            "Phase C stores one when the gene-space pipeline runs.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif fisher_mask and F_list is not None:
        applied_mask = True
        threshold_used = float(min_fisher) if min_fisher is not None else None

    eigvals = np.full((n, n_eig), np.nan + 1j * np.nan, dtype=np.complex128)
    eigvecs = np.full((n, n_eig, dim), np.nan + 1j * np.nan, dtype=np.complex128)
    valid = np.ones(n, dtype=bool)
    tangent: Optional[dict] = None       # only populated in sparse branch
    tangent_rank: Optional[np.ndarray] = None

    if kind == "dense":
        for c in range(n):
            Jc = J[c]
            if not np.all(np.isfinite(Jc)):
                valid[c] = False
                continue
            try:
                evals, V = np.linalg.eig(Jc)
            except np.linalg.LinAlgError:
                valid[c] = False
                continue
            evals, V = _conjugate_free(evals, V)
            # Apply sign convention per entry
            V_fixed = np.column_stack([_fix_eigvec_sign(V[:, k]) for k in range(V.shape[1])])
            eigvals[c] = _pad_to(evals, n_eig, dtype=np.complex128)
            for k in range(min(n_eig, V_fixed.shape[1])):
                eigvecs[c, k] = V_fixed[:, k]
    else:  # kind == "sparse"
        tangent = _ledger_tangent(ledger)
        if tangent is not None:
            tangent_rank = np.asarray(tangent.get("rank"))
            tangent_P = list(tangent.get("projector", []))
        else:
            tangent_rank = None
            tangent_P = []
        # op_eig_reason: per cell per eigenvalue slot. Default to "ok" for
        # populated slots, filled with "off_tangent" for slots past r_c.
        eig_reason = np.full((n, n_eig), "ok", dtype=object)
        for c in range(n):
            Jc = J[c]
            if Jc.nnz == 0:
                valid[c] = False
                eig_reason[c, :] = "no_variation"
                continue
            # Optional Fisher mask: zero data entries below threshold.
            if applied_mask and F_list is not None:
                Fc = F_list[c]
                if Fc.nnz == Jc.nnz and np.array_equal(Fc.indices, Jc.indices) \
                        and np.array_equal(Fc.indptr, Jc.indptr):
                    f_data = Fc.data
                else:
                    f_data = np.asarray(Fc[Jc.nonzero()[0], Jc.nonzero()[1]]).ravel()
                if min_fisher is None:
                    thr_c = float(np.percentile(f_data, 10)) if f_data.size else 0.0
                else:
                    thr_c = float(min_fisher)
                mask = f_data < thr_c
                if mask.any():
                    Jc = Jc.copy()
                    Jc.data[mask] = 0.0
                    Jc.eliminate_zeros()
                threshold_used = thr_c if threshold_used is None else threshold_used

            # Tangent restriction: eigendecompose P_c J_c P_c^T, lift back
            # through P_c. Off-tangent eigvalue slots are left NaN + reason
            # ``off_tangent``. See docs/opspace_figure_spec.md §9 fix (1).
            if tangent is not None and c < len(tangent_P):
                P_c = np.asarray(tangent_P[c], dtype=np.float64)       # (r_c, G)
                r_c = int(P_c.shape[0])
                if r_c == 0:
                    valid[c] = False
                    eig_reason[c, :] = "off_tangent"
                    continue
                # Materialise P_c J_c P_c^T  (r_c × r_c) via sparse multiplies.
                PJ = Jc.T.dot(P_c.T).T          # (r_c, G)
                block = PJ @ P_c.T              # (r_c, r_c)
                if not np.all(np.isfinite(block)):
                    valid[c] = False
                    eig_reason[c, :] = "off_tangent"
                    continue
                try:
                    evals, V = np.linalg.eig(block)       # (r_c,), (r_c, r_c)
                except np.linalg.LinAlgError:
                    valid[c] = False
                    eig_reason[c, :] = "off_tangent"
                    continue
                evals, V = _conjugate_free(evals, V)
                # Fisher-mass tie-break on near-identical Re. Mass per eigvec
                # k: v^H diag(F_c) v where F_c is the per-gene Fisher mass
                # (sum over rows of F) evaluated after lifting.
                if F_list is not None and evals.size >= 2:
                    V_full = P_c.T @ V                    # (G, r_c') complex
                    Fc_diag = np.asarray(F_list[c].sum(axis=0)).ravel()
                    mass = np.array([
                        float(np.sum(Fc_diag * (np.abs(V_full[:, k]) ** 2)))
                        for k in range(V.shape[1])
                    ])
                    # Stable sort: primary key -Re, tie-breaker -mass.
                    order = np.lexsort((-mass, -evals.real))
                    evals = evals[order]
                    V = V[:, order]
                V_fixed = np.column_stack([_fix_eigvec_sign(V[:, k]) for k in range(V.shape[1])])
                eigvals[c] = _pad_to(evals, n_eig, dtype=np.complex128)
                lifted = (P_c.T @ V_fixed).astype(np.complex128)   # (G, kept)
                # Fix sign on the lifted vectors (G-dim is the public space).
                lifted_fixed = np.column_stack([
                    _fix_eigvec_sign(lifted[:, k]) for k in range(lifted.shape[1])
                ])
                for k in range(min(n_eig, lifted_fixed.shape[1])):
                    eigvecs[c, k] = lifted_fixed[:, k]
                # Mark slots past r_c as off_tangent (empty, documented).
                if r_c < n_eig:
                    eig_reason[c, r_c:] = "off_tangent"
                continue

            # Fallback (no tangent restriction): operate on the live block.
            live = _live_cols_of(Jc)
            if live.size < 1:
                valid[c] = False
                eig_reason[c, :] = "no_variation"
                continue
            block = _dense_live_block(Jc, live)
            if not np.all(np.isfinite(block)):
                valid[c] = False
                continue
            try:
                evals, V = np.linalg.eig(block)
            except np.linalg.LinAlgError:
                valid[c] = False
                continue
            evals, V = _conjugate_free(evals, V)
            V_fixed = np.column_stack([_fix_eigvec_sign(V[:, k]) for k in range(V.shape[1])])
            eigvals[c] = _pad_to(evals, n_eig, dtype=np.complex128)
            for k in range(min(n_eig, V_fixed.shape[1])):
                v_full = np.zeros(dim, dtype=np.complex128)
                v_full[live] = V_fixed[:, k]
                eigvecs[c, k] = v_full

    adata.obsm["op_eigvals"] = eigvals
    adata.uns.setdefault("scjdo2", {})
    adata.uns["scjdo2"]["op_eigvecs"] = eigvecs
    adata.uns["scjdo2"]["op_valid"] = valid
    if kind == "sparse":
        adata.uns["scjdo2"]["op_eig_reason"] = eig_reason
        if tangent_rank is not None:
            tr = np.asarray(tangent_rank, dtype=np.int32)
            adata.uns["scjdo2"]["op_tangent_rank"] = tr
            adata.obs["op_rank"] = tr
    adata.uns["scjdo2"]["spectrum_meta"] = {
        "n_eig": int(n_eig),
        "fisher_mask": bool(applied_mask),
        "min_fisher": threshold_used,
        "J_kind": kind,
        "tangent_restricted": bool(tangent is not None and kind == "sparse"),
        "smooth_k": int(smooth_k),
    }
    # Restore the original J if we staged a smoothed copy.
    if orig_J is not None:
        adata.uns["scjdo2"]["J"] = orig_J


# ======================================================================
# 2.2 invariants
# ======================================================================


def _default_near_zero_tol(lead_re: np.ndarray) -> float:
    finite = lead_re[np.isfinite(lead_re)]
    if finite.size == 0:
        return 0.1
    return float(0.1 * np.std(finite))


def invariants(
    adata: AnnData,
    transient_horizon: Optional[int] = None,
    tol: Optional[float] = None,
    horizon: Optional[int] = None,
) -> None:
    """Compute per-cell operator invariants and write to ``adata.obs``.

    Columns written:
        ``lead_re``, ``lead_im``, ``n_stable``, ``n_unstable``,
        ``n_near_zero``, ``n_complex_pairs``, ``spectral_gap``,
        ``henrici``, ``transient_gain``, ``reactivity``,
        ``eigvec_cond``, ``op_confidence``.

    Parameters
    ----------
    tol
        Near-zero threshold for ``lead_re``. Default: ``0.1 · std(lead_re)``
        (``_default_near_zero_tol``), matching :func:`behaviour`.
    horizon, transient_horizon
        Number of steps for the transient-gain integration. ``horizon``
        is the §1.2 spec name; ``transient_horizon`` is the legacy alias
        kept for callers already in the wild. Default: ``5 / median
        |lead_re|`` of the finite leading-eigenvalue real parts (per
        docs/scJDO2_figure_layer_build_instructions.md §1.2), clamped to
        ``[5, 200]`` so pathological fields can't blow up the integration.
    """
    if "op_eigvals" not in adata.obsm:
        raise RuntimeError("Call spectrum(adata) before invariants(adata)")

    J, kind, dim = _get_J(adata)
    eigvals = np.asarray(adata.obsm["op_eigvals"])
    valid = np.asarray(adata.uns["scjdo2"]["op_valid"])
    eigvecs = np.asarray(adata.uns["scjdo2"]["op_eigvecs"])
    n, n_eig = eigvals.shape

    lead_re = np.full(n, np.nan)
    lead_im = np.full(n, np.nan)
    n_stable = np.zeros(n, dtype=np.int32)
    n_unstable = np.zeros(n, dtype=np.int32)
    n_near_zero = np.zeros(n, dtype=np.int32)
    n_complex_pairs = np.zeros(n, dtype=np.int32)
    spectral_gap = np.full(n, np.nan)
    henrici = np.full(n, np.nan)
    transient_gain = np.full(n, np.nan)
    reactivity = np.full(n, np.nan)
    eigvec_cond = np.full(n, np.nan)

    first_valid_re = eigvals.real
    lead_re_arr = first_valid_re[:, 0] if n_eig else first_valid_re.ravel()
    if tol is None:
        tol = _default_near_zero_tol(lead_re_arr)
    # §1.2 default horizon: 5 / median |lead_re|, clamped to [5, 200].
    eff_horizon = horizon if horizon is not None else transient_horizon
    if eff_horizon is None:
        finite_re = lead_re_arr[np.isfinite(lead_re_arr)]
        med = float(np.median(np.abs(finite_re))) if finite_re.size else 0.0
        if med > 0:
            eff_horizon = int(np.clip(round(5.0 / med), 5, 200))
        else:
            eff_horizon = 30
    eff_horizon = int(eff_horizon)

    for c in range(n):
        if not valid[c]:
            continue
        evals = eigvals[c]
        finite_mask = np.isfinite(evals.real) & np.isfinite(evals.imag)
        if not finite_mask.any():
            continue
        ev = evals[finite_mask]
        lead_re[c] = float(ev[0].real)
        lead_im[c] = float(ev[0].imag)
        n_stable[c] = int(np.sum(ev.real < -tol))
        n_unstable[c] = int(np.sum(ev.real > tol))
        n_near_zero[c] = int(np.sum(np.abs(ev.real) <= tol))
        n_complex_pairs[c] = int(np.sum(ev.imag > _EPS_IM))
        if ev.size >= 2:
            spectral_gap[c] = float(ev[0].real - ev[1].real)

        if kind == "dense":
            Jc = J[c]
            if np.all(np.isfinite(Jc)):
                henrici[c] = _henrici(Jc, ev)
                reactivity[c] = _reactivity(Jc, ev)
                transient_gain[c] = _transient_gain(Jc.astype(np.float64), n_max=eff_horizon)
                try:
                    _, V_full = np.linalg.eig(Jc)
                    eigvec_cond[c] = _eigvec_cond(V_full)
                except np.linalg.LinAlgError:
                    eigvec_cond[c] = np.inf
        else:  # sparse: operate on the live block
            Jc = J[c]
            live = _live_cols_of(Jc)
            if live.size == 0:
                continue
            block = _dense_live_block(Jc, live)
            if np.all(np.isfinite(block)):
                henrici[c] = _henrici(block, ev)
                reactivity[c] = _reactivity(block, ev)
                transient_gain[c] = _transient_gain(block, n_max=eff_horizon)
                try:
                    _, V_full = np.linalg.eig(block)
                    eigvec_cond[c] = _eigvec_cond(V_full)
                except np.linalg.LinAlgError:
                    eigvec_cond[c] = np.inf

    adata.uns.setdefault("scjdo2", {})
    adata.uns["scjdo2"]["invariants_meta"] = {
        "tol": float(tol),
        "horizon": int(eff_horizon),
    }
    adata.obs["lead_re"] = lead_re
    adata.obs["lead_im"] = lead_im
    adata.obs["n_stable"] = n_stable
    adata.obs["n_unstable"] = n_unstable
    adata.obs["n_near_zero"] = n_near_zero
    adata.obs["n_complex_pairs"] = n_complex_pairs
    adata.obs["spectral_gap"] = spectral_gap
    adata.obs["henrici"] = henrici
    adata.obs["transient_gain"] = transient_gain
    adata.obs["reactivity"] = reactivity
    adata.obs["eigvec_cond"] = eigvec_cond

    # op_confidence (spec §2.2): median Fisher of entries whose |eigvec
    # weight| > 5 % of max. On Phase B latent (no ledger) we fall back to
    # 1.0 everywhere so the plot layer stays functional.
    ledger = _get_ledger(adata)
    if ledger is None or not hasattr(ledger, "op_confidence_from_eigvec"):
        adata.obs["op_confidence"] = 1.0
    else:
        # Leading eigvec per cell (|Im|≥0 member stored at index 0).
        leading = eigvecs[:, 0, :]
        conf = ledger.op_confidence_from_eigvec(leading)
        if np.all(np.isnan(conf)):
            adata.obs["op_confidence"] = 1.0
        else:
            adata.obs["op_confidence"] = np.where(np.isfinite(conf), conf, 0.0)


# ======================================================================
# 2.3 behaviour
# ======================================================================


def behaviour(adata: AnnData, tol: Optional[float] = None,
              sensitive_pct: float = 90.0,
              sensitive_quantile: Optional[float] = None) -> None:
    """Classify each cell's leading mode and flag non-normality.

    Behaviour categories: holder, drift, fork, damped_oscillator, growing_oscillator.
    Also writes a boolean ``op_sensitive`` where ``henrici`` is above the
    ``sensitive_pct``'th percentile of finite ``henrici`` values.

    ``sensitive_pct`` is on the 0–100 scale (§1.3). The legacy
    ``sensitive_quantile`` kwarg (0–1 scale) is kept for back-compat;
    when supplied it overrides ``sensitive_pct``.
    """
    q = (
        float(sensitive_quantile)
        if sensitive_quantile is not None
        else float(sensitive_pct) / 100.0
    )
    sensitive_quantile = q  # for the rest of the function
    for col in ("lead_re", "lead_im", "henrici"):
        if col not in adata.obs:
            raise RuntimeError(f"Call invariants(adata) before behaviour(adata) (missing {col})")

    lead_re = adata.obs["lead_re"].to_numpy()
    lead_im = adata.obs["lead_im"].to_numpy()
    henrici = adata.obs["henrici"].to_numpy()
    if tol is None:
        tol = _default_near_zero_tol(lead_re)

    cats = np.full(adata.n_obs, "", dtype=object)
    real_mode = np.abs(lead_im) <= _EPS_IM
    cats[real_mode & (lead_re < -tol)] = "holder"
    cats[real_mode & (lead_re > tol)] = "fork"
    cats[real_mode & (np.abs(lead_re) <= tol)] = "drift"
    cats[~real_mode & (lead_re < -tol)] = "damped_oscillator"
    cats[~real_mode & (lead_re > tol)] = "growing_oscillator"
    cats[~real_mode & (np.abs(lead_re) <= tol)] = "drift"
    cats[np.isnan(lead_re)] = ""

    cat = pd.Categorical(cats, categories=["", *BEHAVIOURS])
    adata.obs["op_behaviour"] = cat

    sens_thr = float(np.nanquantile(henrici[~np.isnan(henrici)], sensitive_quantile)) \
        if np.isfinite(henrici).any() else np.nan
    adata.obs["op_sensitive"] = (
        np.isfinite(henrici) & (henrici >= sens_thr) if np.isfinite(sens_thr) else False
    )
    adata.uns.setdefault("scjdo2", {})
    adata.uns["scjdo2"]["behaviour_tol"] = float(tol)
    adata.uns["scjdo2"]["sensitive_threshold"] = float(sens_thr) if np.isfinite(sens_thr) else None


# ======================================================================
# 2.4 drivers — Phase C
# ======================================================================


def drivers(adata: AnnData, which: Literal["lead", "archetype"] = "lead",
            top: int = 15, min_fisher: Optional[float] = None,
            cell: Optional[int] = None,
            archetype_id: Optional[int] = None,
            use_lift: Optional[bool] = None) -> pd.DataFrame:
    """Tidy DataFrame of driver genes for a cell's leading eigenvector or
    an archetype operator.

    Per spec §2.4 the eigvec phase is fixed so the largest-|magnitude|
    entry is real and positive (:func:`_fix_eigvec_sign`). ``weight``
    here is the real part of that phase-fixed eigvec; ``magnitude`` and
    ``phase`` describe the full complex entry so complex-mode drivers
    can render the magnitude × phase bars from spec §3.1 Panel C.

    Columns: ``gene``, ``weight`` (signed real part), ``magnitude``,
    ``phase`` (radians; NaN for pure-real modes), ``fisher``,
    ``fisher_se`` (``1/sqrt(F)``), ``reason``.

    Phase B latent-J adata: not supported (no gene-space eigenvectors);
    raises :class:`NotImplementedError`.

    ``which="lead"`` + ``cell=i`` picks the leading eigvec of cell ``i``
    (defaults to the fork cell with the highest ``lead_re`` if omitted).
    ``which="archetype"`` + ``archetype_id`` requires the archetype
    operator spectrum to be populated by :func:`archetype_operators`.

    V2_PLAN R15. When ``use_lift`` is ``None`` (default), the function
    uses ``uns["scjdo2"]["op_eigvec_lifted"]`` if present — the gene-
    space eigenvector obtained from the chain-rule lift
    ``w_c = ∇η(z_c) · v_latent_lead(c)`` — in place of the independent
    gene-space eigendecomposition. This matches the R15 design rule:
    behaviour comes from the latent operator; Panel C drivers come from
    the gene-space view of the latent leading mode. Set ``use_lift=False``
    to force the pre-R15 behaviour (reads ``uns["scjdo2"]["op_eigvecs"]``
    directly); ``use_lift=True`` requires the lifted eigvec to exist and
    errors otherwise.
    """
    J, kind, dim = _get_J(adata)
    if kind != "sparse":
        raise NotImplementedError(
            "drivers() requires a gene-space Jacobian (Phase C per-cell CSR J). "
            f"Current J is dense latent-space ({dim} dims vs {adata.n_vars} genes). "
            "Run the Phase C rate model + fit_gene_space_jacobians first."
        )
    if dim != adata.n_vars:
        raise RuntimeError(
            f"Gene-space J has dim={dim} but adata.n_vars={adata.n_vars}; "
            "gene-name index mismatch."
        )
    uns_sc = adata.uns.get("scjdo2", {})
    if "op_eigvecs" not in uns_sc and "op_eigvec_lifted" not in uns_sc:
        raise RuntimeError(
            "Call spectrum(adata) (or compute(adata) for R15) before drivers(adata)"
        )
    eigvecs = (
        np.asarray(uns_sc["op_eigvecs"]) if "op_eigvecs" in uns_sc
        else None
    )
    # R15 preference: use the ∇η lift of the latent lead mode when available.
    lifted = uns_sc.get("op_eigvec_lifted")
    if use_lift is None:
        use_lift = lifted is not None
    elif use_lift and lifted is None:
        raise RuntimeError(
            "drivers(use_lift=True) needs uns['scjdo2']['op_eigvec_lifted']; "
            "run opspace.compute(adata) after storing grad_eta."
        )

    if which == "archetype":
        arch = adata.uns.get("scjdo2", {}).get("archetypes")
        if arch is None:
            raise RuntimeError(
                "drivers(which='archetype') needs archetype_operators() to have run first."
            )
        if archetype_id is None:
            raise ValueError("drivers(which='archetype') requires archetype_id=...")
        v = np.asarray(arch[archetype_id]["eigvec"], dtype=np.complex128)
        fisher = np.asarray(arch[archetype_id].get("fisher"), dtype=np.float64)
        if fisher.shape != v.shape:
            fisher = np.full(v.shape, np.nan)
        reasons = np.asarray(arch[archetype_id].get("reason", ["ok"] * v.size))
    else:
        if cell is None:
            # Default: fork cell with the highest lead_re
            if "op_behaviour" in adata.obs and "lead_re" in adata.obs:
                mask = adata.obs["op_behaviour"].astype(str) == "fork"
                lead = adata.obs["lead_re"].to_numpy()
                lead_m = np.where(mask, lead, -np.inf)
                cell = int(np.argmax(lead_m)) if mask.any() else int(np.nanargmax(lead))
            else:
                raise ValueError(
                    "drivers(which='lead') with no `cell=` needs behaviour/lead_re"
                )
        if use_lift and lifted is not None:
            v = np.asarray(lifted)[cell, :]
        else:
            v = eigvecs[cell, 0, :]
        # Fetch per-entry Fisher at the live-gene indices of this cell's J.
        Jc = J[cell]
        live = _live_cols_of(Jc)
        ledger = _get_ledger(adata)
        F_list = _ledger_fisher_list(ledger)
        fisher = np.full(dim, np.nan)
        if F_list is not None:
            Fc = F_list[cell]
            diag_vals = Fc.diagonal()
            fisher[live] = diag_vals[live]
        reasons = np.array(["no_variation"] * dim, dtype=object)
        reasons[live] = "ok"

    # Sign is already fixed by spectrum(); recompute as a safeguard.
    v = _fix_eigvec_sign(v)
    magnitude = np.abs(v)
    weight = v.real
    phase = np.where(magnitude > 0, np.angle(v), np.nan)
    is_complex = bool(np.max(np.abs(v.imag)) > _EPS_IM)
    if not is_complex:
        phase = np.full_like(magnitude, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        fisher_se = 1.0 / np.sqrt(fisher)
    fisher_se = np.where(np.isfinite(fisher_se), fisher_se, np.nan)

    gene_names = (
        np.asarray(adata.var_names)
        if adata.var_names is not None
        else np.array([f"gene_{i}" for i in range(dim)])
    )
    df = pd.DataFrame({
        "gene": gene_names,
        "weight": weight,
        "magnitude": magnitude,
        "phase": phase,
        "fisher": fisher,
        "fisher_se": fisher_se,
        "reason": reasons,
    })
    if min_fisher is not None:
        df = df[df["fisher"] >= float(min_fisher)]
    df = df.sort_values("magnitude", ascending=False).head(int(top)).reset_index(drop=True)
    return df


# ======================================================================
# 2.5 archetype_operators — Phase B Step 8 adapter not yet implemented
# ======================================================================


def archetype_operators(adata: AnnData) -> None:
    """Fisher-weighted mean of ``J_c`` per archetype; apply 2.1–2.4 to each.

    Pending :mod:`scjdo2.archetypes` adapter around v1's ``_seminmf`` (Phase B
    Step 8) and the Fisher ledger (Phase C). Skipped with a clear error until
    then.
    """
    raise NotImplementedError(
        "archetype_operators() needs scjdo2.archetypes semi-NMF adapter and "
        "the Fisher ledger. Scheduled with Phase B Step 8 and Phase C."
    )


# ======================================================================
# 2.6 field_gradient
# ======================================================================


def field_gradient(adata: AnnData, k: int = 15, use_rep: str = "X_pca") -> None:
    """Mean Frobenius distance between ``J_c`` and its k latent-neighbours' J,
    normalised by ``||J_c||_F``. Writes ``adata.obs["op_field_gradient"]``.
    """
    J, kind, _dim = _get_J(adata)
    if use_rep not in adata.obsm:
        raise KeyError(f"adata.obsm[{use_rep!r}] not found")
    z = np.asarray(adata.obsm[use_rep], dtype=np.float32)
    n = adata.n_obs
    kk = min(int(k) + 1, n)
    nn = NearestNeighbors(n_neighbors=kk).fit(z)
    _, idx = nn.kneighbors(z, return_distance=True)
    idx = idx[:, 1:]

    def _frob(M) -> float:
        if issparse(M):
            return float(np.sqrt((M.multiply(M)).sum()))
        return float(np.linalg.norm(M, ord="fro"))

    out = np.full(n, np.nan)
    for c in range(n):
        Jc = J[c]
        if kind == "dense" and not np.all(np.isfinite(Jc)):
            continue
        if kind == "sparse" and Jc.nnz == 0:
            continue
        denom = _frob(Jc) + 1e-12
        diffs = []
        for j in idx[c]:
            Jj = J[j]
            if kind == "dense" and not np.all(np.isfinite(Jj)):
                continue
            if kind == "sparse" and Jj.nnz == 0:
                continue
            diffs.append(_frob(Jj - Jc))
        if diffs:
            out[c] = float(np.mean(diffs)) / denom
    adata.obs["op_field_gradient"] = out


# ======================================================================
# 2.7 lineage_path
# ======================================================================


def lineage_path(adata: AnnData,
                 start_cells: Sequence[int],
                 n_steps: int = 200,
                 dt: Optional[float] = None,
                 use_rep: str = "X_pca",
                 velocity_key: str = "X_drift_v") -> list[dict]:
    """Integrate the drift from each start cell; record nearest-cell
    properties at each step.

    Returns a list (one entry per start cell), each a dict with:
        ``cells``        : (n_steps + 1,) int — nearest-cell index at each step
        ``z``            : (n_steps + 1, d) latent trajectory
        ``lead_re``      : (n_steps + 1,) float
        ``lead_im``      : (n_steps + 1,) float
        ``op_behaviour`` : (n_steps + 1,) str
    """
    if velocity_key not in adata.obsm:
        raise KeyError(
            f"adata.obsm[{velocity_key!r}] not found; run the pipeline first."
        )
    z = np.asarray(adata.obsm[use_rep], dtype=np.float32)
    v = np.asarray(adata.obsm[velocity_key], dtype=np.float32)
    lead_re = adata.obs["lead_re"].to_numpy()
    lead_im = adata.obs["lead_im"].to_numpy()
    beh = adata.obs["op_behaviour"].astype(str).to_numpy()

    nn = NearestNeighbors(n_neighbors=1).fit(z)
    if dt is None:
        dt = float(np.median(np.linalg.norm(v, axis=1))) ** 0  # placeholder, use 1.0
        dt = 1.0
    paths = []
    for sc in start_cells:
        traj = np.zeros((n_steps + 1, z.shape[1]), dtype=np.float32)
        cells = np.zeros(n_steps + 1, dtype=np.int64)
        traj[0] = z[sc]
        cells[0] = sc
        for step in range(1, n_steps + 1):
            # nearest-cell velocity, Euler step
            _, j = nn.kneighbors(traj[step - 1][None], return_distance=True)
            cells[step] = int(j[0, 0])
            traj[step] = traj[step - 1] + dt * v[cells[step]]
        paths.append({
            "start": int(sc),
            "cells": cells,
            "z": traj,
            "lead_re": lead_re[cells],
            "lead_im": lead_im[cells],
            "op_behaviour": beh[cells],
        })
    return paths


# ======================================================================
# 2.8 export_tables
# ======================================================================


def _axis_tag(adata: AnnData) -> str:
    return "" if _is_clock_calibrated(adata) else "_relative"


def export_tables(adata: AnnData, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = _axis_tag(adata)

    cols = {
        f"lead_re{tag}": adata.obs["lead_re"],
        f"lead_im{tag}": adata.obs["lead_im"],
        "n_stable": adata.obs["n_stable"],
        "n_unstable": adata.obs["n_unstable"],
        "n_near_zero": adata.obs["n_near_zero"],
        "n_complex_pairs": adata.obs["n_complex_pairs"],
        f"spectral_gap{tag}": adata.obs.get("spectral_gap", np.nan),
        "henrici": adata.obs.get("henrici", np.nan),
        "transient_gain": adata.obs.get("transient_gain", np.nan),
        "reactivity": adata.obs.get("reactivity", np.nan),
        "eigvec_cond": adata.obs.get("eigvec_cond", np.nan),
        "op_behaviour": adata.obs.get("op_behaviour", np.nan),
        "op_sensitive": adata.obs.get("op_sensitive", False),
        "op_field_gradient": adata.obs.get("op_field_gradient", np.nan),
        "op_confidence": adata.obs.get("op_confidence", 1.0),
    }
    df = pd.DataFrame(cols, index=adata.obs_names)
    inv_path = out / "per_cell_invariants.csv"
    df.to_csv(inv_path)

    paths = {"per_cell_invariants": inv_path}

    archetypes = adata.uns.get("scjdo2", {}).get("archetypes")
    if archetypes is not None:
        arch_rows = []
        for aid, a in enumerate(archetypes):
            arch_rows.append({
                "archetype_id": aid,
                "behaviour": a.get("behaviour"),
                f"lead_re{tag}": a.get("lead_re"),
                f"lead_im{tag}": a.get("lead_im"),
                "henrici": a.get("henrici"),
                "transient_gain": a.get("transient_gain"),
                "n_cells": a.get("n_cells"),
                "mean_confidence": a.get("mean_confidence"),
            })
        p = out / "per_archetype_summary.csv"
        pd.DataFrame(arch_rows).to_csv(p, index=False)
        paths["per_archetype_summary"] = p

    return paths


# ======================================================================
# Back-compat: the (J-list) `compute()` entry point from Phase B first pass
# ======================================================================


def compute_from_J_list(
    J, ledger=None, n_leading: int = 2,
    near_zero_tol: float = 1e-3, transient_n_max: int = 30,
) -> OpCoordinates:
    """Legacy entry point that returns :class:`OpCoordinates` for a list or
    ``(n, d, d)`` array of Jacobians. Prefer :func:`compute` (the AnnData-
    centric §1 wrapper) in new code; this function keeps the Phase B
    first-pass contract for callers already in the wild.
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
        evals, V = np.linalg.eig(Jc)
        order = np.argsort(-evals.real)
        evals = evals[order]
        V = V[:, order]
        eigvals[i] = evals
        n_lead = min(n_leading, V.shape[1])
        leading[i, :n_lead] = V[:, :n_lead].T

        re = evals.real
        im = evals.imag
        max_real_eig[i] = float(re[0])
        if len(re) >= 2:
            spectral_gap[i] = float(re[0] - re[1])
        n_unstable[i] = int(np.sum(re > near_zero_tol))
        n_stable[i] = int(np.sum(re < -near_zero_tol))
        n_near_zero[i] = int(np.sum(np.abs(re) <= near_zero_tol))
        n_complex_pairs[i] = int(np.sum(im > 0))

        henrici[i] = _henrici(Jc, evals)
        reactivity[i] = _reactivity(Jc, evals)
        transient_gain[i] = _transient_gain(Jc.astype(np.float64), n_max=transient_n_max)
        eigvec_cond[i] = _eigvec_cond(V)

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
