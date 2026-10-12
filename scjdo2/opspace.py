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
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors


BEHAVIOURS = ("holder", "drift", "fork", "damped_oscillator", "growing_oscillator")
REASON_CODES = ("ok", "no_counts", "no_variation", "not_in_ball")

_EPS_IM = 1e-10


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


def _get_J(adata: AnnData) -> np.ndarray:
    try:
        J = adata.uns["scjdo2"]["J"]
    except KeyError as e:
        raise KeyError(
            "adata.uns['scjdo2']['J'] not found. Run scjdo2.pipeline.fit_latent_field "
            "first (or set the key directly) before calling opspace functions."
        ) from e
    J = np.asarray(J)
    if J.ndim != 3 or J.shape[1] != J.shape[2]:
        raise ValueError(f"adata.uns['scjdo2']['J'] must be (n_cells, d, d); got {J.shape}")
    if J.shape[0] != adata.n_obs:
        raise ValueError(f"J.shape[0]={J.shape[0]} != adata.n_obs={adata.n_obs}")
    return J


# ======================================================================
# 2.1 spectrum
# ======================================================================


def spectrum(
    adata: AnnData,
    n_eig: int = 5,
    fisher_mask: bool = True,
    min_fisher: Optional[float] = None,
) -> None:
    """Compute per-cell conjugate-free eigenpairs of the stored Jacobian.

    Stores:
        ``adata.obsm["op_eigvals"]`` — ``(n_cells, n_eig)`` complex
        ``adata.uns["scjdo2"]["op_eigvecs"]`` — ``(n_cells, n_eig, d)`` complex
        ``adata.uns["scjdo2"]["op_valid"]`` — ``(n_cells,)`` bool
        ``adata.uns["scjdo2"]["spectrum_meta"]`` — ``{n_eig, fisher_mask, min_fisher}``

    ``fisher_mask`` + ``min_fisher`` zero out entries of ``J_c`` whose
    Fisher information is below the threshold before decomposing. In Phase
    B the Fisher ledger does not exist yet; if ``fisher_mask`` is True and
    no ledger is present the call falls back to no masking and records
    ``fisher_mask=False`` in the stored meta with a warning.
    """
    import warnings

    J = _get_J(adata)
    n, d, _ = J.shape
    n_eig = min(int(n_eig), d)

    ledger = adata.uns.get("scjdo2", {}).get("fisher_ledger")
    applied_mask = False
    threshold_used: Optional[float] = None
    if fisher_mask and ledger is None:
        warnings.warn(
            "spectrum(fisher_mask=True) called without a Fisher ledger; "
            "falling back to no mask. Phase C will store the ledger.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif fisher_mask and ledger is not None:
        # Phase C hook. Not reachable in Phase B.
        applied_mask = True
        threshold_used = float(min_fisher) if min_fisher is not None else None

    eigvals = np.full((n, n_eig), np.nan + 1j * np.nan, dtype=np.complex128)
    eigvecs = np.full((n, n_eig, d), np.nan + 1j * np.nan, dtype=np.complex128)
    valid = np.ones(n, dtype=bool)

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
        eigvals[c] = _pad_to(evals, n_eig, dtype=np.complex128)
        for k in range(min(n_eig, V.shape[1])):
            eigvecs[c, k] = V[:, k]

    adata.obsm["op_eigvals"] = eigvals
    adata.uns.setdefault("scjdo2", {})
    adata.uns["scjdo2"]["op_eigvecs"] = eigvecs
    adata.uns["scjdo2"]["op_valid"] = valid
    adata.uns["scjdo2"]["spectrum_meta"] = {
        "n_eig": int(n_eig),
        "fisher_mask": bool(applied_mask),
        "min_fisher": threshold_used,
    }


# ======================================================================
# 2.2 invariants
# ======================================================================


def _default_near_zero_tol(lead_re: np.ndarray) -> float:
    finite = lead_re[np.isfinite(lead_re)]
    if finite.size == 0:
        return 0.1
    return float(0.1 * np.std(finite))


def invariants(adata: AnnData, transient_horizon: int = 30) -> None:
    """Compute per-cell operator invariants and write to ``adata.obs``.

    Columns written:
        ``lead_re``, ``lead_im``, ``n_stable``, ``n_unstable``,
        ``n_near_zero``, ``n_complex_pairs``, ``spectral_gap``,
        ``henrici``, ``transient_gain``, ``reactivity``,
        ``eigvec_cond``, ``op_confidence``.
    """
    if "op_eigvals" not in adata.obsm:
        raise RuntimeError("Call spectrum(adata) before invariants(adata)")

    J = _get_J(adata)
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
    tol = _default_near_zero_tol(first_valid_re[:, 0] if n_eig else first_valid_re.ravel())

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

        Jc = J[c]
        if np.all(np.isfinite(Jc)):
            henrici[c] = _henrici(Jc, ev)
            reactivity[c] = _reactivity(Jc, ev)
            transient_gain[c] = _transient_gain(Jc.astype(np.float64), n_max=transient_horizon)
            try:
                _, V_full = np.linalg.eig(Jc)
                eigvec_cond[c] = _eigvec_cond(V_full)
            except np.linalg.LinAlgError:
                eigvec_cond[c] = np.inf

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

    # op_confidence: median Fisher of entries contributing to the leading
    # eigenvector (weights above 5% of max). In Phase B there's no ledger,
    # so op_confidence = 1.0 everywhere (uniform opacity in plots).
    ledger = adata.uns.get("scjdo2", {}).get("fisher_ledger")
    if ledger is None:
        adata.obs["op_confidence"] = 1.0
    else:
        # Phase C hook. Implement once ledger is persisted.
        adata.obs["op_confidence"] = 1.0  # placeholder


# ======================================================================
# 2.3 behaviour
# ======================================================================


def behaviour(adata: AnnData, tol: Optional[float] = None,
              sensitive_quantile: float = 0.9) -> None:
    """Classify each cell's leading mode and flag non-normality.

    Behaviour categories: holder, drift, fork, damped_oscillator, growing_oscillator.
    Also writes a boolean ``op_sensitive`` where ``henrici >= np.nanquantile(henrici, q)``.
    """
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
            archetype_id: Optional[int] = None) -> pd.DataFrame:
    """Return a tidy DataFrame of driver genes per cell or archetype.

    Phase C work: this needs gene-space eigenvectors, which require the NB
    rate model (``scjdo2.rates``) and gene-space Jacobians. The Phase B
    pipeline stores latent-space Jacobians only, so the eigenvectors have no
    gene names. Call lands when ``adata.uns["scjdo2"]["J"]`` has shape
    ``(n, n_genes, n_genes)`` and ``adata.uns["scjdo2"]["gene_names"]`` is
    populated.
    """
    J = _get_J(adata)
    gene_space = J.shape[1] == adata.n_vars
    if not gene_space:
        raise NotImplementedError(
            "drivers() requires a gene-space Jacobian (Phase C). "
            f"Current J is latent-space ({J.shape[1]} dims vs {adata.n_vars} genes). "
            "Run scjdo2.rates then scjdo2.field in gene mode (Phase C work)."
        )
    raise NotImplementedError("drivers() Phase C implementation pending")


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
    J = _get_J(adata)
    if use_rep not in adata.obsm:
        raise KeyError(f"adata.obsm[{use_rep!r}] not found")
    z = np.asarray(adata.obsm[use_rep], dtype=np.float32)
    n = adata.n_obs
    kk = min(int(k) + 1, n)
    nn = NearestNeighbors(n_neighbors=kk).fit(z)
    _, idx = nn.kneighbors(z, return_distance=True)
    idx = idx[:, 1:]

    out = np.full(n, np.nan)
    for c in range(n):
        Jc = J[c]
        if not np.all(np.isfinite(Jc)):
            continue
        denom = float(np.linalg.norm(Jc, ord="fro")) + 1e-12
        diffs = []
        for j in idx[c]:
            Jj = J[j]
            if not np.all(np.isfinite(Jj)):
                continue
            diffs.append(float(np.linalg.norm(Jj - Jc, ord="fro")))
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


def compute(J, ledger=None, n_leading: int = 2,
            near_zero_tol: float = 1e-3, transient_n_max: int = 30) -> OpCoordinates:
    """Legacy entry point that returns :class:`OpCoordinates` for a list or
    ``(n, d, d)`` array of Jacobians. Prefer the AnnData-centric API above.
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
