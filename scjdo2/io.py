"""On-disk formats for scJDO v2.

AnnData is the primary container. For the Phase C artefacts that are
not natively h5ad-serialisable (per-cell CSR gene-space J, Fisher
ledger, Fisher-weighted tangent projectors, ``grad_eta``, dense latent
J, complex eigenpairs) a sidecar ``phase_c.npz`` lives alongside
``adata.h5ad``. ``save`` and ``load`` round-trip both; opspace / plot
CLI commands read them without ever re-fitting.

Layout of ``<run_dir>/``::

    adata.h5ad      - the standard AnnData state (var, obs, obsm except
                      complex-dtype columns, layers, uns except the
                      Phase C sidecar keys listed below).
    phase_c.npz     - sidecar for the keys below. Written only when at
                      least one of these keys is populated in adata.

Sidecar keys (all optional; absent keys mean "nothing to restore"):

* ``J``                    — list of ``(G, G)`` sparse CSR. Stored
                             as concatenated ``data``/``indices``/
                             ``indptr`` arrays plus ``n_cells``,
                             ``n_genes``.
* ``fisher`` / ``F``       — same layout as J (ledger.fisher).
* ``reason_per_cell``      — ``(n, G)`` int8.
* ``thresholds``           — JSON-encoded dict.
* ``ridge_lambda*``        — scalars from ``fit_gene_space_jacobians``.
* ``tangent_P``            — concat of ``(r_c, G)`` projectors over all
                             cells + ``tangent_rank`` ``(n,)`` int.
* ``tangent_noise_floor``  — ``(n,)`` float.
* ``J_latent``             — dense ``(n, d, d)``.
* ``grad_eta``             — dense ``(n, G, d)``.
* ``op_eigvals``           — ``(n, n_eig)`` complex.
* ``op_eigvecs``           — ``(n, n_eig, dim)`` complex.
* ``op_valid``             — ``(n,)`` bool.
* ``op_eig_reason``        — ``(n, n_eig)`` object → stored as int codes
                             with a ``op_eig_reason_vocab`` str list.
* ``op_tangent_rank``      — ``(n,)`` int32.
* ``op_eigvec_latent_lead``— ``(n, d)`` complex.
* ``op_eigvec_lifted``     — ``(n, G)`` complex.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse


# Keys we strip from adata before h5ad write; restored on load.
_SIDECAR_UNS_KEYS: tuple[str, ...] = (
    "J",
    "J_latent",
    "fisher_ledger",
    "grad_eta",
    "op_eigvecs",
    "op_valid",
    "op_eig_reason",
    "op_tangent_rank",
    "op_eigvec_latent_lead",
    "op_eigvec_lifted",
    "archetypes",                 # dict[int, dict] with complex arrays + DataFrames
    "archetype_K_curve",          # list[tuple[int, float]]
    "archetype_K_selected",       # int
    "archetype_patterns_latent",  # (K, d, d) float
    "anchor_weights",             # (n, n_anchors) CSR
    "anchor_ids",                 # (n_anchors,) int
    "anchor_bandwidth",           # float
    "anchor_top_k",               # int
)

# obsm columns that h5ad can't write (complex dtype in particular).
_SIDECAR_OBSM_KEYS: tuple[str, ...] = ("op_eigvals",)


# ----------------------------------------------------------------- csr list


def _csr_list_to_arrays(J_list: list[csr_matrix]) -> dict:
    if not J_list:
        return {"n_cells": 0, "n_genes": 0}
    G = int(J_list[0].shape[0])
    data_list, idx_list, ptr_list = [], [], []
    for Jc in J_list:
        data_list.append(np.asarray(Jc.data))
        idx_list.append(np.asarray(Jc.indices, dtype=np.int32))
        ptr_list.append(np.asarray(Jc.indptr, dtype=np.int32))
    return {
        "n_cells": np.int32(len(J_list)),
        "n_genes": np.int32(G),
        "data": np.concatenate(data_list) if data_list else np.zeros(0),
        "indices": np.concatenate(idx_list) if idx_list else np.zeros(0, dtype=np.int32),
        "nnz_per_cell": np.array([Jc.data.size for Jc in J_list], dtype=np.int64),
        # per-cell indptrs stacked as (n, G+1): same number of rows per cell.
        "indptr_stack": np.stack(ptr_list).astype(np.int32),
    }


def _csr_list_from_arrays(d: dict) -> list[csr_matrix]:
    n = int(d["n_cells"])
    G = int(d["n_genes"])
    if n == 0:
        return []
    data = np.asarray(d["data"])
    indices = np.asarray(d["indices"], dtype=np.int32)
    indptr_stack = np.asarray(d["indptr_stack"], dtype=np.int32)
    nnz = np.asarray(d["nnz_per_cell"], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(nnz)])
    out: list[csr_matrix] = []
    for c in range(n):
        s, e = int(offsets[c]), int(offsets[c + 1])
        Jc = csr_matrix((data[s:e], indices[s:e], indptr_stack[c]), shape=(G, G))
        out.append(Jc)
    return out


# ------------------------------------------------------------- tangent list


def _tangent_to_arrays(tangent: dict) -> dict:
    P_list = list(tangent.get("projector", []))
    if not P_list:
        return {}
    ranks = np.array([int(P.shape[0]) for P in P_list], dtype=np.int32)
    G = int(P_list[0].shape[1])
    P_concat = (
        np.concatenate(P_list, axis=0).astype(np.float64)
        if len(P_list) else np.zeros((0, G))
    )
    out = {
        "P_concat": P_concat,
        "rank": ranks,
        "n_genes": np.int32(G),
    }
    nf = tangent.get("noise_floor")
    if nf is not None:
        out["noise_floor"] = np.asarray(nf, dtype=np.float64)
    ev = tangent.get("eigvals")
    if ev is not None:
        # Stack ragged eigval lists into (n, max_d) with NaN padding.
        max_d = max((len(e) for e in ev), default=0)
        arr = np.full((len(ev), max_d), np.nan, dtype=np.float64)
        for i, e in enumerate(ev):
            arr[i, :len(e)] = np.asarray(e)
        out["eigvals"] = arr
    for scalar_key in ("method", "n_perm", "quantile"):
        if scalar_key in tangent:
            out[scalar_key + "_tag"] = np.asarray(
                [str(tangent[scalar_key])], dtype=object,
            )
    return out


def _tangent_from_arrays(d: dict) -> dict:
    if not d or "rank" not in d:
        return {}
    ranks = np.asarray(d["rank"], dtype=np.int32)
    P_concat = np.asarray(d["P_concat"], dtype=np.float64)
    offsets = np.concatenate([[0], np.cumsum(ranks)])
    projectors = [
        P_concat[int(offsets[c]):int(offsets[c + 1])]
        for c in range(len(ranks))
    ]
    out: dict[str, Any] = {
        "projector": projectors,
        "rank": ranks,
    }
    if "noise_floor" in d:
        out["noise_floor"] = np.asarray(d["noise_floor"])
    if "eigvals" in d:
        ev_pad = np.asarray(d["eigvals"])
        out["eigvals"] = [row[np.isfinite(row)] for row in ev_pad]
    for scalar_key in ("method", "n_perm", "quantile"):
        tag_key = scalar_key + "_tag"
        if tag_key in d:
            v = np.asarray(d[tag_key]).item()
            try:
                out[scalar_key] = float(v) if scalar_key in ("n_perm", "quantile") else v
            except Exception:
                out[scalar_key] = v
    return out


# ----------------------------------------------------------------- fisher ledger


def _ledger_to_arrays(ledger) -> dict:
    out: dict[str, Any] = {}
    if ledger is None:
        return out
    fisher = (
        ledger.get("fisher") if isinstance(ledger, dict)
        else getattr(ledger, "fisher", None)
    )
    reason_per_cell = (
        ledger.get("reason_per_cell") if isinstance(ledger, dict)
        else getattr(ledger, "reason_per_cell", None)
    )
    off_tangent = (
        ledger.get("off_tangent") if isinstance(ledger, dict)
        else getattr(ledger, "off_tangent", None)
    )
    thresholds = (
        ledger.get("thresholds") if isinstance(ledger, dict)
        else getattr(ledger, "thresholds", {})
    )
    ridge_lambda = (
        ledger.get("ridge_lambda") if isinstance(ledger, dict)
        else getattr(ledger, "ridge_lambda", None)
    )

    if fisher is not None and len(fisher):
        out["fisher"] = _csr_list_to_arrays(list(fisher))
    if reason_per_cell is not None and len(reason_per_cell):
        out["reason_per_cell"] = np.stack(
            [np.asarray(r, dtype=np.int8) for r in reason_per_cell]
        )
    if off_tangent is not None:
        out["off_tangent"] = _tangent_to_arrays(off_tangent)
    if thresholds:
        out["thresholds_json"] = np.asarray([json.dumps(dict(thresholds))], dtype=object)
    if ridge_lambda is not None:
        out["ridge_lambda"] = np.float64(ridge_lambda)
    return out


def _ledger_from_arrays(d: dict) -> dict | None:
    if not d:
        return None
    out: dict[str, Any] = {}
    if "fisher" in d:
        out["fisher"] = _csr_list_from_arrays(d["fisher"])
    if "reason_per_cell" in d:
        rpc = np.asarray(d["reason_per_cell"], dtype=np.int8)
        out["reason_per_cell"] = [rpc[c].copy() for c in range(rpc.shape[0])]
    if "off_tangent" in d:
        out["off_tangent"] = _tangent_from_arrays(d["off_tangent"])
    if "thresholds_json" in d:
        try:
            out["thresholds"] = json.loads(np.asarray(d["thresholds_json"]).item())
        except Exception:
            out["thresholds"] = {}
    if "ridge_lambda" in d:
        out["ridge_lambda"] = float(np.asarray(d["ridge_lambda"]))
    return out


# ----------------------------------------------------------------- flatten


def _flatten(prefix: str, obj: dict | np.ndarray) -> dict[str, np.ndarray]:
    """Flatten a nested dict of arrays into a single-level {name: ndarray}
    using ``/`` as the separator. Scalars become 0-d arrays."""
    out: dict[str, np.ndarray] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}/{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.update(_flatten(key, v))
            else:
                out[key] = v if isinstance(v, np.ndarray) else np.asarray(v)
    else:
        out[prefix] = obj if isinstance(obj, np.ndarray) else np.asarray(obj)
    return out


def _unflatten(flat: dict[str, np.ndarray]) -> dict:
    """Inverse of :func:`_flatten`."""
    root: dict = {}
    for key, val in flat.items():
        parts = key.split("/")
        d = root
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return root


# ----------------------------------------------------------------- public API


def save(adata: AnnData, out_dir: str | Path) -> dict[str, Path]:
    """Write ``adata.h5ad`` and ``phase_c.npz`` sidecar into ``out_dir``.

    Phase C uns-keys that h5ad can't natively serialise (list of sparse
    CSR, Fisher ledger with sparse ``fisher`` + projector list, complex
    eigenvalues/vectors) are moved to the sidecar. The function restores
    the moved keys on ``adata`` before returning, so the caller still
    sees the in-memory state intact.

    Returns {"h5ad": …, "sidecar": …} with the written paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5 = out_dir / "adata.h5ad"
    npz = out_dir / "phase_c.npz"

    scj = adata.uns.get("scjdo2")
    is_dict = isinstance(scj, dict)
    stashed_uns: dict[str, Any] = {}
    stashed_obsm: dict[str, Any] = {}

    sidecar_payload: dict[str, Any] = {}
    if is_dict:
        if "J" in scj and isinstance(scj["J"], (list, tuple)) and len(scj["J"]) and issparse(scj["J"][0]):
            sidecar_payload["J"] = _csr_list_to_arrays(list(scj["J"]))
        if "J_latent" in scj and scj["J_latent"] is not None:
            sidecar_payload["J_latent"] = np.asarray(scj["J_latent"])
        if "grad_eta" in scj and scj["grad_eta"] is not None:
            sidecar_payload["grad_eta"] = np.asarray(scj["grad_eta"])
        if "fisher_ledger" in scj and scj["fisher_ledger"] is not None:
            sidecar_payload["fisher_ledger"] = _ledger_to_arrays(scj["fisher_ledger"])
        for cx_key in ("op_eigvecs", "op_eigvec_latent_lead", "op_eigvec_lifted"):
            if cx_key in scj and scj[cx_key] is not None:
                arr = np.asarray(scj[cx_key])
                sidecar_payload[cx_key] = arr
        if "op_valid" in scj and scj["op_valid"] is not None:
            sidecar_payload["op_valid"] = np.asarray(scj["op_valid"], dtype=bool)
        if "op_tangent_rank" in scj and scj["op_tangent_rank"] is not None:
            sidecar_payload["op_tangent_rank"] = np.asarray(scj["op_tangent_rank"], dtype=np.int32)
        if "op_eig_reason" in scj and scj["op_eig_reason"] is not None:
            # object array (strings) → store as int codes + vocab.
            arr = np.asarray(scj["op_eig_reason"])
            vocab, codes = np.unique(arr, return_inverse=True)
            sidecar_payload["op_eig_reason_codes"] = codes.reshape(arr.shape).astype(np.int16)
            sidecar_payload["op_eig_reason_vocab"] = np.asarray(vocab.astype(str), dtype=object)
        # Archetype dict + K-curve: dict-of-dicts with pandas DataFrames and
        # complex arrays. Stored as pickled object arrays via npz.
        if "archetypes" in scj and scj["archetypes"]:
            sidecar_payload["archetypes_pkl"] = np.array([scj["archetypes"]], dtype=object)
        if "archetype_K_curve" in scj and scj["archetype_K_curve"]:
            sidecar_payload["archetype_K_curve"] = np.asarray(
                scj["archetype_K_curve"], dtype=np.float64,
            )
        if "archetype_K_selected" in scj and scj["archetype_K_selected"] is not None:
            sidecar_payload["archetype_K_selected"] = np.int32(scj["archetype_K_selected"])
        if "archetype_patterns_latent" in scj and scj["archetype_patterns_latent"] is not None:
            sidecar_payload["archetype_patterns_latent"] = np.asarray(scj["archetype_patterns_latent"])
        # Anchor-path extras.
        if "anchor_ids" in scj and scj["anchor_ids"] is not None:
            sidecar_payload["anchor_ids"] = np.asarray(scj["anchor_ids"], dtype=np.int64)
        if "anchor_weights" in scj and scj["anchor_weights"] is not None:
            aw = scj["anchor_weights"]
            if issparse(aw):
                sidecar_payload["anchor_weights"] = {
                    "data": np.asarray(aw.data),
                    "indices": np.asarray(aw.indices, dtype=np.int32),
                    "indptr": np.asarray(aw.indptr, dtype=np.int32),
                    "shape": np.asarray(aw.shape, dtype=np.int64),
                }
            else:
                sidecar_payload["anchor_weights_dense"] = np.asarray(aw)
        if "anchor_bandwidth" in scj and scj["anchor_bandwidth"] is not None:
            sidecar_payload["anchor_bandwidth"] = np.float64(scj["anchor_bandwidth"])
        if "anchor_top_k" in scj and scj["anchor_top_k"] is not None:
            sidecar_payload["anchor_top_k"] = np.int32(scj["anchor_top_k"])

        for k in _SIDECAR_UNS_KEYS:
            if k in scj:
                stashed_uns[k] = scj.pop(k)

    for k in _SIDECAR_OBSM_KEYS:
        if k in adata.obsm:
            arr = np.asarray(adata.obsm[k])
            if np.iscomplexobj(arr):
                sidecar_payload[k] = arr
                stashed_obsm[k] = adata.obsm[k]
                del adata.obsm[k]

    try:
        adata.write_h5ad(h5)
        if sidecar_payload:
            flat = _flatten("", sidecar_payload)
            np.savez_compressed(npz, **flat)
        elif npz.exists():
            # Stale sidecar from a previous save — remove to avoid loading mismatched data.
            npz.unlink()
    finally:
        if is_dict:
            adata.uns["scjdo2"] = scj
            for k, v in stashed_uns.items():
                adata.uns["scjdo2"][k] = v
        for k, v in stashed_obsm.items():
            adata.obsm[k] = v

    return {"h5ad": h5, "sidecar": npz if sidecar_payload else None}


def load(in_dir: str | Path) -> AnnData:
    """Read ``adata.h5ad`` + ``phase_c.npz`` from a run directory and
    reattach the Phase C sidecar keys onto ``adata.uns["scjdo2"]`` and
    ``adata.obsm`` so downstream consumers see the full state.
    """
    import anndata as ad

    p = Path(in_dir)
    h5 = p / "adata.h5ad" if p.is_dir() else p
    adata = ad.read_h5ad(h5)
    npz_path = h5.parent / "phase_c.npz"
    if not npz_path.exists():
        return adata
    with np.load(npz_path, allow_pickle=True) as z:
        flat = {k: z[k] for k in z.files}
    payload = _unflatten(flat)

    scj = adata.uns.setdefault("scjdo2", {})
    if "J" in payload:
        scj["J"] = _csr_list_from_arrays(payload["J"])
    if "J_latent" in payload:
        scj["J_latent"] = np.asarray(payload["J_latent"])
    if "grad_eta" in payload:
        scj["grad_eta"] = np.asarray(payload["grad_eta"])
    if "fisher_ledger" in payload:
        from .fisher import FisherLedger
        d = _ledger_from_arrays(payload["fisher_ledger"])
        if d is not None:
            try:
                scj["fisher_ledger"] = FisherLedger(
                    fisher=d.get("fisher", []),
                    reason_per_cell=d.get("reason_per_cell", []),
                    thresholds=d.get("thresholds", {}),
                    ridge_lambda=d.get("ridge_lambda"),
                    off_tangent=d.get("off_tangent"),
                )
            except Exception:
                scj["fisher_ledger"] = d
    for cx_key in ("op_eigvecs", "op_eigvec_latent_lead", "op_eigvec_lifted"):
        if cx_key in payload:
            scj[cx_key] = np.asarray(payload[cx_key])
    if "op_valid" in payload:
        scj["op_valid"] = np.asarray(payload["op_valid"], dtype=bool)
    if "op_tangent_rank" in payload:
        scj["op_tangent_rank"] = np.asarray(payload["op_tangent_rank"], dtype=np.int32)
    if "op_eig_reason_codes" in payload:
        vocab = np.asarray(payload.get("op_eig_reason_vocab", np.array([]))).astype(str)
        codes = np.asarray(payload["op_eig_reason_codes"], dtype=np.int32)
        if vocab.size:
            scj["op_eig_reason"] = vocab[codes]
    if "op_eigvals" in payload:
        adata.obsm["op_eigvals"] = np.asarray(payload["op_eigvals"])
    if "archetypes_pkl" in payload:
        obj = np.asarray(payload["archetypes_pkl"], dtype=object).item()
        scj["archetypes"] = obj
    if "archetype_K_curve" in payload:
        kc = np.asarray(payload["archetype_K_curve"])
        scj["archetype_K_curve"] = [(int(k), float(e)) for k, e in kc]
    if "archetype_K_selected" in payload:
        scj["archetype_K_selected"] = int(np.asarray(payload["archetype_K_selected"]))
    if "archetype_patterns_latent" in payload:
        scj["archetype_patterns_latent"] = np.asarray(payload["archetype_patterns_latent"])
    if "anchor_ids" in payload:
        scj["anchor_ids"] = np.asarray(payload["anchor_ids"], dtype=np.int64)
    if "anchor_weights" in payload:
        d = payload["anchor_weights"]
        scj["anchor_weights"] = csr_matrix(
            (np.asarray(d["data"]), np.asarray(d["indices"]), np.asarray(d["indptr"])),
            shape=tuple(np.asarray(d["shape"]).tolist()),
        )
    elif "anchor_weights_dense" in payload:
        scj["anchor_weights"] = np.asarray(payload["anchor_weights_dense"])
    if "anchor_bandwidth" in payload:
        scj["anchor_bandwidth"] = float(np.asarray(payload["anchor_bandwidth"]))
    if "anchor_top_k" in payload:
        scj["anchor_top_k"] = int(np.asarray(payload["anchor_top_k"]))
    return adata
