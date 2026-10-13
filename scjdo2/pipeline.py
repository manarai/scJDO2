"""Phase B + C pipeline — fit the drift + field in latent space and, when
``with_rate_model=True``, also the NB rate model, the per-cell gene-space
Jacobian, the Fisher ledger, the Fisher-weighted tangent projector and
``grad_eta``. The resulting AnnData is what the opspace / plot layers
consume (via :func:`scjdo2.io.save` / :func:`scjdo2.io.load`).

Jacobian estimator (``jacobian_method``):

* ``"autodiff"`` (default) — ``J_c = ∂f/∂z`` at ``(z_c, t_c)`` by autograd
  of the drift network. Exact up to float precision; variance comes only
  from the drift fit, not from a neighbourhood regression. Phase B's
  default per the user's gate feedback.
* ``"neighbourhood"`` — kNN-ball ridge regression (``scjdo2.field``). Phase
  C uses this on the gene-space rate model where the gene-space J has no
  closed-form autodiff path; here it stays for comparison and future use.

Written keys:
    adata.obsm["X_drift_v"]       — drift velocities v_c (n, d)
    adata.uns["scjdo2"]["J"]      — dense Jacobian field (n, d, d)
    adata.uns["scjdo2"]["meta"]   — {clock_calibrated, use_rep, time_key,
                                     jacobian_method, …}
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from anndata import AnnData

from .drift import DriftAdapter, DriftConfig
from .field import FieldConfig, JacobianField


def fit_latent_field(
    adata: AnnData,
    use_rep: str = "X_pca",
    time_key: str = "pseudotime",
    k: int = 40,
    drift_config: Optional[DriftConfig] = None,
    field_config: Optional[FieldConfig] = None,
    clock_scale: Optional[np.ndarray] = None,
    jacobian_method: Literal["autodiff", "neighbourhood"] = "autodiff",
    verbose: bool = False,
) -> AnnData:
    """Fit Steps 2 and 5 on an AnnData and store results.

    Steps 2 (drift) and 5 (local Jacobian) share the ``DriftAdapter`` fit.
    By default the Jacobian is read from the drift network itself by
    autograd (``jacobian_method="autodiff"``); pass ``"neighbourhood"`` to
    use the kNN-ball ridge regression kept in :mod:`scjdo2.field` for
    Phase C's gene-space step.
    """
    if use_rep not in adata.obsm:
        raise KeyError(f"adata.obsm[{use_rep!r}] not found")
    if time_key not in adata.obs:
        raise KeyError(f"adata.obs[{time_key!r}] not found")

    z = np.asarray(adata.obsm[use_rep], dtype=np.float32)
    t = adata.obs[time_key].to_numpy(dtype=np.float32)
    d = z.shape[1]

    drift_config = drift_config or DriftConfig()
    drift = DriftAdapter(dim=d, config=drift_config).fit(
        z, t=t, clock_scale=clock_scale, verbose=verbose,
    )
    v = drift.velocity(z, t)

    field_reason_summary: dict[str, float | bool]
    field_meta: dict[str, object]
    if jacobian_method == "autodiff":
        J = drift.jacobian_autodiff(z, t)
        field_reason_summary = {"ok": 1.0, "autodiff": True}
        field_meta = {"method": "autodiff"}
    elif jacobian_method == "neighbourhood":
        field_config = field_config or FieldConfig(k=k)
        jfield = JacobianField(config=field_config).fit(z, v)
        J = jfield.J_dense()
        field_reason_summary = jfield.reason_summary()
        field_meta = {
            "method": "neighbourhood",
            "k": field_config.k,
            "ridge": field_config.ridge,
            "variation_threshold": field_config.variation_threshold,
        }
    else:
        raise ValueError(
            f"jacobian_method must be 'autodiff' or 'neighbourhood', got {jacobian_method!r}"
        )

    adata.obsm["X_drift_v"] = v
    adata.uns.setdefault("scjdo2", {})
    adata.uns["scjdo2"]["J"] = J
    adata.uns["scjdo2"]["meta"] = {
        "clock_calibrated": bool(drift.config.clock_calibrated),
        "use_rep": use_rep,
        "time_key": time_key,
        "jacobian_method": jacobian_method,
        "drift": {
            "hidden": drift_config.hidden,
            "depth": drift_config.depth,
            "n_epochs": drift_config.n_epochs,
            "seed": drift_config.seed,
            "use_spectral_norm": drift_config.use_spectral_norm,
            "vel_scale": drift_config.vel_scale,
            "sigma": drift_config.sigma,
            "inject_velocity_prior": drift_config.inject_velocity_prior,
        },
        "field": field_meta,
    }
    adata.uns["scjdo2"]["field_reason_summary"] = field_reason_summary
    return adata


def fit_rate_and_gene_field(
    adata: AnnData,
    use_rep: str = "X_pca",
    n_epochs_rate: int = 400,
    rate_hidden: tuple[int, ...] = (128, 128),
    k: int = 30,
    cv_sample_cells: int = 30,
    cv_metric: Literal["nb_nll", "weighted_mse"] = "nb_nll",
    n_perm_tangent: int = 30,
    tangent_quantile: float = 0.5,
    seed: int = 0,
    verbose: bool = False,
    anchor_fit: bool = False,
    n_anchors: int = 300,
    anchor_top_k: int = 10,
    anchor_bandwidth: Optional[float] = None,
    anchor_batch: int = 25,
    anchor_max_live: int = 150,
) -> AnnData:
    """Phase C extension — NB rate model + gene-space J + Fisher ledger
    + Fisher-weighted tangent projector + ``grad_eta``.

    Prerequisites written by :func:`fit_latent_field`:
      - ``adata.obsm[use_rep]`` latent positions,
      - ``adata.obsm["X_drift_v"]`` drift velocities,
      - ``adata.uns["scjdo2"]["J"]`` latent Jacobian field.

    Writes:
      - ``adata.uns["scjdo2"]["J_latent"]`` — the dense latent J moved
        aside (so ``opspace.compute`` can run its R15 behavioural pass
        on it).
      - ``adata.uns["scjdo2"]["J"]`` — overwritten with the per-cell
        gene-space CSR Jacobian list.
      - ``adata.uns["scjdo2"]["fisher_ledger"]`` — Fisher ledger with
        ``off_tangent`` carrying the per-cell projector + rank.
      - ``adata.uns["scjdo2"]["grad_eta"]`` — ``(n, G, d)``.
    """
    from .field import (
        GeneFieldConfig,
        fit_gene_space_jacobians,
        fit_gene_space_jacobians_anchored,
        fit_jacobian_tangent_projectors,
    )
    from .fisher import compute as fisher_compute
    from .rates import GeneRateModel, RatesConfig

    uns = adata.uns.setdefault("scjdo2", {})
    if "J" not in uns:
        raise RuntimeError(
            "fit_rate_and_gene_field needs the latent Jacobian in "
            "uns['scjdo2']['J']; run fit_latent_field first."
        )
    if "X_drift_v" not in adata.obsm:
        raise RuntimeError(
            "fit_rate_and_gene_field needs obsm['X_drift_v']; run "
            "fit_latent_field first."
        )

    z = np.asarray(adata.obsm[use_rep])
    v = np.asarray(adata.obsm["X_drift_v"])
    counts = np.asarray(adata.X, dtype=np.float32)
    # anndata.X may be sparse; densify once for the rate model.
    if hasattr(counts, "toarray"):
        counts = counts.toarray()

    import time as _time
    _t0 = _time.monotonic()
    rm = GeneRateModel(RatesConfig(
        hidden=tuple(rate_hidden), n_epochs=int(n_epochs_rate), lr=1e-3,
        batch_size=256, held_out_frac=0.10, patience=30, seed=int(seed),
    ))
    rm.fit(counts, z, verbose=verbose)
    if verbose:
        print(f"  [pipeline] rate-model fit {_time.monotonic() - _t0:.1f}s")
    else:
        print(f"  [pipeline] rate-model fit {_time.monotonic() - _t0:.1f}s  "
              f"NLL={rm.best_val_nll:.3f}")

    gf_cfg = GeneFieldConfig(
        k=int(k), cv_sample_cells=int(cv_sample_cells),
        cv_metric=cv_metric, seed=int(seed),
        n_anchors=int(n_anchors),
        anchor_top_k=int(anchor_top_k),
        anchor_bandwidth=anchor_bandwidth,
        anchor_batch=int(anchor_batch),
        anchor_max_live=int(anchor_max_live),
    )
    _t0 = _time.monotonic()
    if anchor_fit:
        gs = fit_gene_space_jacobians_anchored(
            z, v, rm, counts=counts, config=gf_cfg, verbose=True,
        )
    else:
        gs = fit_gene_space_jacobians(
            z, v, rm, counts=counts, config=gf_cfg, verbose=True,
        )
    print(f"  [pipeline] gene-space J fit {_time.monotonic() - _t0:.1f}s")

    _t0 = _time.monotonic()
    tangent = fit_jacobian_tangent_projectors(
        z, rm, n_perm=int(n_perm_tangent),
        quantile=float(tangent_quantile), seed=int(seed),
    )
    print(f"  [pipeline] tangent projector fit {_time.monotonic() - _t0:.1f}s")

    _t0 = _time.monotonic()
    ledger = fisher_compute(gs, tangent=tangent)
    print(f"  [pipeline] Fisher ledger assembled {_time.monotonic() - _t0:.1f}s")

    uns["J_latent"] = np.asarray(uns["J"])
    uns["J"] = gs["J"]
    uns["fisher_ledger"] = ledger
    _t0 = _time.monotonic()
    uns["grad_eta"] = rm.grad_eta(z)
    print(f"  [pipeline] grad_eta computed {_time.monotonic() - _t0:.1f}s")
    # Record the ridge λ selection + rate-model held-out NLL for the report.
    uns["rate_meta"] = {
        "held_out_nll": float(rm.best_val_nll),
        "n_epochs": int(n_epochs_rate),
        "cv_metric": cv_metric,
        "ridge_lambda": float(gs.get("ridge_lambda", float("nan"))),
        "ridge_lambda_nll": float(gs.get("ridge_lambda_nll", float("nan"))),
        "ridge_lambda_mse": float(gs.get("ridge_lambda_mse", float("nan"))),
        "variant": gs.get("variant", "per_cell"),
    }
    # Anchor path extras (safe to omit on the per-cell path).
    if gs.get("variant") == "anchor":
        adata.obs["op_anchor"] = np.asarray(gs["op_anchor"], dtype=np.int32)
        uns["anchor_ids"] = np.asarray(gs["anchor_ids"], dtype=np.int64)
        uns["anchor_bandwidth"] = float(gs.get("anchor_bandwidth", 0.0))
        uns["anchor_top_k"] = int(gs.get("anchor_top_k", 0))
        uns["anchor_weights"] = gs["anchor_weights"]
    return adata
