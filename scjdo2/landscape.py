"""One-call API for the behaviour-landscape, energy-landscape and motif
analyses and their figures.

Two entry points:

* :func:`analyse` — runs every analysis step the figures need on a
  fitted AnnData (latent J in ``uns['scjdo2']['J_latent']`` or ``['J']``),
  in the right order, and writes everything back to the AnnData.
* :func:`figures` — renders every figure that the stored state
  supports into one directory, skipping the ones whose inputs are
  missing, and returns a dict ``{name: Path}``.

Both are thin: each step is a public function elsewhere
(:mod:`scjdo2.opspace`, :mod:`scjdo2.motifs`, :mod:`scjdo2.plot`) and
can be called on its own. This module only fixes the order and the
defaults.

Typical use::

    import scjdo2.landscape as L
    L.analyse(adata, cluster_key="cluster", condition_key="age",
              replicate_key="donor", drift=energy_drift)
    paths = L.figures(adata, "figures/pbmc", color_key="cluster",
                      condition_key="age")

What ``analyse`` writes (all under ``obs`` / ``uns['scjdo2']``):

* ``oracle_tol``                      — Gaussian-oracle tolerance
* ``lead_re, spectral_gap, henrici …`` — operator-space invariants
* ``op_behaviour_basin, op_basin_index`` — lead-eig basin classes
* ``op_frac_neg / zero / pos``        — eigenvalue-sign fractions
* ``op_stiffness, op_lambda_min``     — well depth descriptors
* ``energy``                          — ``E(z)`` (when a drift is given)
* ``op_motif`` + ``motifs`` dict      — recurrent motifs (when asked)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from anndata import AnnData

from . import opspace as _opspace


def analyse(
    adata: AnnData,
    cluster_key: str = "cluster",
    use_rep: Optional[str] = None,
    resting_types: Optional[Sequence[str]] = None,
    tol: Optional[float] = None,
    drift: Any = None,
    motifs: bool = False,
    tissue_key: Optional[str] = None,
    replicate_key: Optional[str] = None,
    motif_k: int | str = "auto",
    n_perm: int = 200,
    seed: int = 0,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run the full landscape analysis on a fitted AnnData.

    Parameters
    ----------
    adata
        Needs a dense latent Jacobian in ``uns['scjdo2']['J_latent']``
        (Phase C) or ``uns['scjdo2']['J']`` (Phase B), the latent
        positions in ``obsm[use_rep]``, and ``obs[cluster_key]``.
    cluster_key
        Cell-type column (oracle tol, composition bars, motif contexts).
    use_rep
        Latent key; defaults to ``uns['scjdo2']['meta']['use_rep']`` or
        ``"X_pca"``.
    resting_types
        Clusters used for the oracle tol; all clusters when ``None``.
    tol
        Override the tolerance (skips the oracle).
    drift
        A fitted energy drift (:class:`scjdo2.energy_drift.EnergyDriftAdapter`);
        when given, ``obs['energy']`` is written for the energy landscape.
    motifs
        Also fit recurrent motifs and their occurrence matrix. Contexts
        are ``cluster_key × tissue_key`` when ``tissue_key`` is given,
        else ``cluster_key`` alone (one tissue).
    replicate_key
        Donor / sample column used as the replicate unit for motif
        stability and the occurrence null.

    Returns a dict with the key numbers (``tol``, ``K`` if motifs ran)
    so callers can log them.
    """
    uns = adata.uns.setdefault("scjdo2", {})
    if use_rep is None:
        use_rep = uns.get("meta", {}).get("use_rep", "X_pca")
    if use_rep not in adata.obsm:
        raise KeyError(f"obsm[{use_rep!r}] missing")
    if cluster_key not in adata.obs:
        raise KeyError(f"obs[{cluster_key!r}] missing")
    out: dict[str, Any] = {"use_rep": use_rep}

    # 1. tolerance
    if tol is None:
        tol = _opspace.oracle_tol(adata, use_rep=use_rep, cluster_key=cluster_key,
                                  resting_types=resting_types)
    else:
        uns["oracle_tol"] = float(tol)
    out["tol"] = float(tol)
    if verbose:
        print(f"[landscape] tol = {tol:.4g}")

    # 2. invariants on the latent operator (lead_re, spectral_gap, henrici …)
    J_latent = uns.get("J_latent")
    J_gene = uns.get("J")
    if J_latent is not None and isinstance(J_gene, (list, tuple)):
        # Phase C state: run the latent pass only (R15) — gene-space J
        # stays in place for drivers.
        uns["J"] = np.asarray(J_latent)
        try:
            _opspace.spectrum(adata, n_eig=min(5, np.asarray(J_latent).shape[1]))
            _opspace.invariants(adata)
        finally:
            uns["J"] = J_gene
    else:
        Jd = _opspace._dense_latent_J(adata)
        _opspace.spectrum(adata, n_eig=min(5, Jd.shape[1]))
        _opspace.invariants(adata)

    # 3. basin classes + continuous basin index (locked lead-eig rule)
    _opspace.classify_basin_latent(adata, tol=float(tol), use_rep=use_rep)

    # 4. eigenvalue-sign fractions (ternary) and well-depth descriptors
    _opspace.behaviour_fractions(adata, tol=float(tol))
    _opspace.well_stiffness(adata)

    # 5. energy (Waddington view) when a drift with a scalar potential is given
    if drift is not None and hasattr(drift, "energy"):
        _opspace.store_energy(adata, drift, use_rep=use_rep)
        out["energy"] = True

    # 6. motifs
    if motifs:
        from . import motifs as _motifs

        ctx = (cluster_key, tissue_key) if tissue_key else (cluster_key,)
        if tissue_key is None:
            # one tissue: give motif_occurrence a constant tissue column
            adata.obs["_tissue"] = "all"
            ctx = (cluster_key, "_tissue")
        _motifs.fit_motifs(adata, k=motif_k, replicate_key=replicate_key, seed=seed)
        _motifs.motif_occurrence(adata, context_keys=ctx, replicate_key=replicate_key,
                                 n_perm=n_perm, seed=seed)
        if "op_eigvec_lifted" in uns:
            _motifs.motif_driver_similarity(adata, context_keys=ctx)
        out["K"] = int(uns["motifs"]["K_selected"])
        if verbose:
            print(f"[landscape] motifs: K = {out['K']}")
    return out


def figures(
    adata: AnnData,
    out_dir: str | Path,
    color_key: str = "cluster",
    condition_key: Optional[str] = None,
    conditions: Optional[Sequence[str]] = None,
    null: Optional[dict] = None,
    y: str = "spectral_gap",
    height: str = "stiffness",
    energy_rep: str = "X_umap",
    include: Optional[Sequence[str]] = None,
    verbose: bool = False,
) -> dict[str, Path]:
    """Render every figure the stored state supports into ``out_dir``.

    Figures (name → function), each written as ``<out_dir>/<name>.pdf``
    and ``.png``:

    * ``operator_space``   — :func:`scjdo2.plot.figure_operator_space`
      (needs ``lead_re``, ``op_behaviour``)
    * ``landscape``        — :func:`scjdo2.plot.figure_behaviour_landscape`
    * ``landscape_3d``     — :func:`scjdo2.plot.figure_behaviour_landscape_3d`
    * ``ternary``          — :func:`scjdo2.plot.figure_behaviour_ternary`
    * ``energy``           — :func:`scjdo2.plot.figure_energy_landscape`
      (needs ``obs['energy']`` and ``obsm[energy_rep]``)
    * ``motifs``           — :func:`scjdo2.plot.figure_motif_matrix`
      (needs the motif occurrence table)

    ``include`` restricts to a subset of names. Missing inputs skip the
    figure with a note (when ``verbose``) rather than raising.
    """
    import matplotlib

    matplotlib.use(matplotlib.get_backend())  # keep caller's backend
    import matplotlib.pyplot as plt

    from . import plot as P

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    want = set(include) if include else None
    paths: dict[str, Path] = {}

    def _do(name, fn):
        if want is not None and name not in want:
            return
        try:
            fig = fn(out_dir / name)
        except (RuntimeError, KeyError) as e:
            if verbose:
                print(f"[landscape] skip {name}: {e}")
            return
        plt.close(fig)
        paths[name] = out_dir / name
        if verbose:
            print(f"[landscape] wrote {name}.pdf/.png")

    beh_col = "op_behaviour" if "op_behaviour" in adata.obs else None
    _do("operator_space", lambda p: P.figure_operator_space(
        adata, out=p, by="basin_index" if "op_basin_index" in adata.obs else "lead_re"))
    _do("landscape", lambda p: P.figure_behaviour_landscape(adata, y=y, null=null, out=p))
    _do("landscape_3d", lambda p: P.figure_behaviour_landscape_3d(
        adata, y=y, height=height, null=null, out=p))
    _do("ternary", lambda p: P.figure_behaviour_ternary(
        adata, color_key=color_key, condition_key=condition_key,
        conditions=conditions, out=p))
    if "energy" in adata.obs and energy_rep in adata.obsm:
        _do("energy", lambda p: P.figure_energy_landscape(adata, use_rep=energy_rep, out=p))
    if "motifs" in adata.uns.get("scjdo2", {}) and \
            "occurrence" in adata.uns["scjdo2"]["motifs"]:
        _do("motifs", lambda p: P.figure_motif_matrix(adata, out=p))
    del beh_col
    return paths
