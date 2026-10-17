"""Recurrent operator motifs across contexts (atlas arm).

A *motif* is a local operator pattern that comes back in more than one
context. Two levels:

* **Spectral motif** — a cluster of cells in orientation-free operator
  coordinates (sorted real spectrum, Henrici index, well stiffness).
  Matched across tissues and embeddings because none of these
  coordinates depend on the latent axes.
* **Gene motif** — the gene directions carrying the motif, read from
  the lifted leading eigenvector (``uns["scjdo2"]["op_eigvec_lifted"]``).
  Two contexts share a gene motif when their driver vectors agree
  with the motif's consensus more than the null predicts.

A *context* is ``cell type × tissue`` (any two obs columns), with a
third column (donor) kept as the replicate unit for the null.

Pipeline:

1. :func:`motif_coordinates` — per-cell descriptor matrix.
2. :func:`fit_motifs` — KMeans in that space, ``K`` chosen by
   held-out-replicate stability (adjusted Rand index between
   clusterings fit on disjoint halves of the replicates), writes
   ``obs["op_motif"]``.
3. :func:`motif_occurrence` — per-(motif, context) occupancy, the
   shuffled-context null, enrichment, and the recurrence score (number
   of contexts where the motif is enriched above the null).
4. :func:`motif_driver_similarity` — per-(motif, context) cosine of the
   mean lifted driver vector to the motif consensus (gene level).

Everything is written to ``uns["scjdo2"]["motifs"]`` and read by
:func:`scjdo2.plot.figure_motif_matrix`.

Caveat the plot layer repeats in its footer: operators pooled across
datasets carry batch unless aligned first (scOpAlign); a motif that
tracks sequencing centre fills the matrix exactly as a real one does.
Run within one dataset first, pool only after the alignment kill test.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData

from . import opspace as _opspace


# ----------------------------------------------------------------------
# 1 — coordinates
# ----------------------------------------------------------------------


def motif_coordinates(
    adata: AnnData,
    n_eig: Optional[int] = None,
    standardise: bool = True,
) -> np.ndarray:
    """Per-cell orientation-free descriptor matrix ``(n, n_eig + 2)``:
    the ``n_eig`` largest real eigenvalues (sorted descending), the
    Henrici index and the well stiffness. Standardised column-wise so
    no single descriptor dominates the clustering distance."""
    J = _opspace._dense_latent_J(adata)
    n, d, _ = J.shape
    k = int(n_eig) if n_eig is not None else d
    k = max(1, min(k, d))
    coords = np.full((n, k + 2), np.nan, dtype=np.float64)
    for c in range(n):
        Jc = J[c].astype(np.float64)
        if not np.all(np.isfinite(Jc)):
            continue
        try:
            ev = np.linalg.eigvals(Jc)
        except np.linalg.LinAlgError:
            continue
        re = np.sort(ev.real)[::-1]
        coords[c, :k] = re[:k]
        coords[c, k] = _opspace._henrici(Jc, ev)
        negs = re[re < 0]
        coords[c, k + 1] = float(-negs.mean()) if negs.size else 0.0
    if standardise:
        # Eigenvalues and stiffness share units: scale the whole block
        # by ONE scalar (the pooled eigenvalue spread) so a column that
        # carries only noise is not inflated to unit variance — per-
        # column z-scoring would let five noise columns outweigh the
        # one column that carries a saddle. Henrici has its own units
        # and is scaled on its own; when it has negligible spread
        # (symmetric J → float noise at 1e-8) it is zeroed, not inflated.
        eig_block = coords[:, : k]
        scale = float(np.nanstd(eig_block))
        scale = scale if scale > 1e-12 else 1.0
        mu_eig = float(np.nanmean(eig_block))
        coords[:, : k] = (eig_block - mu_eig) / scale
        coords[:, k + 1] = (coords[:, k + 1] - np.nanmean(coords[:, k + 1])) / scale
        hen = coords[:, k]
        hen_sd = float(np.nanstd(hen))
        if hen_sd > 1e-3 * scale:
            coords[:, k] = (hen - np.nanmean(hen)) / hen_sd
        else:
            coords[:, k] = 0.0
    adata.obsm["X_motif"] = coords.astype(np.float32)
    return coords


# ----------------------------------------------------------------------
# 2 — clustering with stability-selected K
# ----------------------------------------------------------------------


def _kmeans(X: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=int(k), n_init=5, random_state=int(seed)).fit(X)
    return km.labels_, km.cluster_centers_


def _assign(X: np.ndarray, centres: np.ndarray) -> np.ndarray:
    d2 = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
    return np.argmin(d2, axis=1)


def motif_stability(
    X: np.ndarray,
    replicate: np.ndarray,
    k_candidates: Sequence[int] = (2, 3, 4, 5, 6, 8),
    n_splits: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Held-out-replicate stability per ``K``: split replicates into two
    disjoint halves, fit KMeans on each half, assign ALL cells with both
    sets of centres, and score the two labelings by adjusted Rand index.
    Also reports the mean silhouette of a full-data fit at each ``K``
    (on a subsample of ≤ 2000 cells). Returns a DataFrame
    ``K, ari_mean, ari_std, silhouette``."""
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    rng = np.random.default_rng(seed)
    reps = np.asarray(replicate)
    levels = np.array(sorted(set(map(str, reps))))
    reps = reps.astype(str)
    ok = np.all(np.isfinite(X), axis=1)
    rows = []
    for k in k_candidates:
        aris = []
        for s in range(int(n_splits)):
            if levels.size >= 2:
                perm = rng.permutation(levels)
                half = set(perm[: max(1, levels.size // 2)])
                a = ok & np.isin(reps, list(half))
                b = ok & ~np.isin(reps, list(half))
            else:
                # single replicate: random cell split
                mask = rng.random(X.shape[0]) < 0.5
                a, b = ok & mask, ok & ~mask
            if a.sum() < k or b.sum() < k:
                continue
            _, ca = _kmeans(X[a], k, seed + 10 * s)
            _, cb = _kmeans(X[b], k, seed + 10 * s + 1)
            la = _assign(X[ok], ca)
            lb = _assign(X[ok], cb)
            aris.append(adjusted_rand_score(la, lb))
        lab_full, _ = _kmeans(X[ok], k, seed)
        sub = rng.choice(int(ok.sum()), size=min(2000, int(ok.sum())), replace=False)
        try:
            sil = float(silhouette_score(X[ok][sub], lab_full[sub]))
        except ValueError:
            sil = np.nan
        rows.append({"K": int(k),
                     "ari_mean": float(np.mean(aris)) if aris else np.nan,
                     "ari_std": float(np.std(aris)) if aris else np.nan,
                     "silhouette": sil})
    return pd.DataFrame(rows)


def fit_motifs(
    adata: AnnData,
    k: int | str = "auto",
    replicate_key: Optional[str] = None,
    k_candidates: Sequence[int] = (2, 3, 4, 5, 6, 8),
    n_eig: Optional[int] = None,
    stability_min: float = 0.9,
    seed: int = 0,
) -> pd.DataFrame:
    """Cluster cells into spectral motifs. ``k="auto"`` keeps the ``K``
    values that are reproducible across held-out replicates
    (``ari_mean ≥ stability_min``) and, among those, picks the one with
    the best silhouette — stability alone cannot separate nested
    solutions (K=2 and K=3 are both perfectly stable when the three
    groups nest). If none is stable, the most stable ``K`` is taken.
    Writes ``obs["op_motif"]`` (int, −1 for cells with no valid
    operator), ``uns["scjdo2"]["motifs"]["centres"]`` (in standardised
    descriptor space), ``["K_curve"]`` and ``["K_selected"]``. Returns
    the K curve (empty when ``k`` is an int)."""
    X = motif_coordinates(adata, n_eig=n_eig)
    ok = np.all(np.isfinite(X), axis=1)
    if replicate_key is not None and replicate_key in adata.obs:
        rep = adata.obs[replicate_key].astype(str).to_numpy()
    else:
        rep = np.array(["all"] * adata.n_obs)
    if k == "auto":
        curve = motif_stability(X, rep, k_candidates=k_candidates, seed=seed)
        valid = curve.dropna(subset=["ari_mean"])
        if valid.empty:
            raise RuntimeError("fit_motifs: no K had enough cells for a stability split")
        stable = valid[valid["ari_mean"] >= float(stability_min)]
        if stable.empty:
            best = valid["ari_mean"].max()
            K = int(valid.loc[valid["ari_mean"] >= best - 1e-12, "K"].min())
        else:
            sil = stable["silhouette"].fillna(-1.0)
            K = int(stable.loc[sil.idxmax(), "K"])
    else:
        K = int(k)
        curve = pd.DataFrame(columns=["K", "ari_mean", "ari_std", "silhouette"])
    labels_ok, centres = _kmeans(X[ok], K, seed)
    # Order motifs by their mean leading eigenvalue (basin-like first)
    # so motif ids are stable across runs with the same data.
    lead_mean = np.array([X[ok][labels_ok == i, 0].mean() for i in range(K)])
    order = np.argsort(lead_mean)
    remap = np.empty(K, dtype=int)
    remap[order] = np.arange(K)
    labels = np.full(adata.n_obs, -1, dtype=int)
    labels[ok] = remap[labels_ok]
    adata.obs["op_motif"] = labels
    # Raw-unit summary per motif for naming: mean lead_re, mean stiffness.
    desc = _opspace.landscape_descriptors(_opspace._dense_latent_J(adata))
    summary = pd.DataFrame({
        "motif": np.arange(K),
        "n": [int((labels == i).sum()) for i in range(K)],
        "lead_re": [float(np.nanmean(desc["lead_re"][labels == i])) for i in range(K)],
        "stiffness": [float(np.nanmean(desc["stiffness"][labels == i])) for i in range(K)],
        "lambda_min": [float(np.nanmean(desc["lambda_min"][labels == i])) for i in range(K)],
    })
    uns = adata.uns.setdefault("scjdo2", {})
    uns["motifs"] = {
        "centres": centres[order].astype(np.float32),
        "summary": summary,
        "K_selected": K,
        "K_curve": curve,
        "replicate_key": replicate_key,
    }
    return curve


# ----------------------------------------------------------------------
# 3 — occurrence, null, recurrence
# ----------------------------------------------------------------------


def motif_occurrence(
    adata: AnnData,
    context_keys: Sequence[str] = ("cluster", "tissue"),
    replicate_key: Optional[str] = None,
    n_perm: int = 200,
    alpha: float = 0.05,
    min_cells: int = 20,
    seed: int = 0,
) -> pd.DataFrame:
    """Per-(motif, context) occupancy and enrichment against a
    shuffled-context null.

    ``occupancy`` = fraction of the context's cells in the motif. The
    null shuffles the context label within each tissue (the last of
    ``context_keys``) — so cell-type mix within a tissue is preserved
    and the question is "does this type sit in this motif more than a
    random type of the same tissue would". When ``replicate_key`` is
    given, the observed occupancy is the mean over replicates and the
    null is built by shuffling at the replicate level, so a single
    donor cannot create an enriched cell.

    Writes ``uns["scjdo2"]["motifs"]["occurrence"]`` (tidy DataFrame
    with ``motif, <context_keys...>, context, n, occupancy, null_mean,
    null_q, enriched``) and ``["recurrence"]`` (per-motif count of
    enriched contexts). Returns the occurrence table.
    """
    if "op_motif" not in adata.obs:
        raise RuntimeError("call scjdo2.motifs.fit_motifs first")
    rng = np.random.default_rng(seed)
    motif = adata.obs["op_motif"].to_numpy()
    valid = motif >= 0
    K = int(motif[valid].max()) + 1 if valid.any() else 0
    ctx_cols = [adata.obs[k].astype(str).to_numpy() for k in context_keys]
    tissue = ctx_cols[-1]
    context = np.array([" | ".join(parts) for parts in zip(*ctx_cols)])
    rep = (adata.obs[replicate_key].astype(str).to_numpy()
           if replicate_key is not None and replicate_key in adata.obs
           else np.array(["all"] * adata.n_obs))

    def _occupancy(motif_arr, ctx_arr, rep_arr):
        # mean over replicates of per-replicate occupancy
        out: dict[tuple[str, int], list[float]] = {}
        df = pd.DataFrame({"m": motif_arr, "c": ctx_arr, "r": rep_arr})
        df = df[df["m"] >= 0]
        for (c, r), g in df.groupby(["c", "r"], observed=True):
            counts = np.bincount(g["m"].to_numpy(), minlength=K) / max(1, len(g))
            for i in range(K):
                out.setdefault((c, i), []).append(float(counts[i]))
        return {k: float(np.mean(v)) for k, v in out.items()}

    obs_occ = _occupancy(motif, context, rep)
    contexts = sorted(set(context[valid]))
    n_ctx = {c: int(((context == c) & valid).sum()) for c in contexts}

    # Null: within each tissue, permute the context label across cells
    # (or across replicate blocks when replicate_key is given).
    null_store: dict[tuple[str, int], list[float]] = {}
    for p in range(int(n_perm)):
        ctx_perm = context.copy()
        for t in sorted(set(tissue)):
            idx = np.where(tissue == t)[0]
            if replicate_key is not None and replicate_key in adata.obs:
                # permute context labels between replicate blocks
                blocks = {}
                for i in idx:
                    blocks.setdefault(rep[i], []).append(i)
                keys = list(blocks)
                labels_per_block = [context[blocks[k][0]] for k in keys]
                perm = rng.permutation(len(keys))
                for k_i, pi in zip(keys, perm):
                    ctx_perm[blocks[k_i]] = labels_per_block[pi]
            else:
                ctx_perm[idx] = context[idx][rng.permutation(idx.size)]
        occ_p = _occupancy(motif, ctx_perm, rep)
        for key, v in occ_p.items():
            null_store.setdefault(key, []).append(v)

    rows = []
    for c in contexts:
        parts = c.split(" | ")
        for i in range(K):
            o = obs_occ.get((c, i), 0.0)
            nv = np.asarray(null_store.get((c, i), [0.0]))
            q = float(np.quantile(nv, 1.0 - alpha))
            row = {"motif": i, "context": c, "n": n_ctx[c], "occupancy": o,
                   "null_mean": float(nv.mean()), "null_q": q,
                   "enriched": bool(o > q and n_ctx[c] >= min_cells)}
            for k_name, part in zip(context_keys, parts):
                row[k_name] = part
            rows.append(row)
    occ = pd.DataFrame(rows)
    rec = occ.groupby("motif")["enriched"].sum().rename("n_enriched").reset_index()
    rec["n_contexts"] = len(contexts)
    uns = adata.uns.setdefault("scjdo2", {}).setdefault("motifs", {})
    uns["occurrence"] = occ
    uns["recurrence"] = rec
    uns["context_keys"] = list(context_keys)
    uns["null"] = {"n_perm": int(n_perm), "alpha": float(alpha),
                   "replicate_key": replicate_key}
    return occ


# ----------------------------------------------------------------------
# 4 — gene level: driver similarity
# ----------------------------------------------------------------------


def motif_driver_similarity(
    adata: AnnData,
    context_keys: Sequence[str] = ("cluster", "tissue"),
    min_cells: int = 20,
) -> pd.DataFrame:
    """Per-(motif, context) cosine between the context's mean lifted
    driver vector (``|op_eigvec_lifted|``, members of the motif only)
    and the motif's consensus (mean over all its cells). Needs the
    Phase C lift in ``uns["scjdo2"]["op_eigvec_lifted"]``. Writes
    ``uns["scjdo2"]["motifs"]["driver_similarity"]``."""
    uns = adata.uns.get("scjdo2", {})
    W = uns.get("op_eigvec_lifted")
    if W is None:
        raise RuntimeError(
            "motif_driver_similarity needs uns['scjdo2']['op_eigvec_lifted'] "
            "(run scjdo2.opspace.compute after the Phase C fit)",
        )
    W = np.abs(np.asarray(W))
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    W = W / np.where(norms > 1e-12, norms, 1.0)
    motif = adata.obs["op_motif"].to_numpy()
    ctx_cols = [adata.obs[k].astype(str).to_numpy() for k in context_keys]
    context = np.array([" | ".join(parts) for parts in zip(*ctx_cols)])
    K = int(motif.max()) + 1
    rows = []
    for i in range(K):
        mi = motif == i
        if not mi.any():
            continue
        consensus = W[mi].mean(axis=0)
        consensus /= max(1e-12, np.linalg.norm(consensus))
        for c in sorted(set(context[mi])):
            mc = mi & (context == c)
            if mc.sum() < min_cells:
                continue
            v = W[mc].mean(axis=0)
            v /= max(1e-12, np.linalg.norm(v))
            rows.append({"motif": i, "context": c, "n": int(mc.sum()),
                         "driver_cosine": float(v @ consensus)})
    df = pd.DataFrame(rows)
    uns.setdefault("motifs", {})["driver_similarity"] = df
    return df
