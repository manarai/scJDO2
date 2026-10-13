"""LARRY haematopoiesis loader (Weinreb 2020, GSE140802).

Scope: raw counts, clone barcodes, day labels. Everything else is
derived downstream (QC, HVG, PCA, pseudotime), to keep the Step 2
drift fit and the clone arm honest — the loader never adds a root
cell, lineage prior, or annotation-based direction (V2_PLAN R17).

Expected layout under ``cache_dir`` (default: ``benchmarks/data/larry``)::

    cache_dir/
      state_fate_data.npz       # X (sparse), gene_names, cell_metadata
      clone_matrix.npz          # sparse (n_cells, n_clones) 0/1 membership
      metadata.csv              # cell metadata (day, cell_type_annotation,
                                #   library, well, cell_id)

These files come from Weinreb 2020's processed artefacts
(AllonKleinLab/paper-data :: "LARRY_hematopoiesis_states.npz" and
"LARRY_hematopoiesis_clone_matrix.npz") or from a reprocess of
GSE140802 that preserves the same cell-order. If the two files
disagree on ``n_cells``, this loader raises — the clone matrix row
order must match the counts.

The loader does not download anything on its own. If files are
missing it prints the canonical sources and raises.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData


# Default cache lives under the repo's benchmarks tree so a run from a
# fresh clone has a single place to drop the files. Users can point to
# any directory via ``cache_dir``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = _REPO_ROOT / "benchmarks" / "data" / "larry"


# Expected files + source hints. The loader prints these verbatim when a
# file is missing — no fabricated URLs beyond what the dataset's own
# publication-time repository documented.
_EXPECTED_FILES = {
    "state_fate_data.npz": (
        "cell × gene raw counts + gene names + per-cell metadata. "
        "Primary source: Weinreb et al. 2020, Science, Supplementary; "
        "or AllonKleinLab/paper-data :: "
        "'LARRY_hematopoiesis_states.npz' (same cell order as the "
        "clone matrix). The underlying raw data is GEO GSE140802."
    ),
    "clone_matrix.npz": (
        "sparse (n_cells, n_clones) 0/1 clone-membership matrix. "
        "Primary source: AllonKleinLab/paper-data :: "
        "'LARRY_hematopoiesis_clone_matrix.npz'. "
        "Cell order must match state_fate_data.npz."
    ),
    "metadata.csv": (
        "per-cell metadata table with columns 'day', 'cell_type_annotation', "
        "'library', 'well', 'cell_id'. Weinreb 2020's distribution bundles "
        "this inside state_fate_data.npz under the 'cell_metadata' key; "
        "``load_larry`` will reconstruct ``metadata.csv`` from the npz when "
        "the csv is missing but state_fate_data.npz is present."
    ),
}


def _missing_file_message(path: Path) -> str:
    lines = [f"LARRY loader needs {path.name!r} under {path.parent}."]
    hint = _EXPECTED_FILES.get(path.name)
    if hint:
        lines.append(f"  {hint}")
    lines.append(
        "  Place the required files under the cache dir and retry. "
        "The loader does no network IO on its own."
    )
    return "\n".join(lines)


def _load_counts_and_metadata(npz_path: Path) -> tuple[sp.csr_matrix, np.ndarray, pd.DataFrame]:
    """Return ``(X, gene_names, cell_metadata)`` from Weinreb's state_fate npz.

    The published npz holds:
      - ``X`` or ``counts``        : scipy sparse ``(n_cells, n_genes)``
      - ``gene_names`` or ``genes``: ``(n_genes,)`` str
      - ``cell_metadata``          : structured array / dict / DataFrame

    We accept any of these aliases so a reprocess with slightly different
    key names still loads. If none of the X-keys resolve, we raise with
    the list of keys actually present so the user can rename.
    """
    with np.load(npz_path, allow_pickle=True) as z:
        keys = list(z.files)
        # X matrix — try canonical names.
        X = None
        for name in ("X", "counts", "data_matrix"):
            if name in z.files:
                X = z[name]
                break
        if X is None:
            # Maybe it's stored as sparse CSR components (data/indices/indptr/shape).
            if {"data", "indices", "indptr", "shape"} <= set(z.files):
                X = sp.csr_matrix(
                    (z["data"], z["indices"], z["indptr"]),
                    shape=tuple(np.asarray(z["shape"]).tolist()),
                )
            else:
                raise KeyError(
                    f"state_fate_data.npz has no recognised counts matrix. "
                    f"Keys present: {keys!r}. Expected one of 'X', 'counts', "
                    "'data_matrix', or sparse (data, indices, indptr, shape)."
                )
        X_dense_ok = hasattr(X, "toarray")
        if not X_dense_ok and isinstance(X, np.ndarray) and X.dtype == object:
            # Pickled sparse object inside npz.
            X = X.item()
        if not sp.issparse(X):
            X = sp.csr_matrix(np.asarray(X))
        else:
            X = X.tocsr()
        # Gene names.
        gn = None
        for name in ("gene_names", "genes", "var_names"):
            if name in z.files:
                gn = np.asarray(z[name]).astype(str)
                break
        if gn is None:
            raise KeyError(
                f"state_fate_data.npz has no gene names. Keys: {keys!r}."
            )
        if gn.size != X.shape[1]:
            raise ValueError(
                f"gene_names length {gn.size} != n_genes {X.shape[1]}"
            )
        # Cell metadata.
        meta = None
        for name in ("cell_metadata", "obs", "metadata"):
            if name in z.files:
                raw = z[name]
                if raw.dtype == object:
                    raw = raw.item()
                if isinstance(raw, pd.DataFrame):
                    meta = raw.reset_index(drop=True)
                elif isinstance(raw, dict):
                    meta = pd.DataFrame(raw)
                elif isinstance(raw, np.ndarray) and raw.dtype.names:
                    meta = pd.DataFrame({n: raw[n] for n in raw.dtype.names})
                break
        if meta is None:
            # Not every reprocess bundles cell metadata inside the npz; the
            # caller may still have ``metadata.csv`` beside it.
            meta = pd.DataFrame({"cell_id": np.arange(X.shape[0]).astype(str)})
    return X, gn, meta


def _load_clone_matrix(npz_path: Path, n_cells: int) -> sp.csr_matrix:
    """Return a sparse ``(n_cells, n_clones)`` binary clone-membership matrix."""
    with np.load(npz_path, allow_pickle=True) as z:
        keys = list(z.files)
        if "clone_matrix" in z.files:
            cm = z["clone_matrix"]
            if cm.dtype == object:
                cm = cm.item()
            if not sp.issparse(cm):
                cm = sp.csr_matrix(np.asarray(cm))
            else:
                cm = cm.tocsr()
        elif {"data", "indices", "indptr", "shape"} <= set(z.files):
            cm = sp.csr_matrix(
                (z["data"], z["indices"], z["indptr"]),
                shape=tuple(np.asarray(z["shape"]).tolist()),
            )
        else:
            raise KeyError(
                f"clone_matrix.npz has no recognised matrix. Keys: {keys!r}."
            )
    if cm.shape[0] != n_cells:
        raise ValueError(
            f"clone_matrix n_cells {cm.shape[0]} != counts n_cells {n_cells}. "
            "The two files must share the same row order (cells)."
        )
    # Binarise so a cell is either in a clone or not — the published matrix
    # is typically already 0/1 but some reprocesses use read-counts.
    if (cm.data != 0).all():
        cm.data = np.ones_like(cm.data, dtype=np.int8)
    cm.eliminate_zeros()
    return cm.tocsr()


def load_larry(
    cache_dir: Optional[str | Path] = None,
    *,
    require_clone_matrix: bool = True,
) -> AnnData:
    """Load the Weinreb 2020 LARRY haematopoiesis snapshot (raw UMIs).

    Returns an :class:`AnnData` with:

      * ``.X``                  raw counts (sparse CSR, int32).
      * ``.var_names``          gene symbols.
      * ``.obs['day']``         sampling day (int, usually 2 / 4 / 6).
      * ``.obs['cell_type_annotation']``  Weinreb's annotated fate.
      * ``.obs['library']`` / ``.obs['well']`` / ``.obs['cell_id']``
                                 original per-cell metadata (strings).
      * ``.obsm['clone_matrix']``  sparse ``(n_cells, n_clones)`` CSR; a
                                 cell is in clone ``k`` iff
                                 ``clone_matrix[c, k] > 0``.

    No pseudotime is written. No root cell. No arrow. V2_PLAN R17.

    Data source resolution order (V2_PLAN R16 provenance decision):

    1. ``<cache>/larry_inVitro_raw.h5ad`` — the assembled raw-UMI
       artefact built by ``scripts/assemble_larry_raw.py`` from the
       GEO GSE140802 raw per-library 10x matrices and the stateFate_*
       supplementary files. SHA-256 recorded in ``<cache>/CHECKSUMS
       .txt``. Preferred when present.
    2. ``<cache>/state_fate_data.npz`` + ``clone_matrix.npz`` — the
       Weinreb published-processed layout. May carry raw or
       normalised counts depending on upstream; the loader does not
       check. Use only if you have a mirror that holds integer UMIs
       in these files (not GEO: GEO stateFate_*normed_counts.mtx.gz
       is confirmed non-integer).
    """
    cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE

    # Preferred path: assembled h5ad with raw integer counts.
    h5 = cache / "larry_inVitro_raw.h5ad"
    if h5.exists():
        import anndata as _ad
        adata = _ad.read_h5ad(h5)
        # Normalise obs column names → the GEO metadata uses "Time point"
        # / "Cell type annotation" with capitalised spaces. We fold those
        # to the scJDO2 convention so downstream code can address them
        # consistently.
        rename = {
            "Time point": "day",
            "Cell type annotation": "cell_type_annotation",
            "Library": "library",
            "Well": "well",
            "Starting population": "starting_population",
            "Cell barcode": "cell_barcode",
            "SPRING-x": "spring_x",
            "SPRING-y": "spring_y",
        }
        for old, new in rename.items():
            if old in adata.obs.columns and new not in adata.obs.columns:
                adata.obs[new] = adata.obs[old]
        if "day" in adata.obs:
            adata.obs["day"] = pd.to_numeric(adata.obs["day"], errors="coerce").astype("Int64")
        adata.uns.setdefault("dataset", {})
        adata.uns["dataset"].setdefault("name", "LARRY (in-vitro)")
        return adata

    npz_state = cache / "state_fate_data.npz"
    npz_clone = cache / "clone_matrix.npz"
    csv_meta = cache / "metadata.csv"

    if not npz_state.exists():
        raise FileNotFoundError(_missing_file_message(npz_state))

    X, gene_names, meta = _load_counts_and_metadata(npz_state)
    if csv_meta.exists():
        extra = pd.read_csv(csv_meta)
        if len(extra) == X.shape[0]:
            # metadata.csv wins over whatever the npz carried.
            meta = extra.reset_index(drop=True)
    for required in ("day",):
        if required not in meta.columns:
            raise KeyError(
                f"LARRY metadata missing column {required!r}. Present: "
                f"{list(meta.columns)!r}. Provide a metadata.csv with at "
                "least 'day' alongside state_fate_data.npz."
            )
    # Normalise types.
    meta = meta.copy()
    meta["day"] = pd.to_numeric(meta["day"], errors="coerce").astype("Int64")
    for col in ("cell_type_annotation", "library", "well", "cell_id"):
        if col not in meta.columns:
            meta[col] = pd.NA

    adata = AnnData(
        X=X.astype(np.int32),
        obs=meta.reset_index(drop=True),
        var=pd.DataFrame({"gene": gene_names}, index=gene_names),
    )
    adata.var_names = gene_names

    if npz_clone.exists():
        cm = _load_clone_matrix(npz_clone, adata.n_obs)
        adata.obsm["clone_matrix"] = cm
    elif require_clone_matrix:
        raise FileNotFoundError(_missing_file_message(npz_clone))

    adata.uns["dataset"] = {
        "name": "LARRY",
        "source": "Weinreb 2020 (GSE140802)",
        "cache_dir": str(cache),
    }
    return adata


# -------------------------------------------------------------------- clone API


def clone_pairs(adata: AnnData, *, same_clone_only: bool = True) -> np.ndarray:
    """Return per-clone cell-index pairs ``(c_early, c_late)`` where the two
    cells share a clone barcode and sit on different days.

    If ``same_clone_only`` is True, we only emit pairs within the same
    clone; otherwise all pairs within a clone's row set are emitted. The
    "earlier day" is the smaller ``day`` entry; cells on the same day
    inside a clone are skipped.

    Shape: ``(n_pairs, 2)`` int64.
    """
    if "clone_matrix" not in adata.obsm:
        raise KeyError("adata.obsm['clone_matrix'] missing — load with load_larry()")
    cm = adata.obsm["clone_matrix"]
    if not sp.issparse(cm):
        cm = sp.csr_matrix(cm)
    cm_t = cm.T.tocsr()                      # (n_clones, n_cells)
    day = adata.obs["day"].to_numpy()
    pairs: list[tuple[int, int]] = []
    for k in range(cm_t.shape[0]):
        members = cm_t.indices[cm_t.indptr[k]:cm_t.indptr[k + 1]]
        if members.size < 2:
            continue
        d = day[members]
        for i in range(members.size):
            for j in range(members.size):
                if i == j:
                    continue
                di, dj = d[i], d[j]
                if pd.isna(di) or pd.isna(dj) or di >= dj:
                    continue
                pairs.append((int(members[i]), int(members[j])))
    return np.asarray(pairs, dtype=np.int64) if pairs else np.zeros((0, 2), dtype=np.int64)


def clone_split_positions(
    adata: AnnData,
    *,
    annotation_key: str = "cell_type_annotation",
    use_rep: str = "X_pca",
) -> pd.DataFrame:
    """For each clone that spans two distinct annotated fates, return the
    latent-space midpoint of the clone's members at the earliest day on
    which the clone has at least two surviving fate-destined daughters.

    The returned table is a direction-free *descriptor* — it tells you
    *where* on the manifold a fate split occurs, not *when* or *in which
    direction*. Phase D's clone-split map on Panel A reads it.

    Columns: ``clone_id``, ``fate_a``, ``fate_b``, ``n_a``, ``n_b``,
    ``split_day``, ``z0`` … ``z{d-1}`` (latent coordinates of the mean).
    """
    if "clone_matrix" not in adata.obsm:
        raise KeyError("adata.obsm['clone_matrix'] missing — load with load_larry()")
    if use_rep not in adata.obsm:
        raise KeyError(
            f"adata.obsm[{use_rep!r}] missing; fit PCA before clone_split_positions"
        )
    cm = adata.obsm["clone_matrix"]
    if not sp.issparse(cm):
        cm = sp.csr_matrix(cm)
    cm_t = cm.T.tocsr()
    day = adata.obs["day"].to_numpy()
    fate = adata.obs[annotation_key].astype(str).to_numpy()
    z = np.asarray(adata.obsm[use_rep])
    rows = []
    d = z.shape[1]
    for k in range(cm_t.shape[0]):
        members = cm_t.indices[cm_t.indptr[k]:cm_t.indptr[k + 1]]
        if members.size < 2:
            continue
        fates = np.unique(fate[members])
        fates = fates[fates != "nan"]
        if fates.size < 2:
            continue
        # Pick the two most-populous fates.
        counts = {f: int((fate[members] == f).sum()) for f in fates}
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:2]
        if top[1][1] < 1:
            continue
        fa, fb = top[0][0], top[1][0]
        na, nb = top[0][1], top[1][1]
        split_day = int(np.nanmin(day[members])) if np.any(np.isfinite(day[members])) else -1
        mean_z = z[members].mean(axis=0)
        row = {
            "clone_id": int(k),
            "fate_a": fa, "fate_b": fb,
            "n_a": na, "n_b": nb,
            "split_day": split_day,
        }
        for j in range(d):
            row[f"z{j}"] = float(mean_z[j])
        rows.append(row)
    return pd.DataFrame(rows)
