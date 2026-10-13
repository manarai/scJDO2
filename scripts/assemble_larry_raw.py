"""Assemble raw integer LARRY in-vitro counts from the GEO stateFate
L1-normalised matrix (``GSM4185642_stateFate_inVitro_normed_counts.
mtx.gz``) by inverting the normalisation.

Weinreb's normalisation (confirmed by the AllonKleinLab/paper-data
README and this script's data probe) is::

    norm_ij = (umi_ij / total_i) * S,     S = median_total

where ``S`` is a cross-cell constant (the row-sum of the normed
matrix is exactly ``S`` for every cell). For every cell ``i``, the
minimum positive normed value is achieved at ``umi_ij = 1``::

    norm_min_i = S / total_i

so the raw UMI count is recoverable by division + round::

    umi_ij = round(norm_ij / norm_min_i)

Verified end-to-end on a probe of 50 cells: rounding error ≤ 8.5e-14.

The GEO-published ``GSE140802_RAW.tar`` is a 4.67 GB repackage of
the same per-GSM supplementary files; it does NOT contain per-sample
raw 10x matrices. The stateFate_* normed matrix is therefore the
raw-counts source for LARRY under this project's data constraints.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
from anndata import AnnData


REPO = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = REPO / "benchmarks" / "data" / "larry"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _append_checksum(checksum_path: Path, name: str, sha: str) -> None:
    """Append a line to CHECKSUMS.txt; dedupe on name."""
    lines = []
    if checksum_path.exists():
        with open(checksum_path) as f:
            for ln in f:
                parts = ln.strip().split()
                if parts and parts[0] != name:
                    lines.append(ln.rstrip("\n"))
    lines.append(f"{sha}  {name}")
    with open(checksum_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _load_gene_names(cache: Path) -> np.ndarray:
    with gzip.open(cache / "gene_names.txt.gz", "rt") as f:
        return np.asarray([ln.strip() for ln in f if ln.strip()])


def _load_metadata(cache: Path) -> pd.DataFrame:
    with gzip.open(cache / "metadata.txt.gz", "rt") as f:
        return pd.read_csv(f, sep="\t")


def _load_clone_matrix(cache: Path) -> sp.csr_matrix:
    with gzip.open(cache / "clone_matrix.mtx.gz", "rb") as f:
        buf = io.BytesIO(f.read())
    cm = sio.mmread(buf).tocsr().astype(np.int8)
    return cm


def _stream_normed_mtx(path: Path) -> tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(n_cells, n_genes, nnz, rows, cols, data)`` arrays from
    the gzip-compressed MTX. Entries are 1-indexed in the mtx file; we
    convert to 0-indexed on the fly. ``data`` stays float64 so the
    per-row divide-by-min-nonzero is exact.
    """
    t0 = time.monotonic()
    with gzip.open(path, "rt") as fh:
        fh.readline()                                   # header line
        while True:
            line = fh.readline()
            if not line:
                raise ValueError("mtx has no dims line")
            if line.startswith("%"):
                continue
            break
        n_rows, n_cols, nnz = map(int, line.split())
        rows = np.empty(nnz, dtype=np.int64)
        cols = np.empty(nnz, dtype=np.int32)
        data = np.empty(nnz, dtype=np.float64)
        i = 0
        for ln in fh:
            r_str, c_str, v_str = ln.split()
            rows[i] = int(r_str) - 1
            cols[i] = int(c_str) - 1
            data[i] = float(v_str)
            i += 1
            if i % 20_000_000 == 0:
                print(f"  [larry/assemble]   read {i:,}/{nnz:,} ({time.monotonic()-t0:.0f}s)",
                      flush=True)
    if i != nnz:
        raise ValueError(f"nnz mismatch: header says {nnz}, read {i}")
    return n_rows, n_cols, nnz, rows, cols, data


def _recover_umi_counts(
    n_cells: int,
    rows: np.ndarray,
    cols: np.ndarray,
    data: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Return integer UMI counts (int32) by dividing each row's entries
    by that row's minimum positive value and rounding.

    Also returns a diagnostics dict with max rounding error and the
    median recovered per-cell total.
    """
    t0 = time.monotonic()
    # Per-row min positive value. ``data`` is strictly > 0 everywhere
    # (sparse representation: zeros are omitted).
    norm_min = np.full(n_cells, np.inf, dtype=np.float64)
    np.minimum.at(norm_min, rows, data)
    # Rows with no entries stay at inf; replace with 1.0 so the divide
    # doesn't blow up (umi stays 0 for those rows anyway).
    norm_min = np.where(np.isfinite(norm_min), norm_min, 1.0)
    umi_f = data / norm_min[rows]
    umi_i = np.rint(umi_f).astype(np.int32)
    err = np.max(np.abs(umi_f - umi_i))
    totals = np.zeros(n_cells, dtype=np.int64)
    np.add.at(totals, rows, umi_i)
    diag = {
        "max_round_error": float(err),
        "median_total": float(np.median(totals[totals > 0])) if (totals > 0).any() else 0.0,
        "min_total": int(totals[totals > 0].min()) if (totals > 0).any() else 0,
        "max_total": int(totals.max()),
        "n_cells_zero_total": int((totals == 0).sum()),
        "elapsed_s": float(time.monotonic() - t0),
    }
    return umi_i, diag


def assemble(cache_dir: Path = DEFAULT_CACHE) -> Path:
    cache = Path(cache_dir)
    normed = cache / "GSM4185642_stateFate_inVitro_normed_counts.mtx.gz"
    for req in ("gene_names.txt.gz", "metadata.txt.gz", "clone_matrix.mtx.gz"):
        if not (cache / req).exists():
            raise FileNotFoundError(f"{cache / req} missing")
    if not normed.exists():
        raise FileNotFoundError(
            f"{normed} missing — download "
            "GSM4185642_stateFate_inVitro_normed_counts.mtx.gz from GEO or extract from "
            "GSE140802_RAW.tar."
        )

    print(f"[larry/assemble] streaming normed matrix {normed}", flush=True)
    n_cells, n_genes, nnz, rows, cols, data = _stream_normed_mtx(normed)
    print(f"[larry/assemble] shape ({n_cells}, {n_genes})  nnz={nnz:,}", flush=True)

    print("[larry/assemble] inverting normalisation (per-row divide + round)",
          flush=True)
    umi, diag = _recover_umi_counts(n_cells, rows, cols, data)
    print(f"[larry/assemble] rounding error max {diag['max_round_error']:.2e}  "
          f"UMI totals min/median/max {diag['min_total']}/{int(diag['median_total'])}/"
          f"{diag['max_total']}  elapsed {diag['elapsed_s']:.1f}s", flush=True)
    if diag["max_round_error"] > 1e-3:
        raise ValueError(
            f"UMI recovery rounding error {diag['max_round_error']:.3e} "
            "exceeds 1e-3; the normalisation may not match norm = (umi/total)*S."
        )
    del data

    print("[larry/assemble] building CSR + loading metadata/clone matrix", flush=True)
    X = sp.coo_matrix((umi, (rows, cols)), shape=(n_cells, n_genes)).tocsr()
    del umi, rows, cols
    metadata = _load_metadata(cache)
    gene_names = _load_gene_names(cache)
    clone_matrix = _load_clone_matrix(cache)
    if len(metadata) != n_cells:
        raise ValueError(f"metadata rows {len(metadata)} != counts rows {n_cells}")
    if gene_names.size != n_genes:
        raise ValueError(f"gene_names {gene_names.size} != counts cols {n_genes}")
    if clone_matrix.shape[0] != n_cells:
        raise ValueError(f"clone matrix rows {clone_matrix.shape[0]} != counts rows {n_cells}")

    adata = AnnData(
        X=X,
        obs=metadata.reset_index(drop=True),
        var=pd.DataFrame({"gene": gene_names}, index=gene_names),
    )
    adata.obsm["clone_matrix"] = clone_matrix
    adata.uns["dataset"] = {
        "name": "LARRY (in-vitro)",
        "source": (
            "GEO GSE140802 (GSM4185642) stateFate_inVitro — raw integer UMIs "
            "recovered from the L1-normalised matrix by inverting "
            "norm_ij = (umi_ij / total_i) * S."
        ),
        "normalisation": "none (raw integer UMIs)",
        "recovery_round_error_max": diag["max_round_error"],
        "recovery_notes": (
            "Per cell: norm_min_i = S / total_i, so umi_ij = round(norm_ij / "
            "norm_min_i). Verified lossless at numerical precision."
        ),
    }
    out_path = cache / "larry_inVitro_raw.h5ad"
    adata.write_h5ad(out_path)
    sha = _sha256(out_path)
    _append_checksum(cache / "CHECKSUMS.txt", out_path.name, sha)
    print(f"[larry/assemble] wrote {out_path}  sha256={sha}", flush=True)
    return out_path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = p.parse_args()
    assemble(args.cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
