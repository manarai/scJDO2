"""Recurrent-motif contract tests on planted operators.

Three tissues × four cell types × three donors. Planted motifs:
* a basin shared by every (type, tissue) — must be universal;
* a saddle at the "progenitor" type in every tissue — recurs across
  tissues at structurally similar contexts, nowhere else;
* a basin with two soft axes only in lung macrophages — tissue-local.
Checks: K selected ≥ 3 and the three planted groups separate; the
universal motif is enriched in (almost) every context; the saddle is
enriched only in progenitor contexts; the lung-local motif is enriched
only in lung; recurrence scores order accordingly; the figure writes.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from scjdo2 import motifs
from scjdo2.plot import figure_motif_matrix

D = 6
N_PER = 30
TISSUES = ("lung", "blood", "gut")
TYPES = ("T", "B", "macro", "progenitor")
DONORS = ("d1", "d2", "d3")


def _op(ev, rng):
    Q, _ = np.linalg.qr(rng.normal(size=(D, D)))
    return (Q * ev) @ Q.T


def _planted(seed=0) -> AnnData:
    rng = np.random.default_rng(seed)
    J, rows, W = [], [], []
    for t in TISSUES:
        for ct in TYPES:
            for dn in DONORS:
                for _ in range(N_PER):
                    if ct == "progenitor":
                        ev = -np.abs(rng.normal(1.0, 0.08, D)); ev[0] = abs(rng.normal(0.9, 0.08))
                        gene = np.array([0, 0, 1, 1, 0, 0, 0, 0], float)
                    elif t == "lung" and ct == "macro":
                        # basin with two soft axes (lung-local)
                        ev = -np.abs(rng.normal(1.0, 0.08, D)); ev[:2] = -np.abs(rng.normal(0.12, 0.03, 2))
                        gene = np.array([0, 0, 0, 0, 1, 1, 0, 0], float)
                    else:
                        ev = -np.abs(rng.normal(1.0, 0.08, D))
                        gene = np.array([1, 1, 0, 0, 0, 0, 0, 0], float)
                    J.append(_op(ev, rng))
                    W.append(gene + rng.normal(0, 0.05, 8))
                    rows.append({"cluster": ct, "tissue": t, "donor": dn})
    J = np.stack(J).astype(np.float32)
    n = J.shape[0]
    ad = AnnData(X=np.zeros((n, 8), dtype=np.float32),
                 obs=pd.DataFrame(rows, index=[f"c{i}" for i in range(n)]))
    ad.obsm["X_pca"] = rng.normal(size=(n, D)).astype(np.float32)
    ad.uns["scjdo2"] = {"J": J, "meta": {"use_rep": "X_pca"},
                        "op_eigvec_lifted": np.stack(W).astype(np.complex64)}
    return ad


def test_fit_and_occurrence():
    ad = _planted()
    curve = motifs.fit_motifs(ad, k="auto", replicate_key="donor",
                              k_candidates=(2, 3, 4, 5), seed=0)
    K = ad.uns["scjdo2"]["motifs"]["K_selected"]
    assert K >= 3 and len(curve) == 4
    lab = ad.obs["op_motif"].to_numpy()
    cl = ad.obs["cluster"].to_numpy(); ti = ad.obs["tissue"].to_numpy()
    prog = cl == "progenitor"
    lung_m = (ti == "lung") & (cl == "macro")
    rest = ~prog & ~lung_m
    # each planted group is dominated by one motif, and they differ
    dom = lambda m: np.bincount(lab[m][lab[m] >= 0]).argmax()
    assert len({dom(prog), dom(lung_m), dom(rest)}) == 3
    assert (lab[prog] == dom(prog)).mean() > 0.9
    assert (lab[rest] == dom(rest)).mean() > 0.9

    occ = motifs.motif_occurrence(ad, context_keys=("cluster", "tissue"),
                                  replicate_key="donor", n_perm=100, seed=0)
    rec = ad.uns["scjdo2"]["motifs"]["recurrence"].set_index("motif")
    n_ctx = len(TISSUES) * len(TYPES)
    # universal basin: enriched in most non-progenitor, non-lung-macro contexts
    e = occ[occ["motif"] == dom(rest)]
    assert e["enriched"].sum() >= n_ctx - 3 - 2
    # saddle: enriched exactly at progenitor contexts in every tissue
    s = occ[occ["motif"] == dom(prog)]
    assert set(s.loc[s["enriched"], "cluster"]) == {"progenitor"}
    assert set(s.loc[s["enriched"], "tissue"]) == set(TISSUES)
    # lung-local: enriched only in lung
    l = occ[occ["motif"] == dom(lung_m)]
    assert set(l.loc[l["enriched"], "tissue"]) <= {"lung"}
    assert rec.loc[dom(rest), "n_enriched"] > rec.loc[dom(prog), "n_enriched"] \
        >= rec.loc[dom(lung_m), "n_enriched"]

    drv = motifs.motif_driver_similarity(ad)
    assert (drv["driver_cosine"] > 0.9).mean() > 0.9

    with TemporaryDirectory() as tmp:
        out = Path(tmp) / "motifs"
        fig = figure_motif_matrix(ad, out=out)
        assert out.with_suffix(".pdf").exists() and out.with_suffix(".png").exists()
        assert len(fig.axes) >= 3


def test_occurrence_requires_fit():
    ad = _planted()
    with pytest.raises(RuntimeError):
        motifs.motif_occurrence(ad)
