"""scjdo2.landscape one-call API — analyse + figures on planted operators."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scjdo2 import landscape
from tests.test_behaviour_landscape import _planted_adata


class _E:
    def energy(self, z):
        return -np.exp(-np.sum(z[:, :2] ** 2, axis=1))


def test_analyse_and_figures():
    ad = _planted_adata()
    ad.obsm["X_umap"] = ad.obsm["X_pca"][:, :2].copy()
    info = landscape.analyse(ad, cluster_key="cluster", drift=_E(), motifs=True,
                             replicate_key="age", n_perm=20, verbose=False)
    assert info["tol"] > 0 and info["energy"] and info["K"] >= 2
    for col in ("lead_re", "spectral_gap", "henrici", "op_behaviour_basin", "op_basin_index",
                "op_frac_neg", "op_stiffness", "energy", "op_motif"):
        assert col in ad.obs, col
    assert "occurrence" in ad.uns["scjdo2"]["motifs"]
    with TemporaryDirectory() as tmp:
        paths = landscape.figures(ad, tmp, color_key="cluster", condition_key="age",
                                  energy_rep="X_umap")
        for name in ("landscape", "landscape_3d", "ternary", "energy", "motifs"):
            assert name in paths, name
            assert Path(str(paths[name]) + ".png").exists()
