"""Phase C io sidecar round-trip test.

Residual 1 of the Task 41 follow-ups. Confirms that `scjdo2.io.save` +
`scjdo2.io.load` reconstruct the Phase C artefacts (per-cell CSR J,
Fisher ledger, tangent projector, grad_eta, latent J, complex
eigenpairs) end-to-end so `scjdo2 figure *` can read a stored run
directory and never refits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

from scjdo2 import io, opspace


@pytest.fixture()
def phase_c_adata():
    rng = np.random.default_rng(0)
    n, d, G = 30, 2, 8
    z = rng.normal(size=(n, d)).astype(np.float32)
    J_latent = np.array([
        (-0.5 if c % 2 == 0 else 0.3) * np.eye(d) for c in range(n)
    ])
    live = np.arange(5)
    J_gene, F_list = [], []
    for c in range(n):
        data = (0.1 * rng.normal(size=(5, 5))).ravel()
        rows = np.repeat(live, 5)
        cols = np.tile(live, 5)
        J_gene.append(csr_matrix((data, (rows, cols)), shape=(G, G)))
        F_list.append(csr_matrix(
            (np.ones_like(data), (rows, cols)), shape=(G, G),
        ))
    tangent_P = [np.eye(2, G) for _ in range(n)]
    grad_eta = 0.3 * rng.normal(size=(n, G, d))

    adata = AnnData(X=np.zeros((n, G), dtype=np.float32))
    adata.obsm["X_pca"] = z
    adata.var_names = [f"g{i}" for i in range(G)]
    adata.uns["scjdo2"] = {
        "J": J_gene,
        "J_latent": J_latent,
        "fisher_ledger": {
            "fisher": F_list,
            "off_tangent": {
                "projector": tangent_P,
                "rank": np.full(n, 2, dtype=np.int32),
                "noise_floor": rng.uniform(1, 3, size=n),
            },
            "reason_per_cell": [np.zeros(G, dtype=np.int8) for _ in range(n)],
            "thresholds": {"var_floor": 1e-6},
        },
        "grad_eta": grad_eta,
    }
    opspace.compute(adata)
    return adata


def _ledger_part(ledger, key, attr=None):
    if isinstance(ledger, dict):
        return ledger[key]
    return getattr(ledger, attr or key)


def test_sidecar_roundtrip_preserves_phase_c_state(phase_c_adata, tmp_path):
    paths = io.save(phase_c_adata, tmp_path)
    assert paths["h5ad"].exists()
    assert paths["sidecar"] is not None and paths["sidecar"].exists()

    reloaded = io.load(tmp_path)
    scj = reloaded.uns["scjdo2"]
    orig = phase_c_adata.uns["scjdo2"]

    # Behavioural obs columns survive the h5ad round-trip.
    np.testing.assert_array_equal(
        phase_c_adata.obs["op_behaviour"].astype(str).to_numpy(),
        reloaded.obs["op_behaviour"].astype(str).to_numpy(),
    )
    np.testing.assert_allclose(
        phase_c_adata.obs["lead_re"], reloaded.obs["lead_re"],
    )

    # Phase C artefacts come back via the sidecar.
    np.testing.assert_allclose(orig["J"][0].toarray(), scj["J"][0].toarray())
    np.testing.assert_allclose(orig["J_latent"], scj["J_latent"])
    np.testing.assert_allclose(orig["grad_eta"], scj["grad_eta"])

    orig_tan = _ledger_part(orig["fisher_ledger"], "off_tangent")
    new_tan = _ledger_part(scj["fisher_ledger"], "off_tangent")
    np.testing.assert_array_equal(orig_tan["rank"], new_tan["rank"])
    np.testing.assert_allclose(orig_tan["projector"][0], new_tan["projector"][0])

    orig_F = _ledger_part(orig["fisher_ledger"], "fisher")
    new_F = _ledger_part(scj["fisher_ledger"], "fisher")
    np.testing.assert_allclose(orig_F[0].toarray(), new_F[0].toarray())

    # Complex eigenpairs survive.
    np.testing.assert_allclose(
        np.nan_to_num(phase_c_adata.obsm["op_eigvals"]),
        np.nan_to_num(reloaded.obsm["op_eigvals"]),
    )
    np.testing.assert_allclose(
        np.nan_to_num(orig["op_eigvec_lifted"]),
        np.nan_to_num(scj["op_eigvec_lifted"]),
    )

    # And drivers() still works after a cold load (no refit).
    df = opspace.drivers(reloaded, cell=0, top=5)
    assert len(df) == 5
    assert "reason" in df.columns
