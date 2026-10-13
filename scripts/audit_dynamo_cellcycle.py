"""Audit checks (4) + (5) + (3') on the Dynamo hematopoiesis §7 run:

  (4) GATA1 / SPI1 ranks in both responders and drivers_left per
      archetype — written as a compact table.
  (5) Project each archetype's leading responder (right-ev) and
      driver (left-ev) eigenvector onto Seurat S + G2M gene lists;
      report the fraction of squared weight on cell-cycle genes.
  (3') tangent-rank distribution (dumped as a numpy array to CSV).

Full re-run of Phase C with the measured-flow drift so the archetype
dicts carry gene-space eigvec weights. We skip the h5ad sidecar save
(it OOMs on this dataset) and write CSVs instead.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scjdo2 import opspace
from scjdo2.data import load_dynamo_hematopoiesis
from scjdo2.measured_drift import MeasuredDrift, MeasuredDriftConfig
from scjdo2.pipeline import fit_rate_and_gene_field


REPO = Path(__file__).resolve().parent.parent

# Seurat / Tirosh 2016 standard cell-cycle gene panels (human symbols).
# Reproduced from Seurat's bundled ``cc.genes$s.genes`` + ``g2m.genes``.
SEURAT_S_GENES = {
    "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG",
    "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", "MLF1IP",
    "HELLS", "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76",
    "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51",
    "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM",
    "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8",
}
SEURAT_G2M_GENES = {
    "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A",
    "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF",
    "TACC3", "FAM64A", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB",
    "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1", "KIF20B", "HJURP",
    "CDCA3", "HN1", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1",
    "NCAPD2", "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR",
    "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF",
    "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA",
}
CYCLE_UNION = SEURAT_S_GENES | SEURAT_G2M_GENES
MUST_KEEP = ("GATA1", "SPI1")


def _project_on_set(v: np.ndarray, gene_names: np.ndarray,
                    gene_set: set) -> float:
    """Fraction of squared magnitude of ``v`` that falls on ``gene_set``.
    """
    m = np.abs(np.asarray(v)) ** 2
    s = float(m.sum())
    if s <= 0:
        return float("nan")
    mask = np.array([g in gene_set for g in gene_names], dtype=bool)
    return float(m[mask].sum() / s)


def _project_on_cycle(v: np.ndarray, gene_names: np.ndarray,
                      dynamo_cycle: dict) -> dict:
    """Return the squared-magnitude fraction of ``v`` on each cycle
    panel: Seurat S + G2M (Tirosh 2016) and Dynamo's own
    ``uns['cell_phase_genes']`` (curated for this dataset)."""
    dynamo_union: set = set()
    for genes in dynamo_cycle.values():
        dynamo_union |= set(genes)
    present_seurat = (SEURAT_S_GENES | SEURAT_G2M_GENES) & set(map(str, gene_names))
    present_dynamo = dynamo_union & set(map(str, gene_names))
    return {
        "frac_seurat_cycle": _project_on_set(v, gene_names,
                                              SEURAT_S_GENES | SEURAT_G2M_GENES),
        "frac_seurat_S": _project_on_set(v, gene_names, SEURAT_S_GENES),
        "frac_seurat_G2M": _project_on_set(v, gene_names, SEURAT_G2M_GENES),
        "n_seurat_cycle_in_var": int(len(present_seurat)),
        "frac_dynamo_cycle": _project_on_set(v, gene_names, dynamo_union),
        "n_dynamo_cycle_in_var": int(len(present_dynamo)),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-pcs", type=int, default=10)
    p.add_argument("--drift-epochs", type=int, default=1500)
    p.add_argument("--rate-epochs", type=int, default=150)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--n-anchors", type=int, default=200)
    p.add_argument("--anchor-max-live", type=int, default=200)
    p.add_argument("--cv-sample-cells", type=int, default=15)
    p.add_argument("--out", type=Path,
                   default=REPO / "benchmarks" / "v2" / "dynamo_hematopoiesis")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[audit] loading Dynamo hemat (d_latent={args.n_pcs})")
    a = load_dynamo_hematopoiesis(n_pcs=args.n_pcs)
    gene_names = np.asarray(a.var_names).astype(str)
    import anndata as _ad
    _proc = _ad.read_h5ad(
        REPO / "benchmarks" / "data" / "dynamo_hematopoiesis" / "data" / "hematopoiesis.h5ad",
    )
    dynamo_cycle: dict = _proc.uns.get("cell_phase_genes", {})
    dyn_union = set()
    for v_ in dynamo_cycle.values():
        dyn_union |= set(v_)
    print(f"[audit]   shape={a.shape}  "
          f"Seurat cycle in var: {sum(g in set(gene_names) for g in CYCLE_UNION)}/{len(CYCLE_UNION)}  "
          f"Dynamo cycle in var: {len(dyn_union & set(gene_names))}/{len(dyn_union)}")

    # Measured drift fit.
    print(f"[audit] drift fit ({args.drift_epochs} epochs)")
    t0 = time.monotonic()
    z = np.asarray(a.obsm["X_pca"])
    v = np.asarray(a.obsm["velocity_pca"])
    drift = MeasuredDrift(dim=z.shape[1], config=MeasuredDriftConfig(
        hidden=(96, 96, 96), n_epochs=args.drift_epochs,
        weight_decay=0.0, seed=0, clock_calibrated=True,
    )).fit(z, v, verbose=False)
    print(f"[audit]   drift fit {time.monotonic()-t0:.1f}s")

    # Autodiff latent J + per-cell lead_re for the write-back.
    J_latent = drift.jacobian_autodiff(z)
    v_pred = drift.velocity(z)
    scj = a.uns.setdefault("scjdo2", {})
    scj["J"] = J_latent
    scj["meta"] = {
        "clock_calibrated": True, "use_rep": "X_pca", "time_key": None,
        "jacobian_method": "autodiff",
    }
    scj["field_reason_summary"] = {"ok": 1.0, "autodiff": True}
    a.obsm["X_drift_v"] = v_pred
    a.obs["pseudotime"] = a.obsm["X_pca"][:, 0].astype(float)

    # Phase C.
    print(f"[audit] Phase C (anchor max_live={args.anchor_max_live})")
    t0 = time.monotonic()
    fit_rate_and_gene_field(
        a, use_rep="X_pca", n_epochs_rate=args.rate_epochs,
        k=args.k, cv_sample_cells=args.cv_sample_cells, seed=0,
        anchor_fit=True, n_anchors=args.n_anchors, anchor_top_k=10,
        anchor_max_live=args.anchor_max_live,
    )
    print(f"[audit]   Phase C in {time.monotonic()-t0:.1f}s")
    a.uns["scjdo2"]["meta"]["clock_calibrated"] = True

    # (3') Tangent rank from the ledger.
    led = a.uns["scjdo2"].get("fisher_ledger")
    tangent_ranks = None
    if hasattr(led, "off_tangent") and led.off_tangent is not None:
        tangent_ranks = np.asarray(led.off_tangent.get("rank", np.array([])))
    elif isinstance(led, dict) and led.get("off_tangent"):
        tangent_ranks = np.asarray(led["off_tangent"].get("rank", np.array([])))
    if tangent_ranks is not None and tangent_ranks.size:
        tr_series = pd.Series(tangent_ranks).value_counts().sort_index()
        tr_series.to_csv(args.out / "audit_tangent_rank_hist.csv",
                         header=["n_cells"], index_label="rank")
        print(f"\n[audit] (3') tangent rank distribution:\n{tr_series.to_string()}")
        print(f"[audit] (3') min={tangent_ranks.min()} median={int(np.median(tangent_ranks))} "
              f"max={tangent_ranks.max()}")
    else:
        print("[audit] (3') tangent rank unavailable")

    # opspace.
    print("[audit] opspace.compute")
    opspace.compute(a)
    arch = a.uns["scjdo2"].get("archetypes") or {}
    print(f"[audit]   archetypes: {len(arch)}  behaviours: "
          f"{sorted({v['behaviour'] for v in arch.values()})}")

    # (4) GATA1/SPI1 ranks per archetype — more compact than existing CSV.
    rows_rank = []
    rows_proj = []
    rows_top = []
    for ak, info in arch.items():
        for table_name in ("responders", "drivers_left"):
            df = info.get(table_name)
            if df is None:
                continue
            # (5) cell-cycle projection on this archetype's eigvec.
            eig_name = "eigvec" if table_name == "responders" else "eigvec_left"
            v_arch = info.get(eig_name)
            if v_arch is not None:
                proj = _project_on_cycle(v_arch, gene_names, dynamo_cycle)
                proj.update({"archetype": int(ak), "table": table_name,
                             "behaviour": info["behaviour"],
                             "lead_re_per_h": float(info["eigenvalues"][0].real)})
                rows_proj.append(proj)
            for g in MUST_KEEP:
                hit = df.index[df["gene"].astype(str) == g].tolist()
                if not hit:
                    rows_rank.append({"archetype": int(ak), "table": table_name,
                                      "gene": g, "rank": None})
                    continue
                i = int(hit[0])
                r = df.loc[i].to_dict()
                rows_rank.append({
                    "archetype": int(ak), "table": table_name, "gene": g,
                    "rank": i, "fisher": float(r["fisher"]),
                    "snr": float(r.get("snr", float("nan"))),
                    "sensitivity": float(r.get("sensitivity", float("nan"))),
                    "reason": str(r["reason"]),
                })
            # Also dump top-10 of each table for the record.
            top = df.head(10).reset_index(drop=True).copy()
            top.insert(0, "table", table_name)
            top.insert(0, "archetype", int(ak))
            rows_top.append(top)

    pd.DataFrame(rows_rank).to_csv(args.out / "audit_gata1_spi1_ranks.csv", index=False)
    pd.DataFrame(rows_proj).to_csv(args.out / "audit_cellcycle_projection.csv", index=False)
    pd.concat(rows_top, ignore_index=True).to_csv(
        args.out / "audit_archetype_top10.csv", index=False,
    )

    print("\n[audit] (4) GATA1 / SPI1 ranks per archetype:")
    print(pd.DataFrame(rows_rank).to_string(index=False))
    print("\n[audit] (5) cell-cycle projection per archetype eigvec:")
    print(pd.DataFrame(rows_proj).to_string(index=False))
    print(f"\n[audit] wrote CSVs under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
