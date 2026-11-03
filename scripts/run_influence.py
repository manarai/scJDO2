"""Driver / responder influence per archetype on a saved scJDO2 run.

Usage::

    python scripts/run_influence.py benchmarks/v2/paul15 --organism mouse \
        --ref builtin --out benchmarks/v2/paul15/influence

Re-runs ``opspace.archetype_operators`` on the stored archetype weights
(so each entry carries the operator block and projector), then
``influence.archetype_influence`` and the network figure. ``--ref``
picks the v1 reference network source (``builtin`` | ``trrust`` |
``collectri`` | ``none``), or ``--ref-file`` a TSV.

Writes ``influence_long.csv`` (gene × archetype), ``roles_summary.csv``,
``edges.csv``, ``key_genes.csv`` and ``influence_network.{pdf,png}``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from scjdo2 import influence, io, opspace
from scjdo2.plot import figure_influence_network

KEY_GENES = ["Gata1", "Sfpi1", "Spi1", "Klf1", "Cebpa", "Cebpe", "Gata2",
             "Fli1", "Irf8", "Gfi1", "Zfpm1", "Mpo", "Car2", "Hba-a2"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--organism", default="mouse")
    ap.add_argument("--ref", default="builtin")
    ap.add_argument("--ref-file", default=None,
                    help="TSV source/target/mode[/pmid] (e.g. TRRUST raw data); overrides --ref")
    ap.add_argument("--quantile", type=float, default=0.9)
    ap.add_argument("--archetypes", default=None, help="comma-separated ids for the figure")
    args = ap.parse_args()

    out = Path(args.out or Path(args.run_dir) / "influence")
    out.mkdir(parents=True, exist_ok=True)

    adata = io.load(args.run_dir)
    opspace.archetype_operators(adata)
    long = influence.archetype_influence(adata, quantile=args.quantile)
    long.to_csv(out / "influence_long.csv", index=False)

    arch = adata.uns["scjdo2"]["archetypes"]
    rows = []
    for ak, info in sorted(arch.items()):
        tbl = info["influence"]
        counts = tbl["role"].value_counts()
        top = lambda role, col: ", ".join(tbl[tbl["role"] == role].nlargest(5, col)["gene"])
        rows.append({
            "archetype": ak, "behaviour": info["behaviour"], "n_cells": info["n_cells"],
            "lead_re": float(np.real(info["eigenvalues"][0])),
            "t_star": info["influence_t"], "t_rule": info["influence_t_rule"],
            **{f"n_{r}": int(counts.get(r, 0)) for r in influence.ROLES},
            "top_drivers": top("driver", "out_influence"),
            "top_hubs": top("hub", "out_influence"),
            "top_responders": top("responder", "in_influence"),
        })
    pd.DataFrame(rows).to_csv(out / "roles_summary.csv", index=False)

    keys = [g for g in KEY_GENES if g in set(adata.var_names)]
    long[long["gene"].isin(keys)][
        ["archetype", "behaviour", "gene", "role", "out_rank", "in_rank",
         "out_influence", "in_influence", "polarity", "coverage", "reason"]
    ].sort_values(["gene", "archetype"]).to_csv(out / "key_genes.csv", index=False)

    ref = None
    if args.ref_file:
        ref = pd.read_csv(args.ref_file, sep="\t", header=None).iloc[:, :3]
        ref.columns = ["source", "target", "mode"]
        ref["weight"] = ref["mode"].map({"Activation": 1.0, "Repression": -1.0}).fillna(0.0)
        print(f"[network] {args.ref_file}: {len(ref):,} edges, {ref['source'].nunique()} TFs")
    elif args.ref != "none":
        from scjdo.tl import load_network
        ref = load_network(organism=args.organism, source=args.ref)
    edges = []
    for ak in sorted(arch):
        e = influence.annotate_reference(influence.influence_edges(adata, ak), ref)
        edges.append(e.assign(archetype=ak))
    pd.concat(edges, ignore_index=True).to_csv(out / "edges.csv", index=False)

    ids = [int(a) for a in args.archetypes.split(",")] if args.archetypes else None
    for ext in ("pdf", "png"):
        figure_influence_network(adata, archetype_ids=ids, ref_network=ref,
                                 save=out / f"influence_network.{ext}")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
