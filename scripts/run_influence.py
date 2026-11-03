"""Driver / responder analyses + figures on a saved scJDO2 run.

Thin wrapper over :mod:`scjdo2.drivers`::

    python scripts/run_influence.py benchmarks/v2/paul15 \
        --ref-file trrust_rawdata.mouse.tsv --time-key pseudotime \
        --out benchmarks/v2/paul15/drivers

Runs ``drivers.analyse`` (influence → roles → key genes → edges →
reference check → expression-timing QC), writes every table as CSV and
renders ``influence_network``, ``expression_cascade`` and
``reference_auroc`` (pdf + png).

``--ref-file`` is a TRRUST-style TSV; ``--ref`` a v1 network name
(``builtin`` | ``trrust`` | ``collectri`` | ``none``) used when no file is
given. ``--time-key none`` skips the timing QC. ``--branches paul15``
adds the branch-aware timing QC (root 7MEP; erythroid / monocyte /
neutrophil paths).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from scjdo2 import drivers, io

# Branch presets: root clusters + per-fate clusters (obs[cluster_key]).
BRANCH_PRESETS = {
    "paul15": {
        "cluster_key": "paul15_clusters",
        "root": ["7MEP"],
        "branches": {
            "erythroid": ["6Ery", "5Ery", "4Ery", "3Ery", "2Ery", "1Ery"],
            "monocyte": ["9GMP", "10GMP", "14Mo", "15Mo"],
            "neutrophil": ["9GMP", "10GMP", "16Neu", "17Neu"],
        },
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--organism", default="mouse")
    ap.add_argument("--ref", default="none")
    ap.add_argument("--ref-file", default=None)
    ap.add_argument("--time-key", default="pseudotime")
    ap.add_argument("--quantile", type=float, default=0.9)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--archetypes", default=None, help="comma-separated ids for the figures")
    ap.add_argument("--branches", default=None, choices=sorted(BRANCH_PRESETS),
                    help="branch preset for the branch-aware timing QC")
    ap.add_argument("--network-expression", action="store_true",
                    help="also render the per-archetype network-genes-across-states QC "
                         "(needs --branches for the cluster key)")
    args = ap.parse_args()

    out = Path(args.out or Path(args.run_dir) / "drivers")
    adata = io.load(args.run_dir)

    ref = None
    if args.ref_file:
        ref = drivers.load_reference(args.ref_file)
    elif args.ref != "none":
        ref = drivers.load_reference(args.ref, organism=args.organism)
    if ref is not None:
        print(f"[reference] {len(ref):,} edges, {ref['source'].nunique()} TFs")

    time_key = None if args.time_key == "none" else args.time_key
    br = BRANCH_PRESETS.get(args.branches, {})
    res = drivers.analyse(adata, ref_network=ref, time_key=time_key,
                          quantile=args.quantile, n_perm=args.n_perm,
                          branches=br.get("branches"), cluster_key=br.get("cluster_key"),
                          root=br.get("root", ()))
    drivers.save_tables(res, out)
    ids = [int(a) for a in args.archetypes.split(",")] if args.archetypes else None
    paths = drivers.figures(adata, out, ref_network=ref, time_key=time_key,
                            results=res, archetype_ids=ids,
                            cluster_key=br.get("cluster_key") if args.network_expression else None)

    print(res["roles"].to_string(index=False))
    for key in ("reference_summary", "timing_summary", "branch_pooled", "branch_overall",
                "branch_members_pooled", "branch_members_overall"):
        if key in res:
            print(f"\n{key}:\n{res[key].to_string(index=False)}")
    print(f"\nwrote tables + {', '.join(paths)} to {out}")


if __name__ == "__main__":
    main()
