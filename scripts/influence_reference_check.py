"""Reference-network recovery check only (TRRUST etc.).

Thin wrapper over :func:`scjdo2.drivers.reference_check` and
:func:`scjdo2.drivers.plot_reference`::

    python scripts/influence_reference_check.py benchmarks/v2/paul15 \
        --ref-file trrust_rawdata.mouse.tsv --out benchmarks/v2/paul15/influence_trrust

For the full set of analyses use ``scripts/run_influence.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from scjdo2 import drivers, io


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ref-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    adata = io.load(args.run_dir)
    drivers.influence(adata)
    check = drivers.reference_check(adata, drivers.load_reference(args.ref_file),
                                    n_perm=args.n_perm, seed=args.seed)
    for name, df in check.items():
        df.to_csv(out / f"reference_{name}.csv", index=False)
    for ext in ("pdf", "png"):
        drivers.plot_reference(check, save=out / f"reference_auroc.{ext}")
    print(check["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
