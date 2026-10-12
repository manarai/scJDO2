"""scjdo2 CLI.

Two subcommands:

    scjdo2 run --input <h5ad> --embed scvi|fa|pca --k auto --out <dir>
    scjdo2 compare --v1 <dir> --v2 <dir>

Phase D wires the full pipeline; Phase A ships this skeleton so the console
entry point is registered on install.
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scjdo2")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run the scJDO2 pipeline on an h5ad input")
    run.add_argument("--input", required=True)
    run.add_argument("--embed", choices=["scvi", "fa", "pca"], default="scvi")
    run.add_argument("--k", default="auto")
    run.add_argument("--n-archetypes", default="auto")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--out", required=True)

    cmp_ = sub.add_parser("compare", help="produce side-by-side report")
    cmp_.add_argument("--v1", required=True)
    cmp_.add_argument("--v2", required=True)
    cmp_.add_argument("--out", required=True)

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "run":
        raise NotImplementedError("Phase D: wire scjdo2 pipeline end-to-end")
    if args.cmd == "compare":
        raise NotImplementedError("Phase D: delegate to compare.v1_vs_v2")
    return 1


if __name__ == "__main__":
    sys.exit(main())
