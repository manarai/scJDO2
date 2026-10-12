"""scjdo2 CLI.

Subcommands (Phase B):

    scjdo2 run             --input <h5ad> --embed scvi|fa|pca --k auto --out <dir>
    scjdo2 opspace         --input <run_dir> --out <run_dir>/opspace/
    scjdo2 figure operator-space --input <run_dir> --out <figs_dir>
    scjdo2 figure archetypes     --input <run_dir> --out <figs_dir>
    scjdo2 figure compare        --v1 <dir> --v2 <run_dir> --out <figs_dir>
    scjdo2 compare         --v1 <dir> --v2 <dir>

The ``opspace`` and ``figure`` commands read the on-disk run directory and
never re-fit. The run directory is produced by ``scjdo2 run`` (Phase D) or
by programmatic use of :func:`scjdo2.pipeline.fit_latent_field` +
``adata.write_h5ad``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scjdo2")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run the scJDO2 pipeline on an h5ad input")
    run.add_argument("--input", required=True)
    run.add_argument("--embed", choices=["scvi", "fa", "pca"], default="pca")
    run.add_argument("--k", default="auto")
    run.add_argument("--n-archetypes", default="auto")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--out", required=True)

    ops = sub.add_parser("opspace",
                         help="compute operator-space invariants on a stored run")
    ops.add_argument("--input", required=True,
                     help="run directory containing adata.h5ad")
    ops.add_argument("--out", required=True,
                     help="directory to write per_cell_invariants.csv etc.")
    ops.add_argument("--n-eig", type=int, default=5)
    ops.add_argument("--field-gradient-k", type=int, default=15)

    fig = sub.add_parser("figure",
                         help="produce one of the standard figures from a stored run")
    fig_sub = fig.add_subparsers(dest="fig_cmd", required=True)

    f_os = fig_sub.add_parser("operator-space")
    f_os.add_argument("--input", required=True)
    f_os.add_argument("--out", required=True)
    f_os.add_argument("--use-rep", default=None)

    f_arch = fig_sub.add_parser("archetypes")
    f_arch.add_argument("--input", required=True)
    f_arch.add_argument("--out", required=True)

    f_cmp = fig_sub.add_parser("compare")
    f_cmp.add_argument("--v1", required=True)
    f_cmp.add_argument("--v2", required=True)
    f_cmp.add_argument("--out", required=True)

    cmp_ = sub.add_parser("compare", help="produce side-by-side comparison report")
    cmp_.add_argument("--v1", required=True)
    cmp_.add_argument("--v2", required=True)
    cmp_.add_argument("--out", required=True)

    return p


def _cmd_opspace(args) -> int:
    import anndata as ad
    from . import opspace

    run_dir = Path(args.input)
    adata = ad.read_h5ad(run_dir / "adata.h5ad")
    opspace.spectrum(adata, n_eig=args.n_eig, fisher_mask=False)
    opspace.invariants(adata)
    opspace.behaviour(adata)
    opspace.field_gradient(adata, k=args.field_gradient_k)
    out = Path(args.out)
    paths = opspace.export_tables(adata, out)
    adata.write_h5ad(out / "adata.h5ad")
    print(f"opspace: wrote {len(paths)} CSVs and adata.h5ad under {out}")
    return 0


def _cmd_fig_operator_space(args) -> int:
    import anndata as ad
    from .plot import figure_operator_space

    adata = ad.read_h5ad(Path(args.input) / "adata.h5ad")
    out = Path(args.out) / "operator_space"
    figure_operator_space(adata, out=out, use_rep=args.use_rep)
    print(f"figure: wrote {out}.pdf and {out}.png")
    return 0


def _cmd_fig_archetypes(args) -> int:
    import anndata as ad
    from .plot import figure_archetype_sheet

    adata = ad.read_h5ad(Path(args.input) / "adata.h5ad")
    figure_archetype_sheet(adata, out=Path(args.out) / "archetype_sheet")
    return 0


def _cmd_fig_compare(args) -> int:
    import anndata as ad
    from .plot import figure_v1_vs_v2

    adata_v2 = ad.read_h5ad(Path(args.v2) / "adata.h5ad")
    out = Path(args.out) / "v1_vs_v2"
    figure_v1_vs_v2(adata_v2, v1_results=Path(args.v1), out=out)
    print(f"figure: wrote {out}.pdf and {out}.png")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "run":
        raise NotImplementedError("Phase D: wire scjdo2 pipeline end-to-end")
    if args.cmd == "compare":
        raise NotImplementedError("Phase D: delegate to compare.v1_vs_v2")
    if args.cmd == "opspace":
        return _cmd_opspace(args)
    if args.cmd == "figure":
        if args.fig_cmd == "operator-space":
            return _cmd_fig_operator_space(args)
        if args.fig_cmd == "archetypes":
            return _cmd_fig_archetypes(args)
        if args.fig_cmd == "compare":
            return _cmd_fig_compare(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
