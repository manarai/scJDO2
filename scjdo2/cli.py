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
    run.add_argument("--input", required=True,
                     help="input .h5ad (or run dir with adata.h5ad)")
    run.add_argument("--out", required=True,
                     help="output run directory (adata.h5ad + phase_c.npz)")
    run.add_argument("--use-rep", default="X_pca",
                     help="latent key in adata.obsm (default: X_pca)")
    run.add_argument("--n-epochs", type=int, default=1500,
                     dest="n_epochs_drift",
                     help="energy-drift epochs (default: 1500)")
    run.add_argument("--k", type=int, default=30,
                     help="gene-space kNN radius (default: 30)")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--with-rate-model", action="store_true",
                     help="also fit NB rate model + gene-space J + Fisher "
                          "ledger + tangent + grad_eta (Phase C)")
    run.add_argument("--rate-epochs", type=int, default=400,
                     dest="n_epochs_rate",
                     help="NB rate-model epochs (default: 400)")
    run.add_argument("--cv-sample-cells", type=int, default=30,
                     help="sampled cells for ridge-λ CV (default: 30)")
    run.add_argument("--anchor-fit", action="store_true",
                     help="use the anchor-based Step 5 variant (k-means "
                          "anchors + batched ridge + Gaussian kernel lift)")
    run.add_argument("--n-anchors", type=int, default=300,
                     help="anchor count for --anchor-fit (default: 300)")
    run.add_argument("--anchor-top-k", type=int, default=10,
                     help="anchors used per cell's lift (default: 10)")
    run.add_argument("--anchor-batch", type=int, default=25,
                     help="anchor-batch size for the ridge solve (default: 25)")

    ops = sub.add_parser("opspace",
                         help="compute operator-space invariants on a stored run")
    ops.add_argument("--input", required=True,
                     help="run directory containing adata.h5ad")
    ops.add_argument("--out", required=True,
                     help="directory to write per_cell_invariants.csv etc.")
    ops.add_argument("--smooth-k", type=int, default=0,
                     help="pre-smooth the J field by k-NN averaging (§1.1); 0 = off")
    ops.add_argument("--n-eig", type=int, default=5)
    ops.add_argument("--min-fisher", type=float, default=None,
                     help="override the per-cell Fisher-mask threshold")

    fig = sub.add_parser("figure",
                         help="produce one of the standard figures from a stored run")
    fig_sub = fig.add_subparsers(dest="fig_cmd", required=True)

    f_os = fig_sub.add_parser("operator-space")
    f_os.add_argument("--input", required=True)
    f_os.add_argument("--out", required=True)
    f_os.add_argument("--use-rep", default=None)
    f_os.add_argument("--archetype", type=int, default=None,
                      help="archetype id to render in Panel C (default: auto)")
    f_os.add_argument("--lineages", type=str, default=None,
                      help="path to a cells.txt file with one start-cell index per line")

    f_arch = fig_sub.add_parser("archetypes")
    f_arch.add_argument("--input", required=True)
    f_arch.add_argument("--out", required=True)

    f_cmp = fig_sub.add_parser("compare")
    f_cmp.add_argument("--v1", required=True)
    f_cmp.add_argument("--v2", required=True)
    f_cmp.add_argument("--out", required=True)

    f_ledger = fig_sub.add_parser("ledger",
                                  help="supplementary ledger-summary figure")
    f_ledger.add_argument("--input", required=True)
    f_ledger.add_argument("--out", required=True)

    cmp_ = sub.add_parser("compare", help="produce side-by-side comparison report")
    cmp_.add_argument("--v1", required=True)
    cmp_.add_argument("--v2", required=True)
    cmp_.add_argument("--out", required=True)

    return p


def _load_adata(input_arg: str):
    """Load a run directory (adata.h5ad + phase_c.npz sidecar) via
    :func:`scjdo2.io.load`. Falls back to a plain ``read_h5ad`` for a
    bare ``.h5ad`` input."""
    import anndata as ad
    from . import io as _scjdo2_io

    p = Path(input_arg)
    if p.is_dir():
        h5 = p / "adata.h5ad"
    else:
        h5 = p
    if not h5.exists():
        raise SystemExit(
            f"[scjdo2] input not found: {h5}. "
            "Expected a run directory containing adata.h5ad "
            "(produced by `scjdo2 run` or the Phase C pipeline)."
        )
    sidecar = h5.parent / "phase_c.npz"
    if sidecar.exists():
        return _scjdo2_io.load(h5.parent)
    return ad.read_h5ad(h5)


def _require(adata, uns_keys=(), obs_keys=(), producer="scjdo2.opspace.compute"):
    uns = adata.uns.get("scjdo2", {})
    for k in uns_keys:
        if k not in uns:
            raise SystemExit(
                f"[scjdo2] adata.uns['scjdo2']['{k}'] missing; run `{producer}` first."
            )
    for k in obs_keys:
        if k not in adata.obs:
            raise SystemExit(
                f"[scjdo2] adata.obs['{k}'] missing; run `{producer}` first."
            )


def _cmd_opspace(args) -> int:
    from . import io as _scjdo2_io
    from . import opspace

    adata = _load_adata(args.input)
    opspace.compute(
        adata,
        smooth_k=int(args.smooth_k),
        n_eig=int(args.n_eig),
        min_fisher=args.min_fisher,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = opspace.export_tables(adata, out)
    saved = _scjdo2_io.save(adata, out)
    print(
        f"opspace: wrote {len(paths)} CSVs + {saved['h5ad'].name}"
        + (f" + {saved['sidecar'].name}" if saved.get("sidecar") else "")
        + f" under {out}"
    )
    return 0


def _cmd_fig_operator_space(args) -> int:
    from .plot import figure_operator_space

    adata = _load_adata(args.input)
    _require(
        adata, obs_keys=("lead_re", "op_behaviour"),
        producer="scjdo2 opspace",
    )
    lineages = None
    if args.lineages:
        from . import opspace
        cells = [int(line.strip()) for line in Path(args.lineages).read_text().splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
        lineages = opspace.lineage_path(adata, start_cells=cells)
    out = Path(args.out) / "operator_space"
    out.parent.mkdir(parents=True, exist_ok=True)
    figure_operator_space(
        adata, lineages=lineages, archetype=args.archetype,
        out=out, use_rep=args.use_rep,
    )
    print(f"figure: wrote {out}.pdf and {out}.png")
    return 0


def _cmd_fig_archetypes(args) -> int:
    from .plot import figure_archetype_sheet

    adata = _load_adata(args.input)
    _require(adata, uns_keys=("archetypes",),
             producer="scjdo2.opspace.archetype_operators (via `scjdo2 opspace`)")
    out = Path(args.out) / "archetype_sheet"
    out.parent.mkdir(parents=True, exist_ok=True)
    figure_archetype_sheet(adata, out=out)
    print(f"figure: wrote {out}.pdf and {out}.png")
    return 0


def _cmd_fig_compare(args) -> int:
    from .plot import figure_v1_vs_v2

    adata_v2 = _load_adata(args.v2)
    _require(adata_v2, obs_keys=("lead_re",), producer="scjdo2 opspace")
    out = Path(args.out) / "v1_vs_v2"
    out.parent.mkdir(parents=True, exist_ok=True)
    figure_v1_vs_v2(adata_v2, v1_results=Path(args.v1), out=out)
    print(f"figure: wrote {out}.pdf and {out}.png")
    return 0


def _cmd_fig_ledger(args) -> int:
    from .plot import figure_ledger_summary

    adata = _load_adata(args.input)
    out = Path(args.out) / "ledger_summary"
    out.parent.mkdir(parents=True, exist_ok=True)
    figure_ledger_summary(adata, out=out)
    print(f"figure: wrote {out}.pdf and {out}.png")
    return 0


def _cmd_run(args) -> int:
    from . import io as _scjdo2_io
    from .energy_drift import EnergyDriftConfig
    from .pipeline import fit_latent_field, fit_rate_and_gene_field

    adata = _load_adata(args.input)
    if args.use_rep not in adata.obsm:
        raise SystemExit(
            f"[scjdo2] adata.obsm['{args.use_rep}'] missing — provide "
            "the latent representation (e.g. X_pca from `sc.tl.pca`)."
        )
    print(f"[run] fitting energy drift + field "
          f"({args.n_epochs_drift} epochs, use_rep={args.use_rep})")
    fit_latent_field(
        adata, use_rep=args.use_rep,
        drift_config=EnergyDriftConfig(
            hidden=96, depth=3, n_epochs=int(args.n_epochs_drift),
            seed=int(args.seed), batch_size=256,
        ),
    )
    if args.with_rate_model:
        variant = "anchor" if args.anchor_fit else "per_cell"
        print(f"[run] fitting NB rate model + gene-space J "
              f"({args.n_epochs_rate} rate epochs, k={args.k}, variant={variant})")
        fit_rate_and_gene_field(
            adata, use_rep=args.use_rep,
            n_epochs_rate=int(args.n_epochs_rate),
            k=int(args.k),
            cv_sample_cells=int(args.cv_sample_cells),
            seed=int(args.seed),
            anchor_fit=bool(args.anchor_fit),
            n_anchors=int(args.n_anchors),
            anchor_top_k=int(args.anchor_top_k),
            anchor_batch=int(args.anchor_batch),
        )
    out = Path(args.out)
    saved = _scjdo2_io.save(adata, out)
    print(f"[run] wrote {saved['h5ad'].name}"
          + (f" + {saved['sidecar'].name}" if saved.get("sidecar") else "")
          + f" under {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "run":
        return _cmd_run(args)
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
        if args.fig_cmd == "ledger":
            return _cmd_fig_ledger(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
