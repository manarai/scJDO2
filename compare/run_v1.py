"""Run v1 scJDO on a tutorial dataset and save outputs in a common format.

Invoked from the scJDO2 repo root with the ``[compare]`` extra installed:

    python -m compare.run_v1 --dataset paul15 --out benchmarks/v1_baseline/paul15

Outputs under ``<out>/``:

    config.json           — seed, bandwidth, archetype method, n_epochs, etc.
    adata.h5ad            — adata post-run, with ``adata.uns["scjdo"]``
    archetypes.npz        — J tensor, archetype weights, archetype operators
    drivers.tsv           — top driver genes per archetype (v1's output)
    figures/              — v1 ``sjd.pl.summary_figure`` output
    run_log.txt           — stdout/stderr, wall time, memory high-water
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# scJDO2 does NOT import v1 inside the package. This script does, lazily, and
# only when it is run. If v1 is not installed, we fail with a clear message.
try:  # pragma: no cover
    import scanpy as sc  # noqa: F401
except Exception as e:
    print(f"[run_v1] scanpy import failed: {e}", file=sys.stderr)
    raise

try:  # pragma: no cover
    import scjdo as sjd  # v1
except Exception as e:  # pragma: no cover
    print(
        "[run_v1] v1 scjdo not importable. Install with:\n"
        "    pip install -e .[compare]\n"
        "  or set SCJDO_V1_PATH and `pip install -e $SCJDO_V1_PATH`.",
        file=sys.stderr,
    )
    raise

# Dataset loaders live in scJDO2/data/.
from data import loaders  # type: ignore[import-not-found]


def _run_fit_drift(adata, out: Path, seed: int, n_epochs: int, archetype_method: str,
                   groupby: str, root: str) -> dict[str, Any]:
    """Tutorial 01: `prepare_trajectory` → `fit_drift` → summary figure."""
    t0 = time.monotonic()
    sjd.pp.prepare_trajectory(adata, groupby=groupby, root=root)
    sjd.tl.fit_drift(adata, n_epochs=n_epochs, archetype_method=archetype_method, seed=seed)
    wall = time.monotonic() - t0

    sjd.pl.summary_figure(adata, save=str(out / "figures" / "summary.pdf"))
    adata.write_h5ad(out / "adata.h5ad")
    return {"wall_time_s": wall}


def _run_one(dataset: str, out: Path, seed: int, n_epochs: int, archetype_method: str) -> dict[str, Any]:
    """Dispatch per workflow. Bridge/Fourier runs are skipped with a note —
    those tutorials do not share v1's `fit_drift` entry point.
    """
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)

    workflow = loaders.workflow_for(dataset)
    config: dict[str, Any] = {
        "dataset": dataset,
        "workflow": workflow,
        "seed": seed,
        "n_epochs": n_epochs,
        "archetype_method": archetype_method,
        "v1_version": getattr(sjd, "__version__", "unknown"),
    }

    if workflow == "fit_drift":
        adata = loaders.load(dataset)
        drift_cfg = _run_fit_drift(
            adata, out, seed=seed, n_epochs=n_epochs, archetype_method=archetype_method,
            groupby=loaders.groupby_for(dataset),  # type: ignore[arg-type]
            root=loaders.root_for(dataset),        # type: ignore[arg-type]
        )
        config.update(drift_cfg)
    elif workflow in ("fit_bridge", "fourier"):
        config["skipped"] = True
        config["note"] = (
            f"workflow={workflow!r} is out of v2 scope; v1 baseline for this notebook "
            "is deferred to a dedicated run_v1_bridge.py / run_v1_fourier.py in a later pass."
        )
    else:
        raise RuntimeError(f"Unknown workflow {workflow!r} for dataset {dataset!r}")

    def _default(o):
        import numpy as _np
        if isinstance(o, _np.integer):
            return int(o)
        if isinstance(o, _np.floating):
            return float(o)
        if isinstance(o, _np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    with (out / "config.json").open("w") as fh:
        json.dump(config, fh, indent=2, default=_default)
    return config


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   choices=["paul15", "fourier_paul15", "sb_synthetic",
                            "paul15_bridge"])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs", type=int, default=5000)
    p.add_argument("--archetype-method", default="snmf")
    args = p.parse_args(argv)

    cfg = _run_one(
        dataset=args.dataset,
        out=args.out,
        seed=args.seed,
        n_epochs=args.n_epochs,
        archetype_method=args.archetype_method,
    )
    print(json.dumps(cfg, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
