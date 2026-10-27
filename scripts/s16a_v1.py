#!/usr/bin/env python3
"""S16a wrapper for the published scJDO v1 ``_run_fit_drift`` core.

Run with ``SCJDO2_V1_ENV=1 .sandbox/bin/python`` after ``scripts/install_v1_compare.sh``.
The v1 API prepares pseudotime from a root cell rather than accepting an
external ordering, so this wrapper records the requested Palantir input and
uses its S10 root cell in ``prepare_trajectory``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import site
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V1_ENV = ROOT / ".venv-v1"
V1_SOURCE = ROOT / ".sandbox/scjdo-v1"


def bootstrap_v1() -> None:
    """Load the isolated v1 dependencies while keeping the sandbox launcher."""
    site_dir = V1_ENV / "lib/python3.13/site-packages"
    if not site_dir.is_dir():
        raise RuntimeError(f"v1 environment site-packages missing: {site_dir}")
    sys.prefix = str(V1_ENV.resolve())
    sys.exec_prefix = str(V1_ENV.resolve())
    resolved_site_dir = site_dir.resolve()
    site.addsitedir(str(resolved_site_dir))
    # The sandbox launcher starts with the task's primary venv on sys.path.
    # Put v1's packages first and remove that primary site-packages entry so
    # the editable v1 checkout cannot silently mix dependency environments.
    sys.path[:] = [
        entry for entry in sys.path
        if Path(entry or ".").resolve() == resolved_site_dir
        or Path(entry or ".").name not in {"site-packages", "dist-packages"}
    ]
    sys.path.insert(0, str(resolved_site_dir))
    sys.path.insert(0, str(V1_SOURCE.resolve()))


if os.environ.get("SCJDO2_V1_ENV") == "1":
    bootstrap_v1()

import anndata as ad
import numpy as np

DATA_ROOT = Path("/Users/terooatt/Documents/scJDO2-manus/benchmarks/synthetic_multimodal/data/v3")
MANIFEST = Path("/Users/terooatt/Documents/scJDO2-manus/benchmarks/synthetic_multimodal/MANIFEST_v3.csv")
S10_ROOT = Path("/Users/terooatt/Documents/scJDO2-manus/benchmarks/synthetic_multimodal/data/s10")
OUT = ROOT / "benchmarks/baselines/scjdo_v1"
PSEUDOTIME_OUT = OUT / "pseudotime"
DATASETS = ("v3_transition_s2", "v3_transition_s3", "v3_transition_s4", "v3_default_s2", "v3_default_s3")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_pseudotime(dataset: str, adata: ad.AnnData) -> tuple[np.ndarray, str, dict]:
    seed = int(dataset.rsplit("s", 1)[1])
    day_names = adata.obs_names[np.asarray(adata.obs["day"], dtype=int) == 2].to_numpy(dtype=str)
    draw = np.random.default_rng(1000 + seed).choice(day_names, 5, replace=False)
    root = str(draw[0])
    names = adata.obs_names.to_numpy(dtype=str)
    pseudotime_info = {"strict_mode": True, "root_rule": "default_rng(1000 + seed).choice(day-2 cells, 5, replace=False), first"}
    s10 = S10_ROOT / dataset / "palantir.npz"
    if seed == 2 and s10.is_file():
        with np.load(s10, allow_pickle=False) as z:
            stored_names = z["names"].astype(str)
            stored_values = np.asarray(z["day2_strict_r0"], dtype=float)
        if len(stored_names) != len(stored_values) or set(stored_names) != set(names):
            raise RuntimeError(f"S10 Palantir cell names do not align for {dataset}")
        index = {name: i for i, name in enumerate(stored_names)}
        values = np.asarray([stored_values[index[name]] for name in names], dtype=float)
        if root not in day_names:
            raise RuntimeError("selected root is not an observed day-2 cell")
        pseudotime_info.update({"source": str(s10), "key": "day2_strict_r0", "cell_names_key": "names",
                                "pseudotime_sha256": hashlib.sha256(values.tobytes()).hexdigest()})
    else:
        import palantir
        import pandas as pd
        work = ad.AnnData(X=np.empty((adata.n_obs, 0), dtype=np.float32),
                          obs=pd.DataFrame(index=adata.obs_names.copy()),
                          var=pd.DataFrame(index=pd.Index([], dtype=str)),
                          obsm={"X_pca": np.asarray(adata.obsm["X_pca"], dtype=np.float64).copy()})
        palantir.utils.run_diffusion_maps(work, n_components=10, knn=30, seed=0, pca_key="X_pca")
        palantir.utils.determine_multiscale_space(work)
        palantir.core.run_palantir(work, early_cell=root, num_waypoints=500, knn=30, seed=0,
                                   use_early_cell_as_start=True)
        values = work.obs["palantir_pseudotime"].to_numpy(dtype=float)
        pseudotime_info.update({"source": "computed with S03 Palantir settings", "n_components": 10,
                                "knn": 30, "num_waypoints": 500, "palantir_seed": 0,
                                "use_early_cell_as_start": True,
                                "pseudotime_sha256": hashlib.sha256(values.tobytes()).hexdigest()})
    return values, root, {"root_cell": root, "dataset_seed": seed, **pseudotime_info}


def main() -> None:
    started = time.monotonic()
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--pseudotime-only", action="store_true")
    args = p.parse_args()
    rows = list(csv.DictReader(MANIFEST.open(newline="")))
    row = next((r for r in rows if args.dataset in r["path"]), None)
    if row is None: raise RuntimeError(f"manifest row not found for {args.dataset}")
    data_path = DATA_ROOT / f"{args.dataset}.h5ad"
    if sha256(data_path) != row["sha256"].lower(): raise RuntimeError(f"input checksum mismatch: {data_path}")
    adata = ad.read_h5ad(data_path)
    pseudotime, root, pseudo = strict_pseudotime(args.dataset, adata)
    if args.pseudotime_only:
        pseudo["wall_seconds"] = round(time.monotonic() - started, 3)
        PSEUDOTIME_OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(PSEUDOTIME_OUT / f"{args.dataset}.npz", names=adata.obs_names.to_numpy(dtype=str),
                            day2_strict_r0=pseudotime, root=np.asarray(root))
        (PSEUDOTIME_OUT / f"{args.dataset}.json").write_text(json.dumps(pseudo, indent=2, sort_keys=True) + "\n")
        return
    out = args.out or OUT / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)

    try:
        from compare.run_v1 import _run_fit_drift
    except Exception as exc:
        raise RuntimeError("could not import the v1 comparison core in .venv-v1; see the chained import error") from exc
    pseudo.update({"external_pseudotime_used": False,
                   "fallback": "prepare_trajectory computed ordering from the same strict-mode Palantir root; _run_fit_drift has no pseudotime argument"})
    # v1's groupby/root interface takes a category, not an obs name. A unique
    # per-cell group makes DPT start from the exact selected Palantir root.
    adata.obs["s16a_root_cell"] = adata.obs_names.astype(str)
    result = _run_fit_drift(adata, out, seed=0, n_epochs=5000, archetype_method="snmf",
                            groupby="s16a_root_cell", root=root)
    uns = adata.uns.get("scjdo", {})
    arrays = {}
    def collect(prefix: str, value: object) -> None:
        if isinstance(value, dict):
            for k, v in value.items(): collect(f"{prefix}__{k}", v)
        elif isinstance(value, (np.ndarray, list, tuple, np.number, int, float)):
            try: arrays[prefix] = np.asarray(value)
            except Exception: pass
    collect("scjdo", uns)
    np.savez_compressed(out / f"{args.dataset}.npz", **arrays)
    (out / f"{args.dataset}.json").write_text(json.dumps({
        "method": "scJDO v1", "dataset": args.dataset, "input_sha256": row["sha256"],
        "seed": 0, "n_epochs": 5000, "archetype_method": "snmf",
        "preprocessing": "v1 defaults: log-normalise, 2000 HVGs, FA-30",
        "pseudotime": pseudo, "settings": result, "operator_arrays": list(arrays)}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__": main()
