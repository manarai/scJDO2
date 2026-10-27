#!/usr/bin/env python3
"""Run S16a native baseline operators one dataset/method at a time.

The task launcher selects the method environment. CellRank uses
.venv-baselines; spliceJAC uses .venv-splicejac when launched through
.sandbox/bin/python-splicejac; dynamo uses .sandbox/bin/python-dynamo. No
synthetic truth arrays are read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("/Users/terooatt/Documents/scJDO2-manus/benchmarks/synthetic_multimodal/MANIFEST_v3.csv")
DATA_ROOT = Path("/Users/terooatt/Documents/scJDO2-manus/benchmarks/synthetic_multimodal/data/v3")
OUT = ROOT / "benchmarks/baselines"
DATASETS = ("v3_transition_s2", "v3_transition_s3", "v3_transition_s4", "v3_default_s2", "v3_default_s3")
METHODS = ("dynamo_oneshot", "scvelo_dynamical", "splicejac", "cellrank")


def bootstrap_baselines() -> None:
    # spliceJAC uses its own Python 3.11 environment.  Do not prepend the
    # primary Python 3.13 baseline site-packages in that launcher.
    if os.environ.get("SCJDO2_SPLICEJAC_ENV") == "1":
        return
    env = ROOT / ".venv-baselines"
    site = env / "lib/python3.13/site-packages"
    if site.is_dir():
        sys.prefix = str(env.resolve())
        sys.exec_prefix = str(env.resolve())
        sys.path.insert(0, str(site.resolve()))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def input_for(dataset: str) -> tuple[Path, str]:
    rows = list(csv.DictReader(MANIFEST.open(newline="")))
    matches = [r for r in rows if dataset in r.get("path", "")]
    if len(matches) != 1:
        raise RuntimeError(f"expected one manifest row for {dataset}, got {len(matches)}")
    row = matches[0]
    p = DATA_ROOT / f"{dataset}.h5ad"
    expected = row["sha256"].lower()
    if not p.is_file() or sha256(p) != expected:
        raise RuntimeError(f"input checksum mismatch for {p}")
    return p, expected


def _npz(path: Path, **arrays: object) -> None:
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def run_splicejac(data_path: Path, out_path: Path) -> tuple[dict, str]:
    bootstrap_baselines()
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import splicejac as sj

    adata = ad.read_h5ad(data_path)
    # The PCA coordinates are observed inputs supplied with the dataset.
    sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=30, random_state=0)
    sc.tl.leiden(adata, resolution=1.0, random_state=0, key_added="s16a_leiden")
    adata.obs["clusters"] = adata.obs["s16a_leiden"].astype(str)
    # SpliceJAC's documented default estimator and default regression settings.
    sj.tl.estimate_jacobian(adata)
    # transition_genes consumes Scanpy's per-cluster differential expression
    # ranking; spliceJAC documents this prerequisite and asserts its presence.
    sc.tl.rank_genes_groups(adata, groupby="clusters")
    native = adata.uns["average_jac"]
    labels = sorted(map(str, native))
    jacobians = np.stack([np.asarray(native[c][0]) for c in labels])
    eigenvalues = np.stack([np.asarray(native[c][1]) for c in labels])
    positive_fraction = np.mean(eigenvalues.real > 0, axis=1)
    transition_names, transition_weights = [], []
    adata.uns["transitions"] = {}
    for source in labels:
        for target in labels:
            if source != target:
                sj.tl.transition_genes(adata, source, target)
                key = f"{source}-{target}"
                entry = adata.uns["transitions"][key]
                transition_names.append(key)
                transition_weights.append(np.asarray(entry["weights"], dtype=float))
    max_weights = max((len(x) for x in transition_weights), default=0)
    padded = np.full((len(transition_weights), max_weights), np.nan)
    for i, values in enumerate(transition_weights): padded[i, :len(values)] = values
    _npz(out_path, clusters=np.asarray(labels), jacobians=jacobians, eigenvalues=eigenvalues,
         positive_eigenvalue_fraction=positive_fraction, transition_pairs=np.asarray(transition_names),
         transition_gene_weights=padded, transition_gene_names=np.asarray(adata.var_names, dtype=str))
    version = metadata.version("splicejac")
    return {"method": "splicejac", "cluster_key": "s16a_leiden", "resolution": 1.0,
            "neighbors": {"use_rep": "X_pca", "n_neighbors": 30, "random_state": 0},
            "rank_genes_groups": {"groupby": "clusters", "method": "scanpy default (t-test)"},
            "estimate_jacobian": "default arguments (first_moment=True, method=Ridge, alpha=1, beta=1, rescale=True, nsim=10, frac=.9, filter_and_norm=True, min_shared_counts=20, n_top_genes=20, eps=.9, seed=100)",
            "native_uns_keys": ["jacobian_lists", "average_jac", "transitions"],
            "cluster_count": int(adata.obs["s16a_leiden"].nunique()), "transition_pair_count": len(transition_names),
            "operator_arrays": ["jacobians", "eigenvalues", "positive_eigenvalue_fraction", "transition_gene_weights"]}, version


def run_cellrank(data_path: Path, out_path: Path) -> tuple[dict, str]:
    bootstrap_baselines()
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import cellrank as cr
    from moscot.problems.time import TemporalProblem

    source = ad.read_h5ad(data_path)
    # Keep all cells while representing them by the supplied observed PCA-30
    # coordinates. This avoids moscot materializing the full gene matrix.
    pca = np.asarray(source.obsm["X_pca"], dtype=np.float32)
    adata = ad.AnnData(X=pca.copy(), obs=source.obs[["day"]].copy(),
                       var=pd.DataFrame(index=[f"PC{i + 1}" for i in range(pca.shape[1])]),
                       obsm={"X_pca": pca.copy()})
    adata.obs_names = source.obs_names.copy()
    sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=30, random_state=0)
    tp = TemporalProblem(adata).prepare(time_key="day")
    tp.solve()
    rtk = cr.kernels.RealTimeKernel.from_moscot(tp)
    rtk.compute_transition_matrix()
    ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()
    kernel = 0.8 * rtk + 0.2 * ck
    estimator = cr.estimators.GPCCA(kernel)
    estimator.compute_schur()
    estimator.compute_macrostates(n_states=5)
    estimator.predict_terminal_states()
    estimator.compute_fate_probabilities()
    macro = np.asarray(adata.obs["macrostates"].astype(str)) if "macrostates" in adata.obs else np.asarray(estimator.macrostates.astype(str))
    terminal = np.asarray(adata.obs["term_states_fwd"].astype(str)) if "term_states_fwd" in adata.obs else np.asarray(estimator.terminal_states.astype(str))
    fate = np.asarray(estimator.fate_probabilities)
    transition = kernel.transition_matrix.tocsr()
    _npz(out_path, transition_data=transition.data.astype(np.float32), transition_indices=transition.indices,
         transition_indptr=transition.indptr, transition_shape=np.asarray(transition.shape), macrostates=macro,
         terminal_states=terminal, fate_probabilities=fate,
         cell_names=adata.obs_names.to_numpy(dtype=str))
    return {"method": "cellrank", "sampling_time_key": "day", "representation": "observed supplied X_pca (30 components)", "neighbors": {"use_rep": "X_pca", "n_neighbors": 30, "random_state": 0},
            "kernels": "0.8 * RealTimeKernel(moscot TemporalProblem) + 0.2 * ConnectivityKernel", "gpcca": {"n_states": 5, "terminal_states": "predicted", "fate_probabilities": True}}, metadata.version("cellrank")


def run_dynamo(data_path: Path, out_path: Path, method: str) -> tuple[dict, str]:
    # The permitted dynamo environment is an external read-only environment;
    # the dispatcher catches and records import/runtime failures without editing it.
    import anndata as ad
    import numpy as np
    import dynamo as dyn
    adata = ad.read_h5ad(data_path)
    if method == "dynamo_oneshot":
        adata.layers["X_new"] = adata.layers["new"].copy()
        adata.layers["X_total"] = adata.layers["total"].copy()
        n_genes_before = adata.n_vars
        # Dynamo documents these recipe controls: select up to all source
        # genes as features, retain genes failing the default QC mask, and
        # preserve raw layers for the subsequent documented velocity_N pass.
        dyn.tl.recipe_one_shot_data(
            adata,
            tkey="pulse_hours",
            one_shot_method="sci-fate",
            n_top_genes=n_genes_before,
            keep_filtered_genes=True,
            keep_raw_layers=True,
        )
        gene_filter_diagnostics = {
            "genes_before_recipe": int(n_genes_before),
            "genes_after_recipe": int(adata.n_vars),
            "genes_passing_basic_filter": int(adata.var["pass_basic_filter"].sum()) if "pass_basic_filter" in adata.var else None,
            "genes_used_for_pca": int(adata.var["use_for_pca"].sum()) if "use_for_pca" in adata.var else None,
            "recipe_options": {"n_top_genes": int(n_genes_before), "keep_filtered_genes": True, "keep_raw_layers": True},
        }
        if adata.n_vars != n_genes_before or gene_filter_diagnostics["genes_used_for_pca"] != n_genes_before:
            raise RuntimeError(f"Dynamo recipe did not retain all source genes: {gene_filter_diagnostics}")
        dyn.tl.velocity_N(adata, recalculate_pca=True, recalculate_umap=False)
        dynamo_recipe_options = {
            "recipe": "dynamo.tl.recipe_one_shot_data",
            "one_shot_method": "sci-fate",
            "tkey": "pulse_hours",
            "gene_filter_diagnostics": gene_filter_diagnostics,
            "velocity_N": {"recalculate_pca": True, "recalculate_umap": False},
        }
    else:
        import scanpy as sc
        import scvelo as scv
        work = adata.copy(); work.X = work.layers["spliced"].copy()
        scv.pp.filter_genes(work, min_shared_counts=20); scv.pp.normalize_per_cell(work); sc.pp.log1p(work)
        sc.pp.neighbors(work, use_rep="X_pca", n_neighbors=30, random_state=0); scv.pp.moments(work)
        scv.tl.recover_dynamics(work); scv.tl.velocity(work, mode="dynamical"); scv.tl.velocity_graph(work)
        adata = work
        if "velocity_pca" not in work.obsm:
            raise RuntimeError("scVelo did not produce velocity_pca")
        adata.obsm["velocity_pca"] = np.asarray(work.obsm["velocity_pca"]).copy()
        if "X_pca" not in adata.obsm:
            adata.obsm["X_pca"] = np.asarray(work.obsm["X_pca"]).copy()
    vector_field = dyn.vf.VectorField(adata, basis="pca", method="SparseVFC",
                                      velocity_key="velocity_pca", return_vf_object=True,
                                      cores=3, restart_num=5, restart_seed=[0, 100, 200, 300, 400])
    jac = dyn.vf.jacobian(adata, basis="pca", vector_field_class=vector_field,
                          cell_idx=list(range(adata.n_obs)), sampling=None, store_in_adata=False)
    vector_field.find_fixed_points(n_x0=100, sampling_method="random")
    fixed_points, fixed_point_jacobians = vector_field.get_fixed_points()
    classes = vector_field.assign_fixed_points()
    _npz(out_path, velocity=np.asarray(adata.obsm["velocity_pca"]), jacobians=np.asarray(jac),
         fixed_points=np.asarray(fixed_points), fixed_point_jacobians=np.asarray(fixed_point_jacobians),
         fixed_point_classes=np.asarray(classes))
    return {"method": method, "velocity_basis": "PCA-30", "SparseVFC": "dynamo.vf.VectorField(method=SparseVFC)",
            "jacobian": "dynamo.vf.jacobian for all cells", "fixed_points": {"n_x0": 100, "sampling_method": "random"},
            "fixed_point_classes": "dynamo vector-field assignment", "cores": 3,
            **(dynamo_recipe_options if method == "dynamo_oneshot" else {})}, metadata.version("dynamo")


def append_csv(row: dict) -> None:
    path = OUT / "runs.csv"; path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "dataset", "status", "settings", "runtime_seconds", "peak_memory_bytes", "version", "error"]
    old = list(csv.DictReader(path.open())) if path.exists() else []
    old = [r for r in old if (r["method"], r["dataset"]) != (row["method"], row["dataset"])]
    old.append({k: json.dumps(row[k], sort_keys=True) if k == "settings" else str(row.get(k, "")) for k in fields})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(old)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    args = parser.parse_args()
    start = time.monotonic(); unit = f"{args.method}/{args.dataset}"
    result = OUT / args.method / f"{args.dataset}.npz"; detail = result.with_suffix(".json")
    row = {"method": args.method, "dataset": args.dataset, "status": "failed", "settings": {}, "runtime_seconds": 0, "peak_memory_bytes": "external /usr/bin/time -l", "version": "", "error": ""}
    try:
        path, digest = input_for(args.dataset)
        if args.method == "splicejac": settings, ver = run_splicejac(path, result)
        elif args.method == "cellrank": settings, ver = run_cellrank(path, result)
        else: settings, ver = run_dynamo(path, result, args.method)
        row.update(status="complete", settings=settings, version=ver)
        detail.write_text(json.dumps({**row, "input_sha256": digest, "input_path": str(path)}, indent=2, sort_keys=True) + "\n")
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        result.parent.mkdir(parents=True, exist_ok=True)
        detail.write_text(json.dumps({**row, "traceback": traceback.format_exc()}, indent=2, sort_keys=True) + "\n")
        print(f"FAILED {unit}: {row['error']}", file=sys.stderr, flush=True)
    finally:
        row["runtime_seconds"] = round(time.monotonic() - start, 3)
        append_csv(row)
    print(f"{row['status'].upper()} {unit} ({row['runtime_seconds']}s)", flush=True)
    if row["status"] != "complete": raise SystemExit(1)


if __name__ == "__main__":
    main()
