#!/usr/bin/env python3
"""Export one dyngen raw bundle to AnnData with truth kept separate."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks/dyngen"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_matrix(path: Path) -> sp.csr_matrix:
    matrix = scipy.io.mmread(path)
    return matrix.tocsr() if sp.issparse(matrix) else sp.csr_matrix(np.asarray(matrix))


def table_mapping(path: Path) -> dict[str, np.ndarray]:
    """Return a columnar, AnnData-writable representation of a TSV table."""
    frame = pd.read_csv(path, sep="\t", na_values=["NA"])
    result: dict[str, np.ndarray] = {}
    for column in frame.columns:
        values = frame[column]
        if pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
            result[column] = values.to_numpy()
        else:
            result[column] = values.fillna("").astype(str).to_numpy()
    return result


def export_one(raw: Path, output: Path, key: str, backbone: str, seed: int) -> dict[str, object]:
    raw = raw.resolve()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(raw / "cells.tsv", sep="\t", dtype=str)["cell_id"].astype(str).to_numpy()
    obs = pd.read_csv(raw / "obs.tsv", sep="\t", na_values=["NA"])
    if len(obs) != len(cells):
        raise RuntimeError(f"{key}: obs/cell count mismatch")
    obs.index = cells
    obs.index.name = "cell_id"
    var = pd.read_csv(raw / "var.tsv", sep="\t", na_values=["NA"])
    if len(var) == 0 or "feature_id" not in var:
        raise RuntimeError(f"{key}: missing feature metadata")
    var.index = var["feature_id"].astype(str)
    var.index.name = "feature_id"
    var = var.drop(columns=["feature_id"])

    spliced = read_matrix(raw / "spliced.mtx")
    unspliced = read_matrix(raw / "unspliced.mtx")
    if spliced.shape != (len(cells), len(var)) or unspliced.shape != spliced.shape:
        raise RuntimeError(f"{key}: count shape mismatch: {spliced.shape} {unspliced.shape}")
    if not np.equal(spliced.data, np.floor(spliced.data)).all() or not np.equal(unspliced.data, np.floor(unspliced.data)).all():
        raise RuntimeError(f"{key}: raw counts are not integer-valued")

    data = ad.AnnData(X=spliced.copy(), obs=obs, var=var)
    data.layers["spliced"] = spliced.copy()
    data.layers["unspliced"] = unspliced.copy()
    velocity_path = raw / "truth_rna_velocity.mtx"
    if velocity_path.exists():
        data.layers["truth_rna_velocity"] = read_matrix(velocity_path)

    settings = json.loads((raw / "settings.json").read_text())
    data.uns["truth_grn"] = table_mapping(raw / "truth_grn.tsv")
    data.uns["truth_milestone_network"] = table_mapping(raw / "truth_milestone_network.tsv")
    data.uns["dyngen_settings"] = settings
    data.uns["s19"] = {"key": key, "backbone": backbone, "seed": int(seed), "truth_prefix": "truth_"}
    data.write_h5ad(output, compression="gzip")

    reopened = ad.read_h5ad(output, backed="r")
    try:
        if reopened.shape != data.shape:
            raise RuntimeError(f"{key}: h5ad shape changed after write")
        if "spliced" not in reopened.layers or "unspliced" not in reopened.layers:
            raise RuntimeError(f"{key}: missing raw layers after write")
        if "day" not in reopened.obs:
            raise RuntimeError(f"{key}: missing obs['day']")
        truth_obs = [column for column in reopened.obs.columns if column.startswith("truth_")]
        if not truth_obs:
            raise RuntimeError(f"{key}: no truth-prefixed obs fields")
    finally:
        reopened.file.close()

    checkpoint = OUT / "checkpoints" / f"{key}.json"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "complete",
        "task": "orient/S19",
        "key": key,
        "backbone": backbone,
        "seed": int(seed),
        "path": str(output.relative_to(ROOT)),
        "sha256": sha256(output),
        "bytes": output.stat().st_size,
        "cells": int(data.n_obs),
        "genes": int(data.n_vars),
        "settings": settings,
        "finished_utc": now(),
    }
    checkpoint.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def write_manifest() -> None:
    rows = []
    for checkpoint in sorted((OUT / "checkpoints").glob("*.json")):
        payload = json.loads(checkpoint.read_text())
        if payload.get("status") != "complete":
            continue
        rows.append({
            "key": payload["key"],
            "backbone": payload["backbone"],
            "seed": payload["seed"],
            "path": payload["path"],
            "sha256": payload["sha256"],
            "cells": payload["cells"],
            "genes": payload["genes"],
            "settings": f"benchmarks/dyngen/raw/{payload['key']}/settings.txt",
        })
    expected = {f"{backbone}_s{seed}" for backbone in ("bifurcating", "cycle", "linear") for seed in range(3)}
    found = {row["key"] for row in rows}
    if found != expected:
        raise RuntimeError(f"manifest requires all nine datasets; missing={sorted(expected - found)} extra={sorted(found - expected)}")
    pd.DataFrame(rows).sort_values(["backbone", "seed"]).to_csv(OUT / "MANIFEST.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw")
    parser.add_argument("--output")
    parser.add_argument("--key")
    parser.add_argument("--backbone", choices=["bifurcating", "cycle", "linear"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--manifest", action="store_true")
    args = parser.parse_args()
    if args.manifest:
        write_manifest()
        return
    required = (args.raw, args.output, args.key, args.backbone, args.seed)
    if any(value is None for value in required):
        parser.error("one dataset requires --raw, --output, --key, --backbone, and --seed")
    payload = export_one(Path(args.raw), Path(args.output), args.key, args.backbone, args.seed)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
