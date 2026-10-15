#!/usr/bin/env python3
"""Fetch the HLCA core raw counts plus official scArches reference assets.

The script uses the CELLxGENE Census core dataset instead of a browser download
so the Census release and dataset identifier are captured in a local manifest.
It is idempotent: verified local artifacts are reused and all source URLs,
versions, byte counts, and SHA-256 values are retained in manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anndata as ad
import cellxgene_census as cxc

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "benchmarks" / "aging_hlca" / "data"
MANIFEST = DATA_DIR / "manifest.json"
CORE_PATH = DATA_DIR / "hlca_core_raw_counts.h5ad"
MODEL_ZIP = DATA_DIR / "HLCA_reference_model.zip"
MODEL_DIR = DATA_DIR / "HLCA_reference_model"
EMBED_PATH = DATA_DIR / "HLCA_full_v1.1_emb.h5ad"
ZENODO_RECORD = "7599104"
ZENODO_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
ZENODO_CONTENT = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files"
MODEL_URL = f"{ZENODO_CONTENT}/HLCA_reference_model.zip/content"
EMBED_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files/HLCA_full_v1.1_emb.h5ad?download=1"
MAX_TOTAL_GB = 30.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, source_url: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if source_url is not None:
        record["source_url"] = source_url
    return record


def download(url: str, destination: Path, expected_bytes: int) -> None:
    """Resume a public asset in bounded byte ranges with exact size checks."""
    if destination.exists() and destination.stat().st_size == expected_bytes:
        print(f"reusing verified {destination.name} ({expected_bytes} bytes)", flush=True)
        return
    if destination.exists() and destination.stat().st_size != expected_bytes:
        raise RuntimeError(f"existing {destination} has unexpected size; remove it manually before retrying")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    legacy_parts = sorted(destination.parent.glob(destination.name + ".*.part"), key=lambda path: path.stat().st_size, reverse=True)
    if not partial.exists() and legacy_parts:
        legacy_parts[0].replace(partial)
    chunk_bytes = 8 * 1024 * 1024
    while True:
        current = partial.stat().st_size if partial.exists() else 0
        if current == expected_bytes:
            partial.replace(destination)
            print(f"completed {destination.name} ({expected_bytes} bytes)", flush=True)
            return
        if current > expected_bytes:
            raise RuntimeError(f"partial {partial} exceeds expected Zenodo size")
        end = min(current + chunk_bytes - 1, expected_bytes - 1)
        expected_chunk = end - current + 1
        temporary = destination.with_suffix(destination.suffix + f".{current}-{end}.part")
        temporary.unlink(missing_ok=True)
        for attempt in range(1, 4):
            print(
                f"range {current}-{end} ({expected_chunk} bytes), attempt {attempt}/3: {destination.name}",
                flush=True,
            )
            result = subprocess.run(
                [
                    "curl", "--fail", "--location", "--range", f"{current}-{end}",
                    "--connect-timeout", "30", "--max-time", "300", "--retry", "2",
                    "--retry-delay", "5", "--retry-all-errors", "--speed-limit", "1024",
                    "--speed-time", "90", "--output", str(temporary), url,
                ],
                check=False,
            )
            size = temporary.stat().st_size if temporary.exists() else 0
            if result.returncode == 0 and size == expected_chunk:
                with partial.open("ab") as output, temporary.open("rb") as input_handle:
                    shutil.copyfileobj(input_handle, output, length=8 * 1024 * 1024)
                temporary.unlink(missing_ok=True)
                print(f"retained {partial.stat().st_size}/{expected_bytes} bytes for {destination.name}", flush=True)
                break
            temporary.unlink(missing_ok=True)
            print(f"range attempt ended curl={result.returncode}, got {size}/{expected_chunk} bytes", flush=True)
        else:
            raise RuntimeError(f"unable to retrieve byte range {current}-{end} for {destination.name}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value if isinstance(value, (str, int, float, bool)) or value is None else str(value)


def _select_core_dataset(datasets):
    """Identify the single healthy HLCA core dataset from Census metadata."""
    required = {"dataset_id", "dataset_title"}
    missing = required.difference(datasets.columns)
    if missing:
        raise RuntimeError(f"Census dataset catalog lacks required columns: {sorted(missing)}")
    title = datasets["dataset_title"].astype(str).str.lower()
    collection = datasets.get("collection_name", "").astype(str).str.lower()
    matches = datasets[title.str.contains("integrated cell atlas of the lung", na=False)].copy()
    if matches.empty:
        matches = datasets[(title.str.contains("lung", na=False)) & collection.str.contains("lung", na=False)].copy()
    core = matches[title.loc[matches.index].str.contains("core", na=False)].copy()
    if len(core) == 1:
        return core.iloc[0]
    if len(matches) == 1:
        return matches.iloc[0]
    candidate_columns = [column for column in ("dataset_id", "dataset_title", "collection_id", "collection_name", "dataset_total_cell_count") if column in matches]
    candidates = matches[candidate_columns].to_dict(orient="records")
    raise RuntimeError(
        "unable to resolve exactly one HLCA core Census dataset; candidates=" + json.dumps(candidates, sort_keys=True)
    )


def _census_release() -> tuple[str, dict[str, Any]]:
    description = cxc.get_census_version_description("stable")
    if not isinstance(description, dict):
        description = dict(description)
    release = str(description.get("release_build") or description.get("name") or "stable")
    return release, _jsonable(description)


def fetch_core() -> dict[str, Any]:
    release, release_description = _census_release()
    print(f"opening CELLxGENE Census release {release}", flush=True)
    with cxc.open_soma(census_version=release) as census:
        datasets = census["census_info"]["datasets"].read().concat().to_pandas()
        selected = _select_core_dataset(datasets)
        dataset_id = str(selected["dataset_id"])
        print(f"selected HLCA core dataset {dataset_id}: {selected.get('dataset_title', '<unknown>')}", flush=True)
        if not CORE_PATH.exists():
            print("retrieving HLCA raw count matrix and complete observation metadata from Census", flush=True)
            adata = cxc.get_anndata(
                census,
                organism="homo_sapiens",
                measurement_name="RNA",
                X_name="raw",
                obs_value_filter=f"dataset_id == '{dataset_id}'",
                obs_column_names=None,
                var_column_names=None,
            )
            if adata.n_obs == 0 or adata.n_vars == 0:
                raise RuntimeError("Census returned an empty HLCA core AnnData object")
            adata.uns["hlca_h01_fetch"] = {
                "source": "CZ CELLxGENE Census",
                "census_release": release,
                "dataset_id": dataset_id,
                "matrix": "X_name=raw",
            }
            CORE_PATH.parent.mkdir(parents=True, exist_ok=True)
            temporary = CORE_PATH.with_suffix(".h5ad.part")
            print(f"writing {adata.n_obs} cells × {adata.n_vars} genes to {temporary.name}", flush=True)
            adata.write_h5ad(temporary)
            temporary.replace(CORE_PATH)
            print(f"wrote {CORE_PATH.name}", flush=True)
    row = {key: _jsonable(value) for key, value in selected.to_dict().items()}
    return {
        "census_release": release,
        "census_release_description": release_description,
        "dataset": row,
        "dataset_id": dataset_id,
        "core": file_record(CORE_PATH),
        "n_cells": int(ad.read_h5ad(CORE_PATH, backed="r").n_obs),
    }


def fetch_reference_assets() -> dict[str, Any]:
    with urllib.request.urlopen(ZENODO_API, timeout=60) as response:
        zenodo = json.load(response)
    expected = {entry["key"]: int(entry["size"]) for entry in zenodo.get("files", [])}
    for name in (MODEL_ZIP.name, EMBED_PATH.name):
        if name not in expected:
            raise RuntimeError(f"Zenodo record {ZENODO_RECORD} does not provide {name}")
    total_public_bytes = expected[MODEL_ZIP.name] + expected[EMBED_PATH.name]
    if total_public_bytes / 1024**3 > MAX_TOTAL_GB:
        raise RuntimeError(f"reference download would exceed {MAX_TOTAL_GB} GB")
    download(MODEL_URL, MODEL_ZIP, expected[MODEL_ZIP.name])
    download(EMBED_URL, EMBED_PATH, expected[EMBED_PATH.name])
    if MODEL_ZIP.stat().st_size != expected[MODEL_ZIP.name]:
        raise RuntimeError("HLCA reference model zip byte count does not match Zenodo metadata")
    if EMBED_PATH.stat().st_size != expected[EMBED_PATH.name]:
        raise RuntimeError("HLCA embedding byte count does not match Zenodo metadata")
    if not (MODEL_DIR / "model.pt").exists():
        with zipfile.ZipFile(MODEL_ZIP) as archive:
            archive.extractall(DATA_DIR)
    if not (MODEL_DIR / "model.pt").exists():
        raise RuntimeError("HLCA reference model archive did not unpack HLCA_reference_model/model.pt")
    return {
        "zenodo_record": ZENODO_RECORD,
        "zenodo_record_url": f"https://zenodo.org/records/{ZENODO_RECORD}",
        "reference_model": file_record(MODEL_ZIP, MODEL_URL),
        "reference_embedding": file_record(EMBED_PATH, EMBED_URL),
        "zenodo_file_sizes": expected,
        "reference_model_directory": str(MODEL_DIR.relative_to(ROOT)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-total-gb", type=float, default=MAX_TOTAL_GB)
    parser.add_argument("--phase", choices=("all", "reference", "core"), default="all")
    args = parser.parse_args()
    if args.max_total_gb != MAX_TOTAL_GB:
        raise ValueError(f"H01 hard limit is {MAX_TOTAL_GB} GB; do not override it")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    reference = fetch_reference_assets()
    if args.phase == "reference":
        return 0
    core = fetch_core()
    total = sum(path.stat().st_size for path in DATA_DIR.rglob("*") if path.is_file())
    if total / 1024**3 > MAX_TOTAL_GB:
        raise RuntimeError(f"HLCA data directory now exceeds {MAX_TOTAL_GB} GB; stop and ask")
    manifest = {
        "task": "hlca/H01",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hlca_source": "CZ CELLxGENE Census raw matrix plus Zenodo HLCA scArches assets",
        "raw_counts_location": "hlca_core_raw_counts.h5ad:X (Census X_name=raw)",
        "reference": reference,
        "core": core,
        "disk_bytes": total,
        "disk_gb_decimal": total / 1_000_000_000,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
