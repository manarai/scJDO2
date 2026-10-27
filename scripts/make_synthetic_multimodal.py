#!/usr/bin/env python3
"""Write seeded synthetic multimodal H5AD datasets and their manifests.

The historical S01 ``--all`` path is intentionally v1-only so it recreates the
committed local baseline exactly.  V2 data are generated explicitly with
``--v2-all`` and live under ``data/v2/`` with their own manifest.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
from scjdo2.data.synthetic_multimodal import (
    V2_PRESETS,
    V3_PRESETS,
    embed_linear,
    make_multimodal,
    true_velocity_in_embedding,
)

ROOT = Path(__file__).resolve().parents[1]
V1_PRESETS = ("default", "stress", "batch", "stationary_cycle")
ALL_PRESETS = V1_PRESETS + V2_PRESETS + V3_PRESETS
V1_DATA_DIR = ROOT / "benchmarks/synthetic_multimodal/data"
V2_DATA_DIR = V1_DATA_DIR / "v2"
V3_DATA_DIR = V1_DATA_DIR / "v3"
V1_MANIFEST = ROOT / "benchmarks/synthetic_multimodal/MANIFEST.csv"
V2_MANIFEST = ROOT / "benchmarks/synthetic_multimodal/MANIFEST_v2.csv"
V3_MANIFEST = ROOT / "benchmarks/synthetic_multimodal/MANIFEST_v3.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_dataset(preset: str, seed: int, output_dir: Path) -> dict[str, object]:
    adata = make_multimodal(preset=preset, seed=seed)
    is_v2 = preset in V2_PRESETS
    is_v3 = preset in V3_PRESETS
    is_modern = is_v2 or is_v3
    for method in ("pca", "fa"):
        embed_linear(adata, method=method, n_comp=20)
        true_velocity_in_embedding(adata, method=method)
        # The historical v1 H5ADs predate expected-count velocity keys.  Do not
        # mutate their shape: task S05 requires bit-for-bit baseline invariance.
        if is_modern:
            true_velocity_in_embedding(adata, method=method, at="expected")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{preset}_s{seed}.h5ad"
    adata.write_h5ad(output, compression="gzip")
    meta = adata.uns["synthetic_multimodal"]
    detected_fraction = float(adata.obs["clone"].notna().mean())
    knobs = {
        "p_barcode": float(meta["p_barcode"]),
        "labelling_efficiency": float(meta["labelling_efficiency"]),
        "pulse_hours": [float(v) for v in meta["pulse_hours"]],
        "stress_beta_fraction": float(meta["stress_beta_fraction"]),
        "batch_effect": bool(meta["batch_effect"]),
        "stationary_cycle": bool(meta["stationary_cycle"]),
    }
    if is_v2:
        knobs.update(version="v2", capture_eta_unspliced=float(meta["capture_eta_unspliced"]))
    if is_v3:
        knobs.update(version="v3", **{key: meta[key] for key in (
            "mu_b", "sigma_b", "f_constant", "f_switch", "n_prog", "theta0",
            "gamma0", "sigma_gamma", "beta0", "eta_u", "p_barcode", "n_founders", "f_sample",
        )})
        # orient/S10 Step 0/0b fields are stored only when not at their v3 defaults.
        for key in ("z1_crit", "pulse_batch_effect"):
            if key in meta:
                knobs[key] = meta[key].item() if hasattr(meta[key], "item") else meta[key]
    return {
        "preset": preset,
        "seed": int(seed),
        "path": str(output.relative_to(ROOT)),
        "sha256": sha256(output),
        "n_cells": int(adata.n_obs),
        "n_collected_by_day": json.dumps([int(v) for v in meta["n_collected_by_day"]]),
        "n_clones_detected": int(meta["n_clones_detected"]),
        "detected_clone_fraction": detected_fraction,
        "f_sample": float(meta["f_sample"]),
        "knob_values": json.dumps(knobs, sort_keys=True),
    }


def write_manifest(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["preset", "seed", "path", "sha256", "n_cells", "n_collected_by_day",
              "n_clones_detected", "detected_clone_fraction", "f_sample", "knob_values"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def append_manifest(row: dict[str, object], path: Path) -> None:
    """Add or replace exactly one preset/seed entry without changing other rows."""
    existing: list[dict[str, object]] = []
    if path.exists():
        with path.open(newline="") as handle:
            existing = list(csv.DictReader(handle))
    key = (str(row["preset"]), int(row["seed"]))
    rows = [old for old in existing if (str(old["preset"]), int(old["seed"])) != key]
    rows.append(row)
    rows.sort(key=lambda value: (str(value["preset"]), int(value["seed"])))
    write_manifest(rows, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=ALL_PRESETS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--all", action="store_true", help="write historical v1 presets for seeds 0, 1, and 2")
    parser.add_argument("--v2-all", action="store_true", help="write all v2 presets for seeds 0, 1, and 2")
    parser.add_argument("--v3-all", action="store_true", help="write all v3 presets for seeds 0, 1, and 2")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--append-manifest", action="store_true",
                        help="add/replace one --preset/--seed row in --manifest; preserves existing rows")
    args = parser.parse_args()
    if sum(bool(v) for v in (args.all, args.v2_all, args.v3_all)) > 1:
        parser.error("--all, --v2-all, and --v3-all are mutually exclusive")
    if args.all or args.v2_all or args.v3_all:
        if args.preset is not None or args.seed is not None:
            parser.error("--all/--v2-all/--v3-all cannot be combined with --preset or --seed")
        if args.append_manifest:
            parser.error("--append-manifest requires one --preset and --seed")
        presets = V3_PRESETS if args.v3_all else (V2_PRESETS if args.v2_all else V1_PRESETS)
        output_dir = args.output_dir or (V3_DATA_DIR if args.v3_all else (V2_DATA_DIR if args.v2_all else V1_DATA_DIR))
        manifest = args.manifest or (V3_MANIFEST if args.v3_all else (V2_MANIFEST if args.v2_all else V1_MANIFEST))
        rows = [write_dataset(preset, seed, output_dir) for preset in presets for seed in range(3)]
        write_manifest(rows, manifest)
        print(f"wrote {len(rows)} datasets and {manifest}")
        return
    if args.preset is None or args.seed is None:
        parser.error("--preset and --seed are required unless --all or --v2-all is used")
    is_v2 = args.preset in V2_PRESETS
    is_v3 = args.preset in V3_PRESETS
    output_dir = args.output_dir or (V3_DATA_DIR if is_v3 else (V2_DATA_DIR if is_v2 else V1_DATA_DIR))
    row = write_dataset(args.preset, args.seed, output_dir)
    if args.manifest is not None:
        if args.append_manifest:
            append_manifest(row, args.manifest)
        else:
            write_manifest([row], args.manifest)
    print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
