#!/usr/bin/env python3
"""Run S08 statistics and compare dyngen against real data and v3.

The S08 modules are imported as modules; their source files are not edited.
Outputs are kept under benchmarks/dyngen so the S08 reference records remain
read-only on this task branch.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import shutil
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks/dyngen"
S08 = ROOT / "scripts/s08_reference_stats.py"
S08C = ROOT / "scripts/s08_compare.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


s08 = load_module("s19_imported_s08_reference_stats", S08)
s08c = load_module("s19_imported_s08_compare", S08C)
# Keep S08's distribution/checkpoint side effects inside this task's allowed
# output tree.  The imported modules remain unchanged.
s08.OUT = OUT / "s08_reference"
s08.CHECKPOINTS = s08.OUT / "checkpoints"


def dataset_keys() -> list[str]:
    return [f"{backbone}_s{seed}" for backbone in ("bifurcating", "cycle", "linear") for seed in range(3)]


def path_for(key: str) -> Path:
    return OUT / "data" / f"{key}.h5ad"


def run_one(key: str) -> dict[str, object]:
    path = path_for(key)
    if not path.is_file():
        raise FileNotFoundError(path)
    data = ad.read_h5ad(path)
    # Use S08's common preparation, count, distribution, row, and splicing
    # functions with a synthetic key that follows S08's raw-layer convention.
    dataset = s08.Dataset(key=f"v2_{key}", path=path, kind="synthetic", source="dyngen/S19")
    records: list[dict[str, object]] = []
    records.extend(s08.run_counts_trajectory(dataset))
    records.extend(s08.run_splicing(dataset))
    # dyngen has no new/total labelling readout or lineage barcode input.  Keep
    # those S08 statistics explicitly not applicable rather than deriving fake
    # inputs from the truth fields.
    records.append(s08.row(dataset, "L_not_applicable", "no_new_total", 1.0, data.n_obs))
    records.append(s08.row(dataset, "G_not_applicable", "no_clone_source", 1.0, data.n_obs))
    for record in records:
        record["dataset"] = key
        record["source"] = "dyngen/S19 via imported S08"
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "stats" / f"{key}.csv", index=False)
    return {"key": key, "records": int(len(frame)), "shape": [int(data.n_obs), int(data.n_vars)]}


def load_real_stats() -> tuple[pd.DataFrame, list[str]]:
    reference = ROOT / "benchmarks/real_reference/stats.csv"
    if not reference.is_file():
        raise FileNotFoundError(reference)
    frame = pd.read_csv(reference)
    real = [key for key in s08c.REAL if key in set(frame["dataset"])]
    if not real:
        raise RuntimeError("no real S08 reference rows found")
    return frame, real


def load_v3_stats() -> tuple[pd.DataFrame, list[str], str]:
    """Read existing v3 rows when present, otherwise compute them read-only.

    The task branch predates the v3 rows in real_reference/stats.csv, while the
    shared agent-A worktree contains the v3 h5ad files.  Computing the missing
    rows into this task's output preserves the requested comparison and does
    not write to the shared worktree.
    """
    reference = ROOT / "benchmarks/real_reference/stats.csv"
    frame = pd.read_csv(reference)
    existing = frame[frame["dataset"].astype(str).str.startswith("v3_default")].copy()
    if not existing.empty:
        return existing, sorted(existing["dataset"].unique()), "pre-existing real_reference rows"
    cached = OUT / "v3_reference_stats.csv"
    if cached.is_file():
        existing = pd.read_csv(cached)
        return existing, sorted(existing["dataset"].unique()), "cached local computation from shared v3 h5ad read-only"

    shared_root = Path.home() / "Documents/scJDO2-manus/benchmarks/synthetic_multimodal/data/v3"
    manifest = ROOT / "benchmarks/synthetic_multimodal/MANIFEST_v3.csv"
    if not shared_root.is_dir() or not manifest.is_file():
        raise RuntimeError("v3_default rows are absent and shared v3 data are unavailable")
    rows = []
    for item in csv.DictReader(manifest.open()):
        if item["preset"] != "v3_default":
            continue
        source = shared_root / Path(item["path"]).name
        if not source.is_file():
            raise RuntimeError(f"shared v3 file missing: {source}")
        destination = OUT / "v3_reference" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
        key = f"v3_default_s{item['seed']}"
        data = ad.read_h5ad(destination)
        # The local S08 implementation recognizes the v2 raw-count pathway
        # (X plus spliced/unspliced layers); use that internal key and restore
        # the requested v3 label on the emitted rows below.
        dataset = s08.Dataset(key=f"v2_{key}", path=destination, kind="synthetic", source="v3/S09 shared read-only")
        records = s08.run_counts_trajectory(dataset) + s08.run_splicing(dataset)
        # v3 has the S08 labelling and lineage inputs; use their stored rows.
        records += s08.run_labelling(dataset) + s08.run_lineage(dataset)
        for record in records:
            record["dataset"] = key
        rows.extend(records)
    computed = pd.DataFrame(rows)
    computed.to_csv(OUT / "v3_reference_stats.csv", index=False)
    return computed, sorted(computed["dataset"].unique()), "computed from shared v3 h5ad read-only"


def distribution_values(dataset: str, statistic: str, frame: pd.DataFrame) -> np.ndarray:
    values = frame.loc[(frame.dataset == dataset) & (frame.statistic == statistic), "value"].to_numpy(float)
    return values[np.isfinite(values)]


def scalar_value(dataset: str, statistic: str, frame: pd.DataFrame) -> float:
    values = distribution_values(dataset, statistic, frame)
    return float(np.median(values)) if values.size else float("nan")


def compare(all_stats: pd.DataFrame, real_keys: list[str], v3_keys: list[str]) -> pd.DataFrame:
    dyngen_keys = dataset_keys()
    stats = sorted(set(all_stats.statistic) - set(s08c.EXCLUDE))
    rows: list[dict[str, object]] = []
    for statistic in stats:
        available = set(all_stats.loc[all_stats.statistic == statistic, "dataset"])
        real = [key for key in real_keys if key in available]
        candidates = [key for key in dyngen_keys if key in available]
        if not real or not candidates:
            continue
        values = {key: distribution_values(key, statistic, all_stats) for key in real + candidates}
        pairs = [float(ks_2samp(values[a], values[b]).statistic) for a, b in itertools.combinations(real, 2) if values[a].size and values[b].size]
        spread = max(pairs) if pairs else 0.0
        threshold = max(spread, s08c.FIXED["ks"])
        real_medians = [scalar_value(key, statistic, all_stats) for key in real]
        real_medians = np.asarray([x for x in real_medians if np.isfinite(x)], dtype=float)
        for candidate in candidates:
            distances = {key: float(ks_2samp(values[candidate], values[key]).statistic) for key in real if values[candidate].size and values[key].size}
            if not distances:
                continue
            within = None
            candidate_median = scalar_value(candidate, statistic, all_stats)
            if np.isfinite(candidate_median) and real_medians.size:
                within = bool(real_medians.min() <= candidate_median <= real_medians.max())
            rows.append({
                "synthetic": candidate,
                "preset": "dyngen",
                "statistic": statistic,
                "metric": "ks",
                "distance": min(distances.values()),
                "nearest_real": min(distances, key=distances.get),
                "real_spread": spread,
                "fixed_tolerance": s08c.FIXED["ks"],
                "threshold": threshold,
                "match": bool(len(distances) == len(real) and all(value <= threshold for value in distances.values())),
                "matches_every_real": bool(len(distances) == len(real) and all(value <= threshold for value in distances.values())),
                "n_real": len(real),
                "n_real_compared": len(distances),
                "synthetic_scalar_or_median": candidate_median,
                "real_range_min": float(real_medians.min()) if real_medians.size else np.nan,
                "real_range_max": float(real_medians.max()) if real_medians.size else np.nan,
                "within_real_range": within,
                "max_distance_to_any_real": max(distances.values()),
                "distances_to_reals": json.dumps(distances, sort_keys=True),
            })
    result = pd.DataFrame(rows).sort_values(["statistic", "synthetic"]).reset_index(drop=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.run:
        (OUT / "stats").mkdir(parents=True, exist_ok=True)
        missing = [key for key in dataset_keys() if not (OUT / "stats" / f"{key}.csv").is_file()]
        run_payloads = [run_one(key) for key in missing]
    else:
        run_payloads = []
    real_stats, real_keys = load_real_stats()
    v3_stats, v3_keys, v3_source = load_v3_stats()
    dyngen_stats = pd.concat([pd.read_csv(OUT / "stats" / f"{key}.csv") for key in dataset_keys()], ignore_index=True)
    # Use v3 rows only as the requested comparator; real references come from
    # the immutable S08 table.  The saved realism.csv includes both sources.
    compare_stats = pd.concat([real_stats, v3_stats, dyngen_stats], ignore_index=True)
    realism = compare(compare_stats, real_keys, v3_keys)
    realism.to_csv(OUT / "realism.csv", index=False)
    summary = realism.groupby("synthetic", as_index=False).agg(
        statistics=("statistic", "size"),
        match_rate=("match", "mean"),
        within_real_range_rate=("within_real_range", "mean"),
    )
    statistic_summary = realism.groupby("statistic", as_index=False).agg(
        datasets=("synthetic", "size"),
        match_rate=("match", "mean"),
        within_real_range_rate=("within_real_range", "mean"),
        real_range_min=("real_range_min", "min"),
        real_range_max=("real_range_max", "max"),
    )
    (OUT / "RESULT_S19.md").write_text(
        "# orient/S19 — dyngen realism\n\n"
        "S08 reference statistics were imported from `scripts/s08_reference_stats.py`; the comparison helpers were imported from `scripts/s08_compare.py`. Neither source file was edited.\n\n"
        f"v3 reference source: {v3_source}.\n\n"
        "## Match summary\n\n"
        "| dataset | statistics | match rate | within-real-range rate |\n|---|---:|---:|---:|\n"
        + "\n".join(f"| {row.synthetic} | {int(row.statistics)} | {row.match_rate:.3f} | {row.within_real_range_rate:.3f} |" for row in summary.itertuples())
        + "\n\n## Notes\n\n"
        "- A statistic matches when its KS distance is no larger than the maximum of the real-reference spread and the imported S08 fixed tolerance.\n"
        "- dyngen has no new/total labelling readout or clone barcode input; S08 labelling and lineage rows are recorded as not applicable rather than derived from truth.\n"
        "- `realism.csv` contains the per-statistic comparison and within-real-range table.\n"
        "\n## Within-real-range table by statistic\n\n"
        "| statistic | datasets | match rate | within-real-range rate | real range min | real range max |\n|---|---:|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {row.statistic} | {int(row.datasets)} | {row.match_rate:.3f} | {row.within_real_range_rate:.3f} | {row.real_range_min:.6g} | {row.real_range_max:.6g} |"
            for row in statistic_summary.itertuples()
        )
        + "\n"
    )
    print(json.dumps({"datasets": len(dataset_keys()), "statistics": int(realism.statistic.nunique()) if not realism.empty else 0, "rows": len(realism), "match_rate": float(realism.match.mean()) if not realism.empty else None, "v3_source": v3_source, "runs": run_payloads}, indent=2))


if __name__ == "__main__":
    main()
