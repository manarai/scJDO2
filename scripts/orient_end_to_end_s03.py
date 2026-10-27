#!/usr/bin/env python3
"""Run fixed S03 two-output orientation scoring from S02 and user-run inputs.

All observed-data orientation calls precede use of the simulator-only truth
arrays.  The script never refits the S02 energy or measured-drift models.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scjdo2.orient import OrientConfig, orient
from scjdo2.orient.base import get_operator, operator_hash

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks/synthetic_multimodal/MANIFEST.csv"
S02_DIR = ROOT / "benchmarks/synthetic_multimodal/data/s02"
USER_DIR = ROOT / "benchmarks/synthetic_multimodal/data/user"
RESULT_DIR = ROOT / "benchmarks/synthetic_multimodal/s03"
USER_MANIFEST = RESULT_DIR / "user_file_manifest.json"
FIG_DIR = ROOT / "benchmarks/figures/orient/s03"
PRESETS = ("default", "stress", "batch")
METHODS = ("pca", "fa")
VARIANTS = (("raw", "velocity_labelling_raw"), ("smoothed", "velocity_labelling"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def manifest_rows() -> list[dict[str, str]]:
    with MANIFEST.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["preset"] in PRESETS and int(row["seed"]) in (0, 1, 2)]
    expected = {(preset, seed) for preset in PRESETS for seed in range(3)}
    actual = {(row["preset"], int(row["seed"])) for row in selected}
    if actual != expected:
        raise RuntimeError(f"S03 expected exactly nine S01 inputs; mismatch={actual ^ expected}")
    for row in selected:
        source = ROOT / row["path"]
        if not source.is_file() or sha256(source) != row["sha256"]:
            raise RuntimeError(f"S01 checksum mismatch: {source}")
    return sorted(selected, key=lambda r: (PRESETS.index(r["preset"]), int(r["seed"])))


def stationary_rows() -> list[dict[str, str]]:
    with MANIFEST.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["preset"] == "stationary_cycle" and int(row["seed"]) in (0, 1, 2)]
    if len(selected) != 3:
        raise RuntimeError(f"expected three stationary-cycle S02 rows, found {len(selected)}")
    return sorted(selected, key=lambda r: int(r["seed"]))


def user_records() -> dict[str, dict[str, Any]]:
    payload = json.loads(USER_MANIFEST.read_text())
    records = {str(row["unit"]): row for row in payload.get("files", [])}
    expected = {f"{preset}_s{seed}" for preset in PRESETS for seed in range(3)}
    if set(records) != expected:
        raise RuntimeError(f"user-file manifest is incomplete: missing={sorted(expected - set(records))}")
    return records


def assert_user_file(source: ad.AnnData, user: ad.AnnData, record: dict[str, Any]) -> None:
    if not source.obs_names.equals(user.obs_names):
        raise RuntimeError("user file changed S01 cell order")
    for key in ("X_pca", "X_fa"):
        if not np.array_equal(np.asarray(source.obsm[key]), np.asarray(user.obsm[key]), equal_nan=True):
            raise RuntimeError(f"user file changed {key}")
    expected_obsm = {"velocity_pca", "velocity_fa"}
    actual_obsm = set(user.obsm) - set(source.obsm)
    if actual_obsm != expected_obsm:
        raise RuntimeError(f"unexpected user-file obsm keys: {sorted(actual_obsm)}")
    if set(user.obs.columns) - set(source.obs.columns) != {"palantir_pseudotime"}:
        raise RuntimeError("unexpected user-file obs keys")
    if set(user.uns) - set(source.uns) != {"palantir_start_cell"}:
        raise RuntimeError("unexpected user-file uns keys")
    if record.get("keys_added", {}).get("obsm") != sorted(expected_obsm):
        raise RuntimeError("user manifest does not record the required dynamical velocities")
    root = str(user.uns["palantir_start_cell"])
    if root not in user.obs_names:
        raise RuntimeError("recorded Palantir root is not a cell")
    pseudo = user.obs["palantir_pseudotime"].to_numpy(dtype=float)
    if not np.isfinite(pseudo).all():
        raise RuntimeError("Palantir pseudotime contains non-finite values")
    for key in expected_obsm:
        velocity = np.asarray(user.obsm[key], dtype=float)
        if velocity.shape != np.asarray(source.obsm[key.replace("velocity_", "X_")]).shape:
            raise RuntimeError(f"unexpected shape for user {key}")
        if not np.isfinite(velocity).all():
            raise RuntimeError(f"non-finite values in user {key}")


def load_joined(row: dict[str, str], method: str, records: dict[str, dict[str, Any]]) -> tuple[ad.AnnData, dict[str, Any]]:
    preset, seed = row["preset"], int(row["seed"])
    unit = f"{preset}_s{seed}"
    record = records[unit]
    s02_path = S02_DIR / f"{unit}_{method}.h5ad"
    metrics_path = S02_DIR / f"{unit}_{method}.metrics.json"
    if not s02_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"missing S02 unit: {s02_path}")
    metrics = json.loads(metrics_path.read_text())
    if metrics.get("input_sha256") != row["sha256"] or metrics.get("method") != method:
        raise RuntimeError(f"S02 provenance mismatch for {unit} {method}")
    if sha256(s02_path) != metrics.get("result_h5ad_sha256"):
        raise RuntimeError(f"S02 fitted file checksum mismatch for {s02_path}")
    source = ad.read_h5ad(ROOT / row["path"])
    user_path = ROOT / record["path"]
    if record.get("input_sha256") != row["sha256"] or sha256(user_path) != record.get("sha256"):
        raise RuntimeError(f"user-file checksum mismatch for {unit}")
    user = ad.read_h5ad(user_path)
    assert_user_file(source, user, record)
    data = ad.read_h5ad(s02_path)
    if not data.obs_names.equals(user.obs_names):
        raise RuntimeError(f"S02/user cell mismatch for {unit} {method}")
    data.obsm[f"velocity_{method}"] = np.asarray(user.obsm[f"velocity_{method}"], dtype=np.float32)
    data.obs["palantir_pseudotime"] = user.obs["palantir_pseudotime"].to_numpy(dtype=float)
    data.uns["palantir_start_cell"] = str(user.uns["palantir_start_cell"])
    return data, record


def expected_flow_truth(data: ad.AnnData, method: str) -> np.ndarray:
    """Pure expected-count flow truth used only after observed-data calls."""
    meta = data.uns[f"embedding_{method}"]
    ds_dt = np.asarray(data.layers["ds_dt_true"], dtype=np.float64)
    capture = data.obs["capture_factor"].to_numpy(dtype=np.float64)
    batch_index = data.obs["batch_day"].to_numpy(dtype=int) - 1
    batch_effect = np.asarray(data.uns["batch_gene_effects"], dtype=np.float64)
    gene_factor = capture[:, None] * batch_effect[batch_index]
    velocity = ds_dt * gene_factor
    expected = np.asarray(data.layers["s_true"], dtype=np.float64) * gene_factor
    library = expected.sum(axis=1)
    target = float(meta["normalization_target"])
    scaled = expected * target / np.maximum(library[:, None], 1e-12)
    d_library = velocity.sum(axis=1)
    d_scaled = target * (velocity / np.maximum(library[:, None], 1e-12) - expected * d_library[:, None] / np.maximum(library[:, None] ** 2, 1e-12))
    return (d_scaled / (1.0 + scaled) @ np.asarray(meta["linear_map"], dtype=np.float64)).astype(np.float32)


def chord_truth(data: ad.AnnData, method: str) -> np.ndarray:
    """Construct the locked expected-count developmental chord truth for scoring."""
    meta = data.uns[f"embedding_{method}"]
    s_true = np.asarray(data.layers["s_true"], dtype=np.float64)
    capture = data.obs["capture_factor"].to_numpy(dtype=np.float64)
    batch_index = data.obs["batch_day"].to_numpy(dtype=int) - 1
    batch_effect = np.asarray(data.uns["batch_gene_effects"], dtype=np.float64)
    expected = s_true * capture[:, None] * batch_effect[batch_index]
    target = float(meta["normalization_target"])
    transformed = np.log1p(expected * target / np.maximum(expected.sum(axis=1, keepdims=True), 1e-12))
    z_expected = (transformed - np.asarray(meta["mean"], dtype=np.float64)) @ np.asarray(meta["linear_map"], dtype=np.float64)
    clones = data.obs["clone_true"].astype(str).to_numpy()
    days = data.obs["day"].to_numpy(dtype=int)
    truth = np.full_like(z_expected, np.nan, dtype=np.float64)
    for clone in np.unique(clones):
        idx = np.flatnonzero(clones == clone)
        clone_days = np.unique(days[idx])
        if len(clone_days) < 2:
            continue
        early = idx[days[idx] == clone_days.min()]
        late = idx[days[idx] == clone_days.max()]
        truth[idx] = z_expected[late].mean(axis=0) - z_expected[early].mean(axis=0)
    return truth.astype(np.float32)


def truth_masks(axes: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(truth).all(axis=1)
    safe = np.where(finite[:, None], truth, 0.0)
    projection = np.einsum("nd,ndk->nk", safe, axes)
    norm = np.linalg.norm(safe, axis=1)
    sign = np.sign(projection).astype(np.int8)
    moving = finite[:, None] & (norm[:, None] > 0) & (np.abs(projection) / np.maximum(norm[:, None], 1e-12) >= 0.2)
    return sign, moving, finite


def rate(sign: np.ndarray, truth_sign: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(sign[mask] == truth_sign[mask])) if np.any(mask) else float("nan")


def score_group(outcome: dict[str, Any], group: str, truth: np.ndarray, days: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    merged = outcome[group]
    sign = np.asarray(merged["sign"], dtype=np.int8)
    supported = np.asarray(merged["source"], dtype=int) > 0
    truth_sign, moving, _ = truth_masks(np.asarray(outcome["axes"].U), truth)
    core = supported
    moving_core = core & moving
    row = {
        "n_supported_pairs": int(core.sum()),
        "n_supported_moving_pairs": int(moving_core.sum()),
        "sign_agreement": rate(sign, truth_sign, core),
        "sign_agreement_moving": rate(sign, truth_sign, moving_core),
        "oriented_fraction": float(np.mean(core.any(axis=1))),
        "hash_unchanged": bool(outcome["meta"]["hash_unchanged"]),
        "flow_vs_dev_disagreement_rate": float(outcome["meta"]["flow_vs_dev_disagreement_rate"]),
    }
    by_day: list[dict[str, Any]] = []
    for day in sorted(np.unique(days)):
        selected = days == day
        day_core = core & selected[:, None]
        day_moving = moving_core & selected[:, None]
        by_day.append({
            "day": int(day),
            "n_cells": int(selected.sum()),
            "n_supported_pairs": int(day_core.sum()),
            "n_supported_moving_pairs": int(day_moving.sum()),
            "sign_agreement": rate(sign, truth_sign, day_core),
            "sign_agreement_moving": rate(sign, truth_sign, day_moving),
            "oriented_fraction": float(np.mean(core[selected].any(axis=1))),
        })
    return row, by_day


def source_addition(outcome: dict[str, Any], single: dict[str, Any], group: str, high: str, low: str, truth: np.ndarray) -> dict[str, Any]:
    merged = outcome[group]
    source_code = {name: int(code) for code, name in merged["source_codes"].items() if name != "none"}
    high_result = outcome["results"][high]
    low_result = outcome["results"][low]
    sign = np.asarray(merged["sign"], dtype=np.int8)
    source = np.asarray(merged["source"], dtype=int)
    truth_sign, moving, _ = truth_masks(np.asarray(outcome["axes"].U), truth)
    single_source = np.asarray(single[group]["source"], dtype=int)
    high_pair = source == source_code[high]
    lower_pair = source == source_code[low]
    high_cells = single_source.any(axis=1)
    combo_cells = source.any(axis=1)
    added_cells = combo_cells & ~high_cells
    added_pairs = (source > 0) & added_cells[:, None] & moving
    high_decides = high_pair & moving
    conflict = np.asarray(high_result.support, dtype=bool) & np.asarray(low_result.support, dtype=bool) & (np.asarray(high_result.sign) != np.asarray(low_result.sign)) & moving
    high_unchanged = np.array_equal(sign[high_pair], np.asarray(high_result.sign)[high_pair])
    return {
        "coverage_added_fraction": float(np.mean(added_cells)),
        "n_coverage_added_cells": int(added_cells.sum()),
        "coverage_added_agreement": rate(sign, truth_sign, added_pairs),
        "higher_decides_agreement": rate(sign, truth_sign, high_decides),
        "n_higher_decides_pairs": int(high_decides.sum()),
        "combined_oriented_fraction": float(np.mean(combo_cells)),
        "higher_only_oriented_fraction": float(np.mean(high_cells)),
        "coverage_non_decreasing": bool(np.mean(combo_cells) >= np.mean(high_cells)),
        "higher_decision_unchanged_exact": bool(high_unchanged),
        "n_conflict_pairs": int(conflict.sum()),
        "higher_truth_sided_fraction": rate(np.asarray(high_result.sign), truth_sign, conflict),
        "lower_truth_sided_fraction": rate(np.asarray(low_result.sign), truth_sign, conflict),
        "lower_source_code": int(source_code[low]),
        "n_lower_decides_pairs": int((lower_pair & moving).sum()),
    }


def group_equal(left: dict[str, Any], right: dict[str, Any], group: str) -> bool:
    keys = ("sign", "source", "conflict")
    for key in keys:
        if not np.array_equal(np.asarray(left[group][key]), np.asarray(right[group][key])):
            return False
    if group == "flow":
        return np.array_equal(np.asarray(left[group]["magnitude"]), np.asarray(right[group]["magnitude"]), equal_nan=True)
    return True


def call_orient(data: ad.AnnData, sources: Iterable[str], method: str, config: OrientConfig, audit: list[dict[str, Any]]) -> dict[str, Any]:
    before = operator_hash(get_operator(data))
    outcome = orient(data, list(sources), f"X_{method}", config)
    after = operator_hash(get_operator(data))
    if before != after or not outcome["meta"]["hash_unchanged"]:
        raise RuntimeError("orient changed the frozen S02 energy operator")
    audit.append({"sources": list(sources), "method": method, "hash_unchanged": True})
    return outcome


def base_fields(preset: str, seed: int, method: str, variant: str, group: str, combination: str) -> dict[str, Any]:
    return {"preset": preset, "seed": seed, "method": method, "labelling_variant": variant, "group": group, "combination": combination}


def analyse_unit(row: dict[str, str], method: str, records: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    preset, seed = row["preset"], int(row["seed"])
    data, record = load_joined(row, method, records)
    days = data.obs["day"].to_numpy(dtype=int)
    audit: list[dict[str, Any]] = []
    calls = {
        "flow_labelling": (["labelling"], "flow", "labelling"),
        "flow_velocity": (["velocity"], "flow", "velocity"),
        "flow_combined": (["labelling", "velocity"], "flow", "labelling+velocity"),
        "dev_lineage": (["lineage"], "dev", "lineage"),
        "dev_pseudotime": (["pseudotime"], "dev", "pseudotime"),
        "dev_combined": (["lineage", "pseudotime"], "dev", "lineage+pseudotime"),
        "all_four": (["labelling", "velocity", "lineage", "pseudotime"], "both", "all_four"),
    }
    # All observed-data orientation calls complete before truth is constructed.
    variant_outcomes: dict[str, dict[str, dict[str, Any]]] = {}
    for variant, labelling_key in VARIANTS:
        config = OrientConfig(labelling_key=labelling_key)
        variant_outcomes[variant] = {
            name: call_orient(data, sources, method, config, audit)
            for name, (sources, _, _) in calls.items()
        }
    # Simulator truth is available only in this post-fit scoring phase.
    flow_truth = expected_flow_truth(data, method)
    dev_truth = chord_truth(data, method)
    rows: list[dict[str, Any]] = []
    day_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for variant, _ in VARIANTS:
        outcomes = variant_outcomes[variant]
        for name, (_, requested_group, combination) in calls.items():
            outcome = outcomes[name]
            groups = ("flow", "dev") if requested_group == "both" else (requested_group,)
            for group in groups:
                truth = flow_truth if group == "flow" else dev_truth
                scored, per_day = score_group(outcome, group, truth, days)
                common = base_fields(preset, seed, method, variant, group, combination)
                rows.append({"row_type": "combination", **common, **scored})
                day_rows.extend({"row_type": "combination", **common, **entry} for entry in per_day)
        for group, combo_name, single_name, high, low, truth in (
            ("flow", "flow_combined", "flow_labelling", "labelling", "velocity", flow_truth),
            ("dev", "dev_combined", "dev_lineage", "lineage", "pseudotime", dev_truth),
        ):
            detail = source_addition(outcomes[combo_name], outcomes[single_name], group, high, low, truth)
            diagnostic_rows.append({"row_type": "source_addition", **base_fields(preset, seed, method, variant, group, high + "+" + low), **detail})
        flow_identical = group_equal(outcomes["flow_combined"], outcomes["all_four"], "flow")
        dev_identical = group_equal(outcomes["dev_combined"], outcomes["all_four"], "dev")
        diagnostic_rows.append({
            "row_type": "independence", **base_fields(preset, seed, method, variant, "both", "all_four"),
            "flow_identical_without_vs_with_dev": flow_identical,
            "dev_identical_without_vs_with_flow": dev_identical,
            "independence_holds": bool(flow_identical and dev_identical),
            "hash_unchanged_all_calls": bool(all(entry["hash_unchanged"] for entry in audit)),
            "n_orient_calls": len(audit),
            "palantir_start_cell": str(record["palantir_start_cell"]),
            "palantir_start_pseudotime": float(record["palantir_start_pseudotime"]),
            "stochastic_status": str(record["stochastic"]["status"]),
        })
    return rows, day_rows, diagnostic_rows

def analyse_stationary(row: dict[str, str], method: str) -> list[dict[str, Any]]:
    preset, seed = row["preset"], int(row["seed"])
    path = S02_DIR / f"{preset}_s{seed}_{method}.h5ad"
    metrics = json.loads(path.with_suffix(".metrics.json").read_text())
    if sha256(path) != metrics.get("result_h5ad_sha256"):
        raise RuntimeError(f"S02 checksum mismatch for flags-only {path}")
    data = ad.read_h5ad(path)
    output: list[dict[str, Any]] = []
    for variant, labelling_key in VARIANTS:
        outcome = call_orient(data, ["labelling"], method, OrientConfig(labelling_key=labelling_key), [])
        equilibrium = np.asarray(outcome["equilibrium_flag"], dtype=str)
        stationarity = np.asarray(outcome["stationarity_flag"], dtype=str)
        output.append({
            "row_type": "flags_only", **base_fields(preset, seed, method, variant, "flow", "labelling"),
            "fraction_equilibrium": float(np.mean(equilibrium == "equilibrium")),
            "fraction_non_equilibrium": float(np.mean(equilibrium == "non_equilibrium")),
            "fraction_not_identified": float(np.mean(equilibrium == "not_identified")),
            "fraction_stationary": float(np.mean(stationarity == "stationary")),
            "fraction_non_stationary": float(np.mean(stationarity == "non_stationary")),
            "hash_unchanged": bool(outcome["meta"]["hash_unchanged"]),
        })
    return output


def aggregate(results: pd.DataFrame) -> pd.DataFrame:
    combination = results[results["row_type"] == "combination"].copy()
    grouped = combination.groupby(["preset", "method", "labelling_variant", "group", "combination"], dropna=False)
    output = grouped.size().rename("n_seeds").reset_index()
    for column in ("sign_agreement", "sign_agreement_moving", "oriented_fraction", "flow_vs_dev_disagreement_rate"):
        stats = grouped[column].agg(["mean", "min"]).reset_index().rename(columns={"mean": f"{column}_mean", "min": f"{column}_min"})
        output = output.merge(stats, on=["preset", "method", "labelling_variant", "group", "combination"], how="left")
    return output.sort_values(["preset", "method", "labelling_variant", "group", "combination"]).reset_index(drop=True)


def values_for(frame: pd.DataFrame, **where: Any) -> pd.Series:
    selected = frame
    for key, value in where.items():
        selected = selected[selected[key] == value]
    return selected


def gate_table(results: pd.DataFrame, diagnostics: pd.DataFrame) -> list[dict[str, Any]]:
    primary = results[(results["row_type"] == "combination") & (results["preset"] == "default") & (results["method"] == "pca") & (results["labelling_variant"] == "raw")]
    detail = diagnostics[(diagnostics["preset"] == "default") & (diagnostics["method"] == "pca") & (diagnostics["labelling_variant"] == "raw")]
    gates: list[dict[str, Any]] = []
    def threshold_gate(name: str, values: pd.Series, threshold: float) -> None:
        numeric = values.to_numpy(dtype=float)
        gates.append({"gate": name, "minimum": threshold, "per_seed": ", ".join(f"{x:.4f}" for x in numeric), "minimum_observed": float(np.nanmin(numeric)) if len(numeric) else float("nan"), "pass": bool(len(numeric) == 3 and np.all(numeric >= threshold))})
    velocity = values_for(primary, group="flow", combination="velocity")
    threshold_gate("Flow velocity agreement (dynamical; moving axes)", velocity["sign_agreement_moving"], 0.75)
    threshold_gate("Flow velocity oriented fraction", velocity["oriented_fraction"], 0.30)
    flow_add = values_for(detail, row_type="source_addition", group="flow")
    gates.append({"gate": "Flow coverage non-decreasing and labelling decision unchanged", "minimum": 1.0, "per_seed": ", ".join(str(bool(a and b)) for a, b in zip(flow_add["coverage_non_decreasing"], flow_add["higher_decision_unchanged_exact"])), "minimum_observed": float(np.mean(flow_add["coverage_non_decreasing"] & flow_add["higher_decision_unchanged_exact"])) if len(flow_add) else float("nan"), "pass": bool(len(flow_add) == 3 and (flow_add["coverage_non_decreasing"] & flow_add["higher_decision_unchanged_exact"]).all())})
    threshold_gate("Flow conflict truth sides with labelling (moving axes)", flow_add["higher_truth_sided_fraction"], 0.60)
    lineage = values_for(primary, group="dev", combination="lineage")
    threshold_gate("Developmental lineage agreement (moving axes)", lineage["sign_agreement_moving"], 0.95)
    dev_combo = values_for(primary, group="dev", combination="lineage+pseudotime")
    threshold_gate("Developmental lineage+pseudotime oriented fraction", dev_combo["oriented_fraction"], 0.90)
    dev_add = values_for(detail, row_type="source_addition", group="dev")
    threshold_gate("Developmental conflict truth sides with lineage (moving axes)", dev_add["higher_truth_sided_fraction"], 0.80)
    independence = values_for(detail, row_type="independence")
    hash_ok = bool(results.get("hash_unchanged", pd.Series(dtype=bool)).dropna().all()) and bool(diagnostics.get("hash_unchanged_all_calls", pd.Series(dtype=bool)).dropna().all())
    gates.append({"gate": "Flow/dev independence exact in every run", "minimum": 1.0, "per_seed": str(bool(independence["independence_holds"].all())), "minimum_observed": float(bool(independence["independence_holds"].all())), "pass": bool(len(independence) == 3 and independence["independence_holds"].all())})
    gates.append({"gate": "J hash unchanged in every call", "minimum": 1.0, "per_seed": str(hash_ok), "minimum_observed": float(hash_ok), "pass": hash_ok})
    return gates


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    values = frame.loc[:, columns].copy().replace({np.nan: "—"})
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    lines.extend("| " + " | ".join(str(row[column]) for column in columns) + " |" for _, row in values.iterrows())
    return "\n".join(lines)


def write_result_markdown(results: pd.DataFrame, summary: pd.DataFrame, diagnostics: pd.DataFrame, gates: list[dict[str, Any]], user_payload: dict[str, Any]) -> None:
    gate_frame = pd.DataFrame(gates)
    primary = summary[(summary["method"] == "pca") & (summary["labelling_variant"] == "raw")]
    additions = diagnostics[(diagnostics["row_type"] == "source_addition") & (diagnostics["method"] == "pca") & (diagnostics["labelling_variant"] == "raw")]
    text = [
        "# S03 two-output end-to-end results", "",
        "## What ran",
        "- Joined each validated S02 fitted unit to its matched S03 user file using only dynamical `velocity_<embedding>`, `palantir_pseudotime`, and `palantir_start_cell`; S02 models and `J` were not refit.",
        "- Ran the fixed two-output layer: flow uses labelling > scVelo velocity, developmental uses lineage > Palantir pseudotime. Raw labelling is primary; smoothed labelling is the sensitivity.",
        "- The supplied Palantir root is the generator-selected earliest-day progenitor-nearest cell (the task's declared best-case exception); it is recorded, not used as a truth score.",
        "- Exact simulator truth was read only after observed-data orientation calls, for scoring instantaneous flow and expected-count clone chord direction.", "",
        "## Gates (verbatim)",
        "- Flow: velocity (dynamical) sign agreement ≥ 0.75 and oriented fraction ≥ 0.3; combined coverage is non-decreasing and labelling decisions are exact; conflicts side with labelling ≥ 0.60.",
        "- Developmental: lineage agreement ≥ 0.95; lineage+pseudotime oriented fraction ≥ 0.90; conflicts side with lineage ≥ 0.80.",
        "- Flow/dev independence and J hashes hold exactly in every run. Stochastic scVelo is report-only.", "",
        "## Gate results", markdown_table(gate_frame, ["gate", "minimum", "per_seed", "minimum_observed", "pass"]), "",
        "## Primary PCA/raw summary (mean and minimum over seeds)", markdown_table(primary.round(4), ["preset", "group", "combination", "n_seeds", "sign_agreement_moving_mean", "sign_agreement_moving_min", "oriented_fraction_mean", "oriented_fraction_min"]), "",
        "## Source-addition and conflict summary (PCA/raw)", markdown_table(additions.round(4), ["preset", "group", "coverage_added_fraction", "coverage_added_agreement", "higher_decides_agreement", "higher_truth_sided_fraction", "coverage_non_decreasing", "higher_decision_unchanged_exact"]), "",
        "## Complete outputs",
        "- `results.csv`: per-unit combination scores.",
        "- `summary.csv`: per-preset, embedding, variant, group, and combination aggregate scores.",
        "- `by_day.csv`: the corresponding day-stratified scores.",
        "- `RUN_METADATA.json`: user-file manifest, call audit, and gate records.",
        "- Figures: `benchmarks/figures/orient/s03/`.",
        f"- S03 user manifest recorded {len(user_payload.get('files', []))} primary user files; stochastic status is report-only per Revision 2.",
    ]
    (RESULT_DIR / "RESULT_S03.md").write_text("\n".join(text) + "\n")


def render_figures(results: pd.DataFrame, by_day: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    primary = results[(results["row_type"] == "combination") & (results["method"] == "pca") & (results["labelling_variant"] == "raw")]
    combos = ["labelling", "velocity", "labelling+velocity", "lineage", "pseudotime", "lineage+pseudotime"]
    fig, ax = plt.subplots(figsize=(12, 5.8))
    x = np.arange(len(PRESETS)); width = 0.12
    for i, combo in enumerate(combos):
        subset = primary[primary["combination"] == combo]
        values = [subset[subset["preset"] == preset]["sign_agreement_moving"].mean() for preset in PRESETS]
        ax.bar(x + (i - 2.5) * width, values, width, label=combo)
    ax.set_xticks(x, PRESETS); ax.set_ylim(0, 1.05); ax.set_ylabel("agreement on moving truth axes")
    ax.set_title("S03 primary agreement by source combination and preset (PCA, raw labelling)")
    ax.legend(frameon=False, ncol=3, fontsize=8); fig.tight_layout(); fig.savefig(FIG_DIR / "agreement_by_combination.png", dpi=180); plt.close(fig)
    additions = diagnostics[(diagnostics["row_type"] == "source_addition") & (diagnostics["method"] == "pca") & (diagnostics["labelling_variant"] == "raw")]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(PRESETS)); width = 0.34
    for i, group in enumerate(("flow", "dev")):
        subset = additions[additions["group"] == group]
        vals = [subset[subset["preset"] == preset]["coverage_added_fraction"].mean() for preset in PRESETS]
        ax.bar(x + (i - 0.5) * width, vals, width, label=group)
    ax.set_xticks(x, PRESETS); ax.set_ylim(0, 1.05); ax.set_ylabel("fraction of cells newly oriented by lower-ranked source")
    ax.set_title("S03 coverage added by lower-ranked sources (PCA, raw labelling)")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG_DIR / "coverage_added.png", dpi=180); plt.close(fig)
    day = by_day[(by_day["method"] == "pca") & (by_day["labelling_variant"] == "raw") & (by_day["combination"] == "all_four")]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for preset in PRESETS:
        for group, style in (("flow", "-"), ("dev", "--")):
            subset = day[(day["preset"] == preset) & (day["group"] == group)].groupby("day")["sign_agreement_moving"].mean()
            ax.plot(subset.index, subset.values, marker="o", linestyle=style, label=f"{preset} {group}")
    ax.set_ylim(0, 1.05); ax.set_xlabel("sampling day"); ax.set_ylabel("agreement on moving truth axes")
    ax.set_title("S03 all-source agreement by sampling day (PCA, raw labelling)")
    ax.legend(frameon=False, ncol=3, fontsize=8); fig.tight_layout(); fig.savefig(FIG_DIR / "agreement_by_day.png", dpi=180); plt.close(fig)
    ranking = diagnostics[(diagnostics["row_type"] == "source_addition") & (diagnostics["method"] == "pca") & (diagnostics["labelling_variant"] == "raw")]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(PRESETS)); width = 0.34
    for i, group in enumerate(("flow", "dev")):
        subset = ranking[ranking["group"] == group]
        vals = [subset[subset["preset"] == preset]["higher_truth_sided_fraction"].mean() for preset in PRESETS]
        ax.bar(x + (i - 0.5) * width, vals, width, label=f"{group}: truth sides with higher rank")
    ax.axhline(0.60, color="tab:blue", linestyle=":", linewidth=1, label="flow gate")
    ax.axhline(0.80, color="tab:orange", linestyle=":", linewidth=1, label="developmental gate")
    ax.set_xticks(x, PRESETS); ax.set_ylim(0, 1.05); ax.set_ylabel("fraction of conflicts")
    ax.set_title("S03 conflict ranking check (PCA, raw labelling)")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(FIG_DIR / "conflict_ranking.png", dpi=180); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, action="append", help="limit analysis after all user files are validated")
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), action="append", help="limit analysis after all user files are validated")
    args = parser.parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = manifest_rows()
    records = user_records()
    if args.preset:
        rows = [row for row in rows if row["preset"] in set(args.preset)]
    if args.seed:
        rows = [row for row in rows if int(row["seed"]) in set(args.seed)]
    if not rows:
        parser.error("selection contains no rows")
    result_rows: list[dict[str, Any]] = []
    day_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for row in rows:
        for method in METHODS:
            values, per_day, extra = analyse_unit(row, method, records)
            result_rows.extend(values); day_rows.extend(per_day); diagnostics.extend(extra)
            print(f"[S03] complete {row['preset']}_s{row['seed']} {method}", flush=True)
    if not args.preset and not args.seed:
        for row in stationary_rows():
            for method in METHODS:
                diagnostics.extend(analyse_stationary(row, method))
    results = pd.DataFrame(result_rows).sort_values(["preset", "seed", "method", "labelling_variant", "group", "combination"]).reset_index(drop=True)
    by_day = pd.DataFrame(day_rows).sort_values(["preset", "seed", "method", "labelling_variant", "group", "combination", "day"]).reset_index(drop=True)
    detail = pd.DataFrame(diagnostics).sort_values(["preset", "seed", "method", "labelling_variant", "row_type", "group"]).reset_index(drop=True)
    results.to_csv(RESULT_DIR / "results.csv", index=False)
    by_day.to_csv(RESULT_DIR / "by_day.csv", index=False)
    detail.to_csv(RESULT_DIR / "diagnostics.csv", index=False)
    summary = aggregate(results)
    summary.to_csv(RESULT_DIR / "summary.csv", index=False)
    gates = gate_table(results, detail)
    payload = json.loads(USER_MANIFEST.read_text())
    write_result_markdown(results, summary, detail, gates, payload)
    render_figures(results, by_day, detail)
    write_json(RESULT_DIR / "RUN_METADATA.json", {"user_manifest": payload, "gates": gates, "n_results": len(results), "n_by_day": len(by_day), "n_diagnostics": len(detail)})
    print(json.dumps({"n_results": len(results), "gates": gates}, indent=2), flush=True)


if __name__ == "__main__":
    main()
