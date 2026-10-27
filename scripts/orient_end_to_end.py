#!/usr/bin/env python3
"""Run the frozen S02 count-to-orientation analysis on S01 synthetic data.

The fitted path consumes only observed count layers, clone labels, sampling
metadata, and stored PCA/FA embeddings.  Exact simulator keys are accessed only
inside post-fit scoring functions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from scjdo2.data.synthetic_multimodal import true_velocity_in_embedding
from scjdo2.energy_drift import EnergyDriftAdapter, EnergyDriftConfig
from scjdo2.labelled_rates import LabelledRateModel
from scjdo2.measured_drift import MeasuredDrift
from scjdo2.orient import OrientConfig, orient
from scjdo2.orient.base import (
    antisymmetric,
    get_operator,
    local_jacobian,
    neighbours,
    operator_hash,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks/synthetic_multimodal/MANIFEST.csv"
RESULT_ROOT = ROOT / "benchmarks/synthetic_multimodal/s02"
DATA_ROOT = ROOT / "benchmarks/synthetic_multimodal/data/s02"
FIG_ROOT = ROOT / "benchmarks/figures/orient/s02"
PRESETS = ("default", "stress", "batch", "stationary_cycle")
METHODS = ("pca", "fa")
SOURCE_VARIANTS = (
    (("lineage",), "none", None),
    (("labelling",), "smoothed", "velocity_labelling"),
    (("lineage", "labelling"), "smoothed", "velocity_labelling"),
    (("labelling",), "raw", "velocity_labelling_raw"),
    (("lineage", "labelling"), "raw", "velocity_labelling_raw"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def input_rows() -> list[dict[str, str]]:
    with MANIFEST.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {(preset, str(seed)) for preset in PRESETS for seed in range(3)}
    present = {(row["preset"], row["seed"]) for row in rows}
    if present != expected:
        raise RuntimeError(f"manifest preset/seed set differs: {present ^ expected}")
    for row in rows:
        path = ROOT / row["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != row["sha256"]:
            raise RuntimeError(
                f"input checksum mismatch for {path}: {actual} != {row['sha256']}"
            )
    return sorted(rows, key=lambda r: (PRESETS.index(r["preset"]), int(r["seed"])))


def chain_rule_count_velocity(
    adata: ad.AnnData, method: str, gene_velocity: np.ndarray
) -> np.ndarray:
    """Map observed gene-count velocity through the fixed embedding transform.

    This deliberately mirrors the public observed-count chain rule, but accepts
    only S02's measured velocity.  It never reads any ``*_true`` array.
    """
    spliced = np.asarray(adata.layers["spliced"], dtype=np.float64)
    velocity = np.asarray(gene_velocity, dtype=np.float64)
    if spliced.shape != velocity.shape:
        raise ValueError(f"gene velocity {velocity.shape} != spliced {spliced.shape}")
    meta = adata.uns[f"embedding_{method}"]
    library = spliced.sum(axis=1)
    target = float(meta["normalization_target"])
    denominator = np.maximum(library[:, None], 1e-12)
    scaled = spliced * target / denominator
    d_library = velocity.sum(axis=1)
    d_scaled = target * (
        velocity / denominator
        - spliced * d_library[:, None] / np.maximum(denominator**2, 1e-12)
    )
    d_log = d_scaled / (1.0 + scaled)
    linear_map = np.asarray(meta["linear_map"], dtype=np.float64)
    return (d_log @ linear_map).astype(np.float32)


def labelled_velocity(adata: ad.AnnData, z: np.ndarray) -> tuple[np.ndarray, dict[str, Any], LabelledRateModel]:
    """Fit the specified labelled-rate model and derive its raw gene velocity."""
    new = np.asarray(adata.layers["new"], dtype=np.float32)
    total = np.asarray(adata.layers["total"], dtype=np.float32)
    pulse = adata.obs["pulse_hours"].to_numpy(dtype=np.float32)
    if np.any(pulse <= 0):
        raise ValueError("pulse_hours must be positive")
    fitted = LabelledRateModel(seed=0).fit(new, total, z, pulse)
    gamma = np.asarray(fitted.gamma, dtype=np.float64)
    frac = -np.expm1(-pulse[:, None] * gamma[None, :])
    alpha = gamma[None, :] * new / np.maximum(frac, 1e-12)
    velocity = alpha - gamma[None, :] * total
    details = {
        "gamma_min": float(gamma.min()),
        "gamma_max": float(gamma.max()),
        "held_out_nll": float(fitted.best_val_nll),
        "epochs_run": int(len(getattr(fitted, "_gamma_grad_history", ()))),
        "seed": 0,
    }
    return velocity.astype(np.float32), details, fitted


def rotation_summary(z: np.ndarray, velocity: np.ndarray, days: np.ndarray, cfg: OrientConfig) -> dict[str, Any]:
    """Quantify the local antisymmetric share of a measured latent field."""
    idx = neighbours(z, cfg.k_neighbours)
    j_velocity = local_jacobian(z, velocity, idx, cfg.ridge)
    omega = antisymmetric(j_velocity)
    numerator = np.linalg.norm(omega, axis=(1, 2))
    denominator = np.linalg.norm(j_velocity, axis=(1, 2))
    share = numerator / np.maximum(denominator, 1e-12)
    finite = np.isfinite(share)
    result: dict[str, Any] = {
        "rotation_share_median": float(np.median(share[finite])) if finite.any() else float("nan"),
        "rotation_share_n": int(finite.sum()),
    }
    for day in (1, 2, 3, 4):
        selected = finite & (days == day)
        result[f"rotation_share_day_{day}"] = (
            float(np.median(share[selected])) if selected.any() else float("nan")
        )
    return result


def fraction_labels(values: np.ndarray, labels: Iterable[str]) -> dict[str, float]:
    values = np.asarray(values, dtype=str)
    return {f"fraction_{label}": float(np.mean(values == label)) for label in labels}


def score_orientation(outcome: dict[str, Any], truth_velocity: np.ndarray, source_set: str) -> dict[str, Any]:
    """Post-fit scoring only; simulator truth enters here and nowhere upstream."""
    axes = outcome["axes"].U
    truth_projection = np.einsum("nd,ndk->nk", truth_velocity, axes)
    truth_sign = np.sign(truth_projection).astype(np.int8)
    truth_norm = np.linalg.norm(truth_velocity, axis=1)
    moving_axis = np.abs(truth_projection) / np.maximum(truth_norm[:, None], 1e-12) >= 0.2
    supported = np.asarray(outcome["source"], dtype=int) > 0
    sign = np.asarray(outcome["sign"], dtype=np.int8)

    def agreement(mask: np.ndarray) -> float:
        return float(np.mean(sign[mask] == truth_sign[mask])) if np.any(mask) else float("nan")

    results: dict[str, Any] = {
        "sign_agreement": agreement(supported),
        "sign_agreement_moving": agreement(supported & moving_axis),
        "n_supported_pairs": int(supported.sum()),
        "n_supported_moving_pairs": int((supported & moving_axis).sum()),
        "oriented_fraction": float(np.mean(supported.any(axis=1))),
        "hash_unchanged": bool(outcome["meta"]["hash_unchanged"]),
        **fraction_labels(
            outcome["equilibrium_flag"],
            ("equilibrium", "non_equilibrium", "not_identified"),
        ),
        **fraction_labels(
            outcome["stationarity_flag"],
            ("stationary", "non_stationary", "not_identified"),
        ),
    }
    if source_set == "lineage+labelling":
        lineage = outcome["results"]["lineage"]
        labelling = outcome["results"]["labelling"]
        both = np.asarray(lineage.support, dtype=bool) & np.asarray(labelling.support, dtype=bool)
        conflict = np.asarray(outcome["conflict"], dtype=bool) & both
        results["conflict_rate"] = float(conflict.sum() / both.sum()) if both.any() else float("nan")
        results["n_conflict_pairs"] = int(conflict.sum())
        results["conflict_merged_agreement"] = agreement(conflict)
        if conflict.any():
            results["conflict_truth_sides_lineage"] = float(
                np.mean(np.asarray(lineage.sign)[conflict] == truth_sign[conflict])
            )
            results["conflict_truth_sides_labelling"] = float(
                np.mean(np.asarray(labelling.sign)[conflict] == truth_sign[conflict])
            )
        else:
            results["conflict_truth_sides_lineage"] = float("nan")
            results["conflict_truth_sides_labelling"] = float("nan")
    else:
        results.update(
            conflict_rate=float("nan"), n_conflict_pairs=0,
            conflict_merged_agreement=float("nan"),
            conflict_truth_sides_lineage=float("nan"),
            conflict_truth_sides_labelling=float("nan"),
        )
    return results


def save_fitted_models(
    stem: Path,
    energy: EnergyDriftAdapter,
    labelled: LabelledRateModel,
    measured: MeasuredDrift,
) -> None:
    """Persist local fitted-state checkpoints; models remain gitignored."""
    torch.save({"config": asdict(energy.config), "state_dict": energy._model.state_dict()}, stem.with_name(stem.name + "_energy.pt"))
    torch.save({"config": asdict(labelled.config), "state_dict": labelled._model.state_dict()}, stem.with_name(stem.name + "_labelled.pt"))
    torch.save({"config": asdict(measured.config), "state_dict": measured._model.state_dict()}, stem.with_name(stem.name + "_measured.pt"))


def unit_paths(preset: str, seed: int, method: str) -> tuple[Path, Path, Path]:
    stem = DATA_ROOT / f"{preset}_s{seed}_{method}"
    return stem.with_suffix(".h5ad"), stem.with_suffix(".metrics.json"), stem


def fit_unit(row: dict[str, str], method: str, force: bool) -> dict[str, Any]:
    preset, seed = row["preset"], int(row["seed"])
    result_h5ad, result_metrics, stem = unit_paths(preset, seed, method)
    if not force and result_h5ad.exists() and result_metrics.exists():
        cached = json.loads(result_metrics.read_text())
        if cached.get("input_sha256") == row["sha256"] and cached.get("method") == method:
            print(f"[S02] reusing {preset} seed={seed} method={method}", flush=True)
            return cached

    print(f"[S02] fitting {preset} seed={seed} method={method}", flush=True)
    source_path = ROOT / row["path"]
    adata = ad.read_h5ad(source_path)
    z = np.asarray(adata.obsm[f"X_{method}"], dtype=np.float32)
    if z.ndim != 2 or z.shape[0] != adata.n_obs:
        raise ValueError(f"invalid X_{method} shape {z.shape}")
    days = adata.obs["day"].to_numpy(dtype=int)
    cfg_energy = EnergyDriftConfig(
        hidden=96, depth=3, n_epochs=1500, seed=0, batch_size=256,
        loss="esm", lr=5e-4, beta=1.0, esm_laplacian="exact",
    )
    energy = EnergyDriftAdapter(z.shape[1], cfg_energy).fit(z, verbose=True)
    symmetry = energy.symmetry_check(z, seed=0)
    if symmetry["max_asym"] > 1e-4:
        raise RuntimeError(
            f"symmetry check {symmetry['max_asym']:.6g} exceeds frozen 1e-4 gate"
        )
    adata.uns["scjdo2"] = {"J": energy.jacobian_autodiff(z)}
    adata.obsm["X_drift_v"] = energy.velocity(z).astype(np.float32)
    adata.obs["energy"] = energy.energy(z).astype(np.float32)

    raw_gene, fit_details, labelled = labelled_velocity(adata, z)
    raw_latent = chain_rule_count_velocity(adata, method, raw_gene)
    adata.obsm["velocity_labelling_raw"] = raw_latent
    measured = MeasuredDrift(z.shape[1], seed=0).fit(z, raw_latent, verbose=True)
    smooth_latent = measured.velocity(z).astype(np.float32)
    adata.obsm["velocity_labelling"] = smooth_latent

    # Exact truth is requested only after all observed-data fits are complete.
    truth_velocity = true_velocity_in_embedding(adata, method, at="expected")
    common: dict[str, Any] = {
        "preset": preset,
        "seed": seed,
        "method": method,
        "n_cells": int(adata.n_obs),
        "input_path": row["path"],
        "input_sha256": row["sha256"],
        "symmetry_median": float(symmetry["median_asym"]),
        "symmetry_max": float(symmetry["max_asym"]),
        "labelled_fit_nll": fit_details["held_out_nll"],
        "labelled_fit_epochs": fit_details["epochs_run"],
        "labelled_gamma_min": fit_details["gamma_min"],
        "labelled_gamma_max": fit_details["gamma_max"],
    }
    rotations = {
        "smoothed": rotation_summary(z, smooth_latent, days, OrientConfig()),
        "raw": rotation_summary(z, raw_latent, days, OrientConfig(labelling_key="velocity_labelling_raw")),
    }
    rows: list[dict[str, Any]] = []
    final_outcome: dict[str, Any] | None = None
    for sources, variant, labelling_key in SOURCE_VARIANTS:
        config = OrientConfig() if labelling_key is None else OrientConfig(labelling_key=labelling_key)
        before = operator_hash(get_operator(adata))
        outcome = orient(adata, sources, f"X_{method}", config)
        after = operator_hash(get_operator(adata))
        if before != after or not outcome["meta"]["hash_unchanged"]:
            raise RuntimeError("orient changed the frozen energy operator")
        source_set = "+".join(sources)
        scored = score_orientation(outcome, truth_velocity, source_set)
        row_result = {
            **common,
            **scored,
            **rotations.get(variant, {}),
            "source_set": source_set,
            "labelling_variant": variant,
        }
        rows.append(row_result)
        if sources == ("lineage", "labelling") and variant == "smoothed":
            final_outcome = outcome

    if final_outcome is None:
        raise RuntimeError("missing primary lineage+labelling outcome")
    # Preserve the primary call's outputs in the per-unit result H5AD.
    orient(adata, ["lineage", "labelling"], f"X_{method}", OrientConfig())
    result_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(result_h5ad, compression="gzip")
    save_fitted_models(stem, energy, labelled, measured)
    payload = {
        "input_sha256": row["sha256"], "method": method, "rows": rows,
        "result_h5ad": str(result_h5ad.relative_to(ROOT)),
        "result_h5ad_sha256": sha256(result_h5ad),
    }
    write_json(result_metrics, payload)
    print(f"[S02] complete {preset} seed={seed} method={method}", flush=True)
    return payload


def write_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    rows = frame.loc[:, columns].copy()
    rows = rows.replace({np.nan: "—"})
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(str(row[c]) for c in columns) + " |" for _, row in rows.iterrows()]
    return "\n".join([header, divider, *body])


def aggregate(results: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "sign_agreement", "sign_agreement_moving", "oriented_fraction", "conflict_rate",
        "fraction_equilibrium", "fraction_non_equilibrium", "fraction_not_identified",
        "fraction_stationary", "fraction_non_stationary", "rotation_share_median",
        "symmetry_max", "labelled_fit_nll",
    ]
    grouped = results.groupby(["preset", "method", "source_set", "labelling_variant"], dropna=False)
    output = grouped.size().rename("n_rows").reset_index()
    for column in numeric:
        stats = grouped[column].agg(["mean", "min"]).reset_index()
        stats = stats.rename(columns={"mean": f"{column}_mean", "min": f"{column}_min"})
        output = output.merge(stats, on=["preset", "method", "source_set", "labelling_variant"], how="left")
    return output.sort_values(["preset", "method", "source_set", "labelling_variant"]).reset_index(drop=True)


def gate_rows(results: pd.DataFrame) -> list[dict[str, Any]]:
    primary = results[(results["preset"] == "default") & (results["method"] == "pca")]
    specs = [
        ("lineage", "none", "sign_agreement", 0.90), ("lineage", "none", "oriented_fraction", 0.30),
        ("labelling", "smoothed", "sign_agreement", 0.85), ("labelling", "smoothed", "oriented_fraction", 0.30),
        ("lineage+labelling", "smoothed", "sign_agreement", 0.90), ("lineage+labelling", "smoothed", "oriented_fraction", 0.50),
    ]
    gate: list[dict[str, Any]] = []
    for source, variant, metric, threshold in specs:
        values = primary.loc[(primary["source_set"] == source) & (primary["labelling_variant"] == variant), metric]
        gate.append({
            "gate": f"default pca smoothed {source} {metric}", "minimum": threshold,
            "per_seed": ", ".join(f"{v:.4f}" for v in values),
            "minimum_observed": float(values.min()) if len(values) else float("nan"),
            "pass": bool(len(values) == 3 and np.all(values >= threshold)),
        })
    stationary = results[(results["preset"] == "stationary_cycle") & (results["method"] == "pca") & (results["source_set"] == "lineage+labelling") & (results["labelling_variant"] == "smoothed")]
    values = stationary["fraction_non_equilibrium"] / np.maximum(
        stationary["fraction_equilibrium"] + stationary["fraction_non_equilibrium"], 1e-12
    )
    gate.append({
        "gate": "stationary_cycle pca smoothed combined non_equilibrium among identified",
        "minimum": 0.80,
        "per_seed": ", ".join(f"{v:.4f}" for v in values),
        "minimum_observed": float(values.min()) if len(values) else float("nan"),
        "pass": bool(len(values) == 3 and np.all(values >= 0.80)),
    })
    hash_ok = bool(results["hash_unchanged"].all())
    gate.append({"gate": "J hash unchanged in every orient call", "minimum": 1.0, "per_seed": str(hash_ok), "minimum_observed": float(hash_ok), "pass": hash_ok})
    return gate


def render_figures(results: pd.DataFrame) -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    plot = results[(results["method"] == "pca") & (results["labelling_variant"].isin(["none", "smoothed"]))]
    source_order = ["lineage", "labelling", "lineage+labelling"]
    color = {"lineage": "#2563eb", "labelling": "#f59e0b", "lineage+labelling": "#059669"}
    for metric, title, filename in [
        ("sign_agreement", "S02 sign agreement against expected-count truth", "sign_agreement.png"),
        ("oriented_fraction", "S02 oriented-cell fraction", "oriented_fraction.png"),
    ]:
        fig, ax = plt.subplots(figsize=(10.5, 5.2))
        x = np.arange(len(PRESETS))
        width = 0.24
        for offset, source in enumerate(source_order):
            subset = plot[plot["source_set"] == source]
            means = [subset.loc[subset["preset"] == p, metric].mean() for p in PRESETS]
            mins = [subset.loc[subset["preset"] == p, metric].min() for p in PRESETS]
            ax.bar(x + (offset - 1) * width, means, width, label=source, color=color[source])
            ax.scatter(x + (offset - 1) * width, mins, color="black", s=18, zorder=3, label="seed minimum" if offset == 0 else None)
        ax.set_xticks(x, PRESETS)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(title + " (PCA; smoothed labelling where applicable)")
        ax.legend(frameon=False, ncol=4)
        fig.tight_layout()
        fig.savefig(FIG_ROOT / filename, dpi=180)
        plt.close(fig)

    flags = results[(results["method"] == "pca") & (results["source_set"] == "lineage+labelling") & (results["labelling_variant"] == "smoothed")]
    categories = [("fraction_equilibrium", "equilibrium", "#2563eb"), ("fraction_non_equilibrium", "non-equilibrium", "#dc2626"), ("fraction_not_identified", "not identified", "#9ca3af")]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bottom = np.zeros(len(PRESETS))
    for column, label, color_value in categories:
        values = np.array([flags.loc[flags["preset"] == p, column].mean() for p in PRESETS])
        ax.bar(PRESETS, values, bottom=bottom, label=label, color=color_value)
        bottom += values
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction of cells")
    ax.set_title("S02 equilibrium flags (PCA; lineage + smoothed labelling)")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(FIG_ROOT / "flags_by_preset.png", dpi=180)
    plt.close(fig)


def write_result_markdown(results: pd.DataFrame, summary: pd.DataFrame, gate: list[dict[str, Any]]) -> None:
    gate_frame = pd.DataFrame(gate)
    primary = summary[(summary["method"] == "pca") & (summary["labelling_variant"].isin(["none", "smoothed"]))]
    flags_pca = primary[(primary["source_set"] == "lineage+labelling") & (primary["labelling_variant"] == "smoothed")]
    labelling_pca = summary[(summary["method"] == "pca") & (summary["source_set"] == "labelling")]
    contents = [
        "# S02 end-to-end orientation results",
        "",
        "## What ran",
        "- For every S01 manifest input (four presets × seeds 0–2) and PCA/FA embedding, fitted the frozen 1,500-epoch ESM EnergyDriftAdapter (hidden 96, depth 3, batch 256, lr 5e-4, beta 1.0, exact Laplacian; seed 0).",
        "- Fitted LabelledRateModel defaults (seed 0) on observed `new`/`total` counts and pulse hours; derived raw count velocity, mapped it through the stored observed-count transform, and smoothed it with default seed-0 MeasuredDrift.",
        "- Ran lineage, smoothed-labelling, combined, and raw-labelling sensitivity orientation calls. Generator truth was read only after those fits to score signs.",
        "",
        "## Gates (verbatim)",
        "- On default / PCA / smoothed labelling, for all three seeds: lineage sign agreement ≥ 0.90 and oriented fraction ≥ 0.3; labelling ≥ 0.85 and ≥ 0.3; combined ≥ 0.90 and ≥ 0.5.",
        "- On stationary_cycle / PCA / smoothed combined, non_equilibrium is ≥ 0.8 of cells with an identified equilibrium flag for every seed.",
        "- J hash is unchanged in every `orient` call. Symmetry check maximum is ≤ 1e-4 for every energy fit.",
        "",
        "## Gate results",
        write_table(gate_frame, ["gate", "minimum", "per_seed", "minimum_observed", "pass"]),
        "",
        "## PCA summary (mean and minimum over seeds)",
        write_table(primary.round(4), ["preset", "source_set", "labelling_variant", "n_rows", "sign_agreement_mean", "sign_agreement_min", "sign_agreement_moving_mean", "oriented_fraction_mean", "oriented_fraction_min", "conflict_rate_mean", "rotation_share_median_mean"]),
        "",
        "## PCA equilibrium and stationarity flags (combined, smoothed; mean over seeds)",
        write_table(flags_pca.round(4), ["preset", "fraction_equilibrium_mean", "fraction_non_equilibrium_mean", "fraction_not_identified_mean", "fraction_stationary_mean", "fraction_non_stationary_mean"]),
        "",
        "## PCA raw-versus-smoothed labelling sensitivity (mean and minimum over seeds)",
        write_table(labelling_pca.round(4), ["preset", "labelling_variant", "sign_agreement_mean", "sign_agreement_min", "sign_agreement_moving_mean", "oriented_fraction_mean", "rotation_share_median_mean", "labelled_fit_nll_mean"]),
        "",
        "## Complete outputs",
        "- `results.csv`: one row per dataset × embedding × source set × labelling variant.",
        "- `summary.csv`: mean and minimum-over-seed summaries per preset × embedding × source set.",
        "- Fitted H5AD outputs and Torch state dictionaries: `benchmarks/synthetic_multimodal/data/s02/` (local, gitignored).",
        "- Figures: `benchmarks/figures/orient/s02/`.",
    ]
    (RESULT_ROOT / "RESULT_S02.md").write_text("\n".join(contents) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, action="append", help="limit to a preset; default is all")
    parser.add_argument("--seed", type=int, action="append", choices=(0, 1, 2), help="limit to a seed; default is all")
    parser.add_argument("--force", action="store_true", help="refit existing units rather than safely reusing matching local checkpoints")
    args = parser.parse_args()
    rows = input_rows()
    if args.preset:
        rows = [r for r in rows if r["preset"] in set(args.preset)]
    if args.seed:
        rows = [r for r in rows if int(r["seed"]) in set(args.seed)]
    if not rows:
        parser.error("selection contains no manifest inputs")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    for row in rows:
        for method in METHODS:
            unit = fit_unit(row, method, force=args.force)
            units.append(unit)
            all_rows.extend(unit["rows"])
    results = pd.DataFrame(all_rows)
    results = results.sort_values(["preset", "seed", "method", "source_set", "labelling_variant"]).reset_index(drop=True)
    results.to_csv(RESULT_ROOT / "results.csv", index=False)
    summary = aggregate(results)
    summary.to_csv(RESULT_ROOT / "summary.csv", index=False)
    gates = gate_rows(results)
    write_result_markdown(results, summary, gates)
    render_figures(results)
    write_json(RESULT_ROOT / "RUN_METADATA.json", {"units": units, "gates": gates})
    print(json.dumps({"n_results": len(results), "gates": gates}, indent=2), flush=True)


if __name__ == "__main__":
    main()
