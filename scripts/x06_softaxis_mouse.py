#!/usr/bin/env python3
"""EXPLORATORY X06: fixed-axis curvature tests on three mouse basins.

Uses accepted A02/X04 shared spaces only. No model is fitted or changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import zlib
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from scjdo2.compare import centroid_operator, donor_operator, softaxis_curvature_trend

ROOT = Path(__file__).resolve().parents[1]
AGING = ROOT / "benchmarks" / "aging"
OUT = AGING / "x06"
A02_REPORT = ROOT / "coord" / "aging" / "reports" / "A02.json"
X04_REPORT = ROOT / "coord" / "aging" / "reports" / "X04.json"

SEED = 0
N_FLOOR = 30
N_DRAWS = 100
N_CENTROID = 30
N_PERM = 1_000
F_VALUES = (1.25, 1.5, 2.0, 3.0, 5.0)
TARGETS = (
    ("lung", "fibroblast of lung", "lung fibroblast"),
    ("lung", "natural killer cell", "lung NK"),
    ("marrow", "granulocyte", "marrow granulocyte"),
)
INPUTS = (
    ("A02", "benchmarks/aging/lung_shared.h5ad"),
    ("A02", "benchmarks/aging/marrow_shared.h5ad"),
    ("X04", "benchmarks/aging/x04/lung_shared_s1.h5ad"),
    ("X04", "benchmarks/aging/x04/lung_shared_s2.h5ad"),
    ("X04", "benchmarks/aging/x04/marrow_shared_s1.h5ad"),
    ("X04", "benchmarks/aging/x04/marrow_shared_s2.h5ad"),
)


@dataclass
class Context:
    tissue: str
    cell_type: str
    label: str
    seed: int
    z: np.ndarray
    jacobian: np.ndarray
    donor_indices: dict[str, np.ndarray]
    donor_age: dict[str, int]
    n_common: int
    n_young: int
    n_old: int
    n_labellings: int
    min_attainable_p: float
    ref_op: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reported_artifact(report_path: Path, relpath: str) -> dict[str, object]:
    report = json.loads(report_path.read_text())
    item = next((record for record in report["artifacts"] if record["path"] == relpath), None)
    if item is None:
        raise RuntimeError(f"{report_path} does not inventory {relpath}")
    return item


def verify_inputs() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for report_name, relpath in INPUTS:
        report_path = A02_REPORT if report_name == "A02" else X04_REPORT
        item = reported_artifact(report_path, relpath)
        path = ROOT / relpath
        actual = sha256(path)
        if path.stat().st_size != int(item["bytes"]) or actual != item["sha256"]:
            raise RuntimeError(f"upstream checksum mismatch: {relpath}")
        records.append({"path": relpath, "bytes": path.stat().st_size, "sha256": actual, "source_report": str(report_path.relative_to(ROOT))})
    return records


def input_path(tissue: str, seed: int) -> Path:
    if seed == 0:
        return AGING / f"{tissue}_shared.h5ad"
    return AGING / "x04" / f"{tissue}_shared_s{seed}.h5ad"


def load_context(tissue: str, cell_type: str, label: str, seed: int) -> Context:
    path = input_path(tissue, seed)
    data = ad.read_h5ad(path)
    if "X_scVI_shared" not in data.obsm or "scjdo2" not in data.uns:
        raise RuntimeError(f"{path} lacks X_scVI_shared or scjdo2 metadata")
    z = np.asarray(data.obsm["X_scVI_shared"], dtype=np.float64)
    jacobian = np.asarray(data.uns["scjdo2"].get("J", data.uns["scjdo2"].get("J_latent")), dtype=np.float64)
    if z.shape != (data.n_obs, 15) or jacobian.shape != (data.n_obs, 15, 15):
        raise RuntimeError(f"unexpected shared-space dimensions in {path}: {z.shape}, {jacobian.shape}")
    if not np.isfinite(z).all() or not np.isfinite(jacobian).all():
        raise RuntimeError(f"non-finite shared representation/Jacobian in {path}")
    required = {"cell_type", "donor_id", "age_group"}
    missing = sorted(required - set(data.obs.columns))
    if missing:
        raise RuntimeError(f"{path} missing {missing}")
    obs = data.obs.loc[:, ["cell_type", "donor_id", "age_group"]].copy()
    obs["cell_type"] = obs["cell_type"].astype(str)
    obs["donor_id"] = obs["donor_id"].astype(str)
    obs["age_group"] = obs["age_group"].astype(str)
    mask = obs["cell_type"].eq(cell_type).to_numpy()
    if not mask.any():
        raise RuntimeError(f"{cell_type!r} absent from {path}")
    donor_indices: dict[str, np.ndarray] = {}
    donor_age: dict[str, int] = {}
    for donor in sorted(obs.loc[mask, "donor_id"].unique()):
        index = np.flatnonzero(mask & obs["donor_id"].eq(donor).to_numpy())
        groups = set(obs.iloc[index]["age_group"])
        if len(groups) != 1 or groups.pop() not in {"young", "old"}:
            raise RuntimeError(f"invalid age group for {tissue}/{cell_type}/{donor}")
        if len(index) >= N_FLOOR:
            donor_indices[str(donor)] = index
            donor_age[str(donor)] = int(obs.iloc[index]["age_group"].iloc[0] == "old")
    n_young = sum(age == 0 for age in donor_age.values())
    n_old = sum(age == 1 for age in donor_age.values())
    if n_young == 0 or n_old == 0:
        raise RuntimeError(f"{tissue}/{cell_type}/seed{seed} has missing age group after {N_FLOOR}-cell floor")
    n_common = min(len(index) for index in donor_indices.values())
    n_total = n_young + n_old
    n_labellings = math.comb(n_total, n_young)
    young_pool = np.flatnonzero(mask & obs["age_group"].eq("young").to_numpy())
    if len(young_pool) < N_CENTROID:
        raise RuntimeError(f"{tissue}/{cell_type}/seed{seed} has <{N_CENTROID} pooled young cells")
    ref_op = centroid_operator(z[young_pool], jacobian[young_pool], n_cells_centroid=N_CENTROID)
    return Context(
        tissue=tissue, cell_type=cell_type, label=label, seed=seed, z=z, jacobian=jacobian,
        donor_indices=donor_indices, donor_age=donor_age, n_common=int(n_common),
        n_young=int(n_young), n_old=int(n_old), n_labellings=int(n_labellings),
        min_attainable_p=float(1 / n_labellings), ref_op=np.asarray(ref_op, dtype=np.float64),
    )


def context_signature(context: Context) -> tuple[tuple[str, int, int], ...]:
    return tuple((donor, context.donor_age[donor], len(context.donor_indices[donor])) for donor in sorted(context.donor_indices))


def donor_ops(context: Context, jacobian: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    operators: list[np.ndarray] = []
    ages: list[int] = []
    donors: list[str] = []
    for donor in sorted(context.donor_indices):
        index = context.donor_indices[donor]
        operator, _ = donor_operator(
            context.z[index], jacobian[index], n_common=context.n_common,
            n_draws=N_DRAWS, n_cells_centroid=N_CENTROID,
            seed=zlib.crc32(f"{context.cell_type}|{donor}".encode()),
        )
        operators.append(np.asarray(operator, dtype=np.float64))
        ages.append(context.donor_age[donor])
        donors.append(donor)
    return operators, np.asarray(ages, dtype=float), donors


def evaluate(context: Context, jacobian: np.ndarray) -> dict[str, object]:
    operators, ages, donors = donor_ops(context, jacobian)
    result = softaxis_curvature_trend(operators, ages, context.ref_op, n_perm=N_PERM, seed=SEED)
    if not np.isfinite([result.statistic, result.p_value, result.null_q95, result.detail["ref_soft_eigenvalue"]]).all():
        raise RuntimeError(f"non-finite fixed-axis statistic for {context.label}/seed{context.seed}")
    return {
        "seed": context.seed, "tissue": context.tissue, "cell_type": context.cell_type, "label": context.label,
        "n_young": context.n_young, "n_old": context.n_old, "n_common": context.n_common,
        "n_labellings": context.n_labellings, "min_attainable_p": context.min_attainable_p,
        "slope": float(result.statistic), "p": float(result.p_value), "passes": bool(result.passes),
        "ref_soft_eigenvalue": float(result.detail["ref_soft_eigenvalue"]),
        "null_q95_abs_slope": float(result.null_q95), "n_draws": N_DRAWS, "n_perm": N_PERM,
        "donors": ";".join(donors),
    }


def injected_jacobian(context: Context, condition: str, factor: float) -> np.ndarray:
    out = context.jacobian.copy()
    old = np.concatenate([context.donor_indices[donor] for donor in sorted(context.donor_indices) if context.donor_age[donor] == 1])
    if condition == "single_axis":
        _, vectors = np.linalg.eigh(0.5 * (context.ref_op + context.ref_op.T))
        soft_axis = vectors[:, -1]
        projection = np.einsum("i,nij,j->n", soft_axis, out[old], soft_axis)
        out[old] += ((factor ** -1 - 1.0) * projection)[:, None, None] * np.outer(soft_axis, soft_axis)[None, :, :]
    elif condition == "uniform":
        out[old] /= factor
    else:
        raise ValueError(f"unknown injection condition: {condition}")
    return out


def power_rows(context: Context) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for factor in F_VALUES:
        record = evaluate(context, injected_jacobian(context, "single_axis", factor))
        record.update({"condition": "single_axis", "F": float(factor)})
        rows.append(record)
    uniform = evaluate(context, injected_jacobian(context, "uniform", 2.0))
    uniform.update({"condition": "uniform", "F": 2.0})
    rows.append(uniform)
    return rows


def derive_summary(real: pd.DataFrame, power: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for tissue, cell_type, label in TARGETS:
        data = real[(real["tissue"] == tissue) & (real["cell_type"] == cell_type)].sort_values("seed", kind="stable")
        if tuple(data["seed"]) != (0, 1, 2):
            raise RuntimeError(f"missing real seed rows for {label}")
        zero = data[data["seed"] == 0].iloc[0]
        supporting = data[(data["seed"].isin((1, 2))) & (data["p"] <= .05) & (np.sign(data["slope"]) == np.sign(zero["slope"]))]
        verdict = "soft-axis change with age" if bool(zero["p"] <= .05 and not supporting.empty) else "no soft-axis change with age"
        power_rows_for_type = power[(power["tissue"] == tissue) & (power["cell_type"] == cell_type)]
        single = power_rows_for_type[power_rows_for_type["condition"] == "single_axis"].sort_values("F", kind="stable")
        passing = single[single["passes"]]
        mde: str | float = float(passing.iloc[0]["F"]) if not passing.empty else "> max"
        uniform = power_rows_for_type[(power_rows_for_type["condition"] == "uniform") & (power_rows_for_type["F"] == 2.0)]
        if len(uniform) != 1:
            raise RuntimeError(f"missing uniform ×2 record for {label}")
        rows.append({
            "tissue": tissue, "cell_type": cell_type, "label": label,
            "n_young": int(zero["n_young"]), "n_old": int(zero["n_old"]), "n_common": int(zero["n_common"]),
            "n_labellings": int(zero["n_labellings"]), "min_attainable_p": float(zero["min_attainable_p"]),
            "seed0_slope": float(zero["slope"]), "seed0_p": float(zero["p"]),
            "seed1_slope": float(data.iloc[1]["slope"]), "seed1_p": float(data.iloc[1]["p"]),
            "seed2_slope": float(data.iloc[2]["slope"]), "seed2_p": float(data.iloc[2]["p"]),
            "mde_single_axis_F": mde, "uniform_x2_passes": bool(uniform.iloc[0]["passes"]),
            "supporting_seeds": ";".join(str(int(seed)) for seed in supporting["seed"]) or "none",
            "x06_verdict": verdict,
        })
    return pd.DataFrame(rows)


def md_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for values in frame.loc[:, columns].itertuples(index=False, name=None):
        cells: list[str] = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                cells.append(f"{value:.6g}")
            elif isinstance(value, (bool, np.bool_)):
                cells.append(str(bool(value)).lower())
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_result(real: pd.DataFrame, power: pd.DataFrame, summary: pd.DataFrame) -> None:
    rule = "a type shows **soft-axis change with age** if `p ≤ 0.05` in seed 0 **and** in at least one of seeds 1, 2, with the same sign of slope."
    text = "EXPLORATORY — statistic designed after the frozen result\n\n"
    text += "## What ran\nThe verified A02 seed-0 and X04 seeds-1/2 shared spaces were used without model fitting. Per type and seed, mice with at least 30 cells contributed deterministic 100-draw equal-count donor operators. The pooled young 30-cell centroid operator defined the fixed soft axis. Seed 0 also ran the stated old-cell single-axis factors and uniform ×2 specificity injection.\n\n"
    text += "## Reading rule (verbatim)\n" + rule + "\n\n"
    text += "## Real data\n" + md_table(real, ["seed", "tissue", "cell_type", "n_young", "n_old", "n_common", "n_labellings", "min_attainable_p", "slope", "p", "passes", "ref_soft_eigenvalue"]) + "\n\n"
    text += "## Power and specificity (seed 0)\n" + md_table(power, ["tissue", "cell_type", "condition", "F", "slope", "p", "passes"]) + "\n\n"
    text += "## Mechanical summary\n" + md_table(summary, ["label", "n_young", "n_old", "n_common", "n_labellings", "min_attainable_p", "mde_single_axis_F", "uniform_x2_passes", "supporting_seeds", "x06_verdict"]) + "\n"
    (OUT / "RESULT_X06.md").write_text(text)


def run(dry_run: bool) -> dict[str, object]:
    input_records = verify_inputs()
    contexts: dict[tuple[str, str, int], Context] = {}
    for tissue, cell_type, label in TARGETS:
        seed_contexts = [load_context(tissue, cell_type, label, seed) for seed in (0, 1, 2)]
        signature = context_signature(seed_contexts[0])
        if any(context_signature(context) != signature for context in seed_contexts[1:]):
            raise RuntimeError(f"contributor identities/counts differ across shared seeds for {label}")
        for context in seed_contexts:
            contexts[(tissue, cell_type, context.seed)] = context
    plan = {"seed": SEED, "n_floor": N_FLOOR, "n_draws": N_DRAWS, "n_perm": N_PERM, "targets": [{"label": c.label, "seed": c.seed, "n_young": c.n_young, "n_old": c.n_old, "n_common": c.n_common, "n_labellings": c.n_labellings, "min_attainable_p": c.min_attainable_p} for c in contexts.values()]}
    if dry_run:
        print(json.dumps({**plan, "dry_run": True}, indent=2, sort_keys=True)); return plan
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "input_checksums.json").write_text(json.dumps(input_records, indent=2) + "\n")
    real = pd.DataFrame([evaluate(context, context.jacobian) for context in contexts.values()]).sort_values(["tissue", "cell_type", "seed"], kind="stable").reset_index(drop=True)
    power = pd.DataFrame([record for tissue, cell_type, _ in TARGETS for record in power_rows(contexts[(tissue, cell_type, 0)])]).sort_values(["tissue", "cell_type", "condition", "F"], kind="stable").reset_index(drop=True)
    summary = derive_summary(real, power).sort_values(["tissue", "cell_type"], kind="stable").reset_index(drop=True)
    real.to_csv(OUT / "real.csv", index=False)
    power.to_csv(OUT / "power.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    write_result(real, power, summary)
    metrics = {
        "seed": SEED, "verdict_by_type": summary.set_index("label")["x06_verdict"].to_dict(),
        "mde_by_type": summary.set_index("label")["mde_single_axis_F"].to_dict(),
        "uniform_x2_passes_by_type": {label: bool(value) for label, value in summary.set_index("label")["uniform_x2_passes"].items()},
        "min_attainable_p_by_type": {label: float(value) for label, value in summary.set_index("label")["min_attainable_p"].items()},
        "n_types_showing_change": int(summary["x06_verdict"].eq("soft-axis change with age").sum()),
        "n_real_rows": int(len(real)), "n_power_rows": int(len(power)), "n_perm": N_PERM, "n_draws": N_DRAWS,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.seed != SEED:
        raise ValueError("X06 is frozen at seed 0")
    run(args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
