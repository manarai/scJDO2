#!/usr/bin/env python3
"""EXPLORATORY HX03 fixed-axis curvature statistic on real HLCA donor cells.

This task is designed after the frozen H04 result. It verifies specificity against
uniform scaling before applying softaxis_curvature_trend to real H03 data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import zlib
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scjdo2.compare import centroid_operator, donor_operator, softaxis_curvature_trend

ROOT = Path(__file__).resolve().parents[1]
HLCA = ROOT / "benchmarks" / "aging_hlca"
OUT = HLCA / "softaxis"
FIG = ROOT / "benchmarks" / "figures" / "aging_hlca" / "softaxis"
PRIMARY = HLCA / "data" / "h03_primary_drift.h5ad"
SECONDARY = HLCA / "data" / "h03_secondary_drift.h5ad"
TESTS_PRIMARY = HLCA / "tests_primary.csv"
HX02_GRID = HLCA / "power" / "power_grid.csv"
H03_REPORT = ROOT / "coord" / "hlca" / "reports" / "H03.json"
HX02_REPORT = ROOT / "coord" / "hlca" / "reports" / "HX02.json"
HX02_INPUTS = HLCA / "power" / "input_checksums.json"
GRID = OUT / "power_grid.csv"
MDE = OUT / "mde.csv"
REAL = OUT / "real.csv"
SUMMARY = OUT / "summary.csv"
RESULT = OUT / "RESULT_HX03.md"
CHECKSUMS = OUT / "input_checksums.json"
METRICS = OUT / "metrics.json"

SEED = 0
N_COMMON, N_DRAWS, N_CENTROID = 50, 100, 30
N_REPLICATES, DONOR_FRACTION, N_PERM_POWER, N_PERM_REAL = 50, 0.80, 200, 1_000
SINGLE_AXIS_F = (1.25, 1.5, 2.0, 3.0, 5.0, 8.0)
UNIFORM_F = (1.5, 2.0, 3.0)
FOOTER = "EXPLORATORY · hlca/HX03 · statistic designed after the frozen result"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_record(report_path: Path, rel: str) -> dict[str, object]:
    report = json.loads(report_path.read_text())
    record = next((x for x in report["artifacts"] if x["path"] == rel), None)
    if record is None:
        raise RuntimeError(f"{report_path} does not inventory {rel}")
    return record


def verify_record(rel: str, record: dict[str, object], source: str) -> dict[str, object]:
    path = ROOT / rel
    actual = sha256(path)
    if path.stat().st_size != int(record["bytes"]) or actual != record["sha256"]:
        raise RuntimeError(f"checksum mismatch for {rel} against {source}")
    return {"path": rel, "bytes": path.stat().st_size, "sha256": actual, "source": source}


def verify_inputs() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for rel in ("benchmarks/aging_hlca/data/h03_primary_drift.h5ad", "benchmarks/aging_hlca/data/h03_secondary_drift.h5ad"):
        records.append(verify_record(rel, artifact_record(H03_REPORT, rel), "coord/hlca/reports/H03.json"))
    rel = "benchmarks/aging_hlca/power/power_grid.csv"
    records.append(verify_record(rel, artifact_record(HX02_REPORT, rel), "coord/hlca/reports/HX02.json"))
    hx02_inputs = json.loads(HX02_INPUTS.read_text())
    for item in hx02_inputs:
        rel = str(item["path"])
        checked = verify_record(rel, item, f"benchmarks/aging_hlca/power/input_checksums.json ({item['source_report']})")
        previous = next((x for x in records if x["path"] == rel), None)
        if previous is None:
            records.append(checked)
        elif previous["sha256"] != checked["sha256"] or previous["bytes"] != checked["bytes"]:
            raise RuntimeError(f"conflicting inherited checksums for {rel}")
        else:
            previous["source"] = f"{previous['source']}; {checked['source']}"
    expected = {"benchmarks/aging_hlca/data/h03_primary_drift.h5ad", "benchmarks/aging_hlca/data/h03_secondary_drift.h5ad", "benchmarks/aging_hlca/power/power_grid.csv", "benchmarks/aging_hlca/tests_primary.csv"}
    if {str(x["path"]) for x in records} != expected:
        raise RuntimeError("unexpected inherited-input inventory")
    return records


def open_embedding(path: Path, expected_key: str, expected_dim: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, tuple[str, ...]]:
    adata = ad.read_h5ad(path)
    if expected_key not in adata.obsm:
        raise RuntimeError(f"{path} missing {expected_key}")
    z = np.asarray(adata.obsm[expected_key], dtype=np.float64)
    j = np.asarray(adata.uns.get("scjdo2", {}).get("J_latent"), dtype=np.float64)
    required = ("ann_level_3", "donor_id", "age", "study_or_dataset")
    missing = sorted(set(required) - set(adata.obs.columns))
    if missing:
        raise RuntimeError(f"{path} missing obs columns {missing}")
    if z.shape != (adata.n_obs, expected_dim) or j.shape != (adata.n_obs, expected_dim, expected_dim):
        raise RuntimeError(f"unexpected {path.name} dimensions: z={z.shape}, J={j.shape}")
    if not np.isfinite(z).all() or not np.isfinite(j).all():
        raise RuntimeError(f"{path} contains non-finite latent data")
    obs = adata.obs.loc[:, required].copy()
    obs.index = obs.index.astype(str)
    obs["ann_level_3"] = obs["ann_level_3"].astype(str)
    obs["donor_id"] = obs["donor_id"].astype(str)
    obs["study_or_dataset"] = obs["study_or_dataset"].astype(str)
    obs["age"] = pd.to_numeric(obs["age"], errors="raise").astype(float)
    donor_meta = obs.groupby("donor_id", sort=True).agg(age=("age", "nunique"), study=("study_or_dataset", "nunique"))
    if not donor_meta.eq(1).all().all():
        raise RuntimeError(f"age or study varies within a donor in {path.name}")
    return z, j, obs, tuple(adata.obs_names.astype(str))


def load_space_data(z: np.ndarray, j: np.ndarray, obs: pd.DataFrame, types: tuple[str, ...], expected_donors: dict[str, int]) -> dict[str, dict[str, object]]:
    labels = obs["ann_level_3"].to_numpy(str)
    donors = obs["donor_id"].to_numpy(str)
    ages = obs["age"].to_numpy(float)
    studies = obs["study_or_dataset"].to_numpy(str)
    out: dict[str, dict[str, object]] = {}
    for cell_type in types:
        type_idx = np.flatnonzero(labels == cell_type)
        if not len(type_idx):
            raise RuntimeError(f"{cell_type} absent")
        donor_names: list[str] = []
        donor_ages: list[float] = []
        donor_studies: list[str] = []
        donor_positions: list[np.ndarray] = []
        for donor in sorted(set(donors[type_idx])):
            pos = type_idx[donors[type_idx] == donor]
            if len(pos) < N_COMMON:
                continue
            donor_names.append(donor)
            donor_ages.append(float(ages[pos[0]]))
            donor_studies.append(str(studies[pos[0]]))
            donor_positions.append(pos)
        if len(donor_names) != expected_donors[cell_type]:
            raise RuntimeError(f"H04 contributor mismatch for {cell_type}: {len(donor_names)} != {expected_donors[cell_type]}")
        n_sub = int(math.floor(DONOR_FRACTION * len(donor_names)))
        if n_sub < 4:
            raise RuntimeError(f"insufficient 80%-donor subsample for {cell_type}")
        repetitions: list[np.ndarray] = []
        for r in range(N_REPLICATES):
            rng = np.random.default_rng(zlib.crc32(f"{cell_type}|rep{r}".encode()))
            repetitions.append(np.sort(rng.choice(len(donor_names), n_sub, replace=False)))
        young = type_idx[ages[type_idx] < 45]
        if len(young) < N_CENTROID:
            raise RuntimeError(f"fewer than {N_CENTROID} young pooled cells for {cell_type}")
        reference = centroid_operator(z[young], j[young], n_cells_centroid=N_CENTROID)
        eig, vectors = np.linalg.eigh(0.5 * (reference + reference.T))
        out[cell_type] = {
            "z": z, "j": j, "ages_all": ages, "donor_names": tuple(donor_names),
            "donor_ages": np.asarray(donor_ages, dtype=float), "donor_studies": tuple(donor_studies),
            "donor_positions": tuple(donor_positions), "repetitions": tuple(repetitions),
            "reference": reference, "soft_axis": vectors[:, int(np.argmax(eig))],
            "ref_soft_eigenvalue": float(np.max(eig)), "n_sub": n_sub,
        }
    return out


def inject(item: dict[str, object], kind: str, fold: float) -> tuple[np.ndarray, np.ndarray]:
    z = np.asarray(item["z"]); j = np.asarray(item["j"]); ages = np.asarray(item["ages_all"])
    if kind == "none":
        return z, j
    t = (ages - 10.0) / 66.0
    if kind == "single_axis":
        u = np.asarray(item["soft_axis"])
        curvature = np.einsum("i,nij,j->n", u, j, u, optimize=True)
        return z, j + ((np.power(fold, -t) - 1.0) * curvature)[:, None, None] * np.outer(u, u)[None, :, :]
    if kind == "uniform":
        return z, np.power(fold, -t)[:, None, None] * j
    raise ValueError(kind)


def donor_operators(cell_type: str, item: dict[str, object], z: np.ndarray, j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    operators: list[np.ndarray] = []
    centroids: list[np.ndarray] = []
    for donor, pos in zip(item["donor_names"], item["donor_positions"], strict=True):
        op, centroid = donor_operator(z[pos], j[pos], n_common=N_COMMON, n_draws=N_DRAWS, n_cells_centroid=N_CENTROID, seed=zlib.crc32(f"{cell_type}|{donor}".encode()))
        operators.append(op)
        centroids.append(centroid)
    return np.stack(operators), np.stack(centroids)


def replicate_power(item: dict[str, object], operators: np.ndarray) -> float:
    passes: list[bool] = []
    for selected in item["repetitions"]:
        result = softaxis_curvature_trend(operators[selected], np.asarray(item["donor_ages"])[selected], np.asarray(item["reference"]), n_perm=N_PERM_POWER, seed=SEED)
        passes.append(bool(result.passes))
    return float(np.mean(passes))


def mde_single_axis(rows: pd.DataFrame) -> float | str:
    values = rows[(rows["kind"] == "single_axis") & (rows["power"] >= .80)]["size"]
    return float(values.min()) if len(values) else "> max"


def median_mde(series: pd.Series) -> str:
    censored = int((series.astype(str) == "> max").sum())
    if censored >= (len(series) + 1) // 2:
        return ">8"
    numeric = [float(x) for x in series if str(x) != "> max"]
    return f"{np.median(numeric):g}" if numeric else ">8"


def prepare_power(primary: dict[str, dict[str, object]], types: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    rows: list[dict[str, object]] = []
    base: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    started = time.monotonic(); total = len(types) * (1 + len(SINGLE_AXIS_F) + len(UNIFORM_F)); done = 0
    for cell_type in types:
        item = primary[cell_type]
        for kind, sizes in (("none", (1.0,)), ("single_axis", SINGLE_AXIS_F), ("uniform", UNIFORM_F)):
            for size in sizes:
                done += 1
                if kind == "none":
                    z, j = inject(item, kind, size)
                    operators, centroids = donor_operators(cell_type, item, z, j)
                    base[cell_type] = (operators, centroids)
                else:
                    z, j = inject(item, kind, size)
                    operators, _ = donor_operators(cell_type, item, z, j)
                power = replicate_power(item, operators)
                rows.append({"ann_level_3": cell_type, "kind": kind, "size": float(size), "power": power})
                print(f"[{done:03d}/{total}] {cell_type} {kind} F={size:g} power={power:.2f} elapsed={time.monotonic()-started:.1f}s", flush=True)
    power = pd.DataFrame(rows)
    type_order = {x: i for i, x in enumerate(types)}; kind_order = {"none": 0, "single_axis": 1, "uniform": 2}
    power = power.assign(_type=power.ann_level_3.map(type_order), _kind=power.kind.map(kind_order)).sort_values(["_type", "_kind", "size"], kind="stable").drop(columns=["_type", "_kind"]).reset_index(drop=True)
    mde_rows: list[dict[str, object]] = []
    for cell_type in types:
        item = primary[cell_type]
        part = power[power.ann_level_3 == cell_type]
        uniform = part[part.kind == "uniform"]
        mde_rows.append({"ann_level_3": cell_type, "mde_single_axis_F": mde_single_axis(part), "ref_soft_eigenvalue": float(item["ref_soft_eigenvalue"]), "max_uniform_pass_rate": float(uniform.power.max())})
    return power, pd.DataFrame(mde_rows), base


def application_row(analysis: str, cell_type: str, item: dict[str, object], operators: np.ndarray, *, strata: tuple[str, ...] | None, mde: float | str) -> tuple[dict[str, object], np.ndarray, object]:
    result = softaxis_curvature_trend(operators, np.asarray(item["donor_ages"]), np.asarray(item["reference"]), n_perm=N_PERM_REAL, seed=SEED, strata=strata)
    if not np.isfinite([result.statistic, result.p_value, result.detail["ref_soft_eigenvalue"]]).all():
        raise RuntimeError(f"non-finite real-data statistic for {analysis}/{cell_type}")
    return ({"analysis": analysis, "ann_level_3": cell_type, "n_donors": len(operators), "slope": float(result.statistic), "p": float(result.p_value), "passes": bool(result.passes), "ref_soft_eigenvalue": float(result.detail["ref_soft_eigenvalue"]), "mde_single_axis_F": mde}, np.asarray(result.detail["kappa"], dtype=float), result)


def apply_real(primary: dict[str, dict[str, object]], secondary: dict[str, dict[str, object]], mde_table: pd.DataFrame, base: dict[str, tuple[np.ndarray, np.ndarray]], application_types: tuple[str, ...]) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    mde_map = mde_table.set_index("ann_level_3")["mde_single_axis_F"].to_dict()
    rows: list[dict[str, object]] = []
    plot_data: dict[str, dict[str, object]] = {}
    for cell_type in application_types:
        item = primary[cell_type]
        primary_ops, _ = base[cell_type]
        row, kappa, result = application_row("primary", cell_type, item, primary_ops, strata=None, mde=mde_map[cell_type])
        rows.append(row)
        plot_data[cell_type] = {"ages": np.asarray(item["donor_ages"]), "studies": tuple(item["donor_studies"]), "kappa": kappa, "slope": float(result.statistic), "p": float(result.p_value)}
        row, _, _ = application_row("a_within_study", cell_type, item, primary_ops, strata=tuple(item["donor_studies"]), mde=mde_map[cell_type])
        rows.append(row)
        second_item = secondary[cell_type]
        secondary_ops, _ = donor_operators(cell_type, second_item, np.asarray(second_item["z"]), np.asarray(second_item["j"]))
        row, _, _ = application_row("e_secondary_space", cell_type, second_item, secondary_ops, strata=None, mde=mde_map[cell_type])
        rows.append(row)
    columns = ["analysis", "ann_level_3", "n_donors", "slope", "p", "passes", "ref_soft_eigenvalue", "mde_single_axis_F"]
    return pd.DataFrame(rows, columns=columns), plot_data


def summarize(real: pd.DataFrame, mde: pd.DataFrame, types: tuple[str, ...]) -> pd.DataFrame:
    failing = tuple(mde.loc[mde.max_uniform_pass_rate > .20, "ann_level_3"].astype(str))
    if not types:
        return pd.DataFrame([{"types_in_application": 0, "alpha": .05, "corrected_alpha": float("nan"), "n_pass_primary": 0, "n_pass_primary_corrected": 0, "n_pass_within_study": 0, "n_pass_secondary": 0, "n_robust": 0, "robust_types": "", "n_types_failing_uniform_gate": len(failing), "types_failing_uniform_gate": ";".join(failing)}])
    primary = real[real.analysis == "primary"].set_index("ann_level_3")
    within = real[real.analysis == "a_within_study"].set_index("ann_level_3")
    secondary = real[real.analysis == "e_secondary_space"].set_index("ann_level_3")
    threshold = .05 / len(types)
    robust: list[str] = []
    for cell_type in types:
        slopes = [float(primary.loc[cell_type, "slope"]), float(within.loc[cell_type, "slope"]), float(secondary.loc[cell_type, "slope"])]
        if (float(primary.loc[cell_type, "p"]) <= threshold and float(within.loc[cell_type, "p"]) <= .05 and float(secondary.loc[cell_type, "p"]) <= .05 and all(np.sign(x) == np.sign(slopes[0]) and np.sign(x) != 0 for x in slopes)):
            robust.append(cell_type)
    values = {"types_in_application": len(types), "alpha": .05, "corrected_alpha": threshold, "n_pass_primary": int((primary.p <= .05).sum()), "n_pass_primary_corrected": int((primary.p <= threshold).sum()), "n_pass_within_study": int((within.p <= .05).sum()), "n_pass_secondary": int((secondary.p <= .05).sum()), "n_robust": len(robust), "robust_types": ";".join(robust), "n_types_failing_uniform_gate": len(failing), "types_failing_uniform_gate": ";".join(failing)}
    return pd.DataFrame([values])


def plot_power(power: pd.DataFrame, types: tuple[str, ...]) -> list[Path]:
    FIG.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(HX02_GRID)
    fig, ax = plt.subplots(figsize=(12.2, 6.8))
    colours = plt.cm.tab20(np.linspace(0, 1, len(types)))
    for color, cell_type in zip(colours, types, strict=True):
        current = power[(power.ann_level_3 == cell_type) & (power.kind == "single_axis")].sort_values("size")
        hx02 = old[(old.ann_level_3 == cell_type) & (old.kind == "single_axis") & (old.size > 1)].sort_values("size")
        ax.plot(current.size, current.power, color=color, marker="o", linewidth=1.35, markersize=3.5, label=cell_type)
        ax.plot(hx02.size, hx02.power, color=color, linestyle="--", linewidth=.95, alpha=.72)
    ax.axhline(.8, color="#c94c4c", linestyle=":", linewidth=1.3, label="80% power")
    ax.set_xscale("log", base=2); ax.set_ylim(-.02, 1.04); ax.grid(color=".90")
    ax.set_xlabel("single-axis softening fold F"); ax.set_ylabel("power across 50 donor subsamples")
    ax.set_title("Fixed-axis curvature power (solid) vs HX02 isotropy power (dashed)", loc="left", weight="bold")
    ax.legend(ncol=3, fontsize=6.1, frameon=False, loc="lower right")
    fig.text(.5, .01, FOOTER, ha="center", fontsize=7, color=".35"); fig.tight_layout(rect=(0, .04, 1, .98))
    paths: list[Path] = []
    for extension in ("pdf", "png"):
        path = FIG / f"power_curves.{extension}"; fig.savefig(path, dpi=200, bbox_inches="tight"); paths.append(path)
    plt.close(fig)
    return paths


def safe_name(cell_type: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in cell_type).strip("_")


def plot_kappa(plot_data: dict[str, dict[str, object]], types: tuple[str, ...]) -> list[Path]:
    paths: list[Path] = []
    for cell_type in types:
        item = plot_data[cell_type]; ages = np.asarray(item["ages"]); kappa = np.asarray(item["kappa"]); studies = np.asarray(item["studies"])
        fig, ax = plt.subplots(figsize=(7.7, 5.3)); unique = sorted(set(studies)); cmap = plt.cm.tab20
        for index, study in enumerate(unique):
            mask = studies == study
            ax.scatter(ages[mask], kappa[mask], s=23, alpha=.82, color=cmap(index % 20), label=study)
        intercept = float(np.mean(kappa) - item["slope"] * np.mean(ages))
        x = np.linspace(float(ages.min()), float(ages.max()), 100)
        ax.plot(x, intercept + item["slope"] * x, color="#1b1b1b", linewidth=1.4, label="OLS slope")
        ax.axhline(0, color=".7", linewidth=.8)
        ax.set_xlabel("donor age (years)"); ax.set_ylabel("κ_d = uᵀ Op_d u / median|λ(Op_d)|")
        ax.set_title(f"{cell_type}: fixed-axis curvature vs age\nslope={item['slope']:.3g}; p={item['p']:.4g}", loc="left", weight="bold")
        ax.legend(fontsize=6.3, frameon=False, loc="best", ncol=2); ax.grid(color=".92")
        fig.text(.5, .01, FOOTER, ha="center", fontsize=7, color=".35"); fig.tight_layout(rect=(0, .04, 1, .98))
        for extension in ("pdf", "png"):
            path = FIG / f"kappa_{safe_name(cell_type)}.{extension}"; fig.savefig(path, dpi=200, bbox_inches="tight"); paths.append(path)
        plt.close(fig)
    return paths


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame[columns].itertuples(index=False, name=None):
        values = []
        for value in row:
            values.append(f"{float(value):.4g}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_result(power: pd.DataFrame, mde: pd.DataFrame, real: pd.DataFrame, summary: pd.DataFrame) -> None:
    s = summary.iloc[0]
    reading = """* a type shows **soft-axis change with age** in an analysis if `p ≤ 0.05`; it shows it **after correction** if `p ≤ 0.05 / n` (`n` = number of types in step 3);
* a type is **robust** if it shows the change after correction in `primary` **and** at p ≤ 0.05 in both `a_within_study` and `e_secondary_space`, with the same sign of slope;
* slope > 0 means `κ` moves toward zero or positive (the soft axis flattens relative to the rest), slope < 0 the opposite."""
    text = "EXPLORATORY — statistic designed after the frozen result\n\n"
    text += "## What ran\nThe exact HX02 real-cell donor operator and injection protocol was repeated for all 13 primary types: no injection, single-axis F ∈ {1.25, 1.5, 2, 3, 5, 8}, and uniform F ∈ {1.5, 2, 3}. The fixed-axis `softaxis_curvature_trend` used 50 fixed 80%-donor replicates and 200 age permutations per replicate. Types with max uniform false-pass power > 0.20 were excluded before three full-donor real-data analyses at 1,000 permutations.\n\n"
    text += "## Reading rule (verbatim)\n" + reading + "\n\n"
    text += "## Power-grid MDE and specificity gate\n" + markdown_table(mde, ["ann_level_3", "mde_single_axis_F", "ref_soft_eigenvalue", "max_uniform_pass_rate"]) + "\n\n"
    text += "## Real-data statistics\n" + markdown_table(real, ["analysis", "ann_level_3", "n_donors", "slope", "p", "passes", "ref_soft_eigenvalue", "mde_single_axis_F"]) + "\n\n"
    text += "## Mechanical counts\n" + markdown_table(summary, list(summary.columns)) + "\n"
    RESULT.write_text(text)


def run(dry_run: bool) -> dict[str, object]:
    checksums = verify_inputs()
    primary_z, primary_j, primary_obs, primary_names = open_embedding(PRIMARY, "X_scVI_primary", 15)
    secondary_z, secondary_j, secondary_obs, secondary_names = open_embedding(SECONDARY, "X_hlca_published", 30)
    if primary_names != secondary_names or not primary_obs.equals(secondary_obs):
        raise RuntimeError("H03 primary/secondary cell identity or observation metadata mismatch")
    tests = pd.read_csv(TESTS_PRIMARY)
    if len(tests) != 13 or tests.ann_level_3.duplicated().any() or {"ann_level_3", "n_donors"} - set(tests):
        raise RuntimeError("tests_primary must be the frozen unique 13-type table")
    types = tuple(tests.ann_level_3.astype(str)); expected = tests.set_index("ann_level_3").n_donors.astype(int).to_dict()
    primary = load_space_data(primary_z, primary_j, primary_obs, types, expected)
    secondary = load_space_data(secondary_z, secondary_j, secondary_obs, types, expected)
    plan = {"seed": SEED, "n_types": len(types), "n_power_rows_expected": len(types) * (1 + len(SINGLE_AXIS_F) + len(UNIFORM_F)), "n_replicates": N_REPLICATES, "n_perm_power": N_PERM_POWER, "n_perm_real": N_PERM_REAL, "n_donors_per_replicate": {cell_type: int(primary[cell_type]["n_sub"]) for cell_type in types}}
    if dry_run:
        print(json.dumps({**plan, "dry_run": True}, indent=2, sort_keys=True)); return plan
    OUT.mkdir(parents=True, exist_ok=True); CHECKSUMS.write_text(json.dumps(checksums, indent=2) + "\n")
    power, mde, base = prepare_power(primary, types)
    gate_fail = tuple(mde.loc[mde.max_uniform_pass_rate > .20, "ann_level_3"].astype(str))
    application_types = tuple(cell_type for cell_type in types if cell_type not in gate_fail)
    real, plot_data = apply_real(primary, secondary, mde, base, application_types)
    summary = summarize(real, mde, application_types)
    power.to_csv(GRID, index=False); mde.to_csv(MDE, index=False); real.to_csv(REAL, index=False); summary.to_csv(SUMMARY, index=False)
    figure_paths = plot_power(power, types) + plot_kappa(plot_data, application_types)
    write_result(power, mde, real, summary)
    s = summary.iloc[0]
    output = {**plan, "median_mde_single_axis_F": median_mde(mde.mde_single_axis_F), "n_types_mde_beyond_grid": int((mde.mde_single_axis_F.astype(str) == "> max").sum()), "n_types_failing_uniform_gate": int(s.n_types_failing_uniform_gate), "types_in_application": int(s.types_in_application), "n_pass_primary": int(s.n_pass_primary), "n_pass_primary_corrected": int(s.n_pass_primary_corrected), "n_pass_within_study": int(s.n_pass_within_study), "n_pass_secondary": int(s.n_pass_secondary), "robust_types": str(s.robust_types), "types_failing_uniform_gate": str(s.types_failing_uniform_gate), "n_figures": len(figure_paths) // 2, "figure_paths": [str(path.relative_to(ROOT)) for path in figure_paths]}
    METRICS.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True)); return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.seed != SEED:
        raise ValueError("HX03 is frozen at seed 0")
    run(args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
