#!/usr/bin/env python3
"""Render A04 figures from stored A01–A03 artifacts only; no model is fit."""
from __future__ import annotations
import argparse

import csv
import textwrap
import hashlib
import json
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap
from sklearn.neighbors import NearestNeighbors

from scjdo2.energy_drift import EnergyDriftAdapter, EnergyDriftConfig, _EnergyMLP
from scjdo2 import plot

ROOT = Path(__file__).resolve().parents[1]
AGING = ROOT / "benchmarks" / "aging"
OUT = ROOT / "benchmarks" / "figures" / "aging"
FOOTER = "v0.5-basin-lock · aging-basins · classify_basin_latent(rule='lead_eig') · scArches shared space · d_latent=15 · units relative"
SOURCES = [
    "coord/aging/reports/A01.json", "coord/aging/reports/A02.json", "coord/aging/reports/A03.json",
]
MANIFEST: list[dict[str, str]] = []


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_inputs() -> None:
    verified: list[dict[str, str]] = []
    for report in SOURCES:
        data = json.loads((ROOT / report).read_text())
        for art in data["artifacts"]:
            path = ROOT / art["path"]
            actual = _sha(path)
            # RESULT.md is an A04-owned mutable output, listed by A03 only as
            # its then-current write-up; it is not an input to figure rendering.
            if art["path"] == "benchmarks/aging/RESULT.md":
                verified.append({"path": art["path"], "sha256": actual,
                                 "note": "A04-owned mutable result document; excluded from upstream input match"})
                continue
            if actual != art["sha256"]:
                raise RuntimeError(f"checksum mismatch: {art['path']}")
            verified.append({"path": art["path"], "sha256": actual})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "input_checksums.json").write_text(json.dumps(verified, indent=2) + "\n")


def save(fig, rel: str, spec_item: str, tissue: str, age: str, source_files: list[str]) -> None:
    path = OUT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if not any(FOOTER in text.get_text() for text in fig.texts):
        fig.text(0.5, 0.005, FOOTER, ha="center", va="bottom", fontsize=5, color="0.35")
    for suffix in (".pdf", ".png"):
        out = path.with_suffix(suffix)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        MANIFEST.append({"path": str(out.relative_to(ROOT)), "spec_item": spec_item,
                         "tissue": tissue, "age": age,
                         "source_files": ";".join(source_files)})
    plt.close(fig)


def note_skip(spec_item: str, reason: str) -> None:
    MANIFEST.append({"path": "benchmarks/figures/aging/DEVIATIONS.md", "spec_item": spec_item,
                     "tissue": "both", "age": "all", "source_files": reason})


def view(a: ad.AnnData) -> ad.AnnData:
    """In-memory display aliases made solely from stored shared-space fields."""
    b = a.copy()
    b.obsm["X_pca"] = np.asarray(b.obsm["X_scVI_shared"])
    b.obs["lead_re"] = -b.obs["op_basin_index"].to_numpy(float)
    b.obs["lead_im"] = np.zeros(b.n_obs)
    b.obs["op_behaviour"] = b.obs["op_behaviour_basin"].astype(str).to_numpy()
    b.obs["op_confidence"] = np.ones(b.n_obs)
    b.obs["op_rank"] = b.obs.get("op_tangent_r", pd.Series(15, index=b.obs.index)).to_numpy()
    return b


def load_energy(tissue: str) -> EnergyDriftAdapter:
    ckpt = torch.load(AGING / "models" / f"{tissue}_shared_drift.pt", map_location="cpu", weights_only=False)
    config = EnergyDriftConfig(**dict(ckpt["config"]))
    config.device = "cpu"
    drift = EnergyDriftAdapter(dim=int(ckpt["dim"]), config=config)
    model = _EnergyMLP(drift.dim, config.hidden, config.depth).to("cpu")
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    drift._model = model; drift._device = "cpu"; drift._fitted = True
    return drift


def nearest_type(a: ad.AnnData, cell_type: str) -> str:
    z = np.asarray(a.obsm["X_scVI_shared"], float)
    types = a.obs["cell_type"].astype(str).to_numpy()
    centre = z[types == cell_type].mean(0)
    options = [(float(np.linalg.norm(z[types == other].mean(0) - centre)), other)
               for other in sorted(set(types)) if other != cell_type]
    return min(options)[1]


def fig_table(frame: pd.DataFrame, title: str, cols: list[str], formats: dict[str, str] | None = None,
              wrap_first: bool = False):
    formats = formats or {}
    display = frame.loc[:, cols].copy()
    for col, fmt in formats.items():
        if col in display:
            display[col] = display[col].map(lambda x: "—" if pd.isna(x) else format(x, fmt))
    if wrap_first and cols:
        display[cols[0]] = display[cols[0]].astype(str).map(lambda value: textwrap.fill(value, width=20))
    fig, ax = plt.subplots(figsize=(max(11, 1.35 * len(cols) + (2.5 if wrap_first else 0)),
                                   max(2.8, 0.62 * len(display) + 1.8)))
    ax.axis("off")
    ax.set_title(title, fontsize=10, pad=10)
    tab = ax.table(cellText=display.astype(str).values, colLabels=display.columns,
                   loc="center", cellLoc="center")
    tab.auto_set_font_size(False); tab.set_fontsize(6)
    tab.scale(1, 1.8 if wrap_first else 1.35)
    if wrap_first:
        for (_row, col), cell in tab.get_celld().items():
            if col == 0:
                cell.get_text().set_ha("left")
                cell.set_width(cell.get_width() * 1.75)
    return fig


def curvature_ratio(a: ad.AnnData, tissue: str, age: str):
    z = np.asarray(a.obsm["X_scVI_shared"], float)
    J = np.asarray(a.uns["scjdo2"]["J_latent"], float)
    types = a.obs["cell_type"].astype(str).to_numpy()
    rows = []
    for cell_type in sorted(set(types)):
        m = types == cell_type
        mu = z[m].mean(0); pool = np.flatnonzero(m)
        idx = pool[np.argsort(np.linalg.norm(z[pool] - mu, axis=1))[:min(30, len(pool))]]
        learned = np.linalg.eigvalsh(0.5 * (J[idx].mean(0) + J[idx].mean(0).T))
        oracle = np.linalg.eigvalsh(-np.linalg.inv(np.cov(z[m], rowvar=False) + 1e-4 * np.eye(z.shape[1])))
        neg_l = np.sort(np.abs(learned[learned < 0])); neg_o = np.sort(np.abs(oracle[oracle < 0]))
        n = min(len(neg_l), len(neg_o)); rows.append((cell_type, float(np.median(neg_l[-n:] / neg_o[-n:]))) if n else (cell_type, np.nan))
    names, ratios = zip(*rows)
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(names)), 4.2))
    ax.bar(range(len(names)), ratios, color="#378ADD")
    ax.axhspan(1 / 3, 3, color="#378ADD", alpha=0.10, label="1/3–3 agreement band")
    ax.axhline(1, color="0.3", lw=0.7); ax.set_yscale("log")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("median |learned curvature| / |−Σ⁻¹|")
    ax.set_title(f"Gaussian-oracle curvature ratio — {tissue}, {age}"); ax.legend(fontsize=6)
    return fig


def simple_qc(a: ad.AnnData, tissue: str, age: str, kind: str):
    z = np.asarray(a.obsm["X_scVI_shared"], float); J = np.asarray(a.uns["scjdo2"]["J_latent"], float)
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    if kind == "sigma":
        rows = []
        for ct in sorted(a.obs["cell_type"].astype(str).unique()):
            m = a.obs["cell_type"].astype(str).to_numpy() == ct
            rows.append((ct, float(np.sqrt(np.mean(np.var(z[m], axis=0))))))
        names, vals = zip(*rows); scale = float(a.uns["a02_drift"]["lead_eig_tolerance"])
        ax.bar(range(len(names)), vals, color="#378ADD"); ax.axhline(scale, color="#D85A30", ls="--", label="stored lead-eig scale")
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("within-type latent σ"); ax.legend(fontsize=6)
        ax.set_title("within-cluster σ vs drift scale (ESM has no fit σ)")
    elif kind == "symmetry":
        asym = np.linalg.norm(J - J.transpose(0, 2, 1), axis=(1, 2)) / (np.linalg.norm(J, axis=(1, 2)) + 1e-12)
        ax.hist(asym, bins=30, color="#378ADD", edgecolor="white"); ax.set_xlabel("‖J − Jᵀ‖F / ‖J‖F"); ax.set_ylabel("cells")
        ax.set_title("stored Jacobian symmetry check")
    elif kind == "rank":
        vals = a.obs["op_tangent_r"].to_numpy()
        ax.hist(vals, bins=np.arange(vals.min() - .5, vals.max() + 1.5), color="#378ADD", edgecolor="white")
        ax.set_xlabel("tangent rank"); ax.set_ylabel("cells"); ax.set_title("tangent-rank distribution")
    elif kind == "gap":
        nn = NearestNeighbors(n_neighbors=min(21, a.n_obs)).fit(z)
        d, _ = nn.kneighbors(z); low = d[:, -1] >= np.quantile(d[:, -1], .9)
        types = a.obs["cell_type"].astype(str).to_numpy(); centres = np.zeros(a.n_obs, bool)
        for ct in set(types):
            idx = np.flatnonzero(types == ct); centre = z[idx].mean(0)
            centres[idx[np.argsort(np.linalg.norm(z[idx] - centre, axis=1))[:min(30, len(idx))]]] = True
        beh = a.obs["op_behaviour_basin"].astype(str).to_numpy(); cats = ["holder", "fork", "drift"]
        x = np.arange(len(cats)); width = .36
        ax.bar(x - width/2, [np.sum(beh[low] == c) for c in cats], width, label="lowest-density decile")
        ax.bar(x + width/2, [np.sum(beh[centres] == c) for c in cats], width, label="cluster centres")
        ax.set_xticks(x); ax.set_xticklabels(cats); ax.set_ylabel("stored-cell count"); ax.legend(fontsize=6)
        ax.set_title("inter-cluster gap control")
    return fig


def donor_counts(tissue: str, age: str):
    before = pd.read_csv(AGING / "qc" / f"{tissue}_donor_type_counts_before.csv")
    after = pd.read_csv(AGING / "qc" / f"{tissue}_donor_type_counts_after.csv")
    before, after = before[before.age_group == age], after[after.age_group == age]
    types = sorted(set(before.cell_type) | set(after.cell_type)); donors = sorted(set(before.donor_id) | set(after.donor_id))
    def mat(df, val): return df.pivot_table(index="donor_id", columns="cell_type", values=val, aggfunc="sum", fill_value=0).reindex(index=donors, columns=types, fill_value=0)
    b, a = mat(before, "n_cells_before"), mat(after, "n_cells_after")
    fig, axs = plt.subplots(1, 2, figsize=(10, max(3, .28 * len(donors) + 2)), sharey=True)
    for ax, data, title in zip(axs, (b, a), ("before balancing", "after balancing")):
        im = ax.imshow(data, aspect="auto", cmap="Blues"); ax.set_title(title); ax.set_xticks(range(len(types))); ax.set_xticklabels(types, rotation=60, ha="right", fontsize=5)
        ax.set_yticks(range(len(donors))); ax.set_yticklabels(donors, fontsize=5); fig.colorbar(im, ax=ax, shrink=.65)
    fig.suptitle(f"donor × type cell counts — {tissue}, {age}")
    return fig


def age_slice(a: ad.AnnData, age: str) -> ad.AnnData:
    """Slice stored cells while carrying the correspondingly stored Jacobians."""
    mask = a.obs["age_group"].astype(str).to_numpy() == age
    b = a[mask].copy()
    for key in ("J", "J_latent"):
        if key in a.uns["scjdo2"]:
            b.uns["scjdo2"][key] = np.asarray(a.uns["scjdo2"][key])[mask]
    return b


def basin_confusion(shared: ad.AnnData, standalone: ad.AnnData, tissue: str, age: str):
    common = shared.obs_names.intersection(standalone.obs_names)
    s = shared[common].obs["op_behaviour_basin"].astype(str).eq("holder").to_numpy()
    o = standalone[common].obs["op_behaviour_basin"].astype(str).eq("holder").to_numpy()
    cm = np.array([[np.sum(~s & ~o), np.sum(~s & o)], [np.sum(s & ~o), np.sum(s & o)]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8)); im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2): ax.text(j, i, str(cm[i,j]), ha="center", va="center")
    ax.set_xticks([0,1], ["standalone non-basin", "standalone basin"], rotation=25, ha="right", fontsize=7)
    ax.set_yticks([0,1], ["shared non-basin", "shared basin"], fontsize=7); ax.set_title(f"shared vs standalone basin agreement — {tissue}, {age}")
    fig.colorbar(im, ax=ax, shrink=.75); return fig


def rerender_revision1() -> int:
    """Re-render only the six A04 Revision 1 targets; never refit."""
    verify_inputs()
    MANIFEST.clear()
    pooled = pd.read_csv(AGING / "compare_pooled.csv").merge(
        pd.read_csv(AGING / "aging_modes.csv"), on=["tissue", "cell_type"], how="left")
    vector_frame = pooled.rename(columns={
        "median_abs_lambda_young": "young_median_abs_lambda",
        "median_abs_lambda_old": "old_median_abs_lambda",
        "anisotropy_young": "young_anisotropy",
        "anisotropy_old": "old_anisotropy", "mode_label": "mode"})
    save(plot.aging_vectors(vector_frame), "comparison/aging_vectors", "§7.5 aging_vectors", "both", "young_old",
         ["benchmarks/aging/compare_pooled.csv", "benchmarks/aging/aging_modes.csv"])
    for tissue in ("lung", "marrow"):
        comp = pd.read_csv(AGING / f"{tissue}_compare_shared.csv").merge(
            pd.read_csv(AGING / "aging_modes.csv"), on=["tissue", "cell_type"], how="left")
        cs = comp.copy()
        cs["lambda_ratio_soft_axis"] = np.where(cs.basin_fraction_young >= .5, cs.lambda_ratio_soft_axis, np.nan)
        cs["λ ratio"] = cs.lambda_ratio_soft_axis.map(lambda x: "—" if pd.isna(x) else f"{x:.3g}")
        cs["iso pass"] = cs.iso_pass.map({True:"pass",False:"fail"})
        cs["ant pass"] = cs.ant_pass.map({True:"pass",False:"fail"})
        save(fig_table(cs.sort_values("displacement", ascending=False),
                       f"compare summary — {tissue}\nλ ratio shown only when young holder fraction ≥ 0.5",
                       ["cell_type","iso pass","ant pass","S","cos_abs_u_young_d","λ ratio","mode_label","n_young","n_old"],
                       {"S":".3f","cos_abs_u_young_d":".3f"}, wrap_first=True),
             f"{tissue}/compare_summary", "§7b compare_summary", tissue, "young_old",
             [f"benchmarks/aging/{tissue}_compare_shared.csv", "benchmarks/aging/aging_modes.csv"])
        a = view(ad.read_h5ad(AGING / f"{tissue}_shared.h5ad"))
        source = [f"benchmarks/aging/{tissue}_shared.h5ad"]
        drift = load_energy(tissue)
        targets = ["fibroblast of lung", "natural killer cell"] if tissue == "lung" else ["granulocyte"]
        for cell_type in targets:
            save(plot.energy_transect(a, cell_type, nearest_type(a, cell_type), drift, use_rep="X_scVI_shared"),
                 f"{tissue}/energy_transect_{cell_type.replace(' ','_')}", "§7.6 energy_transect", tissue, "young_old",
                 source + [f"benchmarks/aging/models/{tissue}_shared_drift.pt"])
    expected = {"comparison/aging_vectors", "lung/compare_summary", "marrow/compare_summary",
                "lung/energy_transect_fibroblast_of_lung", "lung/energy_transect_natural_killer_cell",
                "marrow/energy_transect_granulocyte"}
    actual = {str((ROOT / row["path"]).with_suffix("").relative_to(OUT)) for row in MANIFEST}
    if actual != expected:
        raise RuntimeError(f"revision renderer wrote unexpected targets: {actual ^ expected}")
    # Manifest columns contain paths and immutable source provenance; rewrite after validation.
    existing = pd.read_csv(OUT / "MANIFEST.csv")
    existing_bases = {str((ROOT / path).with_suffix("").relative_to(OUT))
                      for path in existing.path if path.endswith((".pdf", ".png"))}
    if not expected.issubset(existing_bases):
        raise RuntimeError("revision targets absent from existing manifest")
    existing.to_csv(OUT / "MANIFEST.csv", index=False)
    print(json.dumps({"revision": 1, "figure_pairs_rerendered": len(expected), "n_skipped": 3}, indent=2))
    return 0

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--affected-only", action="store_true", help="render only the A04 Revision 1 targets")
    args = parser.parse_args()
    if args.affected_only:
        return rerender_revision1()
    verify_inputs()
    pooled = pd.read_csv(AGING / "compare_pooled.csv").merge(pd.read_csv(AGING / "aging_modes.csv"), on=["tissue", "cell_type"], how="left")
    fig = plot.aging_vectors(pooled.rename(columns={"median_abs_lambda_young":"young_median_abs_lambda", "median_abs_lambda_old":"old_median_abs_lambda", "anisotropy_young":"young_anisotropy", "anisotropy_old":"old_anisotropy", "mode_label":"mode"}))
    save(fig, "comparison/aging_vectors", "§7.5 aging_vectors", "both", "young_old", ["benchmarks/aging/compare_pooled.csv", "benchmarks/aging/aging_modes.csv"])
    for tissue in ("lung", "marrow"):
        shared = ad.read_h5ad(AGING / f"{tissue}_shared.h5ad")
        a = view(shared); source = [f"benchmarks/aging/{tissue}_shared.h5ad"]
        # §7 across-age items.
        save(plot.spectrum_ladder(a, by="cell_type", groups=("young","old"), use_rep="X_scVI_shared"), f"{tissue}/spectrum_ladder_young_old", "§7.1 spectrum_ladder groups", tissue, "young_old", source)
        save(plot.soft_axis_map(a, by="cell_type", use_rep="X_scVI_shared", groups=("young","old"), displacement_arrow=True), f"{tissue}/soft_axis_map_young_old", "§7.2 soft_axis_map displacement", tissue, "young_old", source)
        # Faceted per-age basin-index strips constructed directly from stored per-cell basin_index.
        fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, age in zip(axs, ("young", "old")):
            sub = a[a.obs.age_group.astype(str) == age]
            types = sorted(sub.obs.cell_type.astype(str).unique()); rng = np.random.default_rng(0)
            for i, ct in enumerate(types):
                vals = sub.obs.loc[sub.obs.cell_type.astype(str) == ct, "op_basin_index"].to_numpy(float)
                ax.scatter(i + rng.uniform(-.25,.25,len(vals)), vals, s=3, alpha=.25, color="#378ADD")
                ax.plot([i-.3,i+.3],[np.median(vals)]*2,color="black",lw=1)
            ax.axhline(0,color="0.4",lw=.5); ax.set_title(age); ax.set_xticks(range(len(types))); ax.set_xticklabels(types,rotation=35,ha="right",fontsize=7)
        axs[0].set_ylabel("basin_index = −lead_re (relative)"); fig.suptitle(f"basin index by type, faceted age — {tissue}")
        save(fig, f"{tissue}/basin_index_age_facet", "§7.3 basin_index_by_type", tissue, "young_old", source)
        for age in ("young", "old"):
            age_a = age_slice(a, age); age_source = source + [f"benchmarks/aging/qc/{tissue}_donor_type_counts_before.csv", f"benchmarks/aging/qc/{tissue}_donor_type_counts_after.csv"]
            save(plot.ledger_map(age_a, use_rep="X_scVI_shared"), f"{tissue}/{age}/ledger_map", "§7.4 ledger_map", tissue, age, source)
            # §7b existing figures and single-group views.
            try: fig = plot.figure_operator_space(age_a, use_rep="X_scVI_shared", by="basin_index")
            except Exception as err: raise RuntimeError(f"figure_operator_space {tissue}/{age}: {err}")
            save(fig, f"{tissue}/{age}/operator_space", "§7b figure_operator_space", tissue, age, source)
            save(plot.figure_ledger_summary(age_a), f"{tissue}/{age}/ledger_summary", "§7b figure_ledger_summary", tissue, age, source)
            save(plot.spectrum_ladder(age_a, by="cell_type", use_rep="X_scVI_shared"), f"{tissue}/{age}/spectrum_ladder", "§7b spectrum_ladder single group", tissue, age, source)
            save(plot.soft_axis_map(age_a, by="cell_type", use_rep="X_scVI_shared"), f"{tissue}/{age}/soft_axis_map", "§7b soft_axis_map", tissue, age, source)
            save(plot.basin_index_by_type(age_a, by="cell_type"), f"{tissue}/{age}/basin_index", "§7b basin_index_by_type single group", tissue, age, source)
            save(curvature_ratio(age_a, tissue, age), f"{tissue}/{age}/gaussian_oracle_curvature_ratio", "§7b Gaussian-oracle curvature ratio", tissue, age, source)
            # The final ELBO is stored, epoch curves are not; panel records the available QC honestly.
            scvi = pd.read_csv(AGING / "qc" / f"{tissue}_scvi_metrics.csv")
            fig, ax = plt.subplots(figsize=(6.2,4)); ax.plot(scvi.fit, scvi.best_val_elbo, marker="o", color="#378ADD"); ax.set_ylabel("stored best validation ELBO"); ax.set_xlabel("fit"); ax.tick_params(axis="x", rotation=25); ax.set_title("final ELBO summaries (epoch curves unavailable)")
            save(fig, f"{tissue}/{age}/qc_scvi_elbo_summary", "§7b scVI ELBO QC", tissue, age, [f"benchmarks/aging/qc/{tissue}_scvi_metrics.csv"])
            fig, ax = plt.subplots(figsize=(5.8,3.8)); ax.axis("off"); ax.text(.5,.58,"d_latent = 15",ha="center",fontsize=20,fontweight="bold"); ax.text(.5,.36,"Frozen A02 shared-space dimension; no selection rerun in A04.",ha="center",fontsize=8); ax.set_title("latent-dimension QC")
            save(fig, f"{tissue}/{age}/qc_d_latent", "§7b d_latent choice", tissue, age, source)
            for kind, item in (("sigma","within-cluster σ versus drift scale"),("symmetry","symmetry check histogram"),("rank","tangent-rank distribution"),("gap","inter-cluster gap control")):
                save(simple_qc(age_a,tissue,age,kind), f"{tissue}/{age}/qc_{kind}", f"§7b {item}", tissue, age, source)
            save(donor_counts(tissue, age), f"{tissue}/{age}/qc_donor_type_counts", "§7b donor × type counts", tissue, age, age_source)
            standalone = ad.read_h5ad(AGING / f"{tissue}_standalone_{age}.h5ad")
            save(basin_confusion(age_a, standalone, tissue, age), f"{tissue}/{age}/qc_shared_standalone_basin_confusion", "§7b standalone-versus-shared basin agreement", tissue, age, source + [f"benchmarks/aging/{tissue}_standalone_{age}.h5ad"])
            comp = pd.read_csv(AGING / f"{tissue}_compare_shared.csv")
            save(fig_table(comp.assign(correction_artifact=comp.correction_artifact.map({True:"yes",False:"no"})), f"correction-artifact list — {tissue}, {age}", ["cell_type","correction_artifact"], {}), f"{tissue}/{age}/qc_correction_artifacts", "§7b correction_artifact list", tissue, age, [f"benchmarks/aging/{tissue}_compare_shared.csv"])
        # §7b across-age summary and shared/standalone quantity control.
        comp = pd.read_csv(AGING / f"{tissue}_compare_shared.csv").merge(pd.read_csv(AGING / "aging_modes.csv"), on=["tissue","cell_type"], how="left")
        cs = comp.copy(); cs["lambda_ratio_soft_axis"] = np.where(cs.basin_fraction_young >= .5, cs.lambda_ratio_soft_axis, np.nan)
        cs["λ ratio"] = cs.lambda_ratio_soft_axis.map(lambda x: "—" if pd.isna(x) else f"{x:.3g}")
        cs["iso pass"] = cs.iso_pass.map({True:"pass",False:"fail"}); cs["ant pass"] = cs.ant_pass.map({True:"pass",False:"fail"})
        save(fig_table(cs.sort_values("displacement", ascending=False), f"compare summary — {tissue}\nλ ratio shown only when young holder fraction ≥ 0.5", ["cell_type","iso pass","ant pass","S","cos_abs_u_young_d","λ ratio","mode_label","n_young","n_old"], {"S":".3f","cos_abs_u_young_d":".3f"}, wrap_first=True), f"{tissue}/compare_summary", "§7b compare_summary", tissue, "young_old", [f"benchmarks/aging/{tissue}_compare_shared.csv", "benchmarks/aging/aging_modes.csv"])
        stand = pd.read_csv(AGING / f"{tissue}_compare_standalone.csv")
        metrics = ["lead_eig","median_abs_lambda","anisotropy","n_soft","basin_fraction","basin_index"]
        fig, axs = plt.subplots(2,3,figsize=(11,6));
        for ax, metric in zip(axs.ravel(),metrics):
            ax.scatter(stand[f"shared_{metric}"], stand[metric], c=stand.age_group.map({"young":"#378ADD","old":"#D85A30"}), s=28)
            lims = np.nanpercentile(np.r_[stand[f"shared_{metric}"], stand[metric]], [2,98]); ax.plot(lims,lims,color="0.4",lw=.6); ax.set_xlabel("shared"); ax.set_ylabel("standalone"); ax.set_title(metric,fontsize=8)
        fig.suptitle(f"standalone versus shared per-type quantities — {tissue}")
        save(fig, f"{tissue}/standalone_shared_quantities", "§7b standalone-versus-shared quantities", tissue, "young_old", [f"benchmarks/aging/{tissue}_compare_standalone.csv"])
        # Required stored-energy transects only for the three locked basin types.
        targets = ["fibroblast of lung", "natural killer cell"] if tissue == "lung" else ["granulocyte"]
        drift = load_energy(tissue)
        for ct in targets:
            nb = nearest_type(a, ct)
            save(plot.energy_transect(a, ct, nb, drift, use_rep="X_scVI_shared"), f"{tissue}/energy_transect_{ct.replace(' ','_')}", "§7.6 energy_transect", tissue, "young_old", source + [f"benchmarks/aging/models/{tissue}_shared_drift.pt"])
    deviations = """# A04 rendering deviations\n\n* `figure_archetype_sheet` was not rendered: no stored archetype weights exist, and A04 prohibits the requested semi-NMF fitting.\n* The ten-refit `lead_re` permutation-null panel was not rendered: no such refits are stored, and recreating them would violate the no-refit instruction.\n* Per-epoch scVI ELBO curves were not stored by A02. The rendered QC panels show the stored best validation ELBO summaries instead.\n"""
    (OUT / "DEVIATIONS.md").write_text(deviations)
    note_skip("§7b figure_archetype_sheet", "no stored archetype weights; A04 prohibits semi-NMF fitting")
    note_skip("§7b permutation null of lead_re", "no stored ten-refit permutation null; A04 prohibits refitting")
    note_skip("§7b scVI ELBO curves", "no stored epoch histories; final-ELBO summary rendered in place")
    pd.DataFrame(MANIFEST, columns=["path","spec_item","tissue","age","source_files"]).to_csv(OUT / "MANIFEST.csv", index=False)
    print(json.dumps({"n_figure_pairs": len({Path(x['path']).with_suffix('') for x in MANIFEST if x['path'].endswith(('.pdf','.png'))}), "n_skipped": 3, "manifest": str(OUT / 'MANIFEST.csv')}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
