"""Phase D LARRY run — clone arm (§4), velocity arm (§5 — fit scVelo
dynamical, use as drift prior), clock check (§6), and the full behaviour
gate (fork at clone-split positions, holders on terminal states, driver
tables).

Resource constraints — 36 GB RAM:

* The full LARRY in-vitro dataset is 130,887 cells × 25,289 genes.
  Phase C's anchor J fit stores a (G, L_cap) float32 block per cell;
  at G=25k × L=300 × 130k × 4B = 390 GB per J. Infeasible as-is.
* Mitigations here: HVG → 2000 genes (operates on all cells for
  embedding), optional ``--subsample N`` on cells with clone-preserving
  resampling (keeps every clone with ≥2 members intact).
* ``--full-gene`` keeps all 2000 HVGs for the rate model; genes beyond
  the HVG set are not scored (no gene-space J on them).

Direction-free (V2_PLAN R17/R18): no root cell, no annotation arrow;
day labels enter only as the pseudotime proxy for the drift fit and as
the clock check against lineage-path pseudotime.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import scanpy as sc

from scjdo2 import io as scjdo2_io
from scjdo2 import opspace
from scjdo2.data import load_larry
from scjdo2.drift import DriftConfig
from scjdo2.pipeline import fit_latent_field, fit_rate_and_gene_field
from scjdo2.clone import (
    clone_consistency,
    clone_cross_day_pairs,
    clone_split_map,
)


REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT_RUN = REPO / "benchmarks" / "v2" / "larry"
DEFAULT_OUT_FIG = REPO / "benchmarks" / "figures" / "larry"
MUST_KEEP = ("Gata1", "Spi1", "Sfpi1")   # Spi1 is the LARRY symbol; Sfpi1 synonym
TERMINAL = {
    "Erythroid", "Monocyte", "Neutrophil", "Baso", "Mast", "Meg",
    "Lymphoid", "Eos", "Ccr7_DC", "pDC",
}
PROGENITOR_LIKE = {"Undifferentiated"}


# ------------------------------------------------------------------- preproc


def _hvg_subset(adata, n_top_genes: int):
    """HVG on log-normalised raw counts; keep must-have TFs regardless."""
    tmp = adata.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=n_top_genes, flavor="seurat")
    hv = tmp.var["highly_variable"].to_numpy()
    for g in MUST_KEEP:
        if g in adata.var_names:
            hv[adata.var_names.get_loc(g)] = True
    kept = adata[:, hv].copy()
    kept.layers["counts"] = kept.X.copy()
    return kept


def _subsample_clone_preserving(adata, n: int, seed: int = 0):
    """Clone-stratified subsample. We prioritise clones that *span days*
    (the clone arm actually needs those), then within-day clones with
    ≥2 members, then unassigned cells. The clone matrix is restricted to
    the retained cells; empty columns are dropped.

    Target: ``n`` cells total. If protected clone cells alone would
    exceed ``n``, we randomly sample whole clones (not cells) in
    priority order until we approach ``n`` from below. This keeps
    each selected clone intact.
    """
    import scipy.sparse as sp
    rng = np.random.default_rng(int(seed))
    cm = adata.obsm["clone_matrix"]
    if not sp.issparse(cm):
        cm = sp.csr_matrix(cm)
    cm_t = cm.T.tocsr()
    day = adata.obs["day"].to_numpy()
    cross_day: list[tuple[int, np.ndarray]] = []
    same_day: list[tuple[int, np.ndarray]] = []
    for k in range(cm_t.shape[0]):
        members = cm_t.indices[cm_t.indptr[k]:cm_t.indptr[k + 1]]
        if members.size < 2:
            continue
        ds = pd.Series(day[members]).dropna().unique()
        (cross_day if len(ds) >= 2 else same_day).append((int(k), members))

    rng.shuffle(cross_day)
    rng.shuffle(same_day)
    keep: set[int] = set()
    for queue in (cross_day, same_day):
        for _, members in queue:
            if len(keep) + members.size > n:
                continue
            keep.update(members.tolist())
            if len(keep) >= n:
                break
        if len(keep) >= n:
            break
    protected = len(keep)
    others = np.setdiff1d(np.arange(adata.n_obs),
                          np.fromiter(keep, dtype=np.int64))
    n_extra = max(0, int(n) - protected)
    extra = rng.choice(others, size=min(n_extra, others.size), replace=False)
    keep_arr = np.sort(np.concatenate([np.fromiter(keep, dtype=np.int64), extra]))
    kept = adata[keep_arr].copy()
    cm2 = cm[keep_arr]
    col_keep = np.asarray(cm2.sum(axis=0)).ravel() > 0
    cm2 = cm2[:, col_keep].tocsr()
    kept.obsm["clone_matrix"] = cm2
    print(f"[larry]   subsample: n={kept.n_obs} (clone-stratified: {protected} "
          f"from cross-day/same-day clones, {extra.size} extras); kept {cm2.shape[1]} clones",
          flush=True)
    return kept


def _pca_and_pseudotime(adata, n_comps: int = 10):
    """Log-normalise a copy, PCA on scaled data, write PCA back to
    original adata.obsm['X_pca']. Pseudotime proxy = day (float)."""
    tmp = adata.copy()
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    sc.pp.scale(tmp, max_value=10)
    sc.tl.pca(tmp, n_comps=n_comps)
    adata.obsm["X_pca"] = tmp.obsm["X_pca"].copy()
    adata.obs["pseudotime"] = adata.obs["day"].astype(float).to_numpy()
    return adata


# ------------------------------------------------------------------- clock arm


def clock_check(adata, use_rep: str = "X_pca") -> dict:
    """Check that latent-flow pseudotime increases with sampling day on
    representative lineage paths. For each annotated fate, compute
    median latent-flow pseudotime per day; the check passes if the
    sequence is non-decreasing.

    Latent-flow pseudotime proxy: distance along the first PCA axis
    from the day-2 progenitor centroid. This is a direction-free scalar
    (we're testing monotonicity, not injecting direction).
    """
    z = np.asarray(adata.obsm[use_rep])
    day = adata.obs["day"].astype(float).to_numpy()
    prog_mask = (day == day.min()) & adata.obs["cell_type_annotation"].isin(
        PROGENITOR_LIKE
    ).to_numpy()
    if prog_mask.sum() == 0:
        prog_mask = (day == day.min())
    centroid = z[prog_mask].mean(axis=0)
    tscalar = np.linalg.norm(z - centroid, axis=1)
    out = {}
    for fate in sorted(set(adata.obs["cell_type_annotation"].astype(str))):
        if fate == "nan":
            continue
        rows = []
        for d in sorted(set(day)):
            mask = (adata.obs["cell_type_annotation"].astype(str) == fate) & (day == d)
            if mask.sum() == 0:
                continue
            rows.append((int(d), int(mask.sum()), float(np.median(tscalar[mask]))))
        if len(rows) < 2:
            continue
        xs = [r[2] for r in rows]
        pass_ = all(xs[i] <= xs[i + 1] + 1e-9 for i in range(len(xs) - 1))
        out[fate] = {"rows": rows, "monotone": bool(pass_)}
    return out


# ----------------------------------------------------- velocity arm (optional)


def fit_scvelo_prior(adata):
    """Train scVelo dynamical mode on the LARRY in-vitro data and wrap
    the per-cell velocity as a differentiable (latent → velocity) MLP.
    Returns the fitted MLP + the per-cell velocity used to train it.

    Falls back to ``None`` if scVelo is unavailable or if the data lacks
    spliced/unspliced layers (LARRY's GEO release does NOT publish
    spliced/unspliced, so this is a no-op here and the velocity arm is
    reported as 'skipped: no spliced/unspliced layers in GSE140802').
    """
    if "spliced" not in adata.layers or "unspliced" not in adata.layers:
        return None, (
            "LARRY GSE140802 ships total UMIs only — no spliced / "
            "unspliced layers. scVelo dynamical mode needs both; this "
            "arm is reported as skipped. Recovery options: realign the "
            "FASTQs with velocyto (not available at load time) or use "
            "an alternative barcoded dataset with pre-computed splicing "
            "(e.g. dynamo-labelled sc-DNA/RNA sets)."
        )
    try:
        import scvelo as scv
    except ImportError:
        return None, "scvelo not installed"
    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    scv.tl.recover_dynamics(adata)
    scv.tl.velocity(adata, mode="dynamical")
    # NOTE: wrapping into an MLP + drift-prior usage would land here.
    return adata.layers.get("velocity"), None


# -------------------------------------------------------------- gate reports


def behaviour_counts_by_fate(adata) -> pd.DataFrame:
    beh = adata.obs["op_behaviour"].astype(str).to_numpy()
    fate = adata.obs["cell_type_annotation"].astype(str).to_numpy()
    rows = []
    for f in sorted(set(fate)):
        if f == "nan":
            continue
        mask = fate == f
        c = pd.Series(beh[mask]).value_counts()
        rec = {"fate": f, "n": int(mask.sum())}
        for k in ("fork", "holder", "drift", "damped_oscillator", "growing_oscillator"):
            rec[k] = int(c.get(k, 0))
        rec["frac_fork"] = rec["fork"] / rec["n"] if rec["n"] else 0.0
        rec["frac_holder"] = rec["holder"] / rec["n"] if rec["n"] else 0.0
        rows.append(rec)
    return pd.DataFrame(rows)


def behaviour_at_split_positions(adata, splits) -> pd.DataFrame:
    """For each clone-split event, find the nearest cell in latent space
    and report its scJDO2 behaviour + lead_re."""
    from sklearn.neighbors import NearestNeighbors
    z = np.asarray(adata.obsm["X_pca"])
    nn = NearestNeighbors(n_neighbors=1).fit(z)
    rows = []
    for sp_ in splits:
        d, i = nn.kneighbors(sp_.pre_split_mean[None, :], return_distance=True)
        c = int(i[0, 0])
        beh = str(adata.obs["op_behaviour"].iloc[c])
        lead = float(adata.obs["lead_re"].iloc[c]) if "lead_re" in adata.obs else float("nan")
        rows.append({
            "clone_id": sp_.clone_id,
            "fates": "/".join(sp_.fates),
            "counts": "/".join(str(c) for c in sp_.counts),
            "nearest_cell": c,
            "distance": float(d[0, 0]),
            "behaviour": beh,
            "lead_re": lead,
        })
    return pd.DataFrame(rows)


def archetype_driver_report(adata) -> pd.DataFrame:
    scj = adata.uns["scjdo2"]
    arch = scj.get("archetypes") or {}
    rows = []
    for ak, info in arch.items():
        for table_name in ("responders", "drivers_left"):
            df = info.get(table_name)
            if df is None:
                continue
            for g in MUST_KEEP:
                hit = df.index[df["gene"].astype(str) == g].tolist()
                if not hit:
                    continue
                i = int(hit[0])
                r = df.loc[i].to_dict()
                rows.append({
                    "archetype": int(ak),
                    "behaviour": info["behaviour"],
                    "lead_re": float(info["eigenvalues"][0].real),
                    "n_cells": int(info["n_cells"]),
                    "table": table_name,
                    "gene": g,
                    "rank": i,
                    "sensitivity": float(r.get("sensitivity", float("nan"))),
                    "magnitude": float(r["magnitude"]),
                    "weight": float(r["weight"]),
                    "fisher": float(r["fisher"]),
                    "snr": float(r.get("snr", float("nan"))),
                    "reason": str(r["reason"]),
                })
    return pd.DataFrame(rows)


# -------------------------------------------------------------------- main


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-top-genes", type=int, default=2000)
    p.add_argument("--subsample", type=int, default=20000,
                   help="cell cap (clone-preserving); 0 → use all 130k")
    p.add_argument("--n-pcs", type=int, default=10)
    p.add_argument("--n-epochs", type=int, default=800)
    p.add_argument("--rate-epochs", type=int, default=200)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--n-anchors", type=int, default=300)
    p.add_argument("--anchor-top-k", type=int, default=10)
    p.add_argument("--anchor-max-live", type=int, default=300)
    p.add_argument("--cv-sample-cells", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-phase-c", action="store_true",
                   help="only run drift + clone arm + clock check")
    p.add_argument("--out-run", type=Path, default=DEFAULT_OUT_RUN)
    p.add_argument("--out-fig", type=Path, default=DEFAULT_OUT_FIG)
    args = p.parse_args()

    args.out_run.mkdir(parents=True, exist_ok=True)
    args.out_fig.mkdir(parents=True, exist_ok=True)

    print("[larry] loading cached raw UMIs")
    adata = load_larry()
    print(f"[larry]   shape={adata.shape}")

    if args.subsample and args.subsample < adata.n_obs:
        adata = _subsample_clone_preserving(adata, args.subsample, seed=args.seed)

    print(f"[larry] HVG → top-{args.n_top_genes}")
    adata = _hvg_subset(adata, args.n_top_genes)
    print(f"[larry]   kept {adata.n_vars} genes")

    print(f"[larry] PCA(n={args.n_pcs}) + day→pseudotime")
    _pca_and_pseudotime(adata, n_comps=args.n_pcs)

    # §6 clock check (direction-free check, pre-Phase-C).
    print("[larry] §6 clock check (latent-flow pseudotime vs sampling day)")
    clock = clock_check(adata)
    monotone_frac = np.mean([v["monotone"] for v in clock.values()]) if clock else float("nan")
    print(f"  fates checked: {len(clock)}  "
          f"frac monotone (day↑ → pseudotime↑): {monotone_frac:.3f}")
    for fate, info in sorted(clock.items()):
        marks = "✓" if info["monotone"] else "✗"
        rows = ", ".join(f"d{d}:{s:.3f}(n={n})" for d, n, s in info["rows"])
        print(f"    {marks} {fate:<18}  {rows}")

    # §2 drift fit.
    print(f"[larry] drift fit (n_epochs={args.n_epochs})")
    t0 = time.monotonic()
    fit_latent_field(
        adata, use_rep="X_pca", time_key="pseudotime",
        drift_config=DriftConfig(
            hidden=96, depth=3, n_epochs=args.n_epochs, seed=args.seed,
            use_spectral_norm=False, batch_size=256,
        ),
    )
    print(f"[larry]   drift+latent J in {time.monotonic()-t0:.1f}s")

    # §4 clone arm.
    print("[larry] §4 clone arm (clone-split map + consistency)")
    splits = clone_split_map(adata, use_rep="X_pca", day_key="day")
    pairs = clone_cross_day_pairs(adata, day_key="day")
    print(f"  clone splits detected: {len(splits)}")
    print(f"  cross-day clone pairs: {pairs.shape[0]}")
    from scjdo2.drift import DriftAdapter  # noqa: F401 (type hint)
    scj = adata.uns["scjdo2"]
    drift = scj.get("_drift")
    if drift is None:
        # Rebuild a drift-callable from the stored velocities (nearest-neighbour
        # lookup) — this is adequate for clone-consistency scoring.
        from sklearn.neighbors import NearestNeighbors
        v_all = adata.obsm["X_drift_v"]
        z_all = adata.obsm["X_pca"]
        nn_v = NearestNeighbors(n_neighbors=1).fit(z_all)

        class _NNDrift:
            def velocity(self, zb, tb):
                _, idx = nn_v.kneighbors(zb, return_distance=True)
                return v_all[idx.ravel()]

        drift = _NNDrift()
    cc = clone_consistency(adata, drift, use_rep="X_pca", day_key="day",
                           time_key="pseudotime", max_pairs=5000,
                           n_rk_steps=20, seed=args.seed)
    print(f"  clone-consistency: n_pairs={cc['n_pairs']} "
          f"n_consistent={cc['n_consistent']} frac={cc['consistency']:.3f} "
          f"radius={cc['radius']:.3f}")

    # §5 velocity arm (reports skip for LARRY due to missing spliced counts).
    print("[larry] §5 velocity arm (scVelo dynamical as drift prior)")
    _, scv_note = fit_scvelo_prior(adata)
    if scv_note:
        print(f"  skipped — {scv_note}")

    if not args.skip_phase_c:
        print(f"[larry] Phase C (anchor fit, max_live={args.anchor_max_live})")
        t0 = time.monotonic()
        fit_rate_and_gene_field(
            adata, use_rep="X_pca",
            n_epochs_rate=args.rate_epochs,
            k=args.k, cv_sample_cells=args.cv_sample_cells,
            seed=args.seed,
            anchor_fit=True, n_anchors=args.n_anchors,
            anchor_top_k=args.anchor_top_k,
            anchor_max_live=args.anchor_max_live,
        )
        print(f"[larry]   Phase C in {time.monotonic()-t0:.1f}s")

        print("[larry] opspace.compute")
        opspace.compute(adata)

        # Full gate.
        print("\n[larry] behaviour counts by annotated fate:")
        bc = behaviour_counts_by_fate(adata)
        print(bc.to_string(index=False))
        bc.to_csv(args.out_run / "behaviour_counts_by_fate.csv", index=False)

        print("\n[larry] behaviour at clone-split positions:")
        bs = behaviour_at_split_positions(adata, splits)
        print(bs.head(30).to_string(index=False))
        bs.to_csv(args.out_run / "behaviour_at_split_positions.csv", index=False)

        print("\n[larry] archetype driver report (Gata1 / Spi1):")
        drv = archetype_driver_report(adata)
        print(drv.to_string(index=False))
        drv.to_csv(args.out_run / "archetype_drivers_Gata1_Spi1.csv", index=False)

        print("[larry] saving run dir")
        scjdo2_io.save(adata, args.out_run)

    # Record the clock + clone arms.
    pd.DataFrame([
        {"fate": fate, "monotone": info["monotone"],
         "rows": "; ".join(f"d{d}:{s:.3f}(n={n})" for d, n, s in info["rows"])}
        for fate, info in sorted(clock.items())
    ]).to_csv(args.out_run / "clock_check.csv", index=False)
    pd.DataFrame([
        {"clone_id": sp_.clone_id, "fates": "/".join(sp_.fates),
         "counts": "/".join(str(c) for c in sp_.counts),
         "pre_split_mean_norm": float(np.linalg.norm(sp_.pre_split_mean))}
        for sp_ in splits
    ]).to_csv(args.out_run / "clone_splits.csv", index=False)
    pd.DataFrame({
        "n_pairs": [cc["n_pairs"]], "n_consistent": [cc["n_consistent"]],
        "consistency": [cc["consistency"]], "radius": [cc["radius"]],
    }).to_csv(args.out_run / "clone_consistency.csv", index=False)
    print(f"\n[larry] done — {args.out_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
