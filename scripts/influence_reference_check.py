"""Do Jacobian influence edges recover a reference TF→target network?

For every reference TF present in the gene set, rank all other genes by
``|M[i, TF](t*)|`` (the change in gene i per small push on the TF) and
score the TF's reference targets with AUROC. Repeated per archetype.

Baselines and null:
* co-expression — rank by ``|Spearman(TF, i)|`` over the archetype's
  member cells (the usual GRN baseline);
* null — the same AUROC with each TF's target set replaced by a random
  gene set of equal size (``--n-perm`` draws), giving a one-sided p for
  the mean AUROC across TFs.
Also reports sign agreement on Activation / Repression edges.

Usage::

    python scripts/influence_reference_check.py benchmarks/v2/paul15 \
        --ref-file trrust_rawdata.mouse.tsv --out benchmarks/v2/paul15/influence_trrust
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from scjdo2 import influence, io, opspace


def _auroc(scores: np.ndarray, pos: np.ndarray) -> float:
    n1 = int(pos.sum())
    n0 = int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ref-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    adata = io.load(args.run_dir)
    opspace.archetype_operators(adata)
    influence.archetype_influence(adata)
    genes = np.asarray(adata.var_names)
    gidx = {g: i for i, g in enumerate(genes)}
    G = genes.size

    ref = pd.read_csv(args.ref_file, sep="\t", header=None).iloc[:, :3]
    ref.columns = ["source", "target", "mode"]
    ref = ref[ref["source"].isin(gidx) & ref["target"].isin(gidx) & (ref["source"] != ref["target"])]
    ref = ref.drop_duplicates(["source", "target"])
    tfs = sorted(ref["source"].unique())

    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    W = np.asarray(adata.obsm["archetype_weights"])
    arch = adata.uns["scjdo2"]["archetypes"]

    per_tf, per_edge, summary, base_pos = [], [], [], []
    for ak, info in sorted(arch.items()):
        B, P = np.asarray(info["block"]), np.asarray(info["projector"])
        t = float(info["influence_t"])
        M = P.T @ (influence._change_operator(B, t) @ P)       # (G, G)
        base_pos.append(float(np.mean(M[~np.eye(G, dtype=bool)] > 0)))
        wk = W[:, ak]
        members = np.where(wk >= 0.5 * wk.max())[0]
        Xm = X[members]
        roles = info["influence"].set_index("gene")["role"]
        auc_j, auc_c, sizes, cols_j, cols_c = [], [], [], [], []
        for tf in tfs:
            j = gidx[tf]
            pos = np.zeros(G, bool)
            pos[[gidx[x] for x in ref.loc[ref["source"] == tf, "target"]]] = True
            keep = np.ones(G, bool)
            keep[j] = False
            s_j = np.abs(M[:, j])
            rho = spearmanr(Xm[:, j], Xm).statistic[0] if Xm[:, j].std() > 0 else np.zeros(G + 1)
            s_c = np.abs(np.nan_to_num(np.asarray(rho)[1:]))
            a_j = _auroc(s_j[keep], pos[keep])
            a_c = _auroc(s_c[keep], pos[keep])
            auc_j.append(a_j)
            auc_c.append(a_c)
            sizes.append(int(pos.sum()))
            cols_j.append(s_j[keep])
            cols_c.append(s_c[keep])
            per_tf.append({"archetype": ak, "tf": tf, "n_targets": int(pos.sum()),
                           "tf_role": roles.get(tf, "?"), "auroc_jacobian": a_j,
                           "auroc_coexpr": a_c})
            for x, mode in ref.loc[ref["source"] == tf, ["target", "mode"]].itertuples(index=False):
                v = M[gidx[x], j]
                per_edge.append({"archetype": ak, "tf": tf, "target": x, "mode": mode,
                                 "M": float(v), "rank_jacobian": int((s_j[keep] > abs(v)).sum()) + 1,
                                 "sign_agrees": (np.sign(v) == (1 if mode == "Activation" else -1))
                                 if mode in ("Activation", "Repression") else np.nan})
        # Permutation null for the mean AUROC across TFs.
        null_j, null_c = np.empty(args.n_perm), np.empty(args.n_perm)
        n_keep = G - 1
        for b in range(args.n_perm):
            vj, vc = [], []
            for sj, sc, n1 in zip(cols_j, cols_c, sizes):
                pos = np.zeros(n_keep, bool)
                pos[rng.choice(n_keep, n1, replace=False)] = True
                vj.append(_auroc(sj, pos))
                vc.append(_auroc(sc, pos))
            null_j[b], null_c[b] = np.nanmean(vj), np.nanmean(vc)
        mj, mc = float(np.nanmean(auc_j)), float(np.nanmean(auc_c))
        summary.append({"archetype": ak, "behaviour": info["behaviour"], "n_tfs": len(tfs),
                        "n_edges": int(sum(sizes)), "mean_auroc_jacobian": mj,
                        "p_jacobian": float((1 + (null_j >= mj).sum()) / (1 + args.n_perm)),
                        "mean_auroc_coexpr": mc,
                        "p_coexpr": float((1 + (null_c >= mc).sum()) / (1 + args.n_perm))})

    pt, pe, sm = pd.DataFrame(per_tf), pd.DataFrame(per_edge), pd.DataFrame(summary)
    pt.to_csv(out / "reference_auroc_per_tf.csv", index=False)
    pe.to_csv(out / "reference_edges_ranked.csv", index=False)
    sm.to_csv(out / "reference_auroc_summary.csv", index=False)
    print(sm.to_string(index=False))
    signed = pe.dropna(subset=["sign_agrees"]).copy()
    signed["sign_agrees"] = signed["sign_agrees"].astype(float)   # object-dtype bools sum as OR
    print(f"\nsign agreement (Activation/Repression edges, all archetypes): "
          f"{signed['sign_agrees'].mean():.2f} over {len(signed)} edge×archetype")
    for mode, d in signed.groupby("mode"):
        print(f"  {mode}: agreement {d['sign_agrees'].mean():.2f} (n={len(d)}); "
              f"fraction M>0 = {(d['M'] > 0).mean():.2f}")
    print(f"  base rate: fraction of all off-diagonal M > 0 = {np.mean(base_pos):.2f} "
          "(sign agreement is informative only relative to this)")
    print("\nper-TF AUROC averaged over archetypes:")
    print(pt.groupby("tf")[["n_targets", "auroc_jacobian", "auroc_coexpr"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
