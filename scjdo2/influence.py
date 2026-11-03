"""Finite-time influence: drivers vs responders per archetype.

A small push ``δ`` on gene ``j`` evolves under the archetype operator as
``δx(t) = e^{J t} e_j δ``. We work on the archetype's tangent-restricted
block ``B = P J_k P^T`` (``P``: ``(r, G)`` orthonormal rows, stored by
:func:`scjdo2.opspace.archetype_operators`), so the gene-space change
operator is::

    M(t) = P^T (e^{B t} - I) P            (G × G)

``M[i, j]`` is the change in gene ``i`` at time ``t`` per unit push on
gene ``j``. Self terms (``i = j``) are excluded throughout, so the
scores measure effects on *other* genes:

* **out-influence** ``out_j(t) = ||M[:, j]||`` over ``i ≠ j`` — how much a
  minimal change in ``j`` moves the rest. High → **driver**.
* **in-influence**  ``in_i(t)  = ||M[i, :]||`` over ``j ≠ i`` — how much
  ``i`` is moved by minimal changes elsewhere. High → **responder**.

Both are cheap: ``P`` has orthonormal rows, so ``||P^T c|| = ||c||`` and
the column / row norms reduce to ``r``-dimensional norms.

Long-time limit: with one dominant mode, ``e^{Bt} ≈ e^{λt} r l^T``, so
``out_j ∝ |l_j|`` (left eigenvector) and ``in_i ∝ |r_i|`` (right
eigenvector). At short and intermediate ``t`` a non-normal operator can
rank genes differently; that is what the time axis shows.

Default time ``t*`` per archetype: for a stable block, the peak of the
transient gain ``g(t) = ||e^{Bt}||_2`` when it exceeds 1.05 (classic
non-normal transient growth); otherwise (no transient, or an unstable
block) the lead-mode time scale ``1/|Re λ_1|``. Times are in the
operator's own units (hours only when the clock is calibrated).

Genes whose column is not measured (aggregated Fisher = 0: ``capped`` /
``no_variation``) or that lie mostly off the archetype tangent get the
role ``unmeasured``. Their influence is zero by construction, not by
measurement, and must not be read as "no influence".
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.linalg import expm

ROLES = ("driver", "hub", "responder", "passenger", "unmeasured")

# ---------------------------------------------------------------- core maths


def _change_operator(B: np.ndarray, t: float) -> np.ndarray:
    """``D(t) = e^{B t} - I`` on the ``r``-dim tangent block."""
    r = B.shape[0]
    return expm(B * float(t)) - np.eye(r)


def influence_scores(B: np.ndarray, P: np.ndarray, t: float) -> dict[str, np.ndarray]:
    """Out- and in-influence of every gene at time ``t``.

    Parameters
    ----------
    B : ``(r, r)`` tangent-restricted operator block.
    P : ``(r, G)`` projector with orthonormal rows.
    t : time, in the operator's units.

    Returns ``{"out", "in", "self", "coverage"}`` — each ``(G,)``.
    ``self`` is the diagonal ``M[j, j]``; ``coverage`` is ``||P[:, j]||``
    (how much of gene ``j`` lies in the tangent; 0 → not representable).
    """
    B = np.asarray(B, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    D = _change_operator(B, t)
    C = D @ P                                   # (r, G): column j = D P e_j
    Ct = D.T @ P                                # (r, G): column i = D^T P e_i
    diag = np.einsum("rg,rg->g", P, C)          # M[j, j] = P_j · D P_j
    out2 = np.einsum("rg,rg->g", C, C) - diag ** 2
    in2 = np.einsum("rg,rg->g", Ct, Ct) - diag ** 2
    return {
        "out": np.sqrt(np.clip(out2, 0.0, None)),
        "in": np.sqrt(np.clip(in2, 0.0, None)),
        "self": diag,
        "coverage": np.linalg.norm(P, axis=0),
    }


def excess_gain(B: np.ndarray, ts: Sequence[float]) -> np.ndarray:
    """``g(t) = ||e^{Bt}||_2 · e^{-max(Re λ_1, 0) t}`` on a time grid.

    For a stable block this is the plain transient gain: ``g ≤ 1`` for a
    normal operator, a peak above 1 is non-normal transient growth. For
    an unstable block the modal growth is divided out so the curve stays
    readable (it usually has no interior peak).
    """
    B = np.asarray(B, dtype=np.float64)
    lam1 = float(np.max(np.linalg.eigvals(B).real)) if B.size else 0.0
    out = np.empty(len(ts))
    for i, t in enumerate(ts):
        out[i] = np.linalg.norm(expm(B * float(t)), ord=2) * np.exp(-max(lam1, 0.0) * float(t))
    return out


def time_grid(B: np.ndarray, n: int = 16, span: tuple[float, float] = (-2.0, 1.0)) -> np.ndarray:
    """Log-spaced grid around the block's typical time scale.

    Scale ``τ = 1 / median |Re λ|`` over non-negligible eigenvalues; the
    grid runs ``τ·10^{span[0]} … τ·10^{span[1]}``. For an unstable block
    the top end is capped at ``5 / Re λ_1`` (five e-foldings): past that
    the lead mode alone sets every ranking.
    """
    ev = np.linalg.eigvals(np.asarray(B, dtype=np.float64)).real
    a = np.abs(ev)
    a = a[a > 1e-9 * max(1.0, a.max(initial=0.0))]
    tau = 1.0 / float(np.median(a)) if a.size else 1.0
    lo, hi = tau * 10.0 ** span[0], tau * 10.0 ** span[1]
    lam1 = float(ev.max()) if ev.size else 0.0
    if lam1 > 0:
        hi = min(hi, 5.0 / lam1)
    if hi <= lo:
        lo = hi / 100.0
    return np.logspace(np.log10(lo), np.log10(hi), int(n))


def default_time(B: np.ndarray, ts: Optional[np.ndarray] = None,
                 min_excess: float = 1.05) -> tuple[float, str]:
    """Pick ``t*``: transient-gain peak if non-normal enough, else ``1/|Re λ_1|``.

    Returns ``(t*, rule)`` with ``rule`` in ``{"transient_peak", "lead_timescale"}``.
    """
    ts = time_grid(B) if ts is None else np.asarray(ts, dtype=np.float64)
    lam1 = float(np.max(np.linalg.eigvals(np.asarray(B, dtype=np.float64)).real))
    if lam1 < 0:
        # Transient growth is only defined for a stable block; for an
        # unstable one an interior "peak" of the rescaled gain is an
        # artefact of dividing out e^{λ₁t}.
        g = excess_gain(B, ts)
        k = int(np.argmax(g))
        if np.isfinite(g[k]) and g[k] >= min_excess and 0 < k < len(ts) - 1:
            return float(ts[k]), "transient_peak"
    t = 1.0 / abs(lam1) if abs(lam1) > 1e-12 else float(np.median(ts))
    return float(np.clip(t, ts[0], ts[-1])), "lead_timescale"


def polarity(out: np.ndarray, inn: np.ndarray) -> np.ndarray:
    """``log2(out / in)`` with a floor at 1e-6 of the largest score."""
    out = np.asarray(out, dtype=np.float64)
    inn = np.asarray(inn, dtype=np.float64)
    eps = 1e-6 * max(float(out.max(initial=0.0)), float(inn.max(initial=0.0)), 1e-300)
    return np.log2((out + eps) / (inn + eps))


def assign_roles(out: np.ndarray, inn: np.ndarray, measurable: np.ndarray,
                 quantile: float = 0.9, n_sd: float = 1.0) -> tuple[np.ndarray, dict]:
    """Driver / hub / responder / passenger / unmeasured.

    Two steps, both among measurable genes:

    1. **Influential** = total influence ``out + in`` at or above the
       ``quantile``. The rest are ``passenger``.
    2. Influential genes are split by **polarity** ``log2(out/in)``
       relative to this archetype: ``z = (polarity − median) / (1.4826·MAD)``.
       ``z ≥ n_sd`` → ``driver``, ``z ≤ −n_sd`` → ``responder``, else ``hub``.

    Relative polarity is used because out- and in-influence are strongly
    correlated when the operator is close to normal (left ≈ right
    eigenvector); absolute cut-offs then call almost every top gene a hub.
    A driver here means "moves others more than it is moved, compared with
    the archetype's typical gene" — read its size from ``polarity``.

    Returns ``(roles, thresholds)`` with keys ``total``, ``polarity_median``,
    ``polarity_lo``, ``polarity_hi``.
    """
    out = np.asarray(out, dtype=np.float64)
    inn = np.asarray(inn, dtype=np.float64)
    m = np.asarray(measurable, dtype=bool)
    roles = np.full(out.shape, "unmeasured", dtype=object)
    nan = float("nan")
    thr = {"total": nan, "polarity_median": nan, "polarity_lo": nan, "polarity_hi": nan}
    if not m.any():
        return roles, thr
    total = out + inn
    pol = polarity(out, inn)
    t_total = float(np.quantile(total[m], quantile))
    med = float(np.median(pol[m]))
    scale = 1.4826 * float(np.median(np.abs(pol[m] - med)))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.std(pol[m])) or 1.0
    lo, hi = med - n_sd * scale, med + n_sd * scale
    infl = m & (total >= t_total) & (total > 0)
    roles[m & ~infl] = "passenger"
    roles[infl & (pol >= hi)] = "driver"
    roles[infl & (pol <= lo)] = "responder"
    roles[infl & (pol > lo) & (pol < hi)] = "hub"
    thr.update(total=t_total, polarity_median=med, polarity_lo=lo, polarity_hi=hi)
    return roles, thr


# ------------------------------------------------------------ AnnData layer


def _archetypes(adata: AnnData) -> dict:
    arch = adata.uns.get("scjdo2", {}).get("archetypes")
    if not arch:
        raise RuntimeError("Run scjdo2.opspace.archetype_operators(adata) first.")
    return arch


def _block_and_projector(info: dict) -> tuple[np.ndarray, np.ndarray]:
    if "block" not in info or "projector" not in info:
        raise RuntimeError(
            "This archetype entry has no 'block'/'projector' (made by an older "
            "scjdo2). Re-run scjdo2.opspace.archetype_operators(adata)."
        )
    return np.asarray(info["block"], dtype=np.float64), np.asarray(info["projector"], dtype=np.float64)


def archetype_influence(
    adata: AnnData,
    t: float | str = "auto",
    t_grid: Optional[Sequence[float]] = None,
    quantile: float = 0.9,
    min_coverage: float = 0.1,
    archetype_ids: Optional[Iterable[int]] = None,
    n_sd: float = 1.0,
) -> pd.DataFrame:
    """Driver / responder delineation for every archetype.

    For each archetype writes into ``uns["scjdo2"]["archetypes"][k]``:

    * ``influence`` — DataFrame (one row per gene) with ``out_influence``,
      ``in_influence``, ``polarity`` (``log2(out/in)``), ``self_effect``,
      ``coverage``, ``fisher``, ``reason``, ``role``, ``out_rank``,
      ``in_rank``; sorted by ``out_influence``.
    * ``influence_t`` (``t*``), ``influence_t_rule``,
      ``influence_thresholds`` (dict from :func:`assign_roles`).

    Roles: see :func:`assign_roles` (``quantile`` sets the influential
    cut, ``n_sd`` the polarity cut).
    * ``influence_curve`` — dict with ``t`` (grid), ``out`` / ``in``
      ``(n_t, G)`` and ``excess_gain`` ``(n_t,)``.

    ``t="auto"`` uses :func:`default_time` per archetype; a float pins the
    same ``t`` for all. Returns all archetypes' tables stacked (long
    format, with an ``archetype`` column).
    """
    arch = _archetypes(adata)
    G = adata.n_vars
    genes = np.asarray(adata.var_names)
    ids = sorted(arch) if archetype_ids is None else [int(a) for a in archetype_ids]
    frames = []
    for ak in ids:
        info = arch[ak]
        B, P = _block_and_projector(info)
        grid = time_grid(B) if t_grid is None else np.asarray(t_grid, dtype=np.float64)
        if t == "auto":
            t_star, rule = default_time(B, grid)
        else:
            t_star, rule = float(t), "fixed"

        sc = influence_scores(B, P, t_star)
        fisher = np.asarray(info.get("fisher", np.full(G, np.nan)), dtype=np.float64)
        reason = np.asarray(info.get("reason", ["ok"] * G), dtype=object)
        measured = (reason == "ok") if reason.shape == (G,) else np.ones(G, bool)
        measurable = measured & (sc["coverage"] >= float(min_coverage))
        roles, thresholds = assign_roles(sc["out"], sc["in"], measurable, quantile, n_sd)
        pol = polarity(sc["out"], sc["in"])
        df = pd.DataFrame({
            "gene": genes,
            "out_influence": sc["out"],
            "in_influence": sc["in"],
            "polarity": pol,
            "self_effect": sc["self"],
            "coverage": sc["coverage"],
            "fisher": fisher if fisher.shape == (G,) else np.nan,
            "reason": reason if reason.shape == (G,) else "ok",
            "role": roles,
        })
        df["out_rank"] = df["out_influence"].rank(ascending=False, method="min").astype(int)
        df["in_rank"] = df["in_influence"].rank(ascending=False, method="min").astype(int)
        df = df.sort_values("out_influence", ascending=False).reset_index(drop=True)

        curve_out = np.empty((grid.size, G))
        curve_in = np.empty((grid.size, G))
        for i, tt in enumerate(grid):
            s = influence_scores(B, P, float(tt))
            curve_out[i] = s["out"]
            curve_in[i] = s["in"]

        info["influence"] = df
        info["influence_t"] = t_star
        info["influence_t_rule"] = rule
        info["influence_thresholds"] = thresholds
        info["influence_curve"] = {
            "t": grid, "out": curve_out, "in": curve_in,
            "excess_gain": excess_gain(B, grid),
        }
        frames.append(df.assign(archetype=int(ak), behaviour=info.get("behaviour"),
                                t_star=t_star))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def influence_edges(
    adata: AnnData,
    archetype_id: int,
    t: Optional[float] = None,
    sources: Optional[Sequence[str]] = None,
    n_sources: int = 5,
    n_targets: int = 4,
    n_hubs: int = 2,
    min_rel_weight: float = 0.05,
) -> pd.DataFrame:
    """Directed, signed edges ``source → target`` from ``M(t)``.

    Sources default to the top ``n_sources`` drivers plus the top
    ``n_hubs`` hubs by out-influence. For each source, keeps its ``n_targets`` largest
    ``|M[target, source]|`` (``target ≠ source``) above
    ``min_rel_weight · max|M[:, source]|``.

    Columns: ``source``, ``target``, ``weight`` (signed), ``abs_weight``,
    ``sign`` (+1 activating / -1 repressing), ``t``.
    """
    info = _archetypes(adata)[int(archetype_id)]
    if "influence" not in info:
        archetype_influence(adata, archetype_ids=[archetype_id])
    B, P = _block_and_projector(info)
    t = float(info["influence_t"]) if t is None else float(t)
    tbl = info["influence"]
    genes = np.asarray(adata.var_names)
    idx = {g: i for i, g in enumerate(genes)}
    if sources is None:
        sources = (tbl[tbl["role"] == "driver"]["gene"].head(int(n_sources)).tolist()
                   + tbl[tbl["role"] == "hub"]["gene"].head(int(n_hubs)).tolist())
    D = _change_operator(B, t)
    rows = []
    for s in sources:
        j = idx[s]
        col = P.T @ (D @ P[:, j])               # M[:, j]
        col[j] = 0.0
        a = np.abs(col)
        if a.max(initial=0.0) <= 0:
            continue
        order = np.argsort(-a)[: int(n_targets)]
        for i in order:
            if a[i] < float(min_rel_weight) * a.max():
                continue
            rows.append((s, genes[i], float(col[i]), float(a[i]), int(np.sign(col[i])), t))
    return pd.DataFrame(rows, columns=["source", "target", "weight", "abs_weight", "sign", "t"])


def annotate_reference(edges: pd.DataFrame, ref_network: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Tag each dynamics edge as in / not in a reference TF→target network.

    ``ref_network`` needs ``source`` and ``target`` columns (e.g. v1
    ``scjdo.tl.load_network``). Matching is case-insensitive so mouse and
    human symbols both work. Adds ``in_reference`` (bool).
    """
    e = edges.copy()
    if ref_network is None or ref_network.empty:
        e["in_reference"] = False
        return e
    ref = set(zip(ref_network["source"].astype(str).str.upper(),
                  ref_network["target"].astype(str).str.upper()))
    e["in_reference"] = [
        (s.upper(), t.upper()) in ref for s, t in zip(e["source"].astype(str), e["target"].astype(str))
    ]
    return e
