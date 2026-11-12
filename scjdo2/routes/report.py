"""Page text, report and tables, written from the diagnostics.

Caveats are generated from the numbers (weak direction, unreached states,
unmeasured genes, symmetric operators, weak reference support), so that a
reader — or an AI agent summarising the run — sees the limits next to the
result.
"""

from __future__ import annotations

import html
import json
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = {
    "clone": ("lineage barcodes",
              "each clone's centroid shift between its first and last time point, per unit time, given "
              "to all its cells. Clone velocity is not a gradient field, so the operator can be "
              "directional. On LARRY held-out clones, a walk on this kind of drift predicted clonal fate "
              "as well as a classifier trained on the fates (macro AUROC 0.89)."),
    "velocity": ("a measured velocity",
                 "a velocity supplied in the latent space (for example from metabolic labelling)."),
    "ot": ("optimal transport between time points",
           "entropic optimal transport between consecutive time points; each cell's displacement per unit "
           "time. Optimal-transport displacement is close to a gradient field, so the operator is mostly "
           "symmetric: expect hubs more than drivers and responders."),
    "pseudotime": ("pseudotime",
                   "the local gradient of pseudotime. This is a gradient field by construction, the weakest "
                   "source: it says which way is later, nothing about rotation or regulation, and the "
                   "operator will be mostly symmetric."),
}


LAMBDA_TXT = ("The leading eigenvalue of scJDO2's flow operator (optimal transport between consecutive time "
              "points, a smooth drift fitted to it, and that drift's Jacobian at each cell). Above 0, nearby cells "
              "move apart (expanding, as at a fate split); below 0 they converge (contracting, as when cells "
              "commit). On LARRY, per-cell λ₁ reproduced across replicate wells (Spearman 0.97); archetypes of "
              "this operator reproduced only as that two-way split, so none are shown. λ₁ is a ranking of "
              "neighbourhoods, not a measured rate: the operator's values depend on the transport space.")


def warnings(diag) -> list[str]:
    """Plain-language caveats triggered by the diagnostics."""
    w = []
    for key in ("alignment_time", "alignment_pseudotime"):
        if key in diag and diag[key] < 0.2:
            w.append(f"The drift barely points along {key.split('_')[1]} (median cosine {diag[key]:.2f}); "
                     "walkers may pool in dense regions instead of differentiating.")
    if diag["never_reached"]:
        w.append("Never reached by any walker: " + ", ".join(diag["never_reached"]) +
                 ". Check whether these states connect to the start state in the data before reading anything into it.")
    arch = diag["archetypes"]
    if arch:
        unm = float(np.median([a["frac_unmeasured"] for a in arch]))
        if unm > 0.8:
            w.append(f"About {unm:.0%} of genes carry no operator weight in a typical archetype; "
                     "absent genes (including known regulators) say nothing about their role.")
        n_dr = sum(a["n_driver"] + a["n_responder"] for a in arch)
        n_hub = sum(a["n_hub"] for a in arch)
        if n_hub > 2 * max(n_dr, 1):
            w.append(f"Most influential genes are hubs ({n_hub} hubs against {n_dr} drivers and responders): "
                     "driver/responder labels are thin here.")
        asym = float(np.median([a["asym"] for a in arch]))
        if asym < 0.5:
            why = (f"as expected from a {diag['source']} drift, which is close to a gradient field"
                   if diag["source"] in ("ot", "pseudotime") else
                   f"although a {diag['source']} drift can be directional")
            w.append(f"Operators are mostly symmetric (median directional/symmetric ratio {asym:.2f}), {why}; "
                     "direction between genes is weak.")
    ref = diag.get("reference")
    if ref:
        if ref["above_coexpression"] < ref["n_archetypes"]:
            w.append(f"Reference network: operator edges beat random targets in {ref['above_random']} of "
                     f"{ref['n_archetypes']} archetypes but plain co-expression in only {ref['above_coexpression']}.")
    else:
        w.append("No reference network given, so gene coupling was not checked against known TF → target pairs.")
    ft = diag.get("fate_test", {}).get("summary")
    if ft and "walk_drift" in ft:
        wd = ft["walk_drift"]["test_macro_auroc"]
        best_base = max(v["test_macro_auroc"] for k, v in ft.items() if k in ("walk_undirected", "proximity"))
        if wd <= best_base:
            w.append(f"Held-out clone test: the drift-guided walk ({wd:.2f}) does not beat the baselines "
                     f"({best_base:.2f}); the drift adds no fate information here.")
    fl = diag.get("flow_lambda")
    if fl:
        if fl["n_times"] == 2:
            w.append("Flow λ₁ rests on two time points (one optimal-transport step); treat it as rough.")
        if fl["frac_last_time"] > 0.3:
            w.append(f"{fl['frac_last_time']:.0%} of the shown cells are on the last time point, where no later "
                     "cells exist: their flow λ₁ comes from the drift's extrapolation, not from transport.")
    if diag["merged_states"]:
        w.append(f"{len(diag['merged_states'])} rare states were merged into 'Other' (the viewer has eight colours).")
    if diag["source"] == "pseudotime":
        w.append("No lineage tracing or time course: where walkers end is not checked against true fates. With "
                 "pseudotime as the only direction, walkers drift toward the states with the highest pseudotime, "
                 "and the end shares can change with the cell sample and number of hops.")
    return w


def _caveat_html(diag, w):
    ref = diag.get("reference")
    ref_txt = (f" Operator edges beat random reference targets in {ref['above_random']} of {ref['n_archetypes']} "
               f"archetypes (AUROC {ref['auroc_range'][0]:.2f}–{ref['auroc_range'][1]:.2f}) and plain "
               f"co-expression in {ref['above_coexpression']}." if ref else "")
    return ("Exploratory. Gene roles come from the scJDO2 finite-time influence of each archetype's operator and "
            "are fixed per archetype: only how strongly each archetype is present changes along a route. "
            "Cell-cycle genes are left out; each archetype's 20 most influential remaining genes are ranked by "
            "polarity and the 5 at each end shown, plus hubs." + ref_txt +
            " The gene coupling panel shows how strongly a small change in one gene moves another within the "
            "archetype's operator. <strong>It is not a regulatory network</strong>; edge signs are not shown. "
            "Relative time: 0 = start, 1 = when the walker settles in its final state. " +
            " ".join(html.escape(x) for x in w))


def _methods_html(diag, fate_tbl):
    src_name, src_txt = SOURCE[diag["source"]]
    fin = ", ".join(f"{n} {c}" for n, c in diag["final"].items() if c)
    p = [f"<p><strong>How this was made.</strong> {diag['n_cells_total']:,} cells, {diag['n_genes']:,} variable "
         f"genes, a {diag['latent_dim']}-dimensional latent space. Direction comes from {src_name}: "
         f"{html.escape(src_txt)} scJDO2's <code>MeasuredDrift</code> is fitted to it on {diag['drift_fit_cells']:,} "
         f"cells (median cosine {diag['drift_fit_median_cos']:.2f}); the gene-level operator and archetypes are fitted "
         f"on {diag['n_cells_operator']:,} cells sampled across states.</p>",
         f"<p><strong>Walkers hop between measured cells.</strong> Each cell links to its {diag['k']} nearest cells "
         f"in the latent space, and a hop to neighbour j is weighted by exp(cos(x<sub>j</sub> − x<sub>i</sub>, drift "
         f"at i) / {diag['sigma']:g}), the CellRank-style velocity kernel; about {diag['hops']} hops per walker. "
         f"{diag['n_walkers']} walkers start in {html.escape(diag['start_state'])}. At the end: {fin}. The floor is "
         f"{html.escape(diag['floor'])}; the surface (valleys from cell density, slope from the projected drift) is a "
         "backdrop only. Playback counts hops, not time.</p>"]
    fl = diag.get("flow_lambda")
    if fl:
        p.append(f"<p><strong>Flow λ₁.</strong> {LAMBDA_TXT} Here: optimal transport between {fl['n_times']} time points "
                 f"on {fl['fit_cells']:,} cells, a {fl['n_comps']}-component PCA, units {html.escape(fl['units'])}.</p>")
    al = [f"{k.split('_')[1]} {v:.2f}" for k, v in diag.items() if k.startswith("alignment_")]
    if al:
        p.append("<p>Median cosine between the drift and the direction of increasing " + "; ".join(al) + ".</p>")
    ft = diag.get("fate_test", {}).get("summary")
    if ft:
        rows = "".join(f"<tr><td>{html.escape(k)}</td><td class='num'>{v['test_macro_auroc']:.2f}</td>"
                       f"<td class='num'>{v['ci_lo']:.2f}–{v['ci_hi']:.2f}</td><td class='num'>{v['top_fate_acc']:.2f}</td></tr>"
                       for k, v in ft.items())
        p.append("<p><strong>Held-out clone fate test.</strong> Clones split in half three times; settings chosen on "
                 "training clones; scored on test clones' earliest cells against each clone's majority fate.</p>"
                 "<table class='ft'><thead><tr><th>Method</th><th class='num'>macro AUROC</th><th class='num'>95% CI</th>"
                 f"<th class='num'>top-fate acc.</th></tr></thead><tbody>{rows}</tbody></table>")
    p.append("<p>Made with <code>scjdo2 simulate</code> (<code>scjdo2.routes</code>); see <code>report.md</code>.</p>")
    return "\n    ".join(p)


def render_page(view, routes, diag, fate_tbl=None, title=None, dataset_name=None) -> str:
    tpl = resources.files("scjdo2.routes").joinpath("assets/viewer.html").read_text(encoding="utf-8")
    w = warnings(diag)
    name = dataset_name or "Cells"
    src_name = SOURCE[diag["source"]][0]
    rep = {"__TITLE__": html.escape(title or "Route Landscape"),
           "__HEADLINE__": html.escape(f"{name} travelling between states"),
           "__SUB__": html.escape(f"Cells hop from measured cell to measured cell, steered by the scJDO2 drift "
                                  f"(direction from {src_name}), so they always stay on real cells. They start in "
                                  f"{diag['start_state']} and change colour as they move into a new state."),
           "__NMEAS__": f"{diag['n_cells_operator']:,}",
           "__METHODS__": _methods_html(diag, fate_tbl),
           "__CAVEAT__": json.dumps(_caveat_html(diag, w)),
           "__DATA__": json.dumps(view, separators=(",", ":")),
           "__ROUTES__": json.dumps(routes, separators=(",", ":"))}
    for k, v in rep.items():
        tpl = tpl.replace(k, v)
    return tpl


def report_md(diag, routes, w) -> str:
    L = ["# scJDO2 route simulation report", ""]
    L += ["## Result", "",
          f"- Direction source: **{diag['source']}** ({SOURCE[diag['source']][0]}); drift fit median cosine "
          f"{diag['drift_fit_median_cos']:.2f} on {diag['drift_fit_cells']:,} cells.",
          f"- {diag['n_walkers']} walkers from **{diag['start_state']}** end in: " +
          ", ".join(f"{n} {c}" for n, c in diag["final"].items() if c) + "."]
    for r in diag["routes"]:
        L.append(f"- Route {' → '.join(r['path'])}: {r['n']} walkers; top archetypes {r['top_archetypes']}.")
    for k, v in diag.items():
        if k.startswith("alignment_"):
            L.append(f"- Drift vs {k.split('_')[1]}: median cosine {v:.2f} (1 = same direction, 0 = unrelated).")
    fl = diag.get("flow_lambda")
    if fl:
        L += ["", "## Flow λ₁ (time course)", "", LAMBDA_TXT, "",
              f"Optimal transport over {fl['n_times']} time points ({', '.join(f'{t:g}' for t in fl['times'])}) on "
              f"{fl['fit_cells']:,} cells; λ₁ {fl['units']}. Shown cells: median {fl['quantiles_5_50_95'][1]:.2f} "
              f"(5–95%: {fl['quantiles_5_50_95'][0]:.2f} to {fl['quantiles_5_50_95'][2]:.2f}); "
              f"{fl['frac_expanding']:.0%} expanding.", "",
              "| state | cells | median λ₁ | expanding |", "|---|---|---|---|"]
        for n, v in fl["by_state"].items():
            L.append(f"| {n} | {v['n']} | {v['median']:.2f} | {v['frac_expanding']:.0%} |")
        L.append("")
        for r in fl["routes"]:
            L.append(f"- Route {' → '.join(r['path'])}: λ₁ {r['start']:.2f} at the start, {r['end']:.2f} on arrival.")
    ft = diag.get("fate_test", {})
    if "summary" in ft:
        L += ["", "## Held-out clone fate test", "", "| method | macro AUROC | 95% CI | top-fate acc. |", "|---|---|---|---|"]
        for k, v in ft["summary"].items():
            L.append(f"| {k} | {v['test_macro_auroc']:.3f} | {v['ci_lo']:.3f}–{v['ci_hi']:.3f} | {v['top_fate_acc']:.3f} |")
    elif "skipped" in ft:
        L += ["", f"Fate test skipped: {ft['skipped']}"]
    L += ["", "## Caveats (generated from the numbers)", ""] + [f"- {x}" for x in w]
    L += ["", "## Archetypes", "", "| id | behaviour | cells | drivers | hubs | responders | directional/symmetric | unmeasured genes |",
          "|---|---|---|---|---|---|---|---|"]
    for a in diag["archetypes"]:
        L.append(f"| {a['id']} | {a['behaviour']} | {a['n_cells']} | {a['n_driver']} | {a['n_hub']} | "
                 f"{a['n_responder']} | {a['asym']:.2f} | {a['frac_unmeasured']:.0%} |")
    L += ["", "## How to read this", "",
          "- Walkers hop between real cells along the drift; where they end is the model's fate tendency from the "
          "start state, not a measured fate unless the fate test above supports it.",
          "- Archetypes are groups of cells sharing a local linear operator; their genes are coupled programs.",
          "- **Do not** present gene coupling edges as a gene regulatory network or TF → target regulation.",
          "- **Do not** read playback hops as real time.",
          "- States never reached may be disconnected in the data, not impossible fates.", "",
          "## Files", "",
          "- `route_page.html` — interactive page (open in a browser).",
          "- `diagnostics.json` — every number above.",
          "- `routes.json` — route panels (archetype curves, genes, coupling edges).",
          "- `routes.csv`, `archetypes.csv`, `fate_test.csv` — tables.", ""]
    return "\n".join(L)


def write_outputs(out: Path, view, routes, diag, fate_tbl=None, title=None, dataset_name=None) -> None:
    w = warnings(diag)
    diag["warnings"] = w
    (out / "route_page.html").write_text(render_page(view, routes, diag, fate_tbl, title, dataset_name), encoding="utf-8")
    (out / "diagnostics.json").write_text(json.dumps(diag, indent=1, default=str))
    (out / "routes.json").write_text(json.dumps(routes, separators=(",", ":")))
    (out / "report.md").write_text(report_md(diag, routes, w))
    pd.DataFrame([{"path": " > ".join(r["path"]), "n_walkers": r["n"], "top_archetypes": r["top_archetypes"]}
                  for r in diag["routes"]]).to_csv(out / "routes.csv", index=False)
    pd.DataFrame(diag["archetypes"]).to_csv(out / "archetypes.csv", index=False)
    if fate_tbl is not None:
        fate_tbl.to_csv(out / "fate_test.csv", index=False)
