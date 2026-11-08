# Route simulation gallery

Interactive pages made by `scjdo2 simulate`: walkers start in one state
and hop between **measured cells** along scJDO2's drift, and each route
shows the operator archetypes it passes and how their genes and gene
coupling change along the way. GitHub shows these files as source; use
the **live** links (GitHub Pages) to run them in the browser, or download
the `.html` and open it locally.

| page | data | direction of the drift | walkers end in (of 300) | check against real fates |
|---|---|---|---|---|
| [Palantir: CD34+ marrow](https://manarai.github.io/scJDO2/gallery/palantir_routes.html) · [report](palantir_report.md) | Setty et al. 2019, 5,780 cells | Palantir pseudotime (weakest source) | DCs 184, Mono 81, Ery 33, CLP 2 | none possible (no lineage or time course) |
| [LARRY: in vitro haematopoiesis](https://manarai.github.io/scJDO2/gallery/larry_routes.html) · [report](larry_report.md) | Weinreb et al. 2020, 130,887 cells | lineage barcodes + days (strongest source) | Neutrophil 134, Monocyte 109, Baso 41, Undifferentiated 12, Mast 3, Meg 1 | held-out clone fate, macro AUROC 0.897 [0.871–0.922] vs undirected walk 0.867 and proximity 0.828 |

Each report lists caveats computed from that run. Read them with the
page. In particular, the gene coupling is **not** a regulatory network:
on every dataset tested it did not recover TF → target regulation beyond
co-expression.

## How the pages are made

The route simulation code (`scjdo2.routes`, `scjdo2 simulate`) is in
[pull request #2](https://github.com/manarai/scJDO2/pull/2) and reaches
`main` with the `aging-basins` branch. With it installed, from the
repository root:

```python
from scjdo2.data import load_setty_cd34
from scjdo2.drivers import load_reference
from scjdo2.routes import simulate

a = load_setty_cd34()
merge = {"HSC_1": "HSC", "HSC_2": "HSC", "Ery_1": "Ery", "Ery_2": "Ery", "Mono_1": "Mono", "Mono_2": "Mono"}
a.obs["state"] = a.obs["cluster"].astype(str).map(lambda c: merge.get(c, c))
simulate(a, "state", "HSC", "results/palantir", pseudotime_key="pseudotime",
         keep_genes=("GATA1", "SPI1"), dataset_name="CD34+ marrow cells",
         title="Palantir Route Simulation",
         reference=load_reference("benchmarks/data/trrust/trrust_rawdata.human.tsv"))
```

```python
from scjdo2.data import load_larry
from scjdo2.drivers import load_reference
from scjdo2.routes import simulate

a = load_larry()
merge = {"Eos": "Baso", "Lymphoid": "Lymphoid/DC", "Ccr7_DC": "Lymphoid/DC", "pDC": "Lymphoid/DC"}
a.obs["state"] = a.obs["cell_type_annotation"].astype(str).map(lambda c: merge.get(c, c))
a.obsm["X_spring"] = a.obs[["spring_x", "spring_y"]].to_numpy(float)
simulate(a, "state", "Undifferentiated", "results/larry", clone_key="clone_matrix", time_key="day",
         basis="X_spring", keep_genes=("Gata1", "Spi1"), dataset_name="LARRY progenitors",
         title="LARRY Route Simulation",
         reference=load_reference("benchmarks/data/trrust/trrust_rawdata.mouse.tsv"))
```

Each run writes `route_page.html`, `report.md`, `diagnostics.json` and
CSV tables to its folder and took about an hour on an Apple-silicon Mac
(defaults: 2,000 operator cells, 2,000 genes, seed 0). The pages here are
those `route_page.html` files with a standalone HTML wrapper
(`<!doctype html>`) added so they open on their own.

The rotating GIF on the front page is rendered from the page itself with
headless Google Chrome and Pillow:

```bash
python scripts/render_route_gif.py --page results/palantir/route_page.html \
    --out docs/gallery/palantir_routes.gif
```
