# Instructions for AI agents: scJDO2 route simulation

Read this before running scJDO2 on a user's data. It tells you what to ask
for, what to run, and what not to claim.

## What it does

`scjdo2 simulate` takes single-cell RNA-seq counts and:

1. turns the best available evidence of direction into a velocity in a
   latent space and fits scJDO2's drift to it;
2. fits scJDO2's gene-level operator on a subsample and groups cells into
   operator archetypes;
3. sends walkers from a start state that **hop between measured cells**
   along the drift (they cannot leave the data);
4. for each route the walkers take, shows which archetypes they pass and
   how those archetypes' genes and gene coupling change along the way;
5. writes `route_page.html` (interactive), `report.md` (plain-language
   result and caveats), `diagnostics.json` and CSV tables.

## Install

```bash
git clone https://github.com/manarai/scJDO2.git
cd scJDO2
pip install -e .
```

Python ≥ 3.10. The route simulation does not need scJDO v1.

## What to ask the user for

| Needed | Why |
|---|---|
| An `.h5ad` file with **raw counts** in `layers['counts']` or `X` | the gene-level operator uses a count likelihood |
| The `obs` column with cell states / types | routes are named by state |
| The **start state** (e.g. `HSC`, `MEF`) | walkers start there |

Then ask which of these exist, strongest first — this decides how much the
result can be trusted:

| Evidence of direction | Option | Strength |
|---|---|---|
| Lineage barcodes (clones) **and** time points | `--clone-key`, `--time-key` | strongest; enables a held-out fate test |
| A velocity already in a latent space (e.g. metabolic labelling) | `--velocity-key` + `--latent-key` | good if the velocity is good |
| Several time points, no lineage | `--time-key` | moderate (optimal transport; operator mostly symmetric) |
| Only a pseudotime | `--pseudotime-key` | weakest; no check against true fates |

Do not use RNA splicing velocity without telling the user it is often
unreliable in differentiation data.

Optional: `--basis` (an `obsm` 2-D layout for the page floor, e.g.
`X_umap`), `--reference` (a TRRUST-style TSV of TF → target pairs for the
species, to check the gene coupling), `--keep-genes` (genes the user cares
about, kept in the gene set).

## Run

```bash
scjdo2 simulate --input data.h5ad --out results/ \
    --state-key cell_type --start-state HSC \
    --clone-key clone --time-key day \
    --reference trrust_rawdata.mouse.tsv
```

or in Python:

```python
import anndata as ad
from scjdo2.routes import simulate
diag = simulate(ad.read_h5ad("data.h5ad"), "cell_type", "HSC", "results",
                clone_key="clone", time_key="day")
```

Cost: the gene-level operator fit dominates. At the default
`--n-cells 2000` and 2,000 genes, one run took about 55 min on an
Apple-silicon Mac CPU (about 45 min of it in the operator fit; LARRY,
130,887 cells, and CD34+ marrow, 5,780 cells, took about the same).
Memory grows with cells × genes: the stored operator for 3,000 cells was
about 9 GB, so expect several GB at the default. Lower `--n-cells` or
`--n-top-genes` on a laptop, and run it in the background. At most eight states are
shown; rarer ones are merged into `Other`.

## Reading the result — report it like this

1. Open `report.md`. Lead with where walkers ended and the direction source.
2. **Always pass on every line under "Caveats (generated from the numbers)"**.
   They are computed from this run, not boilerplate.
3. If a held-out clone fate test ran, report the drift-guided walk against
   the baselines. Only if it beats the undirected walk and proximity may you
   say the model's direction carries fate information for this dataset.
4. Point the user to `route_page.html` for the interactive view.

## Do not claim

- **Gene coupling is not a gene regulatory network.** Edges say how a change
  in one gene moves another inside an archetype's fitted operator. On every
  dataset tested (hematopoiesis, iPSC reprogramming, LARRY, CD34+ marrow)
  they did not recover TF → target regulation beyond what co-expression
  gives. Never call them regulation, and never call a gene a regulator from
  this output alone.
- Genes absent from the panels (including known regulators) are usually
  "unmeasured" by the operator, not unimportant.
- Playback counts hops, not time.
- A state no walker reaches may be disconnected in the data, not an
  impossible fate.
- Without lineage tracing or a time course, where walkers end is the
  model's tendency, not a measured fate.

## Evidence behind the defaults

- On LARRY (Weinreb et al. 2020), clones split in half three times: the
  drift-guided cell walk predicted held-out clonal fate with macro AUROC
  0.89, equal to a classifier trained on the fates and above an undirected
  walk (0.87) in every split; walkers moving freely in the latent space
  (0.81) or rolling down a 2-D landscape (0.50–0.62) did worse than simple
  proximity (0.83). Hence walkers hop between cells, and the 2-D landscape
  on the page is only a backdrop.
- On iPSC reprogramming, a 2-D landscape trapped every walker in
  Trophoblast; the cell walk reaches iPSCs.
- On a snapshot (CD34+ marrow), scJDO2's score-matching drift barely
  followed pseudotime, so the pseudotime source uses the pseudotime
  gradient directly.

## Developing

Tests: `pytest tests/test_routes.py`. The code lives in `scjdo2/routes/`.
