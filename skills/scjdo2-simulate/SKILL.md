---
name: scjdo2-simulate
description: Run the scJDO2 route simulation on a user's single-cell RNA-seq data (h5ad with raw counts) — walkers hop between measured cells along a scJDO2 drift from a start state; produces an interactive route page with archetypes, genes and gene coupling, plus a report with generated caveats. Use when the user asks to simulate cell fate routes, trajectories or transitions with scJDO2, or points at github.com/manarai/scJDO2 with their data.
---

# scJDO2 route simulation

Follow `AGENTS.md` at the root of the scJDO2 repository; it is the full
instruction set. In short:

1. Install: `pip install -e .` in a clone of https://github.com/manarai/scJDO2.
2. Confirm with the user: the h5ad path (raw counts), the state column, the
   start state, and the strongest direction evidence they have — lineage
   barcodes + time points > a latent-space velocity > time points > pseudotime.
3. Run `scjdo2 simulate --input … --out … --state-key … --start-state …` with
   the matching `--clone-key/--time-key`, `--velocity-key/--latent-key`,
   `--time-key` or `--pseudotime-key`. Add `--reference` (TRRUST-style TSV)
   when available. Run it in the background; a run takes
   about an hour on a laptop CPU at the default `--n-cells 2000`.
4. Report from `report.md`: where walkers ended, the direction source, the
   fate test if it ran, the flow λ₁ section if present (expanding vs
   contracting along the routes), and **every generated caveat**. Point to
   `route_page.html`.
5. Never present gene coupling as a regulatory network or call genes
   regulators from this output; playback is hops, not time.
