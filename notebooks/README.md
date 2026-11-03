# Notebooks

Figure and table notebooks for the scJDO preprint (v57) and the Application
Note. They read committed result files under `benchmarks/` only; they do not
refit models. Run all of them in order with

```bash
python notebooks/run_all.py              # or: python notebooks/run_all.py fig3_flow_operator
```

Environment and details: [manuscript/REPRODUCE_v57.md](../manuscript/REPRODUCE_v57.md).

## Preprint figures

| notebook | figure | content |
|---|---|---|
| `fig1_framework.ipynb` | Fig. 1 | framework schematic (drawing only, no data) |
| `fig2_benchmark.ipynb` | Fig. 2 | a benchmark calibrated to real data |
| `fig3_flow_operator.ipynb` | Fig. 3 | the flow operator from a time course alone |
| `fig4_dyngen.ipynb` | Fig. 4 | an independent simulator (dyngen, 8 time points) |
| `fig5_real_scale.ipynb` | Fig. 5 | real data (LARRY) and scaling |
| `fig6_score_operator.ipynb` | Fig. 6 | tests of the score (energy) operator |
| `fig7_archetypes.ipynb` | Fig. 7 | archetypes of dynamics |
| `supp_s10_diffusion.ipynb` | Supp. Fig. S1 | diffusion is not identifiable at 48-h sampling |
| `supp_s21_three_timepoints.ipynb` | Supp. Fig. S2 | dyngen with three sampling times fails |
| `supp_s13_whitening.ipynb` | Supp. Fig. S3 | embedding and whitening grid |
| `supp_s11_operator_agreement.ipynb` | Supp. Fig. S4 | per-cell operator agreement on real data |
| `tables_v57.ipynb` | Supp. Tables S2, S3 | score-operator tables |
| `numbers_v57.ipynb` | — | checks every number in the manuscript against the result files |
| `build_reports.ipynb` | — | assembles the figure reports |

`appnote_fig1.py` draws Figure 1 of the Application Note
(`python notebooks/appnote_fig1.py`).

## Earlier analyses (v0.5, kept for the record)

| notebook | content |
|---|---|
| `pitchfork_gate.ipynb` | basin-rule comparison on the equilibrated pitchfork |
| `basin_analysis.ipynb` | PBMC basin analysis from stored state |
| `perturbseq_k562.ipynb` | K562 CRISPRi Perturb-seq prediction |

These were run at tag `v0.5-basin-lock` and use the energy operator's
holder/fork/drift labels, which describe density curvature, not stability
(see [docs/math.md §3.3](../docs/math.md#33-why-this-is-density-curvature-not-stability)).
