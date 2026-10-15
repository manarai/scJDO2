# A04 rendering deviations

* `figure_archetype_sheet` was not rendered: no stored archetype weights exist, and A04 prohibits the requested semi-NMF fitting.
* The ten-refit `lead_re` permutation-null panel was not rendered: no such refits are stored, and recreating them would violate the no-refit instruction.
* Per-epoch scVI ELBO curves were not stored by A02. The rendered QC panels show the stored best validation ELBO summaries instead.
