# scJDO v1 removal (2026-10-16)

scJDO2 has one drift: `f = −∇E_θ`, the energy network fitted by score
matching (`scjdo2.energy_drift`). The v1 neural drift and everything that
existed only to run it were removed. To recover any removed file:
`git show c3c4223:<path>` (the commit before the removal).

## Removed

**Library**
* `scjdo2/drift.py` — `DriftAdapter`, a wrapper around v1's
  `scjdo.models.drift.DriftField`, v1's DSM/control-energy losses and
  `scjdo.tl._drift._pseudotime_velocity`.

**Changed to the energy drift**
* `scjdo2/pipeline.py::fit_latent_field` — now fits `EnergyDriftAdapter`;
  time-independent, so `time_key` and `clock_scale` are gone;
  `drift_config` is an `EnergyDriftConfig`.
* `scjdo2/cli.py` `run` — energy drift; `--time-key` removed.

**Scripts that studied the v1 drift itself**
`dsm_sanity.py`, `sigma_sweep.py`, `pitchfork_architecture_check.py`,
`pitchfork_convergence_audit.py`, `phase_b_gate_report.py`,
`equilibrated_gate.py` (superseded by `pitchfork_energy_gate.py`).

**Dataset drivers that ran the v1 drift with a pseudotime anchor**
`run_larry.py`, `run_paul15.py`, `run_paul15_demo.py`, `run_setty_v2.py`,
`setty_archetype_region.py`, `phase_c_gate.py`, `make_figures_pitchfork.py`.
Their committed results under `benchmarks/` are v1-drift results and are
kept as records. New energy-drift drivers are written when needed (e.g.
for the orientation layer's Paul15 and LARRY reports).

**Tests**
* `tests/test_phase_b_pitchfork.py` — Phase B gates on the v1 drift.
* `tests/test_opspace_pitchfork.py` — ported to the energy drift; its
  field-gradient and fork-behaviour tests against the pitchfork's τ_crit
  were removed (they tested a time-dependent drift).
* `tests/test_smoke.py` — no longer imports `scjdo2.drift`.

**Imports**
* `scripts/run_pbmc_basin.py` — unused `from scjdo2.drift import …` removed.

## Kept

* `compare/` and `scripts/install_v1_compare.sh` — the separate v1-vs-v2
  benchmark harness (for the v2 revision). It imports the v1 package
  itself, is outside the library and its tests, and is not a dependency:
  `pyproject.toml` lists no scJDO (`compare = []`).
* Benchmark results produced by removed scripts, as records.
* `CHANGELOG.md` history.
