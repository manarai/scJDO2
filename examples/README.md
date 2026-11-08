# Examples

Runnable scripts with built-in test data. Each takes seconds on a laptop CPU
and needs only `pip install -e .` from the repository root.

| script | route | what it shows | time |
|---|---|---|---|
| [flow_quickstart.py](flow_quickstart.py) | time course (`sj.flow`) | a simulated pitchfork sampled at five times; λ₁ is high at the branch point and low in committed cells (AUC 1.00) | ~10 s |
| [flow_motifs_tutorial.py](flow_motifs_tutorial.py) | time course, motif layer | the same pitchfork: archetypes, transitions along the OT plan (approach → branch → arms; spreading → converging shapes), paths, drift paths; writes figures to `runs/flow_motifs_tutorial/`. Notebook version: [../notebooks/tutorial_flow_motifs.ipynb](../notebooks/tutorial_flow_motifs.ipynb) | ~15 s |
| [snapshot_quickstart.py](snapshot_quickstart.py) | snapshot (`sj.run`, energy operator) | a double well sampled once; density peaks are labelled *holder* and the bridge between them *fork*; writes a run folder with figures | ~20 s |

```bash
python examples/flow_quickstart.py
python examples/flow_motifs_tutorial.py     # output in runs/flow_motifs_tutorial/
python examples/snapshot_quickstart.py      # output in runs/snapshot_quickstart/
```

The worked equations for the double well are in
[docs/math.md §3.2](../docs/math.md#32-the-operator).
