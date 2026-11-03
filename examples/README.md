# Examples

Runnable scripts with built-in test data. Each takes seconds on a laptop CPU
and needs only `pip install -e .` from the repository root.

| script | route | what it shows | time |
|---|---|---|---|
| [flow_quickstart.py](flow_quickstart.py) | time course (`sj.flow`) | a simulated pitchfork sampled at five times; λ₁ is high at the branch point and low in committed cells (AUC 1.00) | ~10 s |
| [snapshot_quickstart.py](snapshot_quickstart.py) | snapshot (`sj.run`, energy operator) | a double well sampled once; density peaks are labelled *holder* and the bridge between them *fork*; writes a run folder with figures | ~20 s |

```bash
python examples/flow_quickstart.py
python examples/snapshot_quickstart.py      # output in runs/snapshot_quickstart/
```

The worked equations for the double well are in
[docs/math.md §3.2](../docs/math.md#32-the-operator).
