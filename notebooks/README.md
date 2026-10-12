# Tutorial notebooks (Phase E)

One notebook per v1 tutorial, re-run with scJDO2 using the same datasets and
seeds as the v1 side. Keys match ``data/loaders.py``:

| Notebook                           | Dataset key        |
|------------------------------------|--------------------|
| `01_paul15.ipynb`                  | `paul15`           |
| `02_paul15_fourier.ipynb`          | `fourier_paul15`   |
| `03_sb_synthetic.ipynb`            | `sb_synthetic`     |
| `04_paul15_bridge.ipynb`           | `paul15_bridge`    |

Each notebook, when run, must write its outputs under
`../benchmarks/v2/<dataset>/` so `compare/v1_vs_v2.py` can read them next to
the matching v1 baseline under `../benchmarks/v1_baseline/<dataset>/`.
