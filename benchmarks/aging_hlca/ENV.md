# Environment provenance

## hlca

Collected before the HLCA data fetch and scArches smoke test.

| Component | Version |
|---|---:|
| Python | 3.13.2 |
| anndata | 0.11.4 |
| scanpy | 1.12.4 |
| scvi-tools | 1.5.1 |
| scarches | 0.6.1 |
| cellxgene-census | 1.18.0 |
| torch | 2.14.1 |
| numpy | 2.5.4 |
| pandas | 2.3.3 |
| pytest | 9.1.1 |

`python -m pytest tests/test_compare_aging.py -q` completed successfully: 9 passed.

Compatibility basis: the current scArches HLCA tutorial states that `scanpy>=1.9.3`, `scarches>=0.5.8`, and `scvi-tools>=0.20.3` avoid a known label-transfer issue. The installed versions meet or exceed those minima. The smoke test remains the compatibility gate for the actual Zenodo reference model.
