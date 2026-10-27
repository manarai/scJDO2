# orient/S18b — K562 control-basin occupancy

## Summary

Status: **done**. All predictor/state fits use only 10,691 non-targeting controls; the 865 targets are unchanged. Every energy/reference population and centroid shift uses all controls.
Measured recorded-unit CPU hours: 9.156238 / 12.
Refit-check median relative Frobenius difference: 0; required <=0.05 before use.
gate_H1_reweighting: FAIL; `{"median_explained_fraction": 0.3052877796665334, "required_median": 0.5, "n_exceeds_noise_p95": 235, "fraction_exceeds_noise_p95": 0.27167630057803466, "required_fraction": 0.5, "pass": false}`.
gate_H2_depth_sensitivity: FAIL; `{"rho": 0.05217725354718012, "p_one_sided_negative": 0.6486351364863514, "required_rho_less_than": 0, "required_p_less_than": 0.05, "pass": false}`.
The primary score-J and operator-free Leiden comparisons do not show the predicted negative association; neither demonstrates a predictive advantage in these requested tests.
Original K562 SHA-256: `3e5a63a9e892b21029bb55fca4e12517a49aad7af6c14133ca63d12cf68c6cee` (verified unchanged). Cloud-only execution; no Mac access, merge or push.

## What ran

- `.venv/bin/python scripts/s18b_occupancy.py verify-input` — exit 0, wall 68.738s, CPU 0.003058h.
- `.venv/bin/python -m pip install --no-cache-dir -e .[dev]` — exit 0, wall 173.485s, CPU 0.046509h.
- `.venv/bin/python -m pytest -q tests/test_orient.py tests/test_smoke.py` — exit 0, wall 21.015s, CPU 0.004508h.
- `.venv/bin/python scripts/s18b_occupancy.py refit --k 20` — exit 0, wall 181.073s, CPU 0.216464h.
- `.venv/bin/python scripts/s18b_occupancy.py states --k 20` — exit 0, wall 503.296s, CPU 0.145162h.
- `.venv/bin/python scripts/s18b_occupancy.py occupancy --k 20` — exit 0, wall 14.453s, CPU 0.008708h.
- `.venv/bin/python scripts/s18b_occupancy.py h1 --k 20` — exit 0, wall 1699.470s, CPU 2.390888h.
- `.venv/bin/python scripts/s18b_occupancy.py h2 --k 20` — exit 0, wall 2.770s, CPU 0.001141h.
- `.venv/bin/python scripts/s18b_occupancy.py refit --k 15` — exit 0, wall 134.857s, CPU 0.162823h.
- `.venv/bin/python scripts/s18b_occupancy.py states --k 15` — exit 0, wall 595.952s, CPU 0.171218h.
- `.venv/bin/python scripts/s18b_occupancy.py occupancy --k 15` — exit 0, wall 13.581s, CPU 0.007257h.
- `.venv/bin/python scripts/s18b_occupancy.py h1 --k 15` — exit 0, wall 1929.964s, CPU 2.788509h.
- `.venv/bin/python scripts/s18b_occupancy.py h2 --k 15` — exit 0, wall 2.118s, CPU 0.001112h.
- `.venv/bin/python scripts/s18b_occupancy.py refit --k 30` — exit 0, wall 258.924s, CPU 0.316027h.
- `.venv/bin/python scripts/s18b_occupancy.py states --k 30` — exit 0, wall 521.165s, CPU 0.150833h.
- `.venv/bin/python scripts/s18b_occupancy.py occupancy --k 30` — exit 0, wall 14.774s, CPU 0.008300h.
- `.venv/bin/python scripts/s18b_occupancy.py h1 --k 30` — exit 0, wall 1900.078s, CPU 2.707618h.
- `.venv/bin/python scripts/s18b_occupancy.py h2 --k 30` — exit 0, wall 2.127s, CPU 0.001144h.
- `.venv/bin/python scripts/s18b_occupancy.py aggregate` — exit 0, wall 8.884s, CPU 0.002477h.
- `.venv/bin/python -m pytest -q tests/test_orient.py tests/test_smoke.py` — exit 0, wall 16.806s, CPU 0.004135h.
- `.venv/bin/python scripts/s18b_occupancy.py verify-input` — exit 0, wall 13.448s, CPU 0.002958h.
- `.venv/bin/python scripts/s18b_occupancy.py audit` — exit 1, wall 8.843s, CPU 0.002586h.
- `.venv/bin/python scripts/s18b_occupancy.py repair-precision` — exit 0, wall 2.120s, CPU 0.000710h.
- `.venv/bin/python scripts/s18b_occupancy.py aggregate` — exit 0, wall 9.437s, CPU 0.002638h.
- `.venv/bin/python scripts/s18b_occupancy.py audit` — exit 0, wall 12.344s, CPU 0.003557h.
- `.venv/bin/python scripts/s18b_occupancy.py report --status done` — exit 0, wall 2.971s, CPU 0.000757h.
- `.venv/bin/python coord/coord_check.py check orient S18b --base 56e9c0045387bf6049dbd40c8b049ef2d55a4cae --run-reviews` — exit 0, wall 17.509s, CPU 0.004339h.
- `.venv/bin/python scripts/s18b_occupancy.py delivery-prepare` — exit 0, wall 2.922s, CPU 0.000802h.

Fresh `.venv`; pip freeze committed. Machine/environment: `{"python": "3.12.3", "platform": "Linux-6.18.38+-aarch64-with-glibc2.39", "cpu_count": 6, "ram_bytes": 14727913472, "fresh_venv": true, "preserved_previous_venv": ".venv-S18", "planner_amendment_commit": "56e9c0045387bf6049dbd40c8b049ef2d55a4cae"}`. Per-unit RAM/RSS and thread counts are in JSON checkpoints.

## Gate numbers

| Metric | Value | Requirement/result |
|---|---:|---|
| Primary basins | 59 | Report |
| Refit median relative Frobenius difference | 0 | <=0.05 before use |
| H1 median explained fraction | 0.30528778 | >=0.5 |
| H1 fraction above target-size null95 | 0.271676301 (235/865) | >=0.5; FAIL together |
| stiffness_partial rho | 0.0521772535 | p-=0.648635136; p-two=0.702029797 |
| stiffness_ordinary rho | 0.0598196935 | p-=0.676432357; p-two=0.641035896 |
| energy_depth_partial rho | -0.15084157 | p-=0.133386661; p-two=0.263273673 |
| energy_depth_ordinary rho | 0.0142024284 | p-=0.536446355; p-two=0.914308569 |
| leiden_ordinary rho | 0.0318527177 | p-=0.599440056; p-two=0.798120188 |
| Leiden partial rho/p | null | Undefined by construction; planner amendment |

### PCA dimension sensitivity (report-only for k=15/30)

| k | Basins | H1 median | H1 above noise | Partial stiffness rho | p-negative | Leiden ordinary rho | p-negative |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 59 | 0.305288 | 0.271676 | 0.052177 | 0.648635 | 0.031853 | 0.59944 |
| 15 | 66 | 0.335241 | 0.277457 | -0.136963 | 0.136586 | 0.067551 | 0.70243 |
| 30 | 65 | 0.225469 | 0.239306 | 0.014030 | 0.539846 | 0.120717 | 0.832917 |

H1 uses E=sqrt(2 cross mean minus within means), not squared E in the explained-fraction ratio. Weight optimization uses its monotone-equivalent squared objective, nonnegative weights summing to 1, and equal cell mass within each original basin. The baseline weights are basin occupancy n_b/10,691. No control subsampling or removal is performed. Control null draws only supply pseudo-targets. Partial Spearman residualizes ranks against an intercept, basin-size rank and covariance-trace rank; depth is permuted 10,000 times and re-residualized. Negative-tail p=(1+count(rho_perm<=rho_obs))/10,001; two-sided p uses absolute rho.

## Deviations

- Initial pre-result block resolved by the user's verbatim planner Amendment Revision1, committed as56e9c0045387bf6049dbd40c8b049ef2d55a4cae; Leiden partial is null, one-sided negative p is gated, two-sided absolute p is reported.
- Missing S18 model weights refitted only by the exact S18 adapter/configuration; embedding, cohort and accepted S18 caches were reused unchanged.
- Leiden graph/clustering arguments not specified by the task retain Scanpy defaults; resolution searched on controls only for exact matched state count, with seed0 and no nearest-count substitution.
- Small endpoint clusters merged iteratively into nearest endpoint-centroid cluster; smallest size/source-label order and stable tie rules are recorded. Original memberships define depth, size/spread and uniform H1 mixtures; all cells including controls use the prescribed15-vote classifier for occupancy.
- For identical target sizes, the same200 random-control null subsets are reused; each target has the exact prescribed size-conditional null. Full control pool remains in every mixture and energy denominator.
- H1 minimized exact squared energy by monotone-restart accelerated simplex projection; no statistical regularization. KKT gaps certify numerical optimization, and explained-fraction uncertainty is recorded.
- Only primary k=20 is gated; k=15/30 repeat the corresponding control-only PCA-k pipeline as report-only sensitivities. All centroid shifts use every control.
- Workstream mentions 3 xfails, but the supplied protected test file contains none; tests are executed unchanged and all 48 pass before and after analysis.
- Mechanical scientific-diff check uses the user-authorized planner-amendment commit as baseline because its expressly requested task-file edit lies outside scientific allowed_paths.
- An additional artifact audit caught four k30 sensitivity ranks coalesced by 12-significant-digit CSV formatting. Columns were restored from unchanged cached full-precision arrays, written with 17 digits and read round-trip. Original H1/H2 scientific JSONs, model/state/occupancy caches, correlations and permutation p-values are unchanged; no refits or resampling.

## Observations

- Original SHA matches; 10,691 controls and the same 865 significant targets; original inputs and caches are read-only.
- Primary exact refit median relative Frobenius difference: 0 (stop threshold0.05).
- Primary endpoints formed 7901 initial clusters; nearest-centroid minimum-size merging produced 59 basins. Fraction drift-stopped before/at400 steps: 0.277336.
- Most primary controls reached the fixed 400-step cap rather than the step-norm convergence criterion. The report-only energy depth uses the lowest observed terminal endpoint energy in each basin, not an additionally optimized stationary minimum. No extra integration or endpoint optimization was run.
- H1 median explained fraction 0.305288; above size-matched null95 for 235/865 targets.
- Primary H1 has 299 distinct target sizes and 59800 certified null optimizations. Maximum target explained-fraction numerical bracket width is 8.64e-08; maximum null bracket width is 1.04e-06.
- Operator-free Leiden spread-only comparison does not meet the negative-association significance criterion: ordinary rho=0.031853, p-=0.59944. This is not adjusted equivalently to score J because its partial correlation is undefined.
- Unadjusted score-J stiffness rho=0.059820, p-=0.676432.
- Both the primary score-J partial comparison and the operator-free Leiden ordinary comparison fail the predicted negative-association criterion. These requested tests do not demonstrate a predictive advantage for score J; they are not an equivalently adjusted paired superiority test.
- Completed output audit: pass; assigned-control QC by k and scheme: {"20": {"score_J": {"n_zero_vote_assigned_controls": 0, "min_vote_assigned_controls": 28}, "leiden": {"n_zero_vote_assigned_controls": 0, "min_vote_assigned_controls": 34}}, "15": {"score_J": {"n_zero_vote_assigned_controls": 0, "min_vote_assigned_controls": 31}, "leiden": {"n_zero_vote_assigned_controls": 0, "min_vote_assigned_controls": 10}}, "30": {"score_J": {"n_zero_vote_assigned_controls": 0, "min_vote_assigned_controls": 26}, "leiden": {"n_zero_vote_assigned_controls": 0, "min_vote_assigned_controls": 5}}}.

## Suggestions

- In a separately approved follow-up, assess endpoint-clustering and drift-convergence stability because most controls reached the 400-step cap and 7,901 initial endpoint clusters were merged to 59 basins; also assess original versus vote-classified memberships. No such sensitivity was run here.

## Questions

- None.

## Artifacts

`basins.csv`, `occupancy.csv`, `h1.csv`, `h2.json`, per-k tables, state/model/null caches, figures, unit logs/checkpoints and this report are in the returned benchmark download. Large/ignored files remain uncommitted and are identified by artifact size/hash in JSON. Temporary full-control distance matrices are not serialized. Original raw input/S18 caches are not duplicated; their exact checksums are in `input_verification.json`.

## Revision 1

The authorized planner amendment is appended verbatim to `coord/orient/tasks/S18b.md`. The initial blocked report is preserved in commitc87e793. No scientific result existed when the amendment was fixed.

## Reproduction and reuse

Clone the returned bundle on `manus/orient-S18b`, restore the benchmark ZIP at the repository root, and retain the original S18 inputs/caches at the paths listed in `input_verification.json`. The source file must match the stated SHA-256; restore S18 caches from the S18 benchmark download, not reconstructed or imputed arrays. Create a fresh `.venv` and install `-e ".[dev]"` as recorded. Saved S18b network weights and completed caches are reused; successful refits or null banks are never overwritten.

The ordered timed commands are in the checkpoints. In a genuinely fresh output directory, execute `verify-input`, then for k=20,15,30 the `refit`, `states`, `occupancy`, `h1`, and `h2` subcommands; finally `aggregate`, the required tests, `audit`, and report generation. Each numerical command uses `unit <name> -- .venv/bin/python scripts/s18b_occupancy.py <command>`. For an already restored result, use the saved state/model/occupancy caches for reporting and inspection rather than repeating fits. Exact file-specific regeneration/restoration commands accompany each JSON artifact.
