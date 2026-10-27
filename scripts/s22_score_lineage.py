#!/usr/bin/env python3
"""Fixed S22 LARRY score-J benchmark; cloud-only, checkpointed units."""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'LOKY_MAX_CPU_COUNT'):
    os.environ[_name] = '3'
os.environ['PYTHONUNBUFFERED'] = '1'

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'benchmarks/real_flow/s22'
LOGS = OUT / 'run_logs'
INPUT = ROOT / 'benchmarks/real_flow/data/larry_s22_inputs.npz'
INPUT_SHA256 = '74f01c2a9121531ac4a0af156c2b96e5be63d1af1957075377277f57249286c8'
BASE = 'e90440cfa3d87aa9c76ae0a0c6be697712ac6094'
BUDGET_CPU_SECONDS = 8 * 3600
MAX_PROCESS_RSS = 8 * 1024 ** 3


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def total_cpu_hours() -> float:
    return sum(float(json.loads(p.read_text()).get('cpu_hours', 0)) for p in LOGS.glob('*.checkpoint.json'))


def run_unit(name: str, command: list[str]) -> None:
    import psutil
    LOGS.mkdir(parents=True, exist_ok=True)
    checkpoint = LOGS / f'{name}.checkpoint.json'
    if checkpoint.exists():
        raise RuntimeError(f'Unit {name} already has a checkpoint; do not repeat silently.')
    if command and command[0] == '--':
        command = command[1:]
    if not command:
        raise ValueError('unit requires a command')
    used_seconds = total_cpu_hours() * 3600
    if used_seconds >= BUDGET_CPU_SECONDS - 12:
        raise RuntimeError('S22 CPU budget exhausted: stop and ask.')
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    wall, started = time.monotonic(), utc()
    reason, max_rss, last_progress = None, 0, wall
    with (LOGS / f'{name}.stdout.log').open('w') as stdout, (LOGS / f'{name}.stderr.log').open('w') as stderr:
        p = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, start_new_session=True, env=os.environ.copy())
        proc = psutil.Process(p.pid)
        print(f'{started} S22 unit {name} started; pid={p.pid}; command={command}', flush=True)
        while p.poll() is None:
            processes = [proc]
            try:
                processes += proc.children(recursive=True)
            except psutil.NoSuchProcess:
                pass
            cpu = 0.0
            for q in processes:
                try:
                    r = q.memory_info().rss
                    max_rss = max(max_rss, r)
                    t = q.cpu_times()
                    cpu += t.user + t.system
                    if r > MAX_PROCESS_RSS:
                        reason = 'process exceeded 8 GiB RSS'
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            if used_seconds + cpu >= BUDGET_CPU_SECONDS - 12:
                reason = 'CPU budget reserve reached; stop and ask'
            if reason:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                break
            if time.monotonic() - last_progress >= 30:
                print(f'S22 {name}: elapsed={time.monotonic()-wall:.1f}s CPU={cpu:.1f}s peak RSS={max_rss/1024**2:.1f} MiB', flush=True)
                last_progress = time.monotonic()
            time.sleep(1)
        rc = p.wait()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_seconds = after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
    receipt = {'task': 'orient/S22', 'unit': name, 'command': command, 'started_utc': started, 'finished_utc': utc(), 'wall_seconds': time.monotonic() - wall, 'cpu_hours': cpu_seconds / 3600, 'max_rss_bytes': max(max_rss, int(after.ru_maxrss) * 1024), 'exit_code': rc, 'stop_reason': reason, 'threads_limit': 3, 'process_rss_limit_bytes': MAX_PROCESS_RSS}
    write_json(checkpoint, receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    if rc or reason:
        raise SystemExit(rc if rc > 0 else 1)


def load_input() -> dict:
    import numpy as np
    assert sha256(INPUT) == INPUT_SHA256, 'Input SHA-256 changed: stop and ask.'
    with np.load(INPUT, allow_pickle=False) as raw:
        return {key: raw[key] for key in raw.files}


def guard() -> None:
    pr = json.loads((OUT / 'preregistration.json').read_text())
    am = json.loads((OUT / 'amendment_record.json').read_text())
    assert sha256(ROOT / 'coord/orient/tasks/S22.md') == am['task_sha256']
    assert sha256(ROOT / 'scripts/s11_operators.py') == pr['s11_sha256']
    assert sha256(ROOT / 'scjdo2/energy_drift.py') == pr['energy_adapter_sha256']
    assert am['tied_pairs_L1_included'] and am['tied_pairs_undifferentiated_and_per_type_excluded']
    assert total_cpu_hours() < 8


def preflight() -> None:
    import numpy as np
    import pandas as pd
    data = load_input()
    z, day, sample = data['z_all'], data['day'].astype(np.int64), data['sample_index'].astype(np.int64)
    cell, clone = data['clone_cell'].astype(np.int64), data['clone_id'].astype(np.int64)
    declared, labels = int(data['n_clones']), data['cell_type'].astype(str)
    assert z.shape == (130887, 30) and np.isfinite(z).all()
    assert day.shape == labels.shape == data['cell_id'].shape == (len(z),)
    assert set(np.unique(day)) == {2, 4, 6}
    assert len(sample) == 12000 and len(np.unique(sample)) == 12000
    assert np.all((sample >= 0) & (sample < len(z)))
    assert {int(k): int(v) for k, v in zip(*np.unique(day[sample], return_counts=True))} == {2: 4000, 4: 4000, 6: 4000}
    assert cell.shape == clone.shape and cell.ndim == 1
    assert np.all((cell >= 0) & (cell < len(z))) and np.all((clone >= 0) & (clone < declared))
    assert len(np.unique(np.stack([cell, clone], axis=1), axis=0)) == len(cell)
    rows = []
    for cid, indices in clone_groups(data):
        for t in (2, 4):
            current, future = indices[day[indices] == t], indices[day[indices] == t + 2]
            if len(current) < 2 or len(future) < 3:
                continue
            types, counts = np.unique(labels[current], return_counts=True)
            modes = types[counts == counts.max()]
            rows.append({'clone_id': cid, 't': t, 'n_current': len(current), 'n_future': len(future), 'n_modal_types': len(modes), 'modal_types': '|'.join(modes), 'undifferentiated_in_tie': bool(len(modes) > 1 and 'Undifferentiated' in modes)})
    eligibility = pd.DataFrame(rows)
    eligibility.to_csv(OUT / 'eligibility_preflight.csv', index=False)
    tied = eligibility[eligibility.n_modal_types > 1]
    payload = {'task': 'orient/S22', 'input_sha256': INPUT_SHA256, 'input_bytes': INPUT.stat().st_size, 'shape': list(z.shape), 'sample_cells_by_day': {str(k): int(v) for k, v in zip(*np.unique(day[sample], return_counts=True))}, 'n_clones_declared': declared, 'n_clones_observed': int(len(np.unique(clone))), 'n_membership_pairs': int(len(cell)), 'n_barcoded_day2_or_day4_cells': len(query_index(data)), 'n_eligible_clone_time_pairs': len(eligibility), 'n_eligible_clones': int(eligibility.clone_id.nunique()), 'n_tied_majorities': len(tied), 'n_undifferentiated_majority_ties': int(tied.undifferentiated_in_tie.sum()), 'versions': versions(), 'task_sha256': sha256(ROOT / 'coord/orient/tasks/S22.md'), 's11_code_sha256': sha256(ROOT / 'scripts/s11_operators.py'), 'energy_code_sha256': sha256(ROOT / 'scjdo2/energy_drift.py'), 'fit_has_run': False, 'entropy_or_correlation_has_been_computed': False}
    write_json(OUT / 'input_preflight.json', payload)
    print(json.dumps(payload, indent=2), flush=True)


def versions() -> dict:
    return dict({name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'pandas', 'scikit-learn', 'torch', 'pytest', 'matplotlib')}, python=platform.python_version())


def query_index(data: dict):
    import numpy as np
    cell = data['clone_cell'].astype(np.int64)
    return np.unique(cell[np.isin(data['day'][cell], [2, 4])])


def clone_groups(data: dict):
    import numpy as np
    clone, cell = data['clone_id'].astype(np.int64), data['clone_cell'].astype(np.int64)
    order = np.argsort(clone, kind='stable')
    for group in np.split(order, np.flatnonzero(np.diff(clone[order])) + 1):
        yield int(clone[group[0]]), cell[group]


def score_config(seed: int):
    from scjdo2.energy_drift import EnergyDriftConfig
    return EnergyDriftConfig(hidden=96, depth=3, n_epochs=1500, seed=seed, batch_size=256,
                             loss='esm', lr=5e-4, beta=1.0, esm_laplacian='exact')


def fit_score(seed: int) -> None:
    import numpy as np
    import pandas as pd
    import torch
    from scjdo2.energy_drift import EnergyDriftAdapter
    guard()
    model_path = OUT / f'model_seed{seed}.pt'
    assert not model_path.exists(), 'Saved weights exist: do not refit.'
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    data = load_input()
    train = data['z_all'][data['sample_index'].astype(np.int64)].astype(np.float32)
    index = query_index(data)
    cfg = score_config(seed)
    started = utc()
    fitted = EnergyDriftAdapter(30, cfg).fit(train, verbose=True)
    symmetry = fitted.symmetry_check(train, seed=0)
    if symmetry['max_asym'] > 1e-4 or not np.isfinite(list(symmetry.values())).all():
        write_json(OUT / f'fit_seed{seed}.json', {'seed': seed, 'config': asdict(cfg), 'symmetry': symmetry, 'symmetry_pass': False})
        raise RuntimeError('Frozen S11 symmetry check failed: stop and report.')
    torch.save({'state_dict': {k: v.detach().cpu() for k, v in fitted._model.state_dict().items()}, 'config': asdict(cfg), 'dimension': 30}, model_path)
    fit_meta = {'seed': seed, 'started_utc': started, 'fit_finished_utc': utc(), 'config': asdict(cfg), 'device': fitted._device, 'n_training_cells': len(train), 'training_array_sha256': hashlib.sha256(train.tobytes(order='C')).hexdigest(), 'sample_index_sha256': hashlib.sha256(data['sample_index'].tobytes(order='C')).hexdigest(), 'model_sha256': sha256(model_path), 'symmetry': symmetry, 'symmetry_pass': True, 'torch_threads': torch.get_num_threads(), 'n_query_cells': len(index)}
    write_json(OUT / f'fit_seed{seed}.json', fit_meta)
    eigenvalues = np.empty((len(index), 30), dtype=np.float32)
    max_query_asym = 0.0
    for start in range(0, len(index), 2048):
        stop = min(len(index), start + 2048)
        jac = fitted.jacobian_autodiff(data['z_all'][index[start:stop]].astype(np.float32), chunk=128)
        assert np.isfinite(jac).all()
        eigenvalues[start:stop] = np.linalg.eigvalsh(jac)
        den = np.linalg.norm(jac, axis=(1, 2)) + 1e-12
        asym = np.linalg.norm(jac - jac.transpose(0, 2, 1), axis=(1, 2)) / den
        max_query_asym = max(max_query_asym, float(asym.max()))
        print(f'S22 seed {seed} exact Jacobians/eigenspectra: {stop}/{len(index)}', flush=True)
    assert np.isfinite(eigenvalues).all()
    lambda1 = eigenvalues[:, -1]
    n_positive = (eigenvalues > 0).sum(axis=1).astype(np.int16)
    np.savez_compressed(OUT / f'cells_seed{seed}.npz', cell_index=index, eigenvalues=eigenvalues, lambda1=lambda1, n_positive=n_positive)
    pd.DataFrame({'cell_index': index, 'cell_id': data['cell_id'][index].astype(str), 'day': data['day'][index], 'cell_type': data['cell_type'][index].astype(str), 'lambda1': lambda1.astype(np.float64), 'n_positive': n_positive}).to_csv(OUT / f'cells_seed{seed}.csv', index=False, float_format='%.17g')
    meta = dict(fit_meta, query_finished_utc=utc(), query_max_asym_report_only=max_query_asym, eigenvalue_range=[float(eigenvalues.min()), float(eigenvalues.max())], lambda1_median=float(np.median(lambda1)), positive_eigen_count_distribution={str(k): int(v) for k, v in zip(*np.unique(n_positive, return_counts=True))}, cells_npz_sha256=sha256(OUT / f'cells_seed{seed}.npz'))
    write_json(OUT / f'score_seed{seed}.json', meta)
    print(json.dumps(meta, indent=2), flush=True)


def build_clones() -> None:
    import numpy as np
    import pandas as pd
    guard()
    data = load_input()
    day, labels = data['day'], data['cell_type'].astype(str)
    lambdas, counts = {}, {}
    for seed in (0, 1, 2):
        with np.load(OUT / f'cells_seed{seed}.npz', allow_pickle=False) as a:
            assert np.array_equal(a['cell_index'], query_index(data))
            lambdas[seed] = np.full(len(day), np.nan, dtype=np.float64)
            counts[seed] = np.full(len(day), np.nan, dtype=np.float64)
            lambdas[seed][a['cell_index']] = a['lambda1'].astype(np.float64)
            counts[seed][a['cell_index']] = a['n_positive']
    rows = []
    for cid, indices in clone_groups(data):
        for t in (2, 4):
            current, future = indices[day[indices] == t], indices[day[indices] == t + 2]
            if len(current) < 2 or len(future) < 3:
                continue
            types, current_counts = np.unique(labels[current], return_counts=True)
            modes = types[current_counts == current_counts.max()]
            is_tied = len(modes) > 1
            future_types, future_counts = np.unique(labels[future], return_counts=True)
            probabilities = future_counts.astype(np.float64) / len(future)
            entropy = float(-np.sum(probabilities * np.log(probabilities)))
            switch = float(np.mean(~np.isin(labels[future], modes)))
            row = {'clone_id': cid, 't': t, 'n_current': len(current), 'n_future': len(future), 'is_tied': is_tied, 'modal_types': json.dumps(modes.tolist()), 'majority_type': '' if is_tied else str(modes[0]), 'permutation_type': 'tied' if is_tied else str(modes[0]), 'is_undifferentiated': bool(not is_tied and modes[0] == 'Undifferentiated'), 'H': entropy, 'switch': switch, 'future_type_counts': json.dumps({str(k): int(v) for k, v in zip(future_types, future_counts)}, sort_keys=True)}
            for seed in (0, 1, 2):
                assert np.isfinite(lambdas[seed][current]).all()
                row[f'lambda1_seed{seed}'] = float(np.median(lambdas[seed][current]))
                row[f'n_positive_median_seed{seed}'] = float(np.median(counts[seed][current]))
            rows.append(row)
    df = pd.DataFrame(rows)
    original = pd.read_csv(OUT / 'eligibility_preflight.csv')
    assert np.array_equal(df[['clone_id', 't', 'n_current', 'n_future']].to_numpy(), original[['clone_id', 't', 'n_current', 'n_future']].to_numpy())
    assert int(df.is_tied.sum()) == 67 and not df.loc[df.is_tied, 'is_undifferentiated'].any()
    df.to_csv(OUT / 'clones_truth.csv', index=False, float_format='%.17g')
    summary = {'n_clone_time_pairs': len(df), 'n_unique_clones': int(df.clone_id.nunique()), 'n_pairs_per_t': {str(t): int((df.t == t).sum()) for t in (2, 4)}, 'n_tied_pairs': int(df.is_tied.sum()), 'n_tied_pairs_per_t': {str(t): int(((df.t == t) & df.is_tied).sum()) for t in (2, 4)}, 'n_undifferentiated_pairs': int(df.is_undifferentiated.sum()), 'n_undifferentiated_pairs_per_t': {str(t): int(((df.t == t) & df.is_undifferentiated).sum()) for t in (2, 4)}, 'n_untied_per_type': {str(k): int(v) for k, v in df.loc[~df.is_tied, 'majority_type'].value_counts().sort_index().items()}, 'entropy_range': [float(df.H.min()), float(df.H.max())], 'n_zero_entropy': int((df.H == 0).sum()), 'switch_median': float(df['switch'].median()), 'tied_switch_median': float(df.loc[df.is_tied, 'switch'].median()), 'amendment_applied': True}
    write_json(OUT / 'clones_summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def baselines() -> None:
    import numpy as np
    import pandas as pd
    from sklearn.neighbors import NearestNeighbors
    guard()
    data = load_input()
    z = data['z_all'].astype(np.float64)
    day, labels = data['day'], data['cell_type'].astype(str)
    index = query_index(data)
    centroid = z[(day == 2) & (labels == 'Undifferentiated')].mean(axis=0)
    assert np.isfinite(centroid).all()
    density = np.full(len(z), np.nan)
    distance = np.full(len(z), np.nan)
    distance[index] = np.linalg.norm(z[index] - centroid, axis=1)
    for t in (2, 4):
        reference = np.flatnonzero(day == t)
        queries = index[day[index] == t]
        local_row = np.full(len(z), -1, dtype=np.int64)
        local_row[reference] = np.arange(len(reference))
        nn = NearestNeighbors(n_neighbors=31, metric='euclidean', n_jobs=1).fit(z[reference])
        for start in range(0, len(queries), 512):
            ix = queries[start:start + 512]
            dist, neighbours = nn.kneighbors(z[ix])
            means = np.array([dist[j][neighbours[j] != local_row[cell]][:30].mean() for j, cell in enumerate(ix)])
            assert np.isfinite(means).all() and (means > 0).all(), 'Undefined fixed density baseline: stop.'
            density[ix] = -np.log(means)
            print(f'S22 day {t} all-same-day 30-neighbour baseline: {min(start+512,len(queries))}/{len(queries)}; reference n={len(reference)}', flush=True)
    np.savez_compressed(OUT / 'cell_baselines.npz', cell_index=index, density=density[index], centroid_distance=distance[index], undifferentiated_day2_centroid=centroid)
    df = pd.read_csv(OUT / 'clones_truth.csv', keep_default_na=False, float_precision='round_trip')
    group_map = dict(clone_groups(data))
    d, c = [], []
    for row in df.itertuples():
        current = group_map[row.clone_id][day[group_map[row.clone_id]] == row.t]
        d.append(float(np.median(density[current])))
        c.append(float(np.median(distance[current])))
    df['density'] = d
    df['centroid_distance'] = c
    assert np.isfinite(df[['density', 'centroid_distance']].to_numpy()).all()
    df.to_csv(OUT / 'clones.csv', index=False, float_format='%.17g')
    write_json(OUT / 'baselines.json', {'n_query_cells': len(index), 'n_centroid_cells': int(((day == 2) & (labels == 'Undifferentiated')).sum()), 'n_reference_cells_per_t': {str(t): int((day == t).sum()) for t in (2, 4)}, 'k_neighbours': 30, 'focal_cell_excluded': True, 'metric': 'euclidean', 'coordinates': 'delivered z_all, float64 distance arithmetic; no re-embedding', 'centroid': centroid.tolist(), 'clones_csv_sha256': sha256(OUT / 'clones.csv'), 'cell_baselines_sha256': sha256(OUT / 'cell_baselines.npz')})


def rho(x, y):
    import numpy as np
    from scipy.stats import spearmanr
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    result = float(spearmanr(x, y).statistic)
    return result if np.isfinite(result) else None


def partial_rho(x, y, density):
    import numpy as np
    from scipy.stats import rankdata
    if len(x) < 3:
        return None
    x, y, z = [rankdata(a, method='average') for a in (x, y, density)]
    design = np.column_stack([np.ones(len(z)), z])
    rank = np.linalg.matrix_rank(design)
    if np.linalg.matrix_rank(np.column_stack([design, x])) == rank or np.linalg.matrix_rank(np.column_stack([design, y])) == rank:
        return None
    rx = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    result = float(np.corrcoef(rx, ry)[0, 1])
    return result if np.isfinite(result) else None


def permutation_test(df, x, y):
    import numpy as np
    from scipy.stats import rankdata
    observed = rho(x, y)
    if observed is None:
        return None, None
    x = np.asarray(x, dtype=np.float64)
    rx = rankdata(x, method='average')
    ry = rankdata(y, method='average')
    rx -= rx.mean()
    ry -= ry.mean()
    denominator = np.linalg.norm(rx) * np.linalg.norm(ry)
    groups = []
    for key in sorted(set(zip(df.t.tolist(), df.permutation_type.tolist()))):
        groups.append(np.flatnonzero((df.t.to_numpy() == key[0]) & (df.permutation_type.to_numpy() == key[1])))
    rng = np.random.default_rng(0)
    null = np.empty(1000)
    for i in range(1000):
        permuted = rx.copy()
        for group in groups:
            permuted[group] = rx[rng.permutation(group)]
        null[i] = np.dot(permuted, ry) / denominator
    # Compare to the same centered-rank dot product used for the null,
    # avoiding a self-tie changed only by scipy's correlation roundoff.
    observed_for_tail = np.dot(rx, ry) / denominator
    p = float((1 + np.count_nonzero(null >= observed_for_tail)) / 1001)
    return p, null


def cohorts(df):
    import numpy as np
    tied = df.is_tied.to_numpy(dtype=bool)
    undiff = df.is_undifferentiated.to_numpy(dtype=bool)
    result = [('pooled', np.ones(len(df), dtype=bool)), ('undifferentiated', undiff), ('pooled_without_tied', ~tied)]
    for t in (2, 4):
        result.extend([(f't{t}', df.t.to_numpy() == t), (f'undifferentiated_t{t}', undiff & (df.t.to_numpy() == t))])
    for label in sorted(set(df.loc[~df.is_tied, 'majority_type'])):
        if label != 'Undifferentiated':
            result.append(('type:' + label, (~tied) & (df.majority_type.to_numpy() == label)))
    return result


def scores(seed: int) -> None:
    import numpy as np
    import pandas as pd
    guard()
    df = pd.read_csv(OUT / 'clones.csv', keep_default_na=False, float_precision='round_trip')
    rows, nulls = [], {}
    for name, mask in cohorts(df):
        d = df.loc[mask].reset_index(drop=True)
        x, y = d[f'lambda1_seed{seed}'].to_numpy(), d.H.to_numpy()
        p, null = permutation_test(d, x, y)
        row = {'seed': seed, 'cohort': name, 'n_pairs': len(d), 'n_unique_clones': int(d.clone_id.nunique()), 'rho_lambda1_H': rho(x, y), 'p_permutation_positive': p, 'rho_density_H': rho(d.density.to_numpy(), y), 'rho_centroid_distance_H': rho(d.centroid_distance.to_numpy(), y), 'partial_rho_given_density': partial_rho(x, y, d.density.to_numpy()), 'rho_lambda1_switch_report_only': rho(x, d['switch'].to_numpy()), 'n_tied_pairs': int(d.is_tied.sum()), 'gated': bool(seed == 0 and name in ('pooled', 'undifferentiated'))}
        rows.append(row)
        if null is not None:
            nulls[name] = null
    pd.DataFrame(rows).to_csv(OUT / f'scores_seed{seed}.csv', index=False, float_format='%.17g')
    np.savez_compressed(OUT / f'permutation_nulls_seed{seed}.npz', **nulls)
    write_json(OUT / f'statistics_seed{seed}.json', {'seed': seed, 'rows': rows, 'permutation_rule': '1000 permutations; default_rng(0) reset for each reported cohort; restricted by (t, permutation_type), with tied as its own per-t group; one-sided positive +1 tail', 'partial_rule': 'average ranks; separate OLS residuals of ranked lambda1 and H on intercept plus ranked density; Pearson correlation', 'clones_csv_sha256': sha256(OUT / 'clones.csv')})
    print(json.dumps({'seed': seed, 'rows': rows}, indent=2), flush=True)


def finite_positive(x) -> bool:
    return x is not None and x > 0


def summarize() -> None:
    import pandas as pd
    guard()
    all_rows, seed_rows = [], []
    for seed in (0, 1, 2):
        rows = json.loads((OUT / f'statistics_seed{seed}.json').read_text())['rows']
        all_rows.extend(rows)
        by = {r['cohort']: r for r in rows}
        seed_rows.append({'seed': seed, 'gated': seed == 0, 'pooled': by['pooled'], 'undifferentiated': by['undifferentiated'], 'pooled_without_tied_report_only': by['pooled_without_tied']})
    pd.DataFrame(all_rows).to_csv(OUT / 'scores.csv', index=False, float_format='%.17g')
    primary = seed_rows[0]
    pool, undiff = primary['pooled'], primary['undifferentiated']
    l1 = dict(pool, passed=bool(finite_positive(pool['rho_lambda1_H']) and pool['p_permutation_positive'] is not None and pool['p_permutation_positive'] < 0.01))
    l2 = dict(undiff, passed=bool(finite_positive(undiff['rho_lambda1_H']) and undiff['p_permutation_positive'] is not None and undiff['p_permutation_positive'] < 0.01))
    l3 = {'seed': 0, 'n_pairs': undiff['n_pairs'], 'rho_lambda1_H': undiff['rho_lambda1_H'], 'rho_density_H': undiff['rho_density_H'], 'rho_centroid_distance_H': undiff['rho_centroid_distance_H'], 'partial_rho_given_density': undiff['partial_rho_given_density']}
    r, b1, b2 = undiff['rho_lambda1_H'], undiff['rho_density_H'], undiff['rho_centroid_distance_H']
    l3['passed'] = bool(r is not None and b1 is not None and b2 is not None and abs(r) > abs(b1) and abs(r) > abs(b2) and finite_positive(undiff['partial_rho_given_density']))
    summary = {'gate_L1_pooled': l1, 'gate_L2_undifferentiated': l2, 'gate_L3_baselines': l3, 'n_clones': json.loads((OUT / 'clones_summary.json').read_text()), 'seed_robustness': seed_rows, 'versions': versions(), 'amendment': json.loads((OUT / 'amendment_record.json').read_text()), 'input_sha256': sha256(INPUT)}
    write_json(OUT / 'summary.json', summary)
    figures()
    report('partial')
    print(json.dumps({k: summary[k] for k in ('gate_L1_pooled', 'gate_L2_undifferentiated', 'gate_L3_baselines')}, indent=2), flush=True)


def fmt(x) -> str:
    return 'null' if x is None else f'{x:.6g}'


def figures() -> None:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    data = load_input()
    with np.load(OUT / 'cells_seed0.npz', allow_pickle=False) as a:
        index, lambda1 = a['cell_index'], a['lambda1']
    df = pd.read_csv(OUT / 'clones.csv', keep_default_na=False, float_precision='round_trip')
    folder = ROOT / 'benchmarks/figures/real_flow'
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    norm = Normalize(vmin=float(lambda1.min()), vmax=float(lambda1.max()))
    for ax, t in zip(axes, (2, 4)):
        mask = data['day'][index] == t
        bg = data['z_all'][data['day'] == t]
        ax.scatter(bg[:, 0], bg[:, 1], s=1, c='#dddddd', rasterized=True)
        z = data['z_all'][index[mask]]
        artist = ax.scatter(z[:, 0], z[:, 1], c=lambda1[mask], s=4, cmap='viridis', norm=norm, rasterized=True)
        ax.set(title=f'Day {t}: {mask.sum():,} barcoded cells', xlabel='Delivered PC1', ylabel='Delivered PC2')
    fig.colorbar(artist, ax=axes, label='Largest score-J eigenvalue, seed 0')
    fig.suptitle('S22: exact score J at all barcoded day-2/day-4 cells')
    fig.savefig(folder / 's22_lambda1_embedding.png', dpi=180)
    plt.close(fig)
    stats = {r['cohort']: r for r in json.loads((OUT / 'statistics_seed0.json').read_text())['rows']}
    panels = [('pooled', 'Pooled (L1; ties retained)'), ('undifferentiated', 'Undifferentiated (L2/L3; no ties)'), ('t2', 'Day 2 → 4 (report only)'), ('t4', 'Day 4 → 6 (report only)')]
    masks = dict(cohorts(df))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, (name, title) in zip(axes.flat, panels):
        d = df.loc[masks[name]]
        for t, color in ((2, '#2374ab'), (4, '#d9692a')):
            m = (d.t == t) & ~d.is_tied
            ax.scatter(d.loc[m, 'lambda1_seed0'], d.loc[m, 'H'], s=10, alpha=0.55, c=color, label=f't={t}')
        tied = d.is_tied
        if tied.any():
            ax.scatter(d.loc[tied, 'lambda1_seed0'], d.loc[tied, 'H'], s=18, marker='x', c='#555555', label='tied (no majority)')
        r = stats[name]
        ax.set(title=f'{title}\nn={len(d):,}; ρ={fmt(r["rho_lambda1_H"])}; p={fmt(r["p_permutation_positive"])}', xlabel='Clone median largest eigenvalue, seed 0', ylabel='Next-time cell-type entropy H (nats)')
        ax.legend(fontsize=8, loc='best')
    fig.suptitle('S22 fixed clonal fate-coherence test; restricted positive-tail permutation p')
    fig.savefig(folder / 's22_clonal_coherence.png', dpi=180)
    plt.close(fig)
    labels = sorted(set(df.loc[~df.is_tied, 'majority_type']))
    columns = 3
    rows = (len(labels) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(12, rows * 3.2), constrained_layout=True, squeeze=False)
    for ax, label in zip(axes.flat, labels):
        d = df.loc[(~df.is_tied) & (df.majority_type == label)]
        for t, color in ((2, '#2374ab'), (4, '#d9692a')):
            m = d.t == t
            ax.scatter(d.loc[m, 'lambda1_seed0'], d.loc[m, 'H'], s=13, alpha=0.65, c=color, label=f't={t}')
        ax.set(title=f'{label}: n={len(d)}', xlabel='Clone median λ₁', ylabel='Next-time H (nats)')
    for ax in list(axes.flat)[len(labels):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('S22 per-type strata (ties excluded); seed 0, report only except Undifferentiated')
    fig.savefig(folder / 's22_coherence_by_type.png', dpi=180)
    plt.close(fig)


def report(status: str) -> None:
    summary_path = OUT / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    pre = json.loads((OUT / 'input_preflight.json').read_text())
    cpu = total_cpu_hours()
    receipts = [json.loads(p.read_text()) for p in LOGS.glob('*.checkpoint.json')]
    metrics = {k: summary.get(k) for k in ('gate_L1_pooled', 'gate_L2_undifferentiated', 'gate_L3_baselines', 'n_clones', 'seed_robustness')}
    metrics.update(versions=versions(), cpu_hours=cpu)
    if metrics['n_clones'] is None:
        metrics['n_clones'] = {'eligible_unique': pre['n_eligible_clones'], 'eligible_pairs': pre['n_eligible_clone_time_pairs']}
    observations = ['The fit sample is exactly 12,000 delivered rows: 4,000 each on days 2, 4 and 6.', 'Every one of the 19,623 barcoded day-2/day-4 cells was included in each completed score-J query; no lineage/type filtering was used for fitting.'] if summary else ['The verified input contains 19,623 barcoded day-2/day-4 query cells. Scientific units are pending.']
    for seed in (0, 1, 2):
        score_path = OUT / f'score_seed{seed}.json'
        if score_path.exists():
            saved = json.loads(score_path.read_text())
            observations.append(f"Seed {seed}: max symmetry asymmetry {saved['symmetry']['max_asym']:.6g}; median cell lambda1 {saved['lambda1_median']:.6g}; positive-eigenvalue count distribution (count: cells) {saved['positive_eigen_count_distribution']}.")
    if summary:
        n = summary['n_clones']
        observations += [f"The eligible cohort has {n['n_clone_time_pairs']} clone/time pairs from {n['n_unique_clones']} distinct clones; {n['n_tied_pairs']} tied pairs were retained in L1 and report-only per-t pooled tests in their own per-t permutation strata, and excluded from Undifferentiated and per-type tests.", f"Tied pairs per t: {n['n_tied_pairs_per_t']}; untied Undifferentiated pairs per t: {n['n_undifferentiated_pairs_per_t']}.", f"{n['n_zero_entropy']} pairs have H=0; observed entropy range is {n['entropy_range']}."]
    deviations = ['Planner Amendment 1 was supplied on 2026-10-24 and appended/committed before all score fits and fate statistics; no majority label was chosen for tied pairs.', 'The unchanged delivered tests have 48 passing tests and no xfails; the workstream text mentions three strict xfails.', 'The cloud bundle imported a newer issue snapshot than the local orientation ref. Mechanical review uses the authorized planner amendment commit as --base; the full execution diff is also audited against e90440c, allowing only the exact authorized task amendment. No merge, push or Mac access occurred.']
    if not summary:
        deviations.append('Scientific gate metrics are null because scientific units are not yet complete.')
    if (OUT / 'audit_import_repair.json').exists():
        deviations.append('The auxiliary final audit initially failed before numerical checks because coord is not an importable package from the script directory. Only audit module loading was repaired to use the unchanged checker by absolute repository path; the failed receipt is preserved. Scientific function bodies and all saved scientific files were verified unchanged, and no model was refitted.')
    if (OUT / 'report_cache_filter_repair.json').exists():
        deviations.append('A pre-commit artifact assertion caught autogenerated, ignored Python bytecode incorrectly included in the report manifest. The bytecode cache was excluded from the manifest only; saved scientific files and function bodies were verified unchanged. The completed checker/test receipt was retained and final review rerun after committing the corrected reports.')
    report_dir = ROOT / 'coord/orient/reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    gate_lines = ['| Gate | Fixed condition | Observed seed-0 result | Pass |', '|---|---|---|---|']
    if summary:
        a, b, c = [summary[k] for k in ('gate_L1_pooled', 'gate_L2_undifferentiated', 'gate_L3_baselines')]
        gate_lines += [f'| L1 | pooled ρ > 0; p < 0.01 | ρ={fmt(a["rho_lambda1_H"])}; p={fmt(a["p_permutation_positive"])}; n={a["n_pairs"]} | {a["passed"]} |', f'| L2 | Undifferentiated ρ > 0; p < 0.01 | ρ={fmt(b["rho_lambda1_H"])}; p={fmt(b["p_permutation_positive"])}; n={b["n_pairs"]} | {b["passed"]} |', f'| L3 | abs(ρλ) > both abs(ρbaseline); partial ρ > 0 | λ={fmt(c["rho_lambda1_H"])}; density={fmt(c["rho_density_H"])}; centroid={fmt(c["rho_centroid_distance_H"])}; partial={fmt(c["partial_rho_given_density"])} | {c["passed"]} |']
    else:
        gate_lines += [f'| {g} | fixed task condition | not yet evaluated | null |' for g in ('L1', 'L2', 'L3')]
    robustness = ''
    if summary:
        robustness = '\n### Seeds 1–2 and removal of ties — report only\n\n| Seed | Pooled ρ / p | Undifferentiated ρ / p | Partial ρ | Pooled without ties ρ / p |\n|---|---|---|---|---|\n'
        for r in summary['seed_robustness']:
            a, b, c = r['pooled'], r['undifferentiated'], r['pooled_without_tied_report_only']
            robustness += f'| {r["seed"]} | {fmt(a["rho_lambda1_H"])} / {fmt(a["p_permutation_positive"])} | {fmt(b["rho_lambda1_H"])} / {fmt(b["p_permutation_positive"])} | {fmt(b["partial_rho_given_density"])} | {fmt(c["rho_lambda1_H"])} / {fmt(c["p_permutation_positive"])} |\n'
    command_lines = []
    for r in sorted(receipts, key=lambda r: r['started_utc']):
        command_lines.append(f'- `{r["unit"]}`: exit {r["exit_code"]}; {r["wall_seconds"]:.3f} seconds wall; {r["cpu_hours"]:.8f} CPU-hours. Exact argv/stdout/stderr are in `run_logs/`.')
    state = 'All fixed scientific units completed; gates are reported unchanged.' if summary else 'Amendment adopted; scientific execution is pending.'
    md = f'# orient/S22 — {status}\n\n## Summary\n\n{state} Input SHA-256 is `{INPUT_SHA256}`. Gated results use seed 0 only; seeds 1–2 and the untied-only pooled check are report only. No tuning was performed.\n\n## What ran\n\n' + '\n'.join(command_lines) + f'\n\nCompleted measured unit CPU use: {cpu:.8f} / 8 CPU-hours.\n\n### Fixed methods\n\nFresh S11-exact EnergyDriftAdapter fits on the delivered 12,000-row sample, seeds 0, 1 and 2. Autodiff Jacobians use the protected adapter, internal chunk 128; eigenvalues use `eigvalsh` directly. The S11 symmetry check is on 100 training rows with check seed 0, threshold max_asym ≤ 1e-4. Clone medians are computed over all its day-t cells; H uses natural logarithms over next-time type frequencies. Tied pairs receive no majority label; their permutation type is `tied`, separately by t, and switch counts next-time types outside all tied modal types. Baselines use all same-day cells, exclude the focal cell from 30 Euclidean neighbours, and use the all-day-2 Undifferentiated centroid. Partial Spearman uses average ranks and OLS residualization on ranked density plus an intercept. Each reported cohort resets `default_rng(0)` for 1,000 restricted permutations and uses the positive one-sided +1 tail; undefined correlations are null, never imputed.\n\n## Gate numbers\n\n' + '\n'.join(gate_lines) + '\n' + robustness + '\n## Deviations\n\n' + '\n'.join('- ' + x for x in deviations) + '\n\n## Observations\n\n' + '\n'.join('- ' + x for x in observations) + '\n\n## Suggestions\n\nNone. Only the issued task and planner amendment were executed.\n\n## Questions\n\nNone.\n\n## Artifacts\n\nThe result bundle contains the S22 script, exact saved model weights, per-cell eigenspectra and λ₁/eigen-count tables, clone truth and baseline tables, restricted permutation null arrays, summary statistics, figures and unit receipts. Input NPZ and raw H5ADs are not included. Artifact sizes/SHA-256 values and regeneration commands are in the JSON report. To reproduce, restore the verified input to the task data path, install the project into its venv, and run `fit --seed 0/1/2`, `clones`, `baselines`, `scores --seed 0/1/2`, and `summarize` as separately timed units in a fresh tree; never refit over existing saved weights.\n'
    (report_dir / 'S22.md').write_text(md)
    (OUT / 'RESULT_S22.md').write_text(md)
    tracked = set(subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines())
    paths = [ROOT / 'scripts/s22_score_lineage.py'] + [p for p in OUT.rglob('*') if p.is_file() and not any(v in p.relative_to(OUT).parts for v in ('tmp', 'cache', 'delivery', '__pycache__'))] + list((ROOT / 'benchmarks/figures/real_flow').glob('s22_*'))
    artifacts = []
    for p in sorted(paths):
        rel = str(p.relative_to(ROOT))
        if p.suffix == '.pt' or p.name.startswith(('cells_', 'fit_', 'score_')):
            regenerate = f'.venv/bin/python scripts/s22_score_lineage.py fit --seed {p.stem.split("seed")[-1]}'
        elif p.name.startswith(('statistics_', 'scores_seed', 'permutation_nulls_')):
            regenerate = f'.venv/bin/python scripts/s22_score_lineage.py scores --seed {p.stem.split("seed")[-1]}'
        elif p.name in ('clones_truth.csv', 'clones_summary.json'):
            regenerate = '.venv/bin/python scripts/s22_score_lineage.py clones'
        elif p.name in ('clones.csv', 'cell_baselines.npz', 'baselines.json'):
            regenerate = '.venv/bin/python scripts/s22_score_lineage.py baselines'
        elif p.parent == LOGS:
            regenerate = 'Exact argv in the matching unit checkpoint; run in a fresh restoration'
        else:
            regenerate = '.venv/bin/python scripts/s22_score_lineage.py summarize (or restore tracked provenance/source from this branch)'
        artifacts.append({'path': rel, 'bytes': p.stat().st_size, 'sha256': sha256(p), 'committed': rel in tracked, 'regenerate': regenerate})
    rep = {'task': 'orient/S22', 'status': status, 'started_utc': min(r['started_utc'] for r in receipts), 'finished_utc': utc(), 'compute': {'cpu_hours': cpu, 'gpu_hours': 0.0, 'hardware': 'Linux cloud sandbox, 6 visible CPUs, 3-thread cap, 8 GiB RSS/process'}, 'metrics': metrics, 'artifacts': artifacts, 'deviations': deviations, 'questions': [], 'observations': observations, 'suggestions': [], 'validation': {r['unit']: {'exit_code': r['exit_code'], 'cpu_hours': r['cpu_hours']} for r in receipts if any(k in r['unit'] for k in ('tests', 'audit', 'verify', 'coord_check'))}}
    write_json(report_dir / 'S22.json', rep)


def audit() -> None:
    import numpy as np
    import pandas as pd
    import importlib.util
    spec = importlib.util.spec_from_file_location('s22_coord_check', ROOT / 'coord/coord_check.py')
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    matches, read_task = checker.matches, checker.read_task
    guard()
    data = load_input()
    df = pd.read_csv(OUT / 'clones.csv', keep_default_na=False, float_precision='round_trip')
    assert len(df) == 1818 and int(df.is_tied.sum()) == 67
    assert (df.loc[df.is_tied, 'majority_type'] == '').all()
    assert (df.loc[df.is_tied, 'permutation_type'] == 'tied').all()
    assert not df.loc[df.is_tied, 'is_undifferentiated'].any()
    assert np.isfinite(df.H).all() and np.isfinite(df['switch']).all()
    by_clone = dict(clone_groups(data))
    for seed in (0, 1, 2):
        meta = json.loads((OUT / f'score_seed{seed}.json').read_text())
        assert meta['config'] == json.loads(json.dumps(asdict(score_config(seed))))
        assert meta['symmetry_pass'] and meta['symmetry']['max_asym'] <= 1e-4
        assert sha256(OUT / f'model_seed{seed}.pt') == meta['model_sha256']
        assert sha256(OUT / f'cells_seed{seed}.npz') == meta['cells_npz_sha256']
        with np.load(OUT / f'cells_seed{seed}.npz', allow_pickle=False) as a:
            assert np.array_equal(a['cell_index'], query_index(data))
            assert np.array_equal(a['lambda1'], a['eigenvalues'][:, -1])
            assert np.array_equal(a['n_positive'], (a['eigenvalues'] > 0).sum(axis=1))
            values = np.full(len(data['day']), np.nan)
            values[a['cell_index']] = a['lambda1'].astype(np.float64)
        for row in df.itertuples():
            ids = by_clone[row.clone_id]
            current = ids[data['day'][ids] == row.t]
            assert float(np.median(values[current])) == getattr(row, f'lambda1_seed{seed}')
        stats = json.loads((OUT / f'statistics_seed{seed}.json').read_text())
        for name, mask in cohorts(df):
            d = df.loc[mask].reset_index(drop=True)
            saved = next(r for r in stats['rows'] if r['cohort'] == name)
            observed = rho(d[f'lambda1_seed{seed}'].to_numpy(), d.H.to_numpy())
            assert observed == saved['rho_lambda1_H']
            p, _ = permutation_test(d, d[f'lambda1_seed{seed}'].to_numpy(), d.H.to_numpy())
            assert p == saved['p_permutation_positive']
    for row in df.itertuples():
        ids = by_clone[row.clone_id]
        future = ids[data['day'][ids] == row.t + 2]
        current = ids[data['day'][ids] == row.t]
        types, c = np.unique(data['cell_type'][current].astype(str), return_counts=True)
        modes = types[c == c.max()].tolist()
        assert modes == json.loads(row.modal_types)
        _, c = np.unique(data['cell_type'][future].astype(str), return_counts=True)
        p = c.astype(np.float64) / len(future)
        assert float(-np.sum(p*np.log(p))) == row.H
        assert float(np.mean(~np.isin(data['cell_type'][future].astype(str), modes))) == row.switch
    _, contract = read_task('orient', 'S22')
    changed = subprocess.check_output(['git', 'diff', '--name-only', BASE, 'HEAD'], cwd=ROOT, text=True).splitlines()
    for path in changed:
        assert path == 'coord/orient/tasks/S22.md' or matches(path, contract['allowed_paths']), path
    am = json.loads((OUT / 'amendment_record.json').read_text())
    assert subprocess.check_output(['git', 'show', am['planner_commit'] + ':coord/orient/tasks/S22.md'], cwd=ROOT) == (ROOT / 'coord/orient/tasks/S22.md').read_bytes()
    for path in ('scjdo2/energy_drift.py', 'scripts/s11_operators.py', 'tests/test_orient.py', 'tests/test_smoke.py', 'coord/orient/WORKSTREAM.md'):
        assert subprocess.check_output(['git', 'show', BASE + ':' + path], cwd=ROOT) == (ROOT / path).read_bytes(), path
    write_json(OUT / 'final_audit.json', {'passed': True, 'input_sha256': sha256(INPUT), 'all_19623_query_cells_preserved_per_seed': True, 'saved_eigenspectra_and_clone_medians_verified': True, 'ties_and_switch_verified_against_delivered_labels': True, 'published_rhos_and_1000_permutation_p_values_reproduced_from_round_trip_csv': True, 'all_execution_paths_authorized': True, 'protected_sources_unchanged': True, 'cpu_hours_before_this_receipt': total_cpu_hours()})
    print('S22 final numerical and scope audit PASS', flush=True)


def mechanical_review() -> None:
    am = json.loads((OUT / 'amendment_record.json').read_text())
    env = os.environ.copy()
    env['PATH'] = str(ROOT / '.venv/bin') + os.pathsep + env.get('PATH', '')
    subprocess.run([str(ROOT / '.venv/bin/python'), 'coord/coord_check.py', 'check', 'orient', 'S22', '--base', am['planner_commit'], '--run-reviews'], cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    unit = sub.add_parser('unit')
    unit.add_argument('name')
    unit.add_argument('argv', nargs=argparse.REMAINDER)
    for name in ('preflight', 'clones', 'baselines', 'summarize', 'figures', 'audit', 'mechanical-review'):
        sub.add_parser(name)
    for name in ('fit', 'scores'):
        p = sub.add_parser(name)
        p.add_argument('--seed', type=int, choices=(0, 1, 2), required=True)
    p = sub.add_parser('report')
    p.add_argument('--status', choices=('partial', 'done'), default='partial')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == 'unit':
        run_unit(args.name, args.argv)
    elif args.command == 'preflight':
        preflight()
    elif args.command == 'fit':
        fit_score(args.seed)
    elif args.command == 'clones':
        build_clones()
    elif args.command == 'baselines':
        baselines()
    elif args.command == 'scores':
        scores(args.seed)
    elif args.command == 'summarize':
        summarize()
    elif args.command == 'figures':
        figures()
    elif args.command == 'report':
        report(args.status)
    elif args.command == 'audit':
        audit()
    elif args.command == 'mechanical-review':
        mechanical_review()


if __name__ == '__main__':
    main()
