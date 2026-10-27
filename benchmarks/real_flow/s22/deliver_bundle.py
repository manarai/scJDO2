#!/usr/bin/env python3
"""Finalize the requested incremental S22 bundle on a clean branch and upload it once."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'benchmarks/real_flow/s22'
DELIVERY = OUT / 'delivery'
BASE = 'e90440cfa3d87aa9c76ae0a0c6be697712ac6094'


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open('rb') as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def run(command, cwd=ROOT):
    if command[0] == 'git':
        command = ['git', '-c', 'pack.threads=3'] + command[1:]
    print('RUN', command, flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def gws(args):
    r = subprocess.run(['gws'] + args, cwd=ROOT, capture_output=True, text=True, timeout=300)
    if r.returncode:
        raise RuntimeError(f'gws failed ({r.returncode}): {r.stderr}')
    return json.loads(r.stdout)


def main():
    DELIVERY.mkdir(parents=True, exist_ok=True)
    started, wall, rc = utc(), time.monotonic(), 1
    checkpoint = DELIVERY / 'delivery.checkpoint.json'
    assert not checkpoint.exists(), 'Delivery checkpoint exists: inspect it; do not repeat.'
    stage = 'preflight'
    try:
        assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
        assert subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip() == 'manus/orient-S22'
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        report = json.loads((ROOT / 'coord/orient/reports/S22.json').read_text())
        assert report['status'] == 'done' and report['compute']['cpu_hours'] < 8
        assert json.loads((OUT / 'final_audit.json').read_text())['passed']
        for name in ('tests_after', 'coord_check'):
            assert json.loads((OUT / 'run_logs' / f'{name}.checkpoint.json').read_text())['exit_code'] == 0
        stage = 'bundle_create'
        bundle = DELIVERY / 'S22_result.bundle'
        assert not bundle.exists()
        run(['git', 'bundle', 'create', str(bundle), BASE + '..manus/orient-S22'])
        run(['git', 'bundle', 'verify', str(bundle)])
        heads = subprocess.check_output(['git', 'bundle', 'list-heads', str(bundle)], cwd=ROOT, text=True).strip()
        assert heads == head + ' refs/heads/manus/orient-S22'
        stage = 'offline_restore'
        verify = DELIVERY / 'offline_verify'
        assert not verify.exists()
        run(['git', 'init', str(verify)])
        run(['git', 'fetch', '--no-tags', str(ROOT), BASE + ':refs/heads/prerequisite'], cwd=verify)
        run(['git', 'bundle', 'verify', str(bundle)], cwd=verify)
        run(['git', 'fetch', str(bundle), 'manus/orient-S22:manus/orient-S22'], cwd=verify)
        run(['git', 'checkout', 'manus/orient-S22'], cwd=verify)
        run(['git', 'fsck', '--full'], cwd=verify)
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=verify, text=True).strip() == head
        for filename in ('S22.md', 'S22.json'):
            assert (verify / 'coord/orient/reports' / filename).read_bytes() == (ROOT / 'coord/orient/reports' / filename).read_bytes()
        changed = subprocess.check_output(['git', 'diff', '--name-only', BASE, head], cwd=verify, text=True).splitlines()
        assert not any(p.lower().endswith('.h5ad') or p.endswith('larry_s22_inputs.npz') for p in changed)
        for name in ('clones.csv', 'scores.csv', 'RESULT_S22.md'):
            assert (verify / 'benchmarks/real_flow/s22' / name).is_file()
        stage = 'upload_preflight'
        sha, md5 = digest(bundle, 'sha256'), digest(bundle, 'md5')
        details = {'task': 'orient/S22', 'head': head, 'prerequisite': BASE, 'path': str(bundle), 'bytes': bundle.stat().st_size, 'sha256': sha, 'md5': md5, 'offline_restore_verified': True, 'raw_inputs_excluded': True}
        save(DELIVERY / 'bundle_manifest.json', details)
        query = {'q': "trashed = false and name = 'S22_result.bundle'", 'pageSize': 100, 'fields': 'nextPageToken,files(id,name,size,md5Checksum,sha256Checksum,webViewLink)'}
        candidates = gws(['drive', 'files', 'list', '--params', json.dumps(query)])
        assert not candidates.get('nextPageToken'), 'Paginate before uploading; do not guess.'
        same = [f for f in candidates.get('files', []) if f.get('md5Checksum') == md5 and int(f.get('size', 0)) == bundle.stat().st_size]
        if same:
            assert len(same) == 1
            file_id = same[0]['id']
            print('Recovered an identical existing Drive upload', file_id, flush=True)
        else:
            stage = 'upload_requested'
            assert not (DELIVERY / 'upload_attempt.json').exists(), 'Uncertain prior upload: reconcile, never repeat blindly.'
            save(DELIVERY / 'upload_attempt.json', dict(details, requested_utc=utc(), state='upload_requested'))
            uploaded = gws(['drive', 'files', 'create', '--upload', str(bundle), '--upload-content-type', 'application/octet-stream', '--json', json.dumps({'name': 'S22_result.bundle', 'mimeType': 'application/octet-stream', 'description': 'scJDO2 orient/S22 incremental result: e90440c..manus/orient-S22; planner Amendment 1 and outputs; original input/H5ADs excluded.'}), '--params', json.dumps({'fields': 'id,name,size,md5Checksum,sha256Checksum,webViewLink'})])
            file_id = uploaded['id']
            save(DELIVERY / 'upload_response.json', uploaded)
        stage = 'remote_verification'
        remote = gws(['drive', 'files', 'get', '--params', json.dumps({'fileId': file_id, 'fields': 'id,name,size,md5Checksum,sha256Checksum,webViewLink'})])
        assert remote['name'] == 'S22_result.bundle' and int(remote['size']) == bundle.stat().st_size
        assert remote['md5Checksum'] == md5
        if remote.get('sha256Checksum'):
            assert remote['sha256Checksum'] == sha
        details.update(drive=remote, drive_link=remote.get('webViewLink', 'https://drive.google.com/file/d/' + file_id + '/view'), remote_checksum_verified=True)
        save(DELIVERY / 'delivery_manifest.json', details)
        assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
        print(json.dumps(details, indent=2), flush=True)
        rc, stage = 0, 'complete'
    finally:
        own = resource.getrusage(resource.RUSAGE_SELF)
        child = resource.getrusage(resource.RUSAGE_CHILDREN)
        save(checkpoint, {'task': 'orient/S22', 'unit': 'bundle_offline_verify_and_Drive_delivery', 'stage': stage, 'exit_code': rc, 'started_utc': started, 'finished_utc': utc(), 'wall_seconds': time.monotonic()-wall, 'cpu_hours': (own.ru_utime+own.ru_stime+child.ru_utime+child.ru_stime)/3600, 'max_rss_kib': max(own.ru_maxrss, child.ru_maxrss), 'accounting': 'external delivery receipt; committed report measures completed analysis and review units'})


if __name__ == '__main__':
    main()
