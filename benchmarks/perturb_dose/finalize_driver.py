#!/usr/bin/env python3
"""Final packaging only; no scientific recomputation and no Git pushes."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[2]
CHECKPOINT=ROOT.parent/'S18d_delivery_finalize.checkpoint.json'
if CHECKPOINT.exists():raise RuntimeError('Delivery checkpoint exists; inspect rather than repeat')
started=dt.datetime.now(dt.UTC).isoformat();wall=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN)
env=os.environ.copy();env['PYTHONUNBUFFERED']='1';env['PYTHONPATH']=str(ROOT)
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','LOKY_MAX_CPU_COUNT'):env[key]='6'
print('Final detached delivery started',started,flush=True)
p=subprocess.Popen([sys.executable,'scripts/s18d_dose.py','finalize-delivery'],cwd=ROOT,env=env)
while True:
 try:code=p.wait(timeout=20);break
 except subprocess.TimeoutExpired:print('Final packaging running; wall',round(time.monotonic()-wall),'seconds',flush=True)
after=resource.getrusage(resource.RUSAGE_CHILDREN)
cpu=(after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime)/3600
recorded=sum(json.loads(p.read_text())['cpu_hours'] for p in (ROOT/'benchmarks/perturb_dose/run_logs').glob('*.checkpoint.json'))
r={'unit':'delivery_finalize_external','started_utc':started,'finished_utc':dt.datetime.now(dt.UTC).isoformat(),'exit_code':code,'wall_seconds':time.monotonic()-wall,'cpu_hours':cpu,'total_cpu_hours_including_final_delivery':recorded+cpu,'max_rss_kib':after.ru_maxrss,'no_scientific_recomputation':True}
CHECKPOINT.write_text(json.dumps(r,indent=2)+'\n')
if code:sys.exit(code)
assert recorded+cpu<=16
archive=ROOT.parent/'S18d_benchmarks.zip'
with zipfile.ZipFile(archive,'a',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:z.write(CHECKPOINT,'delivery/S18d_delivery_finalize.checkpoint.json')
with zipfile.ZipFile(archive) as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
manifest=ROOT.parent/'S18d_delivery_manifest.json';m=json.loads(manifest.read_text());m['external_delivery_checkpoint']=r
for item in m['downloads']:
 path=Path(item['path']);h=hashlib.sha256()
 with path.open('rb') as f:
  while block:=f.read(8*1024*1024):h.update(block)
 item.update(bytes=path.stat().st_size,sha256=h.hexdigest());print('FINAL DOWNLOAD',json.dumps(item),flush=True)
manifest.write_text(json.dumps(m,indent=2)+'\n')
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
print('Complete: clean branch, verified bundle/archive/reports; total recorded CPU hours',recorded+cpu,flush=True)
