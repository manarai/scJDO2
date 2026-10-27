#!/usr/bin/env python3
"""Pre-registered orient/S18d. One timed unit; no perturbed counts before freeze."""
from pathlib import Path
import argparse
import datetime as dt
import fnmatch
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmarks/perturb_dose'
CACHE=OUT/'cache'
LOGS=OUT/'run_logs'
FIG=ROOT/'benchmarks/figures/perturb_dose'
TASK=ROOT/'coord/orient/tasks/S18d.md'
BUDGET=16.0
THREADS=min(6,os.cpu_count() or 1)
MEMORY=8*1024**3
SCREENS=('rpe1','k562')
NORMS=('N1','N2')
INPUTS={
 'k562':(ROOT/'benchmarks/perturb_k562/data/K562_essential_raw_singlecell_01.h5ad',10661879995,'3e5a63a9e892b21029bb55fca4e12517a49aad7af6c14133ca63d12cf68c6cee'),
 'rpe1':(ROOT/'benchmarks/perturb_valleys/data/rpe1_raw_singlecell_01.h5ad',8700873216,'9b05ef1f81526216fa008d677e9e0d03dce9a2f7a95499a4fb81e505e9d88ef1'),
}
T_PATTERNS=['MED*','TAF*','GTF2*','POLR2*','INTS*','SUPT4H1','SUPT5H','SUPT6H','SUPT16H','PAF1','CTR9','LEO1','CDC73','RTF1']
M_PATTERNS=['MRPL*','MRPS*','TIMM*','TOMM*','PHB','PHB2','HSPA9','EIF2S1','EIF2S2','EIF2S3','EIF2B*','EIF3*','*ARS','*ARS2','GSPT1','HBS1L']


def now():
 return dt.datetime.now(dt.UTC).isoformat()


def dump(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
 temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while block:=f.read(8*1024*1024):h.update(block)
 return h.hexdigest()


def save_npz(path,**arrays):
 import numpy as np
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():raise RuntimeError('Completed artifact exists; no scientific overwrite: '+str(path))
 temp=path.with_name(path.stem+'.tmp.npz')
 if temp.exists():raise RuntimeError('Partial artifact exists; inspect: '+str(temp))
 np.savez_compressed(temp,**arrays);temp.replace(path)


def cpu_hours():
 return sum(json.loads(p.read_text())['cpu_hours'] for p in LOGS.glob('*.checkpoint.json'))


def run_unit(name,command):
 import psutil
 LOGS.mkdir(parents=True,exist_ok=True);receipt=LOGS/f'{name}.checkpoint.json'
 if receipt.exists():raise RuntimeError('Checkpoint exists; inspect rather than repeat')
 used_before=cpu_hours()
 if used_before>=BUDGET:raise RuntimeError('CPU budget exhausted')
 env=os.environ.copy()
 for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','LOKY_MAX_CPU_COUNT'):env[key]=str(THREADS)
 env['PYTHONPATH']=str(ROOT);env['PATH']=str(ROOT/'.venv/bin')+os.pathsep+env.get('PATH','');env['PYTHONUNBUFFERED']='1'
 started=now();wall=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN);log=LOGS/f'{name}.log';peak=0;reason=None
 print(f'[{started}] {name}: {command}',flush=True)
 with log.open('w') as f:
  p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);next_progress=wall+30
  while p.poll() is None:
   try:
    procs=[psutil.Process(p.pid),*psutil.Process(p.pid).children(recursive=True)]
    values=[x.memory_info().rss for x in procs if x.is_running()];peak=max([peak,*values]);used=sum(sum(x.cpu_times()[:2]) for x in procs if x.is_running())/3600
    if peak>MEMORY or sum(values)>MEMORY:reason='8 GiB conservative process-tree RSS ceiling exceeded'
    if used_before+used+THREADS*10/3600>=BUDGET:reason='CPU budget reserve reached'
    if reason:os.killpg(p.pid,15);break
   except psutil.Error:pass
   if time.monotonic()>=next_progress:
    print(f'[{now()}] {name} running; wall {time.monotonic()-wall:.0f}s; {log.relative_to(ROOT)}',flush=True);next_progress+=30
   time.sleep(1)
  try:code=p.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(p.pid,9);code=p.wait()
 after=resource.getrusage(resource.RUSAGE_CHILDREN)
 exact_cpu_hours=(after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime)/3600
 if used_before+exact_cpu_hours>BUDGET:reason='Exact cumulative CPU budget exceeded';code=124
 record={'unit':name,'command':command,'started_utc':started,'finished_utc':now(),'exit_code':code,'wall_seconds':time.monotonic()-wall,'cpu_hours':(after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime)/3600,'max_rss_kib':max(after.ru_maxrss,int(peak/1024)),'threads':THREADS,'cpu_count':os.cpu_count(),'ram_bytes':os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES'),'per_process_rss_limit_bytes':MEMORY,'log':str(log.relative_to(ROOT)),'stop_reason':reason}
 dump(receipt,record);print('\n'.join(log.read_text().splitlines()[-25:]),flush=True);print(json.dumps(record,indent=2),flush=True)
 return code or (1 if reason else 0)


def check_task():
 pre=json.loads((OUT/'preregistration.json').read_text())
 assert pre['status']=='approved before any S18d result'
 assert digest(TASK)==pre['task_sha256'],'STOP: frozen amended task changed'
 assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='manus/orient-S18d'
 assert json.loads((LOGS/'tests_before.checkpoint.json').read_text())['exit_code']==0
 return pre


def source(screen):
 check_task();d=json.loads((OUT/'input_verification.json').read_text())['inputs'][screen];p,size,sha=INPUTS[screen];s=p.stat()
 assert d['sha256']==sha and s.st_size==size and s.st_mtime_ns==d['mtime_ns'],'STOP: original file changed'
 return p


def preflight():
 assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='manus/orient-S18d'
 inputs={}
 for screen,(path,size,expected) in INPUTS.items():
  assert path.is_file(),'STOP: missing original input';observed=digest(path)
  assert path.stat().st_size==size and observed==expected,'STOP: original checksum/size mismatch'
  inputs[screen]={'path':str(path.relative_to(ROOT)),'bytes':size,'sha256':observed,'mtime_ns':path.stat().st_mtime_ns,'verified_utc':now()};print(screen,'original checksum unchanged',observed,flush=True)
 prior=json.loads((OUT/'reference_sources/S18c.json').read_text());caches=[]
 for screen in INPUTS:
  rel=f'benchmarks/perturb_valleys/cache/controls_{screen}.npz';a=next(a for a in prior['artifacts'] if a['path']==rel);path=ROOT/rel
  if not path.exists():path=OUT/f'reference_inputs/controls_{screen}.npz'
  assert path.stat().st_size==a['bytes'] and digest(path)==a['sha256'],'STOP: prior control cache mismatch'
  caches.append({'path':rel,'bytes':a['bytes'],'sha256':a['sha256'],'control_only':True,'source_ref':'manus/orient-S18c','arrays_not_loaded':True})
 versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','pandas','scanpy','anndata','scikit-learn','scikit-misc','torch','psutil','pytest','matplotlib','h5py')}
 dump(OUT/'environment.json',{'python':platform.python_version(),'executable':sys.executable,'cpu_count':os.cpu_count(),'threads':THREADS,'ram_bytes':os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES'),'per_process_rss_limit_bytes':MEMORY,'versions':versions,'reused_venv':True,'recorded_utc':now()})
 dump(OUT/'input_verification.json',{'inputs':inputs,'control_only_caches':caches,'scientific_arrays_not_opened':True,'hashing_is_integrity_only':True,'checked_utc':now()})
 (OUT/'pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],cwd=ROOT,text=True));return 0


def metadata(path):
 import h5py
 from anndata._io.specs import read_elem
 with h5py.File(path,'r') as f:
  obs=read_elem(f['obs']);var=read_elem(f['var']);x=f['X'];shape=x.shape if isinstance(x,h5py.Dataset) else tuple(x.attrs['shape'])
 assert 'gene' in obs and not obs.gene.isna().any()
 return obs,var,shape


def read_rows(x,indices,n_genes):
 """Exactly selected rows only, including no perturbed X chunks before freeze."""
 import h5py
 import numpy as np
 if isinstance(x,h5py.Dataset):return np.asarray(x[indices,:],dtype=np.float32)
 kind=x.attrs.get('encoding-type','');kind=kind.decode() if isinstance(kind,bytes) else kind
 if kind!='csr_matrix':raise RuntimeError('STOP: unsupported raw storage')
 out=np.zeros((len(indices),n_genes),dtype=np.float32)
 for row,i in enumerate(indices):
  lo,hi=x['indptr'][i:i+2];out[row,x['indices'][lo:hi]]=x['data'][lo:hi]
 return out


def prepare_controls(screen):
 import h5py
 import numpy as np
 import pandas as pd
 from sklearn.decomposition import PCA
 path=source(screen);dest=CACHE/f'controls_{screen}.npz'
 if dest.exists():raise RuntimeError('Control transform already complete')
 assert not (OUT/'controls_frozen.json').exists(),'STOP: controls cannot change after freeze'
 obs,var,shape=metadata(path);labels=obs.gene.astype(str).to_numpy(dtype=str);ci=np.flatnonzero(labels=='non-targeting');n=len(ci)
 assert n=={'rpe1':11485,'k562':10691}[screen]
 old=ROOT/f'benchmarks/perturb_valleys/cache/controls_{screen}.npz';prior=json.loads((OUT/'reference_sources/S18c.json').read_text());art=next(a for a in prior['artifacts'] if a['path']==str(old.relative_to(ROOT)))
 # The delivery archive keeps an identical control-only reference for an offline clone.
 if not old.exists():old=OUT/f'reference_inputs/controls_{screen}.npz'
 assert digest(old)==art['sha256']
 with np.load(old,allow_pickle=False) as a:
  hvg=a['hvg'];pmean=a['pca_mean'];pc=a['pca_components'][:20];zlog=a['z_controls'][:,:20];np.testing.assert_array_equal(ci,a['control_indices']);median=float(a['median_library'])
 assert hvg.sum()==2000
 counts=np.empty((n,len(var)),dtype=np.float32)
 with h5py.File(path,'r') as f:
  for start in range(0,n,512):
   end=min(n,start+512);counts[start:end]=read_rows(f['X'],ci[start:end],len(var))
   if start%4096==0:print(f'[{now()}] {screen}: only controls {end}/{n}',flush=True)
 assert np.isfinite(counts).all() and (counts>=0).all() and np.equal(counts,np.floor(counts)).all()
 lib=counts.sum(1,dtype=np.float64);ng=(counts>0).sum(1);assert (lib>0).all() and (ng>0).all();assert np.median(lib)==median
 frac_mean=(counts/lib[:,None]).mean(0);detect=(counts>0).mean(0)
 symbols=var.gene_name.astype(str).to_numpy(dtype=str)
 # Raw Replogle var.index is Ensembl IDs; gene_name provides its target symbols.
 logged=counts[:,hvg].copy();logged*=(median/lib).astype(np.float32)[:,None];np.log1p(logged,out=logged)
 pca=PCA(n_components=20,svd_solver='full',whiten=False);pca.mean_=pmean;pca.components_=pc;pca.n_features_in_=2000;pca.n_components_=20
 zcheck=pca.transform(logged).astype(np.float32);np.testing.assert_allclose(zcheck,zlog,rtol=1e-5,atol=1e-5);del logged,zcheck
 design=np.column_stack([np.ones(n),np.log(lib),np.log(ng)]);B=np.linalg.lstsq(design,zlog,rcond=None)[0]
 q,r=np.linalg.qr(B[1:].T,mode='reduced');assert np.linalg.matrix_rank(B[1:].T)==2,'STOP: dependent depth directions';projector=np.eye(20)-q@q.T
 zn1=np.asarray(zlog,dtype=np.float64)@projector;axis=B[1]/np.linalg.norm(B[1]);r2=1-((zlog-design@B)**2).sum(0)/((zlog-zlog.mean(0))**2).sum(0)
 mu_g=counts[:,hvg].sum(0,dtype=np.float64)/lib.sum();mu=np.outer(lib,mu_g)
 assert (mu_g>0).all(),'STOP: zero mean selected Pearson HVG'
 residual=(counts[:,hvg]-mu)/np.sqrt(mu+mu*mu/100);np.clip(residual,-np.sqrt(n),np.sqrt(n),out=residual);del counts,mu
 p2=PCA(n_components=20,svd_solver='full',whiten=False).fit(residual);zn2=p2.transform(residual);del residual
 save_npz(dest,z_N1=zn1,z_N2=zn2,z_log=zlog,control_indices=ci,hvg=hvg,pca_log_mean=pmean,pca_log_components=pc,pca_N2_mean=p2.mean_,pca_N2_components=p2.components_,pca_N2_variance=p2.explained_variance_,pearson_gene_proportions=mu_g,median_library=np.array(median),depth_coefficients=B,depth_plane=q,depth_projector=projector,depth_axis=axis,control_fraction_mean=frac_mean,control_detection_fraction=detect,gene_symbols=symbols,control_library=lib,control_genes_detected=ng)
 targets=sorted(set(labels)-{'non-targeting'});groups={g:[t for t in targets if any(fnmatch.fnmatchcase(t,pat) for pat in pats)] for g,pats in [('T',T_PATTERNS),('M',M_PATTERNS)]}
 dump(OUT/f'group_symbols_{screen}.json',{'screen':screen,'patterns':{'T':T_PATTERNS,'M':M_PATTERNS},'targets':targets,'groups':groups,'overlap':sorted(set(groups['T'])&set(groups['M'])),'resolved_utc':now(),'raw_metadata_only':True,'no_perturbed_expression_read':True})
 freq=pd.Series(labels).value_counts();symbol_counts=pd.Series(symbols).value_counts();gene_map={t:i for i,t in enumerate(symbols) if symbol_counts[t]==1}
 assert all(freq[t]<150 for t in targets if t in symbol_counts and symbol_counts[t]>1),'STOP: duplicate symbol target could qualify'
 frame=pd.DataFrame([{'target':t,'n_cells':int(freq[t]),'group_T':t in groups['T'],'group_M':t in groups['M'],'gene_present':t in symbol_counts,'control_detection_fraction':float(detect[gene_map[t]]) if t in gene_map else None,'control_mean_target_fraction':float(frac_mean[gene_map[t]]) if t in gene_map else None} for t in targets]);frame.to_csv(OUT/f'target_control_metadata_{screen}.csv',index=False,float_format='%.17g')
 pd.DataFrame({'gene_symbol':symbols,'control_HVG':hvg,'control_detection_fraction':detect,'control_mean_fraction':frac_mean}).to_csv(OUT/f'hvg_control_{screen}.csv',index=False,float_format='%.17g')
 dump(OUT/f'controls_{screen}.json',{'screen':screen,'n_controls':n,'n_cells':int(shape[0]),'n_genes':len(var),'n_hvg':2000,'fit_cells':'non-targeting controls only','cache_sha256':digest(dest),'original_log_transform_reuse':{'path':str(old.relative_to(ROOT)),'sha256':art['sha256'],'first20_of_full_control_PCA30':True},'depth_regression_R2':r2.tolist(),'depth_plane_rank':2,'N1_coordinate_rank':int(np.linalg.matrix_rank(zn1-zn1.mean(0))),'depth_axis_definition':'B[1]/norm(B[1]); conditional log library coefficient in unprojected log PCA20','N2_theta':100,'N2_clip':float(np.sqrt(n)),'N2_pca_solver':'full','N2_centering':'control mean only','original_gene_symbol_source':'var.index; no alias remapping','median_control_library':median,'no_perturbed_expression_read':True,'finished_utc':now()})
 print(screen,'both control transforms saved; depth rank',np.linalg.matrix_rank(zn1-zn1.mean(0)),flush=True);return 0


def fit_subspace(screen,norm):
 import numpy as np
 import torch
 from scjdo2.energy_drift import EnergyDriftAdapter,EnergyDriftConfig
 source(screen);dest=OUT/f'subspaces_{screen}_{norm}.npz'
 if dest.exists():raise RuntimeError('Subspace complete; no refit')
 assert not (OUT/'controls_frozen.json').exists()
 cpath=CACHE/f'controls_{screen}.npz'
 with np.load(cpath,allow_pickle=False) as a:z=a[f'z_{norm}'];ci=a['control_indices'];axis=a['depth_axis'];plane=a['depth_plane']
 cfg=EnergyDriftConfig(hidden=96,depth=3,n_epochs=1500,seed=0,batch_size=256,loss='esm',lr=5e-4,beta=1.0,esm_laplacian='exact')
 torch.set_num_threads(THREADS);drift=EnergyDriftAdapter(20,cfg).fit(z,verbose=True);H=drift.jacobian_autodiff(z)
 assert np.isfinite(H).all();asym=np.linalg.norm(H-H.transpose(0,2,1),axis=(1,2))/(np.linalg.norm(H,axis=(1,2))+1e-12)
 assert asym.max()<=1e-4,'STOP: required score symmetry check'
 hbulk=H.mean(0,dtype=np.float64);stiff=-.5*(hbulk+hbulk.T);w,v=np.linalg.eigh(stiff);positive=np.flatnonzero(w>0);selected=positive[:3] if len(positive)>=3 else np.argsort(np.abs(w),kind='stable')[:3];sj=v[:,selected]
 cov=np.cov(z,rowvar=False);ew,ev=np.linalg.eigh(cov);sc=ev[:,-3:];root=(ev*np.sqrt(np.maximum(ew,0)))@ev.T
 # Tiny negative roundoff of PSD covariance is set to zero only for its square root; stiffness eigenvalues never thresholded.
 assert ew.min()>=-1e-10*max(1,ew.max()),'STOP: control covariance not PSD'
 g=np.random.default_rng(0).standard_normal((10000,20,3));matched=root@g;qr,R=np.linalg.qr(matched,mode='reduced');assert np.min(np.abs(np.diagonal(R,axis1=1,axis2=2)))>0
 np.testing.assert_allclose(qr.transpose(0,2,1)@qr,np.broadcast_to(np.eye(3),(10000,3,3)),rtol=1e-10,atol=1e-10)
 save_npz(dest,S_J=sj,S_C=sc,H_bulk=hbulk,stiffness_eigenvalues=w,selected_eigenvalue_indices=selected,control_covariance=cov,covariance_eigenvalues=ew,control_covariance_sqrt=root,random_bases=qr,control_mean=z.mean(0,dtype=np.float64),control_indices=ci,depth_axis_before_projection=axis,depth_plane=plane,n_controls=np.array(len(ci)),normalisation=np.array(norm))
 save_npz(CACHE/f'H_{screen}_{norm}.npz',H=H,control_indices=ci)
 torch.save({'configuration':vars(cfg),'dim':20,'state_dict':drift._model.state_dict(),'fit_cells':'controls only','controls_sha256':digest(cpath)},CACHE/f'model_{screen}_{norm}.pt')
 sha=digest(dest);dest.chmod(0o444)
 with (OUT/'subspaces.sha256').open('a') as f:f.write(sha+'  '+dest.name+'\n')
 dump(OUT/f'subspaces_{screen}_{norm}.json',{'screen':screen,'normalisation':norm,'n_fit_controls':len(ci),'configuration':vars(cfg),'n_positive_stiffness_eigenvalues':len(positive),'selected_eigenvalues':w[selected].tolist(),'stiffness_eigenvalues':w.tolist(),'no_eigenvalue_tolerance_or_rule_change':True,'symmetry_max_relative':float(asym.max()),'symmetry_median_relative':float(np.median(asym)),'covariance_eigenvalues':ew.tolist(),'null_basis_generator':'default_rng(0).standard_normal((10000,20,3)); symmetric control covariance sqrt @ G; reduced QR; RNG reset per screen/normalisation','null_covariance_sqrt_roundoff_clip':'max(eigenvalue,0) for PSD square root only','subspace_file':str(dest.relative_to(ROOT)),'sha256':sha,'controls_sha256':digest(cpath),'H_sha256':digest(CACHE/f'H_{screen}_{norm}.npz'),'no_perturbed_expression_or_coordinates_read':True,'saved_utc':now()})
 print(screen,norm,'subspaces+10000 bases SAVED',sha,'selected stiffness',w[selected],flush=True);return 0


def freeze_controls():
 check_task();dest=OUT/'controls_frozen.json'
 if dest.exists():raise RuntimeError('Freeze already complete')
 records=[]
 for screen in SCREENS:
  source(screen)
  assert not (CACHE/f'dose_{screen}.npz').exists()
  for norm in NORMS:
   r=json.loads((OUT/f'subspaces_{screen}_{norm}.json').read_text());assert r['no_perturbed_expression_or_coordinates_read'];assert digest(ROOT/r['subspace_file'])==r['sha256'];records.append(r)
 dump(dest,{'status':'all control predictors and null bases frozen','saved_utc':now(),'task_sha256':digest(TASK),'pre_perturbation_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'files':records,'control_cache_sha256':{s:digest(CACHE/f'controls_{s}.npz') for s in SCREENS},'symbol_list_sha256':{s:digest(OUT/f'group_symbols_{s}.json') for s in SCREENS},'control_target_metadata_sha256':{s:digest(OUT/f'target_control_metadata_{s}.csv') for s in SCREENS},'no_perturbed_expression_or_coordinates_read':True})
 print('GLOBAL PRE-PERTURBATION BARRIER SAVED',flush=True);return 0


def barrier():
 check_task();r=json.loads((OUT/'controls_frozen.json').read_text());assert r['task_sha256']==digest(TASK) and r['no_perturbed_expression_or_coordinates_read']
 for a in r['files']:assert digest(ROOT/a['subspace_file'])==a['sha256'],'STOP: frozen subspace changed'
 for s in SCREENS:
  assert digest(CACHE/f'controls_{s}.npz')==r['control_cache_sha256'][s]
  assert digest(OUT/f'group_symbols_{s}.json')==r['symbol_list_sha256'][s]
  assert digest(OUT/f'target_control_metadata_{s}.csv')==r['control_target_metadata_sha256'][s]
 return r


def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
 u=sub.add_parser('unit');u.add_argument('name');u.add_argument('command',nargs=argparse.REMAINDER)
 sub.add_parser('preflight');sub.add_parser('freeze-controls')
 c=sub.add_parser('controls');c.add_argument('--screen',choices=SCREENS,required=True)
 f=sub.add_parser('subspaces');f.add_argument('--screen',choices=SCREENS,required=True);f.add_argument('--norm',choices=NORMS,required=True)
 a=p.parse_args()
 if a.action=='unit':return run_unit(a.name,a.command[1:] if a.command and a.command[0]=='--' else a.command)
 if a.action=='preflight':return preflight()
 if a.action=='controls':return prepare_controls(a.screen)
 if a.action=='subspaces':return fit_subspace(a.screen,a.norm)
 return freeze_controls()


# The final dispatcher below adds the post-freeze execution stages.


def select_targets(screen):
 import io
 import numpy as np
 import pandas as pd
 barrier();source(screen);dest=OUT/f'target_lists_{screen}.json'
 if dest.exists():raise RuntimeError('Target lists already fixed')
 ref='manus/orient-S18c' if screen=='rpe1' else 'manus/orient-S18'
 rel='benchmarks/perturb_valleys/targets_rpe1.csv' if screen=='rpe1' else 'benchmarks/perturb_k562/targets.csv'
 report_path=OUT/'reference_sources'/('S18c.json' if screen=='rpe1' else 'S18.json')
 if subprocess.run(['git','cat-file','-e',ref+':'+rel],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
  data=subprocess.check_output(['git','show',ref+':'+rel],cwd=ROOT)
  report_bytes=subprocess.check_output(['git','show',ref+':coord/orient/reports/'+('S18c' if screen=='rpe1' else 'S18')+'.json'],cwd=ROOT)
  if not report_path.exists():report_path.write_bytes(report_bytes)
  report=json.loads(report_bytes)
 else:
  data=(OUT/f'accepted_targets_{screen}.csv').read_bytes();report=json.loads(report_path.read_text())
 a=next(a for a in report['artifacts'] if a['path']==rel);assert hashlib.sha256(data).hexdigest()==a['sha256']
 (OUT/f'accepted_targets_{screen}.csv').write_bytes(data)
 metadata_path=OUT/f'target_dose_metadata_{screen}.csv'
 if not metadata_path.exists():metadata_path=OUT/f'target_control_metadata_{screen}.csv'
 old=pd.read_csv(io.BytesIO(data),float_precision='round_trip');control=pd.read_csv(metadata_path,float_precision='round_trip')
 assert old.target.is_unique;np.testing.assert_array_equal(old.significant.to_numpy(),old.p_adjusted_BH.to_numpy()<.01)
 joined=control.merge(old[['target','p_adjusted_BH','effect_size_energy_distance']],on='target',how='left',validate='one_to_one');reasons=[];eligible=[]
 for r in joined.to_dict('records'):
  why=[]
  if not np.isfinite(r['p_adjusted_BH']) or not r['p_adjusted_BH']<.01:why.append('not significant by accepted original log PCA20 BH<0.01')
  if r['n_cells']<150:why.append('fewer than 150 cells')
  if not r['gene_present']:why.append('target gene not present in raw gene_name annotation')
  elif not np.isfinite(r['control_detection_fraction']):
   assert r['n_cells']<150,'STOP: eligible target has ambiguous raw gene_name mapping'
   why.append('duplicate gene_name dose lookup not evaluated; independently excluded by fixed fewer-than-150-cell rule')
  elif not r['control_detection_fraction']>=.5:why.append('target gene detected in fewer than 50% of controls')
  reasons.append('; '.join(why));eligible.append(not why)
 joined['eligible']=eligible;joined['exclusion_reasons']=reasons;joined.to_csv(OUT/f'target_lists_{screen}.csv',index=False,float_format='%.17g')
 selected=joined[joined.eligible].copy();symbols=json.loads((OUT/f'group_symbols_{screen}.json').read_text())
 exclusions=[{'target':r['target'],'reasons':r['exclusion_reasons']} for r in joined[~joined.eligible].to_dict('records')]
 dump(dest,{'screen':screen,'source_ref':ref,'source_path':rel,'source_sha256':a['sha256'],'significance_test_rerun':False,'common_N1_N2_cohort':True,'symbol_groups':symbols['groups'],'analysed_targets':selected.target.tolist(),'T':selected.loc[selected.group_T,'target'].tolist(),'M':selected.loc[selected.group_M,'target'].tolist(),'n_analysed':len(selected),'n_T':int(selected.group_T.sum()),'n_M':int(selected.group_M.sum()),'exclusions':exclusions,'qualification_table_sha256':digest(OUT/f'target_lists_{screen}.csv'),'saved_before_capture':True,'no_S18d_perturbed_expression_read':True,'recorded_utc':now()})
 print(screen,'fixed common cohort:',len(selected),'T',int(selected.group_T.sum()),'M',int(selected.group_M.sum()),flush=True);return 0


def dose_cells(screen):
 import h5py
 import numpy as np
 import pandas as pd
 from sklearn.decomposition import PCA
 freeze=barrier();path=source(screen);dest=CACHE/f'dose_{screen}.npz'
 if dest.exists():raise RuntimeError('Dose/projection complete; no repeat')
 lists=json.loads((OUT/f'target_lists_{screen}.json').read_text());assert lists['saved_before_capture']
 assert digest(OUT/f'target_lists_{screen}.csv')==lists['qualification_table_sha256']
 obs,var,shape=metadata(path);labels=obs.gene.astype(str).to_numpy(dtype=str);n=len(labels);first_access=now()
 with np.load(CACHE/f'controls_{screen}.npz',allow_pickle=False) as a:
  ci=a['control_indices'];hvg=a['hvg'];median=float(a['median_library']);prop=a['pearson_gene_proportions'];symbols=a['gene_symbols'];fracmean=a['control_fraction_mean'];projector=a['depth_projector'];z1c=a['z_N1'];z2c=a['z_N2'];zlc=a['z_log']
  p1=PCA(n_components=20,svd_solver='full');p1.mean_=a['pca_log_mean'];p1.components_=a['pca_log_components'];p1.n_features_in_=2000;p1.n_components_=20
  p2=PCA(n_components=20,svd_solver='full');p2.mean_=a['pca_N2_mean'];p2.components_=a['pca_N2_components'];p2.n_features_in_=2000;p2.n_components_=20
 symbols=var.gene_name.astype(str).to_numpy(dtype=str)
 assert all(int((symbols==t).sum())==1 for t in lists['analysed_targets']),'STOP: analysed target has ambiguous raw gene_name annotation'
 gene_map={t:i for i,t in enumerate(symbols)};qual=set(lists['analysed_targets']);target_cols=np.array([gene_map[t] if t in qual else -1 for t in labels])
 zlog=np.empty((n,20),dtype=np.float32);z2=np.empty((n,20),dtype=np.float64);lib=np.empty(n);ng=np.empty(n,dtype=np.int64);targetcounts=np.full(n,np.nan);kd=np.full(n,np.nan)
 with h5py.File(path,'r') as f:
  for start in range(0,n,2048):
   end=min(n,start+2048);block=read_rows(f['X'],np.arange(start,end),len(var));lb=block.sum(1,dtype=np.float64);assert (lb>0).all();lib[start:end]=lb;ng[start:end]=(block>0).sum(1)
   logged=block[:,hvg].copy();logged*=(median/lb).astype(np.float32)[:,None];np.log1p(logged,out=logged);zlog[start:end]=p1.transform(logged);del logged
   mu=np.outer(lb,prop);res=(block[:,hvg]-mu)/np.sqrt(mu+mu*mu/100);np.clip(res,-np.sqrt(len(ci)),np.sqrt(len(ci)),out=res);z2[start:end]=p2.transform(res)
   cols=target_cols[start:end];valid=np.flatnonzero(cols>=0)
   if len(valid):
    tc=block[valid,cols[valid]];targetcounts[start+valid]=tc;kd[start+valid]=1-(tc/lb[valid])/fracmean[cols[valid]]
   if start%40960==0:print(f'[{now()}] {screen}: frozen-transform projection+raw doses {end}/{n}',flush=True)
 z1=np.asarray(zlog,dtype=np.float64)@projector
 np.testing.assert_allclose(zlog[ci],zlc,rtol=1e-5,atol=1e-5);np.testing.assert_allclose(z1[ci],z1c,rtol=1e-5,atol=1e-5);np.testing.assert_allclose(z2[ci],z2c,rtol=1e-9,atol=1e-9)
 tertile=np.full(n,-1,dtype=np.int8);records=[]
 for t in lists['analysed_targets']:
  rows=np.flatnonzero(labels==t);assert len(rows)>=150 and np.isfinite(kd[rows]).all();order=rows[np.lexsort((rows,kd[rows]))]
  bins=np.array_split(order,3)
  for j,b in enumerate(bins):tertile[b]=j
  records.append({'target':t,'n_cells':len(rows),'tertile_counts':[len(b) for b in bins],'min_knockdown':float(kd[rows].min()),'median_knockdown':float(np.median(kd[rows])),'max_knockdown':float(kd[rows].max()),'zero_target_count_fraction':float((targetcounts[rows]==0).mean()),'tied_knockdown_fraction':float(1-len(np.unique(kd[rows]))/len(rows))})
 selected=np.flatnonzero(tertile>=0);dose=pd.DataFrame({'screen':screen,'original_row':selected,'target':labels[selected],'tertile':np.asarray(['weak','medium','strong'])[tertile[selected]],'target_gene_counts':targetcounts[selected],'library_size':lib[selected],'genes_detected':ng[selected],'knockdown':kd[selected]});dose['group_T']=dose.target.isin(lists['T']);dose['group_M']=dose.target.isin(lists['M']);dose.to_csv(OUT/f'dose_{screen}.csv',index=False,float_format='%.17g')
 save_npz(dest,z_N1=z1,z_N2=z2,z_log=zlog,labels=labels,control_indices=ci,library=lib,genes_detected=ng,target_gene_counts=targetcounts,knockdown=kd,tertile=tertile)
 library=[]
 for scope in ('all_screen_cells_by_symbol','analysed_targets'):
  for group in ('controls','T','M','other'):
   groups=json.loads((OUT/f'group_symbols_{screen}.json').read_text())['groups']
   if group=='controls':rows=ci
   else:
    members=groups[group] if group in ('T','M') else sorted(set(labels)-{'non-targeting'}-set(groups['T'])-set(groups['M']))
    mask=np.isin(labels,members)
    if scope=='analysed_targets':mask&=tertile>=0
    rows=np.flatnonzero(mask)
   if len(rows):library.append({'screen':screen,'scope':scope,'group':group,'n_cells':len(rows),'median_library':float(np.median(lib[rows])),'mean_library':float(lib[rows].mean()),'median_log10_library':float(np.median(np.log10(lib[rows]))),'median_genes_detected':float(np.median(ng[rows]))})
 pd.DataFrame(library).to_csv(OUT/f'library_groups_{screen}.csv',index=False,float_format='%.17g')
 dump(OUT/f'dose_{screen}.json',{'screen':screen,'n_analysed_targets':len(lists['analysed_targets']),'n_dose_cells':len(selected),'target_list_sha256':digest(OUT/f'target_lists_{screen}.json'),'target_summaries':records,'first_perturbed_expression_access_utc':first_access,'control_predictor_freeze_utc':freeze['saved_utc'],'all_controls_reference':len(ci),'control_projection_max_abs_difference':{'N1':float(np.abs(z1[ci]-z1c).max()),'N2':float(np.abs(z2[ci]-z2c).max())},'target_gene_fraction_control_mean':'Mean of target-gene raw counts/full library size over ALL controls','knockdown_not_clamped':True,'tertile_rule':'Ascending knockdown; ties by original row; numpy.array_split(3)','cache_sha256':digest(dest),'csv_sha256':digest(OUT/f'dose_{screen}.csv'),'finished_utc':now()})
 print(screen,'dose/projection complete',len(selected),'cells',flush=True);return 0


def capture_targets(screen,norm):
 import numpy as np
 import pandas as pd
 barrier();source(screen);dest=OUT/f'capture_{screen}_{norm}.csv'
 if dest.exists():raise RuntimeError('Capture complete; no repeat')
 lists=json.loads((OUT/f'target_lists_{screen}.json').read_text());info=json.loads((OUT/f'dose_{screen}.json').read_text());assert info['target_list_sha256']==digest(OUT/f'target_lists_{screen}.json')
 with np.load(CACHE/f'dose_{screen}.npz',allow_pickle=False) as a:z=a['z_'+norm];labels=a['labels'];ci=a['control_indices'];tertile=a['tertile'];lib=a['library'];kd=a['knockdown'];zlog=a['z_log']
 with np.load(OUT/f'subspaces_{screen}_{norm}.npz',allow_pickle=False) as a:sj=a['S_J'];sc=a['S_C'];bases=a['random_bases'];mu=a['control_mean'];depth=a['depth_axis_before_projection']
 np.testing.assert_allclose(z[ci].mean(0,dtype=np.float64),mu,atol=1e-8,rtol=1e-8);logmu=zlog[ci].mean(0,dtype=np.float64)
 records=[];shifts=[]
 for t in lists['analysed_targets']:
  for j,binname in enumerate(('weak','medium','strong')):
   rows=np.flatnonzero((labels==t)&(tertile==j));d=z[rows].mean(0,dtype=np.float64)-mu;before=zlog[rows].mean(0,dtype=np.float64)-logmu;dnorm=np.linalg.norm(d)
   assert len(rows)>0 and dnorm>0 and np.isfinite(d).all(),'STOP: undefined shift; do not filter or impute'
   if norm=='N1':assert np.linalg.norm(np.load(CACHE/f'controls_{screen}.npz',allow_pickle=False)['depth_plane'].T@d)<1e-5
   record={'screen':screen,'normalisation':norm,'target':t,'tertile':binname,'group_T':t in lists['T'],'group_M':t in lists['M'],'n_cells_tertile':len(rows),'reference_n_controls':len(ci),'shift_norm_all_controls':float(dnorm),'capture_S_J':float(np.sum((d@sj)**2)/dnorm**2),'capture_S_C':float(np.sum((d@sc)**2)/dnorm**2),'median_knockdown':float(np.median(kd[rows])),'median_library':float(np.median(lib[rows])),'cos_depth_axis_N1_before_projection':float(abs(before@depth)/np.linalg.norm(before)),'confirmatory_configuration':screen=='rpe1' and binname=='weak'}
   records.append(record);shifts.append(d)
 frame=pd.DataFrame(records);D=np.array(shifts).reshape((-1,20));den=np.sum(D*D,axis=1);null=np.empty((10000,len(D)))
 for start in range(0,10000,500):
  coeff=np.einsum('bdk,td->btk',bases[start:start+500],D,optimize=True);null[start:start+500]=np.sum(coeff*coeff,axis=2)/den
 assert np.isfinite(null).all() and null.min()>=0 and null.max()<=1+1e-10
 for col in ('capture_S_J','capture_S_C'):assert frame[col].min()>=0 and frame[col].max()<=1+1e-10
 frame['capture_null_mean']=null.mean(0);frame['capture_null_p95']=np.quantile(null,.95,axis=0,method='linear');frame.to_csv(dest,index=False,float_format='%.17g')
 save_npz(CACHE/f'shifts_{screen}_{norm}.npz',shifts=D,targets=frame.target.to_numpy(dtype=str),tertiles=frame.tertile.to_numpy(dtype=str),all_controls_mean=mu)
 summaries={};medians={}
 for binname in ('weak','medium','strong'):
  ti=np.flatnonzero(frame.group_T.to_numpy()&(frame.tertile.to_numpy()==binname));mi=np.flatnonzero(frame.group_M.to_numpy()&(frame.tertile.to_numpy()==binname));t=frame.iloc[ti];m=frame.iloc[mi]
  dist=np.median(null[:,ti],axis=1) if len(ti) else None
  if dist is not None:medians[binname]=dist
  summaries[binname]=comparison(t,m,dist)
 save_npz(CACHE/f'null_medians_{screen}_{norm}.npz',**medians)
 dump(OUT/f'capture_{screen}_{norm}.json',{'screen':screen,'normalisation':norm,'n_rows':len(frame),'n_T':lists['n_T'],'n_M':lists['n_M'],'all_shifts_reference':'Mean of ALL controls in the same normalisation','null_definition':'10000 across-T medians, each shared covariance-matched control-only QR basis','summaries':summaries,'target_list_sha256':digest(OUT/f'target_lists_{screen}.json'),'frozen_subspace_sha256':digest(OUT/f'subspaces_{screen}_{norm}.npz'),'shifts_sha256':digest(CACHE/f'shifts_{screen}_{norm}.npz'),'null_medians_sha256':digest(CACHE/f'null_medians_{screen}_{norm}.npz'),'csv_sha256':digest(dest),'finished_utc':now()})
 print(screen,norm,'weak comparisons',json.dumps(summaries['weak']),flush=True);return 0


def comparison(t,m,null_dist):
 import numpy as np
 from scipy.stats import wilcoxon,mannwhitneyu
 nT,nM=len(t),len(m);med=float(t.capture_S_J.median()) if nT else None;p95=float(np.quantile(null_dist,.95,method='linear')) if null_dist is not None else None
 out={'n_T':nT,'n_M':nM,'T_targets':t.target.tolist(),'M_targets':m.target.tolist(),'median_S_J_T':med,'median_S_C_T':float(t.capture_S_C.median()) if nT else None,'median_S_J_M':float(m.capture_S_J.median()) if nM else None,'null_across_T_median_p95':p95,'D1_criterion_met':bool(med>p95) if nT else None,'D2_criterion_met':None,'D3_criterion_met':None,'wilcoxon':None,'mannwhitney':None}
 if nT:
  diff=t.capture_S_J.to_numpy()-t.capture_S_C.to_numpy();res=wilcoxon(diff,alternative='greater')
  out['wilcoxon']={'statistic':float(res.statistic),'p_one_sided_greater':float(res.pvalue) if np.isfinite(res.pvalue) else None,'zero_method':'wilcox','correction':False,'method':'auto','n_pairs':nT,'zero_differences':int((diff==0).sum()),'median_difference_S_J_minus_S_C':float(np.median(diff))};out['D2_criterion_met']=bool(res.pvalue<.05) if np.isfinite(res.pvalue) else None
 if nT and nM:
  res=mannwhitneyu(t.capture_S_J.to_numpy(),m.capture_S_J.to_numpy(),alternative='greater')
  out['mannwhitney']={'statistic':float(res.statistic),'p_one_sided_greater':float(res.pvalue) if np.isfinite(res.pvalue) else None,'method':'auto','use_continuity':True,'n_T':nT,'n_M':nM};out['D3_criterion_met']=bool(res.pvalue<.05) if np.isfinite(res.pvalue) else None
 return out


def aggregate():
 import numpy as np
 import pandas as pd
 barrier();dest=OUT/'gates.json'
 if dest.exists():raise RuntimeError('Gate aggregate already complete; do not reevaluate scientific tests')
 frames=[pd.read_csv(OUT/f'capture_{s}_{n}.csv',float_precision='round_trip') for s in SCREENS for n in NORMS];frame=pd.concat(frames,ignore_index=True);frame.to_csv(OUT/'capture.csv',index=False,float_format='%.17g')
 pd.concat([pd.read_csv(OUT/f'dose_{s}.csv',float_precision='round_trip') for s in SCREENS],ignore_index=True).to_csv(OUT/'dose.csv',index=False,float_format='%.17g')
 pd.concat([pd.read_csv(OUT/f'library_groups_{s}.csv',float_precision='round_trip') for s in SCREENS],ignore_index=True).to_csv(OUT/'library_groups.csv',index=False,float_format='%.17g')
 summaries={s:{n:json.loads((OUT/f'capture_{s}_{n}.json').read_text())['summaries'] for n in NORMS} for s in SCREENS};lists={s:json.loads((OUT/f'target_lists_{s}.json').read_text()) for s in SCREENS};under=lists['rpe1']['n_T']<10
 gates={}
 for gate in ('D1','D2','D3'):
  criteria={n:summaries['rpe1'][n]['weak'][gate+'_criterion_met'] for n in NORMS}
  gates[gate]={'by_normalisation':criteria,'both_normalisations_criterion_met':all(v is True for v in criteria.values()) if all(v is not None for v in criteria.values()) else None,'status':'underpowered' if under else ('undefined' if any(v is None for v in criteria.values()) else 'evaluated'),'passed':None if under or any(v is None for v in criteria.values()) else all(criteria.values())}
 outcome='underpowered' if under else ('pass' if all(g['passed'] is True for g in gates.values()) else 'fail')
 dump(dest,{'task':'orient/S18d','primary_screen':'rpe1','both_normalisations_required':True,'underpowered_threshold_n_T':10,'underpowered':under,'n_T':{s:lists[s]['n_T'] for s in SCREENS},'n_M':{s:lists[s]['n_M'] for s in SCREENS},'gates':gates,'outcome':outcome,'comparisons':summaries,'K562_confirmatory':False,'medium_strong_confirmatory':False,'frozen_task_sha256':digest(TASK),'frozen_predictors_receipt_sha256':digest(OUT/'controls_frozen.json'),'saved_utc':now()})
 pd.DataFrame([{'screen':s,'normalisation':n,**{k:v for k,v in summaries[s][n]['weak'].items() if not isinstance(v,(dict,list))}} for s in SCREENS for n in NORMS]).to_csv(OUT/'weak_summary.csv',index=False,float_format='%.17g')
 make_figures();report('partial');print('Fixed RPE1 outcome:',outcome,flush=True);return 0


def make_figures():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 import pandas as pd
 gates=json.loads((OUT/'gates.json').read_text());df=pd.read_csv(OUT/'capture.csv',float_precision='round_trip');FIG.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axs=plt.subplots(1,2,figsize=(11,4.6),sharey=True)
 for ax,s in zip(axs,SCREENS):
  for j,n in enumerate(NORMS):
   r=gates['comparisons'][s][n]['weak'];values=[r['median_S_J_T'],r['median_S_C_T'],r['null_across_T_median_p95']];ax.bar(np.arange(3)+(j-.5)*.34,values,width=.32,label=n)
  ax.set_xticks(range(3),['Score-J\nmedian T','Covariance\nmedian T','Null95\nT-median']);ax.set_ylim(0,1);ax.set_title(s.upper()+(' — primary '+gates['outcome'] if s=='rpe1' else ' — report only'));ax.legend();ax.set_ylabel('Weak-tertile capture / null threshold')
 fig.suptitle('Control-only soft subspaces: fixed dose-response test');fig.tight_layout();fig.savefig(FIG/'weak_capture_vs_null.png',dpi=170);plt.close(fig)
 fig,axs=plt.subplots(2,2,figsize=(11,7.8),sharey=True)
 for i,s in enumerate(SCREENS):
  for j,n in enumerate(NORMS):
   ax=axs[i,j]
   for group,color in [('T','#315b8a'),('M','#b0763b')]:
    use=df[(df.screen==s)&(df.normalisation==n)&(df.group_T if group=='T' else df.group_M)];v=[use.loc[use.tertile==b,'capture_S_J'].median() for b in ('weak','medium','strong')];ax.plot(range(3),v,'o-',color=color,label=group)
   ax.set_xticks(range(3),['Weak','Medium','Strong']);ax.set_ylim(0,1);ax.set_ylabel('Median score-J capture');ax.set_title(s.upper()+' / '+n+(' — underpowered, descriptive' if s=='rpe1' and gates['underpowered'] else ' — primary weak only' if s=='rpe1' else ' — report only'));ax.legend()
 fig.suptitle('Dose tertiles; medium/strong are report-only');fig.tight_layout();fig.savefig(FIG/'capture_by_dose.png',dpi=170);plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(10,4.3))
 for ax,s in zip(axs,SCREENS):
  use=df[(df.screen==s)&(df.normalisation=='N1')&(df.tertile=='weak')]
  for group,col in [('T','#315b8a'),('M','#b0763b')]:
   sub=use[use.group_T if group=='T' else use.group_M];ax.scatter(sub.cos_depth_axis_N1_before_projection,sub.capture_S_J,s=26,alpha=.75,label=group,c=col)
  ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_xlabel('|cos(weak shift, conditional depth axis)| before projection');ax.set_ylabel('N1 score-J capture after depth-plane removal');ax.set_title(s.upper()+' — depth report only');ax.legend()
 fig.tight_layout();fig.savefig(FIG/'depth_axis_report.png',dpi=170);plt.close(fig)


def fmt(x):
 return 'null' if x is None else f'{x:.6g}' if isinstance(x,(float,int)) else str(x)


def artifact_command(rel):
 name=Path(rel).name
 if name.startswith('subspaces_'):
  parts=name.rsplit('.',1)[0].split('_');return f'.venv/bin/python scripts/s18d_dose.py subspaces --screen {parts[1]} --norm {parts[2]} (fresh output tree only, after controls; never repeat a completed fit)'
 if name.startswith('controls_') and name!='controls_frozen.json':return '.venv/bin/python scripts/s18d_dose.py controls --screen '+('rpe1' if 'rpe1' in name else 'k562')+' (fresh output tree only)'
 if name.startswith('dose_'):return '.venv/bin/python scripts/s18d_dose.py dose --screen '+('rpe1' if 'rpe1' in name else 'k562')+' (fresh post-freeze output tree only)'
 if name.startswith('capture_'):
  p=name.rsplit('.',1)[0].split('_');return f'.venv/bin/python scripts/s18d_dose.py capture --screen {p[1]} --norm {p[2]} (fresh post-dose output tree only)'
 if 'run_logs/' in rel:return 'Original timed execution receipt; retain as provenance, not regenerable without executing its recorded unit.'
 if 'reference_sources/' in rel:return 'git show the source_ref recorded in source_archive.json or its corresponding accepted target metadata; never edit protected source paths.'
 if 'cache/' in rel:return 'Restore the exact saved file from S18d_benchmarks.zip and verify this SHA-256; fresh regeneration via the corresponding controls/subspaces/dose/capture stage in the documented order.'
 return 'Restore this file from the returned S18d benchmark ZIP and verify SHA-256; see REPRODUCE.md for fresh-tree stage order.'


def report(status):
 units=[json.loads(p.read_text()) for p in sorted(LOGS.glob('*.checkpoint.json'))];env=json.loads((OUT/'environment.json').read_text());g=json.loads((OUT/'gates.json').read_text()) if (OUT/'gates.json').exists() else None
 lists={s:json.loads((OUT/f'target_lists_{s}.json').read_text()) if (OUT/f'target_lists_{s}.json').exists() else None for s in SCREENS}
 metrics={k:None for k in ('group_lists','n_T','n_M','gate_D1','gate_D2','gate_D3','depth_report','k562_report_only')};metrics.update(versions=env['versions'],cpu_hours=cpu_hours())
 if g:
  metrics.update(group_lists=lists,n_T=g['n_T'],n_M=g['n_M'],gate_D1=g['gates']['D1'],gate_D2=g['gates']['D2'],gate_D3=g['gates']['D3'],k562_report_only=g['comparisons']['k562'])
  import pandas as pd
  depth=pd.read_csv(OUT/'capture.csv',float_precision='round_trip');depth=depth[(depth.normalisation=='N1')&(depth.tertile=='weak')]
  metrics['depth_report']={'library_size_per_group':pd.read_csv(OUT/'library_groups.csv',float_precision='round_trip').to_dict('records'),'weak_cos_depth_axis_per_target':depth[['screen','target','group_T','group_M','cos_depth_axis_N1_before_projection']].to_dict('records'),'depth_control_regression':{s:json.loads((OUT/f'controls_{s}.json').read_text())['depth_regression_R2'] for s in SCREENS}}
 observations=['All controls are retained in every displacement reference (RPE1 11,485; K562 10,691).','Both screens\' control-only predictors and 10,000 covariance-matched QR bases were saved and hashed before any S18d perturbed expression access.']
 if g:
  observations.append(f"RPE1 qualifying T={g['n_T']['rpe1']}, M={g['n_M']['rpe1']}; fixed outcome: {g['outcome']}.")
  for s in SCREENS:
   for n in NORMS:
    r=g['comparisons'][s][n]['weak'];observations.append(f"{s.upper()} {n} weak T median score-J {fmt(r['median_S_J_T'])}, covariance {fmt(r['median_S_C_T'])}, shared-basis median-null95 {fmt(r['null_across_T_median_p95'])}.")
  for s in SCREENS:
   observations.append(f"{s.upper()} N1 depth-plane removal leaves rank {json.loads((OUT/f'controls_{s}.json').read_text())['N1_coordinate_rank']} in the 20-dimensional coordinates; the literal fixed stiffness-eigenvalue positivity rule was not changed.")
  observations.append('K562 report-only weak T>M score-J comparisons have one-sided p=0.002136716264471399 (N1) and 0.011942540127144782 (N2); its D1 and D2 numerical criteria do not hold in either normalisation.')
 deviations=['Authorized planner amendment ac3f502 fixes median null aggregation, array_split tertiles and common accepted original-log-PCA20 significance cohort before any S18d result.','S18c\'s accepted controls-only log transform/HVGs are reused from verified retained caches; N1 is fitted in its depth-projected coordinates and N2 residual PCA is fitted controls-only. No score fit is reused across different normalisations.','Cloud execution reuses .venv and both original files; no Mac, merge, push, protected-source edit or test modification.']
 if g and g['underpowered']:deviations.append('Fewer than 10 qualifying RPE1 T targets: fixed underpowered rule. Defined raw descriptive medians/tests are still reported, but no confirmatory gate pass is claimed.')
 if (OUT/'gene_annotation_repair.json').exists():deviations.append('An initial gene-ID/symbol implementation mismatch was corrected before perturbed X access: raw var.index is Ensembl IDs, while var.gene_name contains target symbols. Initial empty qualification diagnostic preserved; frozen subspaces, numerical control arrays and hashes unchanged; no refit or significance retest.')
 if (OUT/'compile_repair.json').exists():deviations.append('A final-audit indentation SyntaxError was fixed before the freeze command could execute; no scientific unit was repeated.')
 artifacts=[]
 for p in [ROOT/'scripts/s18d_dose.py',*sorted(OUT.rglob('*')),*sorted(FIG.rglob('*'))]:
  if not p.is_file() or p.name in ('RESULT_S18d.md',) or p.name.endswith('.tmp'):continue
  rel=str(p.relative_to(ROOT));artifacts.append({'path':rel,'bytes':p.stat().st_size,'sha256':digest(p),'committed':subprocess.run(['git','cat-file','-e','HEAD:'+rel],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0,'regenerate':artifact_command(rel)})
 r={'task':'orient/S18d','status':status,'started_utc':min(u['started_utc'] for u in units),'finished_utc':now() if status=='done' else None,'compute':{'cpu_hours':cpu_hours(),'gpu_hours':0,'hardware':env,'units':units},'metrics':metrics,'artifacts':artifacts,'deviations':deviations,'observations':observations,'suggestions':['The planner could consider a separately pre-registered higher-cell-count RPE1 screen to address the one-qualifying-T-target power limitation; no follow-up, parameter change or cohort relaxation was run.'],'questions':[],'outcome':g['outcome'] if g else None,'task_sha256':digest(TASK),'planner_amendment_commit':json.loads((OUT/'preregistration.json').read_text())['planner_amendment_commit'],'integrity':json.loads((OUT/'input_verification.json').read_text()),'final_verification':json.loads((OUT/'final_verification.json').read_text()) if (OUT/'final_verification.json').exists() else None}
 dump(ROOT/'coord/orient/reports/S18d.json',r)
 lines=['# orient/S18d — '+status,'','## Summary','',f"Pre-registered RPE1 confirmation: **{g['outcome'] if g else 'in progress'}**; D1–D3 must hold separately in both N1 and N2.", 'K562 and medium/strong tertiles are report-only. Every shift is against all controls. Predictor hashes precede perturbed access.','',f'Recorded CPU: {cpu_hours():.6f} h / 16 h. No pushes.','','## What ran','', 'Fresh reruns and cache restoration: `benchmarks/perturb_dose/REPRODUCE.md`.','']+[f"- `{' '.join(u['command'])}` — exit {u['exit_code']}; wall {u['wall_seconds']:.3f}s; CPU {u['cpu_hours']:.6f}h; max RSS {u['max_rss_kib']} KiB." for u in sorted(units,key=lambda u:u['started_utc'])]
 lines+=['','## Gate numbers','','| Screen | Norm | n T / M | Median SJ(T) | Median SC(T) | Null95 of T medians | D1 numeric | D2 p (SJ > SC) | D3 p (T > M) |','|---|---|---:|---:|---:|---:|---|---:|---:|']
 if g:
  for s in SCREENS:
   for n in NORMS:
    c=g['comparisons'][s][n]['weak'];wp=c['wilcoxon']['p_one_sided_greater'] if c['wilcoxon'] else None;mp=c['mannwhitney']['p_one_sided_greater'] if c['mannwhitney'] else None
    lines.append(f"| {s.upper()}{' (report)' if s=='k562' else ''} | {n} | {c['n_T']} / {c['n_M']} | {fmt(c['median_S_J_T'])} | {fmt(c['median_S_C_T'])} | {fmt(c['null_across_T_median_p95'])} | {c['D1_criterion_met']} | {fmt(wp)} | {fmt(mp)} |")
  lines+=['',f"Primary gates: `{json.dumps(g['gates'])}`",'',f"RPE1 outcome: **{g['outcome']}**. Underpowered threshold is fixed at fewer than 10 T targets; no discovery pass is asserted in that case.",'','Medium/strong report-only comparisons: see `gates.json` and per-target `capture.csv`.']
 lines+=['','## Deviations','']+['- '+x for x in deviations]+['','## Observations','']+['- '+x for x in observations]+['','## Suggestions','']+['- '+x for x in r['suggestions']]+['','## Questions','','None. Approved conventions are committed in the task.','','## Artifacts','','`benchmarks/perturb_dose/` contains subspaces and SHA-256 file, dose/capture tables, gates, control metadata, immutable target lists and exclusions, exact cache files and timed receipts. Figures are under `benchmarks/figures/perturb_dose/`. Raw H5ADs are excluded from the delivery ZIP; original URLs, sizes and SHA-256 are recorded in `REPRODUCE.md` and integrity metadata. Complete artifact sizes, SHA-256 and regeneration/restoration instructions are in the JSON report.','']
 text='\n'.join(lines);(ROOT/'coord/orient/reports/S18d.md').write_text(text);(OUT/'RESULT_S18d.md').write_text(text);return 0


def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
 u=sub.add_parser('unit');u.add_argument('name');u.add_argument('command',nargs=argparse.REMAINDER)
 for act in ('preflight','freeze-controls','aggregate','figures','verify-final','prepare-delivery','finalize-delivery','repair-gene-mapping'):sub.add_parser(act)
 for act in ('controls','select-targets','dose'):
  x=sub.add_parser(act);x.add_argument('--screen',choices=SCREENS,required=True)
 for act in ('subspaces','capture'):
  x=sub.add_parser(act);x.add_argument('--screen',choices=SCREENS,required=True);x.add_argument('--norm',choices=NORMS,required=True)
 r=sub.add_parser('report');r.add_argument('--status',choices=('partial','done','blocked'),default='partial')
 a=p.parse_args()
 if a.action=='unit':return run_unit(a.name,a.command[1:] if a.command and a.command[0]=='--' else a.command)
 if a.action=='preflight':return preflight()
 if a.action=='controls':return prepare_controls(a.screen)
 if a.action=='subspaces':return fit_subspace(a.screen,a.norm)
 if a.action=='freeze-controls':return freeze_controls()
 if a.action=='select-targets':return select_targets(a.screen)
 if a.action=='dose':return dose_cells(a.screen)
 if a.action=='capture':return capture_targets(a.screen,a.norm)
 if a.action=='aggregate':return aggregate()
 if a.action=='figures':make_figures();return 0
 if a.action=='report':return report(a.status)
 if a.action=='verify-final':return verify_final()
 if a.action=='repair-gene-mapping':return repair_gene_mapping()
 if a.action=='prepare-delivery':return prepare_delivery()
 return finalize_delivery()


# Final integrity and packaging implementations follow.


def repair_gene_mapping():
 import numpy as np
 import pandas as pd
 freeze=barrier();dest=OUT/'gene_annotation_repair.json'
 if dest.exists():raise RuntimeError('Annotation repair already completed')
 assert not any((CACHE/f'dose_{s}.npz').exists() for s in SCREENS),'STOP: annotation fix must precede perturbed X access'
 before={r['subspace_file']:digest(ROOT/r['subspace_file']) for r in freeze['files']}
 cached={s:digest(CACHE/f'controls_{s}.npz') for s in SCREENS};records=[]
 for s in SCREENS:
  obs,var,shape=metadata(source(s));symbols=var.gene_name.astype(str).to_numpy(dtype=str)
  groups=json.loads((OUT/f'group_symbols_{s}.json').read_text())['groups'];labels=obs.gene.astype(str);freq=labels.value_counts();targets=sorted(set(labels)-{'non-targeting'});symbol_counts=pd.Series(symbols).value_counts();mapping={t:i for i,t in enumerate(symbols) if symbol_counts[t]==1}
  duplicate_targets=sorted({t for t in targets if t in symbol_counts and symbol_counts[t]>1});assert all(freq[t]<150 for t in duplicate_targets),'STOP: duplicate raw gene_name target could qualify'
  with np.load(CACHE/f'controls_{s}.npz',allow_pickle=False) as a:det=a['control_detection_fraction'];mean=a['control_fraction_mean']
  frame=pd.DataFrame([{'target':t,'n_cells':int(freq[t]),'group_T':t in groups['T'],'group_M':t in groups['M'],'gene_present':t in symbol_counts,'control_detection_fraction':float(det[mapping[t]]) if t in mapping else None,'control_mean_target_fraction':float(mean[mapping[t]]) if t in mapping else None} for t in targets]);frame.to_csv(OUT/f'target_dose_metadata_{s}.csv',index=False,float_format='%.17g')
  records.append({'screen':s,'var_index_type':'Ensembl gene IDs','target_symbol_column':'var.gene_name','n_raw_gene_symbols':len(symbols),'n_screen_targets_present':int(frame.gene_present.sum()),'duplicate_targets_all_independently_below_150_cells':{t:int(freq[t]) for t in duplicate_targets},'corrected_metadata_sha256':digest(OUT/f'target_dose_metadata_{s}.csv')})
 initial=OUT/'diagnostics/initial_id_mapping';initial.mkdir(parents=True,exist_ok=True)
 for suffix in ('json','csv'):
  p=OUT/f'target_lists_rpe1.{suffix}'
  assert subprocess.run(['git','cat-file','-e','HEAD:'+str(p.relative_to(ROOT))],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0,'STOP: do not move a committed file'
  p.rename(initial/p.name)
 assert before=={p:digest(ROOT/p) for p in before} and cached=={s:digest(CACHE/f'controls_{s}.npz') for s in SCREENS}
 dump(dest,{'issue':'Initial symbol lookup incorrectly used Ensembl var.index rather than the raw gene_name symbols; initial empty RPE1 qualification diagnostic preserved.','repair':'Use the dataset-provided gene_name column; no alias mapping, imputation, threshold/cohort rule change or extra significance test.','records':records,'frozen_subspace_sha256_unchanged':before,'control_cache_sha256_unchanged':cached,'no_perturbed_X_access':True,'no_fit_repeated':True,'repaired_utc':now()})
 return select_targets('rpe1')


def assert_same_output(actual,expected):
 import numpy as np
 if isinstance(expected,dict):
  assert actual.keys()==expected.keys()
  for k in expected:assert_same_output(actual[k],expected[k])
 elif isinstance(expected,list):
  assert len(actual)==len(expected)
  for a,b in zip(actual,expected):assert_same_output(a,b)
 elif isinstance(expected,float):np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
 else:assert actual==expected


def verify_final():
 import numpy as np
 import pandas as pd
 freeze=barrier();g=json.loads((OUT/'gates.json').read_text());assert cpu_hours()<BUDGET
 for test in ('tests_before','tests_after'):assert json.loads((LOGS/f'{test}.checkpoint.json').read_text())['exit_code']==0
 verified={};audits=[]
 for s,(p,size,sha) in INPUTS.items():
  assert p.stat().st_size==size and digest(p)==sha,'STOP: final original checksum mismatch';verified[s]={'bytes':size,'sha256':sha,'checked_utc':now()};print(s,'final original SHA unchanged',flush=True)
 entries=(OUT/'subspaces.sha256').read_text().strip().splitlines();assert len(entries)==4
 for line in entries:
  sha,name=line.split();assert digest(OUT/name)==sha
 for s in SCREENS:
  lists=json.loads((OUT/f'target_lists_{s}.json').read_text());doseinfo=json.loads((OUT/f'dose_{s}.json').read_text());assert freeze['saved_utc']<doseinfo['first_perturbed_expression_access_utc'];assert lists['recorded_utc']<doseinfo['first_perturbed_expression_access_utc'];assert lists['common_N1_N2_cohort']
  with np.load(CACHE/f'dose_{s}.npz',allow_pickle=False) as a:labels=a['labels'];kd=a['knockdown'];tertile=a['tertile'];ci=a['control_indices'];z1=a['z_N1'];z2=a['z_N2']
  for t in lists['analysed_targets']:
   row=np.flatnonzero(labels==t);sort=row[np.lexsort((row,kd[row]))]
   for j,b in enumerate(np.array_split(sort,3)):assert (tertile[b]==j).all()
  for norm,z in [('N1',z1),('N2',z2)]:
   frame=pd.read_csv(OUT/f'capture_{s}_{norm}.csv',float_precision='round_trip');assert len(frame)==3*len(lists['analysed_targets']);assert not frame.duplicated(['target','tertile']).any()
   with np.load(OUT/f'subspaces_{s}_{norm}.npz',allow_pickle=False) as a:sj=a['S_J'];sc=a['S_C'];bases=a['random_bases'];mu=a['control_mean'];cov=a['control_covariance'];root=a['control_covariance_sqrt'];H=a['H_bulk'];w=a['stiffness_eigenvalues'];selected=a['selected_eigenvalue_indices']
   np.testing.assert_allclose(root@root.T,cov,rtol=1e-9,atol=1e-9);np.testing.assert_allclose(sj.T@sj,np.eye(3),atol=1e-10);np.testing.assert_allclose(sc.T@sc,np.eye(3),atol=1e-10)
   vals,v=np.linalg.eigh(-.5*(H+H.T));pos=np.flatnonzero(vals>0);idx=pos[:3] if len(pos)>=3 else np.argsort(np.abs(vals),kind='stable')[:3];np.testing.assert_array_equal(idx,selected);np.testing.assert_allclose(vals,w,atol=0,rtol=0);np.testing.assert_allclose(v[:,idx]@v[:,idx].T,sj@sj.T,atol=1e-12)
   with np.load(CACHE/f'shifts_{s}_{norm}.npz',allow_pickle=False) as a:D=a['shifts'];np.testing.assert_array_equal(frame.target,a['targets']);np.testing.assert_array_equal(frame.tertile,a['tertiles'])
   for i,r in enumerate(frame.to_dict('records')):
    j=('weak','medium','strong').index(r['tertile']);rows=np.flatnonzero((labels==r['target'])&(tertile==j));direct=z[rows].mean(0,dtype=np.float64)-mu;np.testing.assert_allclose(D[i],direct,rtol=0,atol=0);assert r['reference_n_controls']==len(ci)
   den=np.sum(D*D,axis=1);cj=np.sum((D@sj)**2,axis=1)/den;cc=np.sum((D@sc)**2,axis=1)/den;np.testing.assert_allclose(frame.capture_S_J,cj,rtol=1e-12,atol=1e-12);np.testing.assert_allclose(frame.capture_S_C,cc,rtol=1e-12,atol=1e-12)
   mask=frame.group_T.to_numpy()&(frame.tertile.to_numpy()=='weak');DT=D[mask]
   if len(DT):
    coeff=np.einsum('bdk,td->btk',bases,DT,optimize=True);nullmed=np.median(np.sum(coeff*coeff,axis=2)/np.sum(DT*DT,axis=1),axis=1)
    with np.load(CACHE/f'null_medians_{s}_{norm}.npz',allow_pickle=False) as a:np.testing.assert_allclose(nullmed,a['weak'],rtol=1e-12,atol=1e-12)
   else:nullmed=None
   comp=comparison(frame[mask],frame[frame.group_M&(frame.tertile=='weak')],nullmed);saved=g['comparisons'][s][norm]['weak'];assert_same_output(comp,saved)
   audits.append({'screen':s,'normalisation':norm,'n_capture_rows':len(frame),'n_T':len(DT),'all_controls_reference':len(ci),'raw_capture_and_tests_roundtrip_verified':True})
 changed=subprocess.check_output(['git','diff','--name-only','f3e166ed298bce92d812ffd5ea3c24b765b3a490...HEAD'],cwd=ROOT,text=True).splitlines()
 allowed=lambda p:p=='scripts/s18d_dose.py' or p=='coord/orient/tasks/S18d.md' or p.startswith(('benchmarks/perturb_dose/','benchmarks/figures/perturb_dose/','coord/orient/reports/S18d.'))
 assert all(allowed(p) for p in changed),'STOP: unauthorized changed path'
 assert g['underpowered']==(g['n_T']['rpe1']<10)
 for gate,r in g['gates'].items():
  crit={n:g['comparisons']['rpe1'][n]['weak'][gate+'_criterion_met'] for n in NORMS};assert r['by_normalisation']==crit
  if g['underpowered']:assert r['passed'] is None and r['status']=='underpowered'
 dump(OUT/'final_verification.json',{'original_inputs':verified,'published_output_audits':audits,'all_fixed_tertile_assignments_verified':True,'all_subspace_hashes_verified':True,'pre_perturbation_ordering_verified':True,'changed_paths_audited_against_imported_head':changed,'authorized_task_amendment':'ac3f502','mandatory_tests_before_after_pass':True,'cpu_hours_before_audit_receipt':cpu_hours(),'finished_utc':now()})
 print('Final original input, all-control shift, exact table/test, and ordering audit PASS',flush=True);return 0


def prepare_delivery():
 import shutil
 import zipfile
 barrier();dest=ROOT.parent/'S18d_benchmarks.zip'
 if dest.exists():raise RuntimeError('Delivery ZIP exists; inspect rather than overwrite')
 # Small prior control-only caches are retained for exact transform restoration, not perturbation fitting.
 refs=OUT/'reference_inputs';refs.mkdir(exist_ok=True)
 for s in SCREENS:
  src=ROOT/f'benchmarks/perturb_valleys/cache/controls_{s}.npz';target=refs/f'controls_{s}.npz'
  if not target.exists():shutil.copy2(src,target)
  assert digest(target)==digest(src)
 report('done')
 files=[p for p in OUT.rglob('*') if p.is_file()]+[p for p in FIG.rglob('*') if p.is_file()]+[ROOT/'coord/orient/reports/S18d.md',ROOT/'coord/orient/reports/S18d.json',ROOT/'scripts/s18d_dose.py']
 with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(files):
   if p.suffix=='.h5ad':raise RuntimeError('Raw input must not enter delivery ZIP')
   z.write(p,str(p.relative_to(ROOT)))
 with zipfile.ZipFile(dest) as z:assert z.testzip() is None
 dump(OUT/'delivery_preparation.json',{'archive_filename':dest.name,'archive_bytes_before_final_accounting':dest.stat().st_size,'archive_crc_verified':True,'raw_H5ADs_excluded':True,'finished_utc':now()})
 print('Delivery benchmark ZIP prepared and CRC verified',dest.stat().st_size,flush=True);return 0


def finalize_delivery():
 import tempfile
 import zipfile
 dest=ROOT.parent/'S18d_benchmarks.zip';bundle=ROOT.parent/'S18d_result.bundle'
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'Task branch must be clean before bundle'
 assert cpu_hours()<=BUDGET and dest.exists() and not bundle.exists()
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 subprocess.run(['git','-c','pack.threads=6','bundle','create',str(bundle),'manus/orient-S18d'],cwd=ROOT,check=True)
 subprocess.run(['git','bundle','verify',str(bundle)],cwd=ROOT,check=True)
 with tempfile.TemporaryDirectory(prefix='S18d-bundle-check-') as temp:
  repo=Path(temp)/'repo.git';subprocess.run(['git','clone','--bare',str(bundle),str(repo)],check=True);subprocess.run(['git','fsck','--full'],cwd=repo,check=True)
  assert subprocess.check_output(['git','rev-parse','manus/orient-S18d'],cwd=repo,text=True).strip()==head
 old=dest.with_name('S18d_benchmarks.preaccounting.zip');dest.rename(old)
 files=[p for p in OUT.rglob('*') if p.is_file()]+[p for p in FIG.rglob('*') if p.is_file()]+[ROOT/'coord/orient/reports/S18d.md',ROOT/'coord/orient/reports/S18d.json',ROOT/'scripts/s18d_dose.py']
 with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as zout:
  for p in sorted(files):
   assert p.suffix!='.h5ad'
   zout.write(p,str(p.relative_to(ROOT)))
 with zipfile.ZipFile(dest) as z:
  assert z.testzip() is None
  for rel in ('coord/orient/reports/S18d.md','coord/orient/reports/S18d.json'):assert z.read(rel)==(ROOT/rel).read_bytes()
  assert 'benchmarks/perturb_dose/capture.csv' in z.namelist() and 'benchmarks/perturb_dose/dose.csv' in z.namelist()
 receipt={'head':head,'branch_clean':True,'bundle_offline_clone_fsck_verified':True,'archive_crc_verified':True,'raw_H5ADs_excluded':True,'recorded_cpu_hours':cpu_hours(),'downloads':[],'finished_utc':now()}
 for p in (bundle,dest,ROOT/'coord/orient/reports/S18d.md',ROOT/'coord/orient/reports/S18d.json'):
  r={'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)};receipt['downloads'].append(r);print(json.dumps(r),flush=True)
 dump(ROOT.parent/'S18d_delivery_manifest.json',receipt)
 print('FINAL REQUESTED DOWNLOADS VERIFIED; clean branch',head,flush=True);return 0


if __name__=='__main__':sys.exit(main())
