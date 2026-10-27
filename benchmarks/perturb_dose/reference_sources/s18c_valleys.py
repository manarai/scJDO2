#!/usr/bin/env python3
"""Pre-registered orient/S18c; cloud-only, one measured unit at a time.

No perturbed expression/coordinates are read before all control-only valley
files and SHA-256 receipts have been saved. No automatic fit/test retries.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'benchmarks/perturb_valleys'
CACHE = OUT/'cache'
LOGS = OUT/'run_logs'
FIG = ROOT/'benchmarks/figures/perturb_valleys'
TASK = ROOT/'coord/orient/tasks/S18c.md'
BUDGET = 14.0
KS = (20,15,30)
VARIANTS = ((20,5),(15,5),(30,5),(20,3),(20,8))
K562_SHA = '3e5a63a9e892b21029bb55fca4e12517a49aad7af6c14133ca63d12cf68c6cee'
RPE1_SHA = '9b05ef1f81526216fa008d677e9e0d03dce9a2f7a95499a4fb81e505e9d88ef1'
K562_DATA = ROOT/'benchmarks/perturb_k562/data/K562_essential_raw_singlecell_01.h5ad'
RPE1_DATA = OUT/'data/rpe1_raw_singlecell_01.h5ad'
RPE1_URL = 'https://ndownloader.figshare.com/files/35775606'
RPE1_SIZE = 8700873216
RPE1_MD5 = '6a2a9d0d2bf4ec147f4d1104043b268c'
T_PATTERNS = ['MED*','TAF*','GTF2*','POLR2*','INTS*','SUPT4H1','SUPT5H','SUPT6H','SUPT16H','PAF1','CTR9','LEO1','CDC73','RTF1']
M_PATTERNS = ['MRPL*','MRPS*','TIMM*','TOMM*','PHB','PHB2','HSPA9','EIF2S1','EIF2S2','EIF2S3','EIF2B*','EIF3*','*ARS','*ARS2','GSPT1','HBS1L']


def now():
    return dt.datetime.now(dt.UTC).isoformat()


def digest(path,algorithm='sha256'):
    h=hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        while block:=f.read(8*1024*1024): h.update(block)
    return h.hexdigest()


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def save_npz(path,**arrays):
    import numpy as np
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise RuntimeError('Completed artifact exists; no overwrite: '+str(path))
    tmp=path.with_name(path.stem+'.tmp.npz')
    if tmp.exists(): raise RuntimeError('Partial artifact exists; inspect it: '+str(tmp))
    np.savez_compressed(tmp,**arrays);tmp.replace(path)


def cpu_hours():
    return sum(json.loads(p.read_text())['cpu_hours'] for p in LOGS.glob('*.checkpoint.json'))


def run_unit(name,command):
    LOGS.mkdir(parents=True,exist_ok=True)
    checkpoint=LOGS/f'{name}.checkpoint.json'
    if checkpoint.exists(): raise RuntimeError('Checkpoint exists: inspect instead of repeating '+str(checkpoint))
    used=cpu_hours()
    if used>=BUDGET: raise RuntimeError('STOP: CPU budget exhausted')
    threads=min(6,os.cpu_count() or 1)
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','LOKY_MAX_CPU_COUNT'): env[key]=str(threads)
    env['PYTHONUNBUFFERED']='1';env['PATH']=str(ROOT/'.venv/bin')+os.pathsep+env.get('PATH','');env['PYTHONPATH']=str(ROOT)
    started=now();wall=time.monotonic();before=resource.getrusage(resource.RUSAGE_CHILDREN);logpath=LOGS/f'{name}.log'
    print(f'[{started}] {name}: {command}',flush=True)
    with logpath.open('w') as log:
        p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        while True:
            try: code=p.wait(timeout=30);break
            except subprocess.TimeoutExpired:
                elapsed=time.monotonic()-wall
                print(f'[{now()}] {name} running; wall {elapsed:.0f}s; {logpath.relative_to(ROOT)}',flush=True)
                if used+(elapsed+40)*threads/3600>=BUDGET:
                    p.terminate()
                    try: p.wait(timeout=10)
                    except subprocess.TimeoutExpired: p.kill();p.wait()
                    code=124;print('STOP: conservative remaining CPU budget limit',flush=True);break
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt={'unit':name,'command':command,'started_utc':started,'finished_utc':now(),'exit_code':code,'wall_seconds':time.monotonic()-wall,
             'cpu_hours':(after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime)/3600,'cpu_hours_before':used,'max_rss_kib':after.ru_maxrss,
             'threads':threads,'cpu_count':os.cpu_count(),'ram_bytes':os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES'),'log':str(logpath.relative_to(ROOT))}
    dump(checkpoint,receipt)
    print('\n'.join(logpath.read_text(errors='replace').splitlines()[-20:]),flush=True);print(json.dumps(receipt,indent=2),flush=True)
    return code


def check_task():
    pre=json.loads((OUT/'preregistration.json').read_text())
    assert pre['implementation_conventions_status']=='approved verbatim by planner before any RPE1 result'
    assert digest(TASK)==pre['task_sha256'],'STOP: frozen amended task changed'
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='manus/orient-S18c'
    assert json.loads((LOGS/'tests_before.checkpoint.json').read_text())['exit_code']==0
    return pre


def source(screen):
    check_task()
    if screen=='k562':
        d=json.loads((OUT/'input_verification_k562.json').read_text());p=K562_DATA;sha=d['actual_sha256'];size=d['bytes'];mtime=d['mtime_ns'];expected=K562_SHA
    else:
        d=json.loads((OUT/'rpe1_source.json').read_text());p=RPE1_DATA;sha=d['actual_sha256'];size=d['actual_bytes'];mtime=d['verified_mtime_ns'];expected=RPE1_SHA
        assert d['verified'] and d['actual_md5']==RPE1_MD5
    s=p.stat();assert sha==expected and s.st_size==size and s.st_mtime_ns==mtime,'STOP: original file changed'
    return p


def preflight():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'input_verification_k562.json').exists(): raise RuntimeError('Preflight already completed')
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='manus/orient-S18c'
    actual=digest(K562_DATA)
    if actual!=K562_SHA: raise RuntimeError('STOP: K562 checksum differs: '+actual)
    s=K562_DATA.stat()
    dump(OUT/'input_verification_k562.json',{'path':str(K562_DATA.relative_to(ROOT)),'required_sha256':K562_SHA,'actual_sha256':actual,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'verified_utc':now(),'read_only':True})
    versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','pandas','anndata','scanpy','scikit-learn','scikit-misc','torch','matplotlib','pytest')}
    dump(OUT/'environment.json',{'python':platform.python_version(),'platform':platform.platform(),'cpu_count':os.cpu_count(),'ram_bytes':os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES'),'existing_venv_reused':True,'versions':versions})
    print('K562 SHA verified unchanged:',actual,flush=True);return 0


def download_rpe1():
    partial=RPE1_DATA.with_name('rpe1_raw_singlecell_01.partial.h5ad')
    if RPE1_DATA.exists() or partial.exists(): raise RuntimeError('Input or partial exists: inspect; no overwrite')
    RPE1_DATA.parent.mkdir(parents=True,exist_ok=True);sha=hashlib.sha256();md5=hashlib.md5();size=0;next_progress=256*1024*1024;started=now()
    with urllib.request.urlopen(RPE1_URL,timeout=120) as response,partial.open('wb') as f:
        while block:=response.read(8*1024*1024):
            f.write(block);sha.update(block);md5.update(block);size+=len(block)
            if size>=next_progress: print(f'[{now()}] downloaded {size}/{RPE1_SIZE} bytes',flush=True);next_progress+=256*1024*1024
    d={'article_id':20029387,'metadata_url':'https://api.figshare.com/v2/articles/20029387','file_id':35775606,'file_name':RPE1_DATA.name,'download_url':RPE1_URL,
       'expected_bytes':RPE1_SIZE,'actual_bytes':size,'figshare_supplied_md5':RPE1_MD5,'actual_md5':md5.hexdigest(),'actual_sha256':sha.hexdigest(),
       'started_utc':started,'finished_utc':now(),'verified':size==RPE1_SIZE and md5.hexdigest()==RPE1_MD5}
    dump(OUT/'rpe1_source.json',d)
    if not d['verified']: raise RuntimeError('STOP: RPE1 original size/MD5 mismatch')
    partial.replace(RPE1_DATA);RPE1_DATA.chmod(0o444);d['verified_mtime_ns']=RPE1_DATA.stat().st_mtime_ns;dump(OUT/'rpe1_source.json',d)
    (RPE1_DATA.parent/'CHECKSUMS.txt').write_text(sha.hexdigest()+'  '+RPE1_DATA.name+'\n')
    print('RPE1 original verified:',sha.hexdigest(),size,'bytes',flush=True);return 0


def metadata(path):
    import h5py
    from anndata._io.specs import read_elem
    with h5py.File(path,'r') as f:
        obs=read_elem(f['obs']);var=read_elem(f['var']);x=f['X']
        shape=x.shape if isinstance(x,h5py.Dataset) else tuple(x.attrs['shape'])
    assert 'gene' in obs and not obs.gene.isna().any(),'STOP: missing required obs gene'
    return obs,var,shape


def read_rows(x,indices,n_genes):
    """Fetch exactly the listed sorted rows; never global blocks before freeze."""
    import h5py
    import numpy as np
    if isinstance(x,h5py.Dataset): return np.asarray(x[indices,:],dtype=np.float32)
    kind=x.attrs.get('encoding-type','')
    if isinstance(kind,bytes): kind=kind.decode()
    if kind!='csr_matrix': raise RuntimeError('STOP: raw X storage is not dense or CSR')
    out=np.zeros((len(indices),n_genes),dtype=np.float32)
    for row,i in enumerate(indices):
        lo,hi=x['indptr'][i:i+2];out[row,x['indices'][lo:hi]]=x['data'][lo:hi]
    return out


def resolve_groups(screen,labels):
    import fnmatch
    import numpy as np
    import pandas as pd
    p=OUT/f'group_lists_{screen}.json'
    if p.exists(): raise RuntimeError('Group resolution exists; no change')
    counts=pd.Series(labels).value_counts();targets=sorted(t for t in counts.index if t!='non-targeting')
    groups={name:[t for t in targets if any(fnmatch.fnmatchcase(t,pattern) for pattern in patterns)] for name,patterns in [('T',T_PATTERNS),('M',M_PATTERNS)]}
    dump(p,{'screen':screen,'resolved_utc':now(),'patterns':{'T':T_PATTERNS,'M':M_PATTERNS},'targets':targets,'groups':groups,
            'eligible_groups':{name:[t for t in arr if counts[t]>=100] for name,arr in groups.items()},'overlap':sorted(set(groups['T'])&set(groups['M'])),
            'source':'Raw obs gene labels only; before any alignment and without examining perturbed expression'})
    pd.DataFrame([{'target':t,'n_cells':int(counts[t]),'eligible':bool(counts[t]>=100),'group_T':t in groups['T'],'group_M':t in groups['M']} for t in targets]).to_csv(OUT/f'target_metadata_{screen}.csv',index=False)
    return groups


def restore_pca(data):
    from sklearn.decomposition import PCA
    p=PCA(n_components=30,svd_solver='full',whiten=False)
    p.mean_=data['pca_mean'];p.components_=data['pca_components'];p.explained_variance_=data['explained_variance'];p.n_features_in_=len(p.mean_);p.n_components_=30
    return p


def prepare_controls(screen):
    import anndata as ad
    import h5py
    import numpy as np
    import scanpy as sc
    from sklearn.decomposition import PCA
    path=source(screen);dest=CACHE/f'controls_{screen}.npz'
    if dest.exists(): raise RuntimeError('Control embedding exists: no repeat')
    obs,var,shape=metadata(path);labels=obs.gene.astype(str).to_numpy(dtype=str);ci=np.flatnonzero(labels=='non-targeting')
    if len(ci)<2000 or len(var)<2000: raise RuntimeError('STOP: S18 recipe requires >=2000 controls and genes')
    if screen=='k562': assert shape==(310385,8563) and len(ci)==10691
    resolve_groups(screen,labels)
    available=os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')
    if len(ci)*len(var)*4>available*.4: raise RuntimeError('STOP: selected control count matrix would exceed memory reserve')
    counts=np.empty((len(ci),len(var)),dtype=np.float32)
    with h5py.File(path,'r') as f:
        for start in range(0,len(ci),512):
            end=min(start+512,len(ci));counts[start:end]=read_rows(f['X'],ci[start:end],len(var))
            if start%4096==0: print(f'[{now()}] {screen}: selected control rows {end}/{len(ci)} only',flush=True)
    assert np.isfinite(counts).all() and (counts>=0).all() and np.equal(counts,np.floor(counts)).all(),'STOP: invalid raw counts'
    libraries=counts.sum(axis=1,dtype=np.float64)
    assert (libraries>0).all(),'STOP: zero-library control'
    median=float(np.median(libraries));cache_provenance=None
    if screen=='k562':
        old=ROOT/'benchmarks/perturb_k562/cache/embedding.npz'
        report=json.loads((ROOT/'coord/orient/reports/S18.json').read_text())
        artifact=next(a for a in report['artifacts'] if a['path']==str(old.relative_to(ROOT)))
        assert digest(old)==artifact['sha256'],'STOP: accepted S18 embedding changed'
        with np.load(old,allow_pickle=False) as a:
            # Do not read z or labels: they contain perturbed-cell coordinates.
            hvg=a['hvg'];pmean=a['pca_mean'];components=a['pca_components'];variance=a['explained_variance'];np.testing.assert_array_equal(ci,a['control_indices'])
        pca=restore_pca({'pca_mean':pmean,'pca_components':components,'explained_variance':variance})
        assert median==13714.0
        cache_provenance={'path':str(old.relative_to(ROOT)),'sha256':artifact['sha256'],'members_read':['hvg','pca_mean','pca_components','explained_variance','control_indices'],'z_and_labels_not_read':True}
    else:
        a=ad.AnnData(X=counts,var=var.copy());sc.pp.highly_variable_genes(a,flavor='seurat_v3',n_top_genes=2000);hvg=a.var.highly_variable.to_numpy();del a
    assert int(hvg.sum())==2000
    logged=np.array(counts[:,hvg],dtype=np.float32,copy=True);logged*=(median/libraries).astype(np.float32)[:,None];np.log1p(logged,out=logged);del counts
    if screen=='rpe1': pca=PCA(n_components=30,svd_solver='full',whiten=False).fit(logged)
    z=pca.transform(logged).astype(np.float32);del logged
    assert np.isfinite(z).all()
    save_npz(dest,z_controls=z,control_indices=ci,hvg=hvg,pca_mean=pca.mean_,pca_components=pca.components_,explained_variance=pca.explained_variance_,median_library=np.array(median))
    var.assign(selected_control_HVG=hvg).to_csv(OUT/f'hvg_control_{screen}.csv')
    dump(OUT/f'controls_{screen}.json',{'screen':screen,'n_cells':int(shape[0]),'n_genes':int(shape[1]),'n_controls':len(ci),'median_control_library':median,'n_hvg':2000,
         'fit_cells':'non-targeting controls only','pca_dimensions':[15,20,30],'pca_solver':'full','pca_whiten':False,'raw_counts_access':'Exactly sorted selected control rows only; h5py X reads audited by command path',
         'pca_reused_from_S18':cache_provenance,'group_lists_resolved_before_expression':str((OUT/f'group_lists_{screen}.json').relative_to(ROOT)),
         'control_embedding_sha256':digest(dest),'finished_utc':now()})
    print(screen,'control-only embedding:',z.shape,'median library',median,flush=True);return 0


def score_controls(screen,k):
    import numpy as np
    import torch
    from scjdo2.energy_drift import EnergyDriftAdapter,EnergyDriftConfig
    source(screen);dest=CACHE/f'H_{screen}_k{k}.npz'
    if dest.exists(): raise RuntimeError('Completed H cache exists: no refit')
    with np.load(CACHE/f'controls_{screen}.npz',allow_pickle=False) as a: z=a['z_controls'][:,:k];ci=a['control_indices']
    cfg=EnergyDriftConfig(hidden=96,depth=3,n_epochs=1500,seed=0,batch_size=256,loss='esm',lr=5e-4,beta=1.0,esm_laplacian='exact')
    provenance=None
    if screen=='k562':
        old=ROOT/f'benchmarks/perturb_k562/cache/H_k{k}.npz'
        r=json.loads((ROOT/'coord/orient/reports/S18.json').read_text());art=next(a for a in r['artifacts'] if a['path']==str(old.relative_to(ROOT)))
        assert digest(old)==art['sha256'],'STOP: accepted S18 H changed'
        with np.load(old,allow_pickle=False) as a: H=a['H'];np.testing.assert_array_equal(ci,a['control_indices'])
        provenance={'path':str(old.relative_to(ROOT)),'sha256':art['sha256'],'fit_repeated':False}
    else:
        torch.set_num_threads(min(6,os.cpu_count() or 1))
        drift=EnergyDriftAdapter(k,cfg).fit(z,verbose=True);H=drift.jacobian_autodiff(z)
        torch.save({'configuration':vars(cfg),'dim':k,'state_dict':drift._model.state_dict(),'fit_cells':'controls only','controls_sha256':digest(CACHE/f'controls_{screen}.npz')},CACHE/f'model_{screen}_k{k}.pt')
    assert H.shape==(len(ci),k,k) and np.isfinite(H).all()
    asym=np.linalg.norm(H-H.transpose(0,2,1),axis=(1,2))/(np.linalg.norm(H,axis=(1,2))+1e-12)
    if asym.max()>1e-4: raise RuntimeError('STOP: S02 score symmetry check exceeds 1e-4')
    save_npz(dest,H=H,control_indices=ci)
    dump(OUT/f'score_{screen}_k{k}.json',{'screen':screen,'k':k,'n_fit_controls':len(ci),'configuration':vars(cfg),'H_sha256':digest(dest),
         'symmetry_max_relative':float(asym.max()),'symmetry_median_relative':float(np.median(asym)),'S18_control_H_reuse':provenance,'finished_utc':now()})
    print(screen,k,'H saved',H.shape,'max asymmetry',float(asym.max()),flush=True);return 0


def valley_path(screen,k,K):
    return OUT/f'valleys_{screen}.npz' if (k,K)==(20,5) else OUT/f'valleys_{screen}_pca{k}_K{K}.npz'


def build_valleys(screen,k):
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors
    source(screen);receipt=OUT/f'valleys_{screen}_k{k}.json'
    if receipt.exists(): raise RuntimeError('Valley receipt exists: no repeat')
    with np.load(CACHE/f'controls_{screen}.npz',allow_pickle=False) as a: z=a['z_controls'][:,:k];ci=a['control_indices']
    with np.load(CACHE/f'H_{screen}_k{k}.npz',allow_pickle=False) as a: H=a['H'];np.testing.assert_array_equal(ci,a['control_indices'])
    mean=z.mean(axis=0,dtype=np.float64);cov=np.cov(z,rowvar=False);evalues,evectors=np.linalg.eigh(cov)
    assert evalues.min()>0,'STOP: singular control covariance; no regularization allowed'
    centered=z-mean;mahal2=np.einsum('ij,jk,ik->i',centered,np.linalg.inv(cov),centered)
    nedge=math.ceil(.20*len(z));edges=np.lexsort((ci,-mahal2))[:nedge]
    nn=NearestNeighbors(n_neighbors=200,algorithm='brute',n_jobs=1).fit(z);neighbors=nn.kneighbors(z[edges],return_distance=False)
    u=np.empty((nedge,k),dtype=np.float64);s=np.empty(nedge,dtype=np.float64);npositive=np.empty(nedge,dtype=np.int64);dots=np.empty(nedge)
    for i in range(nedge):
        local=H[neighbors[i]].mean(axis=0,dtype=np.float64);values,vectors=np.linalg.eigh(-.5*(local+local.T));positive=np.flatnonzero(values>0)
        j=int(positive[0]) if len(positive) else int(np.argsort(np.abs(values),kind='stable')[0]);vec=vectors[:,j]
        dot=float(vec@(z[edges[i]]-mean))
        if dot<0: vec=-vec;dot=-dot
        if dot==0: raise RuntimeError('STOP: outward signing dot is exactly zero')
        u[i]=vec;s[i]=values[j];npositive[i]=len(positive);dots[i]=dot
    nsoft=math.ceil(.25*nedge);soft=np.lexsort((ci[edges],s))[:nsoft]
    null=np.random.default_rng(0).multivariate_normal(np.zeros(k),cov,size=10000);null/=np.linalg.norm(null,axis=1)[:,None]
    records=[]
    for K in ([5,3,8] if k==20 else [5]):
        p=valley_path(screen,k,K);model=KMeans(n_clusters=K,random_state=0);model.fit(u[soft]);centroids=model.cluster_centers_
        norms=np.linalg.norm(centroids,axis=1)
        if (norms==0).any() or len(np.unique(model.labels_))!=K: raise RuntimeError('STOP: k-means failed to yield K nonzero centroids')
        v=centroids/norms[:,None];baseline=evectors[:,-5:].T;null_A=np.max(np.abs(null@v.T),axis=1);p95=float(np.percentile(null_A,95))
        save_npz(p,valleys=v,covariance_directions=baseline,control_mean=mean,control_covariance=cov,control_covariance_eigenvalues=evalues,
                 edge_control_indices=ci[edges],edge_mahalanobis_distance=np.sqrt(mahal2[edges]),edge_softness=s,edge_unit_vectors=u,
                 soft_control_indices=ci[edges[soft]],kmeans_labels=model.labels_,null_A_J=null_A,null_p95=np.array(p95),pca_k=np.array(k),valley_K=np.array(K),n_controls=np.array(len(z)))
        sha=digest(p);p.with_suffix('.sha256').write_text(sha+'  '+p.name+'\n');p.chmod(0o444)
        records.append({'k':k,'K':K,'path':str(p.relative_to(ROOT)),'sha256':sha,'saved_utc':now(),'null_p95':p95,
                        'kmeans_parameters':model.get_params(),'kmeans_inertia':float(model.inertia_),'kmeans_cluster_sizes':np.bincount(model.labels_,minlength=K).tolist()})
    dump(receipt,{'screen':screen,'k':k,'n_controls':len(z),'n_edge':nedge,'n_soft_edge':nsoft,'n_edges_no_positive_eigenvalues':int((npositive==0).sum()),
         'n_soft_edges_no_positive_eigenvalues':int((npositive[soft]==0).sum()),'minimum_outward_dot':float(dots.min()),'softness_min':float(s.min()),'softness_max':float(s.max()),
         'covariance_condition_number':float(evalues.max()/evalues.min()),'H_source_sha256':digest(CACHE/f'H_{screen}_k{k}.npz'),
         'control_embedding_sha256':digest(CACHE/f'controls_{screen}.npz'),'valley_files':records,'perturbed_expression_or_coordinates_read':False,'finished_utc':now()})
    print(screen,k,'valleys frozen:',json.dumps(records),flush=True);return 0


def freeze_all():
    check_task();dest=OUT/'controls_frozen.json'
    if dest.exists(): raise RuntimeError('Frozen-stage barrier already exists')
    records=[]
    for screen in ('rpe1','k562'):
        source(screen)
        assert (OUT/f'group_lists_{screen}.json').exists()
        for k in KS:
            receipt=json.loads((OUT/f'valleys_{screen}_k{k}.json').read_text());assert receipt['perturbed_expression_or_coordinates_read'] is False
            for rec in receipt['valley_files']:
                p=ROOT/rec['path'];assert digest(p)==rec['sha256'];assert p.with_suffix('.sha256').read_text().split()[0]==rec['sha256'];records.append(rec)
        assert not (CACHE/f'projected_{screen}.npz').exists() and not (OUT/f'targets_{screen}.csv').exists()
    dump(dest,{'status':'controls-only stage passed','task_sha256':digest(TASK),'saved_utc':now(),'git_checkpoint_before_perturbed_access':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         'valley_files':records,'group_list_sha256':{s:digest(OUT/f'group_lists_{s}.json') for s in ('rpe1','k562')},'no_perturbed_expression_or_coordinates_read':True})
    print('Hard barrier SAVED: every screen/dimension/K predictor and hash fixed before perturbed access',flush=True);return 0


def barrier():
    check_task();receipt=json.loads((OUT/'controls_frozen.json').read_text());assert receipt['task_sha256']==digest(TASK) and receipt['no_perturbed_expression_or_coordinates_read']
    for r in receipt['valley_files']: assert digest(ROOT/r['path'])==r['sha256'],'STOP: frozen valley changed'
    for s,sha in receipt['group_list_sha256'].items(): assert digest(OUT/f'group_lists_{s}.json')==sha,'STOP: group list changed'
    return receipt


def project_cells(screen):
    import h5py
    import numpy as np
    barrier();path=source(screen);dest=CACHE/f'projected_{screen}.npz'
    if dest.exists(): raise RuntimeError('Projection exists: no repeat')
    started=now();obs,var,shape=metadata(path);labels=obs.gene.astype(str).to_numpy(dtype=str)
    with np.load(CACHE/f'controls_{screen}.npz',allow_pickle=False) as a:
        control_z=a['z_controls'];ci=a['control_indices'];hvg=a['hvg'];median=float(a['median_library']);pca=restore_pca(a)
    if screen=='k562':
        old=ROOT/'benchmarks/perturb_k562/cache/embedding.npz'
        art=next(a for a in json.loads((ROOT/'coord/orient/reports/S18.json').read_text())['artifacts'] if a['path']==str(old.relative_to(ROOT)))
        assert digest(old)==art['sha256']
        with np.load(old,allow_pickle=False) as a: z=a['z'];np.testing.assert_array_equal(a['labels'],labels);np.testing.assert_array_equal(a['control_indices'],ci)
        reuse={'path':str(old.relative_to(ROOT)),'sha256':art['sha256']}
    else:
        z=np.empty((len(labels),30),dtype=np.float32);reuse=None
        with h5py.File(path,'r') as f:
            for start in range(0,len(labels),2048):
                end=min(start+2048,len(labels));block=read_rows(f['X'],np.arange(start,end),len(var));library=block.sum(axis=1,dtype=np.float64)
                assert (library>0).all(),'STOP: zero-library cell'
                logged=np.array(block[:,hvg],dtype=np.float32,copy=True);logged*=(median/library).astype(np.float32)[:,None];np.log1p(logged,out=logged)
                z[start:end]=pca.transform(logged)
                if start%40960==0: print(f'[{now()}] {screen} projected {end}/{len(labels)} cells; frozen predictor unchanged',flush=True)
    np.testing.assert_allclose(z[ci],control_z,rtol=1e-5,atol=1e-5);assert np.isfinite(z).all()
    save_npz(dest,z=z,labels=labels,control_indices=ci)
    dump(OUT/f'projection_{screen}.json',{'screen':screen,'n_cells':len(labels),'first_perturbed_coordinate_access_utc':started,'frozen_stage_saved_utc':json.loads((OUT/'controls_frozen.json').read_text())['saved_utc'],
         'control_projection_max_absolute_difference':float(np.abs(z[ci]-control_z).max()),'S18_projection_reuse':reuse,'sha256':digest(dest),'finished_utc':now()})
    return 0


def effect_tests(screen):
    import numpy as np
    import pandas as pd
    from scipy.spatial.distance import cdist
    from scipy.stats import false_discovery_control
    barrier();source(screen);dest=OUT/f'targets_{screen}.csv'
    if dest.exists(): raise RuntimeError('Effect results exist: no re-seeding/retest')
    if screen=='k562':
        old=ROOT/'benchmarks/perturb_k562/targets.csv';frame=pd.read_csv(old,float_precision='round_trip')
        artifact=next(a for a in json.loads((ROOT/'coord/orient/reports/S18.json').read_text())['artifacts'] if a['path']==str(old.relative_to(ROOT)))
        assert digest(old)==artifact['sha256'],'STOP: accepted S18 target results changed'
        config=json.loads((ROOT/'benchmarks/perturb_k562/effect_tests.json').read_text());config['accepted_S18_effect_results_reused']={'path':str(old.relative_to(ROOT)),'sha256':digest(old)}
    else:
        with np.load(CACHE/f'projected_{screen}.npz',allow_pickle=False) as a: z=a['z'][:,:20];labels=a['labels'];ci=a['control_indices']
        counts=pd.Series(labels).value_counts();targets=sorted(t for t,n in counts.items() if t!='non-targeting' and n>=100);rng=np.random.default_rng(0);records=[]
        for number,t in enumerate(targets,1):
            x=np.asarray(z[labels==t],dtype=np.float64);chosen=rng.choice(ci,size=2000,replace=False);y=np.asarray(z[chosen],dtype=np.float64);pooled=np.concatenate([x,y]);nx,ny,n=len(x),len(y),len(pooled)
            dist=cdist(pooled,pooled,metric='euclidean');weights=np.full((201,n),-1/ny,dtype=np.float64);weights[0,:nx]=1/nx
            for perm in range(1,201): weights[perm,rng.permutation(n)[:nx]]=1/nx
            energies=np.maximum(-np.einsum('ij,ij->i',weights@dist,weights),0)
            direct=2*dist[:nx,nx:].mean()-dist[:nx,:nx].mean()-dist[nx:,nx:].mean();np.testing.assert_allclose(energies[0],direct,rtol=1e-9,atol=1e-9)
            exceed=int((energies[1:]>=energies[0]).sum());records.append({'target':t,'n_cells':nx,'effect_size_energy_distance':float(np.sqrt(energies[0])),'energy_distance_squared':float(energies[0]),'permutation_exceedances':exceed,'p_value':(exceed+1)/201,'n_controls_effect_test':ny})
            if number%25==0 or number==len(targets): print(f'[{now()}] {screen} energy permutations {number}/{len(targets)}',flush=True)
            del dist,weights,pooled
        frame=pd.DataFrame.from_records(records);frame['p_adjusted_BH']=false_discovery_control(frame.p_value.to_numpy(),method='bh');frame['significant']=frame.p_adjusted_BH<.01
        config={'minimum_cells':100,'PCA_k':20,'random_controls_per_target':2000,'permutations':200,'rng':'default_rng(0), one stream; lexicographically sorted targets','energy_definition':'sqrt(2 cross mean - within means), V statistic including diagonals','p_definition':'(exceedances + 1)/201','BH':'<0.01'}
    frame=frame.sort_values('target',kind='stable').reset_index(drop=True);sig=frame[frame.significant].sort_values(['effect_size_energy_distance','target'],ascending=[False,True],kind='stable');nstrong=math.ceil(.20*len(sig));strong=set(sig.head(nstrong).target)
    groups=json.loads((OUT/f'group_lists_{screen}.json').read_text())['groups'];frame['strong']=frame.target.isin(strong);frame['group_T']=frame.target.isin(groups['T']);frame['group_M']=frame.target.isin(groups['M'])
    frame.to_csv(dest,index=False,float_format='%.17g')
    config.update({'screen':screen,'n_targets_eligible':len(frame),'n_targets_significant':len(sig),'n_strong':nstrong,'n_strong_T':int((frame.strong&frame.group_T).sum()),'n_strong_M':int((frame.strong&frame.group_M).sum()),'strong_targets':sorted(strong),'strong_T':frame.loc[frame.strong&frame.group_T,'target'].tolist(),'strong_M':frame.loc[frame.strong&frame.group_M,'target'].tolist(),'confirmation_underpowered':bool(screen=='rpe1' and (frame.strong&frame.group_T).sum()<8),'finished_utc':now()})
    dump(OUT/f'effects_{screen}.json',config);print(screen,'effect/selection:',json.dumps(config),flush=True);return 0


def align_targets(screen):
    import numpy as np
    import pandas as pd
    barrier();source(screen);dest=OUT/f'alignment_{screen}.csv'
    if dest.exists(): raise RuntimeError('Alignment exists: no repeat')
    targets=pd.read_csv(OUT/f'targets_{screen}.csv',float_precision='round_trip');records=[]
    with np.load(CACHE/f'projected_{screen}.npz',allow_pickle=False) as a: z=a['z'];labels=a['labels'];ci=a['control_indices']
    mean=z[ci].mean(axis=0,dtype=np.float64);centroids=np.stack([z[labels==t].mean(axis=0,dtype=np.float64) for t in targets.target]);shifts=centroids-mean
    save_npz(CACHE/f'shifts_{screen}.npz',targets=targets.target.to_numpy(dtype=str),shifts=shifts,all_controls_mean=mean,centroids=centroids)
    for k,K in VARIANTS:
        with np.load(valley_path(screen,k,K),allow_pickle=False) as a: v=a['valleys'];e=a['covariance_directions'];p95=float(a['null_p95'])
        d=shifts[:,:k];norm=np.linalg.norm(d,axis=1)
        if (norm==0).any(): raise RuntimeError('STOP: a target has zero centroid shift and undefined unit vector')
        d=d/norm[:,None];AJ=np.max(np.abs(d@v.T),axis=1);AC=np.max(np.abs(d@e.T),axis=1)
        for i,r in enumerate(targets.to_dict('records')):
            records.append({'screen':screen,'pca_k':k,'valley_K':K,**r,'shift_norm_all_controls':float(norm[i]),'A_J':float(AJ[i]),'A_C':float(AC[i]),'null_A_J_p95':p95,'A_J_exceeds_null95':bool(AJ[i]>p95),'reference_n_controls':len(ci),'confirmatory_configuration':bool(screen=='rpe1' and (k,K)==(20,5))})
    frame=pd.DataFrame.from_records(records);frame.to_csv(dest,index=False,float_format='%.17g')
    dump(OUT/f'alignment_{screen}.json',{'screen':screen,'n_eligible_targets':len(targets),'n_rows':len(frame),'all_shifts_reference':'Mean of ALL controls','reference_n_controls':len(ci),'group_lists_sha256':digest(OUT/f'group_lists_{screen}.json'),'frozen_receipt_sha256':digest(OUT/'controls_frozen.json'),'finished_utc':now()})
    print(screen,'all-control alignment rows',len(frame),flush=True);return 0


def comparisons(group,allow_tests=True):
    import numpy as np
    from scipy.stats import mannwhitneyu,wilcoxon
    T=group[group.group_T];M=group[group.group_M];nT,nM=len(T),len(M)
    median=float(T.A_J.median()) if nT else None;p95=float(group.null_A_J_p95.iloc[0]) if len(group) else None
    out={'n_T':nT,'n_M':nM,'T_targets':T.target.tolist(),'M_targets':M.target.tolist(),'median_A_J_T':median,'median_A_J_M':float(M.A_J.median()) if nM else None,'median_A_C_T':float(T.A_C.median()) if nT else None,
         'null_A_J_p95':p95,'C1_numeric_exceedance':bool(median>p95) if nT and p95 is not None else None,'mann_whitney':None,'paired_wilcoxon':None}
    if allow_tests and nT and nM:
        r=mannwhitneyu(T.A_J.to_numpy(),M.A_J.to_numpy(),alternative='greater',method='auto')
        out['mann_whitney']={'statistic':float(r.statistic),'p_one_sided_greater':float(r.pvalue),'method':'auto','alternative':'greater','use_continuity':True}
    if allow_tests and nT:
        diff=T.A_J.to_numpy()-T.A_C.to_numpy()
        if not np.all(diff==0):
            r=wilcoxon(diff,alternative='greater',zero_method='wilcox',method='auto')
            out['paired_wilcoxon']={'statistic':float(r.statistic),'p_one_sided_greater':float(r.pvalue),'method':'auto','alternative':'greater','zero_method':'wilcox','median_difference_A_J_minus_A_C':float(np.median(diff))}
        else: out['paired_wilcoxon_reason']='All paired differences exactly zero; stated test undefined, no replacement'
    if not nT or not nM: out['undefined_reason']='One or both fixed gene-group populations empty'
    return out


def summarize():
    import pandas as pd
    barrier();dest=OUT/'gates.json'
    if dest.exists(): raise RuntimeError('Summary exists: do not re-evaluate to change result')
    frames=[pd.read_csv(OUT/f'alignment_{s}.csv',float_precision='round_trip') for s in ('rpe1','k562')];allrows=pd.concat(frames,ignore_index=True);allrows.to_csv(OUT/'alignment.csv',index=False,float_format='%.17g')
    effects={s:json.loads((OUT/f'effects_{s}.json').read_text()) for s in ('rpe1','k562')};under=effects['rpe1']['confirmation_underpowered'];summaries={}
    for screen,frame in zip(('rpe1','k562'),frames):
        summaries[screen]={}
        for k,K in VARIANTS:
            group=frame[(frame.pca_k==k)&(frame.valley_K==K)];strong=group[group.strong&group.significant];alltargets=group[group.significant]
            summaries[screen][f'pca{k}_K{K}']={'screen':screen,'pca_k':k,'valley_K':K,'strong':comparisons(strong,allow_tests=not(screen=='rpe1' and under)),
                                           'all_significant_report_only':comparisons(alltargets,allow_tests=not(screen=='rpe1' and under)),
                                           'report_only':not(screen=='rpe1' and (k,K)==(20,5)),'confirmation_underpowered':bool(screen=='rpe1' and under)}
    r=summaries['rpe1']['pca20_K5']['strong'];mw=r['mann_whitney'];w=r['paired_wilcoxon']
    C1={'median_A_J_strong_T':r['median_A_J_T'],'null_p95':r['null_A_J_p95'],'requirement':'strictly greater than null p95','pass':None if under else r['C1_numeric_exceedance']}
    C2={'n_strong_T':r['n_T'],'n_strong_M':r['n_M'],'statistic':mw['statistic'] if mw else None,'p_one_sided_greater':mw['p_one_sided_greater'] if mw else None,'requirement':'one-sided Mann-Whitney T>M p<0.05','pass':bool(mw['p_one_sided_greater']<.05) if mw and not under else None}
    C3={'n_pairs':r['n_T'],'statistic':w['statistic'] if w else None,'p_one_sided_greater':w['p_one_sided_greater'] if w else None,'requirement':'one-sided paired Wilcoxon AJ>AC p<0.05','pass':bool(w['p_one_sided_greater']<.05) if w and not under else None}
    gates={'primary_screen':'rpe1','primary_pca_k':20,'primary_valley_K':5,'n_strong_T':r['n_T'],'n_strong_M':r['n_M'],'underpowered':under,
           'gate_C1':C1,'gate_C2':C2,'gate_C3':C3,'confirmed':not under and all(g['pass'] is True for g in (C1,C2,C3)),
           'conclusion':'underpowered: confirmation stopped' if under else ('confirmed: all fixed primary gates passed' if all(g['pass'] is True for g in (C1,C2,C3)) else 'not confirmed: one or more fixed primary gates failed or undefined'),
           'report_only_results':summaries,'valley_freeze_receipt_sha256':digest(OUT/'controls_frozen.json'),'task_sha256':digest(TASK),'finished_utc':now()}
    dump(dest,gates);make_figures();write_report('partial');print(json.dumps({x:gates[x] for x in ('underpowered','n_strong_T','n_strong_M','gate_C1','gate_C2','gate_C3','conclusion')},indent=2),flush=True);return 0


def make_figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    FIG.mkdir(parents=True,exist_ok=True);frame=pd.read_csv(OUT/'alignment.csv',float_precision='round_trip');g=json.loads((OUT/'gates.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),constrained_layout=True)
    for ax,screen in zip(axes,('rpe1','k562')):
        r=frame[(frame.screen==screen)&(frame.pca_k==20)&(frame.valley_K==5)&frame.strong];T=r[r.group_T];M=r[r.group_M]
        with np.load(valley_path(screen,20,5),allow_pickle=False) as a: null=a['null_A_J'];p95=float(a['null_p95'])
        ax.hist(null,bins=40,density=True,color='#dce3e7',label='Covariance-matched null');ax.axvline(p95,color='#555555',ls='--',label='Null 95th percentile')
        for values,color,label,y in [(T.A_J,'#326778','Strong T',.06),(M.A_J,'#ac653d','Strong M',.13)]:
            if len(values): ax.scatter(values,np.full(len(values),y),transform=ax.get_xaxis_transform(),color=color,s=20,alpha=.65,label=f'{label}: n={len(values)}, median={values.median():.3f}')
        title=('underpowered: 4 strong T < 8 required' if g['underpowered'] else 'primary confirmation') if screen=='rpe1' else 'discovery replicate, report only'
        ax.set(xlabel='Maximum absolute cosine to 5 saved valleys',ylabel='Null density',xlim=(0,1),title=f'{screen.upper()}: '+title);ax.legend(fontsize=8)
    fig.savefig(FIG/'alignment_vs_null.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.8),constrained_layout=True)
    for ax,screen in zip(axes,('rpe1','k562')):
        r=frame[(frame.screen==screen)&(frame.pca_k==20)&(frame.valley_K==5)&frame.strong];T=r[r.group_T];M=r[r.group_M]
        ax.scatter(M.A_C,M.A_J,s=28,alpha=.7,color='#ac653d',label=f'Strong M (n={len(M)})');ax.scatter(T.A_C,T.A_J,s=34,alpha=.8,color='#326778',label=f'Strong T (n={len(T)})');ax.plot([0,1],[0,1],'--',color='gray')
        suffix=(' underpowered; descriptive only' if g['underpowered'] else ' primary') if screen=='rpe1' else ' report only'
        ax.set(xlabel='A_C: top-5 covariance axes',ylabel='A_J: 5 control-only valleys',title=screen.upper()+suffix,xlim=(0,1),ylim=(0,1));ax.legend(fontsize=8)
    fig.savefig(FIG/'score_vs_covariance.png',dpi=180);plt.close(fig)


def final_verify():
    import fnmatch
    import re
    import numpy as np
    import pandas as pd
    from scipy.stats import false_discovery_control
    barrier();pre=check_task();assert digest(K562_DATA)==K562_SHA and digest(RPE1_DATA)==RPE1_SHA,'STOP: final original checksum mismatch'
    gates=json.loads((OUT/'gates.json').read_text());frame=pd.read_csv(OUT/'alignment.csv',float_precision='round_trip')
    for s in ('rpe1','k562'):
        f=pd.read_csv(OUT/f'targets_{s}.csv',float_precision='round_trip');np.testing.assert_allclose(false_discovery_control(f.p_value,method='bh'),f.p_adjusted_BH,rtol=0,atol=1e-11)
        sig=f[f.significant].sort_values(['effect_size_energy_distance','target'],ascending=[False,True],kind='stable');expected=set(sig.head(math.ceil(.2*len(sig))).target);assert set(f.loc[f.strong,'target'])==expected
        with np.load(CACHE/f'shifts_{s}.npz',allow_pickle=False) as a: shifts=a['shifts'];names=a['targets']
        for k,K in VARIANTS:
            group=frame[(frame.screen==s)&(frame.pca_k==k)&(frame.valley_K==K)].set_index('target').loc[names]
            with np.load(valley_path(s,k,K),allow_pickle=False) as a:
                v=a['valleys'];e=a['covariance_directions'];null=a['null_A_J'];p95=float(a['null_p95']);assert np.percentile(null,95)==p95;np.testing.assert_allclose(np.linalg.norm(v,axis=1),1,rtol=0,atol=1e-12)
            u=shifts[:,:k]/np.linalg.norm(shifts[:,:k],axis=1)[:,None];np.testing.assert_allclose(np.max(np.abs(u@v.T),axis=1),group.A_J,rtol=0,atol=1e-13);np.testing.assert_allclose(np.max(np.abs(u@e.T),axis=1),group.A_C,rtol=0,atol=1e-13)
        assert json.loads((OUT/f'projection_{s}.json').read_text())['first_perturbed_coordinate_access_utc']>json.loads((OUT/'controls_frozen.json').read_text())['saved_utc']
    r=frame[(frame.screen=='rpe1')&(frame.pca_k==20)&(frame.valley_K==5)&frame.strong&frame.significant]
    comp=comparisons(r,allow_tests=not gates['underpowered']);assert comp==gates['report_only_results']['rpe1']['pca20_K5']['strong']
    for name in ('tests_before','tests_after'): assert json.loads((LOGS/f'{name}.checkpoint.json').read_text())['exit_code']==0
    contract=json.loads(re.findall(r'```json\n(.*?)\n```',TASK.read_text(),re.S)[-1]);changed=subprocess.check_output(['git','diff','--name-only','orientation...HEAD'],cwd=ROOT,text=True).splitlines()
    outside=[p for p in changed if p!='coord/orient/tasks/S18c.md' and not any(fnmatch.fnmatchcase(p,pat) for pat in contract['allowed_paths'])];assert not outside
    assert cpu_hours()<BUDGET
    dump(OUT/'final_verification.json',{'status':'pass','k562_sha256':K562_SHA,'rpe1_sha256':RPE1_SHA,'saved_predictor_precedes_all_perturbed_access':True,'alignment_reproduced_from_saved_shifts_and_directions':True,'strong_selection_reproduced':True,'primary_statistics_reproduced':True,'complete_scope_exception':'User-authorized planner task amendment only','outside_allowed_paths':outside,'checked_utc':now()})
    print('Final verification PASS: original hashes, freeze barrier, alignment, strong selection, primary statistics and authorized scope',flush=True);return 0


def regeneration_command(path):
    import re
    rel=str(path.relative_to(ROOT));restore=f"unzip -n ../S18c_benchmarks.zip '{rel}'"
    command=None
    if path.parent==CACHE:
        name=path.name
        m=re.fullmatch(r'controls_(rpe1|k562)\.npz',name)
        if m: command=f'controls --screen {m[1]}'
        m=re.fullmatch(r'(?:H|model)_(rpe1|k562)_k(15|20|30)\.(?:npz|pt)',name)
        if m: command=f'score --screen {m[1]} --k {m[2]}'
        m=re.fullmatch(r'projected_(rpe1|k562)\.npz',name)
        if m: command=f'project --screen {m[1]}'
        m=re.fullmatch(r'shifts_(rpe1|k562)\.npz',name)
        if m: command=f'align --screen {m[1]}'
    return restore if command is None else restore+'; fresh-output-tree regeneration after the recorded prerequisites: .venv/bin/python scripts/s18c_valleys.py '+command+'. Completed-output guards forbid an automatic repeat.'


def write_report(status='partial',question=None):
    pre=check_task();env=json.loads((OUT/'environment.json').read_text());units=[json.loads(p.read_text()) for p in sorted(LOGS.glob('*.checkpoint.json'))]
    gates=json.loads((OUT/'gates.json').read_text()) if (OUT/'gates.json').exists() else None
    sourcefile=json.loads((OUT/'rpe1_source.json').read_text());groups={s:json.loads((OUT/f'group_lists_{s}.json').read_text()) for s in ('rpe1','k562') if (OUT/f'group_lists_{s}.json').exists()}
    metrics={'rpe1_file':sourcefile,'group_lists':groups,'n_strong_T':gates['n_strong_T'] if gates else None,'n_strong_M':gates['n_strong_M'] if gates else None,
             'gate_C1':gates['gate_C1'] if gates else None,'gate_C2':gates['gate_C2'] if gates else None,'gate_C3':gates['gate_C3'] if gates else None,
             'k562_report_only':gates['report_only_results']['k562'] if gates else None,'versions':env['versions'],'cpu_hours':cpu_hours()}
    deviations=['User-authorized Amendment Revision1 appended verbatim and committed before any RPE1 scientific result; only this task-file change is outside scientific allowed_paths.',
                'Cloud-specific execution supersedes Mac folders, launchers and fixed resource split; existing .venv reused as authorized.',
                'Exact full control-only PCA-30 provides nested 15/20/30 axes; no extra scaling or whitening. K562 control-fitted PCA parameters and H caches from accepted S18 reused without refitting; no perturbed coordinates loaded from those caches until saved predictor barrier.',
                'S18 energy test uses 2000 random controls, as specified by the referenced recipe; every centroid displacement uses ALL controls, not that test-reference subset.',
                'All eligible targets are retained in per-target tables; report-only all-target comparisons use every significant analyzed target without the strong restriction.',
                'The supplied protected tests have no xfail annotations; all 48 pass unchanged.']
    if not gates: deviations.append('Scientific metrics not yet computed; null entries are partial/blocked checkpoints, not imputed results.')
    if gates and gates['underpowered']: deviations.append('Fewer than 8 strong RPE1 T targets: confirmation stopped under the fixed rule; inferential confirmation statistics and gate pass fields are null, no threshold or group changed.')
    if question: deviations.append(question)
    if (LOGS/'delivery_finalize_initial.checkpoint.json').exists(): deviations.append('Initial delivery-only ZIP check failed after reusing ZipInfo objects in the open writer. The closed/reopened archive passed CRC and report-byte checks; final packaging now uses fresh entries and validates after closing. The failed checkpoint is retained. No scientific fit, selection, statistic, predictor or original input changed or was rerun.')
    observations=['Original RPE1 size, supplied MD5 and planner-recorded SHA-256 match; original K562 SHA-256 unchanged.','Group symbols were resolved from raw obs metadata before expression/alignment; every learned predictor used controls only.']
    for s in ('rpe1','k562'):
        c=OUT/f'controls_{s}.json';f=OUT/f'effects_{s}.json'
        if c.exists():
            d=json.loads(c.read_text());observations.append(f"{s.upper()}: {d['n_controls']} controls of {d['n_cells']} cells x {d['n_genes']} genes; median control library {d['median_control_library']:.0f}.")
        if f.exists():
            d=json.loads(f.read_text());observations.append(f"{s.upper()}: {d['n_targets_significant']}/{d['n_targets_eligible']} eligible targets significant; {d['n_strong']} strong, including T={d['n_strong_T']}, M={d['n_strong_M']}.")
        for k in KS:
            p=OUT/f'valleys_{s}_k{k}.json'
            if p.exists():
                d=json.loads(p.read_text());observations.append(f"{s.upper()} PCA-{k}: {d['n_edge']} edge controls, {d['n_soft_edge']} soft edges; no-positive-eigenvalue fallback {d['n_edges_no_positive_eigenvalues']}/{d['n_edge']} edges ({d['n_soft_edges_no_positive_eigenvalues']} selected soft edges).")
    if gates:
        observations.append('RPE1 '+gates['conclusion']+'. K562 and all dimension/K/population variants are report-only and cannot rescue or confirm the primary outcome.')
        effect=json.loads((OUT/'effects_rpe1.json').read_text());observations.append('The four strong RPE1 T symbols are '+', '.join(effect['strong_T'])+'. No gene was added to obtain the minimum group size.')
        comp=gates['report_only_results']['k562']['pca20_K5']['strong']
        observations.append(f"K562 primary report-only median A_J(T)={comp['median_A_J_T']:.6f}, A_J(M)={comp['median_A_J_M']:.6f}, null95={comp['null_A_J_p95']:.6f}. M has the higher median, opposite to the fixed T>M direction. Paired A_J>A_C among T has p={comp['paired_wilcoxon']['p_one_sided_greater']:.8g}; this does not make the three-part observation confirmed.")
    validation={n:json.loads((LOGS/f'{n}.checkpoint.json').read_text()) for n in ('tests_before','tests_after','verify_final','coord_check_reviews','coord_check_packaging_repair') if (LOGS/f'{n}.checkpoint.json').exists()}
    artifacts=[]
    for p in [ROOT/'scripts/s18c_valleys.py',*sorted(OUT.rglob('*')),*sorted(FIG.rglob('*'))]:
        if not p.is_file() or p.suffix=='.h5ad' or p.name in ('RESULT_S18c.md',) or p.name.endswith('.tmp'): continue
        rel=str(p.relative_to(ROOT));committed=(subprocess.run(['git','cat-file','-e',f'HEAD:{rel}'],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0 and subprocess.run(['git','diff','--quiet','HEAD','--',rel],cwd=ROOT).returncode==0)
        artifacts.append({'path':rel,'bytes':p.stat().st_size,'sha256':digest(p),'committed':committed,'regenerate':regeneration_command(p)})
    artifacts.append({'path':str(RPE1_DATA.relative_to(ROOT)),'bytes':sourcefile['actual_bytes'],'sha256':sourcefile['actual_sha256'],'committed':False,'regenerate':f'Download {RPE1_URL}; verify size {RPE1_SIZE}, MD5 {RPE1_MD5}, SHA-256 {RPE1_SHA}'})
    report={'task':'orient/S18c','status':status,'started_utc':min(u['started_utc'] for u in units),'finished_utc':now(),'compute':{'cpu_hours':cpu_hours(),'gpu_hours':0,'hardware':env,'units':units},'metrics':metrics,
            'validation':validation,'artifacts':artifacts,'deviations':deviations,'questions':[question] if question else [],'observations':observations,'suggestions':['A separately approved independent confirmation would need adequate representation of the fixed T group (at least 8 strong targets under the unchanged selection). Do not relax this screen\'s cutoff or expand the group post hoc to rescue the underpowered result. No further analysis was run.'],
            'planner_amendment_commit':pre['planner_amendment_commit'],'frozen_task_sha256':digest(TASK),'confirmation':gates['conclusion'] if gates else None}
    dump(ROOT/'coord/orient/reports/S18c.json',report)
    lines=['# orient/S18c — pre-registered soft-valley confirmation','','## Summary','',f'Status: **{status}**. Cloud-only; controls-only fits, every displacement against ALL controls; no Mac, merge or push.',f"Planner amendment: `{pre['planner_amendment_commit']}`; fixed before any RPE1 result.",f'Completed recorded-unit CPU hours: {cpu_hours():.6f} / 14.']
    if gates:
        lines.append(f"**{gates['conclusion']}**; strong RPE1 T={gates['n_strong_T']}, M={gates['n_strong_M']}.")
        for name in ('gate_C1','gate_C2','gate_C3'): lines.append(f"{name}: `{json.dumps(gates[name])}`.")
    lines+=['','## What ran','']+[f"- `{' '.join(u['command'])}` — exit {u['exit_code']}, wall {u['wall_seconds']:.3f}s, CPU {u['cpu_hours']:.6f}h." for u in sorted(units,key=lambda x:x['started_utc'])]
    lines+=['',f"Environment reused: Python {env['python']}; {env['cpu_count']} CPUs; pip freeze committed. Per-unit RAM, threads and peak RSS are recorded in JSON checkpoints.",'',f'RPE1 raw: {RPE1_URL}; {RPE1_SIZE} bytes; SHA-256 `{RPE1_SHA}`. K562 SHA-256 `{K562_SHA}`. Originals are read-only.','','## Gate numbers','','| Gate | Fixed requirement | Observed/result |','|---|---|---|']
    if gates:
        for name in ('gate_C1','gate_C2','gate_C3'): lines.append(f"| {name} | {gates[name]['requirement']} | `{json.dumps(gates[name])}` |")
        lines+=['','### K562 and sensitivity comparisons — report only','','| Screen | PCA k | Valley K | Strong T/M | Median AJ(T) | Null95 | MW p T>M | Paired p AJ>AC |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for screen in ('rpe1','k562'):
            for k,K in VARIANTS:
                r=gates['report_only_results'][screen][f'pca{k}_K{K}']['strong'];mw=r['mann_whitney'];w=r['paired_wilcoxon']
                fmt=lambda a:'null' if a is None else f'{a:.6g}'
                lines.append(f"| {screen.upper()} | {k} | {K} | {r['n_T']}/{r['n_M']} | {fmt(r['median_A_J_T'])} | {fmt(r['null_A_J_p95'])} | {fmt(mw['p_one_sided_greater'] if mw else None)} | {fmt(w['p_one_sided_greater'] if w else None)} |")
        lines+=['','All-significant (not only strong) group statistics, full symbols and every eligible per-target row are provided in `gates.json`, `group_lists_*.json`, `targets_*.csv` and `alignment.csv`. These do not alter any primary gate.']
    else: lines.append('| C1–C3 | Frozen task definitions | null: not computed yet |')
    lines+=['','## Deviations','']+[f'- {x}' for x in deviations]
    for section,arr in [('Observations',observations),('Suggestions',report['suggestions']),('Questions',report['questions'] or ['None.'])]: lines+=['',f'## {section}','']+[f'- {x}' for x in arr]
    lines+=['','## Artifacts','','The benchmark ZIP includes all control-only valley files, SHA receipts, final tables/statistics, fitted weights/H/PCA and projection/shift caches, figures, reports, and timed checkpoints. Raw H5ADs are excluded and must be downloaded/retained with the stated hashes. Large derived caches and model weights remain uncommitted; committed primary valley files remain small. The Git bundle contains complete task-branch history and the verbatim planner amendment.','','`controls_frozen.json` records every saved predictor hash before the first permitted perturbed-coordinate access; `projection_*.json` records that access time. Group lists are resolved and hashed before alignment. For restoration clone the returned bundle on `manus/orient-S18c`, unzip the benchmark archive at its root, and retain the verified K562 input at its recorded path. Use the existing or fresh project venv with `pip install -e \".[dev]\"`; do not repeat successful fits to change a result.','','The recorded sequence is input integrity/tests, original RPE1 download, each screen control embedding then per-k score/valleys, `freeze`, per-screen projection/effects/alignment, `summarize`, required post-tests, `verify-final`, final report/checker and delivery. Every unit uses `.venv/bin/python scripts/s18c_valleys.py unit <name> -- <command>`.','','[Original Figshare deposit](https://plus.figshare.com/articles/dataset/_Mapping_information-rich_genotype-phenotype_landscapes_with_genome-scale_Perturb-seq_Replogle_et_al_2022_processed_Perturb-seq_datasets/20029387); [file metadata](https://api.figshare.com/v2/articles/20029387).','']
    text='\n'.join(lines);(ROOT/'coord/orient/reports/S18c.md').write_text(text);(OUT/'RESULT_S18c.md').write_text(text);return 0


def delivery_prepare():
    import zipfile
    check_task();assert json.loads((ROOT/'coord/orient/reports/S18c.json').read_text())['status']=='done'
    path=ROOT.parent/'S18c_benchmarks.prepared.zip'
    if path.exists(): raise RuntimeError('Prepared archive exists: inspect instead of repeating')
    files=[p for folder in (OUT,FIG) for p in sorted(folder.rglob('*')) if p.is_file() and p.suffix!='.h5ad' and not p.name.endswith('.tmp')]
    files += [ROOT/'scripts/s18c_valleys.py',ROOT/'coord/orient/reports/S18c.md',ROOT/'coord/orient/reports/S18c.json']
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for i,p in enumerate(files,1):
            z.write(p,str(p.relative_to(ROOT)))
            if i%25==0: print(f'[{now()}] archive prepared {i}/{len(files)} files',flush=True)
        assert z.testzip() is None
    dump(OUT/'delivery_prepare.json',{'archive_path':str(path),'n_files':len(files),'bytes_before_final_metadata':path.stat().st_size,'sha256_before_final_metadata':digest(path),'crc_check':'pass','raw_inputs_excluded':True,'prepared_utc':now()})
    return 0


def delivery_finalize():
    import zipfile
    import tempfile
    check_task();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),'STOP: task branch not clean'
    prepared=ROOT.parent/'S18c_benchmarks.prepared.zip';path=ROOT.parent/'S18c_benchmarks.zip';bundle=ROOT.parent/'S18c_result.bundle'
    if path.exists() or bundle.exists(): raise RuntimeError('Final delivery exists: no overwrite')
    files=[p for folder in (OUT,FIG) for p in sorted(folder.rglob('*')) if p.is_file() and p.suffix!='.h5ad' and not p.name.endswith('.tmp')]
    files += [ROOT/'scripts/s18c_valleys.py',ROOT/'coord/orient/reports/S18c.md',ROOT/'coord/orient/reports/S18c.json']
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as new:
        for p in files:
            new.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(path,'r') as verify:
        assert verify.testzip() is None
    subprocess.run(['git','bundle','create',str(bundle),'manus/orient-S18c'],cwd=ROOT,check=True)
    subprocess.run(['git','bundle','verify',str(bundle)],cwd=ROOT,check=True)
    with tempfile.TemporaryDirectory(prefix='S18c-bundle-check-') as d:
        subprocess.run(['git','clone','--bare',str(bundle),d+'/repo.git'],check=True)
        subprocess.run(['git','--git-dir',d+'/repo.git','fsck','--full'],check=True)
        recovered=subprocess.check_output(['git','--git-dir',d+'/repo.git','show','manus/orient-S18c:coord/orient/reports/S18c.json'])
        assert recovered==(ROOT/'coord/orient/reports/S18c.json').read_bytes()
    delivered=[]
    for p in (bundle,path,ROOT/'coord/orient/reports/S18c.md',ROOT/'coord/orient/reports/S18c.json'):
        delivered.append({'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)});print(str(p),p.stat().st_size,'bytes',delivered[-1]['sha256'],flush=True)
    dump(ROOT.parent/'S18c_delivery_manifest.json',{'files':delivered,'bundle_verify':'pass','offline_clone_fsck':'pass','archive_crc':'pass','raw_inputs_excluded':True,'final_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'finished_utc':now()})
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('unit');p.add_argument('name');p.add_argument('command',nargs=argparse.REMAINDER)
    for name in ('preflight','download-rpe1','freeze','summarize','figures','verify-final','delivery-prepare','delivery-finalize'): sub.add_parser(name)
    for name in ('controls','project','effects','align'):
        p=sub.add_parser(name);p.add_argument('--screen',choices=('rpe1','k562'),required=True)
    for name in ('score','valleys'):
        p=sub.add_parser(name);p.add_argument('--screen',choices=('rpe1','k562'),required=True);p.add_argument('--k',type=int,choices=KS,required=True)
    p=sub.add_parser('report');p.add_argument('--status',choices=('partial','blocked','done'),default='partial');p.add_argument('--question')
    a=parser.parse_args()
    if a.action=='unit':
        command=a.command[1:] if a.command and a.command[0]=='--' else a.command
        if not command: parser.error('unit requires command')
        return run_unit(a.name,command)
    if a.action=='report': return write_report(a.status,a.question)
    if a.action=='figures':
        barrier();make_figures();return 0
    if a.action in ('preflight','download-rpe1','freeze','summarize','verify-final','delivery-prepare','delivery-finalize'): return {'preflight':preflight,'download-rpe1':download_rpe1,'freeze':freeze_all,'summarize':summarize,'verify-final':final_verify,'delivery-prepare':delivery_prepare,'delivery-finalize':delivery_finalize}[a.action]()
    if a.action in ('controls','project','effects','align'): return {'controls':prepare_controls,'project':project_cells,'effects':effect_tests,'align':align_targets}[a.action](a.screen)
    return {'score':score_controls,'valleys':build_valleys}[a.action](a.screen,a.k)


if __name__=='__main__':
    sys.exit(main())
