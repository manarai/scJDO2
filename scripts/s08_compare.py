#!/usr/bin/env python3
"""orient/S08: compare saved common-statistic records and render deterministic panels."""
from __future__ import annotations
import itertools, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'benchmarks/real_reference'; FIG=ROOT/'benchmarks/figures/real_reference'
REAL={'scnt_hema','larry','setty','scnt_neuron','pancreas','dentate'}
SYN_PREFIXES=('v1_','v2_')
FIXED={'ks':0.2,'log_ratio':float(np.log(1.25)),'tv':0.15}
CLASS={'S3_phase_portrait_class','S3b_scvelo_switch_class','C8_leiden_share'}
EXCLUDE={'integrity_raw_integer','integrity_shape','CT_not_applicable','S_not_applicable','L_not_applicable','G_not_applicable','T2_not_applicable'}
def label(kind): return kind.split('_',1)[0] if '_' in kind else kind
def distfile(dataset,stat): return OUT/'distributions'/f'{dataset}__{stat}.csv'
def metric_type(stat, datasets):
    if stat in CLASS: return 'tv'
    return 'ks' if all(distfile(d,stat).is_file() for d in datasets) else 'log_ratio'
def value(dataset, stat, full, mtype):
    if mtype=='ks':
        x=pd.read_csv(distfile(dataset,stat)).value.to_numpy(float); return x[np.isfinite(x)]
    rows=full[(full.dataset==dataset)&(full.statistic==stat)].copy()
    if mtype=='tv':
        x=rows.groupby('feature',sort=True).value.mean(); return x/x.sum() if x.sum() else x
    return rows.groupby('feature',sort=True).value.mean()
def scalar_or_median(a, mtype):
    if mtype == 'tv': return float('nan')
    x = a if mtype == 'ks' else a.to_numpy(float)
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float('nan')
def distance(a,b,mtype):
    if mtype=='ks': return float(ks_2samp(a,b).statistic)
    if mtype=='tv':
        both=a.index.union(b.index); aa=a.reindex(both,fill_value=0).to_numpy(float); bb=b.reindex(both,fill_value=0).to_numpy(float)
        return float(0.5*np.abs(aa-bb).sum())
    both=a.index.intersection(b.index)
    if len(both)==0: return float('nan')
    aa=a.loc[both].to_numpy(float); bb=b.loc[both].to_numpy(float)
    ok=np.isfinite(aa)&np.isfinite(bb)&(aa>0)&(bb>0)
    return float(np.median(np.abs(np.log(aa[ok]/bb[ok])))) if ok.any() else float('nan')
def render(compare):
    FIG.mkdir(parents=True,exist_ok=True)
    for stat,g in compare.groupby('statistic',sort=True):
        fig,ax=plt.subplots(figsize=(7.4,3.4),layout='constrained')
        g=g.sort_values(['preset','synthetic'])
        color={'v1':'#6b7280','v2':'#2563eb'}.get(label(g.preset.iloc[0]),'#2563eb')
        ax.axhspan(0,float(g.threshold.iloc[0]),color='#dcfce7',alpha=.75,label='match threshold')
        ax.axhline(float(g.threshold.iloc[0]),color='#15803d',lw=1.1)
        ax.scatter(np.arange(len(g)),g.distance,color=color,s=34,zorder=3)
        ax.set_xticks(np.arange(len(g)),g.synthetic,rotation=45,ha='right',fontsize=7)
        ax.set_ylabel({'ks':'KS distance','tv':'TV distance','log_ratio':'|log ratio|'}[g.metric.iloc[0]])
        ax.set_title(stat+' — nearest real reference distance',fontsize=10)
        ax.grid(axis='y',alpha=.25)
        ax.text(.99,.96,f"real spread = {g.real_spread.iloc[0]:.3g}",ha='right',va='top',transform=ax.transAxes,fontsize=8)
        fig.savefig(FIG/f'{stat}.png',dpi=180); plt.close(fig)
def main():
    full=pd.read_csv(OUT/'stats.csv')
    full=full[np.isfinite(full.value)]
    stats=sorted(s for s in full.statistic.unique() if s not in EXCLUDE)
    rows=[]
    for stat in stats:
        avail=sorted(full.loc[full.statistic==stat,'dataset'].unique())
        real=[d for d in avail if d in REAL]
        syn=[d for d in avail if d.startswith(SYN_PREFIXES)]
        if not real or not syn: continue
        mtype=metric_type(stat, real+syn)
        vals={d:value(d,stat,full,mtype) for d in real+syn}
        pairs=[distance(vals[a],vals[b],mtype) for a,b in itertools.combinations(real,2)]
        pairs=[x for x in pairs if np.isfinite(x)]
        spread=max(pairs) if pairs else 0.0
        fixed=FIXED[mtype]; threshold=max(spread,fixed)
        real_scalars = np.asarray([scalar_or_median(vals[r], mtype) for r in real], dtype=float)
        real_scalars = real_scalars[np.isfinite(real_scalars)]
        rlo = float(real_scalars.min()) if real_scalars.size else float('nan')
        rhi = float(real_scalars.max()) if real_scalars.size else float('nan')
        for s in syn:
            ds=[(r,distance(vals[s],vals[r],mtype)) for r in real]; ds=[x for x in ds if np.isfinite(x[1])]
            if not ds: continue
            nearest,d=min(ds,key=lambda x:x[1])
            synthetic_value = scalar_or_median(vals[s], mtype)
            within = bool(rlo <= synthetic_value <= rhi) if np.isfinite(synthetic_value) and np.isfinite(rlo) and np.isfinite(rhi) else None
            preset='v1' if s.startswith('v1_') else '_'.join(s.split('_')[:2])
            # Frozen rule: the statistic matches only when it is within threshold of EVERY real dataset.
            match_all = bool(len(ds) == len(real) and all(x[1] <= threshold for x in ds))
            distances = {r: float(x) for r, x in ds}
            rows.append({'synthetic':s,'preset':preset,'statistic':stat,'metric':mtype,'distance':d,'nearest_real':nearest,'real_spread':spread,'fixed_tolerance':fixed,'threshold':threshold,'match':match_all,'matches_every_real':match_all,'n_real':len(real),'n_real_compared':len(ds),'synthetic_scalar_or_median':synthetic_value,'real_values_or_medians':json.dumps({r: scalar_or_median(vals[r], mtype) for r in real}, sort_keys=True),'real_range_min':rlo,'real_range_max':rhi,'within_real_range':within,'max_distance_to_any_real':max(distances.values()),'distances_to_reals':json.dumps(distances,sort_keys=True)})
    comp=pd.DataFrame(rows).sort_values(['statistic','synthetic']).reset_index(drop=True)
    comp.to_csv(OUT/'compare.csv',index=False); render(comp)
    summary=comp.groupby('preset').agg(statistics=('statistic','size'),match_rate=('match','mean'),matched=('match','sum')).reset_index()
    lines=['# S08 real-data reference comparison','', 'This record is descriptive: common preprocessing/statistic functions were applied to saved real and synthetic inputs. No generator or orientation model was changed.','', '## What ran','', f'- {len(REAL)} real references and {len(set(comp.synthetic))} synthetic datasets contributed to `{len(comp)}` available synthetic-statistic comparisons.','- Distance: KS for saved distributions, total variation for class-share statistics, and median absolute log ratio for scalar feature vectors.','- A match requires distance ≤ max(real spread, fixed tolerance) for every applicable real reference; `within_real_range` independently records whether the scalar or distribution median falls in the real min–max range.','', '## Match summary','','| preset | statistics | match_rate | matched |\n|---|---:|---:|---:|\n' + '\n'.join(f"| {r.preset} | {int(r.statistics)} | {r.match_rate:.3f} | {int(r.matched)} |" for r in summary.itertuples(index=False)), '', '## Matrix','', 'See `compare.csv`; one deterministic panel per statistic is in `benchmarks/figures/real_reference/`.']
    (OUT/'RESULT_S08.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'comparisons':len(comp),'statistics':int(comp.statistic.nunique()),'match_rate':float(comp.match.mean()),'figures':len(list(FIG.glob('*.png')))},indent=2))
if __name__=='__main__': main()
