"""Paired, label-authorized spatial validation; never reads hidden labels."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys, tempfile, time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT=Path('/app'); OUT=ROOT/'research/results'; OUT.mkdir(exist_ok=True)
PATHS={name:ROOT/'research/parents'/name/'pipeline.py' for name in ['incumbent','s009','s012','s013']}
ALL_TREES=pd.concat([pd.read_csv(ROOT/'data/L_labelled.csv'),pd.read_csv(ROOT/'data/unlabelled.csv')],ignore_index=True)
ORIGINS={e:(v.Long.min(),v.Lat.min(),v.Lat.mean()) for e,v in ALL_TREES.groupby('estate')}

def load(name):
    path=PATHS.get(name,ROOT/'research/candidates'/name/'pipeline.py')
    if name=='coords': return None,'coords15-standardized-distance-exact-union-v2'
    if not path.exists():
        family='parents' if name in PATHS else 'candidates'
        path=ROOT/'datasets/triage_v12/recipes/s013'/family/name/'pipeline.py'
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location('model_'+name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    checksum=hashlib.sha256()
    for p in sorted(path.parent.glob('*.py')):
        checksum.update(p.name.encode());checksum.update(p.read_bytes())
    # Candidate shared implementation changes invalidate its cache.
    if (ROOT/'research/synthesis.py').exists() and name not in PATHS:
        checksum.update((ROOT/'research/synthesis.py').read_bytes())
    if name not in PATHS and (ROOT/'research/engine.py').exists():
        checksum.update((ROOT/'research/engine.py').read_bytes())
    checksum.update(b'exact-union-v2-block-coarsening-v1')
    return module,checksum.hexdigest()

def xy(d):
    lon,lat,meanlat=ORIGINS[str(d.estate.iloc[0])]
    return np.c_[(d.Long.to_numpy()-lon)*111320*np.cos(np.deg2rad(meanlat)),
                 (d.Lat.to_numpy()-lat)*110540]

def blocks(d,size=80,shift=0):
    b=np.floor(xy(d)/80).astype(np.int64)
    if size!=80:
        # Aggregate original 80m blocks using their centres; never split a source block.
        b=np.floor(((b+.5)*80+shift)/size).astype(np.int64)
    elif shift:
        b=np.floor(((b+.5)*80+shift)/size).astype(np.int64)
    return b[:,0]*100000+b[:,1]

def splits(d,protocol):
    size=120 if protocol=='block120' else 160 if protocol=='block160' else 80
    shift=40 if protocol=='shift40' else 0
    group=blocks(d,size,shift);coords=xy(d)
    if protocol in ['quadrant','regions80','medianquad','median80','blockquad','blockquad80','blockquad160']:
        b=np.c_[group//100000,group%100000]
        if protocol.startswith('median'):
            b=coords;mid=np.median(coords,axis=0)
        elif protocol.startswith('blockquad'):
            mid=np.median(b,axis=0)
        else:
            unique=np.unique(b,axis=0);mid=np.median(unique,axis=0)
        region=(b[:,0]>mid[0]).astype(int)+2*(b[:,1]>mid[1]).astype(int)
        split=[(np.flatnonzero(region!=i),np.flatnonzero(region==i)) for i in np.unique(region)]
    else:
        seed=int(protocol.replace('repeat','')) if protocol.startswith('repeat') else None
        split=GroupKFold(5,shuffle=seed is not None,random_state=seed).split(d,groups=group)
    buffer=160 if protocol=='blockquad160' else 80 if protocol in ['buffer80','regions80','median80','blockquad80'] else 40 if protocol=='buffer40' else 0
    for k,(tr,va) in enumerate(split):
        if buffer:
            tr=tr[cKDTree(coords[va]).query(coords[tr])[0]>=buffer]
        if not protocol.startswith('median'):
            assert len(np.intersect1d(group[tr],group[va]))==0
        yield k,tr,va

def budget(y,s,fraction):
    n=len(s)*fraction
    order=np.argsort(-s,kind='stable');scores=s[order];y=y[order]
    boundary=scores[min(int(np.ceil(n))-1,len(s)-1)]
    above=scores>boundary; tie=scores==boundary
    found=y[above].sum()+y[tie].sum()*max(0,n-above.sum())/tie.sum()
    return found/max(1,y.sum())

def metrics(d,s):
    ok=d.label.isin(['Healthy','Unhealthy']).to_numpy();y=(d.label.to_numpy()[ok]=='Unhealthy').astype(int)
    return {'prevalence':y.mean(),'ap':average_precision_score(y,s[ok]),
            'lift':average_precision_score(y,s[ok])-y.mean(), 'roc':roc_auc_score(y,s[ok]),
            'recall25':budget(y,s[ok],.25),'enrich10':budget(y,s[ok],.10)/.10,
            'inclusive_recall25':budget((d.label=='Unhealthy').to_numpy(int),s,.25)}

def run(name,protocols,force=False,only_estates=None):
    module,sha=load(name);all_l=pd.read_csv(ROOT/'data/L_labelled.csv')
    data={e:d.reset_index(drop=True) for e,d in all_l.groupby('estate',sort=True)}
    rows=[]
    for protocol in protocols:
        for estate,d in data.items():
            if only_estates and estate not in only_estates:continue
            dest=OUT/f'{name}_{protocol}_{estate}.npz'
            if dest.exists() and not force:
                z=np.load(dest)
                if str(z['sha'])==sha:
                    rows.append(dict(model=name,protocol=protocol,estate=estate,**metrics(d,z['s']),seconds=float(z['seconds'])))
                    continue
            start=time.monotonic();s=np.full(len(d),np.nan);folds=np.full(len(d),-1,dtype=int);audit=[]
            with tempfile.TemporaryDirectory() as scratch:
                ctx=dict(estate=estate,datasets_dir=ROOT/'datasets',other_labeled={e:v for e,v in data.items() if e!=estate},scratch_dir=Path(scratch),cpus=1)
                for k,tr,va in splits(d,protocol):
                    a=d.iloc[tr].reset_index(drop=True);b=d.iloc[va].drop(columns='label').reset_index(drop=True)
                    binary=a.label.isin(['Healthy','Unhealthy']);y=(a.loc[binary,'label']=='Unhealthy').to_numpy(int)
                    if len(np.unique(y))<2: pred=np.full(len(b),float(y.mean()) if len(y) else 0.)
                    elif name=='coords':
                        scale=StandardScaler().fit(a.loc[binary,['Long','Lat']]);x=scale.transform(a.loc[binary,['Long','Lat']]);t=scale.transform(b[['Long','Lat']])
                        pred=KNeighborsClassifier(n_neighbors=min(15,len(x)),weights='distance').fit(x,y).predict_proba(t)[:,1]
                    else:
                        try:
                            with threadpool_limits(limits=1): pred=module.fit_predict(a,b,ctx)
                        except Exception as error:
                            (OUT/f'{name}_{protocol}_{estate}_failure.json').write_text(json.dumps({'model':name,'protocol':protocol,'estate':estate,'fold':k,'error':str(error)},indent=2))
                            raise
                    pred=np.asarray(pred);assert pred.shape==(len(va),) and np.isfinite(pred).all()
                    s[va]=pred;folds[va]=k
                    dist=cKDTree(xy(d)[tr]).query(xy(d)[va])[0]
                    audit.append(dict(fold=k,ntrain=len(tr),ntest=len(va),train_positive=int(y.sum()),min_distance=float(dist.min()),median_distance=float(np.median(dist))))
            assert np.isfinite(s).all() and (folds>=0).all()
            seconds=time.monotonic()-start
            np.savez_compressed(dest,s=s,fold=folds,sha=sha,seconds=seconds)
            (OUT/f'{name}_{protocol}_{estate}_splits.json').write_text(json.dumps(audit,indent=2))
            row=dict(model=name,protocol=protocol,estate=estate,**metrics(d,s),seconds=seconds);rows.append(row)
            print(f'{name:24s} {protocol:12s} {estate:9s} {row["lift"]:+.6f} {seconds:.2f}s',flush=True)
    pd.DataFrame(rows).to_csv(OUT/f'{name}_metrics.csv',index=False)
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',required=True);p.add_argument('--protocols',nargs='+',default=['official']);p.add_argument('--force',action='store_true');p.add_argument('--estates',nargs='+');a=p.parse_args()
    for name in a.models:
        try:run(name,a.protocols,a.force,a.estates)
        except Exception as error:
            import traceback
            traceback.print_exc()
            (OUT/f'{name}_failure.json').write_text(json.dumps({'model':name,'protocols':a.protocols,'error':str(error)},indent=2))
