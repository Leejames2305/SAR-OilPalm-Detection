"""Fresh worker checks with I/O guards installed after loading authorized inputs."""
from pathlib import Path
import argparse,importlib.util,json,os,resource,subprocess,sys,tempfile,time
import numpy as np,pandas as pd
ROOT=Path('/app')
def worker(estate):
    L=pd.read_csv(ROOT/'data/L_labelled.csv');U=pd.read_csv(ROOT/'data/unlabelled.csv')
    d=L[L.estate==estate].reset_index(drop=True);t=U[U.estate==estate].reset_index(drop=True)
    spec=importlib.util.spec_from_file_location('submission',ROOT/'submission/pipeline.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    with tempfile.TemporaryDirectory() as scratch:
        ctx=dict(estate=estate,datasets_dir=ROOT/'datasets',other_labeled={e:v.reset_index(drop=True) for e,v in L[L.estate!=estate].groupby('estate')},scratch_dir=Path(scratch),cpus=4)
        os.chdir(scratch)
        def guard(event,args):
            if event in ['socket.connect','socket.getaddrinfo']:raise RuntimeError('Network forbidden')
            if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
                p=Path(os.fsdecode(args[0])).absolute();name=str(p)
                if any(name.startswith('/app/'+v+'/') for v in ['data','rasters','prior']):raise RuntimeError('Source read forbidden: '+name)
                mode=args[1];flags=args[2] if len(args)>2 else 0
                writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
                if writing and not name.startswith(scratch+'/'):raise RuntimeError('Write outside scratch: '+name)
        sys.addaudithook(guard)
        before_d=d.copy(deep=True);before_t=t.copy(deep=True);start=time.monotonic()
        s=m.fit_predict(d,t,ctx);elapsed=time.monotonic()-start
        assert s.dtype==np.float64 and s.shape==(len(t),) and np.isfinite(s).all()
        pd.testing.assert_frame_equal(d,before_d);pd.testing.assert_frame_equal(t,before_t)
        assert np.array_equal(s,m.fit_predict(d,t,ctx))
        order=np.random.default_rng(101).permutation(len(t));shuffled=m.fit_predict(d,t.iloc[order],ctx);np.testing.assert_allclose(shuffled,s[order],rtol=1e-10,atol=1e-10)
        idx=np.r_[np.arange(len(t)),0,5,0];duplicate=m.fit_predict(d,t.iloc[idx],ctx);np.testing.assert_allclose(duplicate,s[idx],rtol=1e-10,atol=1e-10)
        binary=d[d.label.isin(['Healthy','Unhealthy'])];np.testing.assert_allclose(m.fit_predict(binary,t,ctx),s,rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(m.fit_predict(d,t,dict(ctx,estate='UnseenEstate')),s,rtol=1e-10,atol=1e-10)
        assert m.fit_predict(d,t.iloc[:0],ctx).shape==(0,)
        assert np.isfinite(m.fit_predict(d[d.label=='Healthy'],t,ctx)).all()
        result=dict(estate=estate,rows=len(t),seconds=elapsed,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,checks=['shape','float64','finite','immutability','determinism','shuffle','duplicates','Middle_ignored','estate_name_independent','empty','single_class','offline_IO_guard'])
        print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker');a=p.parse_args()
    if a.worker:worker(a.worker)
    else:
        rows=[]
        for estate in ['AirHitam','Palong','Serting']:
            r=subprocess.run([sys.executable,'-B',__file__,'--worker',estate],capture_output=True,text=True,check=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
            rows.append(json.loads(r.stdout.strip().splitlines()[-1]));print(r.stdout.strip(),flush=True)
        (ROOT/'report/contract_verification.json').write_text(json.dumps(rows,indent=2))
