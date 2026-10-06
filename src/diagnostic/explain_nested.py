"""Permutation Shapley explanations on the probability scale."""
from pathlib import Path
import argparse,json,sys
import numpy as np,pandas as pd,joblib
from nested_analysis import preparation,fit_predict,fit_subset,SEED

def probability(bundle,x):
    if bundle['family']=='TabPFN':z=x[:,bundle['selected']]
    else:z=bundle['scaler'].transform(bundle['imputer'].transform(np.log1p(x[:,bundle['eligible']])))[:,bundle['selected_local']]
    return np.concatenate([bundle['model'].predict_proba(z[i:i+512])[:,1] for i in range(0,len(z),512)])

def explain(bundle,x,background,seed,permutations=16):
    k=len(bundle['selected']);rng=np.random.default_rng(seed)
    orders=[]
    for _ in range(permutations//2):
        order=rng.permutation(k);orders.extend([order,order[::-1]])
    base=np.tile(background,(len(x),1));baseprob=probability(bundle,base)
    phi=np.zeros((len(x),k));sq=np.zeros_like(phi)
    for t,order in enumerate(orders):
        z=base.copy();states=[]
        for j in order:
            col=bundle['selected'][j];z[:,col]=x[:,col];states.append(z.copy())
        pp=probability(bundle,np.vstack(states)).reshape(k,len(x))
        delta=np.diff(np.vstack([baseprob,pp]),axis=0).T
        phi[:,order]+=delta;sq[:,order]+=delta**2
        print('Shapley permutation',t+1,'/',len(orders),'patients',len(x),flush=True)
    phi/=len(orders)
    score=probability(bundle,x)
    residual=float(np.max(abs(phi.sum(axis=1)+baseprob-score)))
    print('Maximum probability additivity residual',residual,flush=True)
    # CPU inference in different batch shapes can differ at float32 precision.
    # Check probability additivity without altering the estimated contributions.
    assert residual<1e-4
    mcse=np.sqrt(np.maximum(0,sq/len(orders)-phi**2)/len(orders))
    return phi,baseprob,score,mcse

def run(a):
    meta=json.loads(Path(a.selection).read_text());family=meta['winner'];folder=Path(meta['paths'][family])
    if not folder.is_absolute():folder=(Path(a.selection).resolve().parent/folder).resolve()
    d=pd.read_csv(a.input,low_memory=False)
    if a.task=='TreatmentResponse':d['partition']=np.where(d.centre.eq('Sanjiu'),'Development','External')
    else:d=d[d['分组'].isin(['阳性组','癌症非脑膜转移对照组'])]
    dev=d[d.partition.eq('Development')].reset_index(drop=True)
    final=joblib.load(folder/'block5.joblib');names=final['features'];x=dev[names].to_numpy(float);y=dev.y.to_numpy(int)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    for i in range(6):
        if a.block is not None and i!=a.block:continue
        file=out/f'SHAP_block{i}.csv'
        if file.exists():continue
        block=joblib.load(folder/f'block{i}.joblib');tr=block['training'];va=block['validation'];seed=SEED+1000*(i+1)
        if i<5:target=dev.iloc[va].copy();target['split']='Internal OOF'
        else:
            target=d[~d.partition.eq('Development')].copy();target['split']=target.partition.replace({'Nanfang-sheet validation':'External'})
        v=target[names].to_numpy(float)
        p,bundle=fit_subset(x[tr],y[tr],v,family,block['parameters'],block['selected'],seed,a.checkpoint) if a.task=='TreatmentResponse' else fit_predict(x[tr],y[tr],v,family,block['parameters'],len(block['selected']),seed,a.checkpoint,False)
        assert np.array_equal(bundle['selected'],block['selected'])
        assert np.max(abs(p-block['p']))<1e-5
        phi,base,score,se=explain(bundle,v,np.nanmedian(x[tr],axis=0),seed)
        rows=[]
        for n,(_,r) in enumerate(target.iterrows()):
            for j,c in enumerate(bundle['selected']):
                rows.append(dict(patient_id=r['患者ID'],split=r['split'],y=int(r.y),score=score[n],reference_score=base[n],feature=names[c],value=v[n,c],shap=phi[n,j],mc_se=se[n,j],block=i))
        pd.DataFrame(rows).to_csv(file,index=False)
        print('SHAP block',i+1,'complete',flush=True)
    if all((out/f'SHAP_block{i}.csv').exists() for i in range(6)):
        pd.concat([pd.read_csv(out/f'SHAP_block{i}.csv') for i in range(6)],ignore_index=True).to_csv(out/'diagnostic_shap.csv',index=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--selection',required=True);p.add_argument('--output',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--task',default='P_vs_D');p.add_argument('--block',type=int);run(p.parse_args())
