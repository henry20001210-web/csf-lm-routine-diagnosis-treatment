"""Patient-level nested validation of eight CSF prediction algorithms."""
from pathlib import Path
import os, sys, json, argparse, hashlib,atexit
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(key,'2')
os.environ.update(TABPFN_DISABLE_TELEMETRY='1',HF_HUB_OFFLINE='1',TABPFN_ALLOW_CPU_LARGE_DATASET='1')
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score,roc_curve

SEED=20260911
FEATURES=['CSF protein','Glucose','Chloride','Lactate','LDH','ADA','AST','Nucleated cells','Total cells','RBC','LNR','LMR']
FAMILIES=['Logistic','LASSO','Random forest','XGBoost','ExtraTrees','RBF-SVM','ANN/MLP','TabPFN']
PARAMS={'Logistic':[{'C':.1},{'C':1.}], 'LASSO':[{'C':.1},{'C':1.}],
        'Random forest':[{'max_depth':4},{'max_depth':7}], 'ExtraTrees':[{'max_depth':4},{'max_depth':7}],
        'XGBoost':[{'max_depth':2},{'max_depth':3}], 'RBF-SVM':[{'C':.1},{'C':1.}],
        'ANN/MLP':[{'alpha':.01},{'alpha':.1}], 'TabPFN':[{}]}

def estimator(family,param,seed,checkpoint):
    if family in ['Logistic','LASSO']:
        m=LogisticRegression(penalty='l1' if family=='LASSO' else 'l2',solver='liblinear',max_iter=3000,random_state=seed)
    elif family in ['Random forest','ExtraTrees']:
        cls=RandomForestClassifier if family=='Random forest' else ExtraTreesClassifier
        m=cls(n_estimators=150,min_samples_leaf=8,max_features='sqrt',n_jobs=1,random_state=seed)
    elif family=='XGBoost':
        from xgboost import XGBClassifier
        m=XGBClassifier(n_estimators=180,learning_rate=.05,min_child_weight=5,subsample=.85,colsample_bytree=.85,reg_lambda=5,n_jobs=1,tree_method='hist',eval_metric='logloss',random_state=seed)
    elif family=='RBF-SVM':
        m=SVC(kernel='rbf',gamma='scale',probability=True,random_state=seed)
    elif family=='ANN/MLP':
        m=MLPClassifier(hidden_layer_sizes=(32,16),batch_size='auto',max_iter=500,early_stopping=True,validation_fraction=.15,n_iter_no_change=25,random_state=seed)
    else:
        import torch
        from tabpfn import TabPFNClassifier
        torch.set_num_threads(2)
        m=TabPFNClassifier(n_estimators=4,device='cpu',model_path=checkpoint,n_jobs=2,random_state=seed)
    return m.set_params(**param)

def preparation(x,y,seed,treatment_response=False):
    valid=np.ones(x.shape[1],dtype=bool)
    if treatment_response:
        valid=(np.isfinite(x).mean(axis=0)>=.2)&np.array([len(np.unique(c[np.isfinite(c)]))>=3 for c in x.T])
    cols=np.flatnonzero(valid)
    if not len(cols):raise ValueError('No eligible predictors in training fold')
    imp=SimpleImputer(strategy='median',keep_empty_features=True)
    scale=StandardScaler()
    z=scale.fit_transform(imp.fit_transform(np.log1p(x[:,cols])))
    ranker=ExtraTreesClassifier(n_estimators=200,max_depth=7,min_samples_leaf=8,max_features=1.,class_weight='balanced',n_jobs=1,random_state=seed).fit(z,y)
    rank=np.argsort(-ranker.feature_importances_)
    return imp,scale,cols,rank,z

def fit_predict(x,y,v,family,param,k,seed,checkpoint,treatment_response=False,prep=None):
    imp,scale,eligible,rank,z=prep or preparation(x,y,seed,treatment_response)
    ix=rank[:min(k,len(rank))]; cols=eligible[ix]
    model=estimator(family,param,seed,checkpoint)
    if family=='TabPFN':train=x[:,cols];test=v[:,cols]
    else:train=z[:,ix];test=scale.transform(imp.transform(np.log1p(v[:,eligible])))[:,ix]
    model.fit(train,y)
    return model.predict_proba(test)[:,1],dict(model=model,imputer=imp,scaler=scale,eligible=eligible,selected_local=ix,selected=cols,family=family,k=len(cols),parameters=param)

def run(a):
    if a.task=="TreatmentResponse":return treatment_response_floating(a)
    d=pd.read_csv(a.input,low_memory=False)
    if not d['患者ID'].is_unique: raise ValueError('患者ID must be unique')
    if not set(d['y'].dropna().astype(int).unique()).issubset({0,1}): raise ValueError('y must be coded 0/1')
    early=pd.read_csv(a.early,low_memory=False) if a.early else None
    if early is not None:
        if not early['患者ID'].is_unique: raise ValueError('early_data 患者ID must be unique')
        d=d[~d['患者ID'].isin(early['患者ID'])].copy()
    names=FEATURES
    groups=['阳性组','癌症非脑膜转移对照组' if a.task=='P_vs_D' else '阴性组']
    if not set(d['partition'].dropna().unique()).issubset({'Development','Internal test','Nanfang-sheet validation'}): raise ValueError('Unexpected diagnostic partition')
    d=d[d['分组'].isin(groups)].copy()
    assert d['患者ID'].is_unique
    xall=d[names].to_numpy(float)
    if np.isinf(xall).any() or (xall<0).any():raise ValueError('Invalid predictor values')
    dev=d[d.partition.eq('Development')].reset_index(drop=True)
    x=dev[names].to_numpy(float); y=dev.y.to_numpy(int)
    signature=hashlib.sha256(pd.util.hash_pandas_object(d[['患者ID','partition','y']+names],index=False).values.tobytes()+json.dumps(PARAMS,sort_keys=True).encode()+b'nested5-inner3-v1').hexdigest()
    out=Path(a.output)/a.task/a.family.replace('/','_').replace(' ','_')/signature
    out.mkdir(parents=True,exist_ok=True)
    if a.block is not None:
        lock=out/f'block{a.block}.running'
        try:
            with lock.open('x') as handle:handle.write(str(os.getpid()))
        except FileExistsError:
            print('Block already running',a.task,a.family,a.block,flush=True)
            return
        atexit.register(lambda:lock.unlink(missing_ok=True))
    ks=list(range(1,len(names)+1))
    blocks=[(tr,va) for tr,va in StratifiedKFold(5,shuffle=True,random_state=SEED).split(x,y)]
    blocks.append((np.arange(len(y)),np.array([],dtype=int)))
    for block,(tr,va) in enumerate(blocks):
        if a.block is not None and block!=a.block:continue
        if a.block is None and (out/f'block{block}.running').exists():continue
        resultpath=out/f'block{block}.joblib'
        if resultpath.exists():continue
        xx=x[tr];yy=y[tr];seed=SEED+1000*(block+1)
        search=[];ps={}
        for f,(it,iv) in enumerate(StratifiedKFold(3,shuffle=True,random_state=seed).split(xx,yy)):
            prep=preparation(xx[it],yy[it],seed+f,False)
            for pi,param in enumerate(PARAMS[a.family]):
                for k in ks:
                    cache=out/f'b{block}_f{f}_p{pi}_k{k}.npz'
                    if cache.exists():p=np.load(cache)['p']
                    else:
                        p,b=fit_predict(xx[it],yy[it],xx[iv],a.family,param,k,seed+f,a.checkpoint,False,prep)
                        np.savez(cache,p=p,validation=iv,rank=prep[2][prep[3]])
                        del b
                    search.append(dict(fold=f,param=pi,k=k,auc=roc_auc_score(yy[iv],p)))
                    ps.setdefault((pi,k),np.full(len(yy),np.nan))[iv]=p
            print(a.task,a.family,'outer',block+1,'inner',f+1,'complete',flush=True)
        search=pd.DataFrame(search)
        means=search.groupby(['param','k']).auc.mean().reset_index().sort_values(['auc','k','param'],ascending=[False,True,True])
        best=means.iloc[0];pi=int(best.param);k=int(best.k);ip=ps[pi,k]
        fpr,tpr,cut=roc_curve(yy,ip);threshold=float(cut[np.argmax(np.where(np.isfinite(cut),tpr-fpr,-np.inf))])
        if block<5:target=x[va]
        else:
            target=d[~d.partition.eq('Development')][names].to_numpy(float)
        p,bundle=fit_predict(xx,yy,target,a.family,PARAMS[a.family][pi],k,seed,a.checkpoint,False)
        result=dict(block=block,training=tr,validation=va,p=p,threshold=threshold,inner_auc=float(best.auc),search=search,selected=bundle['selected'],parameters=PARAMS[a.family][pi],features=names,signature=signature)
        if block==5:
            if a.family!='TabPFN':joblib.dump(bundle,out/'model.joblib')
            result['patient_id']=d.loc[~d.partition.eq('Development'),'患者ID'].tolist()
        joblib.dump(result,resultpath)
        print(a.task,a.family,'block',block+1,'saved',flush=True)
    if all((out/f'block{i}.joblib').exists() for i in range(6)):
        p=np.full(len(y),np.nan);cuts=np.full(len(y),np.nan)
        for i in range(5):
            b=joblib.load(out/f'block{i}.joblib');p[b['validation']]=b['p'];cuts[b['validation']]=b['threshold']
        assert np.isfinite(p).all()
        pd.DataFrame(dict(patient_id=dev['患者ID'],y=y,p=p,threshold=cuts)).to_csv(out/'nested_oof.csv',index=False)
        final=joblib.load(out/'block5.joblib')
        result=dict(task=a.task,family=a.family,n=len(y),nested_auc=roc_auc_score(y,p),selection_cv_auc=final['inner_auc'],features=[names[i] for i in final['selected']],parameters=final['parameters'],threshold=final['threshold'],signature=signature)
        (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf8')
        print(json.dumps(result),flush=True)

def fit_subset(x,y,v,family,param,selected,seed,checkpoint):
    selected=np.asarray(selected,int)
    imp=SimpleImputer(strategy='median',keep_empty_features=True)
    scale=StandardScaler()
    z=scale.fit_transform(imp.fit_transform(np.log1p(x)))
    model=estimator(family,param,seed,checkpoint)
    train=x[:,selected] if family=='TabPFN' else z[:,selected]
    target=v[:,selected] if family=='TabPFN' else scale.transform(imp.transform(np.log1p(v)))[:,selected]
    model.fit(train,y)
    bundle=dict(model=model,imputer=imp,scaler=scale,eligible=np.arange(x.shape[1]),selected_local=selected,selected=selected,family=family,k=len(selected),parameters=param)
    return model.predict_proba(target)[:,1],bundle

def treatment_response_floating(a):
    d=pd.read_csv(a.input)
    if not d['患者ID'].is_unique: raise ValueError('患者ID must be unique')
    if not set(d['centre'].dropna().unique()).issubset({'Sanjiu','Nanfang'}): raise ValueError('centre must be Sanjiu or Nanfang')
    if not set(d['y'].dropna().astype(int).unique()).issubset({0,1}): raise ValueError('y must be coded 0/1')
    if 'response_status' in d:
        expected=d['response_status'].map({'Responder':1,'Stable disease':0,'Progressive disease':0})
        if expected.isna().any() or not np.array_equal(expected.to_numpy(int),d['y'].to_numpy(int)): raise ValueError('response_status and y are inconsistent')
    if a.early:
        ids=pd.read_csv(a.early)['患者ID']
        d=d[~d['患者ID'].isin(ids)].copy()
    d['partition']=np.where(d.centre.eq('Sanjiu'),'Development','External')
    names=FEATURES;assert len(names)==12 and d['患者ID'].is_unique
    dev=d[d.partition.eq('Development')].reset_index(drop=True)
    x=dev[names].to_numpy(float);y=dev.y.to_numpy(int)
    assert not np.isinf(x).any() and not (x<0).any()
    sig=hashlib.sha256(pd.util.hash_pandas_object(d[['患者ID','partition','y']+names],index=False).values.tobytes()+json.dumps(PARAMS,sort_keys=True).encode()+b'treatment_response12-floating-inner3-outer5-v1').hexdigest()
    out=Path(a.output)/'TreatmentResponse'/a.family.replace('/','_').replace(' ','_')/sig;out.mkdir(parents=True,exist_ok=True)
    blocks=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(x,y));blocks.append((np.arange(len(y)),np.array([],int)))
    for bi,(tr,va) in enumerate(blocks):
        if a.block is not None and bi!=a.block:continue
        path=out/f'block{bi}.joblib'
        if path.exists():continue
        xx=x[tr];yy=y[tr];seed=SEED+1000*(bi+1)
        folds=list(StratifiedKFold(3,shuffle=True,random_state=seed).split(xx,yy))
        eligible=np.ones(len(names),bool)
        for it,iv in folds:
            eligible&=(np.isfinite(xx[it]).mean(axis=0)>=.2)&np.array([len(np.unique(c[np.isfinite(c)]))>=3 for c in xx[it].T])
        pool=set(np.flatnonzero(eligible).tolist());assert pool
        cache={};best_by_size={};trace=[]
        def evaluate(subset):
            key=tuple(sorted(subset))
            if key in cache:return cache[key]
            mask=sum(1<<i for i in key);options=[]
            for pi,param in enumerate(PARAMS[a.family]):
                pp=np.full(len(yy),np.nan);scores=[]
                for fi,(it,iv) in enumerate(folds):
                    file=out/f'b{bi}_f{fi}_p{pi}_s{mask}.npz'
                    if file.exists():pr=np.load(file)['p']
                    else:
                        pr,b=fit_subset(xx[it],yy[it],xx[iv],a.family,param,key,seed+fi,a.checkpoint)
                        np.savez(file,p=pr,validation=iv,selected=key);del b
                    pp[iv]=pr;scores.append(roc_auc_score(yy[iv],pr))
                    trace.append(dict(fold=fi,param=pi,k=len(key),auc=scores[-1],subset=mask))
                options.append(dict(auc=float(np.mean(scores)),param=pi,selected=key,p=pp))
            result=max(options,key=lambda r:(r['auc'],-r['param']))
            cache[key]=result
            prev=best_by_size.get(len(key))
            if prev is None or result['auc']>prev['auc']+1e-12:best_by_size[len(key)]=result
            return result
        current=tuple()
        while len(current)<len(pool):
            choices=[evaluate(set(current)|{j}) for j in sorted(pool-set(current))]
            chosen=max(choices,key=lambda r:(r['auc'],tuple(-j for j in r['selected'])))
            current=chosen['selected']
            while len(current)>2:
                oldbest=best_by_size.get(len(current)-1)
                score_before=oldbest['auc'] if oldbest else -np.inf
                deletions=[evaluate(set(current)-{j}) for j in current]
                backward=max(deletions,key=lambda r:(r['auc'],tuple(-j for j in r['selected'])))
                if backward['auc']>score_before+1e-12:current=backward['selected']
                else:break
            print('TreatmentResponse',a.family,'block',bi+1,'search size',len(current),'subsets',len(cache),flush=True)
        best=max(cache.values(),key=lambda r:(r['auc'],-len(r['selected']),-r['param'],tuple(-j for j in r['selected'])))
        fpr,tpr,cut=roc_curve(yy,best['p']);threshold=float(cut[np.argmax(np.where(np.isfinite(cut),tpr-fpr,-np.inf))])
        target=x[va] if bi<5 else d.loc[~d.partition.eq('Development'),names].to_numpy(float)
        pred,bundle=fit_subset(xx,yy,target,a.family,PARAMS[a.family][best['param']],best['selected'],seed,a.checkpoint)
        result=dict(block=bi,training=tr,validation=va,p=pred,threshold=threshold,inner_auc=best['auc'],search=pd.DataFrame(trace),selected=np.asarray(best['selected']),parameters=PARAMS[a.family][best['param']],features=names,signature=sig,selection_method='sequential_forward_floating')
        if bi==5:
            result['patient_id']=d.loc[~d.partition.eq('Development'),'患者ID'].tolist()
            if a.family!='TabPFN':joblib.dump(bundle,out/'model.joblib')
        joblib.dump(result,path);print('TreatmentResponse',a.family,'block',bi+1,'saved',flush=True)
    if all((out/f'block{i}.joblib').exists() for i in range(6)):
        pp=np.full(len(y),np.nan);cuts=np.full(len(y),np.nan)
        for i in range(5):
            b=joblib.load(out/f'block{i}.joblib');pp[b['validation']]=b['p'];cuts[b['validation']]=b['threshold']
        pd.DataFrame(dict(patient_id=dev['患者ID'],y=y,p=pp,threshold=cuts)).to_csv(out/'nested_oof.csv',index=False)
        b=joblib.load(out/'block5.joblib')
        result=dict(task='TreatmentResponse',family=a.family,n=len(y),nested_auc=roc_auc_score(y,pp),selection_cv_auc=b['inner_auc'],features=[names[j] for j in b['selected']],parameters=b['parameters'],threshold=b['threshold'],signature=sig,selection_method='sequential_forward_floating')
        (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--early');p.add_argument('--output',required=True);p.add_argument('--checkpoint');p.add_argument('--task',choices=['P_vs_D','P_vs_B','TreatmentResponse'],required=True);p.add_argument('--family',choices=FAMILIES,required=True);p.add_argument('--block',type=int);a=p.parse_args()
    if a.family=='TabPFN':
        from filelock import FileLock
        locks=Path(a.output)/'.locks';locks.mkdir(parents=True,exist_ok=True)
        with FileLock(str(locks/f'{a.task}_{a.family}_{a.block}.lock')):run(a)
    else:run(a)
