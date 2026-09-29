from pathlib import Path
import sys,json
import numpy as np,pandas as pd,joblib
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,confusion_matrix
from sklearn.linear_model import LogisticRegression
from scipy.stats import mannwhitneyu
ROOT=Path(__file__).resolve().parents[2]/'analysis_output'
OUT=ROOT/'publication';OUT.mkdir(parents=True,exist_ok=True)
SOURCE=Path(__file__).resolve().parents[2]/'source_data'
FAMILIES=['Logistic','LASSO','Random forest','XGBoost','ExtraTrees','RBF-SVM','ANN/MLP','TabPFN']
FEATURES=['CSF protein','Glucose','Chloride','Lactate','LDH','ADA','AST','Nucleated cells','Total cells','RBC','LNR','LMR']
main=pd.read_csv(SOURCE/'diagnostic_data.csv',low_memory=False)
early=pd.read_csv(SOURCE/'early_data.csv',low_memory=False)
treatment_response=pd.read_csv(SOURCE/'treatment_response_input.csv')
main=main[~main['患者ID'].isin(early['患者ID'])].copy()
treatment_response=treatment_response[~treatment_response['患者ID'].isin(early['患者ID'])].copy()

def primary_site_table():
    table=pd.read_csv(SOURCE/'Table1.csv',dtype=str)
    groups=[main[main.partition.eq(part)&main['分组'].eq(group)]
            for part in ['Development','Internal test','Nanfang-sheet validation']
            for group in ['阳性组','癌症非脑膜转移对照组']]
    for index,row in table.iterrows():
        if row['特征'].startswith('诊断依据：'):
            basis='影像学支持LM' if 'MRI' in row['特征'] else '细胞学阳性'
            for column,patients in zip(table.columns[1:],groups):
                count=int(patients.label_basis.eq(basis).sum())
                table.loc[index,column]=f'{count} ({100*count/len(patients):.1f})' if patients.y.eq(1).all() else '—'
            continue
        if not row['特征'].startswith('原发部位：'):continue
        site=row['特征'].split('：',1)[1]
        for column,patients in zip(table.columns[1:],groups):
            count=int(patients.primary_site.eq(site).sum())
            table.loc[index,column]=f'{count} ({100*count/len(patients):.1f})'
    return table
def wilson(k,n):
    z=1.95996398454;p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return c-h,c+h
def metrics(y,p,cut,bootstrap=1000):
    y=np.asarray(y,int);p=np.asarray(p,float);positive=p>=cut
    tn,fp,fn,tp=confusion_matrix(y,positive,labels=[0,1]).ravel()
    result=dict(n=len(y),positive=int(y.sum()),AUROC=roc_auc_score(y,p),AP=average_precision_score(y,p),Brier=brier_score_loss(y,p),sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),PPV=tp/(tp+fp) if tp+fp else np.nan,NPV=tn/(tn+fn) if tn+fn else np.nan,TP=int(tp),FP=int(fp),TN=int(tn),FN=int(fn))
    bounded=np.clip(p,1e-6,1-1e-6)
    calibration=LogisticRegression(penalty=None,max_iter=3000).fit(np.log(bounded/(1-bounded))[:,None],y)
    cuts=np.asarray(cut,float)
    result.update(Accuracy=(tp+tn)/len(y),F1=2*tp/(2*tp+fp+fn),calibration_slope=float(calibration.coef_[0,0]),calibration_intercept=float(calibration.intercept_[0]),threshold_display=f'{cuts.flat[0]:.4f}' if np.unique(cuts).size==1 else 'Fold-specific')
    rng=np.random.default_rng(20260911);ind=[np.flatnonzero(y==v) for v in [0,1]];aucs=[]
    for _ in range(bootstrap):
        ix=np.r_[rng.choice(ind[0],len(ind[0]),replace=True),rng.choice(ind[1],len(ind[1]),replace=True)]
        aucs.append(roc_auc_score(y[ix],p[ix]))
    result['AUROC_lo'],result['AUROC_hi']=np.quantile(aucs,[.025,.975])
    for key,k,n in [('sensitivity',tp,tp+fn),('specificity',tn,tn+fp),('PPV',tp,tp+fp),('NPV',tn,tn+fn)]:
        result[key+'_lo'],result[key+'_hi']=wilson(k,n) if n else (np.nan,np.nan)
    return result
def taskdata(task):
    if task=='TreatmentResponse':
        d=treatment_response.copy()
        if not d['患者ID'].is_unique: raise ValueError('TreatmentResponse 患者ID must be unique')
        if not set(d['centre'].dropna().unique()).issubset({'Sanjiu','Nanfang'}): raise ValueError('TreatmentResponse centre must be Sanjiu or Nanfang')
        expected=d['response_status'].map({'Responder':1,'Stable disease':0,'Progressive disease':0})
        if expected.isna().any() or not np.array_equal(expected.to_numpy(int),d['y'].to_numpy(int)): raise ValueError('TreatmentResponse response_status and y are inconsistent')
        d['partition']=np.where(d.centre.eq('Sanjiu'),'Development','External');return d
    if not main['患者ID'].is_unique: raise ValueError('Diagnostic 患者ID must be unique')
    if not set(main['partition'].dropna().unique()).issubset({'Development','Internal test','Nanfang-sheet validation'}): raise ValueError('Unexpected diagnostic partition')
    return main[main['分组'].isin(['阳性组','癌症非脑膜转移对照组' if task=='P_vs_D' else '阴性组'])].copy()
def assemble(task):
    folders={f:list((ROOT/'nested'/task/f.replace('/','_').replace(' ','_')).glob('*/metrics.json')) for f in FAMILIES}
    if any(len(v)!=1 for v in folders.values()):
        print('Awaiting',task,[f for f,v in folders.items() if len(v)!=1]);return
    records={f:json.loads(v[0].read_text()) for f,v in folders.items()}
    winner=max(FAMILIES,key=lambda f:records[f]['selection_cv_auc'])
    d=taskdata(task);dev=d[d.partition.eq('Development')].reset_index(drop=True)
    y=dev.y.to_numpy(int);allpred=[];performance=[];selections=[]
    for family in FAMILIES:
        folder=folders[family][0].parent;rec=records[family]
        q=pd.read_csv(folder/'nested_oof.csv');q['task']=task;q['model']=family;q['split']='Internal OOF';allpred.append(q)
        performance.append(dict(task=task,model=family,split='Internal OOF',cv_mean_auc=rec['selection_cv_auc'],k=len(rec['features']),**metrics(q.y,q.p,q.threshold)))
        b=joblib.load(folder/'block5.joblib')
        q=d[~d.partition.eq('Development')][['患者ID','partition','y']].rename(columns={'患者ID':'patient_id','partition':'split'}).copy()
        assert q.patient_id.tolist()==b['patient_id'];q['p']=b['p'];q['threshold']=b['threshold'];q['model']=family;q['task']=task
        q['split']=q.split.replace({'Internal test':'Internal test','Nanfang-sheet validation':'External'});allpred.append(q)
        for split,g in q.groupby('split'):performance.append(dict(task=task,model=family,split=split,cv_mean_auc=rec['selection_cv_auc'],k=len(rec['features']),**metrics(g.y,g.p,g.threshold)))
        for bi in range(6):
            block=joblib.load(folder/f'block{bi}.joblib')
            selections.append(dict(task=task,model=family,block=bi,inner_auc=block['inner_auc'],k=len(block['selected']),features=' | '.join(block['features'][j] for j in block['selected'])))
    # Every outer fold selects the complete workflow only from its inner scores.
    workflow=[]
    for i in range(5):
        blocks={f:joblib.load(folders[f][0].parent/f'block{i}.joblib') for f in FAMILIES}
        selected=max(FAMILIES,key=lambda f:blocks[f]['inner_auc']);b=blocks[selected]
        workflow.append(pd.DataFrame(dict(patient_id=dev.iloc[b['validation']]['患者ID'],y=y[b['validation']],p=b['p'],threshold=b['threshold'],split='Nested workflow',model='Selected workflow',task=task,selected_family=selected)))
    workflow=pd.concat(workflow);allpred.append(workflow)
    performance.append(dict(task=task,model='Selected workflow',split='Nested workflow',cv_mean_auc=records[winner]['selection_cv_auc'],k=len(records[winner]['features']),**metrics(workflow.y,workflow.p,workflow.threshold)))
    pred=pd.concat(allpred,ignore_index=True);pred.to_csv(OUT/f'{task}_predictions.csv',index=False)
    performance=pd.DataFrame(performance)
    if task=='TreatmentResponse':
        composition={}
        for partition,patients in d.groupby('partition'):
            counts=patients.response_status.value_counts()
            composition[partition]='; '.join(f'{status}: {counts.get(status,0)} ({100*counts.get(status,0)/len(patients):.1f}%)' for status in ['Responder','Stable disease','Progressive disease'])
        performance['response_distribution']=[composition['External' if split=='External' else 'Development'] for split in performance.split]
    performance.to_csv(OUT/f'{task}_performance.csv',index=False)
    pd.DataFrame(selections).to_csv(OUT/f'{task}_selections.csv',index=False)
    meta=dict(winner=winner,models=records,paths={f:str(v[0].parent) for f,v in folders.items()})
    (OUT/f'{task}_selection.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    print('Assembled',task,winner,flush=True)
if __name__=='__main__':
    primary_site_table().to_csv(OUT/'Table1.csv',index=False,encoding='utf-8-sig')
    tasks=sys.argv[1:] or ['P_vs_D','P_vs_B','TreatmentResponse']
    if any(task not in ['P_vs_D','P_vs_B','TreatmentResponse'] for task in tasks):
        raise ValueError('Supported tasks: P_vs_D, P_vs_B, TreatmentResponse')
    for task in tasks:assemble(task)
