"""Apply a fixed final model using the manuscript feature schema."""
from pathlib import Path
import argparse,json,os
os.environ.update(TABPFN_DISABLE_TELEMETRY='1',HF_HUB_OFFLINE='1',TABPFN_ALLOW_CPU_LARGE_DATASET='1')
import pandas as pd,numpy as np,joblib
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--task',choices=['P_vs_D','P_vs_B','TreatmentResponse'],required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--checkpoint');a=p.parse_args()
c=json.loads((R/'configs'/f'{a.task}.json').read_text());d=pd.read_csv(a.input);names=c['features']
if d.empty:p.error('Input contains no cases; fill the input template first')
missing=[name for name in names if name not in d]
if missing:p.error('Missing predictors: '+', '.join(missing))
values=d[names].to_numpy(float)
if np.isinf(values).any() or (values<0).any():p.error('Predictors must be nonnegative numeric values or missing')
if c['family']=='TabPFN':
    if not a.checkpoint:p.error('--checkpoint is required for TabPFN')
    if not Path(a.checkpoint).is_file():p.error('Checkpoint file does not exist')
    if not (R/'models'/f'{a.task}_training.npz').is_file():p.error('Restricted training context is not included. Obtain the approved context matching this model configuration.')
    from tabpfn import TabPFNClassifier
    import torch
    torch.set_num_threads(2)
    train=np.load(R/'models'/f'{a.task}_training.npz');m=TabPFNClassifier(n_estimators=4,device='cpu',model_path=a.checkpoint,n_jobs=2,random_state=c['seed']).fit(train['X'],train['y']);prob=m.predict_proba(d[names].to_numpy(float))[:,1]
else:
    b=joblib.load(R/'models'/f'{a.task}.joblib');x=d[c['candidate_features']].to_numpy(float);x=b['scaler'].transform(b['imputer'].transform(np.log1p(x[:,b['eligible']])))[:,b['selected_local']];prob=b['model'].predict_proba(x)[:,1]
result=pd.DataFrame({'probability':prob,'threshold':c['threshold'],'predicted_class':prob>=c['threshold']})
if '患者ID' in d:result.insert(0,'patient_id',d['患者ID'])
elif 'patient_id' in d:result.insert(0,'patient_id',d['patient_id'])
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
result.to_csv(a.output,index=False)
