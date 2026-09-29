"""A4-width manuscript figures for the study."""
from pathlib import Path
import argparse,json,sys
import numpy as np,pandas as pd,joblib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,precision_recall_curve,confusion_matrix
from audit_panel_alignment import require_matplotlib_panel_alignment
PRIMARY='#21665E';ACCENT='#963C50';THIRD='#B08A45';GRAY='#7C8186'
LOW='#416D9C';HIGH='#A52435';FOURTH='#786B99'
COLORS=[PRIMARY,ACCENT,THIRD,GRAY,'#8A769B','#A89465','#698CAD','#173F50']
FAMILIES=['Logistic','LASSO','Random forest','XGBoost','ExtraTrees','RBF-SVM','ANN/MLP','TabPFN']
mpl.rcParams.update({'font.family':'Arial','font.size':8,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':6.5,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'lines.linewidth':1.3,'svg.fonttype':'none','pdf.fonttype':42,'figure.facecolor':'white'})

def setup(rows=2,cols=2,height=175):
    fig,axes=plt.subplots(rows,cols,figsize=(170/25.4,height/25.4),squeeze=False)
    fig.subplots_adjust(left=.15,right=.97,bottom=.10,top=.93,wspace=.70,hspace=.65)
    return fig,axes.ravel()
def label(a,letter,title):
    a.set_gid(letter);a.set_title('',loc='left');a.set_title('')
    a.text(-.06/a.get_position().width,1.07,letter,transform=a.transAxes,fontsize=10,weight='bold',va='bottom')
def save(fig,axes,name,titles,subtitle):
    def clear_titles(axis):
        for loc in ('left','center','right'):axis.set_title('',loc=loc)
        for child in axis.child_axes:clear_titles(child)
    for axis in fig.axes:clear_titles(axis)
    if fig._suptitle is not None:fig._suptitle.set_text('')
    fig.set_size_inches(170/25.4,fig.get_size_inches()[1]*.92)
    grid=axes[0].get_subplotspec().get_topmost_subplotspec().get_gridspec()
    grid.update(top=.90 if name=='SFigure7' else .95,hspace=(grid.hspace if grid.hspace is not None else fig.subplotpars.hspace)*.74)
    if name not in ['SFigure1','SFigure5']:
        for ax,l,t in zip(axes,'ABCDEFGHIJKLMNOPQRSTUVWXYZ',titles):label(ax,l,t)
    else:
        axes[0].set_title('');axes[0].set_title('',loc='left')
    fig.canvas.draw()
    exemptions=[{'panels':['A'],'checks':['vertical-gutter'],'reason':'The full-width search panel is separated from D by a row containing B and C.'}] if False else []
    require_matplotlib_panel_alignment(fig,axes=list(axes),panel_ids=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(axes)]),json_out=OUT/f'{name}.alignment.json',strict=True,require_panel_labels=(name not in ['SFigure1','SFigure5']),tolerance_pt=1.5,gutter_tolerance_pt=1.5,exemptions=exemptions)
    fig.savefig(OUT/f'{name}.pdf');fig.savefig(OUT/f'{name}.svg');fig.savefig(OUT/f'{name}.png',dpi=300)
    plt.close(fig)
def readtask(task):
    meta=json.loads((DATA/f'{task}_selection.json').read_text())
    meta['paths']={k:str((DATA/Path(v)).resolve()) if not Path(v).is_absolute() else v for k,v in meta['paths'].items()}
    return meta,pd.read_csv(DATA/f'{task}_predictions.csv'),pd.read_csv(DATA/f'{task}_performance.csv')
def search(ax,meta):
    from matplotlib.patches import Rectangle
    first=Path(meta['paths'][FAMILIES[0]])
    maximum=len(joblib.load(first/'block5.joblib')['features'])
    if maximum>12:maximum=max(len(np.load(first/f'b5_f{fold}_p0_k1.npz')['rank']) for fold in range(3))
    values=[]
    for f in FAMILIES:
        z=joblib.load(Path(meta['paths'][f])/'block5.joblib')['search']
        group=['param','k']+(['subset'] if 'subset' in z.columns else [])
        g=z.groupby(group).auc.mean().reset_index().groupby('k').auc.max()
        values.append(g.reindex(range(1,maximum+1)).values)
    values=np.array(values);cmap=mpl.colors.LinearSegmentedColormap.from_list('auc',[LOW,'#F7F5F0',HIGH])
    ax.imshow(values,origin='upper',aspect='auto',extent=(.5,maximum+.5,7.5,-.5),cmap=cmap,vmin=np.nanmin(values),vmax=np.nanmax(values))
    for i in range(8):
        for j in range(maximum):
            if np.isfinite(values[i,j]):ax.text(j+1,i,f'{values[i,j]:.3f}',ha='center',va='center',fontsize=5.1,color='white' if (values[i,j]-np.nanmin(values))/(np.nanmax(values)-np.nanmin(values))>.75 or (values[i,j]-np.nanmin(values))/(np.nanmax(values)-np.nanmin(values))<.18 else '#202020')
    row=FAMILIES.index(meta['winner']);k=len(meta['models'][meta['winner']]['features'])
    ax.add_patch(Rectangle((k-.5,row-.5),1,1,fill=False,edgecolor='#202020',lw=1.8))
    ax.set_yticks(range(8),FAMILIES,fontsize=6);ax.set_xticks(np.arange(0,maximum+1,2));ax.set_xlim(0,maximum+.5)
    ax.set_xlabel('Number of selected features; black outline = selected combination',fontsize=6)

def comparison(ax,perf,meta):
    z=perf[perf.split.eq('Internal OOF')].set_index('model').loc[FAMILIES]
    for i,(f,r) in enumerate(z.iterrows()):
        ax.errorbar(r.AUROC,i,xerr=[[r.AUROC-r.AUROC_lo],[r.AUROC_hi-r.AUROC]],fmt='o',color=ACCENT if f==meta['winner'] else PRIMARY,capsize=2,ms=3,lw=.8)
        ax.text(1.005,i,f'{r.AUROC:.3f}',va='center',fontsize=6.2)
    ax.set_yticks(range(8),FAMILIES);ax.set(xlim=(.45,1.12),ylim=(7.6,-.6),xlabel='Outer-fold AUROC (95% CI)',xticks=[.5,.7,.9]);ax.xaxis.label.set_size(6)
def selected(pred,meta,workflow=False):
    internal=pred[pred.model.eq('Selected workflow')&pred.split.eq('Nested workflow')] if workflow else pred[pred.model.eq(meta['winner'])&pred.split.eq('Internal OOF')]
    external=pred[pred.model.eq(meta['winner'])&pred.split.eq('External')]
    return internal,external
def roc(ax,pred,perf,meta,internal_split='Internal OOF'):
    groups=[pred[pred.model.eq(meta['winner'])&pred.split.eq(s)] for s in [internal_split,'External']]
    for g,name,color in zip(groups,[internal_split,'External'],[PRIMARY,ACCENT]):
        fpr,tpr,_=roc_curve(g.y,g.p)
        r=perf[perf.model.eq(meta['winner'])&perf.split.eq(name)].iloc[0]
        display={'Internal test':'Internal test','Internal OOF':'Development cross-validation','External':'External validation'}[name]
        ax.plot(fpr,tpr,color=color,label=f'{display} (n = {len(g)})\n{r.AUROC:.3f} ({r.AUROC_lo:.3f}–{r.AUROC_hi:.3f})')
    ax.plot([0,1],[0,1],ls='--',color=GRAY,lw=.7);ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='1 − Specificity',ylabel='Sensitivity');ax.legend(loc='lower right',frameon=False,fontsize=6)
def dca(ax,g,title=None):
    thresholds=np.arange(.01,1.,.01);y=g.y.to_numpy();p=g.p.to_numpy();n=len(g);net=[]
    for t in thresholds:net.append(((p>=t)&(y==1)).sum()/n-((p>=t)&(y==0)).sum()/n*t/(1-t))
    allnet=y.mean()-(1-y.mean())*thresholds/(1-thresholds)
    ax.plot(thresholds,net,color=PRIMARY,label='Model');ax.plot(thresholds,allnet,color=GRAY,ls='--',label='Treat all');ax.axhline(0,color=ACCENT,lw=.8,label='Treat none')
    ax.set_yscale('asinh',linear_width=.05);ax.set_yticks([-10,-1,-.1,0,.1,1],['−10','−1','−0.1','0','0.1','1']);ax.minorticks_off();ax.set(xlim=(0,1),xlabel='Threshold probability',ylabel='Net benefit (asinh scale)');ax.legend(frameon=False,loc='lower left',fontsize=6)
def custom_grid(rows,height=245,spans=()):
    height={250:210,245:195,240:190}.get(height,height)
    fig=plt.figure(figsize=(170/25.4,height/25.4));grid=fig.add_gridspec(rows,2,left=.17,right=.96,bottom=.08,top=.93,wspace=.76,hspace=.50)
    axes=[]
    for row in range(rows):
        if row in spans:axes.append(fig.add_subplot(grid[row,:]))
        else:axes.extend([fig.add_subplot(grid[row,0]),fig.add_subplot(grid[row,1])])
    return fig,axes

def beeswarm(ax,s,order,limit):
    from scipy.stats import rankdata
    cmap=mpl.colors.LinearSegmentedColormap.from_list('value',[LOW,'#F7F5F0',HIGH]);rng=np.random.default_rng(20260911)
    for i,f in enumerate(order):
        z=s[s.feature.eq(f)].copy();v=z.value.to_numpy(float);finite=np.isfinite(v);color=np.full(len(v),.5)
        if finite.any():color[finite]=(rankdata(v[finite])-.5)/finite.sum()
        x=z.shap.to_numpy(float);bins=np.floor((x+limit)/(2*limit)*40).astype(int);jitter=np.zeros(len(x))
        for b in np.unique(bins):
            ix=np.flatnonzero(bins==b);rng.shuffle(ix);off=(np.arange(len(ix))-(len(ix)-1)/2);jitter[ix]=off/max(1,len(ix))*.65
        ax.scatter(x,i+jitter,c=color,cmap=cmap,vmin=0,vmax=1,s=3.5,alpha=.65,rasterized=True,linewidths=0)
    ax.axvline(0,color=GRAY,lw=.6);ax.set_yticks(range(len(order)),order,fontsize=5.7);ax.set(xlim=(-limit,limit),ylim=(len(order)-.5,-.5));ax.tick_params(axis='x',labelsize=6)
    ax.set_xlabel('SHAP value (probability)',fontsize=6)

def paired_shap(container):
    s=pd.read_csv(DATA/'diagnostic_shap.csv')
    s=s[s.split.isin(['Internal test','External']) & s.block.eq(5)].copy()
    meta=json.loads((DATA/'P_vs_D_selection.json').read_text())
    features=meta['models'][meta['winner']]['features']
    assert set(s.feature)==set(features)
    assert s[s.split.eq('Internal test')].patient_id.nunique()==215
    assert s[s.split.eq('External')].patient_id.nunique()==202
    order=s[s.split.eq('Internal test')].groupby('feature').shap.apply(lambda x:abs(x).mean()).sort_values(ascending=False).index.tolist()
    limit=float(abs(s.shap).max())*1.04;container.axis('off')
    for x,part in [(0,'Internal test'),(.60,'External')]:
        q=s[s.split.eq(part)]
        child=container.inset_axes([x,0,.40,.90]);beeswarm(child,q,order,limit)
        child.set_title('Internal test (n = 215)' if part=='Internal test' else 'External validation (n = 202)',fontsize=6.5,pad=3)
    container.text(.5,-.40,'Feature value: blue = low; red = high (within-feature ranks)',transform=container.transAxes,ha='center',fontsize=6)

def violin_points(ax,groups,labels,colors,probability=False):
    rng=np.random.default_rng(20260911)
    for i,(v,col) in enumerate(zip(groups,colors)):
        v=np.asarray(v,float);v=v[np.isfinite(v)]
        if len(v)>1 and np.ptp(v)>0:
            bodies=ax.violinplot(v,positions=[i],widths=.72,showextrema=False)['bodies']
            for body in bodies:body.set_facecolor(col);body.set_edgecolor(col);body.set_alpha(.13)
        if len(v):
            ax.boxplot(v,positions=[i],widths=.21,showfliers=False,medianprops={'color':'black'},boxprops={'linewidth':.7},whiskerprops={'linewidth':.6},capprops={'linewidth':.6})
            ax.scatter(i+rng.uniform(-.24,.24,len(v)),v,s=4,alpha=.4,color=col,linewidths=0,rasterized=True)
    ax.set_xticks(range(len(groups)),labels,fontsize=5.8);ax.set_xlim(-.6,len(groups)-.4)
    if probability:ax.set_ylim(0,1.03)

def main_model(task,name):
    meta,pred,perf=readtask(task);internal,external=selected(pred,meta)
    if task!='TreatmentResponse':
        internal=pred[pred.model.eq(meta['winner'])&pred.split.eq('Internal test')]
        assert len(internal)==215 and internal.patient_id.is_unique
        fig,ax=custom_grid(4,250,spans=(0,3));search(ax[0],meta);comparison(ax[1],perf,meta);roc(ax[2],pred,perf,meta,'Internal test');dca(ax[3],internal);dca(ax[4],external);paired_shap(ax[5])
        fig.set_size_inches(170/25.4,210/25.4)
        grid=ax[0].get_subplotspec().get_gridspec()
        grid.update(wspace=.60,hspace=.72,bottom=.10,top=.93)
        grid.set_height_ratios([1,1,1,1.12])
        ax[0].set_xticks(range(1,13));ax[0].set_xlim(.5,12.5)
        for a in ax[3:5]:
            for line in a.lines[:2]:
                v=np.asarray(line.get_ydata(),float);line.set_ydata(np.where(v>=-.1,v,np.nan))
            a.set_yscale('linear');a.set_ylim(-.1,1);a.set_yticks([0,.2,.4,.6,.8,1]);a.set_ylabel('Net benefit')
            a.legend(frameon=False,loc='upper right',fontsize=5.5,ncol=3,handlelength=1.3,columnspacing=.6,handletextpad=.4)
            a.text(.02,.22,'Net benefit below −0.1 is not shown',transform=a.transAxes,fontsize=5.5)
        ax[3].text(.02,.31,'Internal test (n = 215)',transform=ax[3].transAxes,fontsize=6)
        ax[4].text(.02,.31,'External validation (n = 202)',transform=ax[4].transAxes,fontsize=6)
        titles=['Algorithm × feature-count search','Cross-validated algorithm performance','Final-model discrimination','Internal test decision curve','External decision curve','Final-model SHAP distributions']
    else:
        assert len(internal)==191 and len(external)==50
        fig,ax=custom_grid(4,250,spans=(0,3))
        grid=ax[0].get_subplotspec().get_gridspec()
        grid.update(wspace=.60,hspace=.72,bottom=.10,top=.93)
        grid.set_height_ratios([1,1,1,1.3])
        search(ax[0],meta);ax[0].set_xticks(range(1,13));ax[0].set_xlim(.5,12.5)
        comparison(ax[1],perf,meta);roc(ax[2],pred,perf,meta)
        ax[2].legend(loc='lower right',frameon=False,fontsize=5,handlelength=1.4,handletextpad=.5)
        dca(ax[3],internal);dca(ax[4],external)
        for a,g,display in [(ax[3],internal,'Development cross-validation'),(ax[4],external,'External validation')]:
            for line in a.lines[:2]:
                v=np.asarray(line.get_ydata(),float);line.set_ydata(np.where(v>=-.1,v,np.nan))
            for line in a.lines[:2]:
                v=np.asarray(line.get_ydata(),float);line.set_ydata(np.where(v>=-.1,v,np.nan))
            a.set_yscale('linear');a.set_ylim(-.1,1);a.set_yticks([0,.2,.4,.6,.8,1]);a.set_ylabel('Net benefit')
            a.legend(frameon=False,loc='upper right',fontsize=5.5,ncol=3,handlelength=1.3,columnspacing=.6,handletextpad=.4)
            a.text(.02,.70,display,transform=a.transAxes,fontsize=6)
            a.text(.02,.61,'Net benefit below −0.1 is not shown',transform=a.transAxes,fontsize=5.5)
        z=pd.read_csv(DATA/'TreatmentResponse_shap.csv')
        features=meta['models'][meta['winner']]['features']
        assert len(features)==8
        z=z[z.feature.isin(features)].copy()
        order=z[z.split.eq('Internal OOF')].groupby('feature').shap.apply(lambda v:abs(v).mean()).reindex(features,fill_value=0).sort_values(ascending=False).index.tolist()
        raw=pd.read_csv(DATA/'treatment_response_input.csv').set_index('患者ID');limit=float(abs(z.shap).max())*1.04
        ax[5].axis('off')
        for x,part in [(0,'Internal OOF'),(.60,'External')]:
            q=z[z.split.eq(part)]
            index=pd.MultiIndex.from_product([q.patient_id.unique(),order],names=['patient_id','feature'])
            q=q.set_index(['patient_id','feature']).reindex(index).reset_index();q['shap']=q.shap.fillna(0)
            q['value']=[raw.at[pid,f] for pid,f in zip(q.patient_id,q.feature)]
            child=ax[5].inset_axes([x,0,.40,.90]);beeswarm(child,q,order,limit)
            display='Cross-validation' if part=='Internal OOF' else 'External validation'
            child.set_title(f'{display} (n = {q.patient_id.nunique()})',fontsize=6.5,pad=3)
        ax[5].text(.5,-.35,'Feature value: blue = low; red = high (within-feature ranks)',transform=ax[5].transAxes,ha='center',fontsize=6)
        titles=['Algorithm × feature-count search','Cross-validated algorithm performance','Selected-algorithm discrimination','Cross-validation decision curve','External decision curve','SHAP distributions']
    save(fig,ax,name,titles,f'{"Treatment response" if task=="TreatmentResponse" else "LM versus cancer controls"} | final model: {meta["winner"]}')

def early_figure():
    from matplotlib.patches import Patch
    meta=json.loads((DATA/'P_vs_D_selection.json').read_text());winner=meta['winner']
    p=pd.read_csv(DATA/'early_identification_predictions.csv');q=p[p.model.eq(winner)&p.split.eq('Converters')].copy()
    s=pd.read_csv(DATA/'early_rates.csv');fig,ax=setup(2,2,180)
    fig.subplots_adjust(wspace=.68,hspace=.64,bottom=.12)
    threshold=float(q.threshold.iloc[0]);assert q.threshold.nunique()==1 and len(q)==81
    assert abs(threshold-float(meta['models'][winner]['threshold']))<1e-12
    q=q.sort_values(['hospital','days'],ascending=[False,False]).reset_index(drop=True)
    for i,r in q.iterrows():
        color=ACCENT if r.detected else GRAY
        ax[0].plot([-r.days,0],[i,i],lw=.7,color=color,alpha=.5)
        ax[0].scatter(-r.days,i,s=7,color=color);ax[0].scatter(0,i,s=4,color=PRIMARY)
    ax[0].axhline(69.5,color=GRAY,ls='--',lw=.6)
    ax[0].text(-180,35,'Sanjiu (n=70)',fontsize=6);ax[0].text(-180,75,'Nanfang (n=11)',fontsize=6)
    ax[0].set(xlim=(-185,8),ylim=(-2,83),xlabel='Days relative to first positive cytology',ylabel='Patients (n = 81)',xticks=[-180,-90,-30,0],yticks=[0,40,80],yticklabels=['1','41','81'])
    colors=[ACCENT,PRIMARY,THIRD];rng=np.random.default_rng(20260911)
    masks=[q.days.le(30),q.days.between(31,90),q.days.between(91,180)]
    labels=['≤30','31–90','91–180']
    for i,(mask,col) in enumerate(zip(masks,colors)):
        v=q.loc[mask,'p'].to_numpy();n=len(v);det=int((v>=threshold).sum())
        ax[1].scatter(i+rng.uniform(-.21,.21,n),v,s=13,color=col,alpha=.7,linewidths=.25,edgecolors='white',zorder=3)
        lo,med,hi=np.quantile(v,[.25,.5,.75]);ax[1].errorbar(i,med,yerr=[[med-lo],[hi-med]],fmt='_',ms=16,color='black',lw=1.1,capsize=4,zorder=4)
        ax[1].text(i,1.075,f'{det}/{n} ({det/n:.1%})',ha='center',fontsize=6.2)
        labels[i]+=f'\n(n = {n})'
    ax[1].axhline(threshold,color=FOURTH,ls='--',lw=.8)
    ax[1].text(.99,.055,f'Dashed line: threshold {threshold:.4f}',transform=ax[1].get_yaxis_transform(),ha='right',fontsize=5.7,color=FOURTH)
    ax[1].set(xticks=[0,1,2],xticklabels=labels,xlim=(-.5,2.5),ylim=(0,1.15),yticks=[0,.25,.5,.75,1],ylabel='First-negative predicted probability',xlabel='Days to first positive cytology')
    hospitals=['All','Sanjiu','Nanfang-sheet'];hnames=['Overall','Sanjiu','Nanfang']
    z=s[s.model.eq(winner)&s.interval.eq('All')].set_index('hospital').loc[hospitals]
    for i,(_,r) in enumerate(z.iterrows()):
        pct=100*r.rate;ax[2].barh(i,pct,color=ACCENT,height=.54)
        ax[2].barh(i,100-pct,left=pct,color='#DEDFE2',height=.54)
        ax[2].text(pct/2,i,f'{pct:.1f}%',ha='center',va='center',color='white',weight='bold',fontsize=7)
        if pct<100:ax[2].text(pct+(100-pct)/2,i,f'{100-pct:.1f}%',ha='center',va='center',fontsize=6.5)
        ax[2].text(103,i,f'{int(r.detected)}/{int(r.n)}',va='center',fontsize=6.5)
    ax[2].set(yticks=[0,1,2],yticklabels=[f'{h}\n(n = {int(n)})' for h,n in zip(hnames,z.n)],ylim=(2.65,-.7),xlim=(0,117),xticks=[0,25,50,75,100],xlabel='Percentage of patients')
    ax[2].legend(handles=[Patch(color=ACCENT,label='Detected'),Patch(color='#DEDFE2',label='Below threshold')],loc='upper left',bbox_to_anchor=(0,.99),ncol=2,frameon=False,fontsize=5.8)
    groups=[q.loc[q.detected.astype(bool)&(True if h=='All' else q.hospital.eq(h)),'days'].to_numpy() for h in hospitals]
    violin_points(ax[3],groups,[f'{h}\n(n = {len(v)})' for h,v in zip(hnames,groups)],colors)
    for i,v in enumerate(groups):
        lo,med,hi=np.quantile(v,[.25,.5,.75]);ax[3].text(i,185,f'{med:g} [{lo:g}, {hi:g}]',ha='center',fontsize=5.8)
    ax[3].set(ylabel='Lead time among detected patients (days)',ylim=(0,200),yticks=[0,30,60,90,120,150,180])
    save(fig,ax,'Figure3',['Patient-level detection intervals','First-negative scores by time interval','Detection by hospital','Lead time among detected patients'],f'First-negative detection | {winner} | n = 81')


def distribution(task,name):
    """Show all 12 diagnostic candidates with BH-adjusted significance."""
    if task!='Diagnostic':
        return treatment_response_distributions(name)
    ef=pd.read_csv(DATA/'univariate_effects.csv')
    parts=['Development','Internal test','Nanfang-sheet validation']
    names=['CSF protein','Glucose','Chloride','Lactate','LDH','ADA','AST','Nucleated cells','Total cells','RBC','LNR','LMR']
    matrix=np.full((len(names),len(parts)),np.nan);qmat=np.full_like(matrix,np.nan)
    for i,f in enumerate(names):
        for j,pt in enumerate(parts):
            z=ef[(ef.task.eq(task))&(ef.feature.eq(f))&(ef.partition.eq(pt))]
            if len(z): matrix[i,j]=float(z.effect.iloc[0]);qmat[i,j]=float(z.q.iloc[0])
    fig,ax=plt.subplots(1,1,figsize=(170/25.4,112/25.4));fig.subplots_adjust(left=.20,right=.88,bottom=.20,top=.84)
    cmap=mpl.colors.LinearSegmentedColormap.from_list('effect',[LOW,'#F7F5F0',HIGH])
    im=ax.imshow(matrix,aspect='auto',cmap=cmap,vmin=-1,vmax=1)
    def stars(q):
        if not np.isfinite(q): return ''
        if q<.001: return '***'
        if q<.01: return '**'
        if q<.05: return '*'
        return ''
    for i in range(len(names)):
        for j in range(len(parts)):
            if np.isfinite(matrix[i,j]):
                frac=(matrix[i,j]+1)/2
                ax.text(j,i,f'{matrix[i,j]:.2f}{stars(qmat[i,j])}',ha='center',va='center',fontsize=6.2,color='white' if frac>.73 or frac<.20 else '#202020')
    ax.set_yticks(range(len(names)),names,fontsize=6.4)
    ax.set_xticks(range(len(parts)),['Development','Internal test','External'],fontsize=6.5)
    ax.set_xlabel('Cohort',fontsize=6.8);ax.set_ylabel('Candidate predictor',fontsize=6.8)
    ax.set_title('Standardized group differences across cohorts',fontsize=7.5,loc='left',pad=13)
    ax.text(0,-.16,'Positive values indicate higher values in LM; * q<0.05, ** q<0.01, *** q<0.001 (Benjamini–Hochberg corrected).',transform=ax.transAxes,fontsize=5.8,va='top')
    cbar=fig.colorbar(im,ax=ax,fraction=.025,pad=.025);cbar.set_label('Standardized difference',fontsize=6);cbar.ax.tick_params(labelsize=5.5)
    save(fig,[ax],name,['All 12 candidate predictors'],f'{task} | candidate predictor effects and adjusted significance')

def treatment_response_distributions(name):
    meta=json.loads((DATA/'TreatmentResponse_selection.json').read_text())
    final=meta['models'][meta['winner']]['features']
    preferred=['LMR','Nucleated cells','Chloride','LDH','Lactate','AST']
    features=[f for f in preferred if f in final]+[f for f in final if f not in preferred]
    data=pd.read_csv(DATA/'treatment_response_data.csv');effect=pd.read_csv(DATA/'univariate_effects.csv').query("task == 'TreatmentResponse'")
    order=effect[effect.partition.eq('Sanjiu')].sort_values('effect',ascending=False).feature.tolist()
    assert len(order)==12 and set(features)==set(final)
    rows=1+(len(features)+1)//2;spans=(rows-1,) if len(features)%2 else ()
    fig,axes=custom_grid(rows,65+40*(rows-1),spans=spans)
    grid=axes[0].get_subplotspec().get_gridspec();grid.set_height_ratios([1.65]+[1]*(rows-1));grid.update(hspace=.60,left=.17,right=.97,bottom=.065)
    for ax,part,color in zip(axes[:2],['Sanjiu','Nanfang'],[PRIMARY,ACCENT]):
        z=effect[effect.partition.eq(part)].set_index('feature').reindex(order)
        ax.errorbar(z.effect,range(len(z)),xerr=[z.effect-z.lo,z.hi-z.effect],fmt='o',color=color,ms=2.6,capsize=1.8,lw=.8)
        ax.axvline(0,color=GRAY,ls='--',lw=.65)
        ax.set_yticks(range(len(z)),order,fontsize=5.8);ax.invert_yaxis();ax.set(xlim=(-1.12,1.15),xlabel='Rank-biserial effect (95% CI)',xticks=[-1,-.5,0,.5,1])
        for i,r in enumerate(z.itertuples()):
            star='***' if r.q<.001 else '**' if r.q<.01 else '*' if r.q<.05 else ''
            if star:
                xpos=min(r.hi+.045,1.03) if r.effect>=0 else max(r.lo-.045,-1.03)
                ax.text(xpos,i,star,va='center',ha='left' if r.effect>=0 else 'right',fontsize=6.2,color=color)
    units={'CSF protein':'g/L','Glucose':'mmol/L','Chloride':'mmol/L','Lactate':'mmol/L','LDH':'U/L','ADA':'U/L','AST':'U/L','Nucleated cells':'10⁶/L','Total cells':'10⁶/L','RBC':'10⁶/L','LMR':'ratio','LNR':'ratio'}
    for ax,feature in zip(axes[2:],features):
        groups=[data.loc[data.centre.eq(part)&data.y.eq(y),feature].dropna().values for part in ['Sanjiu','Nanfang'] for y in [0,1]]
        violin_points(ax,groups,['Dev\n0','Dev\n1','Ext\n0','Ext\n1'],[PRIMARY,THIRD,ACCENT,FOURTH])
        if feature not in ['Chloride','Glucose']:
            ax.set_yscale('symlog',linthresh=.1)
            observed=np.concatenate(groups);minimum=float(np.min(observed))
            ax.set_ylim(bottom=0 if minimum==0 else minimum*.7)
        ax.set_ylabel(f'{feature} ({units[feature]})',fontsize=6)
    save(fig,axes,name,['']*len(axes),'')

def selected_folds(sel):
    return sel[sel.block.lt(5)].sort_values(['block','inner_auc'],ascending=[True,False]).groupby('block').head(1)

def frequency(ax,choices,features=False,candidates=None):
    if features:
        z=choices.features.str.split(' | ',regex=False).explode().value_counts()/5
        if candidates is not None:z=z.reindex(candidates,fill_value=0)
        z=z.sort_values();ax.barh(z.index,z,color=THIRD);ax.tick_params(axis='y',labelsize=5.1 if len(z)>20 else 5.8)
    else:z=choices.model.value_counts().reindex(FAMILIES,fill_value=0)/5;ax.barh(z.index,z,color=FOURTH);ax.tick_params(axis='y',labelsize=6)
    ax.set(xlim=(0,1),xlabel='Selection frequency');ax.set_xticks([0,.5,1],['0%','50%','100%'])

def one_calibration(ax,g,label,col):
    from scipy.stats import norm
    z=g.copy();z['bin']=pd.qcut(z.p,4,duplicates='drop');z=z.groupby('bin',observed=True).agg(p=('p','mean'),y=('y','mean'),n=('y','size'))
    den=1+1.96**2/z.n;center=(z.y+1.96**2/(2*z.n))/den;half=1.96*np.sqrt(z.y*(1-z.y)/z.n+1.96**2/(4*z.n**2))/den
    ax.errorbar(z.p,z.y,yerr=[np.maximum(0,z.y-(center-half)),np.maximum(0,(center+half)-z.y)],fmt='o-',color=col,ms=3,capsize=2,lw=.8,label=label)
    ax.plot([0,1],[0,1],ls='--',color=GRAY,lw=.6);ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='Mean predicted probability',ylabel='Observed proportion');ax.legend(frameon=False,fontsize=6)

def one_pr(ax,g,label,col):
    from sklearn.metrics import average_precision_score
    precision,recall,_=precision_recall_curve(g.y,g.p);ax.plot(recall,precision,color=col,label=f'{label}: AP {average_precision_score(g.y,g.p):.3f}');ax.axhline(g.y.mean(),ls=':',color=col,lw=.6)
    ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='Recall',ylabel='Precision');ax.legend(frameon=False,fontsize=6,loc='lower left')

def stability(task,name):
    from scipy.cluster.hierarchy import linkage,leaves_list
    from scipy.spatial.distance import squareform
    meta,pred,perf=readtask(task);sel=pd.read_csv(DATA/f'{task}_selections.csv');choices=selected_folds(sel);internal,external=selected(pred,meta);candidates=joblib.load(Path(meta['paths'][meta['winner']])/'block5.joblib')['features']
    if task=='P_vs_D':
        fig,ax=setup(2,2,180);fig.subplots_adjust(hspace=.75)
        corr=pd.read_csv(DATA/'diagnostic_correlations.csv',index_col=0);dist=(1-corr.fillna(0)).clip(0,2).to_numpy();np.fill_diagonal(dist,0);order=leaves_list(linkage(squareform(dist,checks=False),method='average'));corr=corr.iloc[order,order]
        ax[0].imshow(corr,vmin=-1,vmax=1,cmap=mpl.colors.LinearSegmentedColormap.from_list('corr',[LOW,'#F7F5F0',HIGH]),aspect='auto');ax[0].set_xticks(range(len(corr)),corr.columns,rotation=90,fontsize=5.5);ax[0].set_yticks(range(len(corr)),corr.index,fontsize=5.5)
        frequency(ax[1],choices,True,candidates);frequency(ax[2],choices);counts=choices.k.value_counts().sort_index();ax[3].bar(counts.index,counts,color=FOURTH);ax[3].set(xlabel='Selected feature count',ylabel='Outer-fold frequency',ylim=(0,5),xticks=range(0,13,2));titles=['Clustered Spearman correlations','Feature-selection frequency','Algorithm-selection frequency','Selected feature-count distribution']
    else:
        fig,ax=custom_grid(2,145,spans=(1,));fig.subplots_adjust(hspace=.55,top=.87)
        one_calibration(ax[0],internal,'Development cross-validation',PRIMARY);one_calibration(ax[1],external,'External validation',ACCENT)
        one_pr(ax[2],internal,'Development CV',PRIMARY);one_pr(ax[2],external,'External validation',ACCENT);titles=['']*3
    save(fig,ax,name,titles,f'{"Treatment response" if task=="TreatmentResponse" else "Diagnostic"} | selection stability and validation')

def confusion(ax,g):
    cm=confusion_matrix(g.y,g.p>=g.threshold,labels=[0,1]);ax.imshow(cm,cmap=mpl.colors.LinearSegmentedColormap.from_list('count',['#F4F7F3',PRIMARY]),aspect='auto')
    for i in range(2):
        for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=10,color='white' if cm[i,j]>cm.max()*.6 else 'black')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Negative','Positive'],yticklabels=['Negative','Positive'],xlabel='Predicted',ylabel='Observed')

def validation_supp(task,name):
    meta,pred,perf=readtask(task)
    internal=pred[pred.model.eq(meta['winner'])&pred.split.eq('Internal test')].copy()
    external=pred[pred.model.eq(meta['winner'])&pred.split.eq('External')].copy()
    threshold=meta['models'][meta['winner']]['threshold']
    assert len(internal)==215 and len(external)==202
    assert internal.patient_id.is_unique and external.patient_id.is_unique
    for g in [internal,external]:assert np.allclose(g.threshold,threshold,rtol=0,atol=1e-12)
    fig,ax=setup(3,2,210);fig.subplots_adjust(hspace=.50)
    one_calibration(ax[0],internal,'Internal test',PRIMARY);one_calibration(ax[1],external,'External validation',ACCENT)
    one_pr(ax[2],internal,'Internal test',PRIMARY);one_pr(ax[3],external,'External validation',ACCENT)
    confusion(ax[4],internal);confusion(ax[5],external)
    save(fig,ax,name,['']*6,'')


def waterfall(ax,z):
    z=z.sort_values('shap',key=lambda x:abs(x),ascending=False);base=float(z.reference_score.iloc[0]);current=0.
    for i,r in enumerate(z.itertuples()):
        end=current+r.shap;ax.barh(i,abs(r.shap),left=min(current,end),color=ACCENT if r.shap>0 else PRIMARY,height=.64);ax.plot([end,end],[i+.32,i+.68],color=GRAY,lw=.5);current=end
    ax.axvline(0,color=GRAY,ls=':',lw=.6);ax.axvline(current,color='black',lw=.6);ax.set_yticks(range(len(z)),z.feature,fontsize=5.8);ax.invert_yaxis();ax.set_xlabel(f'Cumulative SHAP contribution\nReference {base:.3f}; prediction {base+current:.3f}',fontsize=6)

def errors():
    meta,pred,perf=readtask('P_vs_D');s=pd.read_csv(DATA/'diagnostic_shap.csv');s=s[s.block.eq(5) & s.split.isin(['Internal test','External'])].copy();fig,ax=custom_grid(3,245,spans=(0,));titles=['Predictions around the classification threshold'];rng=np.random.default_rng(20260911)
    for j,part in enumerate(['Internal test','External']):
        q=pred[pred.model.eq(meta['winner'])&pred.split.eq(part)];margin=q.p-q.threshold
        for truth,col in [(0,PRIMARY),(1,ACCENT)]:
            v=margin[q.y.eq(truth)];ax[0].scatter(v,j+rng.uniform(-.17,.17,len(v)),s=5,color=col,alpha=.4,label=('Control' if truth==0 else 'LM') if j==0 else None,rasterized=True)
    ax[0].axvspan(-.1,.1,color=THIRD,alpha=.13);ax[0].axvline(0,color='black',ls='--',lw=.7);ax[0].set_yticks([0,1],['Internal test','External validation']);ax[0].set_xlabel('Predicted probability minus corresponding threshold');ax[0].legend(frameon=False,fontsize=6)
    for a,(part,truth) in zip(ax[1:],[('Internal test',0),('External',0),('Internal test',1),('External',1)]):
        q=pred[pred.model.eq(meta['winner'])&pred.split.eq(part)&pred.y.eq(truth)].copy();q=q[(q.p>=q.threshold)!=(q.y==1)];q['margin']=abs(q.p-q.threshold)
        r=q.sort_values(['margin','patient_id'],ascending=[False,True]).iloc[0];z=s[s.patient_id.eq(r.patient_id)&s.split.eq(part)];assert len(z)==len(meta['models'][meta['winner']]['features']);assert np.isclose(z.score.iloc[0],r.p,atol=1e-5);assert np.isclose(z.shap.sum()+z.reference_score.iloc[0],r.p,atol=1e-4);waterfall(a,z);titles.append(f'{part}: {"false positive" if truth==0 else "false negative"}')
    save(fig,ax,'SFigure3',titles,'Threshold proximity and representative SHAP waterfalls')

def benign():
    meta,pred,perf=readtask('P_vs_B')
    internal=pred[pred.model.eq(meta['winner'])&pred.split.eq('Internal test')]
    external=pred[pred.model.eq(meta['winner'])&pred.split.eq('External')]
    fig,ax=setup(2,2,165);fig.subplots_adjust(hspace=.50)
    comparison(ax[0],perf,meta);roc(ax[1],pred,perf,meta,'Internal test')
    one_pr(ax[2],internal,'Internal test',PRIMARY);one_pr(ax[2],external,'External validation',ACCENT)
    ax[2].legend(frameon=False,fontsize=6,loc='lower left',bbox_to_anchor=(0,.16))
    one_calibration(ax[3],internal,'Internal test',PRIMARY);one_calibration(ax[3],external,'External validation',ACCENT)
    save(fig,ax,'SFigure4',['']*4,'')


def longitudinal():
    s=pd.read_csv(DATA/'longitudinal_scores.csv');base=s[s.visit.eq('First negative')];follow=s[s.visit.eq('Follow-up')]
    fig,axes=plt.subplots(1,1,figsize=(170/25.4,110/25.4));fig.subplots_adjust(left=.14,right=.96,top=.95,bottom=.16);ax=[axes];threshold=float(base.threshold.iloc[0])
    repeated=s.copy();ids=repeated.patient_id.unique();grid=np.linspace(-180,0,61);curves=[]
    ax[0].scatter(-repeated.days_to_positive,repeated.score,s=7,color=PRIMARY,alpha=.16,edgecolors='none',zorder=2,rasterized=True)
    for pid in ids:
        q=repeated[repeated.patient_id.eq(pid)];x=-q.days_to_positive.to_numpy(float);v=q.score.to_numpy(float);w=np.exp(-.5*((grid[:,None]-x[None,:])/15)**2);den=w.sum(axis=1);num=w@v;curves.append(np.where(den>.05,num/np.maximum(den,1e-12),np.nan))
    curves=np.array(curves);support=np.isfinite(curves).sum(axis=0);valid=support>=10;mean=np.nanmean(curves,axis=0);rng=np.random.default_rng(20260911)
    with np.errstate(invalid='ignore'):boot=np.array([np.nanmean(curves[rng.integers(0,len(curves),len(curves))],axis=0) for _ in range(500)])
    lo,hi=np.nanquantile(boot,[.025,.975],axis=0);ax[0].fill_between(grid[valid],lo[valid],hi[valid],color=ACCENT,alpha=.14,zorder=1);ax[0].plot(grid[valid],mean[valid],color=ACCENT,lw=1.8,zorder=4)
    ax[0].set(xlabel='Days relative to first positive cytology',ylabel='Predicted LM probability',xlim=(-185,5),ylim=(0,1.03),xticks=[-180,-90,-30,0])
    for a in ax:a.axhline(threshold,color=THIRD,ls='--',lw=.7,zorder=3)
    save(fig,ax,'SFigure5',[''],'Early detection | baseline and longitudinal scores')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--figure',required=True);a=p.parse_args();DATA=Path(a.data);OUT=Path(a.output);OUT.mkdir(parents=True,exist_ok=True)
    jobs={'Figure2':lambda:main_model('P_vs_D','Figure2'),'Figure3':early_figure,'Figure4':lambda:main_model('TreatmentResponse','Figure4'),'SFigure1':lambda:distribution('Diagnostic','SFigure1'),'SFigure2':lambda:validation_supp('P_vs_D','SFigure2'),'SFigure3':errors,'SFigure4':benign,'SFigure5':longitudinal,'SFigure6':lambda:distribution('TreatmentResponse','SFigure6'),'SFigure7':lambda:stability('TreatmentResponse','SFigure7')}
    jobs[a.figure]()

