"""Patient-level diagnostic, early-detection and treatment-response cohorts."""
import sys,json
from pathlib import Path
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch
from audit_panel_alignment import require_matplotlib_panel_alignment
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--data',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
OUT=Path(args.output);OUT.mkdir(parents=True,exist_ok=True)
mpl.rcParams.update({'font.family':'Arial','font.size':8,'axes.titlesize':9,'svg.fonttype':'none','pdf.fonttype':42,'figure.facecolor':'white'})
PRIMARY='#21665E';ACCENT='#963C50';THIRD='#B08A45';GRAY='#7C8186'
LOW='#416D9C';HIGH='#A52435'
d=pd.read_csv(Path(args.data)/'diagnostic_data.csv',low_memory=False)
assert len(d)==9487 and int(d.y.sum())==1040
counts=d.groupby(['partition','分组']).size();assert counts.sum()==9487
fig,axs=plt.subplots(3,1,figsize=(170/25.4,180/25.4))
fig.subplots_adjust(left=.055,right=.975,top=.955,bottom=.035,hspace=.14)
def box(ax,x,y,w,h,title,body,color):
    ax.add_patch(Rectangle((x,y),w,h,facecolor='white',edgecolor=color,lw=.9))
    ax.text(x+.017,y+h-.037,title,weight='bold',fontsize=8,va='top')
    ax.text(x+.017,y+h-.14,body,fontsize=7.3,va='top',linespacing=1.45)
def arrow(ax,x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=9,color=GRAY,lw=.8))
for ax,letter,title in zip(axs,'ABC',['Diagnostic cohorts','Early identification before cytological conversion','Treatment response cohort']):
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.set_gid(letter)
    ax.text(0,1.04,letter,transform=ax.transAxes,fontsize=11,weight='bold')

a=axs[0]
box(a,0,.60,1,.36,'9,487 eligible patients; primary CNS tumours excluded','LM: 1,040 (888 cytology-positive; 152 cytology-negative with imaging-supported LM)\nBenign controls: 8,178; cancer non-LM controls: 269',PRIMARY)
arrow(a,.5,.59,.5,.48)
box(a,0,.01,.49,.46,'Sanjiu: 9,096 patients','Development: 7,283\n723 LM / 6,391 benign / 169 cancer controls\nInternal test: 1,813\n169 LM / 1,598 benign / 46 cancer controls',PRIMARY)
box(a,.51,.01,.49,.46,'Nanfang: 391 patients','External validation\n148 LM / 189 benign / 54 cancer controls\nTwo contrasts: LM vs cancer controls;\nLM vs benign controls',ACCENT)
a=axs[1]
box(a,0,.60,1,.36,'81 patients with cytological conversion within 6 months','Predictors: laboratory measurements at the first cytology-negative examination\nIndependent of diagnostic and treatment-response modelling cohorts',PRIMARY)
arrow(a,.5,.59,.5,.48)
box(a,0,.01,.49,.46,'Sanjiu: 70 converters','First-negative detection rate\nTime to first cytology-positive examination:\nwithin 30 days; 31-90 days;\n91-180 days',PRIMARY)
box(a,.51,.01,.49,.46,'Nanfang: 11 converters','First-negative detection rate\nModel and threshold determined\nusing development data\nEvaluation restricted to the 81 converters',ACCENT)
a=axs[2]
box(a,0,.60,1,.36,'241 patients with baseline CSF and evaluable treatment response','EANO–ESMO criteria: responder versus non-responder\nNon-responders: stable disease or progressive disease',THIRD)
arrow(a,.5,.59,.5,.48)
box(a,0,.01,.49,.46,'Sanjiu development: 191 patients','85 responders / 106 non-responders\nStable disease: 31; progressive disease: 75\nOuter 5-fold / inner 3-fold validation\nCSF laboratory measurements as predictors',THIRD)
box(a,.51,.01,.49,.46,'Nanfang external: 50 patients','25 responders / 25 non-responders\nStable disease: 3; progressive disease: 22\nIndependent external evaluation\nModel and threshold fixed in development',ACCENT)
fig.canvas.draw()
require_matplotlib_panel_alignment(fig,axes=list(axs),panel_ids=list('ABC'),json_out=OUT/'Figure1.alignment.json',strict=True,require_panel_labels=True,tolerance_pt=1.5,gutter_tolerance_pt=1.5)
fig.savefig(OUT/'Figure1.pdf')
fig.savefig(OUT/'Figure1.svg')
fig.savefig(OUT/'Figure1.png',dpi=300)
plt.close(fig)
counts.rename('n').reset_index().to_csv(OUT/'Figure1_counts.csv',index=False)
print('Figure1 rendered')
