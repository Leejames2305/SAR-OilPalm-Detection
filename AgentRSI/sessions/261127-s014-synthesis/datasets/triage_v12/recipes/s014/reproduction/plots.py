from pathlib import Path
import json,sys
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def membership(s,f):
 size=len(s)*f;boundary=np.sort(s)[::-1][min(int(np.ceil(size))-1,len(s)-1)];a=s>boundary;b=s==boundary;return a.astype(float)+b*max(0,size-a.sum())/max(1,b.sum())
ROOT=Path('/app');estates=['AirHitam','Palong','Serting'];models=['incumbent','coords','s009','s012','s013','source_prior']
x=pd.read_csv(ROOT/'report/EXPERIMENTS.csv');official=x[x.protocol=='official'].pivot(index='model',columns='estate',values='lift')
V={'incumbent':[.037027688089114424,.2533059847191583,.06591786288934536],'coords':[.0397917985858249,.1278070936422355,.0317866456964607],'s009':[.065513,.276497,.188360],'s012':[.049552,.363177,.197506],'s013':[.055264,.339137,.257620],'source_prior':[.05928632140157762,.3442070884791901,.27663378162762625]}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(13,4.2),sharey=True)
for j,(ax,e) in enumerate(zip(axes,estates)):
 y=np.arange(len(models));ax.barh(y-.16,official.loc[models,e],height=.30,color='#236ea4',label='Correct union-anchored L CV');ax.barh(y+.16,[V[m][j] for m in models],height=.30,color='#e69a35',label='Noisy V');ax.set(yticks=y,yticklabels=models,title=e,xlabel='AP minus prevalence');ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
axes[0].legend(loc='lower right',fontsize=8);fig.suptitle('Paired lineage audit: local spatial CV and verified V',y=1.01);fig.tight_layout();fig.savefig(ROOT/'report/lineage_comparison.png',dpi=180,bbox_inches='tight');fig.savefig(ROOT/'report/lineage_comparison.pdf',bbox_inches='tight');plt.close(fig)
L=pd.read_csv(ROOT/'data/L_labelled.csv');fig,axes=plt.subplots(1,3,figsize=(13,3.8),sharey=True)
for ax,e in zip(axes,estates):
 d=L[L.estate==e].reset_index(drop=True);y=(d.label=='Unhealthy').to_numpy();budgets=np.arange(.01,.501,.01)
 for m,color in [('incumbent','#999999'),('s013','#cc8b35'),('source_prior','#236ea4')]:
  s=np.load(ROOT/f'research/results/{m}_official_{e}.npz')['s'];recall=[np.sum(membership(s,f)*y)/y.sum() for f in budgets];ax.plot(budgets*100,np.array(recall)*100,label=m,color=color,linewidth=2)
 ax.plot([0,50],[0,50],':',color='black',alpha=.5,label='Random inspections');ax.axhline(50,color='black',lw=.7,alpha=.5);ax.axvline(25,color='black',lw=.7,alpha=.5);ax.set(title=e,xlabel='Inspected fraction of all L trees (%)',xlim=(0,50),ylim=(0,100));ax.grid(alpha=.15)
axes[0].set_ylabel('Unhealthy recall (%)');axes[-1].legend(fontsize=8,loc='lower right');fig.suptitle('Official OOF inspection capacity, including Middle trees',y=1.02);fig.tight_layout();fig.savefig(ROOT/'report/inspection_curves.png',dpi=180,bbox_inches='tight');fig.savefig(ROOT/'report/inspection_curves.pdf',bbox_inches='tight');plt.close(fig)
protocols=['official','block120','block160','buffer40','buffer80','blockquad','blockquad80','blockquad160'];t=x[x.model=='source_prior'].pivot(index='protocol',columns='estate',values='delta_incumbent').reindex(protocols)[estates]
fig,ax=plt.subplots(figsize=(7,4.6));im=ax.imshow(t.to_numpy(),cmap='RdBu',vmin=-.35,vmax=.35,aspect='auto');ax.set(xticks=np.arange(3),xticklabels=estates,yticks=np.arange(len(t)),yticklabels=t.index,title='Source-prior AP lift gain over incumbent')
for i in range(len(t)):
 for j in range(3):ax.text(j,i,f'{t.iloc[i,j]:+.3f}',ha='center',va='center',color='white' if abs(t.iloc[i,j])>.22 else 'black')
fig.colorbar(im,ax=ax,label='Paired AP difference');fig.tight_layout();fig.savefig(ROOT/'report/spatial_robustness.png',dpi=180,bbox_inches='tight');fig.savefig(ROOT/'report/spatial_robustness.pdf',bbox_inches='tight');plt.close(fig)
print('Saved three standalone figure pairs')
