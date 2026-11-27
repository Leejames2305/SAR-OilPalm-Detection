from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/app');df=pd.read_csv(ROOT/'report/EXPERIMENTS.csv');colors=['#405c85','#34977e','#c37d3b'];estates=['AirHitam','Palong','Serting']
fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
models=['incumbent','s009','s010','s011','s012','pooled_sar'];x=np.arange(len(models));width=.24
for i,e in enumerate(estates):
    v=df[(df.protocol=='official')&(df.estate==e)].set_index('model').loc[models,'lift'];axes[0].bar(x+(i-1)*width,v,width,color=colors[i],label=e)
axes[0].set_xticks(x,models,rotation=25,ha='right');axes[0].set_ylabel('AP minus prevalence');axes[0].set_title('Paired official 80 m block CV');axes[0].legend(frameon=False)
budget=pd.read_csv(ROOT/'report/inspection_budget_diagnostics.csv')
for i,e in enumerate(estates):
    v=budget[(budget.estate==e)&budget.include_middle];axes[1].plot(100*v.budget,100*v.recall,marker='o',color=colors[i],label=e)
axes[1].plot([0,50],[0,50],color='gray',ls=':',lw=1);axes[1].axvline(25,color='gray',ls='--',lw=1);axes[1].axhline(50,color='gray',ls='--',lw=1)
axes[1].set(xlabel='Inspection budget (% of all L trees)',ylabel='Unhealthy recall (%)',title='Frozen nominee: Middle consumes capacity',xlim=(0,51),ylim=(0,102))
for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
for extension in ['png','pdf']:fig.savefig(ROOT/f'report/lineage_and_triage.{extension}',dpi=180)
plt.close(fig)
protocols=['official','block120','block160','buffer40','buffer80','blockquad','blockquad80','blockquad160','quadrant','regions80']
fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained');v=df[(df.model=='pooled_sar')&df.protocol.isin(protocols)].pivot(index='protocol',columns='estate',values='lift').loc[protocols,estates]
im=ax.imshow(v.T,cmap='YlGnBu',vmin=0,vmax=.42,aspect='auto');ax.set_xticks(np.arange(len(protocols)),protocols,rotation=30,ha='right');ax.set_yticks(np.arange(3),estates);ax.set_title('Frozen synthesis: spatial holdout sensitivity (AP lift)')
for j in range(3):
    for i in range(len(protocols)):ax.text(i,j,f'{v.iloc[i,j]:.3f}',ha='center',va='center',color='white' if v.iloc[i,j]>.23 else 'black',fontsize=9)
fig.colorbar(im,ax=ax,label='AP lift',shrink=.8)
for extension in ['png','pdf']:fig.savefig(ROOT/f'report/spatial_robustness.{extension}',dpi=180)
