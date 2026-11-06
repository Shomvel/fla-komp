"""Completed coherence quality and conditioning figure, generated from audited statistics."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent/'block_runs_coh'

def main():
 s=json.loads((ROOT/'statistics.json').read_text());assert s['complete']
 d=json.loads((ROOT/'diagnostics-statistics.json').read_text());tag=(s.get('fine_selection') or s['selection'])['selected_tag']
 order=['off','99','995','999','95','80']
 arms=[f'coh_{family}_{t}' for family in ('fifo','gate','learned') for t in order if f'coh_{family}_{t}' in s['arms']]
 labels=[a.split('_')[1].title()+' / '+('off' if a.endswith('off') else '0.'+a.rsplit('_',1)[1]) for a in arms]
 palette={'off':'#777777','99':'#236ab1','995':'#2b8c56','999':'#723d99','95':'#d29422','80':'#c04343'}
 colors=[palette[a.rsplit('_',1)[1]] for a in arms]
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
 fig=plt.figure(figsize=(14,11));grid=fig.add_gridspec(2,3,height_ratios=[1.8,1],hspace=.52)
 for i,(metric,title) in enumerate([('content CE','Content CE (lower is better)'),('Depo gen %','Depo exact generation (%)'),('Brevo valid %','Brevo valid generation (%)')]):
  ax=fig.add_subplot(grid[0,i])
  for y,(arm,color) in enumerate(zip(arms,colors)):
   row=s['arms'][arm][metric];assert row['n']==7
   ax.errorbar(row['mean'],y,xerr=row['sample_sd'],fmt='o',color=color,capsize=3)
  ax.set_yticks(range(len(arms)),labels if i==0 else ['']*len(arms));ax.invert_yaxis();ax.set_title(title,loc='left',fontsize=11)
  ax.grid(axis='x',color='#eeeeee');ax.set_axisbelow(True)
  if i:ax.set_xlim(left=0)
  for y in range(1,len(arms)):
   if arms[y].split('_')[1]!=arms[y-1].split('_')[1]:ax.axhline(y-.5,color='#dddddd',linewidth=.8)
 for i,(family,title) in enumerate([('fifo','FIFO thresholds'),('gate','Gate + selected threshold'),('learned','Learned + selected threshold')]):
  ax=fig.add_subplot(grid[1,i]);subset=[a for a in arms if a.startswith(f'coh_{family}_')]
  for arm in subset:
   idx=arms.index(arm);values=[d[arm][str(l)]['condition_median'] for l in range(4)]
   ax.plot(range(4),values,'o-',label=labels[idx],color=colors[idx])
  ax.set_yscale('log');ax.set_ylim(5,5e5);ax.set_xticks(range(4));ax.set_xlabel('Layer');ax.set_title(title,loc='left',fontsize=11)
  if i==0:ax.set_ylabel('Median Gram condition number')
  ax.grid(color='#eeeeee');ax.legend(fontsize=8,frameon=False)
 fig.suptitle('Coherence admission: quality versus numerical conditioning',x=.03,y=.98,ha='left',fontsize=16,weight='bold')
 fig.text(.03,.025,'Top: 7-seed means ± sample SD, not confidence intervals. All arms: frozen FP32 flex, deterministic algorithms, same 10M tokens.\nBottom: seed731 only; first64 real confirm documents, no padding. Lower conditioning does not establish safe TF32 projection.',fontsize=9,color='#444444')
 fig.subplots_adjust(left=.14,right=.98,top=.91,bottom=.12,wspace=.25)
 for ext in ('png','svg'):fig.savefig(ROOT/f'COMPARISON.{ext}',dpi=180,facecolor='white')
 plt.close(fig)
if __name__=='__main__':main()
