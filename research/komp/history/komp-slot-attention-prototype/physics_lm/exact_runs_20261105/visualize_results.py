"""Create a concise static comparison and populate the inline visualization."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
s=json.loads((root/'statistics.json').read_text())
labels={'exact_scalar_fifo':'Exact scalar FIFO','exact_slot_fifo':'Exact slot FIFO','exact_slot_gate':'Exact gate-only','exact_slot_learned':'Exact learned','block_c64':'Block scalar FIFO','block_slot_c64':'Block slot FIFO','block_slot_gate_c64':'Block gate-only','block_slot_learn_c64':'Block learned','gdn':'GDN','gsa':'GSA','gsa_slot':'Old token KOMP','gsa2_native_lr5e4':'GSA2 · 5e-4','gsa2_native_lr1e3':'GSA2 · 1e-3','gsa2_noconv_lr5e4':'GSA2 no conv'}
core=['exact_scalar_fifo','exact_slot_fifo','exact_slot_gate','exact_slot_learned','block_slot_c64','block_slot_gate_c64','block_slot_learn_c64','gdn','gsa2_native_lr1e3']
matched=[*core[:4],'block_c64',*core[4:7]]
metrics=['Content CE','Depo generation %','Brevo valid %']
fig,axes=plt.subplots(1,3,figsize=(13.5,6.3),sharey=True,layout='constrained')
for ax,metric,title in zip(axes,metrics,['Content CE ↓','Depo generation accuracy ↑','Brevo valid generation ↑']):
 for i,arm in enumerate(core):
  v=s['arms'][arm][metric];color='#2563eb' if arm.startswith('exact_') else '#0d9488' if arm.startswith('block_') else '#6b7280'
  ax.errorbar(v['mean'],i,xerr=v['sd'],fmt='o' if arm.startswith('exact_') else 's',color=color,markersize=6,capsize=3,lw=1.6)
  ax.annotate(f"{v['mean']:.3f}" if metric=='Content CE' else f"{v['mean']:.2f}%",(v['mean']+v['sd'],i),xytext=(7,0),textcoords='offset points',va='center',fontsize=10,color='#273343')
 lo=min(s['arms'][a][metric]['mean']-s['arms'][a][metric]['sd'] for a in core)
 hi=max(s['arms'][a][metric]['mean']+s['arms'][a][metric]['sd'] for a in core)
 ax.set_xlim(max(0,lo-(hi-lo)*.12),hi+(hi-lo)*.43)
 ax.set_title(title,loc='left',fontsize=12,pad=16)
 ax.set_xlabel('Cross entropy' if metric=='Content CE' else 'Percent')
 ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True)
 ax.spines[['top','right','left']].set_visible(False)
 ax.tick_params(axis='y',length=0)
axes[0].set_yticks(range(len(core)),[labels[a]+(' [new]' if a.startswith('exact_') else ' [prior]') for a in core]);axes[0].invert_yaxis()
fig.suptitle('Exact per-token KOMP: quality did not exceed Block-KOMP',fontsize=16,x=.02,ha='left')
fig.supxlabel('~9M parameters · 10M training tokens · 7 seeds · whiskers: sample SD\nFixed 128 generations/task. GSA2 uses its prior tuned LR, native convolution and larger state.',fontsize=10,color='#4b5563')
fig.savefig(root/'comparison.png',dpi=190);fig.savefig(root/'comparison.svg');plt.close(fig)
path=Path('/Users/shom/.codex/visualizations/2026/11/05/01a1aa28-9556-7a83-bbcd-7957803c2f6c/komp-physics-comparison.html')
training={'updated':'2026-11-05','tasks':[{'name':t,'status':'8 卡运行中 · DDP 预检','models':[{'name':m,'status':'预检 / 待训练'} for m in ['GDN','Full','SWA','GSA','GSA2','Block-KOMP','Exact-KOMP']]} for t in ['Depo2','Brevo2']]}
data={'arms':s['arms'],'paired':s['paired'],'labels':labels,'orders':{'core':core,'matched':matched,'all':list(s['arms'])},'training':training}
text=path.read_text();token='__KOMP_DATA__'
if token in text:text=text.replace(token,json.dumps(data,ensure_ascii=False))
else:
 start=text.index('<script type="application/json" id="kp-data">')+len('<script type="application/json" id="kp-data">');end=text.index('</script>',start)
 old=json.loads(text[start:end]);data['training']=old['training'];text=text[:start]+json.dumps(data,ensure_ascii=False)+text[end:]
path.write_text(text)
print(path)
