"""Refresh the inline training snapshot from collected, read-only worker evidence."""
from pathlib import Path
import json
from datetime import datetime
from zoneinfo import ZoneInfo
root=Path(__file__).resolve().parent
progress=json.loads((root.parent/'suite100m_exact_20261105/evidence/ddp-progress.json').read_text())
path=Path('/Users/shom/.codex/visualizations/2026/11/05/01a1aa28-9556-7a83-bbcd-7957803c2f6c/komp-physics-comparison.html')
s=path.read_text();prefix='<script type="application/json" id="kp-data">';start=s.index(prefix)+len(prefix);end=s.index('</script>',start);d=json.loads(s[start:end])
models=[('gdn','GDN'),('full','Full'),('swa','SWA'),('gsa','GSA'),('gsa2','GSA2'),('komp','Block-KOMP'),('exact','Exact-KOMP')]
passed=sum(r['status']=='passed' for r in progress['preflight'].values())
tasks=[]
for task,label in [('depo2','Depo2'),('brevo2','Brevo2')]:
 rows=[];active=False
 for kind,name in models:
  prefix=f'{task}-{kind}-'
  training=next((r for r in progress['training'] if r['name'].startswith(prefix)),None)
  finished=next((r for r in progress['finished'] if r['name'].startswith(prefix)),None)
  if finished: status='训练完成'
  elif training and training['latest']:
   v=training['latest'];status=f"训练 {v['step']:,} / {v['total']:,} 步";active=True
  elif kind in progress['preflight']: status='预检通过 · 待训练'
  else: status='DDP 预检中 / 待预检'
  rows.append({'name':name,'status':status})
 state='8 卡训练中' if active else f'DDP 预检 {passed} / 7 通过'
 if progress['failure']:state='已暂停 · 预检或训练失败待诊断'
 tasks.append({'name':label,'status':state,'models':rows})
d['training']={'updated':datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M CST'),'tasks':tasks}
path.write_text(s[:start]+json.dumps(d,ensure_ascii=False)+s[end:])
(root/'evidence/visual-training-snapshot.json').write_text(json.dumps(d['training'],ensure_ascii=False,indent=2))
print(json.dumps(d['training'],ensure_ascii=False))
