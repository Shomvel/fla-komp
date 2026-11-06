from pathlib import Path
import json
r=Path(__file__).resolve().parent
v=json.loads((r/'latest.json').read_text())
for task,d in v['tasks'].items():
 for field in ('numerical','precision_ddp'):
  if d[field]:d[field].pop('sources',None)
 for c in d['candidates']:
  c.pop('training_config',None)
  if 'summary' in c:
   c['summary']={k:x for k,x in c['summary'].items() if k!='config'}
p=Path('/Users/shom/.codex/visualizations/2026/11/05/01a1aa28-9556-7a83-bbcd-7957803c2f6c/exact-hyperparameter-screen.html')
s=p.read_text().replace('#exact-screen {color:var(--foreground)}','#exact-screen {color:var(--foreground);position:relative}')
start=s.index('<script type="application/json" id="xs-data">')+len('<script type="application/json" id="xs-data">');end=s.index('</script>',start)
p.write_text(s[:start]+json.dumps(v,ensure_ascii=False)+s[end:])
for task,d in v['tasks'].items():
 for c in d['candidates']:
  if c['status']!='not_started':print(task,c['name'],c['status'],c['metrics'][-1:] or c.get('summary'))
print('failure',v['failure'])
