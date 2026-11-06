"""Read-only audit of completed local summaries and saved generation records."""
import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
rows=[]
for p in sorted(root.rglob('summary.json')):
    if 'snapshot' in p.parts:continue
    summary=json.loads(p.read_text());records=json.loads((p.parent/'generations.json').read_text())
    assert summary['status']=='completed' and summary['unique_tokens']==10001108,p
    assert summary['reload_validation_ce_error']<1e-4,p
    for task in ('depo','brevo'):
        subset=[r for r in records if r['task']==task]
        assert len(subset)==128,(p,task,len(subset))
        for r in subset:assert r['exact']==(r['guess']==r['gold']),p
        if task=='depo':assert all(r['valid']==r['exact'] for r in subset),p
        for key in ('exact','valid'):
            actual=sum(r[key] for r in subset)/len(subset)
            assert actual==summary['generation'][task][key],(p,task,key)
    metrics=[json.loads(s) for s in (p.parent/'metrics.jsonl').read_text().splitlines()]
    assert metrics[-1]['step']==2231 and metrics[-1]['unique_tokens']==10001108,p
    assert all(a['step']<b['step'] for a,b in zip(metrics,metrics[1:])),p
    rows.append(dict(path=str(p.relative_to(root)),seed=summary['config']['seed'],arm=summary['config']['architecture'],
                     summary_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),generation_records=len(records)))
assert not list(root.rglob('*.pt')),'Checkpoints must not be copied to deliverables'
receipt=dict(status='passed',completed_runs=len(rows),runs=rows)
(root/'evidence/result-audit.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(dict(status='passed',completed_runs=len(rows))))
