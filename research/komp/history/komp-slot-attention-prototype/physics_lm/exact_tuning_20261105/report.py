"""Summarize observed results only; rank completed runs by final validation CE."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
plan=json.loads((ROOT/'initial-screen.json').read_text())
report={'selection':'final validation CE at equal 10M training budget; provisional at 100M', 'tasks':{}}
lines=['# Exact 初始超参筛选','', '每任务9候选，各10M token，seed731。以下只列已观测结果。','']
for task in ('depo2','brevo2'):
    rows=[]
    for candidate in plan['screen']:
        dest=ROOT/'results'/task/candidate['name']
        row={'candidate':candidate['name'],'status':'not_started'}
        if (dest/'summary.json').exists():
            value=json.loads((dest/'summary.json').read_text())
            row.update(status=value['status'],ce=value['val']['ce'],tokens=value['tokens'],seconds=value['elapsed_seconds'])
        elif (dest/'metrics.jsonl').exists():
            entries=[]
            for line in (dest/'metrics.jsonl').read_text().splitlines():
                try: entries.append(json.loads(line))
                except json.JSONDecodeError: pass
            if entries:
                row.update(status='started',last_observed=entries[-1])
        rows.append(row)
    finished=sorted([r for r in rows if r['status']=='completed'],key=lambda r:r['ce'])
    baseline=next((r for r in finished if r['candidate']=='baseline'),None)
    if baseline:
        for r in finished:r['delta_ce_vs_baseline']=r['ce']-baseline['ce']
    report['tasks'][task]={'candidates':rows,'ranking':finished,'screen_complete':len(finished)==len(rows)}
    lines.extend([f'## {task}','','| 候选 | 状态 | 验证CE | 相对baseline |','|---|---|---:|---:|'])
    for r in rows:
        ce=f"{r['ce']:.6f}" if 'ce' in r else '—'
        delta=f"{r['delta_ce_vs_baseline']:+.6f}" if 'delta_ce_vs_baseline' in r else '—'
        lines.append(f"| {r['candidate']} | {r['status']} | {ce} | {delta} |")
    lines.append('')
(ROOT/'results.json').write_text(json.dumps(report,indent=2))
(ROOT/'RESULTS.md').write_text('\n'.join(lines))
print(json.dumps({t:{'completed':len(v['ranking']),'total':len(v['candidates'])} for t,v in report['tasks'].items()}))
