"""Read-only remote evaluation collection, with local status/report output."""
import json, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REMOTE='/gpfs/users/linlizhi/'+ROOT.name
code=f"""import sys,runpy,json,datetime
from pathlib import Path
r=Path({REMOTE!r});sys.path.insert(0,str(r))
if (r/'sharded/plan.json').exists():
 runpy.run_path(str(r/'collect_sharded_remote.py'),run_name='__main__')
else:
 states={{p.name:json.loads(p.read_text()) for p in (r/'watcher').glob('*.json')}}
 print(json.dumps(dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha=None,states={{}},summaries=[],progress=[],watcher=states)))
"""
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','linlizhi-bp-dev','python3','-'],input=code,text=True,capture_output=True,check=True,timeout=60)
a=json.loads(r.stdout)
helper='/Users/shom/.codex/skills/northjob-gpu-test-bp-dev/scripts/northjob.py'
j=subprocess.run(['python3',helper,'jobs','--alias','komp1b-downstream-eval-20261110'],text=True,capture_output=True,check=True,timeout=60)
a['jobs']=json.loads(j.stdout)
(ROOT/'evidence/latest.json').write_text(json.dumps(a,indent=2)+'\n')
done=sum(v['count'] for v in a['progress']);total=sum(v['total'] for v in a['progress']) or 20457
lines=['# Learned+123 1B 下游评测','',f"采集时间 UTC: {a['captured_utc']}",f'进度: {done}/{total} 条，{len(a["summaries"])}/23 完整分项。','',
       '独立 8 H100 / 8 分片，WeaverRSI / proj-32。具体 checkpoint 和 token 预算固定于 evidence/checkpoint.json。',
       '沿用历史同一批 20457 条请求、相同 batch 形状与生成/评分函数。NIAH single1/2/3 × 1024/2048/4096/8192；PhoneBook 7 档；SWDE/FDA/NQ/DROP。',
       'PhoneBook 为历史 single_break，prompt 约80–244 token；BASED cap2048。FP32 权重+BF16 autocast，全前缀重算，无 cache。','', '| 分项 | 进度 | 完整分数 |','|---|---:|---:|']
for v in a['progress']:
    score=f"{v['score']*100:.2f}%" if v['complete'] else '—'
    lines.append(f"| {v['case']} | {v['count']}/{v['total']} | {score} |")
if a.get('watcher'):lines+=['', '自动衔接状态：'+a['watcher'].get('status.json',{}).get('status','unknown')]
lines+=['','作业:']
for job in a['jobs']: lines.append(f"- {job['name']}: {job['status']} {job['ready_status']}")
failures={k:v for k,v in a['states'].items() if k.endswith('-failure.json')}
for k,v in failures.items(): lines.append(f"- FAILED {k}: {v.get('error')}")
(ROOT/'PROGRESS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(captured_utc=a['captured_utc'],done=done,total=total,complete_cases=len(a['summaries']),
                     acceptances=sum(k.endswith('-acceptance.json') for k in a['states']),failures=failures,jobs=a['jobs'],watcher=a.get('watcher',{})),ensure_ascii=False,indent=2))
