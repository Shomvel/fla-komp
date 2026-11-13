import subprocess,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
code="import runpy;runpy.run_path('/gpfs/users/linlizhi/komp-flame-downstream-eval-20261107/collect_sharded_remote.py',run_name='__main__')"
r=subprocess.run(['ssh','linlizhi-bp-dev','python3','-'],input="import sys;sys.path.insert(0,'/gpfs/users/linlizhi/komp-flame-downstream-eval-20261107')\n"+code,text=True,capture_output=True,check=True)
a=json.loads(r.stdout)
helper='/Users/shom/.codex/skills/northjob-gpu-test-bp-dev/scripts/northjob.py'
job_result=subprocess.run(['python3',helper,'jobs','--prefix','komp340m-downstream-eval-'],text=True,capture_output=True)
if job_result.returncode==0:
 all_jobs=json.loads(job_result.stdout)
 active_aliases={f'komp340m-downstream-eval-20261108-r4g-s{s}' for s in range(3)}
 a['jobs']=[j for j in all_jobs if j.get('user_def_name') in active_aliases]
 a['historical_jobs']=[j for j in all_jobs if j.get('user_def_name') not in active_aliases]
else:a['jobs_error']=job_result.stderr

(ROOT/'evidence/latest.json').write_text(json.dumps(a,indent=2)+'\n')
for job in a.get('jobs',[]):print('JOB',json.dumps(job))
for name,v in a['states'].items():
 if name.endswith(('-progress.json','-failure.json','-complete.json')):print(name,json.dumps({k:x for k,x in v.items() if k not in ['receipt','traceback','cases']}))

for v in a['summaries']:print('COMPLETE',v['model'],v['case'],v['count'],v['score'])

models=['kda','block','block123','exact123','learned','learned123','value','exact_partial']
cases=sorted({v['case'] for v in a['summaries']})
lookup={(v['model'],v['case']):v for v in a['summaries']}
lines=['# KOMP / KDA 下游评测：进行中','','采集时间（UTC）：'+a['captured_utc'],'','Northjob r4g-s0/s1/s2，各8 H100，评测上限24 GPU，每模型3路固定batch分片。r3主动迁移已停止，41,606条完整样本已保留。','', '只填写完整分项分数；未完成留空。PhoneBook 每档1000条。','', '| 模型 | '+' | '.join(cases)+' |','|---|'+ '|'.join(['---:']*len(cases))+'|']
for model in models:
 label=model+('（3.758B partial）' if model=='exact_partial' else '（6.8B final）')
 lines.append('| '+label+' | '+' | '.join(f"{lookup[model,case]['score']*100:.1f}%" if (model,case) in lookup else '—' for case in cases)+' |')
lines+=['','全量范围：NIAH single1/2/3 × 1024/2048/4096/8192；PhoneBook32/64/100/128/150/200/256；SWDE/FDA/NQ/DROP，20457条/模型。','', 'PhoneBook 使用历史 single_break 数据，实际 prompt 约80–244 token，不能代表长上下文能力。BASED 按原 HFLM config cap2048 左截断至2000；NIAH cap65536、PhoneBook cap32768。','', '训练原生 FP32 参数/门控与 BF16 autocast，无 padding、无 cache，和历史模型 BF16 加载数值实现不同，详见 README。原 Exact partial 不与最终6.8B等价排序。','', '8模型已通过加载、运行配置、真实prompt微生成验收。全量运行尚未完成，单项不能代表整体能力。重新采集：python3 collect.py。']
lines+=['','当前作业状态：']
for job in a.get('jobs',[]):lines.append(f"- {job['user_def_name']}: {job['status']} {job['ready_status']} ({job['name']})")
lines+=['','每worker最新进度：']
for name,worker in sorted(a['states'].items()):
 if name.endswith('-progress.json'):lines.append(f"- {worker['model']} shard{worker['shard']}: {worker['case']} / {worker['batch_key']}")
 if name.endswith('-failure.json'):lines.append(f"- FAILED {name}: {worker.get('error')}")
(ROOT/'PROGRESS.md').write_text('\n'.join(lines)+'\n')
