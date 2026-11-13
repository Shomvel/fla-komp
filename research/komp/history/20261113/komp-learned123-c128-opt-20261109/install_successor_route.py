"""Keep the existing hourly collector authoritative if app automation transport is unavailable."""
import datetime,json,shutil
from pathlib import Path
r=Path(__file__).resolve().parent
old=r.with_name('komp-learned123-1b-50b-20261109-mb16-direct')
new=r.with_name('komp-learned123-1b-50b-20261109-opt1')
d=json.loads((new/'evidence/training-latest.json').read_text())
assert len(d['jobs'])==1 and d['jobs'][0]['user_def_name']==new.name
assert d['parent_resume_rank_ids']==list(range(32))
assert d['metrics_first']['step']==d['parent_transition']['step']+1
assert d['metrics_last']['supervised_tokens']>d['parent_transition']['tokens']
assert not d['node_exits'] and not d.get('nonfinite_metric_steps')
marker=dict(local_root=str(new),remote_root=d['remote_root'],alias=new.name,identity=d['jobs'][0]['name'],
 source_sha=d['runs/block/config.json']['source_sha'],parent_step=d['parent_transition']['step'],
 parent_tokens=d['parent_transition']['tokens'],verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 reason='User-authorized optimized continuation; current hourly entry forwards to successor, never reports parent paused/Completed as 50B finished')
p=old/'monitoring/collect_training.py'
backup=old/'evidence/collect_training-before-opt1.py'
assert not backup.exists() and not (old/'successor-run.json').exists()
shutil.copy2(p,backup)
s=p.read_text().replace('import datetime,json,subprocess','import datetime,json,subprocess,shutil');needle='def main():\n';assert s.count(needle)==1
forward='''def main():
 successor_file=ROOT/'successor-run.json'
 if successor_file.exists():
  successor=json.loads(successor_file.read_text())
  target=Path(successor['local_root'])
  assert target==ROOT.with_name('komp-learned123-1b-50b-20261109-opt1')
  result=subprocess.run(['python3',str(target/'monitoring/collect_training.py')],text=True,capture_output=True,check=True,timeout=300)
  current=json.loads((target/'evidence/training-latest.json').read_text())
  assert current['jobs'][0]['name']==successor['identity'],'Successor identity changed'
  current['forwarded_from_local_root']=str(ROOT)
  current['successor_descriptor']=successor
  stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
  prior=ROOT/'evidence/training-latest.json'
  if prior.exists():shutil.copy2(prior,ROOT/f'evidence/training-monitor/before-successor-{stamp}.json')
  tmp=prior.with_suffix('.json.tmp');tmp.write_text(json.dumps(current,indent=2)+'\\n');tmp.replace(prior)
  with (ROOT/'PROGRESS.md').open('a') as f:f.write('\\n当前训练已由完整checkpoint接续到 '+str(target)+'；本采集入口已转发到新作业 '+successor['identity']+'。父作业Completed仅表示安全暂停。\\n')
  print(result.stdout,end='')
  return
'''
s=s.replace(needle,forward)
compile(s,str(p),'exec')
(old/'successor-run.json').write_text(json.dumps(marker,indent=2)+'\n')
p.write_text(s)
print(json.dumps(marker,indent=2))
