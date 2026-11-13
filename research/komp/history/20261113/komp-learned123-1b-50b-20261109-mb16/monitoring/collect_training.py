"""Read-only remote training collection; local snapshots and progress report."""
import datetime,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REMOTE=r'''
from pathlib import Path
import datetime,json,statistics,shutil
r=Path('/gpfs/users/linlizhi/komp-learned123-1b-50b-20261109-mb16')
out={'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'remote_root':str(r),'free_bytes':shutil.disk_usage(r).free}
for name in ['runs/block/config.json','runs/block/latest.json','runs/block/complete.json','runs/block/paused.json','evidence/preflight-block.json']:
 p=r/name
 if p.exists():
  x=json.loads(p.read_text());out[name]=x if 'preflight' not in name else {k:x.get(k) for k in ['status','source_sha','parameters','full_model_adam_rng_loader_restore']}
p=r/'runs/block/metrics.jsonl'
rows=[]
if p.exists():
 for line in p.read_text().splitlines():
  try:rows.append(json.loads(line))
  except json.JSONDecodeError:out['trailing_partial_metric_line']=True
if rows:
 out['metrics_last']=rows[-1];out['metrics_tail']=rows[-10:];out['metrics_mtime']=datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat()
 out['mean_step_seconds_last100']=statistics.mean(x['seconds'] for x in rows[-100:])
 out['mean_loss_last100']=statistics.mean(x['loss'] for x in rows[-100:])
p=r/'runs/block/validation.jsonl';out['validation_last']=json.loads(p.read_text().splitlines()[-1]) if p.exists() and p.stat().st_size else None
out['restore_passed']=sum(json.loads(p.read_text()).get('status')=='passed' for p in (r/'evidence').glob('restore-rank*.json'))
out['parent_transition']=json.loads((r/'configs/resume-parent.json').read_text())
out['parent_resume_passed']=sum(json.loads(p.read_text()).get('status')=='passed' for p in (r/'evidence').glob('parent-resume-rank*.json'))
out['checkpoints']=[]
for p in sorted((r/'runs/block').glob('checkpoint-*/complete.json')):
 d=json.loads(p.read_text());d['path']=str(p.parent);d['model_bytes']=(p.parent/'model.pt').stat().st_size if (p.parent/'model.pt').exists() else None;d['rank_files']=sum((p.parent/f'rank{rank}.pt').exists() for rank in range(32));out['checkpoints'].append(d)
out['node_exits']={p.parent.name:p.read_text() for p in r.glob('nodes/*/exit.txt')}
out['worker_tail']={p.parent.name:p.read_text(errors='replace')[-1800:] for p in r.glob('nodes/*/worker.log')}
out['runtime']={p.parent.name:json.loads(p.read_text()) for p in r.glob('nodes/*/runtime.json')}
print(json.dumps(out))
'''
def main():
 result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','linlizhi-bp-dev','python3','-'],input=REMOTE,text=True,capture_output=True,check=True,timeout=120)
 current=json.loads(result.stdout)
 helper='/Users/shom/.codex/skills/northjob-gpu-test-bp-dev/scripts/northjob.py'
 jobs=subprocess.run(['python3',helper,'jobs','--alias','komp-learned123-1b-50b-20261109-mb16'],text=True,capture_output=True,check=True,timeout=120)
 current['jobs']=json.loads(jobs.stdout)
 assert len(current['jobs'])==1 and current['jobs'][0]['name']=='job-b054fc9711a2-20261109003121','Unexpected job identity'
 p=ROOT/'evidence/training-latest.json';previous=json.loads(p.read_text()) if p.exists() else {}
 now=current.get('metrics_last',{});old=previous.get('metrics_last',{})
 phase='training' if now else ('restoring' if current['parent_resume_passed'] else 'preflight')
 if 'runs/block/complete.json' in current:phase='completed'
 if 'runs/block/paused.json' in current:phase='paused'
 current['delta_steps']=now.get('step',0)-old.get('step',0) if old else None
 current['delta_tokens']=now.get('supervised_tokens',0)-old.get('supervised_tokens',0) if old else None
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 folder=ROOT/'evidence/training-monitor';folder.mkdir(exist_ok=True)
 if previous:(folder/f'previous-{stamp}.json').write_text(json.dumps(previous,indent=2)+'\n')
 (folder/f'{stamp}.json').write_text(json.dumps(current,indent=2)+'\n')
 tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps(current,indent=2)+'\n');tmp.replace(p)
 seconds=current.get('mean_step_seconds_last100',0);tokens=now.get('supervised_tokens',current['parent_transition']['tokens'])
 summary=dict(captured_utc=current['captured_utc'],phase=phase,jobs=current['jobs'],step=now.get('step',current['parent_transition']['step']),tokens=tokens,target=50_000_000_000,percent=tokens/50_000_000_000*100,loss=now.get('loss'),loss_mean100=current.get('mean_loss_last100'),step_seconds_mean100=seconds,global_tokens_per_second=1048576/seconds if seconds else None,delta_steps=current['delta_steps'],delta_tokens=current['delta_tokens'],validation_last=current['validation_last'],checkpoints=current['checkpoints'],node_exits=current['node_exits'],restore_passed=current['restore_passed'],parent_resume_passed=current['parent_resume_passed'],resume_parent_step=current['parent_transition']['step'],resume_parent_tokens=current['parent_transition']['tokens'])
 with (ROOT/'PROGRESS.md').open('a') as f:f.write('\n训练观测：'+json.dumps(summary,ensure_ascii=False)+'\n')
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
