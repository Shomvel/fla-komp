"""Verify observed formal continuation, without treating platform Running as progress."""
import datetime,hashlib,json,math,statistics
from pathlib import Path
r=Path(__file__).resolve().parent.with_name('komp-learned123-1b-50b-20261109-opt1')
p=r/'evidence/training-latest.json';d=json.loads(p.read_text());t=d['parent_transition'];rows=d['metrics_tail']
assert len(d['jobs'])==1 and d['jobs'][0]['status']=='Running' and d['jobs'][0]['ready_status']=='4/4'
assert d['jobs'][0]['user_gpu']==32 and not d['node_exits']
assert set(d['runtime'])=={'0','1','2','3'}
assert d['parent_resume_rank_ids']==list(range(32))
for rec in d['parent_resume_receipts'].values():
 assert rec['step']==t['step'] and rec['tokens']==t['tokens']
 assert rec['strict_model_adam_load'] and rec['rank_input_rng_restored'] and rec['microbatch']==16 and rec['accumulation']==1
assert d['metrics_first']['step']==t['step']+1
assert d['metrics_first']['supervised_tokens']==t['tokens']+1048576
assert not d['nonfinite_metric_steps'] and d['metrics_count']>=35
for row in rows:
 assert row['step']>=t['step']+1 and row['supervised_tokens']==row['step']*1048576
 assert all(math.isfinite(row[k]) for k in ('loss','seconds','grad_norm'))
assert all(b['step']==a['step']+1 for a,b in zip(rows,rows[1:]))
source=d['runs/block/config.json']['source_sha']
assert all(x['source_sha']==source for x in d['runtime'].values())
assert source==hashlib.sha256((r/'source-manifest.json').read_bytes()).hexdigest()
mean=d['steady_mean_seconds_up_to100'];reference=2.857983537297696
out=dict(passed=True,captured_utc=d['captured_utc'],alias=d['jobs'][0]['user_def_name'],identity=d['jobs'][0]['name'],source_sha=source,
 world_size=32,node_ready='4/4',rank_restores=32,parent_step=t['step'],parent_tokens=t['tokens'],first_step=d['metrics_first']['step'],
 latest_step=d['metrics_last']['step'],latest_tokens=d['metrics_last']['supervised_tokens'],target_tokens=50000000000,
 new_steps=d['metrics_count'],seconds_mean=mean,averaging_steps=d['steady_average_count'],reference_seconds=reference,
 formal_speedup=reference/mean,global_supervised_tokens_per_second=1048576/mean,
 remaining_training_hours=(50000000000-d['metrics_last']['supervised_tokens'])/1048576*mean/3600,
 current_loss=d['metrics_last']['loss'],loss_mean100=d['mean_loss_last100'],peak_bytes=d['metrics_last']['peak_bytes'],
 reserved_bytes=d['metrics_last']['peak_reserved_bytes'],validation_last=d['validation_last'],parent_validation_last=d['parent_validation_last'],
 metrics_mtime=d['metrics_mtime'],no_nonfinite_metrics=True,no_node_exits=True,
 verification='Exact parent+1 token continuation and all-rank model/Adam/RNG/input restore; observed finite formal steps; no 50B-completion claim')
(r/'evidence/optimized-resume-acceptance.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
