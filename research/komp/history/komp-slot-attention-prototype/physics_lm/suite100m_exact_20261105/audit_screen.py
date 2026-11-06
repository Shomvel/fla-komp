"""Audit comparable completed short runs using source/config/metrics evidence."""
import json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent;T=ROOT.parent/'exact_tuning_20261105'
plan=json.loads((T/'initial-screen.json').read_text());report={'status':'passed','scope':'18 independent10M-position short-budget runs; FP64geometry/FP32state, BF16 network','tasks':{},'total_training_positions':0}
for task in ['depo2','brevo2']:
 rows=[];orders=set();sources=set();budgets=set();lrs=set()
 for candidate in plan['screen']:
  dest=T/'results'/task/candidate['name'];summary=json.loads((dest/'summary.json').read_text());config=summary['config']
  assert summary['status']=='completed' and config['candidate']==candidate
  assert config['batch']==128 and config['world_size']==8 and config['seed']==731
  assert config['candidate']['projection_dtype']=='float64'
  assert summary['tokens']==config['train_positions'] and 10_000_000<=summary['tokens']<10_002_000
  records=[json.loads(line) for line in (dest/'metrics.jsonl').read_text().splitlines()]
  assert records[-1]['step']==config['steps'] and records[-1]['tokens']==summary['tokens']
  assert all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) for r in records)
  validations=[r for r in records if 'val' in r];assert len(validations)==5
  assert all(math.isfinite(r['val']['ce']) for r in validations)
  assert validations[-1]['val']==summary['val']
  orders.add(config['data_order_sha256']);sources.add(json.dumps(config['sources'],sort_keys=True));budgets.add(summary['tokens']);lrs.add((config['lr'],config['wd']))
  row={'candidate':candidate['name'],'final_val_ce':summary['val']['ce'],'tokens':summary['tokens'],'seconds_including_compile_and_val':summary['elapsed_seconds'],'max_logged_grad_norm':max(r['grad_norm'] for r in records),'last_logged_grad_norm':records[-1]['grad_norm'],'peak_bytes_rank0':summary['peak_bytes_rank0'],'validation_points':5,'finite_logged_metrics':True,'last_validation_supervised_positions':summary['val']['supervised_positions']}
  rows.append(row);report['total_training_positions']+=summary['tokens']
 assert len(orders)==len(sources)==len(budgets)==len(lrs)==1
 baseline=next(x for x in rows if x['candidate']=='baseline')['final_val_ce']
 for row in rows:row.update(delta_ce_vs_baseline=row['final_val_ce']-baseline,relative_ce_change=(row['final_val_ce']/baseline-1))
 rows.sort(key=lambda x:x['final_val_ce'])
 report['tasks'][task]={'ranking':rows,'winner':rows[0]['candidate'],'data_order_sha256':orders.pop(),'tokens_each':budgets.pop(),'lr_wd':list(lrs.pop()),'total_run_seconds':sum(x['seconds_including_compile_and_val'] for x in rows)}
(ROOT/'evidence/screen-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
