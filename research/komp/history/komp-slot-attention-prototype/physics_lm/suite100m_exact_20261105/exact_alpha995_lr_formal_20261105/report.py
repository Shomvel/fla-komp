"""Formal100M LR validation ranking with fixed-source and generation audit."""
import hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CENTERS=ROOT.parent/'exact_alpha995_formal_20261105'
sys.path.insert(0,str(ROOT/'snapshot/suite100m'))
from tasks import Factory
plan=json.loads((ROOT/'plan.json').read_text());manifest=json.loads((ROOT/'source-manifest.json').read_text())
for name,wanted in manifest['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==wanted,name
binding=json.loads((ROOT/'evidence/central-reuse.json').read_text());bound={x['task']:x for x in binding['runs']}
rows=[]
for task in plan['tasks']:
    center_lr=plan['reused_lrs'][task];center_name=f'{task}-exact-final-lr{center_lr:g}-wd0.1-seed731'
    center_dir=CENTERS/'runs-ddp/results'/center_name
    reference=json.loads((center_dir/'summary.json').read_text());prior=reference['config']
    assert hashlib.sha256((center_dir/'summary.json').read_bytes()).hexdigest()==bound[task]['summary_sha256']
    for lr in sorted(plan['new_lrs'][task]+[center_lr]):
        name=f'{task}-exact-final-lr{lr:g}-wd0.1-seed731';reused=lr==center_lr;directory=center_dir if reused else ROOT/'runs-ddp/results'/name
        row={'task':task,'lr':lr,'reused':reused,'training_status':'pending','heldout_status':'pending','alpha0':.995,'projection_dtype':'float32','new_positions':0}
        if (directory/'summary.json').exists():
            summary=json.loads((directory/'summary.json').read_text());cfg=summary['config']
            assert summary['status']=='completed' and summary['tokens']==plan['positions_per_run'][task]
            for key in ('task','phase','seed','wd','batch','world_size','steps','model','sources','data_order_sha256'):assert cfg[key]==prior[key],(task,lr,key)
            assert cfg['lr']==lr
            metrics=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
            assert metrics[-1]['step']==cfg['steps'] and metrics[-1]['tokens']==summary['tokens']
            assert all(math.isfinite(m['loss']) and math.isfinite(m['grad_norm']) for m in metrics)
            row.update(training_status='completed',validation_ce=summary['val']['ce'],validation_ce_delta_vs_center=summary['val']['ce']-reference['val']['ce'],new_positions=0 if reused else summary['tokens'],summary_sha256=hashlib.sha256((directory/'summary.json').read_bytes()).hexdigest())
        elif (directory/'metrics.jsonl').exists():
            lines=(directory/'metrics.jsonl').read_text().splitlines()
            row.update(training_status='running',latest=json.loads(lines[-1]) if lines else None)
        if (directory/'eval-progress.json').exists():
            progress=json.loads((directory/'eval-progress.json').read_text());row.update(heldout_status='generation_in_progress',test_ce=progress['test']['ce'],generation_probes=len(progress['records']))
        if (directory/'heldout.json').exists():
            score=json.loads((directory/'heldout.json').read_text());generations=json.loads((directory/'generations.json').read_text());fac=Factory(task)
            actual=[bool(fac.grade(x['probe'],x['guess'])) for x in generations]
            assert len(actual)==256 and all(a==bool(x['valid']) for a,x in zip(actual,generations))
            assert sum(actual)/256==score['generation_accuracy']
            row.update(heldout_status='completed',test_ce=score['test']['ce'],generation_accuracy=score['generation_accuracy'])
        rows.append(row)
complete=sum(not x['reused'] and x['training_status']=='completed' for x in rows)
winners={}
for task in plan['tasks']:
    candidates=[x for x in rows if x['task']==task]
    if all(x['training_status']=='completed' for x in candidates):winners[task]=min(candidates,key=lambda x:x['validation_ce'])['lr']
report={'new_training_complete':complete,'new_heldout_complete':sum(not x['reused'] and x['heldout_status']=='completed' for x in rows),'expected_new':4,'reused_centers':2,'runs':rows,'validation_only_winners':winners,'new_completed_positions':sum(x['new_positions'] for x in rows),'scope':'100MformalFP32projection alpha995 grid; not the10MFP64pilot. One seed and one hyperparameter; rank only by fullvalidationCE.'}
(ROOT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# Alpha .995 formal learning-rate comparison','',f'New training {complete}/4; centers reused 2/2; full new heldout {report["new_heldout_complete"]}/4.','','Every run uses100Mpositions, FP32projection/state, BF16network, alpha.995, seed731 andWD.1. Rank LR only by final validation CE. These results are separate from the10MFP64screen.','','| Task | LR | Source | Validation CE | Test CE | Generation accuracy |','|---|---:|---|---:|---:|---:|']
for row in rows:
 def fmt(k):return f"{row[k]:.6f}" if k in row else 'pending'
 lines.append(f"| {row['task']} | {row['lr']:g} | {'reused100Mcenter' if row['reused'] else row['training_status']} | {fmt('validation_ce')} | {fmt('test_ce')} | {row['generation_accuracy']:.2%} |" if 'generation_accuracy' in row else f"| {row['task']} | {row['lr']:g} | {'reused100Mcenter' if row['reused'] else row['training_status']} | {fmt('validation_ce')} | {fmt('test_ce')} | pending |")
lines+=['',f'Complete-task validation winners: {winners}.']
(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(report))
