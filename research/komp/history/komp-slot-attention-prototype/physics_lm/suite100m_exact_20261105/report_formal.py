"""Summarize actual completed summaries and heldout artifacts, without selecting recipes."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
LABELS={'gdn':'GDN','full':'Full','swa':'SWA','gsa':'GSA','gsa2':'GSA2','komp':'Block-KOMP','exact':'Exact-KOMP'}
records=[]
for task in ('depo2','brevo2'):
    for kind in LABELS:
        paths=[p for directory in ('runs-ddp','runs-gsa2-consistent') for p in (ROOT/directory/'results').glob(f'{task}-{kind}-final-*/summary.json')]
        assert len(paths)<=1,(task,kind)
        row={'task':task,'kind':kind,'training_status':'pending','heldout_status':'pending','implementation':'numerically-stabilized' if kind=='gsa2' else 'original-frozen'}
        if paths:
            path=paths[0];summary=json.loads(path.read_text())
            row.update(training_status=summary['status'],validation_ce=summary['val']['ce'],positions=summary['tokens'],parameters=summary['config']['model']['parameters'],lr=summary['config']['lr'],wd=summary['config']['wd'],source=summary['config']['sources'],summary_path=str(path.relative_to(ROOT)))
            progress=path.with_name('eval-progress.json')
            if progress.exists():
                partial=json.loads(progress.read_text())
                row.update(test_ce=partial['test']['ce'],generation_probes_completed=len(partial.get('records',[])),heldout_status='generation_in_progress')
            if path.with_name('heldout.json').exists():
                result=json.loads(path.with_name('heldout.json').read_text())
                row.update(heldout_status=result['status'],test_ce=result['test']['ce'],generation_accuracy=result['generation_accuracy'],n_generation=result['n_generation'])
        records.append(row)
report={'training_complete':sum(r['training_status']=='completed' for r in records),'heldout_complete':sum(r['heldout_status']=='completed' for r in records),'test_ce_complete':sum('test_ce' in r for r in records),'expected':14,'runs':records,'comparison_limits':['Each architecture retains its separately selected prior LR/WD; hyperparameters are not identical across models.','Original Exact formal alpha0=.98 and FP32 projection differ from the separate alpha0=.995 FP64 10M screen; screen wins are not formal-training results.','One seed (731) and two adapted PhysicsLM tasks; no multi-seed uncertainty estimate or general superiority claim.','GSA2 uses the separately source-bound stabilized implementation; wholebatch Adam functional equivalence remains failed.'],'scope':'~100M parameters, 100M nonpadding next-token positions per model/task, seed731, frozen prior-selected LR/WD. Exact formal alpha0=.98 FP32 projection; separate tuning winners not substituted. GSA2 candidate runtime is separately identified; wholebatch Adam equivalence remains failed.'}
(ROOT/'formal-results.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# PhysicsLM 100M formal comparison','',f"Training complete: {report['training_complete']}/14. Heldout complete: {report['heldout_complete']}/14.",'','Test CE appears after the full heldout split is scored. Generation accuracy appears only after all 256 prompt-only probes finish; a CE result alone is not counted as a completed heldout evaluator. Original selected recipes and budgets are retained. Exact uses alpha0=.98 and FP32 projection; the separate 10M tuning winners have not replaced its formal recipe. GSA2 uses the separately bound numerical-stability runtime with fresh 8GPU distributed acceptance; its strict wholebatch-Adam witness remains failed.','']
for task in ('depo2','brevo2'):
    lines += [f'## {task}','','| Model | Validation CE | Heldout CE | Generation accuracy |','|---|---:|---:|---:|']
    for row in records:
        if row['task']!=task:continue
        val=f"{row['validation_ce']:.6f}" if 'validation_ce' in row else 'pending'
        test=f"{row['test_ce']:.6f}" if 'test_ce' in row else 'pending'
        accuracy=f"{row['generation_accuracy']:.2%}" if 'generation_accuracy' in row else f"pending ({row.get('generation_probes_completed',0)}/256)"
        lines.append(f"| {LABELS[row['kind']]} | {val} | {test} | {accuracy} |")
    lines.append('')
lines += ['## Comparison scope','','Each architecture retains its own prior validation-selected LR/WD. These are one-seed results on two adapted PhysicsLM tasks. Exact formal alpha0=.98/FP32 projection differs from the separate alpha0=.995/FP64 short-run screen; that screen does not establish the latter recipe at100Mpositions. GSA2 uses a separately source-bound stable runtime, with the original wholebatch-Adam mismatch retained as a limitation.','','| Task | Model | LR | Weight decay |','|---|---|---:|---:|']
for row in records:
    if 'lr' in row:lines.append(f"| {row['task']} | {LABELS[row['kind']]} | {row['lr']:g} | {row['wd']:g} |")
lines.append('')
(ROOT/'FORMAL-RESULTS.md').write_text('\n'.join(lines))
print(json.dumps({'training_complete':report['training_complete'],'heldout_complete':report['heldout_complete']}))
