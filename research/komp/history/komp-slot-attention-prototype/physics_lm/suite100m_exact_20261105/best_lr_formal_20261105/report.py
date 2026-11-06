"""Describe validation-selected existing formal checkpoints without changing baseline."""
import json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;SUITE=ROOT.parent
sys.path.insert(0,str(SUITE/'snapshot/suite100m'))
from tasks import Factory
manifest=json.loads((ROOT/'manifest.json').read_text())
rows=[]
for task,entry in manifest['tasks'].items():
    if entry['status']!='selected_formal_checkpoint':continue
    selected=Path(entry['selected_directory'])
    # Manifest paths bind GPFS; local analysis reads the corresponding synced path.
    if not selected.exists():selected=SUITE/'exact_alpha995_lr_formal_20261105/runs-ddp/results'/selected.name
    summary=json.loads((selected/'summary.json').read_text());assert hashlib.sha256((selected/'summary.json').read_bytes()).hexdigest()==entry['selected_summary_sha256']
    row={'task':task,'label':'Tuned Exact','alpha0':.995,'lr':entry['lr'],'wd':.1,'validation_ce':entry['selected_validation_ce'],'selection':'validation CE only','source':entry['sources'],'checkpoint_sha256':entry['checkpoint_sha256'],'heldout_status':'pending'}
    if (selected/'heldout.json').exists():
        result=json.loads((selected/'heldout.json').read_text());records=json.loads((selected/'generations.json').read_text());fac=Factory(task);scores=[bool(fac.grade(x['probe'],x['guess'])) for x in records]
        assert len(scores)==256 and sum(scores)/256==result['generation_accuracy'] and all(a==bool(x['valid']) for a,x in zip(scores,records))
        row.update(heldout_status='completed',test_ce=result['test']['ce'],generation_accuracy=result['generation_accuracy'],correct=sum(scores))
    elif (selected/'eval-progress.json').exists():
        progress=json.loads((selected/'eval-progress.json').read_text());row.update(heldout_status='generation_in_progress',test_ce=progress['test']['ce'],generation_probes=len(progress['records']))
    rows.append(row)
report={'status':'complete' if len(rows)==2 and all(x['heldout_status']=='completed' for x in rows) else 'pending_remaining_evaluation_or_selection','new_training_positions':0,'action':'Reuse exactsame-seed completed100Mformal best-LR checkpoints; no duplicate run','rows':rows,'manifest_sha256':hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest(),'baseline_table_unchanged':True,'scope':'Tuned Exact is an additional tuning-selected comparison; original14frozen recipes remain independent. One seed; test/generation not used to select rates.'}
(ROOT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# Validation-selected tuned Exact formal results','','Existing100Mformal checkpoints selected only by final validation CE. No duplicate retraining or additional positions. Original frozen14-model/task table remains separate.','','| Task | alpha | LR | Validation CE | Test CE | Generation |','|---|---:|---:|---:|---:|---:|']
for row in rows:
 test=f"{row['test_ce']:.6f}" if 'test_ce' in row else 'pending';gen=f"{row['correct']}/256 ({row['generation_accuracy']:.2%})" if 'correct' in row else 'pending'
 lines.append(f"| {row['task']} | .995 | {row['lr']:g} | {row['validation_ce']:.6f} | {test} | {gen} |")
lines+=['','All candidates have the same100Mbudget, FP32projection/state, BF16network, WD.1, seed731 and originaldataorder. Originalalpha.98 and alpha-onlycenter checkpoints are bound in manifest.json. These are tuning-selected results, not identically tuned comparisons against all other architectures.']
(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(report))
