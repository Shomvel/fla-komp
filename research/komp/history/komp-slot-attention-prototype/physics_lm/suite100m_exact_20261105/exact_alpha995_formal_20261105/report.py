"""Audit and compare only the single alpha change against original formal runs."""
import hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent
sys.path.insert(0,str(ROOT/'snapshot/suite100m'))
from tasks import Factory
manifest=json.loads((ROOT/'source-manifest.json').read_text())
for name,wanted in manifest['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==wanted,name
plan=json.loads((ROOT/'plan.json').read_text())
records=[]
for task in ('depo2','brevo2'):
    lr=plan['lr'][task];name=f'{task}-exact-final-lr{lr:g}-wd0.1-seed731'
    old=PARENT/'runs-ddp/results'/name
    reference=json.loads((old/'summary.json').read_text())
    path=ROOT/'runs-ddp/results'/name
    row={'task':task,'alpha0':.995,'reference_alpha0':.98,'training_status':'pending','heldout_status':'pending','baseline_validation_ce':reference['val']['ce']}
    if (old/'heldout.json').exists():
        result=json.loads((old/'heldout.json').read_text());row.update(baseline_test_ce=result['test']['ce'],baseline_generation_accuracy=result['generation_accuracy'])
    if (path/'summary.json').exists():
        value=json.loads((path/'summary.json').read_text());cfg=value['config'];prior=reference['config']
        assert value['status']=='completed' and value['tokens']==reference['tokens']==plan['positions_per_task'][task]
        for key in ('task','phase','seed','lr','wd','batch','world_size','steps','local_microbatch_token_cap','data_order_sha256'):assert cfg[key]==prior[key],(task,key)
        assert cfg['model']['alpha0']==.995
        assert all(cfg['model'][k]==v for k,v in prior['model'].items())
        assert cfg['sources']=={'snapshot':hashlib.sha256((ROOT/'evidence/snapshot-sha256.json').read_bytes()).hexdigest(),'ddp_train':manifest['files']['ddp_train.py'],'alpha_models':manifest['files']['alpha_models.py']}
        metrics=[json.loads(line) for line in (path/'metrics.jsonl').read_text().splitlines()]
        assert metrics[-1]['step']==cfg['steps'] and metrics[-1]['tokens']==value['tokens']
        assert all(math.isfinite(x['loss']) and math.isfinite(x['grad_norm']) for x in metrics)
        row.update(training_status='completed',validation_ce=value['val']['ce'],validation_ce_delta=value['val']['ce']-reference['val']['ce'],positions=value['tokens'],source=cfg['sources'],summary_sha256=hashlib.sha256((path/'summary.json').read_bytes()).hexdigest())
    if (path/'eval-progress.json').exists():
        value=json.loads((path/'eval-progress.json').read_text());row.update(test_ce=value['test']['ce'],generation_probes=len(value['records']),heldout_status='generation_in_progress')
    if (path/'heldout.json').exists():
        value=json.loads((path/'heldout.json').read_text());generations=json.loads((path/'generations.json').read_text());fac=Factory(task)
        actual=[bool(fac.grade(r['probe'],r['guess'])) for r in generations]
        assert len(actual)==256 and all(a==bool(r['valid']) for a,r in zip(actual,generations))
        assert sum(actual)/256==value['generation_accuracy']
        row.update(heldout_status='completed',test_ce=value['test']['ce'],test_ce_delta=value['test']['ce']-row['baseline_test_ce'],generation_accuracy=value['generation_accuracy'],generation_accuracy_delta=value['generation_accuracy']-row['baseline_generation_accuracy'])
    records.append(row)
report={'status':'audited_completed_artifacts','training_complete':sum(r['training_status']=='completed' for r in records),'heldout_complete':sum(r['heldout_status']=='completed' for r in records),'expected':2,'runs':records,'scope':'Single-factor alpha=.995 vs originalalpha=.98; both FP32projection/state and BF16network, original per-taskLR/WD/seed731/order/100Mbudget. No LR2x substitution. Original FP32stress limitation retained.'}
(ROOT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# Exact-KOMP alpha .995 single-factor comparison','',f"Training {report['training_complete']}/2; complete heldout {report['heldout_complete']}/2.",'','Only initial alpha changed; original FP32projection/state, BF16network, per-taskLR/WD, seed731, exactdataorder and100Mschedule are retained. This is separate from the FP64short-run screen and excludes the Brevo LR=.002 winner. One seed only; original FP32stress numerical limitation remains.','','| Task | alpha.98 valCE | alpha.995 valCE | delta | alpha.98 testCE | alpha.995 testCE |','|---|---:|---:|---:|---:|---:|']
for row in records:
    def fmt(k):return f"{row[k]:.6f}" if k in row else 'pending'
    lines.append(f"| {row['task']} | {fmt('baseline_validation_ce')} | {fmt('validation_ce')} | {fmt('validation_ce_delta')} | {fmt('baseline_test_ce')} | {fmt('test_ce')} |")
lines+=['','Generation accuracy is reported in results.json only after all256probes are independently regraded.']
(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(report))
