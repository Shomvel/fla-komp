"""Read-only audit of completed formal runs; pending runs remain explicit."""
import hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PLAN=json.loads((ROOT/'plan.json').read_text())
DATA=json.loads((ROOT/'evidence/data-audit-console.json').read_text())
EXPECTED={'snapshot':hashlib.sha256((ROOT/'evidence/snapshot-sha256.json').read_bytes()).hexdigest(),
          'ddp_train':hashlib.sha256((ROOT/'ddp_train.py').read_bytes()).hexdigest()}
for name,digest in json.loads((ROOT/'evidence/snapshot-sha256.json').read_text()).items():
    assert hashlib.sha256((ROOT/'snapshot'/name).read_bytes()).hexdigest()==digest,name
CANDIDATE={'snapshot':EXPECTED['snapshot'],
           'inherited_ddp_train':EXPECTED['ddp_train'],
           'ddp_train':hashlib.sha256((ROOT/'ddp_train_gsa2_candidate.py').read_bytes()).hexdigest(),
           'candidate':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('gsa2_fixed_norm_candidate.py','gsa2_packed_eval_candidate.py','gsa2_consistent_candidate.py','gsa2_candidate_preflight.py')}}
sys.path.insert(0,str(ROOT/'snapshot/suite100m'))
from tasks import Factory
rows=[];orders={};seen=set()
for path in sorted([p for folder in ('runs-ddp','runs-gsa2-consistent') for p in (ROOT/folder/'results').glob('*/summary.json')]):
    item=json.loads(path.read_text());cfg=item['config'];task=cfg['task'];kind=cfg['model']['kind']
    assert (task,kind) not in seen,(task,kind)
    seen.add((task,kind))
    assert item['status']=='completed' and cfg['phase']=='final'
    candidate=path.relative_to(ROOT).parts[0]=='runs-gsa2-consistent'
    assert cfg['sources']==(CANDIDATE if candidate else EXPECTED),(path,cfg['sources'])
    if candidate:
        assert kind=='gsa2'
        release=json.loads((ROOT/'evidence/gsa2-numerical-release.json').read_text())
        assert release['status']=='released_with_limitations' and release['sources']==CANDIDATE
    assert cfg['seed']==731 and cfg['batch']==128 and cfg['world_size']==8
    assert cfg['local_microbatch_token_cap']==8192
    assert 99_000_000<cfg['model']['parameters']<101_000_000
    assert cfg['model']['layers']==15 and cfg['model']['width']==512
    assert cfg['model']['short_conv'] and cfg['model']['conv_size']==4
    chosen=PLAN['selections'][f'{task}-{kind}']
    assert (cfg['lr'],cfg['wd'])==(chosen['lr'],chosen['wd'])
    assert item['tokens']==DATA['tasks'][task]['train']['positions']
    assert math.isfinite(item['val']['ce'])
    orders.setdefault(task,cfg['data_order_sha256'])
    assert orders[task]==cfg['data_order_sha256']
    metrics=[json.loads(line) for line in path.with_name('metrics.jsonl').read_text().splitlines()]
    assert metrics[-1]['step']==cfg['steps'] and metrics[-1]['tokens']==item['tokens']
    assert all(math.isfinite(m['loss']) and math.isfinite(m['grad_norm']) for m in metrics)
    assert all(m['world_size']==8 and m['global_documents']<=128 for m in metrics)
    result={'task':task,'kind':kind,'implementation':'numerically-stabilized' if candidate else 'original-frozen','validation_ce':item['val']['ce'],'positions':item['tokens'],
            'train_seconds_including_compilation':item['train_seconds'],'summary_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    heldout=path.with_name('heldout.json')
    if heldout.exists():
        value=json.loads(heldout.read_text());generations=json.loads(path.with_name('generations.json').read_text())
        assert value['status']=='completed' and value['n_generation']==len(generations)==256
        assert math.isfinite(value['test']['ce'])
        factory=Factory(task)
        rescored=[bool(factory.grade(record['probe'],record['guess'])) for record in generations]
        assert all(actual==bool(record['valid']) for actual,record in zip(rescored,generations))
        assert value['generation_accuracy']==sum(rescored)/256
        result['heldout']=value
    rows.append(result)
report={'status':'passed_for_completed_runs','training_complete':len(rows),'heldout_complete':sum('heldout' in r for r in rows),
        'expected_runs':14,'pending':[f'{t}/{k}' for t in PLAN['tasks'] for k in PLAN['kinds'] if (t,k) not in seen],
        'sources':{'original':EXPECTED,'gsa2_stabilized':CANDIDATE},'data_order_sha256':orders,'runs':rows,'scope':'Completed-source/configuration/token-budget/finite-metrics audit; heldout generations independently rescored from frozen task grader; GSA2 wholebatch-Adam equivalence is not claimed.'}
(ROOT/'evidence/formal-results-audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2))
