"""CPU-only source and zero-training reuse receipts; never rewrites old results."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUITE = ROOT.parent
OLD = SUITE/'exact_tuning_20261105'
PLAN = json.loads((ROOT/'lr-screen.json').read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8*1024*1024):
            h.update(chunk)
    return h.hexdigest()


(ROOT/'evidence').mkdir(exist_ok=True)
(ROOT/'logs').mkdir(exist_ok=True)
for task, reuse in PLAN['reuse'].items():
    result = OLD/'results'/task/reuse['candidate']
    summary = json.loads((result/'summary.json').read_text())
    config = summary['config']
    assert summary['status'] == 'completed'
    assert config['candidate'] == PLAN['fixed_candidate']
    assert config['lr'] == reuse['lr'] and config['wd'] == PLAN['weight_decay']
    assert config['seed'] == PLAN['seed'] and config['batch'] == PLAN['batch_documents']
    assert config['world_size'] == PLAN['world_size_per_run'] and summary['tokens'] == config['train_positions']
    old_numerical = OLD/'evidence'/f'numerical-{task}.json'
    old_ddp = OLD/'preflight'/task/'runs-ddp/ddp-preflight/exact.json'
    for gate in (old_numerical, old_ddp):
        ticket = json.loads(gate.read_text())
        assert ticket['status'] == 'passed' and ticket['sources'] == config['sources']
    receipt = {'status': 'verified', 'task': task, 'lr': reuse['lr'], 'candidate': reuse['candidate'],
               'new_training_steps': 0, 'new_training_positions': 0, 'val': summary['val'],
               'original_config': config, 'files': {str(result/name): sha(result/name) for name in ('summary.json', 'config.json', 'checkpoint.pt', 'metrics.jsonl')},
               'inherited_acceptance': {str(p): sha(p) for p in (old_numerical, old_ddp)},
               'scope': 'Read-only reference to completed 1x LR run; new LR runs inherit unchanged operator/backbone numerical and DDP acceptance, not training outcomes'}
    path = ROOT/'evidence'/f'reuse-{task}.json'
    if path.exists():
        assert json.loads(path.read_text()) == receipt
    else:
        path.write_text(json.dumps(receipt, indent=2)+'\n')
    print(task, reuse['lr'], summary['val']['ce'], flush=True)
files = list(ROOT.glob('*.py'))+[ROOT/'lr-screen.json']
inherited = list(OLD.glob('*.py'))+[OLD/'initial-screen.json', SUITE/'ddp_train.py', SUITE/'evidence/snapshot-sha256.json']
manifest = {'files': {str(p.relative_to(ROOT)): sha(p) for p in files},
            'inherited_files': {str(p.relative_to(SUITE)): sha(p) for p in inherited}}
(ROOT/'source-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
