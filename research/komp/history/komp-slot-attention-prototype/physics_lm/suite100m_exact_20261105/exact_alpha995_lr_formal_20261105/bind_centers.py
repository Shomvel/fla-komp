"""Bind completed100M central LR checkpoints without copying or retraining."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'exact_alpha995_formal_20261105'
plan=json.loads((ROOT/'plan.json').read_text());records=[]
for task,lr in plan['reused_lrs'].items():
    name=f'{task}-exact-final-lr{lr:g}-wd0.1-seed731';directory=SOURCE/'runs-ddp/results'/name
    summary=json.loads((directory/'summary.json').read_text())
    assert summary['status']=='completed' and summary['tokens']==plan['positions_per_run'][task]
    assert summary['config']['model']['alpha0']==.995 and summary['config']['model']['projection_dtype']=='float32'
    assert summary['config']['lr']==lr and summary['config']['wd']==.1
    h=hashlib.sha256()
    with (directory/'checkpoint.pt').open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
    records.append({'task':task,'lr':lr,'new_training_positions':0,'directory':str(directory),'checkpoint_sha256':h.hexdigest(),'summary_sha256':hashlib.sha256((directory/'summary.json').read_bytes()).hexdigest(),'source':summary['config']['sources'],'validation_ce':summary['val']['ce']})
(ROOT/'evidence/central-reuse.json').write_text(json.dumps({'status':'bound','runs':records},indent=2)+'\n')
print(json.dumps(records))
