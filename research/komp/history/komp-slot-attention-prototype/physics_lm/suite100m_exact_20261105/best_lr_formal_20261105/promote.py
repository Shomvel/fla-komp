"""Select completed100M formal checkpoints only by final validation CE."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SUITE=ROOT.parent
LR=SUITE/'exact_alpha995_lr_formal_20261105'
CENTER=SUITE/'exact_alpha995_formal_20261105'
plan=json.loads((LR/'plan.json').read_text());old_path=ROOT/'manifest.json'
old=json.loads(old_path.read_text()) if old_path.exists() else {'tasks':{}}
manifest={'status':'partial','selection_rule':'min final full-validation CE among three prospectively fixed100Mformal LRs per task; no test or generation consulted','new_training_positions':0,'action':'promote existing exactsame-seed100Mformal checkpoint; no duplicate retraining','tasks':{}}
for task in plan['tasks']:
    candidates=[]
    for lr in sorted(plan['new_lrs'][task]+[plan['reused_lrs'][task]]):
        directory=(CENTER if lr==plan['reused_lrs'][task] else LR)/'runs-ddp/results'/f'{task}-exact-final-lr{lr:g}-wd0.1-seed731'
        p=directory/'summary.json'
        if not p.exists():continue
        r=json.loads(p.read_text());cfg=r['config']
        assert r['status']=='completed' and r['tokens']==plan['positions_per_run'][task]
        assert cfg['lr']==lr and cfg['wd']==.1 and cfg['seed']==731 and cfg['model']['alpha0']==.995 and cfg['model']['projection_dtype']=='float32'
        candidates.append({'lr':lr,'validation_ce':r['val']['ce'],'summary_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'directory':str(directory),'sources':cfg['sources'],'positions':r['tokens'],'data_order_sha256':cfg['data_order_sha256']})
    if len(candidates)!=3:
        manifest['tasks'][task]={'status':'awaiting_all_validation_results','completed_candidates':candidates};continue
    winner=min(candidates,key=lambda x:x['validation_ce'])
    directory=Path(winner['directory']);checkpoint=directory/'checkpoint.pt';stat=checkpoint.stat();prior=old['tasks'].get(task,{})
    if prior.get('selected_summary_sha256')==winner['summary_sha256'] and prior.get('checkpoint_stat')=={'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}:
        digest=prior['checkpoint_sha256']
    else:
        h=hashlib.sha256()
        with checkpoint.open('rb') as f:
            for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
        digest=h.hexdigest()
    original=next((SUITE/'runs-ddp/results').glob(task+'-exact-*/summary.json'))
    central=next((CENTER/'runs-ddp/results').glob(task+'-exact-*/summary.json'))
    manifest['tasks'][task]={'status':'selected_formal_checkpoint','alpha0':.995,'lr':winner['lr'],'wd':.1,'seed':731,'positions':winner['positions'],'projection_dtype':'float32','state_dtype':'float32','network_dtype':'bfloat16','selected_validation_ce':winner['validation_ce'],'selected_directory':str(directory),'selected_summary_sha256':winner['summary_sha256'],'checkpoint_sha256':digest,'checkpoint_stat':{'size':stat.st_size,'mtime_ns':stat.st_mtime_ns},'sources':winner['sources'],'data_order_sha256':winner['data_order_sha256'],'all_candidates':candidates,'original_alpha098_summary_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'central_alpha995_summary_sha256':hashlib.sha256(central.read_bytes()).hexdigest()}
if all(r['status']=='selected_formal_checkpoint' for r in manifest['tasks'].values()):manifest['status']='both_tasks_selected'
old_path.write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
