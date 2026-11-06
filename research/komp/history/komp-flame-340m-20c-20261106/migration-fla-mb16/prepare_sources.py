import pathlib,json,hashlib
b=pathlib.Path('/gpfs/users/linlizhi/komp-flame-340m-20c-20261106');dest=b/'migration-fla-mb16';(dest/'runtime/evidence').mkdir(parents=True,exist_ok=True)
for kind in ('exact','block'):
 if kind=='exact':
  r=b/'migration-fla/runtime/runs/exact';latest=json.loads((r/'latest.json').read_text());folder=pathlib.Path(latest['path']);complete=json.loads((folder/'complete.json').read_text());final=json.loads((r/'metrics.jsonl').read_text().splitlines()[-1]);assert latest['step']==final['step']==2492
  stop=dict(kind=kind,status='graceful_paused',latest=latest,complete=complete,final_logged=final,rollback_steps=0,rollback_tokens=0)
 else:stop=json.loads((b/f'migration-fla/stop-{kind}.json').read_text());folder=pathlib.Path(stop['latest']['path'])
 (dest/f'stop-{kind}.json').write_text(json.dumps(stop,indent=2))
 hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()};(dest/f'runtime/evidence/checkpoint-{kind}-hashes.json').write_text(json.dumps(dict(checkpoint=str(folder),step=stop['latest']['step'],files=hashes),indent=2))
print('STOP_AND_HASHES_SAVED')
