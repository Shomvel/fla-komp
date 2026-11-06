import hashlib,json,os,sys,time
from pathlib import Path
r=Path('/gpfs/users/linlizhi/physics100m-komp-variants-20261102'); old=Path('/gpfs/users/linlizhi/physics100m-20261101/snapshot/suite100m'); q=r/'queue'
a=r/'evidence/repair-source-20261102T0619Z';a.mkdir(exist_ok=False)
failure=json.loads((q/'FAILED.json').read_text()); ident='300-capo-komp_gdn_slot-final-lr0.001-wd0-seed731'
assert failure['name']==ident and failure['worker']=='4' and failure['exit_code']==1
before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'suite100m').glob('*.py')}
source=r/'suite100m/source';assert not source.exists() and not source.is_symlink()
assert (r/'suite100m/tasks.py').read_bytes()==(old/'tasks.py').read_bytes()
source.symlink_to(old/'source',target_is_directory=True)
sys.path.insert(0,str(r/'suite100m'))
from tasks import Factory
checks={}
for task in ('mano','lano','depo2','brevo2','capo'):
 fac=Factory(task);probes=json.loads((r/'data'/task/'test.probes.json').read_text())[:256]
 assert len(probes)==256
 if task=='capo':
  regenerated=fac.bios.probes('test',len(json.loads((r/'data'/task/'test.probes.json').read_text())))
  assert regenerated[:256]==probes
  assert all(fac.grade(p,p['answer']) and p['answer'][-1]==fac.bios.vocab['.'] for p in probes)
 elif task=='lano': assert fac.grammar.valid(fac.grammar.generate(__import__('random').Random(731)))
 else: assert all(fac.grade(p,p['answer']) for p in probes)
 checks[task]='Factory and grading inputs passed; Capo regenerated probes exactly match' if task=='capo' else 'Factory and grading inputs passed'
assert before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'suite100m').glob('*.py')}
files={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
# Retain complete failed attempt before allowing the idle existing workers to retry.
claim=q/(ident+'.4.running');done=q/(ident+'.done');log=r/'logs'/(ident+'.log')
assert json.loads(done.read_text())==failure
assert json.loads(claim.read_text())==failure['spec']
assert not (q/(ident+'.json')).exists()
for p in (claim,done,log,q/'FAILED.json'):(a/p.name).write_bytes(p.read_bytes())
done.unlink();log.rename(a/(log.name+'.original'));claim.rename(q/(ident+'.json'))
receipt=dict(at=time.time(),failure=failure,source_link=str(source),target=str(source.resolve()),source_files=files,checks=checks,training_sources_unchanged=True,action='restore shared source link; archive failed attempt; requeue unchanged evaluation; release failure barrier')
(a/'receipt.json').write_text(json.dumps(receipt,indent=2))
(q/'FAILED.json').unlink()
print(json.dumps(receipt))
