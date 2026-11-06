"""Check every fixture dependency before creating outputs; stdlib CPU only."""
import argparse,hashlib,json
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def validate(checkpoint,manifest_path,audit_path):
 names=['model.pt',*[f'rank{i}.pt' for i in range(8)],'complete.json']
 required=[manifest_path,audit_path,*[checkpoint/n for n in names]]
 missing=[str(p) for p in required if not p.is_file()]
 if missing:raise FileNotFoundError('Missing fixture dependencies: '+', '.join(missing))
 manifest=json.loads(manifest_path.read_text());audit=json.loads(audit_path.read_text())
 assert sha(audit_path)==manifest['data_audit_sha256'],'data audit identity changed'
 required=[*[Path(p) for p in manifest['train_files']],Path(manifest['tokenizer_path'])/'tokenizer.json',Path(manifest['excluded_text_sha256_path'])]
 required += [Path(manifest['tokenizer_path'])/name for name in audit['tokenizer_files']]
 missing=[str(p) for p in required if not p.is_file()]
 if missing:raise FileNotFoundError('Missing runtime data dependencies: '+', '.join(missing))
 for name,digest in audit['tokenizer_files'].items():assert sha(Path(manifest['tokenizer_path'])/name)==digest,name
 assert sha(Path(manifest['excluded_text_sha256_path']))==audit['exclusion_sha256']
 complete=json.loads((checkpoint/'complete.json').read_text());assert complete['world_size']==8
 return dict(status='passed',train_files=len(manifest['train_files']),checkpoint_step=complete['step'],data_manifest_sha=sha(manifest_path),audit_sha=sha(audit_path))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);a=p.parse_args();print(json.dumps(validate(a.checkpoint,a.manifest,a.audit)))
