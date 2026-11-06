import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE=Path('/Users/shom/Downloads/komp-flame-340m-20c-20261106/migration-fla-mb16/runtime')
EXPECTED='1dde0fbb5668f1105425de6774a4d188535b278819d3af3f1c569a987359f5ea'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert sha(SOURCE/'source-manifest.json')==EXPECTED
 manifest=json.loads((SOURCE/'source-manifest.json').read_text())
 for name,digest in manifest['files'].items():assert sha(SOURCE/name)==digest,name
 for variant in ('baseline','candidate'):
  target=ROOT/variant
  if target.exists():raise FileExistsError(target)
  for name in [*manifest['files'],'source-manifest.json']:
   dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(SOURCE/name,dest)
 evidence=ROOT/'evidence';evidence.mkdir(exist_ok=True)
 for name in ('memory-exact.json','environment.json'):
  shutil.copy2(SOURCE/'evidence'/name,evidence/('original-'+name))
 (evidence/'provenance.json').write_text(json.dumps(dict(source=str(SOURCE),manifest_sha256=EXPECTED,verified_files=len(manifest['files']),scope='Local frozen files, no remote job operations'),indent=2))
 print('verified and copied',len(manifest['files']),'files per variant')
if __name__=='__main__':main()
