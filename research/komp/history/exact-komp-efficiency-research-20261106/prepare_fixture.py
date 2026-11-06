"""CPU-only capture into a new private fixture; reads a completed checkpoint.
Run on the data host after choosing a stable checkpoint. No training or writes
outside --out. Full checkpoint copy is intentional to avoid retention races.
"""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'baseline'))
from data import Stream
from fixture_preflight import validate

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);a=p.parse_args()
 preflight=validate(a.checkpoint,a.manifest,a.audit)
 complete=a.checkpoint/'complete.json';assert complete.is_file()
 assert not a.out.exists(),'Use a new fixture directory'
 a.out.mkdir(parents=True)
 files=['model.pt',*[f'rank{r}.pt' for r in range(8)],'complete.json']
 sources={}
 for name in files:
  src=a.checkpoint/name;before=sha(src);shutil.copy2(src,a.out/name)
  assert before==sha(a.out/name)==sha(src),f'checkpoint changed during capture: {name}'
  sources[name]=before
 manifest=json.loads(a.manifest.read_text());shutil.copy2(a.manifest,a.out/'data-manifest.json')
 # This fixture binds exact token IDs and loader witnesses. Never consume test split.
 for rank in range(8):
  saved=torch.load(a.out/f'rank{rank}.pt',map_location='cpu',weights_only=False)
  stream=Stream(manifest,rank);stream.load_state_dict(saved['stream'])
  first=stream.next();stream.load_state_dict(saved['stream']);assert torch.equal(first,stream.next())
  stream.load_state_dict(saved['stream']);ids=torch.stack([stream.next() for _ in range(32)])
  after=stream.state_dict();stream.close()
  torch.save(dict(ids=ids,before=saved['stream'],after=after,rng=saved['rng']),a.out/f'input-rank{rank}.pt')
 receipt=dict(preflight=preflight,source_checkpoint=str(a.checkpoint.resolve()),source_hashes=sources,files={p.name:sha(p) for p in a.out.iterdir() if p.is_file()},shape=[32,2049],input_scope='Two fixed real-data B16 microbatches per rank; repeated for controlled benchmarking, not counted as formal tokens')
 (a.out/'fixture.json').write_text(json.dumps(receipt,indent=2));print(a.out)
if __name__=='__main__':main()
