"""CPU-only real loader cost from fixture cursor; no production stream mutation."""
import argparse,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'baseline'))
from data import Stream
p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True);p.add_argument('--rank',type=int,default=0);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
saved=torch.load(a.fixture/f'input-rank{a.rank}.pt',map_location='cpu',weights_only=False)
stream=Stream(json.loads((a.fixture/'data-manifest.json').read_text()),a.rank);rows=[]
for repeat in range(4):
 stream.load_state_dict(saved['before']);start=time.perf_counter()
 batches=[torch.stack([stream.next() for _ in range(16)]) for _ in range(2)]
 elapsed=time.perf_counter()-start
 assert torch.equal(torch.cat(batches),saved['ids'])
 rows.append(dict(repeat=repeat,seconds=elapsed,shape=[2,16,2049]))
stream.close();a.out.write_text(json.dumps(dict(rows=rows,scope='Single rank CPU JSONL/tokenizer/packing; OS cache not flushed; no H2D, concurrent rank contention or checkpoint IO'),indent=2))
