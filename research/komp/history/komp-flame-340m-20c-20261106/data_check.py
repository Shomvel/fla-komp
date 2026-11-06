"""Exercise real heldout filtering and byte/token-buffer restoration."""
import hashlib,json,tempfile
from pathlib import Path
import torch
from data import Stream
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'configs/data-manifest.json').read_text())
samples=json.loads((ROOT/'evidence/tokenizer-parity-samples.json').read_text())
with tempfile.TemporaryDirectory(dir=ROOT/'evidence') as directory:
 p=Path(directory);excluded=samples[0]['text'];keep='Independent training document. '+('alpha beta gamma '*800)
 file=p/'source.jsonl';file.write_text('\n'.join(json.dumps({'text':text}) for text in [excluded,keep,excluded,keep])+'\n')
 hashes=p/'excluded.json';hashes.write_text(json.dumps([hashlib.sha256(excluded.encode()).hexdigest()]))
 m=dict(manifest,train_files=[str(file)],excluded_text_sha256_path=str(hashes));stream=Stream(m,0,1)
 a=stream.next();assert stream.state['excluded_documents']==1
 before=stream.state_dict();b=stream.next();stream.load_state_dict(before);assert torch.equal(stream.next(),b)
 # Expected packing consists of allowed documents only, with one shared boundary ID.
 target=stream.tokenizer.encode(keep,add_special_tokens=True).ids*2
 assert a.tolist()==target[:2049] and b.tolist()==target[2048:4097]
 stream.close()
# Real shards: per-rank independent source set, next sequence deterministic after restore.
source_sets=[]
for rank in range(8):
 stream=Stream(manifest,rank);source_sets.append(set(stream.files));a=stream.next();s=stream.state_dict();b=stream.next();stream.load_state_dict(s);assert torch.equal(stream.next(),b);stream.close()
assert len(set.union(*source_sets))==len(manifest['train_files'])
assert sum(map(len,source_sets))==len(set.union(*source_sets))
(ROOT/'evidence/data-loader.json').write_text(json.dumps(dict(status='passed',heldout_exact_text_exclusion=True,byte_and_buffer_resume=True,rank_file_shards_disjoint=True,real_source_sequences_checked=16),indent=2))
