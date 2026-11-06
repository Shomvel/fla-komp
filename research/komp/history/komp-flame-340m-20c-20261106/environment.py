import json,sys,importlib.metadata
from pathlib import Path
import torch
from google.protobuf.internal.api_implementation import Type
from transformers import AutoTokenizer
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parent
path='/gpfs/shared_data/qiji/huggingface.co/fla-hub/transformer-1.3B-100B'
a=AutoTokenizer.from_pretrained(path,local_files_only=True);b=Tokenizer.from_file(path+'/tokenizer.json')
texts=['The quick brown fox jumps over the lazy dog.','A physics experiment: E = mc².','Hello\nWorld!',' 中文和 English.']
gold=json.loads((ROOT/'evidence/tokenizer-parity-samples.json').read_text())+json.loads((ROOT/'evidence/tokenizer-edge-gold.json').read_text())
for sample in gold:assert b.encode(sample['text'],add_special_tokens=True).ids==sample['ids']
auto_matches=all(a(s['text'])['input_ids']==s['ids'] for s in gold)
result=dict(python=sys.version,gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,protobuf_backend=Type(),versions={n:importlib.metadata.version(n) for n in ['triton','transformers','tokenizers','protobuf']},tokenizer_native_cpu_flame_gold_equivalent=True,parity_examples=len(gold),hf5_auto_matches_frozen_contract=auto_matches,tokenizer_implementation="tokenizers.Tokenizer.from_file(frozen tokenizer.json); HF5 AutoTokenizer alters normalizer",tokenizer_vocab=len(a),special_tokens=a.special_tokens_map)
(ROOT/'evidence/environment.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
