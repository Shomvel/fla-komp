import json,importlib.metadata,inspect
from pathlib import Path
from transformers import AutoTokenizer,LlamaTokenizerFast
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parent
path='/gpfs/shared_data/qiji/huggingface.co/fla-hub/transformer-1.3B-100B'
a=AutoTokenizer.from_pretrained(path,local_files_only=True);b=Tokenizer.from_file(path+'/tokenizer.json');c=LlamaTokenizerFast.from_pretrained(path,local_files_only=True)
texts=['The quick brown fox jumps over the lazy dog.','A physics experiment: E = mc².','Hello\nWorld!',' 中文和 English.']
report=dict(versions={n:importlib.metadata.version(n) for n in ['transformers','tokenizers']},auto_class=type(a).__name__,explicit_class=type(c).__name__,auto_special=a.special_tokens_map,rows=[])
for text in texts:
 ids=a(text)['input_ids'];native=b.encode(text,add_special_tokens=True).ids;explicit=c(text)['input_ids']
 report['rows'].append(dict(text=text,auto=ids,native=native,explicit=explicit))
report['auto_backend']=json.loads(a.backend_tokenizer.to_str()) if hasattr(a,'backend_tokenizer') else None
(ROOT/'evidence/tokenizer-diagnosis.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='auto_backend'},indent=2))
