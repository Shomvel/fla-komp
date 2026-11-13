"""CPU audit against the exact historical HFLM and tokenizer, before GPU launch."""
import json,sys,hashlib,copy,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
from transformers import AutoTokenizer,GenerationConfig
from tokenizers import Tokenizer
from lm_eval.models.huggingface import HFLM
from lm_eval.api.task import TaskConfig
spec=importlib.util.spec_from_file_location('eval_runner',ROOT/'evaluate.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
MODELS,context_ids,sha=runner.MODELS,runner.context_ids,runner.sha
BASE=Path('/gpfs/users/linlizhi')
manifest=json.loads((ROOT/'evidence/requests-manifest.json').read_text())
tok=AutoTokenizer.from_pretrained(manifest['tokenizer'])
raw=Tokenizer.from_file(str(Path(manifest['tokenizer'])/'tokenizer.json'))
assert tok.is_fast and not tok.clean_up_tokenization_spaces
assert tok.eos_token_id==2 and tok.bos_token_id==1
hf=object.__new__(HFLM);hf.tokenizer=tok;hf.backend='causal';hf.add_bos_token=False;hf._max_length=None
historical=BASE/'flame/exp/gated_deltanet_hop_340M_shift_4_state_passing_0_1/config.json'
from types import SimpleNamespace
hf._config=SimpleNamespace(**json.loads(historical.read_text()));hf._model=SimpleNamespace(config=hf._config)
assert hf.max_length==2048
evidence=dict(status='passed',historical_config_path=str(historical),historical_config_sha=sha(historical),historical_hflm_max_length=hf.max_length,tokenizer_class=type(tok).__name__,tokenizer_cleanup=False,add_special_tokens=False,bos=False,eos=2,caps=dict(niah=65536,phonebook=32768,based=2048),cases=[],configs=[],differences=['Frozen training adapters retain FP32 parameters/residuals/gates with BF16 autocast; historical HF run used BF16 parameter conversion. Learned low-rank FP32 gate operations require this training-native precision.','Cache disabled because frozen KOMP adapters reject it; complete prefix is recomputed.','No padding, at most eight equal effective-length samples/batch versus historical left-padding batch16/64; batch-shape rounding may differ.'])
for name,folder,kind,partial in MODELS:
 for p in sorted((BASE/folder/'configs').rglob('config.json')):
  cfg=json.loads(p.read_text());assert cfg['max_position_embeddings']==2048
  for key,expected in [('num_beams',1),('do_sample',False),('repetition_penalty',1.0),('no_repeat_ngram_size',0),('min_length',0)]:assert cfg.get(key,expected)==expected,(p,key)
  evidence['configs'].append(dict(model=name,path=str(p),sha256=sha(p),max_position_embeddings=cfg['max_position_embeddings']))
assert TaskConfig().filter_list is None # ConfigurableTask installs the take_first default.
for text in [' hello world',"I can't do it .",'<s>hello</s>','电话 你好\n123-456','\n\n']:
 ids=tok.encode(text,add_special_tokens=False)
 assert ids==raw.encode(text,add_special_tokens=False).ids
 for special in [False,True]:assert tok.decode(ids,skip_special_tokens=special)==raw.decode(ids,skip_special_tokens=special)
allids=list(range(len(tok)))
for special in [False,True]:assert tok.decode(allids,skip_special_tokens=special)==raw.decode(allids,skip_special_tokens=special)
for case in manifest['cases']:
 p=ROOT/'requests'/f"{case['case']}.jsonl";assert sha(p)==case['sha256']
 rows=[json.loads(x) for x in p.read_text().splitlines()];truncated=0
 for start in range(0,len(rows),32):
  chunk=rows[start:start+32]
  ids=tok([r['prompt'] for r in chunk],add_special_tokens=False)['input_ids']
  assert ids==[r['input_ids'] for r in chunk]
 for row in rows:
  kw=row['generation'];assert set(kw)<={'until','max_gen_toks','do_sample','temperature'},kw
  assert not kw.get('do_sample',False) and kw.get('temperature',0)==0
  expected=128 if row['task'].startswith('niah') else 20 if row['task'].startswith('phonebook') else 48
  assert kw['max_gen_toks']==expected
  cap=65536 if row['task'].startswith('niah') else 32768 if row['task'].startswith('phonebook') else hf.max_length
  assert context_ids(row,2048)==row['input_ids'][-(cap-expected):]
  truncated+=len(context_ids(row,2048))<len(row['input_ids'])
 # Direct HFLM encoding/truncation comparison on shortest/longest/first request.
 for row in [rows[0],min(rows,key=lambda x:len(x['input_ids'])),max(rows,key=lambda x:len(x['input_ids']))]:
  tok.pad_token=tok.eos_token
  ids,mask=hf.tok_batch_encode([row['prompt']],left_truncate_len=cap-expected)
  assert ids[0].tolist()==context_ids(row,2048) and mask.all()
 evidence['cases'].append(dict(case=case['case'],count=len(rows),generation=rows[0]['generation'],truncated=truncated,request_sha=sha(p)))
 print('AUDIT',case['case'],len(rows),'truncated',truncated,flush=True)
(ROOT/'evidence/generation-contract-audit.json').write_text(json.dumps(evidence,indent=2)+'\n')
print('CONTRACT_AUDIT_PASS',flush=True)
