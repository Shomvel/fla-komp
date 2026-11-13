"""Portable structural/score validation; no GPU and no model inference."""
from pathlib import Path
import hashlib,json,re,collections
root=Path(__file__).resolve().parent
d=json.loads((root/'fixtures.json').read_text());m=json.loads((root/'manifest.json').read_text())
assert hashlib.sha256((root/'fixtures.json').read_bytes()).hexdigest()==m['fixtures_sha256']
seen=set();golden_rows=0
for b in d['batches']:
 key=(b['model'],b['batch_key']);assert key not in seen;seen.add(key)
 rows=[d['requests'][x] for x in b['request_ids']]
 assert len(rows)==b['batch_size'] and len({r['id'] for r in rows})==len(rows)
 assert all(len(r['effective_input_ids'])==b['effective_prompt_tokens'] for r in rows)
 assert all(r['generation']==rows[0]['generation'] for r in rows)
 for r in rows:assert r['effective_input_ids']==r['input_ids'][-(r['context_cap']-r['generation']['max_gen_toks']):]
 if 'golden' not in b:continue
 g=b['golden'];assert [x['id'] for x in g['results']]==b['request_ids']
 assert g['checkpoint_sha']==d['models'][b['model']]['checkpoint_sha'] and g['plan_sha']==d['plan_sha']
 for r,x in zip(rows,g['results']):
  assert x['generated_tokens']==len(x['generated_ids'])<=r['generation']['max_gen_toks']
  assert x['prompt_tokens']==len(r['effective_input_ids']) and x['truncated_prompt_tokens']==len(r['input_ids'])-x['prompt_tokens']
  stops=r['generation'].get('until',[]);stops=[stops] if isinstance(stops,str) else stops
  text=x['decoded_before_stop_split']
  for stop in list(stops)+['</s>']:
   if stop:text=text.split(stop)[0]
  assert text==x['prediction']
  if r['metric']=='string_match_all':
   text=re.sub(r'[\x00-\x1f]','\n',text.strip()).strip();score=sum(v.lower() in text.lower() for v in r['labels'])/len(r['labels'])
  else:
   phone=r['task'].startswith('phonebook');score=max(int(bool(re.search(re.escape(v.strip() if phone else v),text.strip() if phone else text,re.I))) for v in r['labels'])
  assert score==x['score'];golden_rows+=1
assert set(d['models'])=={b['model'] for b in d['batches']}
print(json.dumps(dict(status='passed',requests=len(d['requests']),batches=len(d['batches']),golden_rows=golden_rows,golden_batches=sum('golden' in b for b in d['batches']),models=len(d['models']))))
