"""Read frozen remote artifacts only; emit a portable regression fixture bundle."""
from pathlib import Path
import json,hashlib,datetime,sys
BASE=Path('/gpfs/users/linlizhi')
ROOT=BASE/'komp-flame-downstream-eval-20261107'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((ROOT/'sharded/plan.json').read_text());plan_sha=sha(ROOT/'sharded/plan.json')
manifest=json.loads((ROOT/'evidence/requests-manifest.json').read_text())
from tokenizers import Tokenizer
tok=Tokenizer.from_file(str(Path(manifest['tokenizer'])/'tokenizer.json'))
assert sha(Path(manifest['tokenizer'])/'tokenizer.json')==manifest['tokenizer_sha']
cases=['phonebook_32','phonebook_256','based_swde','based_fda','niah_single_1_1024','niah_single_1_8192','niah_single_3_2048']
bundle=dict(version=1,captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_root=str(ROOT),plan_sha=plan_sha,request_manifest=manifest,models=plan['models'],requests={},batches=[],coverage=[],protocol_audit=json.loads((ROOT/'evidence/generation-contract-audit.json').read_text()),source_files={})
for f in ['evaluate.py','evaluate_sharded.py','sharding.py','sharded/plan.json','evidence/generation-contract-audit.json']:
 bundle['source_files'][f]=dict(path=str(ROOT/f),sha256=sha(ROOT/f))
for case in [c for c in plan['cases'] if c['case'] in cases]:
 path=ROOT/'requests'/f"{case['case']}.jsonl";assert sha(path)==case['sha256']
 rows={r['id']:r for r in map(json.loads,path.read_text().splitlines())}
 def request(r):
  budget=r['generation']['max_gen_toks'];cap=65536 if r['task'].startswith('niah') else 32768 if r['task'].startswith('phonebook') else 2048
  return dict(**{k:r[k] for k in ['id','task','case','index','prompt','input_ids','labels','generation','metric']},effective_input_ids=r['input_ids'][-(cap-budget):],context_cap=cap,source_path=str(path),source_sha256=case['sha256'])
 for model in plan['models']:
  available=[];missing=[]
  for b in case['batches']:
   p=ROOT/'sharded/batches'/model/f"s{b['shard']}"/f"{b['key']}.json"
   if not p.exists():missing.append(b);continue
   val=json.loads(p.read_text());assert val['plan_sha']==plan_sha and val['checkpoint_sha']==plan['models'][model]['checkpoint_sha']
   assert [x['id'] for x in val['results']]==b['ids']
   events=set()
   if len({len(x['generated_ids']) for x in val['results']})>1:events.add('mixed_stop_lengths')
   for x in val['results']:
    row=rows[x['id']];raw=tok.decode(x['generated_ids'],skip_special_tokens=True)
    until=row['generation'].get('until',[]);until=[until] if isinstance(until,str) else until
    if 2 in x['generated_ids']:events.add('eos')
    if any(s and s in raw for s in until):events.add('stop_string')
    if len(x['generated_ids'])==row['generation']['max_gen_toks']:events.add('max_generation')
    if x['truncated_prompt_tokens']>0:events.add('left_truncation')
   available.append((b,p,val,events))
  chosen={}
  def choose(item,reason):
   if item is None:return
   key=item[0]['key']
   if key not in chosen:chosen[key]=[item,[]]
   chosen[key][1].append(reason)
  if available:
   choose(available[0],'first_completed_original_batch');choose(available[-1],'last_completed_original_batch')
   for event in ['eos','stop_string','mixed_stop_lengths','left_truncation','max_generation']:
    choose(next((x for x in available if event in x[3]),None),event)
  for item,reasons in chosen.values():
   b,p,val,events=item
   for id in b['ids']:bundle['requests'][id]=request(rows[id])
   enriched=[]
   for result in val['results']:
    r=rows[result['id']];raw=tok.decode(result['generated_ids'],skip_special_tokens=True)
    enriched.append(dict(**result,decoded_before_stop_split=raw))
   bundle['batches'].append(dict(model=model,case=case['case'],batch_key=b['key'],shard=b['shard'],request_ids=b['ids'],batch_size=len(b['ids']),effective_prompt_tokens=b['tokens'],selection_reasons=reasons,events=sorted(events),golden_status='observed_complete_batch',golden=dict(path=str(p),sha256=sha(p),checkpoint_sha=val['checkpoint_sha'],plan_sha=val['plan_sha'],eval_script_sha=val.get('eval_script_sha',plan['original_eval_sha']),inherited=val['inherited'],results=enriched)))
  # Preserve uncompleted shortest and longest original batches as input-only tests.
  for b in [case['batches'][0],case['batches'][-1]]:
   if b['key'] in chosen:continue
   for id in b['ids']:bundle['requests'][id]=request(rows[id])
   bundle['batches'].append(dict(model=model,case=case['case'],batch_key=b['key'],shard=b['shard'],request_ids=b['ids'],batch_size=len(b['ids']),effective_prompt_tokens=b['tokens'],selection_reasons=['input_coverage_endpoint'],events=[],golden_status='input_only_no_golden_selected'))
  bundle['coverage'].append(dict(model=model,case=case['case'],complete_batches_observed=len(available),selected_golden_batches=len(chosen),available_events=sorted(set().union(*(x[3] for x in available))) if available else []))
new=BASE/'komp-value-decay-learned-123-340m-20c-20261107'
extra=dict(model='value_learned123',root=str(new),status='not_in_existing_eval',golden_available=False)
for f in ['source-manifest.json','models.py','configs/data-manifest.json','configs/model-block/config.json','runs/block/latest.json','runs/block/complete.json']:
 p=new/f
 if p.exists():extra[f]=dict(path=str(p),sha256=sha(p),data=json.loads(p.read_text()) if p.suffix=='.json' and f!='source-manifest.json' else None)
extra['final_checkpoint_available']=(new/'runs/block/complete.json').exists()
extra['checkpoint_policy']='No checkpoint weights opened or hashed; latest.json is observed metadata only. Freeze and record an existing checkpoint separately before candidate numeric validation.'
extra['request_fixture_policy']='Reuse all input-only original batches with identical tokenizer/hash; do not reuse another model golden output.'
bundle['models']['value_learned123']=extra
bundle['derived_logits_tests']=dict(lengths=[1,2,3,4,7,127,128,129,255,256,257,1023,1024,1025,1999,2000],policy='Use actual effective input prefix of selected batches only when long enough; this is a derived input, has no saved logits/generation golden. Compare candidate vs frozen reference at same B,T,dtype and fixed teacher-forced tokens. Never compare B=1 against a saved B>1 golden.')
for case_name in sorted({b['case'] for b in bundle['batches']}):
 refs=[b for b in bundle['batches'] if b['model']=='kda' and b['case']==case_name]
 for b in [min(refs,key=lambda x:x['effective_prompt_tokens']),max(refs,key=lambda x:x['effective_prompt_tokens'])]:
  if any(x['model']=='value_learned123' and x['batch_key']==b['batch_key'] for x in bundle['batches']):continue
  bundle['batches'].append(dict(model='value_learned123',case=case_name,batch_key=b['batch_key'],shard=b['shard'],request_ids=b['request_ids'],batch_size=b['batch_size'],effective_prompt_tokens=b['effective_prompt_tokens'],selection_reasons=['new_model_input_coverage'],events=[],golden_status='input_only_new_model_no_frozen_checkpoint'))
 bundle['coverage'].append(dict(model='value_learned123',case=case_name,complete_batches_observed=0,selected_golden_batches=0,available_events=[]))
bundle['synthetic_protocol_tests']=[dict(name='eos_immediate',type='token_stream_only_not_model_golden',generated_stream=[2],expected_generated_ids=[2],expected_prediction='',expected_stopped=True),dict(name='eos_before_unused_tail',type='token_stream_only_not_model_golden',generated_stream=[2,1,2],expected_generated_ids=[2],expected_prediction='',expected_stopped=True),dict(name='empty_until_still_eos_stop',type='token_stream_only_not_model_golden',generation={'until':[],'max_gen_toks':128},generated_stream=[2],expected_generated_ids=[2],expected_prediction='',expected_stopped=True)]
bundle['eos_coverage_note']='No EOS was observed among the completed batches scanned in these seven selected cases; do not claim observed model EOS coverage. Synthetic stream tests cover protocol EOS handling only; real candidate EOS is a further acceptance requirement when encountered.'
print(json.dumps(bundle,ensure_ascii=False))
