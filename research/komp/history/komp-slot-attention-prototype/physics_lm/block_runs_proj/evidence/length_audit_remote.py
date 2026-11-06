from pathlib import Path
import hashlib,json,statistics
R=Path('/gpfs/users/linlizhi/komp-proj-swa-20261101')
D=Path('/gpfs/users/linlizhi/softmax-10m-20261023/data')
manifest=json.loads((R/'snapshot/physics_lm/data-manifest.json').read_text())
out={}
for split in ('train','confirm'):
 p=D/(split+'.jsonl');assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest['splits'][split]['sha256']
 rows=[json.loads(x) for x in p.read_text().splitlines()]
 for task in ('depo','brevo'):
  selected=[r for r in rows if r['task']==task]
  lengths=[len(r['input_ids']) for r in selected]
  # Output at position i predicts gold token at i+1.
  query_positions=[i for r in selected for i in range(len(r['input_ids'])-1) if r['loss_mask'][i+1] and r['input_ids'][i+1]<9000]
  out[f'{split}/{task}']=dict(documents=len(selected),min_length=min(lengths),mean_length=statistics.mean(lengths),max_length=max(lengths),
     content_prediction_positions=len(query_positions),fraction_predictions_after_first_C64=sum(i>=64 for i in query_positions)/len(query_positions),
     fraction_predictions_after_first_C128=sum(i>=128 for i in query_positions)/len(query_positions))
(R/'sequence-length-audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
