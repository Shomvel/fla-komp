"""Replay already saved generation probes with padding; require exact guesses."""
import hashlib,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'snapshot'))
sys.path.insert(0,str(ROOT/'snapshot/suite100m'))
from block_eval_padding_adapter import LM
from evaluate import generate
RUN='brevo2-komp-final-lr0.001-wd0.1-seed731'
path=ROOT/'runs-ddp/results'/RUN
saved=torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=False)
model=LM('komp',731).cuda().eval();model.load_state_dict(saved['model']);del saved
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
existing=json.loads((path/'eval-progress.json').read_text())['records'][:24];assert len(existing)==24
actual=generate(model,'brevo2',[x['probe'] for x in existing])
assert len(actual)==24 and all(a==b for a,b in zip(existing,actual))
report={'status':'passed','run':RUN,'replayed_probes':24,'complete_record_exact_equality':True,'adapter_sha256':hashlib.sha256((ROOT/'block_eval_padding_adapter.py').read_bytes()).hexdigest(),'scorer_sha256':hashlib.sha256((ROOT/'snapshot/evaluate.py').read_bytes()).hexdigest(),'records_sha256':hashlib.sha256(json.dumps(existing,sort_keys=True).encode()).hexdigest(),'precision':'unchanged BF16network, original FP32Blockstate/projection','scope':'Exact full recorded guesses/grade match for first24already-completed probes; no additional evaluation samples'}
(ROOT/'evidence/block-padding-generation-replay.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
