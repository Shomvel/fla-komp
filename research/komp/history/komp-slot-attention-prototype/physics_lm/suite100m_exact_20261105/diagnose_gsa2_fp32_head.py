"""Head-only FP32 experiment with BF16 recurrent/FFN computation."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import ddp_train as d
import gsa2_consistent_candidate as candidate
configuration=candidate.install()
# Preserve the native loss chunking and checkpoint schedule. Only the linear
# head itself moves to FP32; the old implementation casts its BF16 result.
import inspect
source=inspect.getsource(d.base.loss_sum)
old="def ce(a,b):return F.cross_entropy(F.linear(a,model.head.weight).float(),b,reduction='sum')"
new="def ce(a,b):\n  with torch.autocast('cuda',enabled=False):return F.cross_entropy(F.linear(a.float(),model.head.weight.float()),b,reduction='sum')"
assert source.count(old)==1
exec(compile(source.replace(old,new),'<isolated-fp32-loss-head>','exec'),d.base.__dict__)
spec={'sources':d.digest(),'configuration':configuration,'precision':'BF16 network; FP32 linear loss head and CE','candidate_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ('gsa2_fixed_norm_candidate.py','gsa2_packed_eval_candidate.py','gsa2_consistent_candidate.py','diagnose_gsa2_fp32_head.py')}}
(ROOT/'evidence/gsa2-fp32-head-source.json').write_text(json.dumps(spec,indent=2)+'\n')
source=(ROOT/'diagnose_gsa2_adam_witness.py').read_text()
source=source.replace("prior=ROOT/'evidence/gsa2-bf16-gradient-v1.json'", "prior=ROOT/'evidence/no-reuse-fp32-head.json'")
source=source.replace("for kind in ('gsa2','full') for case in ('mano','depo2')", "for kind in ('gsa2',) for case in ('mano','depo2')")
source=source.replace('gsa2-bf16-adam-witness.json','gsa2-fp32-head-adam.json')
source=source.replace('gsa2-layout-acceptance.json','gsa2-fp32-head-acceptance.json')
source=source.replace('BF16 training only; FP32 whole-batch shape sensitivity retained as a limitation','BF16 network with FP32 loss head, fixed normalization configs and packed long cache-free eval; isolated candidate, not formal acceptance')
exec(compile(source,str(ROOT/'diagnose_gsa2_adam_witness.py'),'exec'),globals())
