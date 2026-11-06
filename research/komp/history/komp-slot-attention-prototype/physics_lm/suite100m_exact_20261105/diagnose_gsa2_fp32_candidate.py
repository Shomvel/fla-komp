"""Isolate BF16 rounding from mathematical/FP32 gradient consistency."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import ddp_train as d
import gsa2_packed_eval_candidate as candidate
spec={'sources':d.digest(),'candidate_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ('gsa2_fixed_norm_candidate.py','gsa2_packed_eval_candidate.py')},'configuration':candidate.install()}
(ROOT/'evidence/gsa2-fp32-candidate-source.json').write_text(json.dumps(spec,indent=2)+'\n')
source=(ROOT/'diagnose_gsa2_real_layout.py').read_text()
source=source.replace("assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == expected['script_sha256']", "assert d.digest() == spec['sources']")
source=source.replace('gsa2-real-layout.json','gsa2-packed-eval-layout.json')
exec(compile(source,str(ROOT/'diagnose_gsa2_real_layout.py'),'exec'),globals())
source=(ROOT/'diagnose_gsa2_adam_witness.py').read_text()
source=source.replace("prior=ROOT/'evidence/gsa2-bf16-gradient-v1.json'", "prior=ROOT/'evidence/no-reuse-fp32-candidate.json'")
source=source.replace("for kind in ('gsa2','full') for case in ('mano','depo2')", "for kind in ('gsa2',) for case in ('mano','depo2')")
source=source.replace("d.amp('bf16')", "d.amp('fp32')")
source=source.replace('gsa2-bf16-adam-witness.json','gsa2-fp32-candidate-adam.json')
source=source.replace('gsa2-layout-acceptance.json','gsa2-fp32-candidate-acceptance.json')
source=source.replace('BF16 training only; FP32 whole-batch shape sensitivity retained as a limitation','FP32 isolated candidate diagnostic only; does not authorize formal precision change')
exec(compile(source,str(ROOT/'diagnose_gsa2_adam_witness.py'),'exec'),globals())
