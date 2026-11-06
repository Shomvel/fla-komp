"""Isolated fixed-norm forward/gradient check on the real-data witness."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import ddp_train as d
import gsa2_fixed_norm_candidate as candidate
spec={'sources':d.digest(),'candidate_sha256':hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),'configuration':candidate.install()}
(ROOT/'evidence/gsa2-fixed-norm-source.json').write_text(json.dumps(spec,indent=2)+'\n')
# Reuse the established diagnostic algorithms, with fresh output paths. Their
# original script SHA check is replaced by the full source binding above.
source=(ROOT/'diagnose_gsa2_real_layout.py').read_text()
source=source.replace("assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == expected['script_sha256']", "assert d.digest() == spec['sources']")
source=source.replace('gsa2-real-layout.json','gsa2-fixed-norm-layout.json')
exec(compile(source,str(ROOT/'diagnose_gsa2_real_layout.py'),'exec'),globals())
layout=json.loads((ROOT/'evidence/gsa2-fixed-norm-layout.json').read_text())
# Forward comparison precedes the expensive backward witness.
if layout['checks']['actual_rank_microbatch'][-1]['rms_relative'] < .02:
    source=(ROOT/'diagnose_gsa2_adam_witness.py').read_text()
    source=source.replace("prior=ROOT/'evidence/gsa2-bf16-gradient-v1.json'", "prior=ROOT/'evidence/no-reuse-fixed-norm.json'")
    source=source.replace("for kind in ('gsa2','full') for case in ('mano','depo2')", "for kind in ('gsa2',) for case in ('mano','depo2')")
    source=source.replace('gsa2-bf16-adam-witness.json','gsa2-fixed-norm-adam.json')
    source=source.replace('gsa2-layout-acceptance.json','gsa2-fixed-norm-acceptance.json')
    exec(compile(source,str(ROOT/'diagnose_gsa2_adam_witness.py'),'exec'),globals())
else:
    print('FORWARD_GATE_FAILED: gradient/update witness skipped',flush=True)
