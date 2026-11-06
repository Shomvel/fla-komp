from pathlib import Path
import json,time
r=Path('/gpfs/users/linlizhi/komp-coh-trials-20261101')
receipt=dict(status='cancelled_by_user',reason='Stop coherence fine scan; switch to projection/SWA experiments',time=time.time())
(r/'fine-cancellation.json').write_text(json.dumps(receipt,indent=2))
# Workers check this before starting any dependent run; active jobs are deleted by exact alias next.
if not (r/'fine-failure.json').exists():(r/'fine-failure.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
