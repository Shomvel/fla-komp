"""Resume the immutable trainer after verifying checkpoint restoration on every rank."""
import sys,hashlib,json,random
from pathlib import Path
ROOT=Path('/gpfs/users/linlizhi/komp-kda-options-ff2272-340m-20c-20261106')
sys.path.insert(0,str(ROOT))
import train
import torch
import numpy as np
import torch.distributed as dist
original=train.resume

def checked_resume(out,model,opt,stream,rank,source_sha,data_sha):
    step,tokens=original(out,model,opt,stream,rank,source_sha,data_sha)
    assert (step,tokens)==(1536,805306368)
    folder=Path(json.loads((out/'latest.json').read_text())['path'])
    saved=torch.load(folder/f'rank{rank}.pt',map_location='cpu',weights_only=False)
    assert stream.state_dict()==saved['stream']
    assert torch.equal(torch.get_rng_state(),saved['rng']['torch'])
    assert torch.equal(torch.cuda.get_rng_state(),saved['rng']['cuda'])
    assert random.getstate()==saved['rng']['python']
    n=np.random.get_state();want=saved['rng']['numpy']
    assert n[0]==want[0] and np.array_equal(n[1],want[1]) and n[2:]==want[2:]
    assert len(opt.state)==sum(1 for _ in model.parameters())
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert all(torch.isfinite(s[k]).all() for s in opt.state.values() for k in ('exp_avg','exp_avg_sq'))
    h=hashlib.sha256()
    for name,p in model.state_dict().items():h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    hashes=[None]*8;dist.all_gather_object(hashes,h.hexdigest());assert len(set(hashes))==1
    receipt=dict(status='passed',rank=rank,step=step,tokens=tokens,source_sha=source_sha,data_sha=data_sha,model_sha=hashes[0],adam_states=len(opt.state),rng_loader_match=True,replicas_identical=True)
    train.atomic_json(ROOT/f'evidence/block-resume-20261106-r1/restore-rank{rank}.json',receipt)
    print('RESUME_VERIFIED '+json.dumps(receipt),flush=True)
    dist.barrier()
    return step,tokens
train.resume=checked_resume
sys.argv=['train.py','--kind','block']
train.main()
