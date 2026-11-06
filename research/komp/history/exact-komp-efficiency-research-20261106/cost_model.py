"""Static tensor/operation accounting; not a GPU latency estimator."""
import json
from pathlib import Path
B,T,H,D,M,V,C,L,A,W=16,2048,16,64,32,64,16,24,2,8
BH=B*H;N=BH*T;NC=(T+C-1)//C
shapes={
 'qn':(BH,T,D),'pool':(BH,T+M,D),'ids':(BH,T+1,M),'drop':(BH,T),
 'p':(BH,T,M),'h':(BH,T,M),'r':(BH,T,M),'n':(BH,T),'be':(BH,T),
 'u':(BH,T,2,M),'gc_saved_but_unread_in_backward':(BH,NC,C,M),
 'f':(BH,NC,M,M),'qm':(BH,NC,C,M),'entry':(BH,NC,M,V),'z':(BH,T,2,V)}
import math
rows={k:dict(shape=s,bytes=math.prod(s)*4,mib=math.prod(s)*4/2**20) for k,s in shapes.items()}
cache=N*(3*M+1)*4
result=dict(shape=dict(B=B,T=T,H=H,K=D,M=M,V=V,C=C,L=L,accumulation=A,world=W),
 scope='Static frozen FIFO FP32 source accounting; excludes inputs, allocator, temporary overlap, parameters, FLA/MLP/head, DDP and optimizer. Atomic counts are logical contributions, not HBM bytes or measured time.',
 saved_internal_tensors=rows,
 geometry_cache_bytes_per_layer=cache,geometry_cache_gib_24_layers=cache*L/2**30,
 baseline_peak_gib_observed=32.098,
 baseline_plus_cache_gib_arithmetic_only=32.098+cache*L/2**30,
 baseline_solves_per_rank_update=3*N*L*A,candidate_solves_per_rank_update=2*N*L*A,
 avoided_solves_per_rank_update=N*L*A,
 geometry_backward_dpool_atomic_contributions_per_layer=N*(2*M*D+2*D),
 geometry_backward_atomic_payload_gib_per_layer=N*(2*M*D+2*D)*4/2**30,
 local_gradients_atomic_contributions_per_layer=N*M*(V//32)*5,
 full_matrix_tape_gib_per_layer=N*M*M*4/2**30,
 conditional_speedup=[dict(saved_replay_geometry_fraction=f,speedup_no_overhead=1/(1-f)) for f in [.05,.1,.2,.3]],
 gc_note='Baseline GC is written and saved, but backward only unpacks it; candidate --prune-gc removes its allocation/global store while keeping local cumsum arithmetic. Cached plus pruned B16x2 is the measured combined candidate.')
path=Path(__file__).resolve().parent/'evidence/cost-model.json';path.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
