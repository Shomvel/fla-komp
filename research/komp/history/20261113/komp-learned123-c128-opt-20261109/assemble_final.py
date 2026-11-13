"""Compose only bounded, measured candidates; final validation happens after assembly."""
import argparse,difflib,shutil
from pathlib import Path
r=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--output',default='snapshot-final');ap.add_argument('--operator',default='snapshot-combined');ap.add_argument('--norm',action='store_true');ap.add_argument('--cache',action='store_true');args=ap.parse_args()
out=r/args.output
assert not out.exists(),'Do not overwrite a validated or published candidate'
shutil.copytree(r/args.operator,out,ignore=shutil.ignore_patterns('__pycache__'))
architecture='snapshot-norm' if args.norm else 'snapshot-ff'
for name in ['fla/modules/rounded_swiglu.py','fla/models/komp/modeling_komp.py']:
 shutil.copy2(r/architecture/name,out/name)
if args.norm:
 for name in ['fla/modules/komp_rmsnorm.py','fla/layers/komp.py']:
  shutil.copy2(r/'snapshot-norm'/name,out/name)
if args.cache:
 target=out/'fla/ops/komp/block_fast.py';s=target.read_text()
 base=(r/'snapshot-decay/fla/ops/komp/block_fast.py').read_text()
 cached=(r/'snapshot-cache/fla/ops/komp/block_fast.py').read_text()
 start=base.index('    def forward(ctx, q, k, v, g, beta, keys0, values0, scores, init_scores, c, tau, ridge):')
 end=base.index('    @staticmethod\n    @once_differentiable\n    def backward',start)
 cstart=cached.index('    def forward(ctx, q, k, v, g, beta, keys0, values0, scores, init_scores, c, tau, ridge):')
 cend=cached.index('    @staticmethod\n    @once_differentiable\n    def backward',cstart)
 assert s.count(base[start:end])==1,'Selected variant changed forward; merge needs inspection'
 s=s.replace(base[start:end],cached[cstart:cend])
 s=s.replace('from .geometry import normalize_bwd_kernel, normalize_kernel','from .geometry import normalize_bwd_kernel, normalize_kernel\nfrom .geometry_cache import cached_geometry')
 target.write_text(s)
print(out)
