"""Summarize frozen result JSON without loading multi-GB gradient artifacts."""
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();out={'comparisons':{},'runs':{}}
for path in sorted(a.root.glob('*comparison.json')):
 x=json.loads(path.read_text());categories={}
 for category in ('grad','delta'):
  rows=[r for r in x['checks'] if f'/{category}/' in r['name']]
  categories[category]=dict(whole_relative_l2=(sum(r['error_squared_norm'] for r in rows)/max(sum(r['reference_squared_norm'] for r in rows),1e-30))**.5,worst_rms=max(r['rms'] for r in rows),worst_normalized_max=max(r['normalized_max'] for r in rows))
 out['comparisons'][path.name]=dict(status=x['status'],failures=len(x['failures']),max_loss_difference=max(r['loss_absolute'] for r in x['losses']),categories=categories)
for path in sorted(a.root.glob('bench-*/result-rank0.json')):
 x=json.loads(path.read_text());warm=[r for r in x['rows'] if r['phase']=='warm']
 if not warm:continue
 out['runs'][path.parent.name]=dict(variant=x['args']['variant'],cache=x['args']['cache'],prune_gc=x['args'].get('prune_gc',False),world=x['world'],samples=len(warm),median_seconds=statistics.median(r['update_wall_seconds'] for r in warm),median_forward_ms=statistics.median(r['forward_ms'] for r in warm),median_backward_ms=statistics.median(r['backward_ms_including_ddp'] for r in warm),median_clip_adam_ms=statistics.median(r['clip_adam_ms'] for r in warm),peak_allocated_gib=max(r['peak_allocated'] for r in warm)/2**30,peak_reserved_gib=max(r['peak_reserved'] for r in warm)/2**30)
a.out.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
