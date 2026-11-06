"""Summarize completed ABBA runs; never mix profiler/capture rows with timing."""
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('results',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();runs=[]
for run in ('a1','b1','b2','a2'):
 rows=[json.loads(x.read_text()) for x in sorted((a.results/f'bench-{run}').glob('result-rank*.json'))]
 assert rows and len(rows)==rows[0]['world']
 warm=[[r for r in x['rows'] if r['phase']=='warm'] for x in rows]
 assert len({len(x) for x in warm})==1 and len(warm[0])>=10
 times=[max(x[i]['update_wall_seconds'] for x in warm) for i in range(len(warm[0]))]
 med=statistics.median(times);tokens=warm[0][0]['effective_tokens']
 runs.append(dict(run=run,world=rows[0]['world'],variant=rows[0]['args']['variant'],cache=rows[0]['args']['cache'],fixture_sha=rows[0]['fixture_sha'],median_seconds=med,tokens_per_second=tokens/med,min_seconds=min(times),max_seconds=max(times),peak_allocated=max(r['peak_allocated'] for x in warm for r in x),peak_reserved=max(r['peak_reserved'] for x in warm for r in x)))
assert len({r['fixture_sha'] for r in runs})==1
baseline=statistics.median(r['median_seconds'] for r in runs if not r['cache']);cached=statistics.median(r['median_seconds'] for r in runs if r['cache'])
result=dict(runs=runs,median_ratio_baseline_over_cached=baseline/cached,scope='Repeated fixed-input ABBA, slowest rank, warm rows only; excludes real loader and checkpoint IO; not long-run quality proof')
a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
