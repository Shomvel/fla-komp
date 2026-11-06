"""Generate projection/SWA report directly from archived summaries, with paired seed bootstrap."""
import argparse
import json
from pathlib import Path
import block_report_v4 as report


def main():
    p=argparse.ArgumentParser();p.add_argument('--allow-partial',action='store_true');args=p.parse_args()
    root=Path(__file__).parent/'block_runs_proj'
    plan=json.loads((root/'evidence/plan.json').read_text())
    arms=plan['main']+plan['optional']
    report.ARMS=[(a,'results') for a in arms]
    report.PAIRS=[tuple(x) for x in plan['comparisons']]
    report.EXPLORATORY_PAIRS=[]
    if args.allow_partial:
        complete={a for a in arms if len(list((root/'results').glob(f'{a}-seed*/summary.json')))==7}
        report.PAIRS=[(a,b) for a,b in report.PAIRS if a in complete and b in complete]
    report.FROZEN={k.removeprefix('./'):v for k,v in json.loads((root/'evidence/frozen-verification.json').read_text()).items()}
    table=report.generate(root,args.allow_partial)
    table=table.replace('Scalar-KAPA versus Block/GDN contrasts are explicitly exploratory additions after six KAPA seeds were inspected; they were not in the original contrast plan.','Projection contrasts and SWA comparisons were specified before training; C128 is the prespecified optional extension.')
    runtime=['| arm | seed | train loop + bookkeeping s | evaluation s | generation s | total train + eval s |','|---|---:|---:|---:|---:|---:|']
    configs=['| arm | parameters | persistent elements / head |','|---|---:|---:|']
    initial={};details=[]
    for arm in arms:
        rows=[]
        for path in sorted((root/'results').glob(f'{arm}-seed*/summary.json')):
            row=json.loads(path.read_text());cfg=row['config'];seed=cfg['seed'];rows.append(row)
            initial.setdefault(seed,[]).append(cfg['shared_init_sha256'])
            timing=row['timing']
            runtime.append(f"| {arm} | {seed} | {timing['train_loop_and_bookkeeping_seconds']:.2f} | {timing['evaluation_seconds']:.2f} | {timing['generation_seconds']:.2f} | {row['elapsed_seconds']:.2f} |")
            details.append(dict(arm=arm,seed=seed,**timing,total_seconds=row['elapsed_seconds']))
        if rows:configs.append(f"| {arm} | {rows[0]['config']['params']} | {rows[0]['config']['persistent_state_elements_per_head']} |")
    for seed,hashes in initial.items():assert all(h==hashes[0] for h in hashes),('all-shared initialization',seed)
    table+='\n'+'\n'.join(configs)+'\n\nPer-run synchronized wall time. Training excludes measured evaluation/generation, includes loop/logging overhead. Different implementations: these are not kernel speed comparisons.\n\n'+'\n'.join(runtime)+'\n'
    if args.allow_partial:table='INCOMPLETE interim results; do not treat partial seed sets as the final experiment.\n\n'+table
    (root/'TABLE.md').write_text(table)
    (root/'timing.json').write_text(json.dumps(details,indent=2))
    print(table)


if __name__=='__main__':main()
