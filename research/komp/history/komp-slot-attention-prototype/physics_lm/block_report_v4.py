"""Seed-paired frozen-kernel trials report; summaries are the sole metric source."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics

ARMS=[('block_c64','results'),('block_slot_c64','results'),('block_slot_gate_c64','results'),
      ('block_slot_learn_c64','results'),('block_slot_learn_c32','results'),
      ('gdn','baselines/results'),('gsa','baselines/results'),('gsa_slot','baselines/results'),
      ('kapa_scalar_p2_c16','kapa/results'),('kapa_slot_p2_c16','kapa/results')]
METRICS={'content CE':('confirm','mean_content_ce'), 'Depo CE':('confirm','depo','content_ce'),
         'Brevo CE':('confirm','brevo','content_ce'),'Depo gen %':('generation','depo','valid'),
         'Brevo valid %':('generation','brevo','valid'),'seconds':('elapsed_seconds',)}
PAIRS=[('block_slot_learn_c64','block_slot_c64'),('block_slot_gate_c64','block_slot_c64'),
       ('block_slot_gate_c64','block_slot_learn_c64'),('block_slot_c64','gsa_slot'),
       ('block_slot_c64','gdn'),('block_slot_c64','block_c64'),
       ('block_slot_learn_c32','block_slot_learn_c64'),('kapa_slot_p2_c16','kapa_scalar_p2_c16'),
       ('kapa_slot_p2_c16','block_slot_c64'),('kapa_slot_p2_c16','gdn')]
EXPLORATORY_PAIRS=[('kapa_scalar_p2_c16','block_slot_c64'),('kapa_scalar_p2_c16','gdn')]


def value(row,metric):
    for key in METRICS[metric]:row=row[key]
    return row*(100 if metric.endswith('%') else 1)


def interval(differences,seed=91731,replicates=20000):
    rng=random.Random(seed);n=len(differences)
    means=sorted(statistics.mean(rng.choices(differences,k=n)) for _ in range(replicates))
    def quantile(q):
        pos=q*(len(means)-1);i=int(pos);f=pos-i
        return means[i]*(1-f)+means[min(i+1,len(means)-1)]*f
    return [quantile(.025),quantile(.975)]


def generate(root,allow_partial=False):
    expected=set(range(731,738));arms={};files={};fingerprints={};initial={}
    for arm,sub in ARMS:
        runs={}
        for p in sorted((root/sub).glob(arm+'-seed*/summary.json')):
            r=json.loads(p.read_text());seed=r['config']['seed']
            assert r['status']=='completed' and seed in expected and seed not in runs,p
            assert r['unique_tokens']==10001108 and r['reload_validation_ce_error']<1e-4,p
            assert r['config']['architecture']==arm,p
            assert r['config']['steps']==2231 and r['config']['batch']==64,p
            assert all(r['generation'][t]['n']==128 for t in ('depo','brevo')),p
            runs[seed]=r;files[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
            cfg=r['config'];fingerprints.setdefault(arm,set()).add(json.dumps(cfg['data'],sort_keys=True))
            initial.setdefault(seed,[]).append(cfg['common_init_sha256'])
            for rel,digest in (cfg.get('trials_source_sha256') or cfg.get('ablation_source_sha256') or cfg.get('source_sha256',{})).items():
                if rel.startswith('komp_slot/'):
                    wanted=FROZEN[rel];assert digest==wanted,(p,rel)
        if not allow_partial:assert set(runs)==expected,(arm,sorted(runs))
        if runs:arms[arm]=runs
    all_data=set().union(*fingerprints.values()) if fingerprints else set()
    assert len(all_data)<=1,'Dataset differs across arms'
    for seed,entries in initial.items():assert all(e==entries[0] for e in entries),('common initialization differs',seed)
    table=['| arm | n | '+ ' | '.join(METRICS)+' |','|---|---:|'+ '|'.join('---:' for _ in METRICS)+'|']
    stats={}
    for arm,runs in arms.items():
        cells=[];stats[arm]={}
        for metric in METRICS:
            values=[value(r,metric) for r in runs.values()];mean=statistics.mean(values)
            sd=statistics.stdev(values) if len(values)>1 else None
            stats[arm][metric]=dict(n=len(values),mean=mean,sample_sd=sd)
            digits=1 if metric=='seconds' else 2 if metric.endswith('%') else 4
            cells.append(f'{mean:.{digits}f} ± {sd:.{digits}f}' if sd is not None else f'{mean:.{digits}f} ± NA')
        table.append(f'| {arm} | {len(runs)} | '+' | '.join(cells)+' |')
    paired=['| A − B | metric | paired n | mean difference | bootstrap 95% CI |','|---|---|---:|---:|---|']
    pair_results=[]
    for a,b in PAIRS+EXPLORATORY_PAIRS:
        if a not in arms or b not in arms:continue
        seeds=sorted(set(arms[a])&set(arms[b]))
        if not seeds:continue
        for metric in list(METRICS)[:-1]:
            ds=[value(arms[a][s],metric)-value(arms[b][s],metric) for s in seeds]
            ci=interval(ds);mu=statistics.mean(ds);digits=2 if metric.endswith('%') else 4
            exploratory=(a,b) in EXPLORATORY_PAIRS
            pair_results.append(dict(a=a,b=b,metric=metric,seeds=seeds,differences=ds,mean=mu,ci95=ci,exploratory=exploratory))
            label=f'{a} − {b}'+(' (exploratory)' if exploratory else '')
            paired.append(f'| {label} | {metric} | {len(seeds)} | {mu:+.{digits}f} | [{ci[0]:+.{digits}f}, {ci[1]:+.{digits}f}] |')
    text=('Values are mean ± sample standard deviation across seeds. Generation is percent; paired generation differences are percentage points. '
          'CE: lower is better. Generation: higher is better.\n\n'+'\n'.join(table)+'\n\n'
          'Paired A − B; percentile bootstrap resamples the matched seed pairs, 20,000 draws, fixed seed 91731. '
          'Intervals describe initialization-seed variability conditional on this fixed dataset and evaluation subset; no multiple-comparison correction. '
          'Scalar-KAPA versus Block/GDN contrasts are explicitly exploratory additions after six KAPA seeds were inspected; they were not in the original contrast plan.\n\n'
          +'\n'.join(paired)+'\n')
    (root/'TABLE.md').write_text(text)
    (root/'statistics.json').write_text(json.dumps(dict(arms=stats,paired=pair_results,input_sha256=files,
        seeds=sorted(expected),bootstrap=dict(replicates=20000,seed=91731,method='paired percentile'),
        integrity='complete' if not allow_partial else 'partial'),indent=2))
    return text


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).parent/'block_runs_v4')
    parser.add_argument('--allow-partial',action='store_true');args=parser.parse_args()
    FROZEN={k.removeprefix('./'):v for k,v in json.loads((args.root/'evidence/frozen-verification.json').read_text()).items()}
    print(generate(args.root,args.allow_partial))
