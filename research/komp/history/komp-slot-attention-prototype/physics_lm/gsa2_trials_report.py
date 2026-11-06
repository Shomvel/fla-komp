"""Supplemental native GSA2 comparisons against frozen v4 runs; no old results overwritten."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import block_report_v4 as old

ROOT=Path(__file__).parent/'block_runs_v4'
ARMS=old.ARMS+[(x,'gsa2/results') for x in ('gsa2_native_lr5e4','gsa2_native_lr1e3','gsa2_noconv_lr5e4')]
PAIRS=[(g,b) for g in ('gsa2_native_lr5e4','gsa2_native_lr1e3') for b in ('block_slot_c64','block_slot_gate_c64','block_slot_learn_c64','kapa_scalar_p2_c16','gdn')]
PAIRS+=[('gsa2_native_lr1e3','gsa2_native_lr5e4'),('gsa2_native_lr5e4','gsa2_noconv_lr5e4'),('gsa2_noconv_lr5e4','block_slot_c64')]


def main(partial=False):
    expected=set(range(731,738));arms={};files={};data=set();backbones={}
    frozen=json.loads((ROOT/'evidence/frozen-verification.json').read_text())
    initial_by_seed={}
    for name,sub in ARMS:
        rows={}
        for p in sorted((ROOT/sub).glob(name+'-seed*/summary.json')):
            r=json.loads(p.read_text());cfg=r['config'];seed=cfg['seed']
            assert seed in expected and seed not in rows and r['status']=='completed'
            assert cfg['architecture']==name and cfg['steps']==2231 and cfg['batch']==64
            assert r['unique_tokens']==10001108 and r['reload_validation_ce_error']<1e-4
            data.add(json.dumps(cfg['data'],sort_keys=True))
            for task in ('depo','brevo'):assert r['generation'][task]['n']==128
            # Shared mixer parameters differ for native GSA2; only compare non-mixer fingerprints.
            shared={k:v for k,v in cfg['common_init_sha256'].items() if '.mix.' not in k}
            initial_by_seed.setdefault(seed,[]).append(shared)
            if name.startswith('gsa2_'):
                assert cfg['slots']==64 and cfg['recurrent_elements_per_head']==8192
                for path,digest in cfg['source_sha256'].items():
                    if path.startswith('komp_slot/'):assert frozen['./'+path]==digest
                bb=cfg['nonmixer_init_sha256'];backbones.setdefault(seed,[]).append(bb)
            rows[seed]=r;files[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        if not partial:assert set(rows)==expected,(name,sorted(rows))
        if rows:arms[name]=rows
    assert len(data)==1
    for entries in initial_by_seed.values():assert all(x==entries[0] for x in entries)
    for entries in backbones.values():assert all(x==entries[0] for x in entries)
    metrics=old.METRICS;summary={}
    table=['| arm | n | '+ ' | '.join(metrics)+' |','|---|---:|'+ '|'.join('---:' for _ in metrics)+'|']
    config=['| arm | params | slots | short conv | recurrent elements/head | peak LR |','|---|---:|---:|---|---:|---:|']
    for name,runs in arms.items():
        cells=[];summary[name]={}
        for metric in metrics:
            values=[old.value(r,metric) for r in runs.values()];mu=statistics.mean(values);sd=statistics.stdev(values) if len(values)>1 else None
            summary[name][metric]=dict(n=len(values),mean=mu,sample_sd=sd)
            digits=1 if metric=='seconds' else 2 if metric.endswith('%') else 4
            cells.append(f'{mu:.{digits}f} ± {sd:.{digits}f}' if sd is not None else f'{mu:.{digits}f} ± NA')
        table.append(f'| {name} | {len(runs)} | '+' | '.join(cells)+' |')
        cfg=next(iter(runs.values()))['config']
        state=cfg.get('recurrent_elements_per_head',2048 if name.startswith('kapa') else 4096)
        config.append(f"| {name} | {cfg['params']} | {cfg['slots']} | {cfg.get('short_conv',False)} | {state} | {cfg.get('peak_lr',.0005):g} |")
    pair_rows=[];pair_table=['| A − B | metric | n | difference | paired bootstrap 95% CI |','|---|---|---:|---:|---|']
    for a,b in PAIRS:
        if a not in arms or b not in arms:continue
        seeds=sorted(set(arms[a])&set(arms[b]))
        for metric in list(metrics)[:-1]:
            diff=[old.value(arms[a][s],metric)-old.value(arms[b][s],metric) for s in seeds]
            mu=statistics.mean(diff);ci=old.interval(diff);digits=2 if metric.endswith('%') else 4
            pair_rows.append(dict(a=a,b=b,metric=metric,seeds=seeds,mean=mu,ci95=ci,differences=diff))
            pair_table.append(f'| {a} − {b} | {metric} | {len(seeds)} | {mu:+.{digits}f} | [{ci[0]:+.{digits}f}, {ci[1]:+.{digits}f}] |')
    text=('Mean ± sample SD across seeds731–737. Percent rates; paired differences in percentage points. '
          'Same data/backbone/training tokens; native GSA2 has a different mixer, at least64 slots, native WD exclusions, and conv unless labelled noconv. '
          'The 1e-3 recipe was selected in the earlier validation search, not this comparison. '
          'Recurrent elements count mixer memory only, excluding convolution caches; KAPA centers are fixed model parameters. '
          'GDN slots is the harness M setting, not a slot dictionary. '
          'Seconds include training, validation and final generation, excluding process startup and checkpoint reload; this is not a kernel benchmark.\n\n'
          +'\n'.join(config)+'\n\n'+'\n'.join(table)+'\n\n'
          'Seed-paired percentile bootstrap, 20,000 resamples, RNG91731; fixed dataset and128 generation cases/task, no multiplicity correction. '
          'All new contrasts are supplemental following the completed Block/KAPA study.\n\n'+'\n'.join(pair_table)+'\n')
    (ROOT/'gsa2/TABLE.md').write_text(text)
    (ROOT/'gsa2/statistics.json').write_text(json.dumps(dict(arms=summary,paired=pair_rows,input_sha256=files,complete=not partial,
        seeds=sorted(expected),bootstrap=dict(replicates=20000,seed=91731,method='paired percentile'),
        scope='Supplemental comparison; intervals condition on fixed data and evaluation cases; no multiplicity correction'),indent=2))
    print(text)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--allow-partial',action='store_true');main(p.parse_args().allow_partial)
