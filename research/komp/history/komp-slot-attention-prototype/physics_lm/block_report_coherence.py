"""Generate coherence tables solely from collected summaries and diagnostic receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import block_report_v4 as common

ROOT = Path(__file__).resolve().parent/'block_runs_coh'
E1 = ['coh_fifo_off','coh_fifo_99','coh_fifo_95','coh_fifo_80']
SEEDS = set(range(731,738))


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def summary(values):
    return dict(n=len(values), mean=statistics.mean(values), sample_sd=statistics.stdev(values) if len(values)>1 else None)
def cell(values,digits=4):
    r=summary(values)
    return f"{r['mean']:.{digits}f} ± {r['sample_sd']:.{digits}f}" if r['sample_sd'] is not None else f"{r['mean']:.{digits}f} ± NA"


def main(partial=False, original_only=False):
    selection=read(ROOT/'evidence/selection.json') if (ROOT/'evidence/selection.json').exists() else None
    arms=E1+(['coh_gate_off','coh_gate_'+selection['selected_tag'],'coh_learned_off','coh_learned_'+selection['selected_tag']] if selection else [])
    if not partial: assert selection and selection['status']=='locked'
    fine_selection=read(ROOT/'evidence/fine-selection.json') if (ROOT/'evidence/fine-selection.json').exists() else None
    fine_screen=read(ROOT/'evidence/fine-screening.json') if (ROOT/'evidence/fine-screening.json').exists() else None
    if not original_only and (ROOT/'fine/preflight.json').exists():
        arms += ['coh_fifo_995','coh_fifo_999']
        if fine_selection and fine_selection['selected_tag']!='99' and not original_only:
            arms += [f"coh_{f}_{fine_selection['selected_tag']}" for f in ('gate','learned')]
        if not partial:assert fine_selection and fine_screen
    def armroot(arm):return ROOT/'fine' if arm.rsplit('_',1)[1] in ('995','999') else ROOT
    data={};files={};initial={};datasets=set();stats={};merges={}
    source=read(ROOT/'preflight.json')['source_sha256']
    table=['| arm | n | '+' | '.join(common.METRICS)+' |','|---|---:|'+'---:|'*len(common.METRICS)]
    merge_table=['| arm | layer | n | first 224 steps merge % | last 224 steps merge % |','|---|---:|---:|---:|---:|']
    for arm in arms:
        runs={};merge_runs={}
        directory=armroot(arm)
        arm_source=read(directory/'preflight.json')['source_sha256']
        for path in sorted((directory/'results').glob(arm+'-seed*/summary.json')):
            r=read(path);cfg=r['config'];seed=cfg['seed'];assert seed in SEEDS and seed not in runs
            assert cfg['architecture']==arm and r['status']=='completed'
            assert cfg['source_sha256']==arm_source
            assert cfg['implementation']=='block_flex FP32 for every arm; BF16 surrounding backbone'
            assert cfg['steps']==2231 and r['unique_tokens']==10001108 and cfg['batch']==64
            assert r['reload_validation_ce_error']<1e-4
            assert all(r['generation'][t]['n']==128 for t in ('depo','brevo'))
            initial.setdefault(seed,[]).append(cfg['common_init_sha256']);datasets.add(json.dumps(cfg['data'],sort_keys=True))
            runs[seed]=r;files[str(path.relative_to(ROOT))]=sha(path)
            mpath=directory/f'{arm}-seed{seed}-coherence.json'
            if partial and not mpath.exists(): continue
            m=read(mpath);assert m['status']=='completed' and m['training_tokens']==10001108 and m['steps']==2231
            for win in ('first10pct','last10pct'):
                assert len(m['windows'][win])==4
                for row in m['windows'][win].values():
                    assert row['calls']==224 and 0<=row['merged']<=row['tokens']
                    assert row['fraction']==row['merged']/row['tokens']
            merge_runs[seed]=m;files[str(mpath.relative_to(ROOT))]=sha(mpath)
        if not partial:
            wanted={731} if arm in ('coh_fifo_995','coh_fifo_999') and arm.rsplit('_',1)[1] not in fine_screen['expand_tags'] else SEEDS
            assert set(runs)==set(merge_runs)==wanted,(arm,sorted(runs))
        if not runs: continue
        data[arm]=runs;merges[arm]={};stats[arm]={};cells=[]
        for metric in common.METRICS:
            vals=[common.value(r,metric) for r in runs.values()];stats[arm][metric]=summary(vals)
            cells.append(cell(vals,1 if metric=='seconds' else 2 if metric.endswith('%') else 4))
        table.append(f'| {arm} | {len(runs)} | '+' | '.join(cells)+' |')
        for layer in range(4):
            cols=[];merges[arm][str(layer)]={}
            for window in ('first10pct','last10pct'):
                vals=[m['windows'][window][str(layer)]['fraction']*100 for m in merge_runs.values()]
                if vals:
                    merges[arm][str(layer)][window]=summary(vals);cols.append(cell(vals,2))
                else:cols.append('NA')
            merge_table.append(f'| {arm} | {layer} | {len(merge_runs)} | '+' | '.join(cols)+' |')
    assert len(datasets)==1
    for entries in initial.values(): assert all(x==entries[0] for x in entries)
    comparisons=[(a,'coh_fifo_off') for a in E1[1:]]
    interactions=[]
    if selection:
        tag=selection['selected_tag']
        comparisons += [(f'coh_{family}_{tag}',f'coh_{family}_off') for family in ('gate','learned')]
        comparisons += [(f'coh_{family}_{tag}',f'coh_fifo_{tag}') for family in ('gate','learned')]
        for family in ('gate','learned'):
            interactions.append((f'coh_{family}_{tag}',f'coh_{family}_off',f'coh_fifo_{tag}','coh_fifo_off'))
    if fine_screen and not original_only:
        comparisons += [(f'coh_fifo_{tag}','coh_fifo_off') for tag in fine_screen['expand_tags']]
    if fine_selection and fine_selection['selected_tag']!='99' and not original_only:
        tag=fine_selection['selected_tag']
        comparisons += [(f'coh_{f}_{tag}',f'coh_{f}_off') for f in ('gate','learned')]
        comparisons += [(f'coh_{f}_{tag}',f'coh_{f}_99') for f in ('gate','learned')]
        comparisons += [(f'coh_{f}_{tag}',f'coh_fifo_{tag}') for f in ('gate','learned')]
        interactions += [(f'coh_{f}_{tag}',f'coh_{f}_off',f'coh_fifo_{tag}','coh_fifo_off') for f in ('gate','learned')]
    paired=[];pairtable=['| contrast | metric | n | mean difference | paired bootstrap 95% CI |','|---|---|---:|---:|---|']
    for terms in comparisons+interactions:
        if any(a not in data for a in terms): continue
        seeds=sorted(set.intersection(*(set(data[a]) for a in terms)))
        if not seeds:continue
        name=f'{terms[0]} − {terms[1]}' if len(terms)==2 else f'({terms[0]} − {terms[1]}) − ({terms[2]} − {terms[3]})'
        for metric in list(common.METRICS)[:-1]:
            diffs=[sum(sign*common.value(data[arm][s],metric) for sign,arm in zip((1,-1,-1,1),terms)) for s in seeds]
            mu=statistics.mean(diffs);ci=common.interval(diffs);digits=2 if metric.endswith('%') else 4
            paired.append(dict(terms=terms,name=name,metric=metric,seeds=seeds,mean=mu,ci95=ci,differences=diffs))
            pairtable.append(f'| {name} | {metric} | {len(seeds)} | {mu:+.{digits}f} | [{ci[0]:+.{digits}f}, {ci[1]:+.{digits}f}] |')
    if selection:
        recomputed={a:statistics.mean(r['confirm']['mean_content_ce'] for r in data[a].values()) for a in E1}
        assert all(len(data[a])==7 for a in E1)
        best=min(E1[1:],key=lambda a:recomputed[a]);assert best=='coh_fifo_'+selection['selected_tag']
    text=('Mean ± sample SD; seeds731–737. CE lower is better; generation percent higher is better. '
          'All arms use the same frozen FP32 flex kernel and10,001,108 training tokens.\n\n'+'\n'.join(table)+'\n\n'
          'Seed-paired percentile bootstrap:20,000 resamples, RNG91731. Generation differences are percentage points. '
          'Fixed data and evaluation cases; no multiplicity correction. Selected-threshold intervals do not account for threshold selection. '
          'Difference-in-differences are supplemental interaction diagnostics.\n\n'+'\n'.join(pairtable)+'\n\n'
          'Merge fractions: nonpadding token-head weighted within each run; mean ± sample SD across seeds. '
          'Only optimizer steps1–224 and2008–2231, no evaluation forwards.\n\n'+'\n'.join(merge_table)+'\n')
    (ROOT/('ORIGINAL_TABLE.md' if original_only else 'TABLE.md')).write_text(text)
    result=dict(complete=not partial,fine_selection=fine_selection,fine_screening=fine_screen,arms=stats,paired=paired,merges=merges,selection=selection,input_sha256=files,
        bootstrap=dict(replicates=20000,seed=91731,method='paired percentile'))
    (ROOT/('original-statistics.json' if original_only else 'statistics.json')).write_text(json.dumps(result,indent=2))
    diagnostics={};dtable=['| arm | layer | block-head observations | merged % | Gram condition median | Gram condition max | cos>0.95 pairs mean / max | coefficient Frobenius median / max |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in arms:
        path=armroot(arm)/f'{arm}-seed731-diagnostic.json'
        if not path.exists():
            if not partial:raise AssertionError(path)
            continue
        d=read(path);assert d['status']=='passed' and d['samples']==64
        diagnostics[arm]={}
        for layer in range(4):
            rows=[r for r in d['rows'] if r['layer']==layer];counts=[r for r in d['merge_counts'] if r['layer']==layer]
            assert len(rows)==2*sum((n+63)//64 for n in d['input_lengths'])
            assert sum(r['tokens'] for r in counts)==2*sum(d['input_lengths'])
            vals=[r['condition'] for r in rows];norms=[r['coefficient_frobenius'] for r in rows];pairs=[r['cosine_pairs_gt95'] for r in rows]
            row=dict(n=len(rows),merge_fraction=sum(r['merged'] for r in counts)/sum(r['tokens'] for r in counts),
                condition_median=statistics.median(vals),condition_max=max(vals),pairs_mean=statistics.mean(pairs),pairs_max=max(pairs),
                coefficient_median=statistics.median(norms),coefficient_max=max(norms))
            diagnostics[arm][str(layer)]=row
            dtable.append(f"| {arm} | {layer} | {row['n']} | {100*row['merge_fraction']:.2f} | {row['condition_median']:.2f} | {row['condition_max']:.2f} | {row['pairs_mean']:.2f} / {row['pairs_max']} | {row['coefficient_median']:.4f} / {row['coefficient_max']:.4f} |")
    dtext=('E3:seed731 only, first64 confirm full documents, no padding. Every real block end including final partial blocks; '
           'Gram=exp((K Kᵀ−1)/0.5)+1e-4 I. FP32 frozen state measured in FP64. '
           'Pairs count unordered off-diagonal pairs once; coefficient Frobenius norm per head. No TF32 experiment.\n\n'+'\n'.join(dtable)+'\n')
    (ROOT/('ORIGINAL_DIAGNOSTICS.md' if original_only else 'DIAGNOSTICS.md')).write_text(dtext);(ROOT/('original-diagnostics-statistics.json' if original_only else 'diagnostics-statistics.json')).write_text(json.dumps(diagnostics,indent=2))
    print(text);print(dtext)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--allow-partial',action='store_true');p.add_argument('--original-only',action='store_true');args=p.parse_args();main(args.allow_partial,args.original_only)
