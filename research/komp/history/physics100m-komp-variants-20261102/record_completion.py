"""Seal completion only after full scores, audit, archive and exact cleanup."""
import hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;E=ROOT/'evidence';C=E/'cleanup-final-20261102T1049Z'
d=json.loads((E/'monitor-20261102T1049Z.json').read_text());audit=json.loads((E/'final-generation-audit.json').read_text());archive=json.loads((E/'final-archive-verification.json').read_text())
assert len(d['final'])==30 and all(x['heldout'] and x['generations'] for x in d['final'])
assert audit['status']=='passed' and audit['records']==7680 and not audit['mismatches']
assert archive['summaries']==165 and archive['heldout']==archive['generations']==30 and archive['no_checkpoints']
assert json.loads((C/'all-absent.json').read_text())==[]
identities={}
for i in range(1,13):
 terminal=json.loads((C/f'r{i}-terminal.json').read_text());assert len(terminal)==1 and terminal[0]['status']=='Completed'
 assert json.loads((C/f'r{i}-absent.json').read_text())==[]
 assert (C/f'r{i}-delete.json').stat().st_size
 identities[terminal[0]['user_def_name']]=terminal[0]['name']
manifest=json.loads((E/'SOURCE_MANIFEST.json').read_text());verified=json.loads((E/'source-cache-final.json').read_text());assert all(manifest[k]==verified[k] for k in manifest)
base=json.loads((ROOT.parent/'physics100m-20261101/evidence/BASELINE_COMPLETE.json').read_text())
paths=[ROOT/'FINAL_RESULTS.md',ROOT/'results/latest-all/summary.json',ROOT/'results/latest-all/TABLE.md',E/'monitor-20261102T1049Z.json',E/'final-generation-audit.json',E/'final-archive-verification.json',E/'source-cache-final.json',C/'all-absent.json']
receipt=dict(status='complete',at=time.time(),final_runs=30,complete_evaluations=30,tuning_runs=135,baseline_formal_runs=60,total_formal_runs=90,generation_records_audited=7680,total_generation_records_audited=23040,source_hashes_unchanged=True,archive=archive,deleted_job_identities=identities,cleanup=str(C),all_owned_jobs_absent=True,evidence_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},automation='Ready to delete physicslm-100m; tool receipt recorded separately')
(E/'VARIANTS_COMPLETE.json').write_text(json.dumps(receipt,indent=2))
state=json.loads((E/'monitor-state.json').read_text());state.update(checked_at=time.time(),reported_final=sorted(x['name'] for x in d['final']),reported_evaluations=sorted(x['name'] for x in d['final']),heldout_completed=30,final_completed=30,formal_completed=30,decision='NOTIFY',running_jobs=0,active_workers=[],active_formal=[],active_evaluations=[],current_job_identities={},last_job_identities=identities,last_snapshot='evidence/monitor-20261102T1049Z.json',jobs_snapshot='evidence/cleanup-final-20261102T1049Z/all-absent.json',cache_evidence='evidence/source-cache-final.json',last_generation_audit='evidence/final-generation-audit.json',note='All30variant formal runs and full evals complete;7680generation records independently rescored with zero mismatch. Full90run report saved.165training summaries and all logs/JSON archived without checkpoints.12jobsCompleted,exactdeletedandconfirmedabsent; original16remainintentionallydeleted. CompletionreceiptVARIANTS_COMPLETE.json. Stop automation.')
temp=E/'monitor-state.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(E/'monitor-state.json')
with (ROOT/'STATUS.md').open('a') as f:f.write('''

## COMPLETE, Nov2 18:49 China monitoring cycle

All30/30variant formal100Mtraining runs and all30heldout+256generation evaluations completed. Original60+new30=90complete runs. Script-generated `FINAL_RESULTS.md` and `results/latest-all/` contain both-seed validation/testCE, generation scores, hyperparameters, paired comparisons, means and ranks. Independent scorer verified7680variant records without mismatch (23040including baseline). No significance claim from two seeds; CE not independently recomputed.

FinalLano mean generation: learned10.546875%,GDN-slot7.03125%,combined7.6171875%,originalKOMP11.328125%. GDN-slot seed731/732 differs substantially(13.671875%/0.390625%), so mean does not imply consistency. Overall per-task mean generation winners: learnedMano20.8984375%,originalKOMPLano11.328125%,GDNDepo2 6.4453125%,learnedBrevo2 20.703125%,GDN-slotCapo80.859375%.

`final_archive/` preserves1285JSON/log/source files with per-fileSHA256,165training summaries,15selections,30heldoutandgenerations; no*.ptcopied. Source manifest verified unchanged. All12variant jobs observedCompleted0/1,exactdeletedandconfirmedabsent; receipts `evidence/cleanup-final-20261102T1049Z/`. All16baseline aliases were alreadydeleted; do not recreate any. Final completionreceipt `evidence/VARIANTS_COMPLETE.json`. Monitoring is no longer needed and is being deleted via the app tool.
''')
print(json.dumps(dict(status='complete',runs=90,variant_audit_records=7680,cleaned_jobs=12)))
