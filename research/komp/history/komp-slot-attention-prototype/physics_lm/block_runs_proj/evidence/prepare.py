from pathlib import Path
import hashlib,json,shutil
P=Path('/Users/shom/Downloads/komp-slot-attention-prototype')
E=P/'physics_lm/block_runs_proj/evidence';S=E/'snapshot';S.mkdir(exist_ok=True)
F=Path('/Users/shom/Downloads/komp-accel-scratch/frozen-20261101-proj')
frozen={}
for line in (F/'SHA256SUMS').read_text().splitlines():
 h,n=line.split();assert hashlib.sha256((F/n).read_bytes()).hexdigest()==h,n
 frozen[n]=h
for name in ('komp_slot','komp_bench','tools'):
 shutil.copytree(F/name,S/name,ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
shutil.copy2(F/'benchmark_fla.py',S/'benchmark_fla.py')
(S/'physics_lm').mkdir(exist_ok=True)
for name in ('experiment.py','block_experiment.py','block_ablation.py','validate_data.py','data-manifest.json'):
 shutil.copy2(P/'physics_lm'/name,S/'physics_lm'/name)
(S/'previous-reference').mkdir(exist_ok=True)
shutil.copy2('/Users/shom/Downloads/komp-accel-scratch/frozen-20261101-coh/komp_slot/block_flex.py',S/'previous-reference/block_flex.py')
(E/'frozen-verification.json').write_text(json.dumps(frozen,indent=2))
manifest={str(p.relative_to(S)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(S.rglob('*')) if p.is_file()}
(E/'snapshot-sha256.json').write_text(json.dumps(manifest,indent=2))
plan=dict(seeds=list(range(731,738)),main=['fifo_proj_c64','fifo_noproj_c64','learned_proj_c64','learned_noproj_c64','swa_w32','swa_w96','swa_w96_nope','full_attn'],
 optional=['fifo_proj_c128','fifo_noproj_c128'],workers=16,optional_after_main=True,source=frozen,
 comparisons=[['fifo_noproj_c64','fifo_proj_c64'],['learned_noproj_c64','learned_proj_c64'],['fifo_proj_c64','swa_w32'],['fifo_proj_c64','swa_w96'],['swa_w96','swa_w96_nope'],['fifo_noproj_c128','fifo_proj_c128']],
 bootstrap=dict(replicates=20000,seed=91731))
(E/'plan.json').write_text(json.dumps(plan,indent=2))
old=P/'physics_lm/block_runs_coh/evidence'
runner=(old/'runner.sh').read_text().replace('komp-coh-trials-20261101','komp-proj-swa-20261101').replace('COH_WORKER','PROJ_WORKER')
(E/'runner.sh').write_text(runner)
for i in range(1,17):
 y=(old/'job-r1.yaml').read_text().replace('komp-coh-trials-20261101','komp-proj-swa-20261101').replace('-r1\n',f'-r{i}\n').replace('COH_WORKER=0',f'PROJ_WORKER={i-1}').replace('/r1/',f'/r{i}/')
 (E/f'job-r{i}.yaml').write_text(y)
submit=(old/'submit_jobs.py').read_text().replace('komp-coh-trials-20261101','komp-proj-swa-20261101').replace('else 1,17','else 1,int(sys.argv[2]) if len(sys.argv)>2 else 17')
(E/'submit_jobs.py').write_text(submit)
(E/'cleanup_jobs.py').write_text((old/'cleanup_jobs.py').read_text().replace('komp-coh-trials-20261101','komp-proj-swa-20261101'))
print(json.dumps(dict(frozen_files=len(frozen),snapshot_files=len(manifest))))
