"""Each task runs three isolated candidates; every stage is source-bound and gated."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
import triton
import google.protobuf
from google.protobuf.internal.api_implementation import Type

ROOT = Path(__file__).resolve().parent
TASK = os.environ['GROUP_TASK']
plan = json.loads((ROOT / 'plan.json').read_text())
manifest = json.loads((ROOT / 'source-manifest.json').read_text())
for name, wanted in manifest['files'].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == wanted, name
for name, wanted in json.loads((ROOT / 'evidence/snapshot-sha256.json').read_text()).items():
    assert hashlib.sha256((ROOT / 'snapshot' / name).read_bytes()).hexdigest() == wanted, name
assert torch.cuda.device_count() == 8
(ROOT / f'evidence/environment-{TASK}.json').write_text(json.dumps(dict(
    gpu=[torch.cuda.get_device_name(i) for i in range(8)], torch=str(torch.__version__),
    triton=triton.__version__, cuda=torch.version.cuda, protobuf=google.protobuf.__version__,
    protobuf_backend=Type()), indent=2) + '\n')


def run(command, env):
    assert not (ROOT / 'FAILED.json').exists(), 'Peer failure blocks new stages'
    result = subprocess.run(command, cwd=ROOT, env=env)
    if result.returncode:
        (ROOT / 'FAILED.json').write_text(json.dumps(dict(task=TASK, command=command,
            variant=env['VARIANT_ROOT'], exit_code=result.returncode)) + '\n')
        raise SystemExit(result.returncode)


for candidate in plan['new_candidates']:
    beta, slots = candidate['beta0'], candidate['slots']
    variant = ROOT / 'variants' / f'{TASK}-beta{beta:g}-m{slots}'
    (variant / 'evidence').mkdir(parents=True, exist_ok=True)
    for name in (*manifest['files'], 'source-manifest.json', 'snapshot'):
        path = variant / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.symlink_to(ROOT / name)
    env = dict(os.environ, VARIANT_ROOT=str(variant), EXACT_BETA0=str(beta), EXACT_SLOTS=str(slots),
               SUITE_RUNS=str(variant / 'runs-ddp'))
    name = f'{TASK}-exact-beta{beta:g}-m{slots}-final-lr{plan["lr"][TASK]:g}-wd0.1-seed731'
    summary = variant / 'runs-ddp/results' / name / 'summary.json'
    if summary.exists():
        assert json.loads(summary.read_text())['status'] == 'completed'
        continue
    print('CANDIDATE', name, flush=True)
    if not (variant / 'evidence/numerical.json').exists():
        run([sys.executable, str(ROOT / 'numerical_preflight.py')], dict(env, CUDA_VISIBLE_DEVICES='0'))
    benchmark = variant / f'evidence/benchmark-beta{beta:g}-m{slots}.json'
    if not benchmark.exists():
        run([sys.executable, str(ROOT / 'benchmark.py')], dict(env, CUDA_VISIBLE_DEVICES='0'))
    if beta == .1 and slots == 32:
        baseline = variant / 'evidence/benchmark-beta0.2-m32.json'
        if not baseline.exists():
            run([sys.executable, str(ROOT / 'benchmark.py')], dict(env, EXACT_BETA0='.2', CUDA_VISIBLE_DEVICES='0'))
    prefix = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8', str(ROOT / 'ddp_train.py')]
    if not (variant / 'runs-ddp/ddp-preflight/exact.json').exists():
        run(prefix + ['--preflight', '--kind', 'exact'], env)
    run(prefix + ['--kind', 'exact', '--task', TASK, '--seed', '731', '--lr', str(plan['lr'][TASK]), '--wd', '.1'], env)
    assert summary.exists() and json.loads(summary.read_text())['status'] == 'completed'
(ROOT / f'evidence/training-{TASK}-complete.json').write_text(json.dumps(dict(status='completed', candidates=plan['new_candidates'])) + '\n')
