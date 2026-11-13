"""Run one isolated GPU validation attempt, serially, stopping at the first failure."""

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from verify_sources import ROOT, fingerprint, plan, verify_candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--base', default='/gpfs/users/linlizhi')
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--stage', choices=('operators', 'models', 'fixtures', 'benchmark', 'all'), default='operators')
    parser.add_argument('--operator-attempt')
    parser.add_argument('--acceptance-attempt')
    parser.add_argument('--fixtures')
    args = parser.parse_args()
    if not args.attempt.replace('-', '').replace('_', '').isalnum():
        raise ValueError('attempt must contain only letters, digits, hyphen and underscore')
    candidate = verify_candidate()
    order = plan()['order']
    requested = set(args.models or order)
    if requested-set(order):
        raise ValueError(f'unknown models: {sorted(requested-set(order))}')
    models = [name for name in order if name in requested]
    stages = ['operators', 'models', 'fixtures', 'benchmark'] if args.stage == 'all' else [args.stage]
    if 'fixtures' in stages and not args.fixtures:
        raise ValueError('fixture stage requires the frozen fixture path')
    # root must expose the verified idle GPU explicitly; this script never chooses or claims a GPU
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('root must set CUDA_VISIBLE_DEVICES to the independently verified idle GPU')
    import torch
    if torch.cuda.device_count() != 1:
        raise ValueError('exactly one visible GPU is required')
    attempt = ROOT/'attempts'/args.attempt
    attempt.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    record = dict(status='running', candidate_source=candidate, models=models, stages=stages,
                  started=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  cuda_visible_devices=env['CUDA_VISIBLE_DEVICES'], gpu=torch.cuda.get_device_name(), commands=[])
    target = attempt/'receipt.json'
    target.write_text(json.dumps(record, indent=2)+'\n')
    try:
        for stage in stages:
            output = attempt/stage
            output.mkdir()
            for name in models:
                if stage == 'operators':
                    command = [sys.executable, str(ROOT/'gpu_validate.py'), '--operators-only', '--model', name,
                               '--base', args.base, '--evidence-dir', str(output)]
                elif stage == 'models':
                    prior = Path(args.operator_attempt) if args.operator_attempt else attempt
                    operator = prior/'operators'/f'gpu-{name}.json'
                    command = [sys.executable, str(ROOT/'gpu_validate.py'), '--model-only', '--model', name,
                               '--base', args.base, '--evidence-dir', str(output), '--operator-receipt', str(operator)]
                else:
                    prior = Path(args.acceptance_attempt) if args.acceptance_attempt else attempt
                    acceptance = prior/'models'/f'gpu-{name}.json'
                    command = [sys.executable, str(ROOT/('fixture_validate.py' if stage == 'fixtures' else 'benchmark.py')),
                               '--model', name, '--base', args.base, '--acceptance', str(acceptance),
                               '--evidence-dir', str(output)]
                    if stage == 'fixtures':
                        if name == 'value_learned123':
                            # input-only fixtures cannot establish an observed-golden pass
                            record['commands'].append(dict(stage=stage, model=name, status='not_applicable_no_golden'))
                            continue
                        command += ['--fixtures', args.fixtures]
                verify_candidate()
                item = dict(stage=stage, model=name, command=command, status='running')
                record['commands'].append(item)
                target.write_text(json.dumps(record, indent=2)+'\n')
                print(json.dumps(item), flush=True)
                log = output/f'{name}.log'
                with log.open('x') as stream:
                    result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                item.update(exit_code=result.returncode, log_sha256=fingerprint(log))
                if result.returncode:
                    item['status'] = 'failed'
                    raise RuntimeError(f'{stage}/{name} failed; stopped without retry, see {log}')
                item['status'] = 'passed'
                target.write_text(json.dumps(record, indent=2)+'\n')
        record['status'] = 'passed'
    except Exception as error:
        record['status'], record['error'] = 'failed', repr(error)
        raise
    finally:
        record['finished'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        target.write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
