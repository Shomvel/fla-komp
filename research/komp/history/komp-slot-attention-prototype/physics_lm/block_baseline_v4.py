"""Dispatch unchanged baseline harnesses into isolated v4 result directories."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import experiment as base
import decay_experiment as decay
import torch

ROOT=Path(__file__).resolve().parent/os.environ.get('TRIAL_BASELINES','block_runs_v4/baselines')
KINDS=('gdn','gsa','gsa_slot')


def sources():
    root=Path(base.__file__).parents[1]
    paths=[Path(__file__),Path(base.__file__),Path(decay.__file__),root/'benchmark_fla.py',
           *sorted((root/'komp_slot').glob('*.py'))]
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def configure():
    ROOT.mkdir(parents=True,exist_ok=True)
    decay.ROOT=ROOT/'decay'


def preflight():
    # The original data and baseline checks run unchanged, inside the snapshot.
    base.preflight();decay.preflight()
    report=dict(status='passed',kinds=KINDS,source_sha256=sources(),
                base_preflight=json.loads((base.ROOT/'preflight.json').read_text()),
                decay_preflight=json.loads((decay.ROOT/'preflight.json').read_text()))
    (ROOT/'preflight.json').write_text(json.dumps(report,indent=2))


def train(kind,seed):
    ticket=json.loads((ROOT/'preflight.json').read_text())
    assert ticket['status']=='passed' and ticket['source_sha256']==sources()
    def meta(model):
        extra=decay.metadata(model) if kind=='gsa_slot' else {}
        extra.update(trials_source_sha256=sources(),short_conv=False,
                     harness='unchanged decay_experiment' if kind=='gsa_slot' else 'unchanged experiment',
                     weight_decay_policy='decay.bias/A_log/dt_bias=0; others=.1' if kind=='gsa_slot' else 'all parameters .1')
        return extra
    kwargs=dict(result_root=ROOT/'results',config_extra=meta,preflight_path=ROOT/'preflight.json')
    if kind=='gsa_slot':kwargs.update(model_factory=decay.DecayLM,optimizer_factory=decay.optimizer)
    base.train(kind,seed,**kwargs)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true')
    p.add_argument('--kind',choices=KINDS);p.add_argument('--seed',type=int,default=731)
    args=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;configure()
    if args.preflight:preflight()
    else:train(args.kind,args.seed)
