"""Use the unchanged FLA GSA2 layer in the same backbone and LR/WD protocol."""
import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys
import types

import lrwd_experiment as study
import shortconv_experiment as conv
import experiment as base
import torch
from torch import nn

# Restore native exports and compatibility aliases skipped by the operator-only
# loader. Execute upstream initialization unchanged for both Oja2 and GDN2.
module_path = base.ops.FLA_ROOT / 'fla/modules/__init__.py'
module_spec = importlib.util.spec_from_file_location('fla.modules', module_path,
                                                    submodule_search_locations=[str(module_path.parent)])
module_package = sys.modules['fla.modules']
module_package.__file__ = str(module_path)
module_package.__package__ = 'fla.modules'
module_package.__spec__ = module_spec
module_package.__loader__ = module_spec.loader
module_spec.loader.exec_module(module_package)
layer_package = types.ModuleType('fla.layers')
layer_package.__path__ = [str(base.ops.FLA_ROOT / 'fla/layers')]
layer_package.__spec__ = importlib.machinery.ModuleSpec('fla.layers', loader=None, is_package=True)
sys.modules['fla.layers'] = layer_package
from fla.layers.gsa2 import GatedSlotAttention2

ROOT = Path(__file__).resolve().parent / 'gsa2_runs'
PIN = '035f54b64e22abdb640725b3cf13db47d33692c5'
ORIGINAL_SOURCES = study.source_hashes
ORIGINAL_DIMENSIONS = study.dimensions


def dimensions(scale):
    ORIGINAL_DIMENSIONS(scale)
    base.M = 64


class NativeGSA2(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.layer = GatedSlotAttention2(hidden_size=base.D, head_dim=base.K, num_heads=base.H,
                                         num_slots=64, expand_v=1., mode='chunk', layer_idx=layer_idx)

    def forward(self, x):
        return self.layer(x, use_cache=False)[0]


def make_model(kind, seed):
    assert kind == 'gsa2_conv'
    model = base.LM('gdn', seed)
    model.shortconv_architecture = 'gsa2'
    model.use_short_conv = True
    for i, layer in enumerate(model.layers):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 7000 + i)
            layer.mix = NativeGSA2(i)
    return model


def optimizer(model, wd):
    groups = {}
    for name, parameter in model.named_parameters():
        residual = name.endswith(('mix.layer.o_proj.weight', '.down.weight'))
        no_decay = bool(getattr(parameter, '_no_weight_decay', False))
        groups.setdefault((residual, no_decay), []).append(parameter)
    return torch.optim.AdamW([{'params': ps, 'weight_decay': 0. if no_decay else wd,
                              'residual_output': residual}
                             for (residual, no_decay), ps in groups.items()],
                            lr=5e-4, betas=(.9, .95), fused=True)


def sources():
    result = ORIGINAL_SOURCES()
    for name in ('gsa2_experiment.py', 'gsa2_launch.py'):
        result['physics_lm/' + name] = hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
    result['FLA_COMMIT'] = PIN
    for path in sorted((base.ops.FLA_ROOT / 'fla').rglob('*.py')):
        result['upstream/' + str(path.relative_to(base.ops.FLA_ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def preflight():
    ROOT.mkdir(parents=True, exist_ok=True)
    assert sys.modules['fla.modules.l2norm'] is sys.modules['fla.modules.norm.l2norm']
    assert sys.modules['fla.modules'].ShortConvolution is conv.ShortConvolution
    study.dimensions('small')
    sample = study.load('val', 'small')[:4]
    model = make_model('gsa2_conv', 731).cuda()
    reference = base.LM('gdn', 731)
    assert conv.hashes(model, lambda n: '.mix.' not in n) == conv.hashes(reference, lambda n: '.mix.' not in n)
    del reference
    initial_causality = conv.check_model_causality(model, sample)
    no_decay_names = []
    for wd in study.WDS:
        opt = optimizer(model, wd)
        actual = {id(p): g['weight_decay'] for g in opt.param_groups for p in g['params']}
        residual_ids = {id(p) for g in opt.param_groups if g['residual_output'] for p in g['params']}
        assert residual_ids == {id(p) for n, p in model.named_parameters()
                                if n.endswith(('mix.layer.o_proj.weight', '.down.weight'))}
        assert len(residual_ids) == base.L * 2
        for name, p in model.named_parameters():
            expected = 0. if getattr(p, '_no_weight_decay', False) else wd
            assert actual[id(p)] == expected
            if wd == .1 and expected == 0.:
                no_decay_names.append(name)
    assert len(no_decay_names) == base.L * 4
    layer = model.layers[0].mix
    x = torch.randn(2, 129, base.D, device='cuda', requires_grad=True)
    with base.amp():
        y = layer(x)
    grad = torch.autograd.grad(y[:, 63].float().square().sum(), x)[0]
    assert int(torch.count_nonzero(grad[:, 64:])) == 0
    opt = optimizer(model, .1)
    losses = []
    for _ in range(16):
        opt.zero_grad(set_to_none=True)
        with base.amp():
            loss = base.loss_items(model, sample)[0].mean()
        loss.backward()
        assert bool(torch.isfinite(nn.utils.clip_grad_norm_(model.parameters(), 1.)))
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
    info = {'status': 'passed', 'sources': sources(), 'environment': study.environment(),
            'native_layer': 'fla.layers.gsa2.GatedSlotAttention2', 'fla_commit': PIN,
            'initial_causality': initial_causality, 'future_gradient': 0, 'losses': losses,
            'no_weight_decay_names': no_decay_names, 'backbone_initialization': 'matches other architectures',
            'layer_initialization': 'unmodified constructor with named seed',
            'small_params': sum(p.numel() for p in model.parameters()),
            'recurrent_elements_per_head': 8192, 'other_models_recurrent_elements_per_head': 4096,
            'num_slots': 64, 'use_short_conv': True, 'use_w_conv': True}
    del model, opt
    study.dimensions('large')
    large = make_model('gsa2_conv', 731)
    info['large_params'] = sum(p.numel() for p in large.parameters())
    (ROOT / 'preflight.json').write_text(json.dumps(info, indent=2))
    print('GSA2 PREFLIGHT PASSED', info['small_params'], info['large_params'], losses[0], losses[-1], flush=True)


study.ROOT = ROOT
study.dimensions = dimensions
conv.KINDS = ('gsa2',)
conv.make_model = make_model
study.optimizer = optimizer
study.source_hashes = sources
study.CONFIG_EXTRA = lambda model: {
    'native_layer': 'fla.layers.gsa2.GatedSlotAttention2', 'fla_commit': PIN,
    'short_conv': 'native q/k/v/w causal width4 + SiLU, bias=False',
    'wd_exclusions': [n for n, p in model.named_parameters() if getattr(p, '_no_weight_decay', False)],
    'recurrent_elements_per_head': 8192, 'initialization': 'native layer constructor; matched non-mixer backbone',
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=('preflight', 'train', 'select', 'evaluate'), required=True)
    parser.add_argument('--seed', type=int, default=731)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=.1)
    parser.add_argument('--scale', choices=('small', 'large'), default='small')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(9041)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.action == 'preflight': preflight()
    elif args.action == 'select': study.select()
    elif args.action == 'evaluate': study.evaluate_selected(args.scale)
    else: study.train('gsa2', args.seed, args.lr, args.wd, args.scale)
