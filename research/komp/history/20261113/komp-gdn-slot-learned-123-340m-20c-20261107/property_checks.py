"""Source-bound properties of GDN slot + learned eviction + all eight 123 choices.

These diagnostics do not advance the formal training loader or optimizer. FP64
references compose projection matrices explicitly; hard selection is separately
checked for its intentionally absent gradient.
"""
import copy
import itertools
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

import models
from models import ROOT, D, H, M, LearnedSlotMixer, initialize_mixer_options, head_log_decays
from fla.ops.komp.block_fast import block_komp_fast


def _error(name, actual, reference, tolerance=.003, absolute_tolerance=3e-4):
    actual, reference = actual.detach().double(), reference.detach().double()
    delta = actual - reference
    absolute = float(delta.abs().max())
    relative = float(delta.square().mean().sqrt() / reference.square().mean().sqrt().clamp_min(1e-12))
    return dict(tensor=name, absolute=absolute, rmsrel=relative,
                passed=bool(torch.isfinite(actual).all()) and
                (absolute <= absolute_tolerance or relative <= tolerance))


def _linear_reference(x, params, name, low_rank):
    # Compose the effective matrix in FP64, independently of Sequential's two GEMMs.
    weight = params[name + '.1.weight'] @ params[name + '.0.weight'] if low_rank else params[name + '.weight']
    return x @ weight.T


def _formula_reference(x, params, config):
    z = _linear_reference(x, params, 'decay', config.low_rank_gates).reshape(*x.shape[:-1], H, M)
    g = -torch.exp(params['A_log']) * torch.logaddexp(torch.zeros_like(z), z + params['dt_bias'])
    scores = x @ params['score.weight'].T + params['score.bias']
    raw_beta = x @ params['beta.weight'].T + params['beta.bias']
    beta = (1 / (1 + torch.exp(-raw_beta))) * (2 / (1 + torch.exp(-scores)))
    gate = _linear_reference(x, params, 'gate', config.low_rank_gates)
    sigmoid = 1 / (1 + torch.exp(-gate))
    output_gate = sigmoid if config.output_sigmoid else gate * sigmoid
    return g, beta, scores, output_gate


def _formula_gradients(mix, config):
    # Nonzero up weights exercise both factors; initial zero-up behavior is tested
    # by the real full/tail DDP trial, including its first pure-WD down update.
    with torch.no_grad():
        if config.low_rank_gates:
            mix.decay[1].weight.normal_(0, .01)
        else:
            mix.decay.weight.normal_(0, .005)
        mix.A_log.normal_(0, .02)
        mix.dt_bias.add_(torch.randn_like(mix.dt_bias) * .01)
        mix.score.weight.normal_(0, .01)
        mix.score.bias.normal_(0, .01)
    x = (torch.randn(1, 17, D, device='cuda') * .1).requires_grad_()
    params = {name: param for name, param in mix.named_parameters()
              if name in ('A_log', 'dt_bias') or name.startswith(('decay.', 'gate.', 'score.', 'beta.'))}
    ref_x = x.detach().double().requires_grad_()
    ref_params = {name: param.detach().double().requires_grad_() for name, param in params.items()}
    operands = mix.operands(x)
    gate_logits = mix.gate(x)
    gate = gate_logits.sigmoid() if config.output_sigmoid else F.silu(gate_logits)
    actual = (*operands[3:5], operands[7], gate)
    reference = _formula_reference(ref_x, ref_params, config)
    weights = [torch.randn_like(t) for t in actual]
    da = torch.autograd.grad(actual, (x, *params.values()), weights)
    dr = torch.autograd.grad(reference, (ref_x, *ref_params.values()), [w.double() for w in weights])
    rows = [_error(n, a, r) for n, a, r in zip(('g', 'beta', 'score', 'output_gate'), actual, reference)]
    rows += [_error('grad:' + n, a, r) for n, a, r in zip(('input', *params.keys()), da, dr)]
    return dict(rows=rows, passed=all(row['passed'] for row in rows),
                comparison='FP32 production projections against FP64 composed-matrix and analytic gate formulas')


@torch.no_grad()
def _output_path(mix, config, x):
    # Capture the real operator output and the real projection input: this proves
    # the activated option reaches the production mixer output, without a second
    # operator invocation and without replacing its numerical implementation.
    captured = {}
    original = models.block_komp_fast
    def observed_operator(*args, **kwargs):
        result = original(*args, **kwargs)
        captured['operator_output'] = result[0]
        captured['eviction'] = kwargs.get('evict')
        return result
    def projection_input(module, inputs):
        captured['out_input'] = inputs[0]
    hook = mix.out.register_forward_pre_hook(projection_input)
    try:
        models.block_komp_fast = observed_operator
        output = mix._mix(x)
    finally:
        models.block_komp_fast = original
        hook.remove()
    y = mix.norm(captured['operator_output'].float()).to(x.dtype).flatten(-2)
    z = mix.gate(x)
    expected = y * (z.sigmoid() if config.output_sigmoid else F.silu(z))
    row = _error('production_output_projection_input', captured['out_input'], expected,
                 tolerance=1e-6, absolute_tolerance=1e-7)
    row.update(eviction=captured['eviction'], output_finite=bool(torch.isfinite(output).all()))
    row['passed'] = row['passed'] and captured['eviction'] == 'learned' and row['output_finite']
    return row


def _selection_gradient(mix):
    x = torch.randn(1, 129, D, device='cuda') * .1
    with torch.no_grad():
        ops = mix.operands(x)
    xs = [a[:, :, :2].detach().contiguous() for a in ops[:5]]
    xs[0].requires_grad_()
    keys, values = [a[:, :2].detach().contiguous() for a in ops[5:7]]
    scores = ops[7][:, :, :2].detach().contiguous().requires_grad_()
    initial_scores = ops[8][:, :2].detach().contiguous().requires_grad_()
    out, _ = block_komp_fast(*xs, (keys, values), chunk_size=128,
                             evict='learned', scores=scores, initial_scores=initial_scores)
    gradients = torch.autograd.grad(out.square().sum(), (xs[0], scores, initial_scores), allow_unused=True)
    score_leaf = scores.detach().clone().requires_grad_()
    beta_reference = torch.full_like(score_leaf, .2) * (2 * score_leaf.sigmoid())
    write_gradient, = torch.autograd.grad(beta_reference.sum(), score_leaf)
    return dict(hard_selection_score_gradient_absent=gradients[1] is None,
                hard_selection_initial_score_gradient_absent=gradients[2] is None,
                continuous_q_gradient_finite=bool(torch.isfinite(gradients[0]).all()),
                score_write_gate_gradient_min_abs=float(write_gradient.abs().min()),
                passed=gradients[1] is None and gradients[2] is None and
                bool(torch.isfinite(gradients[0]).all()) and bool((write_gradient.abs() > 0).all()))


def run_property_checks(model, source_sha):
    rows = []
    # Preserve rank0 RNG so these diagnostics cannot alter subsequent trial inputs.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        for sigmoid, low_rank, timescales in itertools.product((False, True), repeat=3):
            config = copy.copy(model.config)
            config.output_sigmoid, config.low_rank_gates, config.multi_timescale_init = sigmoid, low_rank, timescales
            torch.manual_seed(20261107)
            mix = LearnedSlotMixer(config, 0)
            initialize_mixer_options(mix, config, 42, 0)
            mix = mix.cuda().eval()
            x = torch.randn(1, 129, D, device='cuda') * .1
            with torch.no_grad():
                ops = mix.operands(x)
                target = (head_log_decays(config.timescale_spread, config.alpha0) if timescales else
                          torch.full((H,), math.log(config.alpha0), dtype=torch.float64)).to(ops[3]).view(1, 1, H, 1)
                initial = dict(head_log_decay_error=float((ops[3] - target).abs().max()),
                               mean_log_alpha_error=abs(float(ops[3].double().mean()) - math.log(config.alpha0)),
                               beta_error=float((ops[4] - config.beta0).abs().max()),
                               score_error=float(ops[7].abs().max()),
                               head_alpha=target.exp().flatten().tolist(),
                               negative_log_decay=bool((ops[3] < 0).all()),
                               initial_score_fixed_zero=not mix.initial_score.requires_grad and bool((mix.initial_score == 0).all()))
                initial['passed'] = (initial['head_log_decay_error'] < 2e-6 and initial['mean_log_alpha_error'] < 2e-6 and
                                     initial['beta_error'] < 2e-6 and initial['score_error'] == 0 and
                                     initial['negative_log_decay'] and initial['initial_score_fixed_zero'])
                rank_contract = dict(enabled=low_rank, passed=True)
                if low_rank:
                    for name in ('gate', 'decay'):
                        module = getattr(mix, name)
                        valid = (isinstance(module, nn.Sequential) and len(module) == 2 and
                                 all(isinstance(layer, nn.Linear) and layer.bias is None for layer in module) and
                                 module[0].out_features == config.gate_rank and module[1].in_features == config.gate_rank)
                        rank_contract[name] = dict(rank_upper_bound=config.gate_rank, no_intermediate_nonlinearity=valid,
                                                   no_bias=all(layer.bias is None for layer in module))
                        rank_contract['passed'] &= valid
                    rank_contract['initial_decay_up_zero'] = bool((mix.decay[1].weight == 0).all())
                    rank_contract['initial_decay_down_nonzero'] = bool((mix.decay[0].weight != 0).any())
                    rank_contract['passed'] &= rank_contract['initial_decay_up_zero'] and rank_contract['initial_decay_down_nonzero']
                output_path = _output_path(mix, config, x)
                gate_values = mix.gate(x)
                if sigmoid:
                    gate_values = gate_values.sigmoid()
                    output_path['sigmoid_range_valid'] = bool(((gate_values >= 0) & (gate_values <= 1)).all())
                    output_path['passed'] &= output_path['sigmoid_range_valid']
            formula = _formula_gradients(mix, config)
            selection = _selection_gradient(mix) if sigmoid and low_rank and timescales else None
            row = dict(options=dict(output_sigmoid=sigmoid, low_rank_gates=low_rank, multi_timescale_init=timescales),
                       initial=initial, low_rank_contract=rank_contract, production_output=output_path,
                       fp64_gate_comparison=formula, selection_gradient=selection)
            row['passed'] = all((initial['passed'], rank_contract['passed'], output_path['passed'], formula['passed'])) and (selection is None or selection['passed'])
            rows.append(row)
            # Persist every completed case before an assertion can terminate the job.
            report = dict(status='running' if row['passed'] else 'failed', source_sha=source_sha, cases=rows,
                          scope='eight single-mixer configurations; full-model DDP trial separately in preflight',
                          selection_gradient='hard ranking detached; score learns through write strength',
                          bitwise_equivalence_claimed=False)
            (ROOT / 'evidence/properties-block.json').write_text(json.dumps(report, indent=2))
            assert row['passed'], row
            del mix
    report['status'] = 'passed'
    (ROOT / 'evidence/properties-block.json').write_text(json.dumps(report, indent=2))
    print('GDN_SLOT_LEARNED_123_PROPERTIES_PASS', flush=True)
    return report
