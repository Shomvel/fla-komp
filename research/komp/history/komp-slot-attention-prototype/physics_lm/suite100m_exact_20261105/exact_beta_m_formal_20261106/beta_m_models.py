"""Fixed-backbone beta/M variants of the frozen Exact model."""
import math
import os
import torch
from torch import nn
from torch.nn import functional as F
from models import LM as OriginalLM, metadata as original_metadata, exact_komp

ALPHA0 = .995
BETA0 = float(os.environ.get('EXACT_BETA0', '.2'))
SLOTS = int(os.environ.get('EXACT_SLOTS', '32'))


class VariableSlotMixer(nn.Module):
    def __init__(self, original, beta0, slots):
        super().__init__()
        assert slots in (32, 64) and beta0 in (.1, .2)
        self.kind = 'exact'
        self.window = original.window
        for name in ('q', 'k', 'v', 'out', 'gate', 'convs', 'norm', 'beta'):
            setattr(self, name, getattr(original, name))
        self.register_buffer('rope_freq', original.rope_freq, persistent=False)
        if slots == 32:
            self.decay = original.decay
            self.keys = original.keys
        else:
            # Preserve the original atoms exactly; append a complete orthogonal basis.
            basis = torch.linalg.qr(original.keys.detach().transpose(-1, -2), mode='complete').Q
            self.keys = nn.Parameter(torch.cat((original.keys.detach(), basis[..., 32:].transpose(-1, -2)), -2))
            with torch.random.fork_rng(devices=[]):
                self.decay = nn.Linear(512, 8 * slots)
        with torch.no_grad():
            self.decay.weight.zero_()
            self.decay.bias.fill_(math.log(ALPHA0 ** 8 / (1 - ALPHA0 ** 8)))
            self.beta.bias.fill_(math.log(beta0 / (1 - beta0)))

    def forward(self, x):
        b, t, _ = x.shape
        h, m, kdim = self.keys.shape
        q, k, v = [conv(proj(x))[0].reshape(b, t, h, kdim)
                   for conv, proj in zip(self.convs, (self.q, self.k, self.v))]
        with torch.autocast('cuda', enabled=False):
            g = F.logsigmoid(F.linear(x.float(), self.decay.weight, self.decay.bias)).reshape(b, t, h, m) / 8
            beta = F.linear(x.float(), self.beta.weight, self.beta.bias).sigmoid()
        keys = self.keys[None].expand(b, h, m, kdim).contiguous()
        values = torch.zeros(b, h, m, kdim, device=x.device)
        out, _, _ = exact_komp(q, k, v, g, beta, (keys, values), eviction='fifo',
                              compute_chunk=16, projection_dtype='float32')
        y = self.norm(out.float()).to(x.dtype).reshape(b, t, h * kdim)
        return self.out(y * F.silu(self.gate(x)))


class LM(OriginalLM):
    def __init__(self, kind, seed, *, beta0=BETA0, slots=SLOTS, **kwargs):
        assert kind == 'exact'
        # This original M32 construction fixes shared parameters and FF1984.
        super().__init__(kind, seed, **kwargs)
        assert self.ff == 1984
        self.beta0, self.slots = beta0, slots
        for layer in self.layers:
            layer.mix = VariableSlotMixer(layer.mix, beta0, slots)


def metadata(model):
    info = original_metadata(model)
    info.update(alpha0=ALPHA0, beta0=model.beta0, block_slots=model.slots,
                experiment='beta/M formal with fixed original M32 backbone',
                persistent_state_bytes_per_sequence=15 * 8 * model.slots * (64 + 64) * 4,
                coefficient_state_bytes_per_sequence=15 * 8 * model.slots * 64 * 4)
    return info


def audit_initialization():
    reference = OriginalLM('exact', 731)
    candidate = LM('exact', 731)
    a, b = reference.state_dict(), candidate.state_dict()
    assert a.keys() == b.keys()
    changed, resized = [], []
    for name, original in a.items():
        target = b[name]
        if original.shape != target.shape:
            assert SLOTS == 64 and name.endswith(('.mix.decay.weight', '.mix.decay.bias', '.mix.keys')), name
            resized.append(name)
            if name.endswith('.mix.keys'):
                assert torch.equal(original, target[:, :32]), name
            elif name.endswith('.mix.decay.weight'):
                assert torch.count_nonzero(target) == 0
        elif not torch.equal(original, target):
            assert name.endswith('.mix.decay.bias') or (BETA0 == .1 and name.endswith('.mix.beta.bias')), name
            changed.append(name)
    for layer in candidate.layers:
        mixer = layer.mix
        torch.testing.assert_close(mixer.keys @ mixer.keys.transpose(-1, -2),
                                   torch.eye(SLOTS).expand(8, SLOTS, SLOTS), rtol=1e-5, atol=1e-5)
        assert torch.count_nonzero(mixer.decay.weight) == torch.count_nonzero(mixer.beta.weight) == 0
        assert torch.equal(mixer.decay.bias, torch.full_like(mixer.decay.bias, math.log(ALPHA0 ** 8 / (1 - ALPHA0 ** 8))))
        assert torch.equal(mixer.beta.bias, torch.full_like(mixer.beta.bias, math.log(BETA0 / (1 - BETA0))))
    info = metadata(candidate)
    assert info['parameters'] == {32: 100526392, 64: 102742072}[SLOTS]
    return dict(status='passed', model=info, changed_same_shape=changed, resized=resized,
                all_other_tensors_identical=True, initial_first32_atoms_identical=True,
                reference_alpha0=.98, reference_beta0=.2, reference_slots=32)
