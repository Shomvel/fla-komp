"""CPU adapter contract; CUDA tests separately cover the real frozen 340M modules."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from komp_inference.model import RecurrentKOMPLM


class Convolution(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(width, 1, 4)*.03)

    def forward(self, x, cache=None, output_final_state=False):
        b, _, d = x.shape
        cache = x.new_zeros(b, d, 4) if cache is None else cache
        history = torch.cat((cache, x.transpose(1, 2)), -1)
        y = F.conv1d(history[:, :, 1:], self.weight, groups=d)
        return F.silu(y.transpose(1, 2)), history[:, :, -4:].clone()


class Mixer(nn.Module):
    def __init__(self, kind, axis, learned):
        super().__init__()
        d, h, k, m = 8, 2, 4, 4
        self.kind = kind
        self.config = SimpleNamespace(output_sigmoid=learned, kind=kind)
        self.keys = nn.Parameter(torch.randn(h, m, k))
        for name in ('q', 'k', 'v', 'out', 'gate'):
            setattr(self, name, nn.Linear(d, d, bias=False))
        self.convs = nn.ModuleList([Convolution(d) for _ in range(3)])
        self.norm = nn.RMSNorm(k)
        self.decay = nn.Linear(d, h*(m if axis == 'slot' else k))
        self.beta = nn.Linear(d, h)
        if learned:
            self.score = nn.Linear(d, h)
            self.A_log = nn.Parameter(torch.zeros(h, m if axis == 'slot' else k))
            self.dt_bias = nn.Parameter(torch.full_like(self.A_log, -4.))
            self.register_buffer('initial_score', torch.zeros(h, m))


def model(kind, axis, learned):
    torch.manual_seed(42)
    root = nn.Module()
    root.config = SimpleNamespace(fuse_norm=False, kind=kind, output_sigmoid=learned)
    root.model = nn.Module()
    root.model.embeddings = nn.Embedding(100, 8)
    root.model.layers = nn.ModuleList()
    for _ in range(2):
        layer = nn.Module()
        layer.attn = Mixer(kind, axis, learned)
        layer.attn_norm, layer.mlp_norm = nn.RMSNorm(8), nn.RMSNorm(8)
        layer.mlp = nn.Sequential(nn.Linear(8, 12), nn.SiLU(), nn.Linear(12, 8))
        root.model.layers.append(layer)
    root.model.norm = nn.RMSNorm(8)
    root.lm_head = nn.Linear(8, 100, bias=False)
    return root.eval()


@pytest.mark.parametrize('kind,axis,learned', [('exact', 'slot', False), ('block', 'slot', False),
                                             ('block', 'slot', True), ('block', 'value', True)])
def test_model_cache_decode_and_reset(kind, axis, learned):
    original = model(kind, axis, learned)
    adapter = RecurrentKOMPLM(original, decay_axis=axis)
    weights = {name: tensor.clone() for name, tensor in original.state_dict().items()}
    ids = torch.randint(0, 100, (2, 131))
    expected, whole = adapter(ids)
    first, cache = adapter(ids[:, :127])
    rest, cache = adapter(ids[:, 127:], cache=cache)
    torch.testing.assert_close(torch.cat((first, rest), 1), expected, atol=1e-5, rtol=1e-5)
    assert cache.position == whole.position == 131
    reorder = torch.tensor([1, 0, 1])
    more = torch.tensor([[3], [4]])
    baseline, _ = adapter(torch.cat((ids, more), 1), logits_to_keep=1)
    actual, _ = adapter(more[reorder], cache=cache.index_select(reorder), logits_to_keep=1)
    torch.testing.assert_close(actual, baseline[reorder], atol=1e-5, rtol=1e-5)
    cache.reset()
    reset, _ = adapter(ids, cache=cache)
    torch.testing.assert_close(reset, expected, atol=0, rtol=0)
    assert cache.nbytes() > 0
    for name, tensor in original.state_dict().items():
        torch.testing.assert_close(tensor, weights[name], atol=0, rtol=0)


def test_model_rejects_padding_and_training():
    original = model('block', 'slot', False)
    with pytest.raises(ValueError, match='eval'):
        RecurrentKOMPLM(original.train())
    adapter = RecurrentKOMPLM(original.eval())
    with pytest.raises(ValueError, match='padded'):
        adapter(torch.ones(2, 4, dtype=torch.long), attention_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]))
