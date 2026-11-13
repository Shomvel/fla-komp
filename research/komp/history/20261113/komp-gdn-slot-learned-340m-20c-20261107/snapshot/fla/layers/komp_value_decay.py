# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""FLA mixer with value-coordinate decay and FIFO Exact/Block compression."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from fla.modules import ShortConvolution
from fla.ops.komp import block_komp_value_decay, exact_komp_value_decay


class KOMPValueDecayMixer(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.config, self.layer_idx = config, layer_idx
        self.kind, self.checkpoint_mixer = config.kind, config.checkpoint_mixer
        d, h, k, m = config.hidden_size, config.num_heads, config.head_dim, config.num_slots
        for name in ("q", "k", "v", "out"):
            setattr(self, name, nn.Linear(d, d, bias=False))
        self.convs = nn.ModuleList(
            [ShortConvolution(d, config.conv_size, bias=False, activation="silu", backend="triton") for _ in range(3)]
        )
        self.norm = nn.RMSNorm(k, eps=config.norm_eps)
        if config.low_rank_gates:
            self.gate = nn.Sequential(nn.Linear(d, config.gate_rank, bias=False), nn.Linear(config.gate_rank, d, bias=False))
            self.decay = nn.Sequential(nn.Linear(d, config.gate_rank, bias=False), nn.Linear(config.gate_rank, d, bias=True))
            decay_up = self.decay[1]
        else:
            self.gate = nn.Linear(d, d, bias=False)
            self.decay = nn.Linear(d, d)
            decay_up = self.decay
        self.beta = nn.Linear(d, h)
        self.keys = nn.Parameter(torch.stack([torch.linalg.qr(torch.randn(k, m)).Q.T for _ in range(h)]))
        factors = torch.ones(h)
        if config.multi_timescale_init:
            factors = torch.logspace(-math.log10(config.timescale_spread), math.log10(config.timescale_spread), h)
            factors = factors / factors.mean()
        log_p = (8 * math.log(config.alpha0) * factors[:, None]).expand(h, k).reshape(d)
        nn.init.zeros_(decay_up.weight)
        with torch.no_grad():
            decay_up.bias.copy_(log_p - torch.log(-torch.expm1(log_p)))
        nn.init.zeros_(self.beta.weight)
        nn.init.constant_(self.beta.bias, math.log(config.beta0 / (1 - config.beta0)))

    def operands(self, x):
        b, t, _ = x.shape
        h, k, m = self.config.num_heads, self.config.head_dim, self.config.num_slots
        q, key, v = [conv(proj(x))[0].reshape(b, t, h, k) for conv, proj in zip(self.convs, (self.q, self.k, self.v))]
        with torch.autocast(x.device.type, enabled=False):
            if self.config.low_rank_gates:
                logits = F.linear(
                    F.linear(x.float(), self.decay[0].weight.float()), self.decay[1].weight.float(), self.decay[1].bias.float()
                )
            else:
                logits = F.linear(x.float(), self.decay.weight.float(), self.decay.bias.float())
            g = F.logsigmoid(logits).reshape(b, t, h, k) / 8
            beta = F.linear(x.float(), self.beta.weight.float(), self.beta.bias.float()).sigmoid()
        keys = self.keys.float()[None].expand(b, h, m, k).contiguous()
        values = torch.zeros(b, h, m, k, device=x.device, dtype=torch.float32)
        return q, key, v, g, beta, keys, values

    def _mix(self, x):
        q, k, v, g, beta, keys, values = self.operands(x)
        options = dict(temperature=self.config.kernel_temperature, ridge=self.config.ridge)
        if self.kind == "exact":
            o, _ = exact_komp_value_decay(q, k, v, g, beta, (keys, values), **options)
        else:
            o, _ = block_komp_value_decay(q, k, v, g, beta, (keys, values), chunk_size=self.config.chunk_size, **options)
        y = self.norm(o.float()).to(x.dtype).flatten(-2)
        gate = self.gate(x)
        gate = gate.sigmoid() if self.config.output_sigmoid else F.silu(gate)
        return self.out(y * gate)

    def forward(
        self, hidden_states, attention_mask=None, past_key_values=None, use_cache=False, output_attentions=False, **kwargs
    ):
        if use_cache or past_key_values is not None:
            raise ValueError("value-decay KOMP does not implement a generation cache")
        if attention_mask is not None and not bool(attention_mask.all()):
            raise ValueError("dense pretraining requires fully valid inputs; mask labels for tail loss")
        if kwargs:
            raise TypeError(f"Unsupported value-decay KOMP mixer arguments: {sorted(kwargs)}")
        out = (
            checkpoint(self._mix, hidden_states, use_reentrant=False)
            if self.training and self.checkpoint_mixer
            else self._mix(hidden_states)
        )
        return out, None, None
