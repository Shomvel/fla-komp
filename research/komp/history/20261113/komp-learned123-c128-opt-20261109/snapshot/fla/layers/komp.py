# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""KOMP1B mixer with the frozen training parameter layout."""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from fla.modules import ShortConvolution
from fla.ops.komp import block_komp_fast, exact_komp
from fla.ops.komp.geometry_cache import geometry_contexts

D, H, K, M = 2048, 32, 64, 32
ALPHA, BETA = 0.995, 0.2


class KOMPMixerBase(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        for name in ("q", "k", "v", "out", "gate"):
            setattr(self, name, nn.Linear(D, D, bias=False))
        self.convs = nn.ModuleList([ShortConvolution(D, 4, bias=False, activation="silu", backend="triton") for _ in range(3)])
        self.norm = nn.RMSNorm(K, eps=1e-6)
        self.decay = nn.Linear(D, H * M)
        self.beta = nn.Linear(D, H)
        self.keys = nn.Parameter(torch.stack([torch.linalg.qr(torch.randn(K, M)).Q.T for _ in range(H)]))
        nn.init.zeros_(self.decay.weight)
        nn.init.constant_(self.decay.bias, math.log(ALPHA**8 / (1 - ALPHA**8)))
        nn.init.zeros_(self.beta.weight)
        nn.init.constant_(self.beta.bias, math.log(BETA / (1 - BETA)))

    def operands(self, x):
        b, t, _ = x.shape
        q, k, v = [conv(proj(x))[0].reshape(b, t, H, K) for conv, proj in zip(self.convs, (self.q, self.k, self.v))]
        with torch.autocast("cuda", enabled=False):
            g = F.logsigmoid(F.linear(x.float(), self.decay.weight, self.decay.bias)).reshape(b, t, H, M) / 8
            beta = F.linear(x.float(), self.beta.weight, self.beta.bias).sigmoid()
        keys = self.keys[None].expand(b, H, M, K).contiguous()
        values = torch.zeros(b, H, M, K, device=x.device)
        return q, k, v, g, beta, keys, values

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v, g, beta, keys, values = self.operands(x)
        if self.kind == "exact":
            o, _, _ = exact_komp(
                q, k, v, g, beta, (keys, values), eviction="fifo", compute_chunk=16, projection_dtype="float32"
            )
        else:
            o, _ = block_komp_fast(q, k, v, g, beta, (keys, values), chunk_size=128)
        y = self.norm(o.float()).to(x.dtype).reshape(b, t, D)
        return self.out(y * F.silu(self.gate(x)))


class KOMPMixer(KOMPMixerBase):
    def __init__(self, config, layer_idx):
        super().__init__(config.kind)
        self.layer_idx = layer_idx
        self.checkpoint_mixer = config.checkpoint_mixer
        self.cache_geometry = config.cache_geometry

    def forward(
        self, hidden_states, attention_mask=None, past_key_values=None, use_cache=False, output_attentions=False, **kwargs
    ):
        if use_cache or past_key_values is not None:
            raise ValueError("KOMP pretraining adapter does not implement generation cache")
        if attention_mask is not None and not bool(attention_mask.all()):
            raise ValueError("Packed pretraining requires fully valid inputs; mask labels for tail loss")
        options = {}
        if self.training and self.checkpoint_mixer and self.cache_geometry:
            if self.kind != "exact":
                raise ValueError("geometry cache is Exact-only")
            if torch.compiler.is_compiling():
                raise RuntimeError("geometry cache requires eager checkpoint execution")
            options["context_fn"] = geometry_contexts
        y = (
            checkpoint(super().forward, hidden_states, use_reentrant=False, **options)
            if self.training and self.checkpoint_mixer
            else super().forward(hidden_states)
        )
        return y, None, None
