# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Legacy KOMP1B checkpoint layout, retained for checkpoint conversion."""

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from fla.layers.komp import KOMPMixerBase as Mixer
from fla.models.komp.configuration_komp import ALPHA, BETA, FF, VOCAB, D, H, K, L, M


class Block(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.n1 = nn.RMSNorm(D, eps=1e-6)
        self.n2 = nn.RMSNorm(D, eps=1e-6)
        self.mix = Mixer(kind)
        self.up = nn.Linear(D, 2 * FF, bias=False)
        self.down = nn.Linear(FF, D, bias=False)

    def forward(self, x):
        x = x + self.mix(self.n1(x))
        a, b = self.up(self.n2(x)).chunk(2, -1)
        return x + self.down(F.silu(a) * b)


class LM(nn.Module):
    def __init__(self, kind, seed=42, checkpoint_layers=True):
        super().__init__()
        assert kind in ("exact", "block")
        self.kind = kind
        self.checkpoint_layers = checkpoint_layers
        torch.manual_seed(seed)
        self.embed = nn.Embedding(VOCAB, D)
        self.layers = nn.ModuleList()
        for i in range(L):
            torch.manual_seed(seed + 100 + i)
            self.layers.append(Block(kind))
        self.norm = nn.RMSNorm(D, eps=1e-6)
        self.head = nn.Linear(D, VOCAB, bias=False)
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)) and not name.endswith((".decay", ".beta")):
                torch.manual_seed(seed + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))
                # FLA-style depth scaling for residual output projections.
                std = 0.02 / math.sqrt(2 * L) if name.endswith((".mix.out", ".down")) else 0.02
                nn.init.normal_(module.weight, 0, std)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
        for i, layer in enumerate(self.layers):
            for j, conv in enumerate(layer.mix.convs):
                torch.manual_seed(seed + 6000 + i * 3 + j)
                nn.init.normal_(conv.weight, 0, 0.02)

    def forward(self, ids, labels=None, valid_tokens=None):
        x = self.embed(ids)
        for layer in self.layers:
            x = checkpoint(layer, x, use_reentrant=False) if self.training and self.checkpoint_layers else layer(x)
        x = self.norm(x)
        if labels is None:
            return x
        # Bound vocabulary-logit memory, keeping the head in the DDP graph.
        loss = x.new_zeros((), dtype=torch.float32)
        flat = x.flatten(0, 1)
        target = labels.flatten()
        if valid_tokens is not None:
            target = target.clone()
            target[valid_tokens:] = -100
        for start in range(0, len(flat), 256):
            loss = loss + checkpoint(
                lambda h, y: F.cross_entropy(self.head(h).float(), y, reduction="sum", ignore_index=-100),
                flat[start: start + 256],
                target[start: start + 256],
                use_reentrant=False,
            )
        return loss


def optimizer(model):
    groups = {True: [], False: []}
    for name, p in model.named_parameters():
        groups[p.ndim < 2 or name.endswith(".keys")].append(p)
    return torch.optim.AdamW(
        [dict(params=ps, weight_decay=0.0 if flag else 0.1) for flag, ps in groups.items()],
        lr=3e-4,
        betas=(0.8, 0.95),
        eps=1e-15,
        fused=True,
    )


def metadata(model):
    return dict(
        kind=model.kind,
        parameters=sum(p.numel() for p in model.parameters()),
        width=D,
        layers=L,
        ff=FF,
        heads=H,
        head_dim=K,
        slots=M,
        alpha0=ALPHA,
        beta0=BETA,
        seed=42,
        vocab=VOCAB,
        short_conv=4,
        compression_tokens=1 if model.kind == "exact" else 128,
        exact_compute_chunk=16 if model.kind == "exact" else None,
        eviction="fifo",
        temperature=0.5,
        ridge=1e-4,
        projection="float32",
        state_compute="float32",
        chunk_tape="float32" if model.kind == "exact" else "bfloat16",
        residual_init_std=0.02 / math.sqrt(2 * L),
    )
