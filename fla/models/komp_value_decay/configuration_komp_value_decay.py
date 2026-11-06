# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Configuration for independently initialized value-decay KOMP models."""

import math

from fla.models.gated_deltanet.configuration_gated_deltanet import GatedDeltaNetConfig


class KOMPValueDecayConfig(GatedDeltaNetConfig):
    model_type = "komp_value_decay"

    def __init__(
        self,
        kind="exact",
        num_slots=32,
        chunk_size=128,
        checkpoint_mixer=True,
        alpha0=0.995,
        beta0=0.2,
        kernel_temperature=0.5,
        ridge=1e-4,
        output_sigmoid=True,
        low_rank_gates=True,
        gate_rank=64,
        multi_timescale_init=True,
        timescale_spread=4.0,
        cache_geometry=False,
        **kwargs,
    ):
        defaults = dict(
            hidden_size=1024,
            num_hidden_layers=24,
            num_heads=16,
            head_dim=64,
            expand_v=1.0,
            intermediate_size=2272,
            vocab_size=32000,
            norm_eps=1e-6,
            fuse_norm=False,
            fuse_swiglu=False,
            fuse_cross_entropy=False,
            fuse_linear_cross_entropy=True,
            use_cache=False,
            tie_word_embeddings=False,
            output_gate_activation="sigmoid" if output_sigmoid else "silu",
        )
        defaults.update(kwargs)
        super().__init__(**defaults)
        if kind not in ("exact", "block"):
            raise ValueError("kind must be exact or block")
        for name, value in [
            ("num_slots", num_slots),
            ("gate_rank", gate_rank),
            ("hidden_size", self.hidden_size),
            ("num_heads", self.num_heads),
            ("head_dim", self.head_dim),
            ("num_hidden_layers", self.num_hidden_layers),
            ("intermediate_size", self.intermediate_size),
            ("vocab_size", self.vocab_size),
        ]:
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_size != self.num_heads * self.head_dim or self.expand_v != 1.0:
            raise ValueError("value-decay KOMP requires hidden_size=num_heads*head_dim and expand_v=1")
        if not num_slots <= min(64, self.head_dim) or self.head_dim > 128:
            raise ValueError("requires num_slots<=min(64, head_dim) and head_dim<=128")
        if chunk_size not in (16, 32, 64, 128):
            raise ValueError("chunk_size must be 16, 32, 64 or 128 (Exact always compresses per token)")
        if not 0 < alpha0 < 1 or not 0 < beta0 < 1:
            raise ValueError("alpha0 and beta0 must be strictly between zero and one")
        if any(not math.isfinite(x) or x <= 0 for x in (kernel_temperature, ridge)):
            raise ValueError("kernel_temperature and ridge must be positive and finite")
        if not math.isfinite(timescale_spread) or timescale_spread < 1:
            raise ValueError("timescale_spread must be finite and >=1")
        if cache_geometry:
            raise ValueError("value-decay geometry caching has not been validated; use cache_geometry=False")
        if self.use_cache or self.attn is not None or self.attnres_block_size is not None:
            raise ValueError("value-decay KOMP supports dense pretraining without generation cache or attention hybrids")
        if self.fuse_norm or self.fuse_swiglu or not self.use_gate or not self.use_short_conv:
            raise ValueError("requires unfused norms/MLP, output gate and short convolution")
        if self.tie_word_embeddings or self.num_v_heads not in (None, self.num_heads) or self.allow_neg_eigval:
            raise ValueError("requires untied embeddings, equal key/value head counts and beta in [0,1]")
        if self.hidden_act not in ("swish", "silu"):
            raise ValueError("value-decay KOMP requires a SiLU MLP")
        if self.output_gate_activation != ("sigmoid" if output_sigmoid else "silu"):
            raise ValueError("output_gate_activation must agree with output_sigmoid")
        self.kind, self.num_slots, self.chunk_size = kind, num_slots, chunk_size
        self.checkpoint_mixer, self.cache_geometry = checkpoint_mixer, cache_geometry
        self.alpha0, self.beta0 = alpha0, beta0
        self.kernel_temperature, self.ridge = kernel_temperature, ridge
        self.output_sigmoid, self.low_rank_gates, self.gate_rank = output_sigmoid, low_rank_gates, gate_rank
        self.multi_timescale_init, self.timescale_spread = multi_timescale_init, timescale_spread
