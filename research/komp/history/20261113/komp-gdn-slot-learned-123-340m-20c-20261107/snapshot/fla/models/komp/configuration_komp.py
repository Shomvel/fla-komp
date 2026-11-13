# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from fla.models.gated_deltanet.configuration_gated_deltanet import GatedDeltaNetConfig

D, H, K, M, L, FF, VOCAB = 1024, 16, 64, 32, 24, 2272, 32000
ALPHA, BETA = 0.995, 0.2


class KOMPConfig(GatedDeltaNetConfig):
    model_type = "komp"

    def __init__(self, kind="exact", checkpoint_mixer=True, cache_geometry=False, kernel_temperature=0.5, **kwargs):
        # Legacy configs used the reserved Transformers generation field for kernel geometry.
        legacy_temperature = kwargs.pop("temperature", kernel_temperature)
        if kernel_temperature != 0.5 or legacy_temperature != 0.5:
            raise ValueError("Frozen KOMP340M requires kernel_temperature=0.5")
        defaults = dict(
            hidden_size=D,
            num_hidden_layers=L,
            num_heads=H,
            head_dim=K,
            expand_v=1.0,
            intermediate_size=FF,
            vocab_size=VOCAB,
            norm_eps=1e-6,
            fuse_norm=False,
            fuse_swiglu=False,
            fuse_cross_entropy=False,
            fuse_linear_cross_entropy=True,
            use_cache=False,
            tie_word_embeddings=False,
        )
        defaults.update(kwargs)
        super().__init__(**defaults)
        self.kind = kind
        self.checkpoint_mixer = checkpoint_mixer
        self.cache_geometry = cache_geometry
        for name, want in [
            ("hidden_size", D),
            ("num_hidden_layers", L),
            ("num_heads", H),
            ("head_dim", K),
            ("intermediate_size", FF),
            ("vocab_size", VOCAB),
            ("fuse_norm", False),
            ("fuse_swiglu", False),
            ("expand_v", 1.0),
            ("conv_size", 4),
            ("norm_eps", 1e-6),
            ("use_gate", True),
            ("use_short_conv", True),
            ("tie_word_embeddings", False),
        ]:
            if getattr(self, name) != want:
                raise ValueError(f"Frozen KOMP340M config requires {name}={want}")
        if kind not in ("exact", "block"):
            raise ValueError("kind must be exact or block")

        if self.hidden_act not in ("swish", "silu") or self.output_gate_activation not in ("swish", "silu"):
            raise ValueError("KOMP340M requires SiLU activations")
        if cache_geometry and not checkpoint_mixer:
            raise ValueError("Geometry caching requires mixer checkpointing")
        if cache_geometry and kind != "exact":
            raise ValueError("geometry cache is Exact-only")
        if self.attn is not None or self.attnres_block_size is not None or self.use_cache:
            raise ValueError("KOMP340M supports dense pretraining without attention hybrids or generation cache")
        self.num_slots = M
        self.alpha0 = ALPHA
        self.beta0 = BETA
        self.kernel_temperature = kernel_temperature
        self.ridge = 1e-4
