# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Native FLA KOMP340M pretraining model with compatible checkpoint names.

Constructors retain the frozen adapter's initialization. Load a converted
legacy checkpoint to reproduce an existing experiment's initial conditions.
"""

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from fla.layers.komp import KOMPMixer
from fla.models.gated_deltanet.modeling_gated_deltanet import GatedDeltaNetBlock, GatedDeltaNetPreTrainedModel
from fla.models.komp.configuration_komp import KOMPConfig
from fla.models.modeling_utils import FLAForCausalLM, FLAModel
from fla.modules import GatedMLP


class RoundingCompatibleMLP(GatedMLP):
    """Native FLA projections, preserving the previous BF16 SiLU-before-multiply rounding."""

    def forward(self, x, **kwargs):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class KOMPBlock(GatedDeltaNetBlock):
    def __init__(self, config, layer_idx):
        # Native block creates the native FLA MLP and norm/residual contract.
        super().__init__(config, layer_idx)
        self.attn = KOMPMixer(config, layer_idx)
        self.mlp = RoundingCompatibleMLP(config.hidden_size, intermediate_size=config.intermediate_size, fuse_swiglu=False)


class KOMPModel(GatedDeltaNetPreTrainedModel, FLAModel):
    config_class = KOMPConfig

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList(KOMPBlock(config, i) for i in range(config.num_hidden_layers))
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.use_attnres = False
        self.gradient_checkpointing = False
        # All weights come from a validated legacy checkpoint; no post_init reinitialization.


class KOMPForCausalLM(GatedDeltaNetPreTrainedModel, FLAForCausalLM):
    config_class = KOMPConfig
    _tied_weights_keys = {}

    def __init__(self, config):
        super().__init__(config)
        self.model = KOMPModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size
        self.criterion = None
        self.kind = config.kind

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        shift_labels=None,
        num_items_in_batch=None,
        use_cache=False,
        past_key_values=None,
        return_dict=True,
        logits_to_keep=0,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"Unsupported KOMP model arguments: {sorted(kwargs)}")
        if use_cache or past_key_values is not None:
            raise ValueError("Generation cache is not implemented")
        output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, use_cache=False, return_dict=True
        )
        loss, logits = self._compute_logits_and_loss(
            output.last_hidden_state,
            labels=labels,
            shift_labels=shift_labels,
            num_items_in_batch=num_items_in_batch,
            logits_to_keep=logits_to_keep,
        )
        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def forward_train(self, input_ids, aligned_labels=None, valid_tokens=None):
        hidden = self.model(input_ids=input_ids, use_cache=False, return_dict=True).last_hidden_state
        if aligned_labels is None:
            return hidden
        aligned = aligned_labels
        if valid_tokens is not None:
            aligned = aligned_labels.reshape(-1).clone()
            aligned[valid_tokens:] = -100
            aligned = aligned.reshape(aligned_labels.shape)
        if torch.is_autocast_enabled("cuda"):
            hidden = hidden.to(torch.get_autocast_dtype("cuda"))
        loss, _ = self._compute_logits_and_loss(hidden, shift_labels=aligned, num_items_in_batch=None)
        return loss * aligned.ne(-100).sum()


class TrainingLM(KOMPForCausalLM):
    """Thin adapter for the existing trainer's aligned-label sum-loss contract."""

    def forward(self, input_ids, labels=None, valid_tokens=None):
        return self.forward_train(input_ids, labels, valid_tokens)
