# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Native FLA language models for Exact and Block value-decay KOMP."""

from fla.layers.komp_value_decay import KOMPValueDecayMixer
from fla.models.komp.modeling_komp import KOMPBlock, KOMPForCausalLM, KOMPModel
from fla.models.komp_value_decay.configuration_komp_value_decay import KOMPValueDecayConfig


class KOMPValueDecayBlock(KOMPBlock):
    mixer_class = KOMPValueDecayMixer


class KOMPValueDecayModel(KOMPModel):
    config_class = KOMPValueDecayConfig
    block_class = KOMPValueDecayBlock


class KOMPValueDecayForCausalLM(KOMPForCausalLM):
    config_class = KOMPValueDecayConfig
    model_class = KOMPValueDecayModel


class ValueDecayTrainingLM(KOMPValueDecayForCausalLM):
    """Aligned-label summed-loss adapter for the portable trainer."""

    def forward(self, input_ids, labels=None, valid_tokens=None):
        return self.forward_train(input_ids, labels, valid_tokens)
