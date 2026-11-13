# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.komp_value_decay.configuration_komp_value_decay import KOMPValueDecayConfig
from fla.models.komp_value_decay.modeling_komp_value_decay import KOMPValueDecayForCausalLM, KOMPValueDecayModel

AutoConfig.register("komp_value_decay", KOMPValueDecayConfig, exist_ok=True)
AutoModel.register(KOMPValueDecayConfig, KOMPValueDecayModel, exist_ok=True)
AutoModelForCausalLM.register(KOMPValueDecayConfig, KOMPValueDecayForCausalLM, exist_ok=True)

__all__ = ["KOMPValueDecayConfig", "KOMPValueDecayForCausalLM", "KOMPValueDecayModel"]
