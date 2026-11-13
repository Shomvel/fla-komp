# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.komp.configuration_komp import KOMPConfig
from fla.models.komp.modeling_komp import KOMPForCausalLM, KOMPModel

AutoConfig.register("komp", KOMPConfig, exist_ok=True)
AutoModel.register(KOMPConfig, KOMPModel, exist_ok=True)
AutoModelForCausalLM.register(KOMPConfig, KOMPForCausalLM, exist_ok=True)

__all__ = ["KOMPConfig", "KOMPForCausalLM", "KOMPModel"]
