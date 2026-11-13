# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Lossless tensor/name mapping from the frozen KOMP1B trainer."""

import copy

import torch

from fla.models.komp import legacy
from fla.models.komp.configuration_komp import FF


def mappings(old_name):
    if old_name == "embed.weight":
        return [("model.embeddings.weight", None)]
    if old_name == "head.weight":
        return [("lm_head.weight", None)]
    if old_name == "norm.weight":
        return [("model.norm.weight", None)]
    fields = old_name.split(".")
    assert fields[0] == "layers", old_name
    stem = "model.layers." + fields[1] + "."
    tail = ".".join(fields[2:])
    if tail == "up.weight":
        return [(stem + "mlp.gate_proj.weight", slice(0, FF)), (stem + "mlp.up_proj.weight", slice(FF, None))]
    for old, new in [("n1.", "attn_norm."), ("n2.", "mlp_norm."), ("mix.", "attn."), ("down.", "mlp.down_proj.")]:
        if tail.startswith(old):
            return [(stem + new + tail[len(old):], None)]
    raise KeyError(old_name)


def map_weights(state):
    mapped = {}
    for name, value in state.items():
        for target, part in mappings(name):
            mapped[target] = value if part is None else value[part].clone()
    return mapped


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


def map_optimizer(old_model, old_state, new_model, new_opt):
    old_groups = legacy.optimizer(old_model).param_groups
    old_names = {id(p): n for n, p in old_model.named_parameters()}
    names = {}
    for parameters, saved in zip(old_groups, old_state["param_groups"]):
        for p, index in zip(parameters["params"], saved["params"]):
            names[index] = old_names[id(p)]
    transformed = {}
    source_group = {}
    for group in old_state["param_groups"]:
        for index in group["params"]:
            for target, part in mappings(names[index]):
                source_group[target] = group
                transformed[target] = {
                    k: (
                        v[part].clone()
                        if torch.is_tensor(v) and v.ndim > 0 and part is not None
                        else v.clone()
                        if torch.is_tensor(v)
                        else copy.deepcopy(v)
                    )
                    for k, v in old_state["state"].get(index, {}).items()
                }
    new_names = {id(p): n for n, p in new_model.named_parameters()}
    out = new_opt.state_dict()
    for params, group in zip(new_opt.param_groups, out["param_groups"]):
        reference = None
        for p, index in zip(params["params"], group["params"]):
            name = new_names[id(p)]
            out["state"][index] = transformed[name]
            source = source_group[name]
            if reference is None:
                reference = source
            else:
                for key in ["lr", "betas", "eps", "weight_decay"]:
                    assert source[key] == reference[key]
        for key, value in reference.items():
            if key != "params":
                group[key] = copy.deepcopy(value)
    return out
