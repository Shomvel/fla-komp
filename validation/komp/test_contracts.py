# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Host contracts for the installed KOMP package (no GPU allocation)."""

import tempfile

import numpy as np
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from examples.komp.data import TokenBatches
from fla.models.komp import KOMPConfig, KOMPForCausalLM
from fla.models.komp.conversion import map_weights, mappings
from fla.models.komp.legacy import LM
from fla.models.komp.modeling_komp import RoundingCompatibleMLP
from fla.ops.komp import block_komp, block_komp_fast, block_komp_flex, exact_komp, projected_slot_attention
from fla.ops.komp.block_reference import block_komp_reference


def test_public_api_and_config_roundtrip():
    assert all(callable(x) for x in (exact_komp, block_komp, block_komp_fast, block_komp_flex, projected_slot_attention))
    config = KOMPConfig(cache_geometry=True)
    assert isinstance(AutoConfig.for_model("komp"), KOMPConfig)
    assert AutoModelForCausalLM._model_mapping[KOMPConfig] is KOMPForCausalLM
    with tempfile.TemporaryDirectory() as folder:
        config.save_pretrained(folder)
        restored = AutoConfig.from_pretrained(folder)
    original_fields = config.to_dict()
    restored_fields = restored.to_dict()
    original_fields.pop("_name_or_path", None)
    restored_fields.pop("_name_or_path", None)
    assert restored_fields == original_fields
    assert restored.cache_geometry and restored.checkpoint_mixer and restored.kind == "exact"
    legacy = KOMPConfig.from_dict({"model_type": "komp", "temperature": 0.5})
    assert legacy.kernel_temperature == 0.5
    assert "temperature" not in legacy.to_dict()
    with tempfile.TemporaryDirectory() as folder:
        legacy.save_pretrained(folder)
        assert AutoConfig.from_pretrained(folder).kernel_temperature == 0.5
    for kwargs in (
        {"kind": "bogus"},
        {"kind": "block", "cache_geometry": True},
        {"use_cache": True},
        {"hidden_size": 128},
        {"cache_geometry": True, "checkpoint_mixer": False},
        {"conv_size": 8},
    ):
        with pytest.raises(ValueError):
            KOMPConfig(**kwargs)


@pytest.mark.parametrize("kind", ["exact", "block"])
def test_complete_checkpoint_layout(kind):
    # Meta construction checks every parameter name/shape without allocating 2.7 GB.
    with torch.device("meta"):
        old = LM(kind)
        new = KOMPForCausalLM(KOMPConfig(kind=kind))
    old_state = old.state_dict()
    mapped = map_weights(old_state)
    expected = new.state_dict()
    assert set(mapped) == set(expected)
    assert all(mapped[name].shape == expected[name].shape for name in mapped)
    new.load_state_dict(mapped, strict=True)
    assert sum(p.numel() for p in new.parameters()) == sum(p.numel() for p in old.parameters()) == 339966848
    assert not new.model.gradient_checkpointing
    assert all(layer.attn.checkpoint_mixer for layer in new.model.layers)


def test_mapped_mlp_values_and_gradients():
    torch.manual_seed(2)
    width, intermediate = 12, 1824
    old_up = torch.nn.Linear(width, intermediate * 2, bias=False)
    old_down = torch.nn.Linear(intermediate, width, bias=False)
    native = RoundingCompatibleMLP(width, intermediate_size=intermediate, fuse_swiglu=False)
    mapping = mappings("layers.0.up.weight")
    with torch.no_grad():
        for name, part in mapping:
            getattr(native, name.split(".")[-2]).weight.copy_(old_up.weight[part])
        native.down_proj.weight.copy_(old_down.weight)
    inputs = torch.randn(2, 3, width, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    gate, up = old_up(reference_inputs).chunk(2, -1)
    reference = old_down(torch.nn.functional.silu(gate) * up)
    actual = native(inputs)
    torch.testing.assert_close(actual, reference)
    actual.square().sum().backward()
    reference.square().sum().backward()
    torch.testing.assert_close(inputs.grad, reference_inputs.grad)
    torch.testing.assert_close(native.gate_proj.weight.grad, old_up.weight.grad[:intermediate])
    torch.testing.assert_close(native.up_proj.weight.grad, old_up.weight.grad[intermediate:])


def test_distributed_token_batches_and_resume(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(65, dtype="<u2").tofile(path)
    complete = TokenBatches(path, sequence_length=4, global_batch=4)
    shards = [TokenBatches(path, 4, 4, rank, 2) for rank in range(2)]
    for step in [0, 1, 3, 4, 7]:
        x, y = complete.batch(step)
        pieces = [shard.batch(step) for shard in shards]
        assert torch.equal(torch.cat([p[0] for p in pieces]), x)
        assert torch.equal(torch.cat([p[1] for p in pieces]), y)
        assert torch.equal(y, x + 1)
        restored = TokenBatches(path, 4, 4)
        assert torch.equal(restored.batch(step)[0], x)
    assert complete.fingerprint() == shards[0].fingerprint()
    with pytest.raises(ValueError):
        TokenBatches(path, 4, 3, world_size=2)


def test_reference_gradients():
    torch.manual_seed(4)
    shape = (1, 3, 1, 2)
    q, k, v = [torch.randn(shape, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    g = torch.full((1, 3, 1), -0.05, dtype=torch.float64, requires_grad=True)
    beta = torch.full((1, 3, 1), 0.2, dtype=torch.float64, requires_grad=True)
    keys, state = [torch.randn(1, 1, 2, 2, dtype=torch.float64, requires_grad=True) for _ in range(2)]

    def apply(q, k, v, g, beta, keys, state):
        output, (_, final) = block_komp_reference(q, k, v, g, beta, (keys, state), chunk_size=1)
        return output, final

    assert torch.autograd.gradcheck(apply, (q, k, v, g, beta, keys, state))


def test_optimizer_moments_follow_weight_mapping():
    from fla.models.komp.conversion import map_optimizer, optimizer

    old = torch.nn.Module()
    old.layers = torch.nn.ModuleList([torch.nn.Module()])
    old.layers[0].up = torch.nn.Linear(3, 3648, bias=False)
    old.layers[0].mix = torch.nn.Module()
    old.layers[0].mix.keys = torch.nn.Parameter(torch.randn(2, 2))
    new = torch.nn.Module()
    new.model = torch.nn.Module()
    new.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    new.model.layers[0].mlp = torch.nn.Module()
    new.model.layers[0].mlp.gate_proj = torch.nn.Linear(3, 1824, bias=False)
    new.model.layers[0].mlp.up_proj = torch.nn.Linear(3, 1824, bias=False)
    new.model.layers[0].attn = torch.nn.Module()
    new.model.layers[0].attn.keys = torch.nn.Parameter(torch.randn(2, 2))
    old_opt, new_opt = optimizer(old), optimizer(new)
    for parameter in old.parameters():
        old_opt.state[parameter] = dict(
            step=torch.tensor(5.0),
            exp_avg=torch.arange(parameter.numel()).reshape(parameter.shape).float(),
            exp_avg_sq=torch.full_like(parameter, 0.25),
        )
    new_opt.load_state_dict(map_optimizer(old, old_opt.state_dict(), new, new_opt))
    targets = dict(new.named_parameters())
    for name, parameter in old.named_parameters():
        for target, part in mappings(name):
            for key in ("exp_avg", "exp_avg_sq"):
                expected = old_opt.state[parameter][key]
                torch.testing.assert_close(new_opt.state[targets[target]][key], expected if part is None else expected[part])
            assert new_opt.state[targets[target]]["step"] == 5
    assert sorted(g["weight_decay"] for g in new_opt.param_groups) == [0.0, 0.1]
