# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Native value-decay model initialization, train/eval, serialization and checkpoint replay."""

import math

import pytest
import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.komp_value_decay import KOMPValueDecayConfig, KOMPValueDecayForCausalLM, KOMPValueDecayModel
from fla.utils import IS_NVIDIA, device


def tiny_config(kind="exact", **kwargs):
    return KOMPValueDecayConfig(
        kind=kind,
        hidden_size=128,
        num_heads=2,
        head_dim=64,
        num_hidden_layers=2,
        intermediate_size=192,
        vocab_size=256,
        **kwargs,
    )


def test_value_decay_config_and_default_layout(tmp_path):
    config = KOMPValueDecayConfig()
    config.save_pretrained(tmp_path)
    restored = AutoConfig.from_pretrained(tmp_path)
    assert isinstance(restored, KOMPValueDecayConfig)
    assert restored.kind == "exact" and restored.low_rank_gates and restored.multi_timescale_init
    assert AutoModel._model_mapping[KOMPValueDecayConfig] is KOMPValueDecayModel
    assert AutoModelForCausalLM._model_mapping[KOMPValueDecayConfig] is KOMPValueDecayForCausalLM
    with torch.device("meta"):
        exact = AutoModelForCausalLM.from_config(config)
        block = AutoModelForCausalLM.from_config(KOMPValueDecayConfig(kind="block"))
    assert sum(p.numel() for p in exact.parameters()) == 341552000
    assert {n: p.shape for n, p in exact.named_parameters()} == {n: p.shape for n, p in block.named_parameters()}
    assert exact.model.layers[0].attn.decay[1].out_features == 1024
    assert exact.all_tied_weights_keys == exact.model.all_tied_weights_keys == {}
    base = AutoModel.from_config(tiny_config())
    base.save_pretrained(tmp_path)
    restored_base = AutoModel.from_pretrained(tmp_path)
    for key, value in base.state_dict().items():
        torch.testing.assert_close(value, restored_base.state_dict()[key], atol=0, rtol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"use_cache": True}, id="cache"),
        pytest.param({"cache_geometry": True}, id="geometry-cache"),
        pytest.param({"num_slots": 65}, id="slots"),
        pytest.param({"expand_v": 2}, id="value-expansion"),
        pytest.param({"output_sigmoid": False, "output_gate_activation": "sigmoid"}, id="gate-contract"),
    ],
)
def test_value_decay_config_rejects_unsupported(kwargs):
    with pytest.raises(ValueError):
        tiny_config(**kwargs)


def test_value_decay_initialization():
    torch.manual_seed(42)
    model = KOMPValueDecayForCausalLM(tiny_config())
    mix = model.model.layers[0].attn
    bias = mix.decay[1].bias.detach().reshape(2, 64)
    g = torch.nn.functional.logsigmoid(bias) / 8
    torch.testing.assert_close(g.mean(), torch.tensor(math.log(0.995)))
    assert not torch.equal(g[0], g[1])
    assert mix.decay[1].weight.count_nonzero() == 0
    assert mix.decay[0].weight.count_nonzero() > 0
    assert mix.gate[1].bias is None


@pytest.mark.skipif(not IS_NVIDIA, reason="model mixer requires NVIDIA CUDA")
@pytest.mark.parametrize(
    ("kind", "options"),
    [
        pytest.param("exact", {}, id="exact-defaults"),
        pytest.param("block", {}, id="block-defaults"),
        pytest.param(
            "block", dict(output_sigmoid=False, low_rank_gates=False, multi_timescale_init=False), id="block-dense-gates"
        ),
    ],
)
def test_value_decay_model_train_restore(kind, options, tmp_path):
    torch.manual_seed(42)
    config = tiny_config(kind, **options)
    model = AutoModelForCausalLM.from_config(config).to(device)
    ids = torch.randint(0, config.vocab_size, (1, 17), device=device)
    labels = ids.clone()
    labels[:, -2:] = -100
    model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(input_ids=ids, labels=labels).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    # Mixer checkpointing must reproduce the non-checkpointed model's update.
    eager = KOMPValueDecayForCausalLM(tiny_config(kind, checkpoint_mixer=False, **options)).to(device)
    eager.load_state_dict(model.state_dict())
    eager.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        eager_loss = eager(input_ids=ids, labels=labels).loss
    eager_loss.backward()
    torch.testing.assert_close(loss, eager_loss, atol=2e-5, rtol=2e-5)
    for p, q in zip(model.parameters(), eager.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=2e-4, rtol=0.003)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    opt.step()
    model.eval()
    with torch.no_grad():
        expected = model(ids).logits
    model.save_pretrained(tmp_path)
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path).to(device).eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded(ids).logits, expected, atol=1e-5, rtol=1e-5)
    assert isinstance(loaded, KOMPValueDecayForCausalLM)
    with pytest.raises(ValueError, match="cache"):
        loaded(ids, use_cache=True)
    with pytest.raises(ValueError, match="valid"):
        loaded(ids, attention_mask=torch.zeros_like(ids))
