"""Checkpoint-preserving adapter for frozen KOMP training models."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .recurrent import initialize_state, recurrent_komp


@dataclass
class LayerCache:
    recurrent_state: object
    conv_state: tuple


@dataclass
class ModelCache:
    layers: list
    position: int = 0

    def reset(self):
        self.layers[:] = [None] * len(self.layers)
        self.position = 0

    def index_select(self, indices):
        return ModelCache([None if layer is None else LayerCache(
            layer.recurrent_state.index_select(indices), tuple(x.index_select(0, indices) for x in layer.conv_state)
        ) for layer in self.layers], self.position)

    def nbytes(self):
        return sum(layer.recurrent_state.nbytes() + sum(x.numel()*x.element_size() for x in layer.conv_state)
                   for layer in self.layers if layer is not None)


class RecurrentKOMPLM:
    """Use the original modules and weights with an external, request-owned inference cache.

    The source model and its state_dict are not patched or renamed. This adapter
    is inference-only and accepts dense equal-length batches without padding.
    """
    def __init__(self, model, *, decay_axis='slot', backend='torch', prefill_operator=None):
        if model.training:
            raise ValueError('call model.eval() before constructing the inference adapter')
        if getattr(model.config, 'fuse_norm', False) or getattr(model.model, 'use_attnres', False):
            raise ValueError('this adapter supports the frozen unfused-norm KOMP blocks')
        if decay_axis not in ('slot', 'value'):
            raise ValueError('decay_axis must be explicit: slot or value')
        self.model, self.decay_axis, self.backend = model, decay_axis, backend
        self.prefill_operator = prefill_operator

    def new_cache(self):
        return ModelCache([None] * len(self.model.model.layers))

    def _mixer(self, mixer, x, cache):
        b, t, _ = x.shape
        h, m, d = mixer.keys.shape
        conv_state = (None, None, None) if cache is None else cache.conv_state
        projected, next_conv = [], []
        for projection, convolution, prior in zip((mixer.q, mixer.k, mixer.v), mixer.convs, conv_state):
            # short-conv decode mutates its cache, so each request owns these tensors
            result, final = convolution(projection(x), cache=prior, output_final_state=True)
            projected.append(result.reshape(b, t, h, d))
            next_conv.append(final)
        q, k, v = projected
        axis = d if self.decay_axis == 'value' else m
        with torch.autocast(x.device.type, enabled=False):
            z = mixer.decay(x.float()).reshape(b, t, h, axis)
            if hasattr(mixer, 'A_log'):
                g = -mixer.A_log.float().exp()*F.softplus(z+mixer.dt_bias.float())
            else:
                g = F.logsigmoid(z)/8
            beta = F.linear(x.float(), mixer.beta.weight.float(), mixer.beta.bias.float()).sigmoid()
            scores = None
            if hasattr(mixer, 'score'):
                scores = F.linear(x.float(), mixer.score.weight.float(), mixer.score.bias.float())
                beta = beta*(2*scores.sigmoid())
        if cache is None:
            kind = getattr(mixer, 'kind', self.model.config.kind)
            initial_scores = mixer.initial_score[None].expand(b, h, m) if scores is not None else None
            state = initialize_state(
                mixer.keys.float()[None].expand(b, h, m, d), x.new_zeros(b, h, m, d, dtype=torch.float32),
                chunk_size=1 if kind == 'exact' else 128, decay_axis=self.decay_axis,
                eviction='learned' if scores is not None else 'fifo', initial_scores=initial_scores,
            )
        else:
            state = cache.recurrent_state
        if cache is None and self.prefill_operator is not None:
            from .prefill import native_prefill
            output, state = native_prefill(q, k, v, g, beta, state, self.prefill_operator,
                                           scores=scores, backend=self.backend)
        else:
            output, state = recurrent_komp(q, k, v, g, beta, state, scores=scores, backend=self.backend)
        y = mixer.norm(output.float()).to(x.dtype).reshape(b, t, h*d)
        gate = mixer.gate(x)
        config = getattr(mixer, 'config', self.model.config)
        sigmoid = getattr(mixer, 'output_sigmoid', getattr(config, 'output_sigmoid', False))
        gate = gate.sigmoid() if sigmoid else F.silu(gate)
        return mixer.out(y*gate), LayerCache(state, tuple(next_conv))

    @torch.inference_mode()
    def __call__(self, input_ids, *, cache=None, attention_mask=None, logits_to_keep=0):
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError('input_ids must be nonempty [B,T]')
        if attention_mask is not None and (attention_mask.shape != input_ids.shape or not bool(attention_mask.all())):
            raise ValueError('padded/unequal-length batches are unsupported')
        cache = self.new_cache() if cache is None else cache
        if len(cache.layers) != len(self.model.model.layers):
            raise ValueError('cache layer count mismatch')
        for layer in cache.layers:
            if layer is not None and layer.recurrent_state.position != cache.position:
                raise ValueError('inconsistent cache token positions')
        hidden = self.model.model.embeddings(input_ids)
        for i, layer in enumerate(self.model.model.layers):
            residual = hidden
            update, cache.layers[i] = self._mixer(layer.attn, layer.attn_norm(hidden), cache.layers[i])
            hidden = residual+update
            hidden = hidden+layer.mlp(layer.mlp_norm(hidden))
        hidden = self.model.model.norm(hidden)
        selected = hidden[:, -logits_to_keep:] if logits_to_keep else hidden
        logits = self.model.lm_head(selected).float()
        cache.position += input_ids.shape[1]
        return logits, cache

    @torch.inference_mode()
    def generate(self, input_ids, *, max_new_tokens, eos_token_id=None):
        logits, cache = self(input_ids, logits_to_keep=1)
        generated, done = [], torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        for i in range(max_new_tokens):
            token = logits[:, -1].argmax(-1)
            if eos_token_id is not None:
                token = torch.where(done, eos_token_id, token)
                done |= token == eos_token_id
            generated.append(token)
            if bool(done.all()) or i+1 == max_new_tokens:
                break
            logits, cache = self(token[:, None], cache=cache, logits_to_keep=1)
        return torch.stack(generated, 1) if generated else input_ids[:, :0], cache
