"""r3 model adapter: bounded native Block slot decoding, original r1 Exact path."""
import torch
from torch.nn import functional as F

from .model import LayerCache, RecurrentKOMPLM
from .native_chunk import initialize_native_state, native_chunk_append, native_chunk_prefill


class NativeChunkKOMPLM(RecurrentKOMPLM):
    def __init__(self, model, *, decay_axis='slot', backend='triton', prefill_operator=None):
        if decay_axis != 'slot':
            raise NotImplementedError('r3 native chunk cache does not support Value decay: prefix-wide scaling/fallback is not validated')
        if prefill_operator is None:
            raise ValueError('frozen native operator required')
        super().__init__(model, decay_axis=decay_axis, backend=backend, prefill_operator=prefill_operator)

    def _mixer(self, mixer, x, cache):
        kind = getattr(mixer, 'kind', self.model.config.kind)
        if kind == 'exact':
            return super()._mixer(mixer, x, cache)
        b, t, _ = x.shape
        h, m, d = mixer.keys.shape
        prior = (None, None, None) if cache is None else cache.conv_state
        projected, conv = [], []
        for projection, convolution, old in zip((mixer.q, mixer.k, mixer.v), mixer.convs, prior):
            result, final = convolution(projection(x), cache=old, output_final_state=True)
            projected.append(result.reshape(b, t, h, d))
            conv.append(final)
        with torch.autocast(x.device.type, enabled=False):
            z = mixer.decay(x.float()).reshape(b, t, h, m)
            g = -mixer.A_log.float().exp()*F.softplus(z+mixer.dt_bias.float()) if hasattr(mixer, 'A_log') else F.logsigmoid(z)/8
            beta = F.linear(x.float(), mixer.beta.weight.float(), mixer.beta.bias.float()).sigmoid()
            scores = None
            if hasattr(mixer, 'score'):
                scores = F.linear(x.float(), mixer.score.weight.float(), mixer.score.bias.float())
                beta = beta*(2*scores.sigmoid())
        if cache is None:
            state = initialize_native_state(mixer.keys.float()[None].expand(b, h, m, d),
                    x.new_zeros(b, h, m, d, dtype=torch.float32),
                    initial_scores=None if scores is None else mixer.initial_score[None].expand(b, h, m))
            output, state = native_chunk_prefill(*projected, g, beta, state, self.prefill_operator, scores=scores)
        else:
            output, state = native_chunk_append(*projected, g, beta, cache.recurrent_state,
                                               self.prefill_operator, scores=scores)
        y = mixer.norm(output.float()).to(x.dtype).reshape(b, t, h*d)
        gate = mixer.gate(x)
        config = getattr(mixer, 'config', self.model.config)
        sigmoid = getattr(mixer, 'output_sigmoid', getattr(config, 'output_sigmoid', False))
        gate = gate.sigmoid() if sigmoid else F.silu(gate)
        return mixer.out(y*gate), LayerCache(state, tuple(conv))
