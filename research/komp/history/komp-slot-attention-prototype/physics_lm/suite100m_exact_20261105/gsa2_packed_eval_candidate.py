"""Canonical cache-free dense evaluation uses training's packed chunk layout.

A process-local adapter; original class and weights remain unchanged on disk.
"""
import torch
import fla.layers.gsa2 as native
from gsa2_fixed_norm_candidate import install as fix_norm


def install():
    configuration = fix_norm()
    original = native.GatedSlotAttention2.forward

    def forward(self, hidden_states, attention_mask=None, past_key_values=None,
                use_cache=False, output_attentions=False, **kwargs):
        batch, length, width = hidden_states.shape
        canonical = (not self.training and attention_mask is None
                     and past_key_values is None and not use_cache
                     and kwargs.get('cu_seqlens') is None and batch > 1
                     and length > 64)
        if not canonical:
            return original(self, hidden_states, attention_mask, past_key_values,
                            use_cache, output_attentions, **kwargs)
        boundaries = torch.arange(0, (batch + 1) * length, length,
                                  device=hidden_states.device, dtype=torch.int32)
        output, attention, cache = original(
            self, hidden_states.reshape(1, batch * length, width),
            None, None, False, output_attentions,
            **{**kwargs, 'cu_seqlens': boundaries})
        return output.reshape(batch, length, -1), attention, cache

    native.GatedSlotAttention2.forward = forward
    return configuration + [{'cache_free_eval_layout': 'packed for B>1 and T>64; short recurrent path unchanged'}]
