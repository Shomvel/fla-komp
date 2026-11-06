"""Process-local experiment: keep gated norm reductions independent of row count.

Does not modify frozen FLA source or any checkpoint. Formula/dtypes are unchanged.
"""
import triton
import fla.modules.norm.fused_norm_gate as norm


def install():
    changed = []
    for name in ('layer_norm_gated_fwd_kernel', 'layer_norm_gated_bwd_kernel'):
        kernel = getattr(norm, name)
        tuner = kernel.fn
        assert hasattr(tuner, 'configs') and hasattr(tuner, 'cache'), type(tuner)
        tuner.configs = [triton.Config({'BT': 32}, num_warps=4)]
        tuner.cache.clear()
        changed.append({'kernel': name, 'BT': 32, 'num_warps': 4})
    return changed
