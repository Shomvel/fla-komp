"""Isolated GSA2 arithmetic candidate; no frozen source is edited."""
import importlib
import triton
from gsa2_packed_eval_candidate import install as install_packed


def install():
    configuration=install_packed()
    module=importlib.import_module('fla.modules.norm.l2norm')
    for name in ('l2norm_fwd_kernel','l2norm_bwd_kernel'):
        tuner=getattr(module,name)
        tuner.configs=[triton.Config({'BT':32},num_warps=4)]
        tuner.cache.clear()
        configuration.append({'kernel':name,'BT':32,'num_warps':4})
    return configuration
