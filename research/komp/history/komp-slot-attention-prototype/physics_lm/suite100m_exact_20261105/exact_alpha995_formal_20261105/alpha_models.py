"""Single-factor alpha0=.995 initialization of the original formal Exact model."""
import math
import torch
from models import LM as OriginalLM, metadata as original_metadata

ALPHA0=.995


class LM(OriginalLM):
    def __init__(self,kind,seed,**kwargs):
        assert kind=='exact','This isolated experiment only permits Exact-KOMP'
        super().__init__(kind,seed,**kwargs)
        original=math.log(.98**8/(1-.98**8))
        replacement=math.log(ALPHA0**8/(1-ALPHA0**8))
        with torch.no_grad():
            for layer in self.layers:
                assert torch.all(layer.mix.decay.bias==torch.full_like(layer.mix.decay.bias,original))
                assert torch.count_nonzero(layer.mix.decay.weight)==0
                layer.mix.decay.bias.fill_(replacement)


def metadata(model):
    return {**original_metadata(model),'alpha0':ALPHA0,'alpha0_reference':.98,
            'experiment':'single-factor alpha995 formal; other original Exact settings unchanged'}
