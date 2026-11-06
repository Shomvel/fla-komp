"""Right-pad only the frozen Block model input to its128-token chunk boundary."""
import torch.nn.functional as F
from models import LM as OriginalLM


class LM(OriginalLM):
    def forward(self,ids):
        assert self.kind=='komp' and not self.training,'Evaluation-only Block adapter'
        length=ids.shape[1]
        padded=F.pad(ids,(0,(-length)%128))
        return super().forward(padded)[:,:length]
