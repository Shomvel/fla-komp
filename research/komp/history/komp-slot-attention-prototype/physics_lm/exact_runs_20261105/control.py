import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'snapshot/physics_lm'))
import experiment as base
import torch
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
base.train('gdn', 731, result_root=ROOT/'control', preflight_path=ROOT/'preflight.json')
