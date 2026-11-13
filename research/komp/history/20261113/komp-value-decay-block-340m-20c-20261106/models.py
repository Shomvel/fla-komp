"""Frozen FLA value-decay model with the established trainer contract."""
from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'snapshot'))
from fla.models.komp_value_decay import KOMPValueDecayConfig
from fla.models.komp_value_decay.modeling_komp_value_decay import ValueDecayTrainingLM
from fla.models.komp.conversion import optimizer

def LM(kind, seed=42):
    if kind != 'block':
        raise ValueError('This package authorizes Block only')
    torch.manual_seed(seed)
    model = ValueDecayTrainingLM(KOMPValueDecayConfig(kind=kind))
    assert sum(p.numel() for p in model.parameters()) == 341552000
    return model

def metadata(model):
    return dict(kind=model.kind, parameters=sum(p.numel() for p in model.parameters()),
                model_type=model.config.model_type, config=model.config.to_dict(),
                width=1024, layers=24, ff=2272, heads=16, head_dim=64, slots=32,
                alpha0=.995, beta0=.2, decay_axis='value',
                initialization='native seed42; no legacy remapping',
                checkpoint_policy='mixer_only', head_checkpoint=False,
                projection='float32', state_compute='float32', chunk_tape='bfloat16')
