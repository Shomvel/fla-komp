"""Same-shape single-GPU warm full-model forward/backward cost comparison."""
import json
import os
from pathlib import Path
import statistics
import sys
import torch

ROOT = Path(os.environ['VARIANT_ROOT'])
sys.path.insert(0, str(ROOT / 'snapshot/suite100m'))
from beta_m_models import LM, metadata, BETA0, SLOTS
import train as base

torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
model = LM('exact', 731).cuda().train()
ids = torch.arange(1024, device='cuda')[None].remainder(1000) + 1
mask = torch.ones_like(ids, dtype=torch.bool)


def step():
    model.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss, count = base.loss_sum(model, ids, mask)
        (loss / count).backward()


for _ in range(2):
    step()
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
times = []
for _ in range(5):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    step()
    end.record()
    end.synchronize()
    times.append(start.elapsed_time(end))
report = dict(model=metadata(model), device=torch.cuda.get_device_name(), batch=1, sequence_length=1024,
              dtype='BF16 network/head; FP32 geometry/state', forward_backward_ms=times,
              median_ms=statistics.median(times), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
              peak_reserved_bytes=torch.cuda.max_memory_reserved(),
              scope='Warm synthetic same-shape full-model forward/backward, activation checkpointing on, no optimizer/DDP; not training wall-clock throughput.')
(ROOT / f'evidence/benchmark-beta{BETA0:g}-m{SLOTS}.json').write_text(json.dumps(report, indent=2) + '\n')
print('BENCHMARK', json.dumps(report), flush=True)
