"""Locate real-Depo batch sensitivity without changing frozen model weights/kernels."""
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import ddp_train as d
from models import LM

expected = json.loads((ROOT/'evidence/gsa2-diagnostic-source.json').read_text())
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == expected['script_sha256']
assert d.digest() == expected['inherited_sources']
for name, wanted in json.loads((ROOT/'evidence/snapshot-sha256.json').read_text()).items():
    assert hashlib.sha256((ROOT/'snapshot'/name).read_bytes()).hexdigest() == wanted, name

torch.set_num_threads(2)
torch.cuda.set_device(0)
torch.backends.cuda.matmul.allow_tf32 = False
rows = [d.base.Corpus('depo2', 'train', 'tune').row(i) for i in range(17)]
ids, mask = d.base.batch(rows)
valid = torch.arange(ids.shape[1])[None] < torch.tensor([len(r['input_ids']) for r in rows])[:, None]
supervised = torch.zeros_like(valid)
supervised[:, :-1] = mask[:, 1:].cpu()
assert ids.shape[1] > 64, 'Eval-mode dense comparison requires the same chunk kernel'
model = LM('gsa2', 731, checkpoint_layers=False).cuda()


def forward(x, training):
    model.train(training)
    outputs = []
    hooks = [layer.register_forward_hook(lambda m, i, o: outputs.append(o.detach().float().cpu()))
             for layer in model.layers]
    with torch.no_grad(), d.amp('bf16'):
        outputs.append(model(x).detach().float().cpu())
    for hook in hooks:
        hook.remove()
    return outputs


def compare(reference, actual, selection=valid):
    result = []
    for index, (a, b) in enumerate(zip(reference, actual)):
        x, y = a[selection], b[selection]
        finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
        result.append({'layer': index if index < 15 else 'final_norm', 'finite': finite,
                       'rms_relative': float((x-y).square().mean().sqrt()/x.square().mean().sqrt().clamp_min(1e-8)) if finite else None,
                       'max_abs': float((x-y).abs().max()) if finite else None})
    return result


report = {'status': 'running', 'sources': d.digest(), 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'scope': 'Forward-only localization on the exact real Depo rows used in failed gradient witness; zero optimizer steps',
          'shape': list(ids.shape), 'lengths': [len(r['input_ids']) for r in rows], 'checks': {}}
dest = ROOT/'evidence/gsa2-real-layout.json'


def record(name, reference, actual, selection=valid):
    report['checks'][name] = compare(reference, actual, selection)
    print('REAL_LAYOUT', name, json.dumps(report['checks'][name]), flush=True)
    dest.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


baseline = forward(ids, True)
record('same_shape_repeat', baseline, forward(ids, True))
record('dense_eval_same_chunk', baseline, forward(ids, False))
permutation = torch.arange(len(rows)-1, -1, -1, device='cuda')
record('permutation', baseline, [x.flip(0) for x in forward(ids[permutation], True)])
changed = ids.clone()
changed[1:] = 1
first = torch.zeros_like(valid)
first[0] = valid[0]
record('other_document_intervention', baseline, forward(changed, True), first)
for global_padding in (True, False):
    combined = [torch.zeros_like(x) for x in baseline]
    for rank in range(8):
        indices = list(range(rank, len(rows), 8))
        local, _ = d.base.batch([rows[i] for i in indices])
        if global_padding:
            local = torch.nn.functional.pad(local, (0, ids.shape[1]-local.shape[1]))
        for target, output in zip(combined, forward(local, True)):
            target[indices, :output.shape[1]] = output
    name = 'partition_global_pad' if global_padding else 'partition_local_pad'
    record(name, baseline, combined)
    record(name+'_supervised', baseline, combined, supervised)
# Ranks can use multiple microsteps: preserve the actual trainer's rank-local shape rule.
padded = math.ceil(max(len(r['input_ids']) for r in rows)/16)*16
micro = min(math.ceil(len(rows)/8), 2**int(math.log2(max(1, 8192/padded))))
combined = [torch.zeros_like(x) for x in baseline]
for rank in range(8):
    indices = list(range(rank, len(rows), 8))
    for start in range(0, len(indices), micro):
        take = indices[start:start+micro]
        local, _ = d.base.batch([rows[i] for i in take])
        for target, output in zip(combined, forward(local, True)):
            target[take, :output.shape[1]] = output
record('actual_rank_microbatch', baseline, combined)
report.update(status='completed', microbatch_documents=micro, training_steps=0)
dest.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
