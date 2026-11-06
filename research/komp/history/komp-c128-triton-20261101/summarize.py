"""Generate the C128 acceptance report directly from GPU evidence."""
import hashlib,json
from pathlib import Path
root=Path(__file__).parent
E=root/'evidence'
lines=['# Block-KOMP Triton C=128','', 'C=128 uses four 32×32 blocks for the unit-lower triangular inverse. The scan uses one pipeline stage; the backward pass splits output gradients into two Triton kernels, followed by triangular-solve and decay gradients. Projection, dictionary selection, scans and gradients remain on GPU.', '', 'The C≤64 default path is preserved. Supported slots remain M≤64; eviction supports FIFO and learned. Coherence/project=False are not added to the Triton API.', '', '## Acceptance','']
for filename in ['smoke-final.json','c128-validation.json','legacy-validation.json','integration.json']:
 data=json.loads((E/filename).read_text());assert data['status']=='passed',(filename,data.get('error'))
 checks=data['checks'];errs=[x['rms_relative'] for x in checks if 'rms_relative' in x]
 lines.append(f'- `{filename}`: passed, {len(checks)} checks'+(f', maximum RMS relative error {max(errs):.6g}.' if errs else '.'))
lines += ['', 'FP64 references: token recurrence for FIFO; flex for learned. RMS relative tolerances are 0.005 for FP32 and 0.01 for BF16. C128 near-duplicate cases use 0.005. Exact equality is required for future-input interventions and the C16/32/64 regression. Streaming permits 0.001 for scan reassociation.', '', '## Timings','', 'H100; per-slot decay; M=32, Dk=Dv=64. Inputs are BF16 for Triton and FP32 for flex, matching their intended experiment paths. Flex timings therefore compare implementations at their intended precision, not identical low-level arithmetic. C64 changes chunk boundaries and is not numerically equivalent to C128. Forward+backward includes autograd overhead; compilation is excluded.', '', '| B | T | H | Eviction | Implementation | Forward ms | Forward+backward ms | Extra peak MiB |','|---:|---:|---:|---|---|---:|---:|---:|']
for row in json.loads((E/'benchmark-stable.json').read_text()):
 b,t,h,*_=row['shape'];lines.append(f"| {b} | {t} | {h} | {row['evict']} | {row['implementation']} | {row['forward_ms']:.3f} | {row['forward_backward_ms']:.3f} | {row['incremental_peak_bytes']/2**20:.1f} |")
lines+=['', '## Reproducibility','',f"- Kernel SHA256: `{hashlib.sha256((root/'work/komp_slot/block_fast.py').read_bytes()).hexdigest()}`.",f'- Base source hashes, patch, environment, acceptance details and exact job cleanup receipts: `{root.resolve()}/evidence/`.', '- This establishes C128 numerical and execution support. It does not establish results for the pending 100M-parameter, five-task training suite.', '']
(root/'C128_TRITON.md').write_text('\n'.join(lines))
