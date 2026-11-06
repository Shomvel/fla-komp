# Block-KOMP Triton C=128

C=128 uses four 32×32 blocks for the unit-lower triangular inverse. The scan uses one pipeline stage; the backward pass splits output gradients into two Triton kernels, followed by triangular-solve and decay gradients. Projection, dictionary selection, scans and gradients remain on GPU.

The C≤64 default path is preserved. Supported slots remain M≤64; eviction supports FIFO and learned. Coherence/project=False are not added to the Triton API.

## Acceptance

- `smoke-final.json`: passed, 40 checks, maximum RMS relative error 0.00356111.
- `c128-validation.json`: passed, 600 checks, maximum RMS relative error 0.00527463.
- `legacy-validation.json`: passed, 573 checks, maximum RMS relative error 0.00716693.
- `integration.json`: passed, 18 checks, maximum RMS relative error 0.00351007.

FP64 references: token recurrence for FIFO; flex for learned. RMS relative tolerances are 0.005 for FP32 and 0.01 for BF16. C128 near-duplicate cases use 0.005. Exact equality is required for future-input interventions and the C16/32/64 regression. Streaming permits 0.001 for scan reassociation.

## Timings

H100; per-slot decay; M=32, Dk=Dv=64. Inputs are BF16 for Triton and FP32 for flex, matching their intended experiment paths. Flex timings therefore compare implementations at their intended precision, not identical low-level arithmetic. C64 changes chunk boundaries and is not numerically equivalent to C128. Forward+backward includes autograd overhead; compilation is excluded.

| B | T | H | Eviction | Implementation | Forward ms | Forward+backward ms | Extra peak MiB |
|---:|---:|---:|---|---|---:|---:|---:|
| 1 | 512 | 8 | fifo | triton_c64 | 0.333 | 0.949 | 20.7 |
| 1 | 512 | 8 | fifo | triton_c128 | 0.332 | 1.044 | 29.5 |
| 1 | 512 | 8 | fifo | flex_c128 | 13.602 | 29.494 | 32.5 |
| 1 | 512 | 8 | learned | triton_c64 | 0.422 | 1.100 | 19.3 |
| 1 | 512 | 8 | learned | triton_c128 | 0.426 | 1.195 | 28.5 |
| 1 | 512 | 8 | learned | flex_c128 | 13.675 | 29.528 | 32.5 |
| 8 | 512 | 8 | fifo | triton_c64 | 0.273 | 0.836 | 165.8 |
| 8 | 512 | 8 | fifo | triton_c128 | 0.377 | 1.959 | 235.8 |
| 8 | 512 | 8 | fifo | flex_c128 | 12.800 | 27.618 | 307.5 |
| 8 | 512 | 8 | learned | triton_c64 | 0.367 | 0.985 | 154.4 |
| 8 | 512 | 8 | learned | triton_c128 | 0.506 | 2.088 | 228.4 |
| 8 | 512 | 8 | learned | flex_c128 | 12.782 | 27.637 | 307.5 |
| 1 | 8192 | 8 | fifo | triton_c64 | 0.336 | 0.958 | 325.1 |
| 1 | 8192 | 8 | fifo | triton_c128 | 0.803 | 3.976 | 465.0 |
| 1 | 8192 | 8 | fifo | flex_c128 | 213.481 | 455.627 | 565.7 |
| 1 | 8192 | 8 | learned | triton_c64 | 1.107 | 1.618 | 301.4 |
| 1 | 8192 | 8 | learned | triton_c128 | 2.321 | 5.437 | 449.2 |
| 1 | 8192 | 8 | learned | flex_c128 | 211.883 | 460.878 | 565.7 |

## Reproducibility

- Kernel SHA256: `79629c0cf0096624c7004a7cab9bd7d41acffb6de93b2a70ded0ed6d0275a9e7`.
- Base source hashes, patch, environment, acceptance details and exact job cleanup receipts: `/Users/shom/Downloads/komp-c128-triton-20261101/evidence/`.
- This establishes C128 numerical and execution support. It does not establish results for the pending 100M-parameter, five-task training suite.
