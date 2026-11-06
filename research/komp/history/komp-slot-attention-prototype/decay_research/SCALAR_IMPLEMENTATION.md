# GDN scalar timescale implementation

Also exposes `gsa_decay(logits, normalizer=8.)`, computing FP32
`exp(logsigmoid(logits)/normalizer)` in one Triton forward and one backward
kernel. It accepts any nonempty CUDA FP32/FP16/BF16 shape and noncontiguous
layouts. `normalizer` is a positive finite Python scalar; 1 reduces to sigmoid.
The GSA gate only generates alpha; it does not couple beta to `1-alpha`.

`komp_slot.decay.gdn_decay(logits, A_log, dt_bias)` returns FP32 alpha with
`alpha = exp(-exp(A_log) * softplus(logits + dt_bias))`.

`logits` is `[B,T,H]` or `[B,T,H,M]`; both parameters have `logits.shape[2:]`.
The extra slot dimension is supported by the gate, not by the existing scalar
recurrence. Parameter gradients reduce batch/time with two Triton kernels and
FP32 partial sums; no atomics or CPU synchronization. Inputs may be noncontiguous
and FP32/FP16/BF16; leaf gradients retain their leaf dtype. FP32 parameters are
recommended. Only first-order autograd is supported.

Gate arithmetic uses a stable log-product to avoid `exp(A_log) * softplus(x)`
becoming `inf * 0`. CUDA libdevice `log1p` preserves softplus accuracy for negative
logits. `log(softplus(x))` uses its FP32-accurate `x` limit below -20. Above
`log(rate * softplus(x)) = 20`, alpha and its derivatives round to zero in FP32;
the exponent is capped there to avoid intermediate infinities. Finite inputs
are assumed, matching model-internal gate use.

## Integration

Use a bias-free linear projection for logits, initialized to zero. Initialize
`A_log = zeros(H)` and `dt_bias = full((H,), matched_dt_bias(.98))`; this preserves
the previous initial alpha=.98. Continue passing alpha to the existing
`triton_chunk`, retaining independent sigmoid beta and the current pre-decay
ordering. Do not add a second scalar multiplier.

For strict matched experiments, retain the existing weight-decay policy first,
or explicitly exclude A_log/dt_bias and rebuild a matched baseline with its gate
bias excluded too. `_no_weight_decay` attributes alone do not affect the current
experiment's manually constructed optimizer groups. The helper sets no optimizer
attributes, so the choice remains explicit.

## Acceptance

Run `python validate_decay_gate.py --output evidence/decay-gate.json` on CUDA.
The script checks forward and all input/parameter gradients against independent
Torch FP64 arithmetic, FP32/BF16/FP16 inputs, noncontiguous layouts, scalar and
slot-shaped gates, partial blocks, model-size reductions, extreme logits/rates,
matched .98 initialization and rate-one sigmoid equivalence. It emits source
hashes. Additional GSA checks cover normalizers 1/8, gradients, all supported
dtypes, scalar/slot layouts and extreme inputs. It emits environment, errors
and pass/fail JSON. Local syntax compilation alone
does not establish GPU numerical acceptance.
