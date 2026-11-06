# Value-decay Exact and Block KOMP

`exact_komp_value_decay` compresses after each token's read. `block_komp_value_decay` compresses after each chunk, including a final partial chunk. Both apply forgetting to coefficient **value columns**, instead of dictionary slots. They use the normalized Gaussian kernel, delta-rule writes, FIFO selection and ridge projection from KOMP.

```python
from fla.ops.komp import exact_komp_value_decay, block_komp_value_decay

# q/k: [B,T,H,K], v/log_decay: [B,T,H,V], beta: [B,T,H]
# keys: [B,H,M,K], coefficients: [B,H,M,V]
y, (keys, coefficients) = exact_komp_value_decay(
    q, k, v, log_decay, beta, (initial_keys, initial_coefficients),
)
y, state = block_komp_value_decay(
    q, k, v, log_decay, beta, (initial_keys, initial_coefficients), chunk_size=128,
)
```

The returned keys and coefficients are oldest-first. For Block, splitting an input across calls is algebraically equivalent to a single call only at compression boundaries; no unfinished-chunk cache is retained. The layer/model APIs support dense pretraining, without padding masks, generation caches, varlen or context parallelism.

## Native route and stable fallback

For each head, write `g[t,v] = s[t] + delta[t,v]`, where `s[t]` is the mean across value channels. Set `r[t,v] = cumsum(delta)[t,v]`. All dictionary-side linear operations commute with right diagonal scaling, so the transformed coefficients `C' = C diag(exp(-r))` obey scalar-decay KOMP with values `v' = v exp(-r)`. Scale the output and final coefficients back by `exp(r)`. A constant coordinate origin centers each sequence/channel's range and also scales the initial coefficients. The origin is detached; the result is invariant to its choice. The complete value-gate gradient still flows through `s`, `r` and all scaling operations.

The native route directly uses the existing Exact/Block Triton geometry, solve, state scan and analytic backward. Scaling uses PyTorch autograd. This is an algebraic change of coordinates, with the native kernels' existing floating-point policies, not a bitwise-equivalent execution order.

The accumulated residual range includes the initial time. If any channel's range exceeds 24, the entire call selects the stable hybrid implementation, which reconstructs decayed states without inverse decay. Unsupported native chunk sizes and FP16 inputs also select this route. No gate clipping, extra compression boundaries or discarded value channels are introduced. Native Block uses its original TF32/BF16 arithmetic: different coordinate origins or streaming splits can change rounding, including a prefix when future gates change the centering origin. Agreement is within the operator numerical budget, not bitwise or a strict 1e-5 guarantee. Use the hybrid/reference route when tighter prefix or streaming agreement is required.

| Property       | Contract                                                                                   |
| -------------- | ------------------------------------------------------------------------------------------ |
| Domain         | Finite inputs, `g<=0`, `0<=beta<=1`, positive finite temperature/ridge                        |
| CUDA backends  | `triton` (native plus fallback), `hybrid` (force fallback); currently validated on NVIDIA     |
| Reference      | `recurrent`, pure PyTorch with FP64 when any input is FP64                                   |
| Limits         | CUDA `M<=64`, `chunk_size<=128`; native `K,V<=128`, C=1/16/32/64/128                          |
| Precision      | FP32 state/scaling; original Block TF32/BF16 attention/tape policy retained                  |
| Gradients      | First order for q/k/v/g/beta/initial keys/initial coefficients, including final-state losses |
| Output         | Same dtype as v; oldest-first FP32 state on CUDA                                            |
| Unsupported    | Higher-order Triton gradients, generation caches, varlen, CP and CUDA graph capture         |

Input validation and data-dependent fallback synchronize with the host. Near-collinear dictionary limitations of the original FP32 solve remain. The fallback is numerically stable for strong forgetting but can be slower and use substantially more memory, especially for Exact. A finite exponent range does not promise a universal condition-number/error bound.

## Model definitions

```python
from transformers import AutoModelForCausalLM
from fla.models.komp_value_decay import KOMPValueDecayConfig

config = KOMPValueDecayConfig(kind='block')  # or 'exact'
model = AutoModelForCausalLM.from_config(config)
# Dense inputs; labels follow the standard causal-LM shift convention.
loss = model(input_ids=ids, labels=ids).loss
loss.backward()
model.save_pretrained('/path/to/value-komp')
```

`KOMPValueDecayModel`, `KOMPValueDecayForCausalLM` and `KOMPValueDecayConfig` are exported from `fla.models`; the standalone mixer is `fla.layers.KOMPValueDecayMixer`. The registered model type is `komp_value_decay`. `kind` selects the operator. The base model and LM share FLA's native block/MLP/loss loop with the existing KOMP adapter.

Default architecture: D1024/L24/H16/K64/M32, FF2272, vocabulary 32000, conv4, FP32 gate computation, beta=.2, and mixer-only activation checkpointing. With the default flags it has **341,552,000 parameters**, including 798,720 more than the FF2272 slot-decay combination. This is a separate architecture and initialization; old slot-decay state dicts are not interchangeable with it.

The three existing combination options are independent configuration fields, enabled by default: `output_sigmoid`, `low_rank_gates`, `multi_timescale_init`. Low-rank output and decay projections have rank `gate_rank=64`, no intermediate activation, and no output-gate bias. Decay uses random down/zero up initialization. Initial per-head decay rates have factors from 1/4 to 4 normalized by their arithmetic mean, preserving mean log(alpha)=log(.995); all value channels within a head start equal. Set `multi_timescale_init=False` for uniform alpha=.995. Supported dimensions and FFN/vocabulary/layer counts can be changed through the new config; the old frozen `KOMPConfig` contract is unchanged.

Use `--decay-axis value --kind exact|block` in `python -m examples.komp.train`. New models initialize directly from the configured seed; they do not invoke the legacy slot checkpoint converter. Resume compares the complete serialized model configuration along with the existing data/batching/schedule contract. `--cache-geometry` is rejected for value decay pending dedicated validation; mixer checkpointing remains enabled by default.

## Validation

The port originates from the independently validated value-decay prototype, including the inverse-free fallback and the native coordinate transform. Integration tests live in `tests/ops/test_komp_value_decay.py` and `tests/models/test_komp_value_decay.py`. Existing slot-model compatibility tests remain in `validation/komp/test_contracts.py`.

```bash
FLA_CI_ENV=0 pytest -x tests/ops/test_komp_value_decay.py tests/models/test_komp_value_decay.py \
    validation/komp/test_contracts.py validation/komp/test_geometry_cache.py
```

The numerical gate retains the prototype's FP32 relative RMS .003 / BF16 .025, with an absolute-error shortcut of 3e-4. Tests exercise all seven gradients and final states against the FP64 token oracle, strong forgetting, fallback, chunk tails, streaming, causal prefixes, model checkpoint recomputation, an Adam update, and Hugging Face save/load. The prototype's earlier speed measurements are not a new full-model benchmark of this package.
