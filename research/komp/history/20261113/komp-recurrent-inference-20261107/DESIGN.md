# KOMP recurrent inference

This isolated package adds inference state without mutating the frozen training or live evaluation packages. It follows GSA's separation of convolution state and recurrent state, while retaining KOMP's dictionary compression semantics.

## Supported variants

| Frozen model | Compression | Decay axis/formula | Eviction | Output/projection |
| --- | --- | --- | --- | --- |
| Exact, Exact+123 | Every token | physical slot, log-sigmoid / 8 | FIFO | original modules |
| Block, Block+123 | every 128 tokens | physical slot, log-sigmoid / 8 | FIFO | original modules |
| learned, learned+123 | every 128 tokens | physical slot, GDN | score top-M | original modules |
| old Value | every 128 tokens | value column, log-sigmoid / 8 | FIFO | original modules |
| Value+learned+123 | every 128 tokens | value column, GDN | score top-M | original modules |

The score write gate is applied once by the original formula `beta * 2 * sigmoid(score)`. Rankings are nondifferentiable and ties retain newer atoms. The three options change original projection/gate modules, which are reused rather than reinitialized. Weight keys, state dictionaries and model residual/MLP modules are unchanged.

## Persistent state

Each layer owns three native short-convolution states plus `KOMPState`: normalized keys, FP32 coefficients, score and birth metadata, physical slot assignments and global position. The independent FP64 reference route stores FP64 keys/coefficients. Exact keeps M atoms. Block keeps M plus the current open chunk's temporary atoms, at most M+127. No prompt history is retained by the operator.

Block output is read before end-of-chunk projection. An unfinished chunk **must not be compressed at a prefill/decode API boundary**. `compressed_state` returns a terminal view for comparison/export; it is explicitly rejected if reused to continue a partial chunk. Whole native prefill chunks are retained, followed by an open recurrent tail. Full C128 boundaries coincide with the M32 physical-slot cycle.

`ModelCache` belongs to one request and supports reset and batch selection/duplication. Native short-convolution decode mutates its own cache tensors. Sharing one cache across concurrent requests is not supported. No pad tokens are treated as valid state updates: masked or unequal-length batches are rejected until a separate length/mask acceptance exists.

## Numerical and performance contract

This is an inference-only API. No backward support is claimed. The single-token Triton kernel fuses decay, delta write and read, with runtime dictionary length to avoid one compilation per length. Projection/compression currently uses batched PyTorch solve. This is a hybrid recurrent implementation, not an all-Triton claim. Native whole-chunk prefill preserves the frozen operator's numerical route; recurrent decode changes reduction order and must pass checkpoint tests before use in evaluation.

Frozen operator tolerances: FP32 RMSrel < .003 OR abs < .0003; BF16 RMSrel < .025 OR abs < .0003. Model logits use RMSrel < .003 OR abs < .0003 **and** identical greedy top1 on tested sequences. These gates do not imply bitwise equality. The independent CPU FP64 token comparisons use atol/rtol 2e-11; arbitrary split continuity is exact for the Torch route. True cache efficacy and numerical acceptability must be established separately.

The current 24-GPU evaluator and 8-GPU training job remain unchanged. Integration is allowed only after operator and real-checkpoint acceptance; retaining existing evaluation scores does not authorize silently mixing a new decoding implementation into the old result set.

## Commands

CPU: `.venv/bin/python -m pytest -q tests/test_recurrent.py`.

`source-manifest.json` binds all implementation files, tests, entrypoints, this design and `validation-plan.json`; every GPU entry verifies the complete manifest. `python verify_sources.py --base /gpfs/users/linlizhi` checks the candidate, all frozen training source files, data manifest and fixed checkpoint files without loading CUDA. `--source-only` omits checkpoint files, while `--candidate-only` checks just the portable package.

All checkpoints are fixed in `validation-plan.json`; there is no `latest.json` lookup. Original Exact is checkpoint-007168; the six completed KOMP variants use checkpoint-012970. New Value+learned+123 is checkpoint-001792 / 939,524,096 effective tokens, SHA `dd381260e7aaced707d1be708416a8cf0225dc53b20111bceec841e5652ab37e`. Its complete marker and all eight rank files were confirmed; this is a partial training checkpoint, not its eventual final model.

Single isolated GPU, one variant per fresh Python process: `python gpu_validate.py --model block --operators-only`. Available model names: `block`, `exact`, `block123`, `exact123`, `learned`, `learned123`, `value`, `value_learned123`. The source/model mapping is explicit for all eight: original/combo packages use their `legacy_models` operators, learned packages use their frozen native FLA operators, old Value uses its value-decay operator, and new Value uses its own value-learned operator.

The root must independently identify an idle GPU and set `CUDA_VISIBLE_DEVICES` to expose exactly that one device. The entry never selects, claims or submits GPU resources. Run `python gpu_entry.py --attempt r1 --stage operators` for all eight in the frozen order. Every child has its own log and evidence file, and the first failure terminates the entire sequence without retry. Existing attempt directories are rejected.

After successful operators, run `python gpu_entry.py --attempt r1-models --stage models --operator-attempt /absolute/package/attempts/r1`. This consumes each successful operator receipt bound to the same full source manifest, so passing operator tests are not repeated. `--models block exact` restricts a run while retaining frozen order. Fixtures and benchmark use analogous `--stage fixtures` / `--stage benchmark` plus `--acceptance-attempt /absolute/package/attempts/r1-models`; fixtures additionally require `--fixtures /absolute/fixtures.json`. `--stage all` runs operators, models, observed fixtures and benchmark serially in one attempt and requires `--fixtures`.

Do not overwrite a frozen manifest after code changes. Preserve the complete attempt and source bundle, make a new revision, then run `python freeze_sources.py` once in that new directory. The manifest generator uses exclusive creation.

After full acceptance: `python benchmark.py --model block`. B1/2/4/8 and prompt128/2048/8192 are measured against the cache-free reference on the same GPU, with cold/warm prefill, decode latency, state size and allocated peak recorded separately. Profiling is not included in timed measurements.

Whole original-batch greedy gate: `python fixture_validate.py --model block --fixtures /path/to/komp-recurrent-inference-fixtures-20261107/fixtures.json`. This first reproduces immutable saved golden IDs/text/scores with the old cache-free implementation, then compares independent recurrent generation at unchanged B and row order. Stopped rows remain in the batch; only active-row outputs matter. The currently training Value+learned+123 has input-only fixtures and cannot pass an observed-golden gate until it has its own frozen evaluation outputs.

## Current evidence

21 CPU tests passed for independent frozen slot and value references, arbitrary partial-chunk splits, score ties, native-prefill metadata, batch reordering, model cache reset, unchanged parameters and rejected padding. GPU correctness and real-checkpoint generation are pending. No inference speedup is claimed before measurements.
