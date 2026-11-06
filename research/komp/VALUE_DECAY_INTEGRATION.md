# Value-decay integration

Base commit: `439072f54a67cc35b6310651e43ce56b4ccbe4a7`. This change adds value-column decay to the maintained operator/model API. It does not change any frozen training package or launch a pretraining experiment.

Public design: `fla.ops.komp.{exact_komp_value_decay,block_komp_value_decay}`, with `triton`, `hybrid` and `recurrent` backends; `fla.layers.KOMPValueDecayMixer`; and the `komp_value_decay` Transformers registration. The operator contract and model defaults are documented in [VALUE_DECAY.md](../../fla/ops/komp/VALUE_DECAY.md). Existing `KOMPConfig` dimensions, parameters, initialization and slot-decay kernels retain their defaults. Three class factories allow the new model to share the established FLA block/model/LM and loss behavior.

The coordinate transformation and hybrid fallback originate from `/Users/shom/Downloads/komp-slot-attention-prototype`, previously validated on H100 with 288 checks. That historical result does not substitute for testing the package imports, wrappers, configuration or new layer/model. New validation uses the package's FP64 recurrent oracle and NaN-poisoned operator allocations.

## Fixed numerical contract

Relative RMS <= .003 for FP32 and <= .025 for BF16, with absolute-error shortcut <= 3e-4, inherited from the value-decay prototype. The unchanged original slot tests retain their own criteria. Value-decay tests cover outputs, final keys/coefficients and all seven supported gradients, Exact ring conversion, C16/128/tails, small non-native geometry fallback, strong forgetting, the residual-range 24 fallback boundary, streaming and causal prefixes. CPU FP64 gradcheck validates the independent oracle.

Model tests cover default parameter count, Auto registration/config roundtrip, decay initialization, model loss/backward, checkpoint/eager gradient agreement, Adam update, save_pretrained/from_pretrained logits, and unsupported cache/mask rejection. The standalone short E2E also runs both default 341M models through the public CLI with checkpoint save and resume loading.

## Evidence

Each GPU run uses a frozen source manifest and an isolated single-H100 WeaverRSI/proj-32 job, with logs and structured receipts under `/Users/shom/Downloads/fla-komp-value-integration-20261106-rN` and the same basename under `/gpfs/users/linlizhi`.

The first run failed during test collection because the implicit `examples` namespace was shadowed by an installed regular package. Both runnable example directories now have explicit `__init__.py` files. No numerical tests ran in that attempt; its log and terminal/cleanup evidence are preserved. Subsequent result details will be recorded after execution, not inferred from compilation or import success.

The second run passed 11 cases before a new 1e-5 streaming assertion failed on Block C16. A separate unchanged-prototype versus integrated comparison reproduced identical numbers: output max 3.44247e-4 / RMSrel .0003314; state max .0011513 / RMSrel .0003995; edited-future prefix max .0003263 / RMSrel .0002703. The FP64 oracle agreed at ~2e-16. Thus the original proposed strict native streaming criterion was incorrect for the inherited TF32 route. The native streaming check now uses the already established FP32 .003 / abs3e-4 numerical budget, while a separate hybrid streaming case retains the original strict criterion. Forward/state/all-VJP oracle tolerances are unchanged. The failed run remains failed, and no universal strict-causality claim is made. The diagnostic's unrotated slot-state streaming control was not a valid per-physical-slot gate comparison and is not used to justify acceptance.

The third run's model test passed loss/backward, checkpoint/eager gradient comparison and Adam, then exposed missing Transformers `all_tied_weights_keys` metadata during `from_pretrained`. Shared KOMP constructors skipped `post_init` to preserve custom gate initialization. They now call it under Transformers' `no_init_weights` context to establish loading metadata without modifying weights or RNG. Both base-model and causal-LM roundtrips are covered.

## Final integration result

Run r4 passed **34 tests, zero failures/skips** on NVIDIA H100 80GB, PyTorch 2.10.0a0+b558c986e8.nv25.11, Triton 3.7.1, Transformers 5.6.2 and CUDA 13.0. Coverage includes the new operator/model tests and existing slot-model contract and geometry-cache tests. Pytest reported 14 dependency/configuration warnings. The [receipt](evidence/value-decay-20261106/r4/integration.json), [JUnit results](evidence/value-decay-20261106/r4/tests.xml), log and source manifest are retained alongside r1–r3 failure/diagnosis records.

Both default 341,552,000-parameter models completed a BF16-autocast forward/backward/Adam update through `python -m examples.komp.train --decay-axis value`, using B=1, T=17 and synthetic uint16 tokens. Exact loss was 10.4783451 with gradient norm 16.6698475; Block loss was 10.4782140 with gradient norm 16.6666088. Each checkpoint contained finite model tensors and 507 optimizer states, and the `--resume` invocation loaded successfully. Resume stopped at the configured one-step limit: this validates loading, not a subsequent-step continuation-equivalence test. The measured first-step times include compilation and are not warm throughput measurements.

The 851 files in the frozen manifest matched the working tree when checked. Job `job-edcf9dec1543-20261106193320` reached Completed; exact-alias deletion was followed by an empty job query. No GPU test resources remain from r1–r4. This integration has no 8-rank, full-length production-batch or long-pretraining quality acceptance; previous prototype operator timings remain separate evidence.
