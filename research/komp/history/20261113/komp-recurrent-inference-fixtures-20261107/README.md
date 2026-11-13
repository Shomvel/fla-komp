# Recurrent inference regression fixtures

This directory is independent of running evaluation/training. `fixtures.json`
contains exact materialized prompts, full and effective input IDs, generation
parameters, whole original batches, frozen model receipts and copied observed
outputs. `manifest.json` pins the portable file checksum and coverage. Everything
was read from existing artifacts; no GPU inference or queue operation was run.

Use `batches[*].request_ids` in its existing order. Look up each row in `requests`.
Use `effective_input_ids` for prefill. They already implement the historical
context reservation and left truncation. Never add BOS, template, padding or a
second truncation. The original limits are NIAH 65536 minus128, PhoneBook32768
minus20 and BASED2048 minus48. Tokenizer/hash and raw requests hashes are recorded.

The seven cases cover PhoneBook32/256, SWDE, FDA, NIAH single1 at1024/8192 and
single3 at2048. Some KOMP models have not reached the long tasks in the running
evaluation: those entries are explicitly input-only, not invented goldens.
`golden.path` references the exact immutable remote batch and `golden.sha256`
pins its bytes. Saved IDs, decoded text, scores and stop lengths are copied.
No logits or recurrent states were saved by the old runner.

Eight models have frozen evaluation receipts. `value_learned123` has input-only
coverage and observed training metadata. At collection it has no complete.json;
no checkpoint weights were opened or hashed. Its latest.json is a pointer, not a
frozen reference checkpoint. Freeze an existing checkpoint and record its hash
before running numeric comparisons. Never reuse another model's golden output.

No EOS was observed in completed batches of the selected cases. Three clearly
labelled synthetic token-stream tests cover EOS protocol behavior, without
claiming real model EOS coverage. Real newline stops, generation-budget stops,
mixed row stop lengths and left truncation are present in observed goldens.

## Candidate acceptance

1. Pin the exact frozen checkpoint, source, tokenizer, dtype and model options.
   All old runs retain FP32 weights/residuals/gates with BF16 autocast. Treat KDA,
   Block, Exact, learned eviction, options123 and Value as separate contracts.
2. Reproduce an observed full-prefix greedy batch with its original batch size,
   row order and no padding. Compare its complete generated IDs with the saved
   golden before assessing the candidate. If this fails, first resolve reference
   environment/batch/numerics drift; do not blame recurrent cache.
3. For logits/state tests, keep reference and candidate B,T, fixed input token
   stream and precision identical. Compare full-prefix last-token logits with
   prefill+one-token recurrent decode at every teacher-forced step. Teacher force
   the same old generated IDs to avoid amplification from diverged greedy paths.
   Record max absolute and RMS-relative error, argmax, and top-two margin. Do not
   select tolerances after seeing failures or equate a changed argmax with a large
   logit error; near ties need explicit evidence.
4. Test actual prompt prefixes at short-conv boundaries and T=127/128/129,
   255/256/257, 1023/1024/1025,1999/2000, then full8192-class contexts. These are
   derived input tests, not saved generation goldens. Prefill split equivalence,
   reset, independent sequences, same batch with different contents, and repeated
   decode must not leak state. Do not mutate training checkpoints.
5. Run independent greedy generation at the original whole batch shape, with
   original per-row EOS/newline/max-generation termination. Require unchanged
   generated IDs/text/score for regression batches; where exact greedy identity
   fails, investigate fixed-token logits before claiming task equivalence.

The old generator keeps all rows in a fixed B dimension after a row stops; it
freezes that row's logical length and keeps EOS in its unused buffer while other
rows continue. It gathers each row's last logical position. A candidate may mask
inactive recurrent rows, but must preserve outputs of all active rows. Do not
compact the active batch during the first comparison. A B=1 rerun cannot prove
parity against a B=8 saved golden: different GEMM/kernel paths can change BF16
rounding independently of caching. Batch invariance is a separate test matrix.

Original Exact padding previously changed logits (RMS-relative0.00414); padding
was rejected rather than widening tolerances. Cached prefill must likewise match
true lengths. Verify causal behavior across chunk boundaries before assuming a
training chunk operator is exactly equivalent to token recurrence.

Performance measurement comes after numerical acceptance. Warm both paths and
exclude compilation. Use the same batch/contexts/generated token counts, report
prefill and decode separately, synchronize CUDA, and report GPU memory. Compare
at1024/2048/8192-class contexts and short PhoneBook; never mix speedups from a new
batch shape with speedups from caching. Existing long cold-compile pauses are not
a fair cached-decode baseline.
