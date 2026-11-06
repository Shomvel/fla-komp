# Block-KOMP startup compilation, 2026-11-05

The two formal jobs initially logged only step1 for many minutes. This was not sufficient evidence of a deadlock. Northjob telemetry showed0–1.28% GPU utilization and approximately889W per8GPU group, while Triton kept generating Block kernels. A later snapshot recorded1,235 modified/new compilation files within240seconds, including a `block_mapgrad_kernel.cubin` only1.1seconds old. Brevo subsequently reached step25 at elapsed1280.29seconds, confirming optimizer progress after the sparse log interval.

At20:45, Depo-specific console inspection confirmed all8 training ranks had open temporary PTX/compiler-log files and each owned a newly started `ptxas` subprocess consuming99–100% CPU. This directly attributes active compilation to Depo, unlike shared-cache activity alone. All8 training ranks had accumulated approximately21–22CPU minutes. GPU memory was approximately8165MiB per GPU, with0% utilization at the snapshot. The console contract was read from the deployed frontend; only `ps`, file-descriptor links and GPU telemetry were read. Nonblocking py-spy sampling was denied by ptrace permissions; no signal, process restart or permission change was attempted. Evidence: `depo-process-20261105-2042.txt` and `depo-compile-fds-20261105-2045.txt`.

`block-shape-audit.json` reconstructs the exact formal data order and asserts its SHA against the training config. Depo has55 distinct `(B,T)` shapes in its first25steps and114 over the whole run. Brevo has31 in its first25steps and80 overall. Rank-local padding to16 and multiple rank-local microbatches produce several lengths in one global batch. The training logger records step1 and then every25steps, obscuring this slow startup.

Static optimization candidates, not implemented in the frozen training source:

- Most Block kernels declare `T` and `NC` as `tl.constexpr`, although these values are used for pointer strides, bounds/masks and loop limits. Tensor tile shapes come from `C`, `BM`, `BD`, `BV`, `BU`, not `T` or `NC`. Removing unnecessary length specialization could share the same compiled kernel across many document lengths.
- Propagate runtime `T/NC` through helper signatures too: `_dict`, `_sel`, setup helpers, `block_mapgrad_kernel`, final-row kernels, and the imported scan/offset/output kernels. Changing only public kernel signatures can still force specialization in callees.
- Scan `G/NG` are also potential runtime scalars. Preserve compile-time tile dimensions and numerical modes. Runtime loop bounds are supported by Triton; do not convert tile-shaping `arange` dimensions to runtime values.
- Consider `do_not_specialize` for scalar length arguments if residual1/divisibility specialization remains. Expected code-count reduction is not a measured hot-kernel speedup; constant-stride optimization may be lost.
- Avoid changing document order or training padding merely to hide kernel specialization in this formal comparison. The present frozen jobs keep their original recipes and source hashes.

A future implementation must be isolated from these running snapshots, checked against the same outputs/gradients, then measure both cold compile count/time and warmed forward/backward on representative short, partial-chunk and long shapes. No dynamic-kernel change or benchmark has been run here.

Evidence: `block-stall-gpu.json`, `block-compile-evidence.json`, `block-shape-audit.json`, and the original formal metrics/logs on GPFS.
