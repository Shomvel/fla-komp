# Exact per-token KOMP Triton

```python
from komp_slot.exact_komp import exact_komp

out, (keys, coefficients), metadata = exact_komp(
    q, k, v, log_decay, beta, (initial_keys, initial_coefficients),
    eviction="learned", scores=scores,
    compute_chunk=16, projection_dtype="float32",
)
```

每 token 插入、读出并压缩回 M 个原子；FIFO 与固定 input-score learned 规则。
计算块 16/32 不改变 C=1 语义，投影使用直接消元，不引入近似 CG。
前后向核心计算均由 Triton 执行。

默认投影与状态均为 FP32，支持 FP32/BF16 q/k/v；M<=64、K/V<=128。
M32、K=V64 时，字典 keys+coefficients 共 4096 个 FP32 元素/head，与 GDN 的持久矩阵状态相同；learned 元数据另计。
`projection_dtype="float64"` 可用于精度对照；正常训练无需默认开启。
FP32 精度结论基于默认 temperature=0.5、ridge=1e-4 等验收配置，不是任意病态系统的精度保证。

learned 默认额外应用 `beta *= 2*sigmoid(scores)`；若已包含此门，请设 `score_write_gate=False`。
流式续算需携带返回的 scores/births，并正确设置 initial_scores、initial_births、start_position。
硬选择本身无梯度，score 的梯度来自 write gate。只支持一阶反向。

参考 FLA GDN/Mesa 的分块状态、boundary adjoint 与局部重算方式，移除了逐 token 的 M×V 状态及 M×M Gram/inverse tape。
长序列采用分层边界扫描，局部状态/adjoint 改为稳定仿射扫描。learned 在稀疏 32-token 窗口内复用同一字典的 LDLᵀ 分解，批量解 RHS 并合并 Gram 梯度；密集替换窗口用直接 RHS sweep。反向读取保存的 H/R/N。默认仍保留 FP32 状态计算。

最终实现：`komp_slot/exact_komp/{operator,geometry,recurrence,chunk_backward,hierarchical,versioned}.py`。
可复现材料：

- `/Users/shom/Downloads/komp-exact-opt2-20261105/README.md`
- `/Users/shom/Downloads/komp-exact-opt2-20261105/RESEARCH.md`
- `/Users/shom/Downloads/komp-exact-opt2-20261105/RESULTS.md`
- `/Users/shom/Downloads/komp-exact-opt2-20261105/evidence/`

H100 8K 实测、完整梯度与优化器验收见 RESULTS.md。原始实现与优化前测速保留在 komp-exact-triton-20261105 与 komp-gdn-8k-20261105。
当前仍慢于 FLA GDN，主要成本为每 token 投影求解、chunk 边界扫描与局部状态重算。未进行新的 PhysicsLM 质量训练。

第二轮 H100 8K BF16、B1/H2/K=V64/M32、C16、CUDA Graph：FIFO 前反向 2.121 ms，learned 1.688 ms；上一轮分别 3.967/4.138 ms，GDN 0.466 ms。
训练增量峰值约 89.75/91.75 MiB。learned 收益依赖字典接受率；本组每次前向或反向的分解次数减少 92.4%。混合状态操作数仅作为实验，没有设为默认。
