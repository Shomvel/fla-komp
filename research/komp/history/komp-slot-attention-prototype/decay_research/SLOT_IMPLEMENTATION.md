# Slot pre-decay implementation

The public operator accepts `alpha[B,T,H,M]` with `backend='all_triton'` or `'torch'`. Existing scalar `alpha[B,T,H]` calls retain their original coefficient kernel. Unsupported older backends reject vector gates explicitly.

The semantics are old-slot pre-decay, with independent scalar beta:

```
pre[m,v] = alpha[m] * old[m,v]
err[v] = value[v] - sum_m(hist[m] * pre[m,v])
new[m,v] = pre[m,v] + (proj[m]-onehot[m])*pre[slot,v]
           + beta*onehot[m]*err[v]
```

`prepare_slot_kernel` builds OS/OV/FS/FV by composing the diagonal + rank-two affine maps. Each chunk is scheduled independently. Its internal 16/32 token loop is serial, and the existing block scan consumes the maps. This is an exact, zero-gate-safe correctness-first implementation, not a claim that the slot preparation is parallel within a chunk or faster than the scalar implementation. All coefficients, recurrence, recomputation, analytic backward, and tile reductions execute in Triton.

The coefficient recurrence carries F[M,M] and G[M,C], initially identity and zero. It applies each transition to their rows, adds the current write to G, and reads the output maps with the current read vector. This avoids dividing cumulative gates or constructing a full [2C,2C,M] pair tensor. Complexity is O(C M² + C² M), with chunk states independent of value width. Padding steps apply identity and zero output maps.

Local backward returns vector da by summing only over values and multiplies dpre rowwise by alpha. Value-tile partials are reduced with the existing Triton reducer. Input checking now uses separate alpha/beta/slot element counts, avoiding out-of-bounds reads when alpha grows by M.

Run `python validate_slot_decay.py --benchmark` on the pinned H100 environment. It checks FP64 dense recurrence outputs/state/all gradients, c16/c32, zero gates, repeated routes, tails, M1/5/32/64, V1/37/64/65, T192/T2048, BF16 values, scalar degeneration (including summed da), streaming gradients, slot permutations, noncontiguous input, output-only/final-only cotangents, finite differences, single-step write order, full geometry gradients, and public input rejection. Results are written to `evidence/slot-decay-gpu.json` including source hashes.

Benchmarks compare scalar, broadcast-identical slot gates, and random slot gates at B64/T192/H2/M32/V64 and B2/T2048/H4/M32/V64, both eager and CUDA Graph. They are core-only measurements; full kernel geometry is excluded. GPU execution is coordinated by the parent agent. Local static Python compilation passed; no GPU pass is asserted by this document before the JSON result exists.
