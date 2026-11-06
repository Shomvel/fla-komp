# FP32 geometry: fixed checkpoint test

Tested the completed alpha0=.995 checkpoint in each task; it was the best of7completed candidates when selected, not the final nine-candidate winner. Both checkpoints were trained withFP64geometry/FP32recurrent state and evaluated withBF16network activations. Only projection geometry changes in this test. No optimizer steps or training tokens were added.

| Task | FP64 full validation CE | FP32 full validation CE | ΔCE FP32−FP64 | Supervised hidden relative RMS | Maximum selected gradient relative RMS |
|---|---:|---:|---:|---:|---:|
| depo2 | 1.0889589048 | 1.0889503448 | -0.00000856 | 0.001251 | 0.036637 |
| brevo2 | 1.2575677709 | 1.2575364858 | -0.00003129 | 0.001744 | 0.016833 |

FP64 validation reproduced the saved screen scores exactly. All outputs, CE and selected gradients were finite. The largest gradient difference was the first-layer key projection (3.66% Depo2;1.68% Brevo2). The initial-dictionary gradients were zero for this particular real validation batch, so this batch does not test sensitivity of the early FIFO dictionary.

The separate stress matrix used the existing seed917 and M32/K64/T65, baseline alpha=.98,beta=.2,temperature=.5,ridge=1e-4. Both GPU groups reproduced79/80checks passing; the only failure was FP32-input near-collinear initial-dictionary gradient, relativeRMS0.0130098 against unchanged0.005tolerance. Its normalized maximum0.00184783 passed0.005. Random, exact-duplicate and strong-forgetting cases passed, includingBF16 inputs. This numerical boundary failure remains unresolved; it was not converted into acceptance.

The full-validation pass timings were recorded, butFP32 triggered new compilation whileFP64 was warm; these single measurements are not a speed comparison. Fixed-checkpoint inference and gradient proximity does not establish equivalentFP32training trajectories.

Evidence: geometry-transfer-{depo2,brevo2}.json (checkpoint/source SHA included), fp32-projection-diagnostic-{depo2,brevo2}.json. Both interrupted ridge1e-5runs retained checkpoints and resumed from their saved optimizer boundaries.
