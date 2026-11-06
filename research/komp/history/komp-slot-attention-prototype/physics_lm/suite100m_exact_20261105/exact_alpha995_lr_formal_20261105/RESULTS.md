# Alpha .995 formal learning-rate comparison

New training 4/4; centers reused 2/2; full new heldout 4/4.

Every run uses100Mpositions, FP32projection/state, BF16network, alpha.995, seed731 andWD.1. Rank LR only by final validation CE. These results are separate from the10MFP64screen.

| Task | LR | Source | Validation CE | Test CE | Generation accuracy |
|---|---:|---|---:|---:|---:|
| depo2 | 0.00025 | completed | 0.701141 | 0.704862 | 5.86% |
| depo2 | 0.0005 | reused100Mcenter | 0.669557 | 0.671253 | 5.47% |
| depo2 | 0.001 | completed | 0.630805 | 0.631022 | 7.81% |
| brevo2 | 0.0005 | completed | 0.761777 | 0.768237 | 10.16% |
| brevo2 | 0.001 | reused100Mcenter | 0.734830 | 0.742074 | 10.94% |
| brevo2 | 0.002 | completed | 0.708866 | 0.711004 | 11.72% |

Complete-task validation winners: {'depo2': 0.001, 'brevo2': 0.002}.
