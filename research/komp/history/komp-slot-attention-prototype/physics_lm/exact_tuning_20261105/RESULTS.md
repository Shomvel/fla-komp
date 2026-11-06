# Exact 初始超参筛选

每任务9候选，各10M token，seed731。以下只列已观测结果。

## depo2

| 候选 | 状态 | 验证CE | 相对baseline |
|---|---|---:|---:|
| baseline | completed | 1.097439 | +0.000000 |
| alpha0-0.995 | completed | 1.088959 | -0.008480 |
| alpha0-0.999 | completed | 1.093032 | -0.004407 |
| beta0-0.1 | completed | 1.093708 | -0.003731 |
| beta0-0.4 | completed | 1.098083 | +0.000644 |
| temperature-0.25 | completed | 1.098999 | +0.001561 |
| temperature-1 | completed | 1.098340 | +0.000901 |
| ridge-1e-05 | completed | 1.098250 | +0.000811 |
| ridge-0.001 | completed | 1.096624 | -0.000815 |

## brevo2

| 候选 | 状态 | 验证CE | 相对baseline |
|---|---|---:|---:|
| baseline | completed | 1.280947 | +0.000000 |
| alpha0-0.995 | completed | 1.257568 | -0.023379 |
| alpha0-0.999 | completed | 1.280341 | -0.000606 |
| beta0-0.1 | completed | 1.265581 | -0.015366 |
| beta0-0.4 | completed | 1.343016 | +0.062069 |
| temperature-0.25 | completed | 1.283687 | +0.002740 |
| temperature-1 | completed | 1.385542 | +0.104595 |
| ridge-1e-05 | completed | 1.288882 | +0.007935 |
| ridge-0.001 | completed | 1.276946 | -0.004001 |
