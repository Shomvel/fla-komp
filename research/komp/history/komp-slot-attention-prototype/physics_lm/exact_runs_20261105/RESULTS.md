# Exact KOMP：Physics of LM 质量实验

**状态：28 个新训练与 1 个 GDN 复现对照完成。**

沿用冻结 Depo/Brevo 数据：10,001,108 token、4 层宽度128、M32、batch64、2231步，seeds731–737。新四臂均使用 LR5e-4、原 AdamW 配方、BF16 模型投影和 FP32 exact KOMP。算法逐 token 压缩；计算 C16。下表 CE 来自 confirm，生成使用每任务固定128例。

Exact KOMP 在旧字典上计算残差，加入当前原子并读出，然后投影压缩回M；这是 Block-KOMP 的算法C=1。旧gsa_slot采用替换/投影后写入与读出的另一条规则，二者不是仅kernel实现不同。

均值 ± seed 样本标准差；生成以百分比表示。GDN/GSA/GSA2/Block 行为历史结果，未冒充本轮复跑。

| 方法 | n | Content CE ↓ | Depo CE ↓ | Brevo CE ↓ | Depo 生成 % ↑ | Brevo 合法 % ↑ | 总耗时 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| exact_scalar_fifo | 7 | 2.4805 ± 0.0610 | 3.1855 ± 0.0253 | 1.7756 ± 0.1242 | 3.35 ± 1.67 | 11.27 ± 3.69 | 63.8 ± 0.3 |
| exact_slot_fifo | 7 | 2.2440 ± 0.0475 | 3.2178 ± 0.0510 | 1.2701 ± 0.0610 | 4.13 ± 1.25 | 33.71 ± 2.61 | 64.5 ± 0.2 |
| exact_slot_gate | 7 | 2.2015 ± 0.0549 | 3.1853 ± 0.0877 | 1.2177 ± 0.0570 | 4.02 ± 1.99 | 35.49 ± 3.46 | 68.9 ± 0.3 |
| exact_slot_learned | 7 | 2.2704 ± 0.0502 | 2.9874 ± 0.1231 | 1.5534 ± 0.1515 | 5.36 ± 1.65 | 26.12 ± 4.75 | 69.1 ± 0.2 |
| block_c64 | 7 | 2.4135 ± 0.0906 | 3.0830 ± 0.1099 | 1.7440 ± 0.0940 | 5.47 ± 1.28 | 12.61 ± 3.01 | 60.9 ± 0.2 |
| block_slot_c64 | 7 | 2.0917 ± 0.0793 | 3.0440 ± 0.1505 | 1.1393 ± 0.0252 | 5.13 ± 1.34 | 36.50 ± 2.54 | 62.0 ± 0.1 |
| block_slot_gate_c64 | 7 | 2.0519 ± 0.0774 | 3.0133 ± 0.1566 | 1.0905 ± 0.0423 | 6.70 ± 2.06 | 40.07 ± 3.43 | 66.5 ± 0.3 |
| block_slot_learn_c64 | 7 | 2.0067 ± 0.0772 | 2.8728 ± 0.1520 | 1.1406 ± 0.0482 | 7.03 ± 1.19 | 37.28 ± 3.49 | 67.5 ± 0.3 |
| gdn | 7 | 2.3759 ± 0.1088 | 3.0066 ± 0.1257 | 1.7451 ± 0.1584 | 3.35 ± 1.33 | 11.16 ± 3.49 | 71.8 ± 1.2 |
| gsa | 7 | 2.7163 ± 0.0172 | 3.2943 ± 0.0224 | 2.1382 ± 0.0160 | 2.90 ± 1.08 | 3.01 ± 1.31 | 77.2 ± 2.9 |
| gsa_slot | 7 | 2.2469 ± 0.0467 | 3.2131 ± 0.0407 | 1.2808 ± 0.0731 | 4.13 ± 1.33 | 33.93 ± 2.34 | 67.7 ± 0.3 |
| gsa2_native_lr5e4 | 7 | 2.6069 ± 0.1486 | 3.2139 ± 0.3478 | 1.9999 ± 0.1143 | 5.58 ± 1.83 | 11.94 ± 3.46 | 195.5 ± 1.6 |
| gsa2_native_lr1e3 | 7 | 1.9819 ± 0.0963 | 2.2753 ± 0.2214 | 1.6884 ± 0.0729 | 11.94 ± 3.24 | 20.31 ± 2.51 | 194.2 ± 0.7 |
| gsa2_noconv_lr5e4 | 7 | 2.3151 ± 0.1493 | 2.8243 ± 0.3210 | 1.8060 ± 0.0941 | 5.02 ± 2.66 | 13.28 ± 3.19 | 154.2 ± 4.0 |

耗时包括训练、验证与最终生成，排除进程启动和 checkpoint reload；跨历史作业的耗时仅供参考，不能替代同卡算子测速。

GSA2 native 使用64 slots、8192状态元素/head及短卷积，LR1e-3为此前验证集选定的配方；KOMP/GDN为4096状态元素/head，learned元数据另计。旧gsa_slot还有decay.bias的WD豁免差异。

## 配对比较

A−B；CE负值更好，生成正值更好。固定数据集上的seed配对bootstrap95%区间，20,000次抽样；未做多重比较校正。

| A − B | 指标 | n | 差值 | 95% CI |
|---|---|---:|---:|---|
| exact_slot_learned − exact_slot_fifo | Content CE | 7 | +0.0265 | [-0.0103, +0.0650] |
| exact_slot_learned − exact_slot_fifo | Depo CE | 7 | -0.2304 | [-0.3188, -0.1570] |
| exact_slot_learned − exact_slot_fifo | Brevo CE | 7 | +0.2833 | [+0.2125, +0.3671] |
| exact_slot_learned − exact_slot_fifo | Depo generation % | 7 | +1.2277 | [-0.4464, +2.9018] |
| exact_slot_learned − exact_slot_fifo | Brevo valid % | 7 | -7.5893 | [-9.7098, -5.3571] |
| exact_slot_gate − exact_slot_fifo | Content CE | 7 | -0.0425 | [-0.0616, -0.0265] |
| exact_slot_gate − exact_slot_fifo | Depo CE | 7 | -0.0325 | [-0.0761, -0.0078] |
| exact_slot_gate − exact_slot_fifo | Brevo CE | 7 | -0.0524 | [-0.0755, -0.0324] |
| exact_slot_gate − exact_slot_fifo | Depo generation % | 7 | -0.1116 | [-1.6741, +1.6741] |
| exact_slot_gate − exact_slot_fifo | Brevo valid % | 7 | +1.7857 | [-0.7812, +4.5759] |
| exact_slot_learned − exact_slot_gate | Content CE | 7 | +0.0689 | [+0.0328, +0.1001] |
| exact_slot_learned − exact_slot_gate | Depo CE | 7 | -0.1979 | [-0.2974, -0.1228] |
| exact_slot_learned − exact_slot_gate | Brevo CE | 7 | +0.3357 | [+0.2648, +0.4163] |
| exact_slot_learned − exact_slot_gate | Depo generation % | 7 | +1.3393 | [-0.4464, +3.4598] |
| exact_slot_learned − exact_slot_gate | Brevo valid % | 7 | -9.3750 | [-13.2812, -6.0268] |
| exact_slot_fifo − exact_scalar_fifo | Content CE | 7 | -0.2366 | [-0.2629, -0.2102] |
| exact_slot_fifo − exact_scalar_fifo | Depo CE | 7 | +0.0323 | [+0.0096, +0.0541] |
| exact_slot_fifo − exact_scalar_fifo | Brevo CE | 7 | -0.5055 | [-0.5595, -0.4576] |
| exact_slot_fifo − exact_scalar_fifo | Depo generation % | 7 | +0.7812 | [-0.5580, +2.7902] |
| exact_slot_fifo − exact_scalar_fifo | Brevo valid % | 7 | +22.4330 | [+19.9777, +24.5536] |
| exact_scalar_fifo − block_c64 | Content CE | 7 | +0.0670 | [+0.0360, +0.1101] |
| exact_scalar_fifo − block_c64 | Depo CE | 7 | +0.1024 | [+0.0480, +0.1839] |
| exact_scalar_fifo − block_c64 | Brevo CE | 7 | +0.0316 | [+0.0006, +0.0668] |
| exact_scalar_fifo − block_c64 | Depo generation % | 7 | -2.1205 | [-3.3482, -1.2277] |
| exact_scalar_fifo − block_c64 | Brevo valid % | 7 | -1.3393 | [-3.2366, +0.1116] |
| exact_slot_fifo − block_slot_c64 | Content CE | 7 | +0.1523 | [+0.1294, +0.1838] |
| exact_slot_fifo − block_slot_c64 | Depo CE | 7 | +0.1737 | [+0.1113, +0.2543] |
| exact_slot_fifo − block_slot_c64 | Brevo CE | 7 | +0.1309 | [+0.1034, +0.1627] |
| exact_slot_fifo − block_slot_c64 | Depo generation % | 7 | -1.0045 | [-2.1205, +0.2232] |
| exact_slot_fifo − block_slot_c64 | Brevo valid % | 7 | -2.7902 | [-4.6875, -1.1161] |
| exact_slot_gate − block_slot_gate_c64 | Content CE | 7 | +0.1496 | [+0.1059, +0.1936] |
| exact_slot_gate − block_slot_gate_c64 | Depo CE | 7 | +0.1720 | [+0.0772, +0.2720] |
| exact_slot_gate − block_slot_gate_c64 | Brevo CE | 7 | +0.1272 | [+0.1089, +0.1477] |
| exact_slot_gate − block_slot_gate_c64 | Depo generation % | 7 | -2.6786 | [-4.7991, -0.4464] |
| exact_slot_gate − block_slot_gate_c64 | Brevo valid % | 7 | -4.5759 | [-5.9152, -3.2366] |
| exact_slot_learned − block_slot_learn_c64 | Content CE | 7 | +0.2637 | [+0.2237, +0.2959] |
| exact_slot_learned − block_slot_learn_c64 | Depo CE | 7 | +0.1146 | [-0.0050, +0.2267] |
| exact_slot_learned − block_slot_learn_c64 | Brevo CE | 7 | +0.4128 | [+0.3420, +0.4965] |
| exact_slot_learned − block_slot_learn_c64 | Depo generation % | 7 | -1.6741 | [-2.7902, -0.5580] |
| exact_slot_learned − block_slot_learn_c64 | Brevo valid % | 7 | -11.1607 | [-14.5089, -8.8170] |
| exact_slot_fifo − gsa_slot | Content CE | 7 | -0.0030 | [-0.0084, +0.0043] |
| exact_slot_fifo − gsa_slot | Depo CE | 7 | +0.0047 | [-0.0064, +0.0141] |
| exact_slot_fifo − gsa_slot | Brevo CE | 7 | -0.0106 | [-0.0215, -0.0003] |
| exact_slot_fifo − gsa_slot | Depo generation % | 7 | +0.0000 | [-1.4509, +1.5625] |
| exact_slot_fifo − gsa_slot | Brevo valid % | 7 | -0.2232 | [-2.2321, +1.8973] |
| exact_slot_learned − gdn | Content CE | 7 | -0.1055 | [-0.1731, -0.0300] |
| exact_slot_learned − gdn | Depo CE | 7 | -0.0192 | [-0.1675, +0.1426] |
| exact_slot_learned − gdn | Brevo CE | 7 | -0.1917 | [-0.2451, -0.1523] |
| exact_slot_learned − gdn | Depo generation % | 7 | +2.0089 | [+0.5580, +3.3482] |
| exact_slot_learned − gdn | Brevo valid % | 7 | +14.9554 | [+12.9464, +16.9643] |
| exact_slot_learned − gsa | Content CE | 7 | -0.4459 | [-0.4845, -0.4046] |
| exact_slot_learned − gsa | Depo CE | 7 | -0.3069 | [-0.3964, -0.2269] |
| exact_slot_learned − gsa | Brevo CE | 7 | -0.5848 | [-0.6830, -0.4736] |
| exact_slot_learned − gsa | Depo generation % | 7 | +2.4554 | [+1.5625, +3.4598] |
| exact_slot_learned − gsa | Brevo valid % | 7 | +23.1027 | [+19.7545, +26.3393] |
| exact_slot_learned − gsa2_native_lr1e3 | Content CE | 7 | +0.2885 | [+0.2031, +0.3673] |
| exact_slot_learned − gsa2_native_lr1e3 | Depo CE | 7 | +0.7121 | [+0.4698, +0.8912] |
| exact_slot_learned − gsa2_native_lr1e3 | Brevo CE | 7 | -0.1350 | [-0.2385, -0.0174] |
| exact_slot_learned − gsa2_native_lr1e3 | Depo generation % | 7 | -6.5848 | [-9.5982, -4.0179] |
| exact_slot_learned − gsa2_native_lr1e3 | Brevo valid % | 7 | +5.8036 | [+1.8973, +9.0402] |

## 验收与产物

新实验及复现对照已独立核验 7424 条生成记录：gold、exact、合法性和汇总率一致。数据指纹、共享初始化、源码指纹、训练token数与checkpoint重载误差均检查。

四组完整模型预检通过：33-token模型输出对FP64逐token参考的最大RMS相对偏差为 0.926%，低于预先设定的1%门槛；训练后40次跨边界因果干预最大绝对误差 0。各组16步小样本训练loss下降、实际batch64反向通过，gate/learned的score梯度非零。

GDN seed731 复现对照差值：`{"Content CE": 0.003229753352839815, "Depo CE": -0.004540896386599602, "Brevo CE": 0.011000403092279454, "Depo generation %": -2.34375, "Brevo valid %": -0.78125, "Elapsed seconds": 71.89152783085592}`。

预检细节见 preflight.json；每次训练的完整曲线、summary、generation位于results/。checkpoint保存在远端独立实验目录。统计可用 report.py 重建；实验设计见 PLAN.md。

![Validation curves](curves.png)

本轮单卡GPU作业已完成并按精确alias清理，查询确认不存在；远端结果和checkpoint保留。
