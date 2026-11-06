# KOMP decay 消融结果

完成 8/8 次预定训练。四组都使用原IEEE geometry、相同初始alpha=.98、独立beta=.2。
每组相同10M唯一输入token、4层hidden128、LR配方、公共初始化；decay bias/时间尺度参数无weight decay。

| variant | seed | confirm内容CE | Depo生成 | Brevo合法生成 | 常态step ms | 峰值GiB |
|---|---:|---:|---:|---:|---:|---:|
| gdn_scalar | 731 | 2.4756 | 3.12% | 10.94% | 27.23 | 1.52 |
| gdn_scalar | 732 | 2.5745 | 4.69% | 7.03% | 27.29 | 1.52 |
| gsa_scalar | 731 | 2.4804 | 1.56% | 10.16% | 27.18 | 1.52 |
| gsa_scalar | 732 | 2.6173 | 3.91% | 6.25% | 27.69 | 1.52 |
| gsa_slot | 731 | 2.2672 | 2.34% | 32.03% | 27.72 | 1.54 |
| gsa_slot | 732 | 2.3251 | 4.69% | 29.69% | 28.99 | 1.54 |
| sigmoid | 731 | 2.4764 | 1.56% | 10.94% | 27.30 | 1.52 |
| sigmoid | 732 | 2.5742 | 4.69% | 8.59% | 27.13 | 1.52 |

## 两seed均值

| variant | 内容CE mean±SD | Depo生成 | Brevo合法生成 |
|---|---:|---:|---:|
| sigmoid | 2.5253 ± 0.0691 | 3.12% | 9.77% |
| gdn_scalar | 2.5251 ± 0.0700 | 3.91% | 8.98% |
| gsa_scalar | 2.5489 ± 0.0968 | 2.73% | 8.20% |
| gsa_slot | 2.2961 ± 0.0410 | 3.52% | 30.86% |

## 分任务内容CE

| variant | Depo CE | Brevo CE |
|---|---:|---:|
| sigmoid | 3.1984 | 1.8522 |
| gdn_scalar | 3.1991 | 1.8510 |
| gsa_scalar | 3.2067 | 1.8911 |
| gsa_slot | 3.2375 | 1.3548 |

全部非decay参数的初始化逐seed hash一致；checkpoint回读验证通过。两seed是探索性证据，不足以证明统计显著性。
常态step排除夹有验证的日志区间，包含数据准备、前反向和optimizer。峰值allocated包括训练和评估。
gsa_scalar与gsa_slot用于分离门参数化和粒度；与旧GSA完整mixer结果不可混淆。
slot与scalar初始数学函数在FP32通过原3e-4门槛；BF16下初始隐藏状态RMS相对差异约0.171%，未声称逐元素完全相同。

![Validation curves](validation-curves.png)

## 生成分桶

| variant | 任务 | hop或节点数 | 正确/总数（两个seed） |
|---|---|---:|---:|
| sigmoid | brevo | 3 | 15/22 |
| sigmoid | brevo | 4 | 8/28 |
| sigmoid | brevo | 5 | 2/22 |
| sigmoid | brevo | 6 | 0/20 |
| sigmoid | brevo | 7 | 0/10 |
| sigmoid | brevo | 8 | 0/18 |
| sigmoid | brevo | 9 | 0/12 |
| sigmoid | brevo | 10 | 0/6 |
| sigmoid | brevo | 11 | 0/8 |
| sigmoid | brevo | 12 | 0/10 |
| sigmoid | brevo | 13 | 0/6 |
| sigmoid | brevo | 14 | 0/12 |
| sigmoid | brevo | 15 | 0/2 |
| sigmoid | brevo | 16 | 0/14 |
| sigmoid | brevo | 17 | 0/8 |
| sigmoid | brevo | 18 | 0/16 |
| sigmoid | brevo | 19 | 0/8 |
| sigmoid | brevo | 20 | 0/6 |
| sigmoid | brevo | 21 | 0/10 |
| sigmoid | brevo | 22 | 0/6 |
| sigmoid | brevo | 24 | 0/12 |
| sigmoid | depo | 1 | 1/56 |
| sigmoid | depo | 2 | 2/64 |
| sigmoid | depo | 3 | 1/68 |
| sigmoid | depo | 4 | 4/68 |
| gdn_scalar | brevo | 3 | 14/22 |
| gdn_scalar | brevo | 4 | 8/28 |
| gdn_scalar | brevo | 5 | 1/22 |
| gdn_scalar | brevo | 6 | 0/20 |
| gdn_scalar | brevo | 7 | 0/10 |
| gdn_scalar | brevo | 8 | 0/18 |
| gdn_scalar | brevo | 9 | 0/12 |
| gdn_scalar | brevo | 10 | 0/6 |
| gdn_scalar | brevo | 11 | 0/8 |
| gdn_scalar | brevo | 12 | 0/10 |
| gdn_scalar | brevo | 13 | 0/6 |
| gdn_scalar | brevo | 14 | 0/12 |
| gdn_scalar | brevo | 15 | 0/2 |
| gdn_scalar | brevo | 16 | 0/14 |
| gdn_scalar | brevo | 17 | 0/8 |
| gdn_scalar | brevo | 18 | 0/16 |
| gdn_scalar | brevo | 19 | 0/8 |
| gdn_scalar | brevo | 20 | 0/6 |
| gdn_scalar | brevo | 21 | 0/10 |
| gdn_scalar | brevo | 22 | 0/6 |
| gdn_scalar | brevo | 24 | 0/12 |
| gdn_scalar | depo | 1 | 2/56 |
| gdn_scalar | depo | 2 | 2/64 |
| gdn_scalar | depo | 3 | 2/68 |
| gdn_scalar | depo | 4 | 4/68 |
| gsa_scalar | brevo | 3 | 12/22 |
| gsa_scalar | brevo | 4 | 8/28 |
| gsa_scalar | brevo | 5 | 1/22 |
| gsa_scalar | brevo | 6 | 0/20 |
| gsa_scalar | brevo | 7 | 0/10 |
| gsa_scalar | brevo | 8 | 0/18 |
| gsa_scalar | brevo | 9 | 0/12 |
| gsa_scalar | brevo | 10 | 0/6 |
| gsa_scalar | brevo | 11 | 0/8 |
| gsa_scalar | brevo | 12 | 0/10 |
| gsa_scalar | brevo | 13 | 0/6 |
| gsa_scalar | brevo | 14 | 0/12 |
| gsa_scalar | brevo | 15 | 0/2 |
| gsa_scalar | brevo | 16 | 0/14 |
| gsa_scalar | brevo | 17 | 0/8 |
| gsa_scalar | brevo | 18 | 0/16 |
| gsa_scalar | brevo | 19 | 0/8 |
| gsa_scalar | brevo | 20 | 0/6 |
| gsa_scalar | brevo | 21 | 0/10 |
| gsa_scalar | brevo | 22 | 0/6 |
| gsa_scalar | brevo | 24 | 0/12 |
| gsa_scalar | depo | 1 | 2/56 |
| gsa_scalar | depo | 2 | 3/64 |
| gsa_scalar | depo | 3 | 1/68 |
| gsa_scalar | depo | 4 | 1/68 |
| gsa_slot | brevo | 3 | 22/22 |
| gsa_slot | brevo | 4 | 26/28 |
| gsa_slot | brevo | 5 | 15/22 |
| gsa_slot | brevo | 6 | 8/20 |
| gsa_slot | brevo | 7 | 2/10 |
| gsa_slot | brevo | 8 | 4/18 |
| gsa_slot | brevo | 9 | 1/12 |
| gsa_slot | brevo | 10 | 0/6 |
| gsa_slot | brevo | 11 | 1/8 |
| gsa_slot | brevo | 12 | 0/10 |
| gsa_slot | brevo | 13 | 0/6 |
| gsa_slot | brevo | 14 | 0/12 |
| gsa_slot | brevo | 15 | 0/2 |
| gsa_slot | brevo | 16 | 0/14 |
| gsa_slot | brevo | 17 | 0/8 |
| gsa_slot | brevo | 18 | 0/16 |
| gsa_slot | brevo | 19 | 0/8 |
| gsa_slot | brevo | 20 | 0/6 |
| gsa_slot | brevo | 21 | 0/10 |
| gsa_slot | brevo | 22 | 0/6 |
| gsa_slot | brevo | 24 | 0/12 |
| gsa_slot | depo | 1 | 2/56 |
| gsa_slot | depo | 2 | 1/64 |
| gsa_slot | depo | 3 | 2/68 |
| gsa_slot | depo | 4 | 4/68 |

## 最后一步门诊断

取最后一个验证minibatch，含padding；不是全验证集分布。

| variant | seed | 参数量 | 各层平均alpha | 各层slot标准差 |
|---|---:|---:|---|---|
| gdn_scalar | 731 | 8,932,760 | 0.997, 0.922, 0.995, 0.986 | — |
| gdn_scalar | 732 | 8,932,760 | 0.996, 0.978, 0.997, 0.994 | — |
| gsa_scalar | 731 | 8,932,752 | 0.997, 0.947, 0.995, 0.987 | — |
| gsa_scalar | 732 | 8,932,752 | 0.996, 0.990, 0.994, 0.991 | — |
| gsa_slot | 731 | 8,964,744 | 0.983, 0.955, 0.983, 0.980 | 0.0074, 0.0249, 0.0113, 0.0096 |
| gsa_slot | 732 | 8,964,744 | 0.977, 0.977, 0.986, 0.987 | 0.0139, 0.0098, 0.0053, 0.0036 |
| sigmoid | 731 | 8,932,752 | 0.996, 0.925, 0.994, 0.986 | — |
| sigmoid | 732 | 8,932,752 | 0.995, 0.978, 0.996, 0.992 | — |
