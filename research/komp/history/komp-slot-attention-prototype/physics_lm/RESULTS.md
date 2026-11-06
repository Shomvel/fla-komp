# Depo/Brevo 匹配小模型结果

已完成 6/6 次有效训练；每个run完整遍历同一10M-token数据。GSA使用修复slot初始化后的r3，GDN/KOMP使用r2；失败对照不计入此表。

| 模型 | seed | 参数量 | confirm内容CE | Depo内容准确率 | Brevo内容准确率 | Depo生成 | Brevo合法生成 | 秒 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gdn | 731 | 8,916,368 | 2.3351 | 13.10% | 42.84% | 4.69% | 14.84% | 71.5 |
| gdn | 732 | 8,916,368 | 2.5023 | 12.74% | 29.28% | 2.34% | 7.81% | 76.0 |
| gsa | 731 | 8,947,328 | 2.7093 | 11.97% | 25.28% | 0.78% | 3.12% | 78.1 |
| gsa | 732 | 8,947,328 | 2.7167 | 12.52% | 22.69% | 3.91% | 1.56% | 78.2 |
| komp | 731 | 8,932,752 | 2.4813 | 12.87% | 39.33% | 0.78% | 10.16% | 66.0 |
| komp | 732 | 8,932,752 | 2.5858 | 13.45% | 29.24% | 3.91% | 6.25% | 66.7 |

## 两个 seed 的平均值

| 模型 | 内容CE mean±SD | Depo生成均值 | Brevo合法生成均值 |
|---|---:|---:|---:|
| gdn | 2.4187 ± 0.1182 | 3.52% | 11.33% |
| gsa | 2.7130 ± 0.0052 | 2.34% | 2.34% |
| komp | 2.5335 ± 0.0739 | 2.34% | 8.20% |

## 完整模型训练吞吐

| 模型 | seed | 常态 step 毫秒 | 非padding token/s | 峰值 allocated GiB | 回读CE误差 |
|---|---:|---:|---:|---:|---:|
| gdn | 731 | 28.76 | 155,881 | 0.97 | 0 |
| gdn | 732 | 30.71 | 145,624 | 0.97 | 0 |
| gsa | 731 | 31.57 | 141,857 | 1.08 | 0 |
| gsa | 732 | 31.66 | 141,539 | 1.08 | 0 |
| komp | 731 | 26.77 | 167,593 | 1.52 | 0 |
| komp | 732 | 26.97 | 165,967 | 1.52 | 0 |

吞吐取连续日志区间的中位数，排除夹有验证的区间及首次编译；包含数据组batch、前反向和optimizer，未包含周期验证及生成。峰值allocated覆盖训练和评估，不是专门的训练显存microbenchmark。三个架构分别占一张H100，端到端吞吐不是同卡串行算子速度比较。

记录的共同模块初始化SHA-256逐seed一致。所有run固定同一LR配方、数据顺序、token预算；未按确认集选模型。两seed仅供探索，不足以证明稳定统计优势。

KOMP采用循环硬路由，并非完整最小删除代价KOMP。小模型、特定训练配方和合成任务结论不能直接推广到语言模型容量或大规模预训练。

旧LR研究的181M结果不参与此表；旧FLA快照有已记录的换底接口问题，本研究使用新的干净commit。

三种mixer嵌入统一骨干，GSA也采用L2-normalized q/k；GDN/KOMP均为sigmoid scalar gate，没有原生GDN的A_log/dt_bias；并非上游原生完整网络复现。先前GSA零门权重导致slot对称，原结果移至excluded-symmetric-gsa。详见PLAN.md的实施审计。

![Validation curves](validation-curves.png)

## 生成分桶

| 模型 | 任务 | 难度桶 | 正确/总数（两seed合并） |
|---|---|---|---:|
| gdn | brevo | 10 | 0/6 |
| gdn | brevo | 11 | 0/8 |
| gdn | brevo | 12 | 0/10 |
| gdn | brevo | 13 | 0/6 |
| gdn | brevo | 14 | 0/12 |
| gdn | brevo | 15 | 0/2 |
| gdn | brevo | 16 | 0/14 |
| gdn | brevo | 17 | 0/8 |
| gdn | brevo | 18 | 0/16 |
| gdn | brevo | 19 | 0/8 |
| gdn | brevo | 20 | 0/6 |
| gdn | brevo | 21 | 0/10 |
| gdn | brevo | 22 | 0/6 |
| gdn | brevo | 24 | 0/12 |
| gdn | brevo | 3 | 16/22 |
| gdn | brevo | 4 | 10/28 |
| gdn | brevo | 5 | 2/22 |
| gdn | brevo | 6 | 1/20 |
| gdn | brevo | 7 | 0/10 |
| gdn | brevo | 8 | 0/18 |
| gdn | brevo | 9 | 0/12 |
| gdn | depo | 1 | 1/56 |
| gdn | depo | 2 | 3/64 |
| gdn | depo | 3 | 3/68 |
| gdn | depo | 4 | 2/68 |
| gsa | brevo | 10 | 0/6 |
| gsa | brevo | 11 | 0/8 |
| gsa | brevo | 12 | 0/10 |
| gsa | brevo | 13 | 0/6 |
| gsa | brevo | 14 | 0/12 |
| gsa | brevo | 15 | 0/2 |
| gsa | brevo | 16 | 0/14 |
| gsa | brevo | 17 | 0/8 |
| gsa | brevo | 18 | 0/16 |
| gsa | brevo | 19 | 0/8 |
| gsa | brevo | 20 | 0/6 |
| gsa | brevo | 21 | 0/10 |
| gsa | brevo | 22 | 0/6 |
| gsa | brevo | 24 | 0/12 |
| gsa | brevo | 3 | 4/22 |
| gsa | brevo | 4 | 2/28 |
| gsa | brevo | 5 | 0/22 |
| gsa | brevo | 6 | 0/20 |
| gsa | brevo | 7 | 0/10 |
| gsa | brevo | 8 | 0/18 |
| gsa | brevo | 9 | 0/12 |
| gsa | depo | 1 | 1/56 |
| gsa | depo | 2 | 1/64 |
| gsa | depo | 3 | 3/68 |
| gsa | depo | 4 | 1/68 |
| komp | brevo | 10 | 0/6 |
| komp | brevo | 11 | 0/8 |
| komp | brevo | 12 | 0/10 |
| komp | brevo | 13 | 0/6 |
| komp | brevo | 14 | 0/12 |
| komp | brevo | 15 | 0/2 |
| komp | brevo | 16 | 0/14 |
| komp | brevo | 17 | 0/8 |
| komp | brevo | 18 | 0/16 |
| komp | brevo | 19 | 0/8 |
| komp | brevo | 20 | 0/6 |
| komp | brevo | 21 | 0/10 |
| komp | brevo | 22 | 0/6 |
| komp | brevo | 24 | 0/12 |
| komp | brevo | 3 | 11/22 |
| komp | brevo | 4 | 8/28 |
| komp | brevo | 5 | 2/22 |
| komp | brevo | 6 | 0/20 |
| komp | brevo | 7 | 0/10 |
| komp | brevo | 8 | 0/18 |
| komp | brevo | 9 | 0/12 |
| komp | depo | 1 | 0/56 |
| komp | depo | 2 | 2/64 |
| komp | depo | 3 | 1/68 |
| komp | depo | 4 | 3/68 |
