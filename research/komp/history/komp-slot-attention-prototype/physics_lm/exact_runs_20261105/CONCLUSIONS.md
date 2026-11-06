# 本轮结论

四组各7个seed、共28次训练全部完成，另复跑GDN seed731。保持原10M token数据和训练配方，未重调LR。

- **逐token exact压缩没有超过Block-KOMP。** 对应scalar/FIFO/gate/learned四组，Exact的Content CE均更差。逐slot FIFO相对Block C64的CE增加0.1523，配对95%区间[0.1294,0.1838]；Brevo合法率下降2.79个百分点，区间[-4.69,-1.12]。
- **新Exact slot FIFO与旧逐token gsa_slot效果接近。** CE 2.2440 vs 2.2469；Depo均4.13%；Brevo33.71% vs33.93%。三个差值的区间均跨零；这不证明数学等价，只说明本实验未检出收益。
- **逐slot衰减仍是主要收益。** 相对Exact scalar，slot FIFO的CE下降0.2366，Brevo上升22.43个百分点；两者区间均不跨零。
- **写入门改善CE，learned选择损害Brevo。** Gate-only相对FIFO的CE下降0.0425，95%区间[-0.0616,-0.0265]；Brevo上升1.79个百分点的区间跨零。Learned相对gate-only的CE增加0.0689，区间[0.0328,0.1001]，Brevo下降9.375个百分点，区间[-13.2812,-6.0268]。Learned对Depo生成的收益区间跨零，暂不声称稳定改善。
- **相对GDN/GSA仍有Brevo优势，但不是全面优于调参后的GSA2。** GSA2 LR1e-3的CE1.9819、Depo11.94%更好，Exact gate的Brevo35.49%更好；GSA2参数、状态、卷积与LR不同。

因此100M比较采用Block/Exact同slot FIFO、同任务既定LR/WD，直接检验压缩频率变化。不能从这些结果断言FP32精度导致差距，也不能断言所有数据或训练预算下逐token压缩都更差。

GDN复现CE偏移+0.00323，生成有少量离散变化，说明历史比较并非逐位可复现。所有区间只描述固定语料与固定128例/任务下的seed变动，未做多重比较校正。7424条本轮生成记录全部独立核验通过。完整均值、标准差、配对区间与曲线见RESULTS.md；checkpoint保留在远端。
