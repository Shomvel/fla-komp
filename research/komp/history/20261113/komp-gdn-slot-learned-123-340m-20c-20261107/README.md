# GDN slot + learned Block + 123 — 340M / 6.8B

用户授权：独立新增一组 Block，8 H100，验收通过后 fresh seed42 正式训练 6.8B 有效监督 token。项目 WeaverRSI / proj-32；不修改现有训练。

## 模型

D1024 / L24 / H16 / K64 / M32 / C128 / FF2272。精确参数 341,159,168，比原 KOMP 339,966,848 多 0.3507%；比原 learned FF1824 多 0.2311%。

- 物理 slot 衰减：`g = -exp(A_log) * softplus(W_up W_down x + dt_bias)`。
- rank64 输出门和遗忘投影，无中间非线性，无投影 bias；输出门使用 sigmoid。
- 遗忘 down 随机、up 为零；首步 down 数据梯度为零但 AdamW WD 会更新，第二步应有非零数据梯度。
- A_log 初始为零；dt_bias 匹配 head 间 1/4..4 倍因子、按均值归一的衰减率，平均 log alpha = log(.995)，每个 head 内 32 slot 同初值。A_log/dt_bias 均 WD0。
- learned score 为 full Linear(D,H)，零初始；beta = sigmoid(beta_projection) * 2 sigmoid(score)，初始 beta=.2。
- 每 chunk 从旧槽和新 token 选 top-M，硬排序不反传，同分取新；score 仅通过写入强度获得梯度。初始 score 为固定零 buffer。
- 保留 KOMP 残差写入与原 Triton Block 算子。
- 同 FF2272 legacy seed42 共享骨干；named 低秩门种子与原组合 123 一致。FF1824 learned 对照的 FFN 形状不同，不能声称整模型初值相同。
- native FLA layer/block loop、独立 MLP projection、fused linear CE；兼容 BF16 SiLU 两次舍入；FP32 residual/state/projection，BF16网络和Block tape，仅 mixer checkpoint。

## 数据与预算

沿用冻结 FineWeb-Edu 253 shards / flame32K tokenizer / heldout hash 排除 / 顺序 / seed42。每卡 mb16、accum2、global256、seq2048。AdamW lr3e-4 betas(.8,.95) WD.1 eps1e-15，warmup1024，cosine 至 .1。每256步完整 checkpoint，每512步 val，最终 test；保留最近两个完整 checkpoint。6.8B 监督 token 精确停止，尾部 padding 不计预算；预检不计预算，正式训练重建 fresh 模型、Adam、RNG 和原始 loader cursor。

## 验收

新源码独立冻结，不能复用旧 PASS。8 开关组合的门性质和 FP64 梯度；random/real 算子 outputs/state/7 operands 对 FP64；learned retain-early/retain-initial/ties-newest；实际8rank full/tail 更新有限且 replicas 一致；24层门学习和 WD；初始/更新后 prefix1024/1031 因果绝对差 <1e-5；完整 model/Adam/RNG/loader 恢复。门限见 configs/native-acceptance.json，失败先保留证据并静态排查，不放宽门限重试。

所有初始化、计算和梯度检查结果在 evidence。硬排序和 indexed 原子梯度不承诺 bitwise。长期质量按同有效 token 验证比较；组合机制无法归因于单个开关。

## 运行和跟踪

`job-block.yaml` 依次执行 preflight 与 train；只有新源预检通过才训练。日志 `logs/train-block.log`；正式指标 `runs/block/metrics.jsonl` 和 `validation.jsonl`。本机 `python3 reports/collect.py` 只读采集。加入现有每小时跟踪，最终训练、val/test、来源/预算/恢复审计与精确 GPU 清理完成后结束。
