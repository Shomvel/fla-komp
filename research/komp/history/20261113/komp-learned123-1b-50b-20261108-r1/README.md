# Learned+123 1B / 50B

用户授权：BP Dev 单个 32 GPU 训练，参考 Flame 之前配置。新目录、全新 seed42 初始化，既有 340M 评测不改动。

模型：D2048 / L24 / H32 / K64 / M32 / FF3072 / chunk128，Learned eviction + GDN slot decay + options 123（sigmoid output、rank64 gates、多时间尺度初始化）。参数预计 1,003,181,056，实际实例化由 preflight 核对。

Flame 来源：exp/gdn_1B_edu_100B.sh、configs/transformer_1B.json 与 flame/models/fla.toml。沿用 seq2048、global512、AdamW lr3e-4 betas(.8,.95) eps1e-15 WD.1、warmup4000、cosine至.1、seed42。KOMP 几何不同于 GDN，因此不直接复制其 head 配置。FF3072 使本结构接近 1B。每卡 mb4 × accum4 ×32 保持 global512。沿用此前 KOMP 的 frozen FineWeb-Edu/32K tokenizer 和 heldout exclusions，不改用另一 tokenizer。

预算：50,000,000,000 supervised tokens，47,683 个完整 global step 加一个 750,592 token tail，总47,684步。预验收2步不计入正式预算；正式重置 seed42 和输入游标。最后尾部 label mask，额外计算 token 单独记录。checkpoint/validation 每1024步，保留两个完整 checkpoint。非有限数立即失败；不启用 Flame skip_nan_inf 或 torch.compile。

4 个节点各8 H100，DDP/IB，WeaverRSI/proj-32。全32rank验收包括真实 full/tail batch、FP64算子/option检查、前后因果性、梯度有限性、参数更新、跨rank一致性以及完整模型/Adam/RNG/输入恢复；全部通过才 fresh 正式训练。预验收显存上限72GiB。无自动重启。

数据 audit 为253 shards的大小/mtime/首64KiB指纹及tokenizer/heldout全量哈希，并非训练语料每个字节的全量哈希。证据在 evidence/，运行日志 nodes/，正式指标 runs/block/metrics.jsonl。

修订 r1：移除 save 内重复局部 import shutil，修复新增空间检查的 UnboundLocalError。旧版本与32份失败回执保留。已执行实际 save 函数的32rank文件/元数据/保留逻辑回归并复现旧错误，所有顶层任务Python的局部导入作用域已静态排查。GPU全链路验收将重新执行。
