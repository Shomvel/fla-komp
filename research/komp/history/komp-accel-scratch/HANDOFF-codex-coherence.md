请跑 Block-KOMP 的 **coherence 准入（coherence criterion + QKLMS 式并入）** 质量实验，用 Depo/Brevo 7 seeds 回答：它是否优于不开启的 per-slot FIFO，最优阈值是多少，以及它与 gate、learned 淘汰叠加的效果。你只负责实验：kernel 已冻结，不改任何 `komp_slot/` 代码。

## 背景（先读）
- 项目：`~/Downloads/komp-slot-attention-prototype`（非 git）。先读 `~/Downloads/AGENTS.md`、`BLOCK_KOMP.md` 的 Round 2 一节（flex 路径与各 flag）、`BLOCK_KOMP_TRIALS.md`（你上一轮的 7-seed 结果、统计方法和 harness 约定）。
- 语义：参数 `coherence=θ`（余弦阈值，None 表示关闭）。
  - 新 key 与当前字典的最大余弦 ≤ θ 时，才成为新原子占新槽。当前字典包括旧原子和本块此前已准入的原子。
  - 否则这次写入并到最相似的已有原子上，原子的 key 和槽位都不变。
  - 只有已准入的原子参与块末淘汰。准入判断只看 key，不传梯度。
  - 只支持 fifo/learned 淘汰，不能与 projectron 同开。
  - 公式见 `block_flex.py` 文档串。
- 正确性：`tools/check_coherence.py` 在 CPU 上 FP64 对拍 72 组配置（flex 对比 token 级参考，覆盖输出、最终状态和全部梯度），最大误差 1e-12。关闭开关时与旧 `block_flex` 逐位一致。
- 动机：在已训练的 per-slot FIFO 模型上，用真实 key 探测并入比例（`tools/probe-merge-trained-slot-fifo.txt`）：
  - cos > 0.99：第 0 层 72%，第 1 层 34%，第 2、3 层约 16%。
  - cos > 0.95：约 50%。
  说明字典里有大量近重复原子，它们占槽却几乎不增加表达力，还让 Gram 近奇异。

## 冻结快照（必须用这一版）
- `~/Downloads/komp-accel-scratch/frozen-20261101-coh/`，带 `SHA256SUMS`：
  - 内容：r3 冻结 kernel + 新的 `komp_slot/block_flex.py`（sha 04e04936…）和 `block_reference.py`（sha c36873c0…）。
  - `block_fast.py` 与 r3 相同（a8e4c725…）。
  - `tools/` 下有我的参考 wrapper `block_coh.py`、`probe_merge.py`、`train_coh.sh`、`report_coh.py` 和对拍脚本。
- 主树的 `komp_slot/` 会被 Claude 继续修改。快照里的 `komp_slot` 一律从冻结目录拷贝。preflight 会哈希源码，必须在快照内先跑 preflight 再训练。
- 全部臂（包括对照组）一律走 FP32 PyTorch 的 `block_flex`，不走 Triton 的 `block_fast`，保证同一实现下配对比较。
  - 参考做法：wrapper 把 `block_experiment.block_komp_fast` 替换为带 `coherence` 的 `block_komp_flex`（见 `tools/block_coh.py`）。
  - 要用主树当前带 gate 臂的 `physics_lm/block_experiment.py`；我的 `tools/block_coh.py` 基于旧的 snap-v3 版本，需要改造。
- 每个 run 记录各层并入比例（训练前 10% 和后 10%），只统计非 padding token：
  - padding 的 token id 为 0，`stats['valid']` 用法见 `block_coh.py`。
  - 写入 `<kind>-seed<seed>-coherence.json`。
- 注意：flex 路径每个 run 约 250–290 s，比 Triton 慢约 4 倍。

## 实验
**E1 阈值扫描。** 4 个臂，seeds 731–737，都是 per-slot 衰减、C=64、FIFO：
- 不开 coherence
- θ = 0.99
- θ = 0.95
- θ = 0.80

**E2 叠加。** 先在 E1 的 7 seeds 上按 content CE 均值最低（预先规定）选出 θ*，再跑以下 4 个臂，seeds 731–737，全部走 flex：
- gate-only
- gate-only + θ*
- learned
- learned + θ*

**E3 诊断。** 每臂 seed 731 的 checkpoint，在 confirm 前 64 条样本上统计各层：
- 并入比例；
- 块末保留字典的 Gram（κ(K,K) + ridge 1e-4）条件数，给中位数和最大值，并以不开 coherence 为对照；
- 互余弦 > 0.95 的原子对数量；
- 系数范数。

E3 用来检验“coherence 改善 Gram 条件数”这一假设，这关系到投影侧能否降到 TF32。只做测量，不改 kernel。

## GPU
- 用 `~/.codex/skills/northjob-gpu-test-bp-dev` skill，SSH 主机 `linlizhi-bp-dev`，项目 WeaverRSI，单卡 H100。
- 模板：`~/Downloads/komp-accel-scratch/job/coh-job.yaml` 和 `runner-coh.sh`。镜像、环境变量、tolerations 照抄。
- 换成你自己的目录和 alias：`/gpfs/users/linlizhi/komp-coh-trials-20261101`，alias 用 `komp-coh-trials-20261101-r1`；如果拆成两个 job 并行或串行，第二个用 `-r2`。单个 job 的 deadline 是 4 小时，E1 和 E2 各约 2.1 小时。
- **不要碰** Claude 正在用的 job `komp-block-coh-20261101` 和它的目录 `/gpfs/users/linlizhi/komp-block-coh-20261101`。那是 3-seed 的预实验，用的是旧版 block_experiment，不与你的结果合并。
- 数据：`PHYSICS_DATA`，只读。

## 约束
- `physics_lm/` 里另一个 agent 的文件（lrwd/gsa2/shortconv/decay/experiment 等）不要改，也不要写进它们已有的 runs 目录。
  - 你可以改或新增 `block_experiment.py`、`block_report*.py`，以及新的 wrapper，例如 `physics_lm/block_coherence.py`。
  - 结果拷回 `physics_lm/block_runs_coh/`，不含 `*.pt`。
- 统计方法沿用 `BLOCK_KOMP_TRIALS.md`：
  - 每臂给 n、均值和样本标准差；
  - 按 seed 配对求差，20,000 次 bootstrap 求 95% CI；
  - 关键对比：各 θ 对比不开 coherence、gate+θ* 对比 gate、learned+θ* 对比 learned。
  - 表格必须由脚本从 `summary.json` 生成，禁止手填。
- 结论写进新文件 `BLOCK_KOMP_COHERENCE.md`，表格原样粘贴脚本输出。不改 `BLOCK_KOMP.md` 和 `README.md`，由 Claude 合并。
- 远端探测用独立脚本经 stdin 传入，不用 `python -c`。不打印、不拷贝、不持久化任何 token、密码或密钥。
- 某类错误首次出现时，暂停重复运行；先静态排查同类问题，批量修复，再恢复运行。
- 结束时：记录 job 终态和证据（json），按精确 alias 删除自建 job，并重新查询确认已消失。

## 交付
- `physics_lm/block_runs_coh/`：日志、summary、并入统计和 TABLE.md。
- E3 诊断的 json 与表格。
- `BLOCK_KOMP_COHERENCE.md`：方法、生成的表、结论和局限。
- 一段中文总结，回答四个问题：
  - coherence 是否在 7 seeds 下站得住，在哪个任务（Depo/Brevo）上；
  - 最优阈值；
  - 与 gate、learned 是否可叠加；
  - 是否改善了 Gram 条件数。
