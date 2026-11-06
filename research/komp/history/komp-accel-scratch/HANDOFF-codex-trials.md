你来跑 Block-KOMP 的 Depo/Brevo 质量实验。目标是把目前 2 个 seed 的探索性结论做实，并拆清 learned 淘汰的收益来源。kernel 由另一个 agent（Claude）继续加速；你只跑实验，不改 kernel。

## 背景（先读）
- 项目：`~/Downloads/komp-slot-attention-prototype`，不是 git 仓库。先读 `AGENTS.md`（`~/Downloads/AGENTS.md`）、`BLOCK_KOMP.md` 的 Round 2/3 两节、`physics_lm/block_experiment.py`、`physics_lm/block_report.py`。
- 现有结论（2 seeds，`physics_lm/block_runs_v2/TABLE.md`）：
  - per-slot 衰减 + FIFO：content CE 2.155，Brevo 37.1%。
  - per-slot + learned 淘汰：CE 2.070，Brevo 37.5%，Depo gen 7.4%。
  - GDN：2.419 / 11.3%。逐 token gsa_slot：2.296 / 30.9%。
  - learned 的 Triton 版（`block_runs_v3`）：CE 2.0745，Brevo 36.3%。
- learned 臂同时做了两件事：按 score 淘汰，以及 `beta *= 2*sigmoid(score)` 写入门控。目前不知道收益来自哪一项。

## 冻结的 kernel（必须用这一版）
- `~/Downloads/komp-accel-scratch/frozen-20261101-r3/`：`komp_slot/`、`komp_bench/`、`benchmark_fla.py`，带 `SHA256SUMS`。其中 `block_fast.py` 的 sha256 为 `a8e4c725…74ce`，已通过 573 项 FP64 验收。
- 主树里的 `komp_slot/*.py` 会被 Claude 持续修改。快照里的 `komp_slot` 一律从冻结目录拷贝，不要从主树拷。
- 各 harness 的 preflight ticket 会哈希 `komp_slot/*.py`，所以 preflight 要在快照里跑，并且必须先于训练。

## GPU 方式
- 用 `~/.codex/skills/northjob-gpu-test-bp-dev` skill，SSH 主机 `linlizhi-bp-dev`，项目 WeaverRSI，单卡 H100。
- 可参考的模板：`~/Downloads/komp-accel-scratch/job/dev-job2.yaml` 和 `runner2.sh`（单卡 runner 轮询 `queue/*.sh`），以及 `job/train_block.sh`（快照加训练）。
  - 镜像、环境变量（`PYTHONPATH` 指向 triton371 overlay、`FLA_SOURCE` 等）、tolerations 都照抄。
  - task 目录和 alias 换成你自己的：目录 `/gpfs/users/linlizhi/komp-block-trials-20261101`，alias `komp-block-trials-20261101-r1`。不要复用 `komp-accel-*` 目录。
- 数据在 `PHYSICS_DATA`，默认 `/gpfs/users/linlizhi/softmax-10m-20261023/data`，只读。
- 每个 run 的 train+eval 约 65–80 s。

## 实验
E1 seed 扩充：新增 seeds 733–737，加上已有的 731/732，每臂共 7 个 seed。
- `block_experiment.py`：`block_slot_c64`、`block_slot_learn_c64`（Triton，默认 `BLOCK_TRITON_LEARNED=1`）、`block_c64`。
- 基线（各自 harness，在快照里跑）：
  - `experiment.py --kind gdn|gsa`
  - `decay_experiment.py --kind gsa_slot`
- 已有的 731/732 结果：block 臂在 `physics_lm/block_runs_v2/results`（flex learned）和 `block_runs_v3/results`（Triton learned、fused per-slot）。
- 冻结 kernel 与 v3 训练时的 kernel 在舍入层面有差异（之后又加了旋转槽列、keygrad 合并等改动）。为了让所有 seed 用同一实现，block 臂的 731/732 也用冻结 kernel 重跑。

E2 拆解 learned。在 `block_experiment.py` 的 VARIANTS 里加两个臂，seeds 731–737：
- `block_slot_gate_c64`：FIFO 淘汰，但保留 `beta *= 2*sigmoid(score)` 门控（score 头与 learned 臂相同）。用来分离门控和淘汰各自的作用。
- `block_slot_learn_c32`：learned，chunk 32。
- 新臂先过 preflight（因果性、loss 下降）再训练。

E3 可选，只在 E1/E2 完成且 job 时间够时做：`block_slot_learn_c64` 用更长序列或更大 M。前提是 harness 不改共享常量就能做到；做不到就跳过并写明原因。

## 约束
- `physics_lm/` 里另一个 agent 的文件（lrwd/gsa2/shortconv/decay/experiment 等）不要改，也不要写进它们已有的 runs 目录。
  - 你可以改 `block_experiment.py` 和 `block_report.py`（我方文件）。
  - 结果拷回 `physics_lm/block_runs_v4/`；基线放在 `block_runs_v4/baselines/`。
- 表格必须由脚本从 `summary.json` 生成，禁止手填数字。
  - 扩展 `block_report.py`（或新写 `block_report_v4.py`）：每臂给 n、均值、样本标准差。
  - 关键对比给 paired by seed 的差值和 bootstrap 95% CI：learned vs slot FIFO、gate vs FIFO、gate vs learned、block_slot vs gsa_slot、vs GDN。
- 结论写进新文件 `BLOCK_KOMP_TRIALS.md`，表格从脚本输出原样粘贴。不要改 `BLOCK_KOMP.md` 和 `README.md`，由 Claude 合并。
- 远端探测用独立脚本经 stdin 传入，不用 `python -c`。不要打印、拷贝或持久化任何 token、密码、密钥。
- 某类错误第一次出现时，暂停重复运行；先静态排查同类问题，批量修复，再恢复。不要靠反复重跑逐个发现问题。
- 结束时：记录 job 终态和证据（json），按精确 alias 删除本次自建 job，并重新查询确认已消失。只停止自己启动的进程。

## 交付
- `physics_lm/block_runs_v4/`：日志、summary、TABLE.md，不含 `*.pt`。
- `BLOCK_KOMP_TRIALS.md`：方法、生成的表、结论和局限。
- job 删除证据。
- 一段中文总结：哪些结论在 7 seeds 下站得住，learned 的收益主要来自门控还是淘汰。
