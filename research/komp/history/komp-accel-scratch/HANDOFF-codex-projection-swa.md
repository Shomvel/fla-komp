请跑一组 **Block-KOMP 投影消融 + softmax sliding-window attention（SWA）基线** 的 Depo/Brevo 7-seed 实验，回答两个问题：

1. Block-KOMP 块末的联合投影（被淘汰原子的系数投影到留下的原子上）到底值多少？
2. 在同等状态预算下，Block-KOMP 和普通的 softmax SWA 相比如何？

你只负责实验：kernel 已冻结，不改任何 `komp_slot/` 代码。

## 背景（先读）
- 项目：`~/Downloads/komp-slot-attention-prototype`（非 git）。先读：
  - `~/Downloads/AGENTS.md`
  - `BLOCK_KOMP.md` 开头的算法说明和 Round 2 一节（flex 路径与各 flag）
  - `BLOCK_KOMP_TRIALS.md`：你上一轮的 7-seed 方法、统计方法和 harness 约定，沿用它们
- 动机：FIFO 的 Block-KOMP 在“用哪些 key 读出”上就是一个滑动窗口——字典里是最近 M 到 M+C 个 token 的 key。区别在于，被淘汰的 key 不是直接丢掉，而是把它的系数用 S_kept += P·S_old 投影到留下的原子上，其中 P = (G_kept+λI)⁻¹ κ(K_kept, K_old)。
  - 投影是 kernel 里最贵的部分：8k 时约 3 ms，GDN 整个前反向是 4.4 ms。
  - 如果去掉投影质量基本不变，就可以省掉这部分；如果明显变差，说明投影保住了窗口外的信息。
- 已有结论：C=128 比 C=64 好（Claude 的 3-seed 预实验，CE −0.156）；coherence 并入在 θ ≤ 0.95 时让 Brevo 下降约 25 pp，属于负收益。

## 冻结快照（必须用这一版）
- `~/Downloads/komp-accel-scratch/frozen-20261101-proj/`，带 `SHA256SUMS`：
  - 内容：r3 冻结 kernel，加上新的 `komp_slot/block_flex.py`（sha 0d1700ec…）和 `block_reference.py`（sha e195d47a…）。
  - 新增参数 `project`（默认 True）。`project=False` 时，被淘汰原子连同系数直接丢弃，相当于在留下的 key 上做 delta rule 的滑动窗口；块内计算完全不变。同一文件里也包含 coherence/dedupe 开关，本实验不用。
  - `block_fast.py` 与 r3 相同（a8e4c725…）。
  - `tools/check_project.py`：CPU 上的 FP64 对拍，flex 与 token 级参考对比 64 组配置，最大误差 3.5e-12；`project=True` 时与旧版逐位一致。
  - `tools/block_coh.py`：一个把 `block_experiment.block_komp_fast` 替换成 `block_komp_flex` 的 wrapper，可作模板。
- 主树的 `komp_slot/` 会被 Claude 继续修改，快照里的 `komp_slot` 一律从冻结目录拷贝。preflight 会哈希源码，必须在快照内先跑 preflight 再训练。
- 所有 Block-KOMP 臂（包括投影开启的对照组）一律走 FP32 PyTorch 的 `block_flex`，保证同一实现下配对比较。每个 run 约 250–290 s。
- 用主树当前的 `physics_lm/block_experiment.py`，其中有 gate/learned 臂。

## 实验
**E1 投影消融**（per-slot 衰减，M=32，seeds 731–737，全部走 flex）：

| 臂 | C | 淘汰规则 | project |
|---|---:|---|---|
| slot FIFO | 64 | FIFO | True（对照组） |
| slot FIFO, no projection | 64 | FIFO | False |
| slot learned | 64 | learned | True |
| slot learned, no projection | 64 | learned | False |

learned 臂按现有 harness 的做法，score 头同时作为写入门控 `beta *= 2σ(score)`。

**E2 softmax SWA 基线**（seeds 731–737）。在同一 backbone 上用 softmax 注意力替换 mixer：
- 复用 KOMP mixer 的 q/k/v/out/gate/norm 模块和初始化，保持共享参数初始指纹一致；decay/beta/keys 不使用。
- 打分为 softmax(q·kᵀ/√K)，因果加窗口掩码；用 BF16 autocast 下的 `F.scaled_dot_product_attention` 即可。
- backbone 本身没有位置编码。默认加 RoPE，另加一个不加位置编码（NoPE）的臂做对照。

| 臂 | 窗口 W | 位置编码 | 说明 |
|---|---:|---|---|
| swa_w32 | 32 | RoPE | KV cache = W·(Dk+Dv)，与 Block-KOMP 的状态 M·(Dk+Dv) 相等 |
| swa_w96 | 96 | RoPE | 等于 M+C，即 Block-KOMP 块内最多可见的原子数 |
| swa_w96_nope | 96 | 无 | 检验 RoPE 的作用 |
| full_attn | 192 | RoPE | 全因果注意力，作为质量上限参考 |

每个 SWA 臂先过 preflight：干预未来 token 不影响过去输出、窗口外的 token 不影响当前输出、loss 下降。

**E3（可选，前两项完成且 job 时间够时做）**：C=128 的 slot FIFO，project 开与关各一个臂，看窗口更长时投影的价值。

## GPU
- 用 `~/.codex/skills/northjob-gpu-test-bp-dev` skill，SSH 主机 `linlizhi-bp-dev`，项目 WeaverRSI，单卡 H100。模板：`~/Downloads/komp-accel-scratch/job/coh-job.yaml` 和 `runner-coh.sh`，镜像、环境变量、tolerations 照抄。
- 换成你自己的目录和 alias：目录 `/gpfs/users/linlizhi/komp-proj-swa-20261101`，alias `komp-proj-swa-20261101-r1`。可以拆成两个 job，第二个用 `-r2`。E1 约 2 小时，E2 约 30–40 分钟；单个 job 的 deadline 是 4 小时。
- **不要碰** Claude 的 job `komp-block-coh-20261101` 和它的目录，以及其他 `komp-accel-*` 目录。
- 数据：`PHYSICS_DATA`，只读。

## 约束
- `physics_lm/` 里另一个 agent 的文件（lrwd/gsa2/shortconv/decay/experiment 等）不要改，也不要写进它们已有的 runs 目录。你可以新增 wrapper，例如 `physics_lm/block_ablation.py`，以及报告脚本。结果拷回 `physics_lm/block_runs_proj/`，不含 `*.pt`。
- 统计沿用 `BLOCK_KOMP_TRIALS.md`：每臂给 n、均值和样本标准差；按 seed 配对求差，20,000 次 bootstrap 求 95% CI。
  - 关键对比：
    - no-projection 对比 projection（FIFO、learned 各一组）
    - slot FIFO 对比 swa_w32、swa_w96
    - swa_w96 对比 swa_w96_nope
    - full_attn 作为上限参考
  - 表格必须由脚本从 `summary.json` 生成，禁止手填。
- 同一份报告里列出各臂每个 run 的训练时间。注意 flex 与 SWA 的实现不同，这个时间不能当作 kernel 速度比较。
- 结论写进新文件 `BLOCK_KOMP_PROJECTION.md`，表格原样粘贴脚本输出。不改 `BLOCK_KOMP.md` 和 `README.md`，由 Claude 合并。
- 远端探测用独立脚本经 stdin 传入，不用 `python -c`。不打印、不拷贝、不持久化任何 token、密码或密钥。
- 某类错误首次出现时，暂停重复运行；先静态排查同类问题，批量修复，再恢复运行。
- 结束时：记录 job 终态和证据（json），按精确 alias 删除自建 job，并重新查询确认已消失。
- 如果你还在跑之前那份 coherence handoff：coherence 已确认为负收益，可以停掉。所有自建 job 都要按 alias 删除并确认。

## 交付
- `physics_lm/block_runs_proj/`：日志、summary、TABLE.md。
- `BLOCK_KOMP_PROJECTION.md`：方法（包括 SWA mixer 的实现细节、RoPE 设置和可比性的局限）、生成的表、结论。
- job 删除证据。
- 一段中文总结，回答四个问题：
  - 投影值多少，分 FIFO 和 learned 两种情况，分 Depo 和 Brevo 两个任务；
  - 同等状态下，Block-KOMP 对比 SWA 的结果；
  - 全注意力的上限在哪里；
  - Block-KOMP 是否只是“带 delta rule 的 sliding window”。
