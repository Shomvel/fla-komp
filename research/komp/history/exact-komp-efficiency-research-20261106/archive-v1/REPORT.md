# Exact-KOMP 效率研究：首个可评估候选

2026-11-06。研究范围是冻结 native FLA / B16×2 / 340M Exact 版本，不是旧全层 checkpoint 或旧 head 重算循环。当前交付完成源码分析、隔离候选和可运行验证入口；**没有运行 CUDA，没有测得候选加速，也没有获得新增 GPU 授权**。所有修改位于本目录。正式 runtime、checkpoint、指标、队列和监控均未修改，未发送跨会话消息。

**最先值得验证的是：以约 4.55 GiB 持久缓存，省掉 checkpoint 重放中的逐 token 几何求解。** 这是基于源码成本的优先级，不是 profiler 已确认的第一瓶颈。候选复用原始 FP32 `p/h/r/n`，不改变 ridge、FIFO、每 token 压缩、TF32x3、chunk16、状态精度、BF16 网络舍入、原解析 backward、batch 或 loss scaling。真正的吞吐和质量结论必须等待同 checkpoint、同输入、同硬件对照。

**来源与测量边界**

- `evidence/provenance.json`：源 `/Users/shom/Downloads/komp-flame-340m-20c-20261106/migration-fla-mb16/runtime`；manifest SHA256 为 `1dde0fbb5668f1105425de6774a4d188535b278819d3af3f1c569a987359f5ea`。全部 658 个 manifest 文件逐一核验并复制；结束前再次核验源文件未变。`baseline/` 是原样副本；`candidate/` 只改两个文件、增加一个文件，差异见 `candidate.patch`。
- 已阅读并留存 STATUS、EXPERIMENT、FLA-MIGRATION、DATA。最新事实只使用这些本地冻结证据，**没有查询远端实时状态**。
- `evidence/original-exact-startup.json` 和原 STATUS：Exact 暖更新约 5.3433 s / 524288 token，约 98,120 token/s，rank0 peak allocated 32.098 GiB。Block 约 2.54575 s、205,946 token/s、33.076 GiB。不是同硬件受控 A/B，且 Block 压缩频率和 tape 精度不同，不能把差值全归因于 Exact 的某个 kernel。
- `evidence/original-memory-exact.json`：真实 B16 单层 saved-storage 外推为 mixer-only 34.9438 GiB、全部不 checkpoint 97.1477 GiB。外推包含 12 GiB 保守固定项，不是实际整模型峰值。全关 checkpoint 不作为候选。
- 既有 Exact/Block strict trajectory failures 与 FP32 近共线限制均保留。CPU 测试不证明 Triton 数值、H100 吞吐、8 卡稳定性或长期训练质量。

**源码能确定的成本与优先级**

| 优先级 | 源码事实 | 可验证方向及边界 |
|---|---|---|
| P1，已实现 | `baseline/models.py:34` checkpoint 整个 mixer。`operator.py:33` 几何前向被重放；`geometry.py:257` backward 再解伴随系统。 | 缓存首次前向 `p/h/r/n`，重放跳过一个 geo.forward launch 及其中所有 Gram/solve 工作。保留 backward 求解。只减少几何系统次数，不等于训练时间减 1/3。 |
| P2，待 profile | `geometry.py:296` 起对 kept/old/dropped/new 的梯度分别 atomic_add；FIFO 字典在相邻 token 高度重叠。 | 每层 B16T2048 约 2,214,592,512 个标量原子贡献，8.25 GiB 单次写入载荷当量。不是实际 HBM 流量，未包含 RMW、缓存合并、冲突和其他访存。需测 geometry backward 占比、L2/atomic 压力，再考虑聚合/按 key 收集。改变求和次序会带来数值风险。 |
| P2，待 profile | `chunk_backward.py:83` 对 chunk×BH×M×NV 启动 local-gradients；当前 NV=2。四种向量梯度及 beta 使用原子累加，每层约 167,772,160 次，不含 scalar kernel。 | 测量 local-adjoint / local-gradients 的时间、寄存器和占用。V=64 单 tile 或分段确定归约可能减少原子，但要对拍并测 occupancy，不能先宣称更快。 |
| P3，低风险后续 | `operator.py:37,55,63` 分配、保存和解包 `gc`；`recurrence.py:19` 写它；backward 没有读取它。 | 每次前向可去除 64 MiB 全局写入/保存；mixer checkpoint 下并非每层一直保留这块显存，不能直接宣称节约 24×64 MiB 峰值。内部局部 gc 仍被 maps 使用，不能删掉计算。本候选故意未混入此改动，以便归因。 |
| P3，待 profile | `recurrence.py:34,67` 有 scalar 快路，但实际 mixer 输出每槽独立 decay，shape 为 B,T,H,M。 | 不可因初始化相同就改走 scalar。需测 M 次耦合计算、map 构造及 hierarchical scan；改变 scan 分组/融合可能改变舍入。NC=128 走两层 scan，serial 与 parallel 成本都要分开看。 |
| P3，待 profile | `operator.py:155` 把 q/k/v/g 等转为 BH-major contiguous，重放再次搬运；state scans、updates、outputs 分开启动。 | 当前 qkv 合计至少 192 MiB、g 64 MiB 的显式重排规模/层/次，另有 beta/scores 等。先测 copy 和 launch 时间，再研究原生布局或读写融合；融合必须保持输出舍入。 |
| P3，待 profile | `train.py:84` CPU stream 同步组 batch 后阻塞 H2D；首个 microbatch no_sync，第二个同步；step 含 clip/Adam/loss collective。 | 既有日志只有整步时间，不能断言是 GPU 核心或 DDP 主导。约 1.27 GiB FP32 参数梯度/更新/卡是逻辑 payload，不能当成实际 wire 流量或未重叠时间。准备了阶段计时与 NCCL trace；`profile_input.py` 单独测真实 tokenizer/packing。 |

`versioned.py` 的 reuse 仅在 learned eviction 路径启用（`operator.py:31`），当前 FIFO 不适用；不能把现有 learned 快路当成 FIFO 的无损开关。保存完整 M×M Gram/factor/inverse 每层约 2 GiB，24 层约 48 GiB，且因分解/求解方法不同会改变数值；优先级低于只保存几何结果。

**可复现成本模型与收益假设**

运行 `python3 cost_model.py` 重建 `evidence/cost-model.json`。固定 B=16、T=2048、H=16、K=V=64、M=32、C=16、L=24、accum=2。

- `p/h/r/n` 保存量 = `B×H×T×(3M+1)×4` = 203,423,744 B = **194 MiB/层**；24 层 **4.546875 GiB**。缓存不转 BF16、不保存完整 per-token M×V state、不全开 tape。
- 每卡每个 global update：原路径几何求解计数 `3×BHT×24×2` = **75,497,472**；候选为 **50,331,648**，减少 **25,165,824**。三个系统调用中，两个是重复前向投影，一个是不同 RHS 的 backward 伴随求解。此计数只适用于当前 FIFO、mixer checkpoint、训练路径。
- `32.098 + 4.546875 = 36.644875 GiB` 只是加法参考，**不是候选峰值预测或 80GB 安全证明**。真实峰值还受缓存释放、临时空间、allocator、优化器、DDP bucket 及梯度生命周期影响。必须测所有 rank allocated/reserved；不据此扩大 batch。
- 如果被省掉的重放几何占原整步时间 f，忽略新增缓存成本的速度上界为 `1/(1-f)`：f=5%/10%/20% 时约 **1.053×/1.111×/1.25×**。这些是条件算例，不是测得或承诺的收益。缓存读写/驻留及调度成本可能使收益接近零或为负。

**候选实现的契约**

`candidate/geometry_cache.py` 为每次 non-reentrant checkpoint 创建独立 context；缓存由当前计算图持有，通过 ContextVar 区分原前向与重放。重复 backward 可重用，图释放后释放缓存。没有进程级张量缓存，不跨 microbatch/层/训练步骤复用。重放检查调用位置与 shape/device/dtype/参数契约；它不逐元素比较重算输入，依赖 PyTorch checkpoint 的确定性重算约定。原 mixer 没有跨调用可变状态。

`candidate/models.py` 的开关默认为 **False**，研究 harness 显式打开。仅 eager、Exact、FIFO、FP32 路径使用缓存；learned 与 FP64 不缓存，torch.compile 明确不支持。缓存值在 custom Function.forward 内产生，原 custom Function.backward 和它保存的张量顺序不变；没有用 detach 旁路替代梯度。没有修改任何 Triton kernel 数学代码。

这种机制依赖 checkpoint 内部行为，已在本机 PyTorch 2.12.1 CPU 做功能验证。生产镜像 torch 2.10.0a0 / Triton 3.7.1 的 CUDA 行为仍须验证。缓存持有的额外内存不一定出现在 saved_tensors_hooks 的统计里，所有内存结论必须以 CUDA peak 为准。

**验证与 benchmark 入口**

`README.md` 给出执行命令；所有 GPU 入口尚未执行。

1. `prepare_fixture.py` 在 CPU 上复制一个指定的已完成 checkpoint 到新的私有目录，逐文件校验 SHA，保存原模型、Adam、8 rank RNG/loader，读取每 rank 后续 32×2049 真数据 ID。不会推进正式 loader。需在源 checkpoint 被 retention 删除前完成；捕获遇到变化则失败。当前没有访问远端，fixture 还未生成。
2. `operator_check.py` 对 FP32/BF16、T=1/15/16/17/496/497/511/512/513/2048，检查输出、两个 state 输出与全部输入梯度，覆盖非完整 chunk、NC=31/32 的 scan 路径。使用真实 checkpoint 重放机制；仍是合成算子输入，不冒充真实语言模型输入。固定原 FP32 .005/.005、BF16 .035/.04 门槛，保留 nearzero 1e-7。不宣称已解决近共线压力限制。
3. `benchmark_layer.py` 用真实 checkpoint 的 embedding 和第 0 个 native FLA block，在 B16T2048 测冷启动、warm forward/backward、allocated/reserved、saved storage 和 profiler trace。不是全模型性能。
4. `benchmark.py --mode check` 用**相同 native FLA checkpoint，B16×2 对 B16×2**，不拿旧 B32/legacy 作参考。full=65536/rank，tail=63616/rank；两个微批 valid 为32768+32768或32768+30848，都除以总 local valid。保存每个参数的 clip 前 grad 与 clip+Adam 后 delta；校验 finite、model/Adam/RNG restore、前缀因果（eval 路径），8 卡时校验更新后 replica hash。真实 loader 恢复 witness 由 fixture 工具执行。
5. `compare.py` 逐 tensor 使用原 .035 RMS/.04 normalized max 或1e-7 absolute门槛，loss<.005；不因 whole-model 接近就把 strict failure 改成通过。先验证 baseline 对 candidate-cache-off 的控制，再测 cache-on。单卡不等价于 global256。
6. `benchmark.py --mode bench` 顺序 ABBA，独立进程恢复相同模型/Adam/RNG。首步单列 compile/初始化，3 个额外 warmup 后统计10次 warm；每个 microbatch forward/backward 用 CUDA event，optimizer 单列，host 总时间包含阻塞H2D、DDP及loss collective。8卡按最慢 rank 整步统计。trace/profile 与抓取数值的步骤不进入速度统计。
7. `profile_input.py` 单独测原 JSONL/tokenizer/packing，同一 cursor 重放并核对真实 IDs。其单 rank/OS缓存结果不代表8rank竞争。主 benchmark 重复固定 batch，**不包含真实逐步加载、定期 validation/checkpoint I/O、学习率调度变化与整个6.8B过程**。原长期指标也是仅更新计时；如要评估 wall-clock 完成时间，仍需独立测这些额外成本。
8. `run_stage1.sh` / `run_stage2.sh` 均 `set -euo pipefail`，首次失败停止剩余动态验证，先静态排查同类问题再修复。不得通过反复跑到一次通过或放宽阈值获取接受。

后续推广还需要：同硬件 ABBA 的收益超过波动、生产镜像重复结果、短程固定 token 的轨迹与 heldout CE 对照。一次更新通过不足以证明长期质量等价。不存在自动替换正式 runtime 的入口。

**当前已完成与尚未完成**

`evidence/cpu-tests.txt`：5项CPU测试通过，覆盖重放次数、不同图隔离、retained graph 二次 backward、嵌套、异常后 context 恢复、缓存生命周期、数值比较器对 NaN/Inf/真实偏差的拒绝。Python语法、CLI help、shell语法已检查。`evidence/local-validation.json`：658个原文件重新核验、0 GPU作业、0远端修改。

未验证：GPU kernel执行、缓存路径在生产torch版本上的行为、实际峰值/吞吐、DDP通信占比、真实输入IO耗时、真实模型 loss/grad/update 对拍及长期训练质量。因此候选是**准备好验收的研究版本，未获训练放行**。

**资源申请准备（没有提交）**

依照 `/Users/shom/.codex/skills/northjob-gpu-test-bp-dev/SKILL.md` 准备独立 GPFS 目录与 pinned image YAML，project=`WeaverRSI`、queue=`proj-32`。提交前必须重新核实项目/队列与容量。用户已经明确当前16卡全部占用且未授权额外GPU，所以这里停止于可审阅的配置，不抢占、停止或改动原任务。

| 阶段 | 预备作业 | 资源/上限 | 内容 |
|---|---|---|---|
| 1 | `exact-efficiency-stage1-20261106` / `job-stage1.yaml` | 1×H100 80GB，8CPU，64GiB RAM，最多45分钟，0.75 GPU小时 | 算子门槛；full/tail控制组与缓存组；真实层profile；340M单卡ABBA与trace。冷编译是否能在上限内完成尚未实测；超时保留证据，不自动续租。 |
| 2，阶段1通过后 | `exact-efficiency-stage2-20261106` / `job-stage2.yaml` | 8×H100 80GB，64CPU，256GiB RAM，最多30分钟，4 GPU小时 | global256、full/tail、8rank replica/restore、ABBA吞吐。需新增授权或原队列自然释放后的明确分配。 |

两个阶段串行，总上限4.75 GPU小时；最大一次新增8卡。这是资源请求方案，不是已获批预算。输出保留完整grad/delta会占用较多GPFS空间：8rank每次检查约20GiB，阶段2四次检查约80GiB，另加fixture/trace；准备前检查独立目录空间。所有测试更新都从私有fixture出发，不写回正式checkpoint、不计入原训练token预算。镜像、工具版本及输入hash在结果中记录；候选不会自动进入正式训练。
