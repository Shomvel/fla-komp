# r1 模型失败的静态排查与 r2 诊断

r2 仅新增诊断，`komp_inference/`、模型验收门限及原 r1 测试代码逐字节保留。尚未提出或部署数值修复。原 r1 manifest 为 `f0ebd0c1b3375e671d849a99e70c2019b3cf20be89d588982e4e5fe86dd050f8`，原失败回执独立复制到 `evidence/parent-r1-model-failure`。正式训练和下游评测源未修改。

原 Block final checkpoint 首个 T31 前缀 logits RMSrel `0.003007770501277965`、maxabs `0.0625`，超过冻结 `RMSrel < .003 OR abs < .0003`。r1 在此抛出异常，top1 尚未知；不能称模型通过，也不能从 120 项算子通过外推模型通过。

## 已由代码确认

1. `prefill.py` 只把 `T//128*128` 个 token 交给冻结 Block。T<128 整段 recurrent；T=129/257 的尾段 recurrent。所有六个 Block 变体共享此分支。Exact 走另一条 C1 分支，仍需要完整模型验收。
2. 冻结 `block_fast.py:_BlockFast.forward` 的 qn（BF16输入）、HK/HN/RK/RN、TT、OS/OV 均 BF16；readout `_adot` 把状态和映射转 BF16，再 tensor-core 累加 FP32。候选 `_step_kernel` 的 normalized q/k、kernel权重、delta write/read 及状态均 FP32，最终输出才转 BF16。代数相同不代表这些计算路径舍入相同。
3. 冻结 Block 即使 T31 仍建立 C128 矩阵，通过有效 token mask 处理尾段，终态压缩至M。候选保留 M+31 个开放原子，直到全局128边界。终态压缩视图只可用于比较，不能用于恢复半块，否则改变算法。
4. 候选 `initialize_state` 先 normalize 初始keys，之后 native整块调用内部再次normalize；冻结原生只在内部normalize。该额外舍入也可能影响完整chunk路径，诊断单列 `native_double_normalization`。
5. T>1 的 `ShortConvolution.forward` 两种 final_state 标志均调用 `causal_conv1d`；T=1 才进入step。原快照 `causal_conv1d_fwd` 在完成同一输出 kernel 后，另调用 `causal_conv1d_update_states` 提取最终状态。静态未发现 flag 改变输出公式，但环境 backend 覆盖或编译差异仍由同一projection的双调用实测排除。实际decode另比原生完整前缀。
6. 原模型 `fuse_norm=False`、`use_attnres=False`。候选 attention norm → mixer → residual → MLP norm → 原MLP → residual → final norm/head 顺序与冻结 GatedDeltaNetBlock 相同。FP32残差沿用；MLP使用原模块，保留旧BF16两次舍入。诊断使用原模型真实 forward hooks 与手工同输入控制，不靠仅看最终logits推断。
7. learned的score只乘beta一次；GDN `-exp(A_log)*softplus(z+dt_bias)`、旧slot/value `logsigmoid(z)/8`、输出门sigmoid/SiLU均静态匹配。诊断仍逐层对q/k/v/g/beta/keys/coef/scores核对。
8. Value两类冻结算子使用完整输入范围的 `amin/amax` 决定坐标scale origin，span>24时走独立稳定后端。截短前缀会改变origin及可能改变后端，故 full-prefix oracle 的早期输出自身可能存在数值漂移。原门限允许数值因果，不能称严格因果或bitwise。诊断逐次记录 frozen_prefix_stability，与候选缓存误差分开。

## 尚未证明的原因

BF16中间舍入是明确路径差异，但尚未测得其对本次 .00300777 的贡献。额外keys normalization、卷积最终状态、局部归约及误差穿过 norm/MLP 的放大都须按同输入实测。不得把近门限失败改称通过或仅凭差值很小放宽门限。

## 诊断入口

由主 agent 指定现有空闲GPU。每模型独立进程，输出目录必须不存在：

```bash
CUDA_VISIBLE_DEVICES=<主agent已选GPU> python diagnose_model.py --model block --lengths 31 127 128 129 257 --decode 4 --output-dir attempts/r2-diagnostic-block
```

可先 `--lengths 31` 定位首个失败；这是新的逐层诊断，不是重跑原失败验收。全矩阵覆盖8个模型时，每模型启动独立进程以隔离冻结FLA模块导入。该脚本不提交/清理作业；只允许恰好1张可见GPU。

每个长度完整捕获原24层输入、norm、mixer和输出。固定原层输入，比较conv final标志/缓存、全部operands、冻结算子与Torch/Triton recurrence、终态压缩视图、开放状态、r1混合prefill、后处理及残差MLP。再运行r1级联，记录每层误差和最终logits、top1、top2 margin，即使门限失败也继续记录。

另有诊断用 `NativeOutputProbe`：整个prefill输出交给原native算子，同时独立构建r1开放状态。它用于检验“保留原生prefill输出能否去掉首步logits漂移”；下4个token仍记录缓存对全前缀baseline的偏差，不能由prefill改善就认定decode修复。输入为同一固定NIAH请求，逐步teacher forcing原baseline token，避免greedy分歧改变比较输入。报告保存请求原始IDs、请求文件SHA、模型/source/data/checkpoint pins、候选完整manifest、运行时TF32状态。

`status=diagnosed` 仅指采集完成。比较条目的既有数值门限字段不是新的验收结果。GPU异常会留下 `diagnostic_execution_failed` 回执及已flush比较，不再自动重跑。
