# KOMP、GSA、GDN 速度比较

比较固定于 FLA commit `97874af2400916e2885c33ea30559a70ff686b66`，使用本目录已验收的全 Triton KOMP 原型。复现入口为 `benchmark_fla.py`，GPU 作业配置为 `gpu-fla-benchmark-job.yaml`。

## 比较口径

主配置为 `B=2, H=4, Dk=Dv=64, M=32`，序列长度 256/1024/2048/4096，BF16 输入、FP32 门和初始状态。这样三者每个 head 都有 4096 个状态元素：

| 算子 | 每个 head 的状态 | 元素数 |
|---|---|---:|
| KOMP | key 字典 `M×Dk` + value 系数 `M×Dv` | 4096 |
| GSA | key-slot 状态 `Dk×M` + slot-value 状态 `M×Dv` | 4096 |
| GDN | key-value 状态 `Dk×Dv` | 4096 |

补充 `T=1024` 的 FP32 输入，以及 BF16 下 M=16/64。后两组的 GSA/KOMP 状态大小变为 2048/8192，GDN 始终为 4096；它们用于观察槽位数的影响，不能视为等状态预算。

三者使用同一份原始 q/k/v，所有初态非零，均输出终态，训练计时包含初态和终态的梯度。GSA 使用独立的 s/g 输入，GDN 使用融合 QK 归一化，KOMP 使用完整核几何。GSA/GDN 使用原生 chunk size 64，KOMP 使用已验证较快的 16；GSA 保留默认 checkpoint level 2。没有强行把不同算法的块长设成相同值。

**这是序列算子比较，不是完整网络层比较。** 不含输入/输出线性投影、卷积、门网络、路由网络、优化器和 loss。KOMP 的整数路由提前给定；GSA 的 slot 写入权重/门、GDN 的门提前给定。三者的数值输出没有等价关系，等状态存储量也不意味着等模型容量或等任务效果。

## 计时范围

- `komp`：归一化、按路由追踪 key 来源、kernel Gram、投影求解、块系数、recurrence，以及训练时的相应全部梯度。
- `komp_core`：已预计算 h/p/r 后的块算子，训练时只求 h/p/r 等输入梯度，不包含几何及其反向。这个数字只用于定位成本，不用来宣称完整 KOMP 快于 GSA/GDN。
- `gsa`：上游 `chunk_gsa`，含两个记忆阶段及 slot softmax。
- `gdn`：上游 `chunk_gated_delta_rule`，含融合 QK 归一化。
- KOMP 的高层数值/索引检查会同步 CPU。主表使用不含该检查、但计算完全相同的底层组合入口，以便与 FLA 算子比较和捕获 CUDA Graph。高层 API 的 eager 前向另外记录。

每项先预热完成 JIT/autotune，再记录三轮 eager 和 CUDA Graph replay，取中位数。eager 每轮 warmup 20 ms、rep 80 ms；graph 每轮 rep 80 ms，捕获成本不计入 replay。两种模式分别报告，防止将主机调度开销误认为 GPU 算术成本。

此次性能测试关闭未初始化内存 NaN 填充，沿用生产模式的内存分配行为，因此不应与之前开启该检测的报告直接计算速度比。所有方法在同一作业、同一张 GPU 上测量。

FP32/BF16 指输入 dtype：KOMP 内部状态与几何仍使用 FP32，块乘法为 tf32x3；FLA 使用原生精度策略。比较并不要求三个不同算法采用完全相同的指令精度。

## 来源与验证

上传的 FLA 算子和工具文件保持原样。脚本只初始化 `fla`、`fla.ops`、`fla.modules` 包命名空间，跳过可选 Transformer 模型注册；没有替换或修改 GSA/GDN kernel。`FLA_DISABLE_BACKEND_DISPATCH=1` 确保比较原生 Triton，避免自动切换外部 FlashQLA 后端；关闭 tensor result cache，避免同一输入对象复用影响计时。

计时前，各算子与自己的逐 token reference 比较输出、终态和梯度，检查 FP32/BF16。不同算子之间不做数值相等断言。结果 JSON 保存所用全部 FLA Python 源码的 SHA-256，以及原型和 benchmark 的源码指纹、设备/软件版本、逐轮计时与验证误差。

环境预检发现原固定镜像的 Triton3.6.0 命中上游 `common/chunk_o.py` 对 Hopper gated backward 的已知错误保护。r1因此停止，未产生可用性能结果；没有移除该保护。后续统一用 Triton3.7.1 的隔离overlay，wheel SHA-256为 `7e40869937a68206ec70d7f25bb7ec6433cb083f9135e1f36dbd318dc449a728`。镜像中的Torch/CUDA和其他依赖不变。

数值验收r2发现BF16 GDN的衰减门梯度RMS误差1.003%，超过脚本统一1%门槛。静态核对上游`tests/ops/test_gdn.py:test_chunk`第261–262行，dg和dbeta使用2%容差。r3仅将这两类BF16梯度容差对齐2%，其他指标保持不变，完整重跑验收。原失败日志和JSON保留，不将它记为1%精度通过。
