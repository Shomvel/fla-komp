# Exact KOMP 与 FLA GDN：8K 算子比较

`compare.py` 对相同的 B=1、T=8192、H=2、K=V=64 输入进行比较。KOMP M=32，每个 head 的字典 keys 与 coefficients 共 4096 个 FP32 持久状态元素；GDN K×V=4096。learned 另有少量 score/birth 元数据。状态元素数相同不代表模型容量或质量相同。

两种输入精度：FP32、BF16。标量 log decay、beta 以及初始状态均为 FP32。两边包括 Q/K 归一化、返回最终状态及完整输入反向；KOMP learned 包括其 score write gate。KOMP 保留原实现的 FP64 几何投影，GDN 使用原生 FLA 精度。KOMP 计算块 16/32 均保持逐 token 压缩语义；GDN 使用默认计算块 64。

前向在 no_grad 下测量。前向＋反向使用随机输出和最终状态 cotangent，以 autograd.grad 返回梯度，不累加 .grad。两边不包括模型 QKV 投影、短卷积、输出层、优化器和 learned score 网络。全部 8K 输出和梯度检查有限值；GDN 另做 T65 的 FP32/BF16 naive 参考对拍。KOMP 数值差分证据见上一轮 komp-exact-triton-20261105，不将两种不同算法互相作数值参考。

5 次预热后，用 CUDA events 测量 3 组、每组 20 次，报告每次耗时的组间中位数。另捕获单次完整调用，通过 CUDA Graph replay 按同样方法测量，降低 Python dispatch 对计时的影响。编译、autotune 和 graph capture 均不计入。显存是 eager 调用相对常驻输入及上游梯度的峰值 allocated 增量，不是 reserved，不包含 graph pool。

固定镜像、项目和队列见 job.yaml。runner.sh 在 worker 中运行；FLA operator namespace 从之前验收的远端 fla-source 复制，避免导入无关 transformer 模型注册，算子源码未修改。comparison.json 保存所有 FLA Python 文件及本次 KOMP 实现的 SHA256，environment.json 保存实际运行环境。原生 Triton dispatch 与 FLA tensor cache 设置均在 runner.sh 明示。

原始数据和任务回执保存在 evidence/，汇总见 RESULTS.md。
