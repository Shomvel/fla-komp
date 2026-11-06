# 全 Triton 实现

默认 `projected_slot_attention(..., backend='all_triton', chunk_size=16)`。保持原核、正则化投影、给定路由和 rank-2 delta 更新规则。支持 `chunk_size=16/32`；原混合实现仍可用 `backend='chunk'` 对照，逐 token fused recurrent 仍为 `backend='triton'`。

## Kernel 划分

1. `geometry.py`：归一化 query/key/初态；按路由生成各槽位最近写入 token 的索引；各 token 并行计算核 Gram 和投影。Gram 是对称正定矩阵时用无主元 LDLᵀ 分解并求解。默认 ridge 保持 `1e-4`；`ridge=0` 要求 Gram 非奇异。
2. `triton_chunk.py::prepare_kernel`：每块一个 program，融合 decay、rank-2 内积、分块三角求逆、四个 transfer map。零门采用不除法的乘积扫描。思路对应 FLA 的 `generalized_delta_rule/iplr/wy_fast.py` 和 `ops/utils/solve_tril.py`，原型没有修改上游文件。
3. `scan_kernel`：仅沿块边界递推，块内输出由矩阵乘法计算，保存入口状态。
4. 反向先用 `boundary_bwd_kernel` 扫描块边界伴随状态，再由 `local_bwd_kernel` 在各块内并行重算 token 状态并求局部导数。它直接求原转移的导数，无需对四个 transfer map 做通用 autograd。归约也由 Triton 完成。
5. 几何反向求解 `Gᵀ z = dp`，使用 `dG = -z pᵀ` 和核函数、归一化的解析导数。不同 token 对同一个历史 key 的梯度用 FP32 atomic add 汇总，因此不保证逐位确定性。
6. `utils.py`：Triton 清零及有限数值、门值和 slot 范围检查。Python/PyTorch 仅承担内存分配、形状检查、必要的连续布局复制、autograd 接口和取回一个错误码；输入检查保留一次 CPU 同步。

逐slot alpha 路径使用 `prepare_slot_kernel`：每个chunk独立组合diagonal+rank-2仿射变换，chunk内部16/32步顺序执行，chunk间并行；零门不需要除法。后续scan接口不变，local backward的alpha梯度只归约value维。scalar仍使用原并行块系数路径。`decay.py`提供GDN/GSA门的Triton前反向。

来源追踪已加入3阶段整数max scan；`T>=192,M>=16`默认采用，其余尺寸保留串行。`triton_geometry_serial`是冻结旧基线。gather LDL与TF32x3几何候选虽通过本轮数值验收，但跨shape收益不稳定，只保留显式入口。见 [最新加速结果](DECAY_ACCELERATION_RESULTS.md)。

块矩阵乘法采用 Triton `tf32x3`，FP32 累加；Gram 和几何梯度的矩阵乘法采用 `ieee`。没有把状态或投影求解降成 BF16。FP16/BF16 输入在 kernel 内转成 FP32，输出和状态为 FP32，输入梯度回写输入 dtype。全 Triton 路径不依赖外部 autocast 设置。

## 内存和适用范围

槽位数 1–64，dense `[B,T,H,D]` 输入，整数路由固定。不会求路由 argmin 的导数，也不包含完整 KOMP 删除候选搜索。TF32x3 路径面向支持它的 NVIDIA GPU，目前验收设备为 H100。

前向只保存块入口 value 状态，但局部反向会临时分配 `O(B H ceil(T/C) C M Dv)` 的重算缓冲；几何训练另保存 `O(B T H M²)` 的 LDLᵀ 因子，以及 `O(B T H M)` 的来源索引。尚未做大规模训练的峰值内存优化。无梯度前向不保存这两个训练缓冲。

不支持 varlen、GQA、二阶梯度或可学习温度。正则化过弱、极端温度、近奇异 Gram 或长序列不稳定转移仍可能产生数值问题；目前的加速不改变模型稳定性条件。

## 验收

运行 `python validate_all_triton.py --benchmark`。沿用既有 FP32 数值容差，不为优化放宽。检查各块系数、输出/终态、所有输入梯度、零门、重复路由、尾块、非连续输入、低精度输入、流式和因果性；同时用 profiler 检查前反向不存在 ATen 矩阵运算、核运算或归约。

性能对照包括同一转移的旧混合 chunk 和 fused recurrent。端到端计时包含归一化、核几何、投影求解、输入检查和 chunk 算子，排除首次编译；它不是与 GSA/GDN 模型吞吐或模型效果的比较。

此前版本验收：422 项检查通过；环境、逐项误差、profiler 事件及源码指纹见 `evidence/all-triton-gpu.json`。本轮decay和并行route的独立验收及最终集成回归存放在 `evidence/decay-20261031/`，不覆盖旧证据。
