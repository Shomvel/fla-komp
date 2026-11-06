# 投影 / SWA 实验方法

**冻结与隔离。** kernel 完全来自 `frozen-20261101-proj`，上传前逐文件核对 `SHA256SUMS`。当前主树的 `experiment.py`、`block_experiment.py` 只复制到独立快照；没有修改共享 harness 或 kernel。所有 Block 臂使用同一份 FP32 `block_flex`，q/k/v 由外围 BF16 autocast 产生。关闭 coherence、dedupe、projectron 和 short conv。全程启用确定性算法、设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，关闭 TF32。

**实验臂。** 主实验为 FIFO/learned × 投影开/关四臂，以及 SWA32-RoPE、SWA96-RoPE、SWA96-NoPE、全因果192-RoPE 四臂；每臂 seeds731–737，共56次训练。Block 使用 M32、C64、逐 slot 衰减；learned 同时使用现有 score 头的 `beta *= 2*sigmoid(score)` 写入门控。C128 的 FIFO 开/关投影为预先规定的14次补充实验，在主实验全部成功后才开始。沿用用户授权的16张独立H100并行；首个作业完成全部 preflight 后才提交其余作业。首次错误保留 failure.json 并阻止后续训练，无自动重试。

**SWA 模块与初始化。** 直接复用原 KOMP 已初始化的 q/k/v/out/gate/norm 模块对象，删除不用的 decay、beta 和初始字典参数。逐层核对全部共享参数，以及 embedding、head、FFN 和各归一化参数的 SHA256；报告再次按 seed 检查臂间一致性。SWA 的 q/k 不做 L2 normalization，注意力分数为 `q·k/sqrt(64)`；SDPA 不加 dropout，布尔掩码为 `0 <= query_position-key_position < W`，W 包含当前位置。注意力输出继续使用原有逐 head 的 FP32 RMSNorm、SiLU 输出门控和输出投影。

**RoPE。** 旋转完整的64维 head，按相邻坐标成对，base10000，序列第一个位置为0；角度与旋转在FP32计算，再转回 q/k dtype，v 不旋转。NoPE 仅跳过这一步。W192 覆盖长度192内的全部因果前缀。文档仅右侧补零；padding 位于有效预测的未来，因果掩码阻止它影响非padding的训练损失与生成查询。

**状态预算。** W32 与 M32 均为每 head 4096个常驻状态元素，即每层8192、四层32768。W96 为每 head12288，W192为24576。这是元素数对齐，不是字节数对齐：SWA KV 为BF16，Block的常驻key/系数状态为FP32，learned还携带score记录；Block临时字典在块内从M增长到M+C。本轮未实现增量KV cache，也未测量cache显存或kernel吞吐，生成沿用原harness重算完整序列的方式。跨家族还存在 RoPE、Gaussian/softmax读出、delta写入、可训练初始原子及有效参数量差异，因此它是匹配backbone的质量比较，而非仅投影的单变量比较。

**训练与评估。** width128、4层、2 heads、K/V64、FF256、词表32000；同一份10,001,108个非padding训练token，1 epoch、2231 steps、batch64，固定样本顺序。AdamW betas=(.9,.95)、WD=.1；峰值LR5e-4、5% warmup、cosine到10%，残差输出LR倍率由.1升到.3。没有逐家族调参。confirm CE使用完整split，总content CE为Depo与Brevo两任务content CE的算术平均；生成各取固定前128条样本、greedy。Depo合法率等于exact，Brevo允许其他合法拓扑序。confirm此前已用于探索，不能视作全新外部测试集。

**Preflight。** 十个臂均检查初始化与16步更新后的未来后缀干预，覆盖窗口和chunk边界；同输入重复forward必须逐位一致；16步loss下降；learned score梯度非零且有限。SWA另在单个mixer输入上检查窗口外扰动不影响当前输出，而最早仍在窗口内的位置能够影响输出。窗口局部性只针对单层mixer，多层网络会扩大感受野。W192的边界检查使用长度201输入；正式质量实验仍为长度192。

**统计。** 每臂报告均值和样本标准差；按seed配对差值，以RNG91731做20,000次配对bootstrap，给出percentile95%CI。未做多重比较校正；区间只描述固定数据、顺序与生成样本下的初始化波动。预设对比为FIFO/learned各自的无投影减有投影、FIFO有投影减SWA32/96、RoPE96减NoPE96，以及C128无投影减有投影。全注意力是本配方下的参考，不能视作已证明的性能上限。

**时间。** 对验证、评估和生成使用CUDA同步计时。训练循环加记录开销 = summary总时间 − 测得的评估时间 − 生成时间；checkpoint重载评估另行记录，不在summary总时间内。这不是纯优化步骤GPU计时。flex与SDPA实现不同，不能把逐run墙钟时间当作kernel速度比较。

**解释边界。** 关闭投影只去掉一条明确的跨chunk系数转移路径，并没有去掉全部较早token依赖：保留下来的delta系数本身来自含旧记忆的预测残差。FIFO约束的是字典key的时间范围，不一定是系数所含信息的全部历史；learned还可以保留非最近的key。因果性验收通过，也不意味着整个模型只有W-token感受野。
