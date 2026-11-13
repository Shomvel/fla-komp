# r3：有界 native Block cache 候选，尚未GPU验收

本revision针对r2真实Block checkpoint诊断定位的数值路径差异，增加`NativeChunkKOMPLM`，没有覆盖原r1的`RecurrentKOMPLM`。所有r1/r2包、失败证据、正式训练和正式评测保持不变。当前仅完成CPU状态性质验收，不声称速度提升、模型验收通过或bitwise兼容。

## 范围

候选支持原Block、Block123、learned slot、learned slot123；Exact与Exact123仍使用原r1 C1 token路径，必须在r3完整源下重新通过模型/greedy门禁。两个Value变体在新入口明确抛出unsupported，禁止悄悄退回已失败的r1完整模型路径。它们原实现与固定checkpoint pins保留用于未来工作。

诊断的top1错误已修正：直接调用`argmax`，topk仅用于第二名margin。平局回归覆盖较小索引必须获胜。r2原回执里零mismatch仍是错误诊断数据；纠正分析复制在evidence，原始文件不改。

## 算法与状态

fresh prefill仍一次调用冻结native算子得到完整输出。若有尾段，再对完整块前缀提取其真实native边界state，保留尾段原始q/k/v/g/beta/score。decode只投影新token，每层用“最后已完成块的state＋当前开放块（最多128token）”调用冻结native算子；仅取新增token输出。开放块返回的临时压缩终态被丢弃，恰好到128边界才提交。API跨多个块时依次处理；不为每个token提前压缩。

持久缓存为M个raw keys及FP32系数、真实scores/births，加最多127个投影operands和原shortconv状态。切片均clone，避免小tail视图保留整个prompt底层storage。缓存可按batch reorder/duplicate及reset，position按全局token累计。全程复用模型模块与state_dict，不改权重。

保留raw而非normalized keys：原native完整调用只normalize每个pool key一次；每块返回的normalized key再次送入native会引入重复舍入。因此根据字典路由保留对应原始key，系数仍来自native。FIFO C128整块边界且M32时age order与physical slot一致。learned保留物理slot；选择时把所有M+当前N个atom的真实(score,global birth)顺序编码为唯一小整数rank，仅传给native选择核。原始score仍用于beta写门和持久元数据，没有修改写入强度。此举避免native独立调用把旧birth重置为-slot序而破坏同分newest规则。

## 数值边界

当前块的BF16 QN/HK/HN/RK/RN/TT/OS/OV、矩阵乘和输出都由原native核执行，替代r1 FP32逐token路径。仍不承诺和全prefix数值完全一致：原native在B1时用分组affine scan、B4时用serial scan，其全序列组合顺序与边界state接续存在FP32/TF32舍入差异；prefill截到完整块长度也可能改变group布局。GPU必须覆盖B1及B4和127/128/129/257边界，真实模型门限没有放宽。

Value算子使用整段cumulative residual的min/max决定center origin，span>24时选稳定fallback；未来token能改变origin或后端，从而影响历史BF16运算及旧输出。仅保存局部块state并重新选择scale无法保证复刻全prefix历史，不能从slot通过外推Value通过。本revision未实现Value bounded cache。

## CPU证据

23项新增/相关诊断与chunk性质测试通过：独立冻结token oracle对FIFO/learned、prefill31/127/128/129/257、任意跨块追加到385；partial终态毒化证明未提前提交；raw keys、global birth同分选择、reorder/duplicate、257到513同余位置cache字节数不增长；真实score/beta不变、模型权重不变/reset、Value拒绝、argmax平局规则。CPU不验证native GPU矩阵舍入，也不等于340M checkpoint验收。

## 固定GPU验收，root唯一执行者

沿用`validation-plan.json`全部不可变source/data/checkpoint pins，包括原Exact partial7168和其余final12970。r3完整manifest必须验证，不可复用r1/r2通过回执。主agent显式选择现有空闲GPU，入口不申请资源。

```bash
CUDA_VISIBLE_DEVICES=<已核对空闲GPU> python gpu_entry.py --attempt r3-block-operators --stage operators --models block
CUDA_VISIBLE_DEVICES=<同一GPU> python gpu_entry.py --attempt r3-block-models --stage models --models block --operator-attempt attempts/r3-block-operators
```

算子阶段除保留原r1算子对拍外，增加真实native C128 cache对整前缀的B1 FP32/BF16、B4 BF16，五种prefill切分，含learned同分分数。模型阶段B1五个边界、B4三个边界，每点8步teacher-force，对每步`.003 relative OR .0003 absolute`与精确argmax同时门禁，先打印argmax再执行数值断言。首错即停且保留新attempt；不重试、不放宽。默认六项候选，Value不能进入新门禁。

模型通过后才用同源`fixture_validate.py`对原完整batch先复现golden，再对新cache生成IDs/文本/分数，保留原B/顺序/真实长度/stop。只有这些通过后才能运行同卡benchmark；benchmark也是新cache入口，不能拿旧r1时间冒充。
