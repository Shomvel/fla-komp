"""Build the supplemental Chinese report from completed statistics and receipts."""
from pathlib import Path
import json

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / 'physics_lm/block_runs_v4/gsa2'


def main():
    stats = json.loads((ROOT / 'statistics.json').read_text())
    audit = json.loads((ROOT / 'evidence/result-audit.json').read_text())
    cleanup = json.loads((ROOT / 'evidence/job-cleanup.json').read_text())
    assert stats['complete'] and audit['status'] == 'passed'
    # The helper's exact-alias list must be empty before declaring cleanup.
    assert cleanup == []

    def pair(a, b, metric):
        row = next(r for r in stats['paired'] if (r['a'], r['b'], r['metric']) == (a, b, metric))
        digits = 2 if metric.endswith('%') else 4
        lo, hi = row['ci95']
        unit = ' 个百分点' if metric.endswith('%') else ''
        sign = '区间跨零' if lo <= 0 <= hi else '区间不跨零'
        return f"{row['mean']:+.{digits}f}{unit}，95% CI [{lo:+.{digits}f}, {hi:+.{digits}f}]（{sign}）"

    high, low, noconv = 'gsa2_native_lr1e3', 'gsa2_native_lr5e4', 'gsa2_noconv_lr5e4'
    learned, gate, kapa = 'block_slot_learn_c64', 'block_slot_gate_c64', 'kapa_scalar_p2_c16'
    table = (ROOT / 'TABLE.md').read_text()
    text = f'''# GSA2 与 Block-KOMP / KAPA：7-seed 补充比较

GSA2 新增 21/21 次训练完成，三种配置各 7 seeds（731–737）；与上一轮 70 次冻结 Block/KAPA/基线结果合并展示，共 91 次。preflight、源码哈希、训练预算、checkpoint 重载和独立生成审计全部通过。没有更改冻结 kernel 或原有 70 次实验报告。

## 结果怎样解读

1. **GSA2 的 LR 选择会显著影响这里的比较。** 原生层 LR1e-3 相对 LR5e-4：content CE {pair(high, low, 'content CE')}；Depo 生成 {pair(high, low, 'Depo gen %')}；Brevo 合法生成 {pair(high, low, 'Brevo valid %')}。1e-3 来自此前验证集搜索，本轮没有重新选参；不能仅凭统一 LR5e-4 的表现判定 GSA2 上限。
2. **高 LR GSA2 与 Block learned 应分任务看。** GSA2 LR1e-3 相对 Block learned-C64：content CE {pair(high, learned, 'content CE')}；Depo 生成 {pair(high, learned, 'Depo gen %')}；Brevo 合法生成 {pair(high, learned, 'Brevo valid %')}。与 Brevo 均值最好的 Block gate-only 相比，Brevo 差值为 {pair(high, gate, 'Brevo valid %')}。总 CE 的接近不能解释成两项任务能力相同，也不是统计等效性证明。
3. **KAPA 与 GSA2 的 Depo 生成差距仍需谨慎。** GSA2 LR1e-3 相对 scalar KAPA：Depo 生成 {pair(high, kapa, 'Depo gen %')}；Depo CE {pair(high, kapa, 'Depo CE')}；Brevo 合法生成 {pair(high, kapa, 'Brevo valid %')}。不同任务的均值优势不能单独归因于某一种 transition。
4. **原生 short conv 在 LR5e-4 下没有带来更好的 CE。** 同 LR 下 conv 相对 no-conv：content CE {pair(low, noconv, 'content CE')}；Depo 生成 {pair(low, noconv, 'Depo gen %')}；Brevo 合法生成 {pair(low, noconv, 'Brevo valid %')}。这是原生层配置消融，包含其对应激活和初始化行为；不能推导 short conv 普遍无效。本轮没有 LR1e-3 的 no-conv 臂。
5. GSA2 LR1e-3 相对 GDN：content CE {pair(high, 'gdn', 'content CE')}，Depo 生成 {pair(high, 'gdn', 'Depo gen %')}，Brevo 合法生成 {pair(high, 'gdn', 'Brevo valid %')}。两者参数、状态和原生层配方并非完全一致。

## 配方、可比性与局限

- 直接使用已有 FLA 源码中的 `fla.layers.gsa2.GatedSlotAttention2`，Oja2 slot memory → GDN2 value memory；没有重写 transition 或改内核。上游目录无 Git 元数据，以冻结文件哈希为身份依据。`fla/layers/gsa2.py` SHA256：`831e7658047a92ad2422526fc7e6f1f2095ea6c45a3f1340d5928562e7fbd234`。
- 三个臂：原生 q/k/v/w conv4 + LR5e-4；相同原生 conv4 + LR1e-3；关闭 q/k/v/w conv + LR5e-4。WD=.1，并尊重原生 `_no_weight_decay` 标记。峰值 LR 以外沿用 5% warmup、cosine 到10%、残差输出 LR multiplier .1→.3。原生 optimizer 的 WD 例外与旧 Block/KAPA 配方不同。
- `1e-3 / WD .1` 来自此前 `gsa2_runs/selection.json` 的验证集选参记录。本轮只是把该配方与统一 LR 一并复验；不是在同等调参预算下比较所有架构的最优成绩。选参记录副本及哈希在本轮 evidence 中。
- 数据、训练顺序、非 mixer backbone、非 mixer 初始参数均与 v4 对齐：width128、4层、2 heads、K/V64、FFN256、vocab32000，10,001,108 个实际输入 token、2231 steps、batch64、1 epoch，右侧 padding 至192。每个 run 实际计数核对一致。
- **不是等状态预算比较。** 原生 GSA2 只支持 M64/128/256，本轮 M64；循环矩阵每 head 8192 元素。Block 的 keys+coefficients、GSA 两层 memory、GDN 矩阵均为4096；fixed-center KAPA 动态 coefficients 为2048，另有固定可训练中心参数。上述计数不含 short conv cache。GSA2 约938万参数，Block 约897万；详见配置表。
- **不是只替换 transition 的单变量实验。** GSA2 原生 mixer 的初始化、激活、gate、short conv、WD 规则与旧模型不同。只对齐并核验非 mixer 参数初值，不声称所有 mixer 初始权重相同。no-conv 臂保留原生关闭卷积时的行为。旧 Block/KAPA 均无 short conv。
- preflight 对三臂分别在初始化与16次更新后，检查 train/eval 模式下的未来 token 干预、31/32/63/64/65/127/128/129 边界、batch 隔离；未来梯度严格为0，所有参数有有限梯度，16步 loss 下降。preflight 在正式训练前执行，每次训练先核对 ticket 的源码哈希。
- 同一 confirm split；每任务固定前128条做 greedy 生成。Depo valid=exact；Brevo 允许不同合法拓扑序。独立审计重新执行输入字典跳转与输入 DAG 祖先闭包，新增 **5,376 条生成记录**全部与 summary 一致。
- 均值±样本标准差，seed 配对 percentile bootstrap 20,000次，固定 RNG91731；生成差值单位为百分点。CI只描述固定数据/数据顺序/评测样本下的初始化波动，没有多重比较校正。7个seed、128个生成样本仍有限；confirm 此前已被查看，这不是新的外部留出测试。
- `seconds` 是训练+验证+最终生成的整体时长，排除进程启动和 checkpoint 重载；它不是纯 kernel benchmark。GSA2 的更大状态、双层 memory 和原生 conv 都影响成本，不能把时长差直接归因于单个算子。

## 均值与 seed 波动

![Comparison]({ROOT / 'COMPARISON.png'})

误差棒是样本标准差，不是置信区间。正式配对区间见下面的脚本生成表。

## 脚本生成的完整表

以下逐字插入 `physics_lm/block_runs_v4/gsa2/TABLE.md`。数值全部由 `gsa2_trials_report.py` 从91个 summary 生成；原先 v4 的 TABLE.md 与 statistics.json 未覆盖。

{table}
## KAPA 与 Block-KOMP 融合

融合推导与 CPU FP64 参考实现见 [KAPA_BLOCK_FUSION.md]({PROJECT / 'KAPA_BLOCK_FUSION.md'})。结论是可以保留块仿射转移和 scan：分别加入“因果局部 KAPA 写入方向”和“块末保护式压缩”，优先测试后者。p=0 精确恢复冻结 Block baseline；505项数值检查通过，涵盖输出、终态、所有输入梯度与跨块因果性。

不能使用依赖整个 chunk 未来 key 的 Gram inverse 来定义当前写入方向；参考中有实际泄露反例。当前成果是数学原型和验收证据，没有融合 Triton、融合模型质量收益或 learned 淘汰 anchor-cache 的实现证据。既有 KAPA 与 Block 的任务偏好仅构成研究动机，不能证明融合必然两者兼得。

## 产物、重现与清理

- `physics_lm/gsa2_trials.py`：独立原生层实验 harness；未改旧 gsa2/lrwd/shortconv/decay/experiment harness。
- `physics_lm/gsa2_trials_report.py`：完整表与 statistics.json；`gsa2_trials_plot.py`：PNG/SVG 图；`gsa2_comparison_document.py`：本报告。
- `physics_lm/block_runs_v4/gsa2/`：21个 summary、manifest、metrics、生成记录、日志、preflight，不含 `.pt`。
- `evidence/`：冻结快照、源码哈希、计划、选参来源、H100环境、启动配置、source/result/generation审计与退出状态。
- 独立远端目录 `/gpfs/users/linlizhi/komp-gsa2-trials-20261101`；alias `komp-gsa2-trials-20261101-r1`，单H100。已保存作业终态，按精确 alias 删除，并查询确认为空。`job-terminal.json`、`job-delete.json`、`job-cleanup.json` 保存证据。

重建统计：`python3 physics_lm/gsa2_trials_report.py`。绘图：`uv run --no-project --python 3.12 --with matplotlib==3.10.7 python physics_lm/gsa2_trials_plot.py`。重建报告：`python3 physics_lm/gsa2_comparison_document.py`。
'''
    (PROJECT / 'GSA2_BLOCK_KAPA_COMPARISON.md').write_text(text)
    print(PROJECT / 'GSA2_BLOCK_KAPA_COMPARISON.md')


if __name__ == '__main__':
    main()
