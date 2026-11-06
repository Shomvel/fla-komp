"""Build the final document only from complete, audited 7-seed results."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT/'physics_lm/block_runs_proj'


def main():
    stats=json.loads((RUNS/'statistics.json').read_text())
    assert stats['integrity']=='complete'
    audit=json.loads((RUNS/'evidence/local-result-audit.json').read_text());assert audit['status']=='passed'
    gen=json.loads((RUNS/'evidence/independent-generation-audit.json').read_text());assert gen['completed_runs']==70
    for i in range(1,17):assert json.loads((RUNS/f'evidence/final-cleanup/r{i}-cleanup.json').read_text())==[]
    narrative=(RUNS/'evidence/conclusions.md').read_text()
    methods=(RUNS/'evidence/METHOD.md').read_text()
    table=(RUNS/'TABLE.md').read_text()
    lengths=json.loads((RUNS/'evidence/sequence-length-audit.json').read_text())
    length_table=['| split / task | documents | mean / max length | content predictions after C64 | after C128 |',
                  '|---|---:|---:|---:|---:|']
    for name,r in lengths.items():
        length_table.append(f"| {name} | {r['documents']} | {r['mean_length']:.2f} / {r['max_length']} | {100*r['fraction_predictions_after_first_C64']:.2f}% | {100*r['fraction_predictions_after_first_C128']:.2f}% |")
    (RUNS/'LENGTHS.md').write_text('\n'.join(length_table)+'\n')
    text='# Block-KOMP 投影消融与 softmax SWA：Depo/Brevo 7-seed 实验\n\n'
    text+='70/70 次训练完成：8 个主实验臂与 2 个 C128 补充臂，各 7 seeds（731–737）。所有 preflight、训练预算、共享参数初始指纹、checkpoint 重载和独立生成审计通过。kernel 未修改。\n\n'
    text+=narrative+'\n\n## 方法与可比性\n\n'+methods.split('\n',2)[2]
    text+='\n## 序列长度与投影影响范围\n\n'
    text+='下表由只读数据脚本统计；预测位置 i 用于预测 token i+1。只有跨过块末的预测能在该层直接受到投影影响。C128 的答案大多出现在第一次压缩之前，不能把这组短序列结果外推到长上下文。\n\n'+'\n'.join(length_table)
    text+='\n\n## 脚本生成的完整结果\n\n以下内容从 `physics_lm/block_runs_proj/TABLE.md` 原样插入，由 `block_report_projection.py` 从 `summary.json` 生成。\n\n'+table
    text+='\n## 验收与资源清理\n\n'
    text+=f"- 冻结 manifest 校验、FP64 数值对拍、所有十臂 preflight 均存档。提供的 CPU 脚本实际枚举 32 组配置；验收以实际输出计数为准。\n- 独立生成审计检查 {gen['checked_generations']:,} 条记录：Depo 从输入字典执行跳转，Brevo 从输入图重新求祖先闭包，与所有 summary 一致。\n"
    text+='- 两个有投影C64对照臂的14次独立重跑，其confirm指标和生成统计与上一轮确定性coherence-off对照完全一致；未复用旧结果文件。证据为 `evidence/control-reproduction.json`。\n'
    text+='- 训练均为 2231 steps、10,001,108 个非 padding token；所有共享参数按 seed 完整对齐。最终源文件哈希保持不变。\n'
    text+='- 结果与日志在 `physics_lm/block_runs_proj/`，不含 `.pt`。源码快照、计划、环境、启动配置及验收 JSON 在 `evidence/`。\n'
    text+='- 本轮独立目录 `/gpfs/users/linlizhi/komp-proj-swa-20261101`；16 个自建作业 alias 为 `komp-proj-swa-20261101-r1` 至 `-r16`。均记录正常终态、按精确 alias 删除，并重新查询确认消失；证据在 `evidence/final-cleanup/`。\n'
    text+='- 上一轮 coherence 的 r19–r30 已按用户要求取消，保存取消前 Running 状态、删除回执和空查询；没有把取消伪装为训练完成。证据在 `physics_lm/block_runs_coh/evidence/fine-cancel-cleanup/`。未停止 Claude 的作业。\n'
    dest=ROOT/'BLOCK_KOMP_PROJECTION.md';dest.write_text(text)
    assert table in dest.read_text()
    print(dest)


if __name__=='__main__':main()
