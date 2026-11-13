# 续训与评测状态

北京时间2026-11-10 17:26核验：训练已恢复并实际推进。

- 作业：job-35a0f599ed70-20261110172046，Running 4/4，32 H100。
- 32/32 rank恢复模型、Adam、数据位置与RNG状态。
- 从step31745开始，当前step31792，33,336,328,192/50,000,000,000有效tokens。
- 已新增48步；最新loss 2.228664，无非有限指标或节点退出。
- 排除首次冷启动后平均1.670秒/步，剩余纯训练约7.4小时，另加验证/保存开销。
- 自动衔接已在BP持久运行，仅在精确50B、最终checkpoint和退出状态核验通过后启动8H100下游评测。
- 评测范围：NIAH1/2/3 1K–8K、PhoneBook七档、SWDE/FDA/NQ/DROP，23项20,457条。当前未提交评测GPU作业。

恢复证据：evidence/resume-startup-acceptance.json。评测进度见../komp-learned123-1b-downstream-eval-20261110/PROGRESS.md。
