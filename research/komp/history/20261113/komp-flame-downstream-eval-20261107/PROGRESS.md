# KOMP / KDA 下游评测：全量完成并核验通过

采集时间（UTC）：2026-11-08T19:54:45.069692+00:00

Northjob r4g-s0/s1/s2，各8 H100，评测上限24 GPU，每模型3路固定batch分片。r3主动迁移已停止，41,606条完整样本已保留。

只填写完整分项分数；未完成留空。PhoneBook 每档1000条。

| 模型 | based_drop | based_fda | based_nq_1024 | based_swde | niah_single_1_1024 | niah_single_1_2048 | niah_single_1_4096 | niah_single_1_8192 | niah_single_2_1024 | niah_single_2_2048 | niah_single_2_4096 | niah_single_2_8192 | niah_single_3_1024 | niah_single_3_2048 | niah_single_3_4096 | niah_single_3_8192 | phonebook_100 | phonebook_128 | phonebook_150 | phonebook_200 | phonebook_256 | phonebook_32 | phonebook_64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| kda（6.8B final） | 15.5% | 2.7% | 10.2% | 8.0% | 52.0% | 17.6% | 4.2% | 3.6% | 74.6% | 34.4% | 18.6% | 5.0% | 2.2% | 0.6% | 0.2% | 0.0% | 35.8% | 21.8% | 15.4% | 6.5% | 3.1% | 32.8% | 34.8% |
| block（6.8B final） | 11.5% | 1.7% | 6.2% | 6.7% | 8.6% | 1.0% | 1.4% | 0.4% | 10.0% | 2.2% | 3.4% | 1.6% | 0.0% | 0.6% | 0.6% | 0.0% | 33.2% | 24.1% | 19.9% | 10.5% | 6.9% | 31.7% | 31.9% |
| block123（6.8B final） | 13.9% | 3.8% | 7.0% | 5.8% | 51.6% | 10.0% | 1.6% | 0.6% | 15.4% | 2.6% | 3.4% | 2.2% | 0.0% | 1.6% | 1.2% | 0.0% | 32.6% | 23.7% | 19.1% | 11.4% | 8.0% | 31.5% | 31.3% |
| exact123（6.8B final） | 14.7% | 1.3% | 6.4% | 6.8% | 26.4% | 1.6% | 0.0% | 0.0% | 13.6% | 6.2% | 2.2% | 1.4% | 0.0% | 0.0% | 0.0% | 0.0% | 23.6% | 16.8% | 13.4% | 8.8% | 6.8% | 21.7% | 23.8% |
| learned（6.8B final） | 11.8% | 1.5% | 7.7% | 5.9% | 71.2% | 47.6% | 19.4% | 7.2% | 73.2% | 57.6% | 17.8% | 8.6% | 0.0% | 0.0% | 0.0% | 0.0% | 35.9% | 25.7% | 20.8% | 12.0% | 9.0% | 33.0% | 34.3% |
| learned123（6.8B final） | 14.1% | 2.1% | 8.2% | 8.1% | 91.6% | 83.0% | 44.6% | 55.2% | 94.6% | 78.4% | 26.6% | 3.2% | 0.6% | 0.8% | 0.4% | 0.0% | 35.1% | 25.2% | 20.9% | 10.9% | 7.6% | 32.9% | 34.3% |
| value（6.8B final） | 12.4% | 1.6% | 6.0% | 6.1% | 10.8% | 1.0% | 0.2% | 0.2% | 6.4% | 2.2% | 3.0% | 2.0% | 0.0% | 0.0% | 0.0% | 0.0% | 35.9% | 25.3% | 20.9% | 12.6% | 9.4% | 33.9% | 33.7% |
| exact_partial（3.758B partial） | 10.9% | 0.5% | 5.4% | 5.0% | 3.8% | 0.2% | 0.0% | 0.0% | 20.2% | 6.0% | 3.6% | 2.2% | 0.0% | 0.0% | 0.0% | 0.0% | 6.3% | 2.6% | 2.1% | 0.5% | 0.9% | 5.0% | 6.6% |

全量范围：NIAH single1/2/3 × 1024/2048/4096/8192；PhoneBook32/64/100/128/150/200/256；SWDE/FDA/NQ/DROP，20457条/模型。

PhoneBook 使用历史 single_break 数据，实际 prompt 约80–244 token，不能代表长上下文能力。BASED 按原 HFLM config cap2048 左截断至2000；NIAH cap65536、PhoneBook cap32768。

训练原生 FP32 参数/门控与 BF16 autocast，无 padding、无 cache，和历史模型 BF16 加载数值实现不同，详见 README。原 Exact partial 不与最终6.8B等价排序。

8模型已通过加载、运行配置、真实prompt微生成验收。全量运行已完成，184/184分项均已核验。重新采集：python3 collect.py。

当前作业状态：
- komp340m-downstream-eval-20261108-r4g-s2: Completed 0/1 (job-573d9d01ad05-20261108183240)
- komp340m-downstream-eval-20261108-r4g-s1: Completed 0/1 (job-2f01a6119448-20261108183228)
- komp340m-downstream-eval-20261108-r4g-s0: Completed 0/1 (job-555935e78289-20261108183216)

每worker最新进度：
- block shard0: based_drop / based_drop-00528
- block shard1: based_drop / based_drop-00529
- block shard2: based_drop / based_drop-00527
- block123 shard0: based_drop / based_drop-00528
- block123 shard1: based_drop / based_drop-00529
- block123 shard2: based_drop / based_drop-00527
- exact123 shard0: based_drop / based_drop-00528
- exact123 shard1: based_drop / based_drop-00529
- exact123 shard2: based_drop / based_drop-00527
- exact_partial shard0: based_drop / based_drop-00528
- exact_partial shard1: based_drop / based_drop-00529
- exact_partial shard2: based_drop / based_drop-00527
- kda shard0: based_drop / based_drop-00528
- kda shard1: based_drop / based_drop-00529
- kda shard2: based_drop / based_drop-00527
- learned shard0: based_drop / based_drop-00528
- learned shard1: based_drop / based_drop-00529
- learned shard2: based_drop / based_drop-00527
- learned123 shard0: based_drop / based_drop-00528
- learned123 shard1: based_drop / based_drop-00529
- learned123 shard2: based_drop / based_drop-00527
- value shard0: based_drop / based_drop-00528
- value shard1: based_drop / based_drop-00529
- value shard2: based_drop / based_drop-00527

本次追踪：{"captured_utc": "2026-11-08T19:54:45.069692+00:00", "previous_utc": "2026-11-08T18:59:34.560727+00:00", "count": 163656, "total": 163656, "percent": 100.0, "delta": 1063, "completed_items": 184, "delta_items": 2, "hashed_batches_in_complete_items": 34000, "delta_hashed_batches_in_complete_items": 1060, "jobs": [{"name": "job-573d9d01ad05-20261108183240", "user_def_name": "komp340m-downstream-eval-20261108-r4g-s2", "status": "Completed", "ready_status": "0/1", "namespace": "qiji", "job_creator": "linlizhi", "user_gpu": 8}, {"name": "job-2f01a6119448-20261108183228", "user_def_name": "komp340m-downstream-eval-20261108-r4g-s1", "status": "Completed", "ready_status": "0/1", "namespace": "qiji", "job_creator": "linlizhi", "user_gpu": 8}, {"name": "job-555935e78289-20261108183216", "user_def_name": "komp340m-downstream-eval-20261108-r4g-s0", "status": "Completed", "ready_status": "0/1", "namespace": "qiji", "job_creator": "linlizhi", "user_gpu": 8}], "acceptance_passed": 12, "workers_complete": 12, "failures": [], "integrity_issues": [], "plan_unchanged": true, "new_completed": [{"model": "learned", "case": "based_drop", "count": 2087, "score": 0.11787254432199329}, {"model": "learned123", "case": "based_drop", "count": 2087, "score": 0.1413512218495448}], "unfinished": [], "worker_progress": {"block-s0-progress.json": {"case": "based_drop", "batch_key": "based_drop-00528", "time": 1794139113.9769638, "latest_batch_seconds": 3.2609556149691343}, "block-s1-progress.json": {"case": "based_drop", "batch_key": "based_drop-00529", "time": 1794157432.8311749, "latest_batch_seconds": 0.9549551047384739}, "block-s2-progress.json": {"case": "based_drop", "batch_key": "based_drop-00527", "time": 1794139290.9930463, "latest_batch_seconds": 3.1932125203311443}, "block123-s0-progress.json": {"case": "based_drop", "batch_key": "based_drop-00528", "time": 1794139358.9675045, "latest_batch_seconds": 3.4800029564648867}, "block123-s1-progress.json": {"case": "based_drop", "batch_key": "based_drop-00529", "time": 1794158260.4244146, "latest_batch_seconds": 0.7346259877085686}, "block123-s2-progress.json": {"case": "based_drop", "batch_key": "based_drop-00527", "time": 1794139691.0547943, "latest_batch_seconds": 3.3750384598970413}, "learned-s0-progress.json": {"case": "based_drop", "batch_key": "based_drop-00528", "time": 1794147308.2182605, "latest_batch_seconds": 3.6380869243294}, "learned-s1-progress.json": {"case": "based_drop", "batch_key": "based_drop-00529", "time": 1794165369.1528268, "latest_batch_seconds": 3.729850087314844}, "learned-s2-progress.json": {"case": "based_drop", "batch_key": "based_drop-00527", "time": 1794147664.8412702, "latest_batch_seconds": 3.572256363928318}, "learned123-s0-progress.json": {"case": "based_drop", "batch_key": "based_drop-00528", "time": 1794146830.0147078, "latest_batch_seconds": 3.946521122008562}, "learned123-s1-progress.json": {"case": "based_drop", "batch_key": "based_drop-00529", "time": 1794166275.6784036, "latest_batch_seconds": 3.833855949342251}, "learned123-s2-progress.json": {"case": "based_drop", "batch_key": "based_drop-00527", "time": 1794147686.778299, "latest_batch_seconds": 3.7407975886017084}}}

用户最新要求：已立即切为每小时跟踪eval与1B训练；eval完成核验后仅跟踪1B训练。

最终核验通过：{"status": "passed", "captured_utc": "2026-11-08T19:56:36.757796+00:00", "plan_sha": "2abeb5704dc2e10579a99e5b7aca858818851781d46471bb84c1c517b94aa741", "total_results": 163656, "total_items": 184, "old_artifacts_unchanged": 26820, "old_batches_unchanged": 26731, "complete_batches": 34000, "checkpoint_hashes_verified": 8}
完整分数表：FINAL_RESULTS.md；机器可读分数：FINAL_SCORES.csv。评测停止重复采集，后续每小时仅观测1B训练。

2026-11-09 03:58：eval最终核验已通过，FINAL_RESULTS.md和FINAL_SCORES.csv已生成。automation_update 两次返回 Transport closed，尚未确认自动跟踪名称和prompt精简成功；现有每小时频率无需更改。下次先读取本完成标记，跳过已完成eval采集，重试工具更新为仅1B训练。
