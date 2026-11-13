# s1/s2恢复验收：2026-11-08

用户授权继续运行后，仅替换两个中断的8GPU分片。项目WeaverRSI / proj-32；s0、训练任务未修改。

| 分片 | 新作业 | identity | 状态 |
|---|---|---|---|
| s1 | komp340m-downstream-eval-20261108-r4d-s1 | job-b975c4a6b8da-20261108013315 | Running 1/1 |
| s2 | komp340m-downstream-eval-20261108-r4d-s2 | job-ab970a5f32c2-20261108013328 | Running 1/1 |

最终恢复验收：**passed**，时间2026-11-07T17:50:52.176095+00:00。12/12 worker加载、真实prompt推理验收通过，并均已发布新的完整batch。累计新增204条。

两旧作业Aborted且所有实例Failed终态，平台overview未给出原因；旧日志无failure/OOM。旧claim和evidence归档在claim-evidence-archive-s1/s2，旧完整batch保留原位。12,874个旧batch SHA在恢复前和恢复后均一致；冻结runner、sharding、plan、请求manifest和协议审计hash不变。

KDA及旧Value两项已完成，其claims未迁移且未启动worker；每个新作业只启动index 1/2/3/4/5/7。12项独立SSH客户端mkdir/link排他探针通过。

恢复首批有长尾：部分Block worker一度GPU利用率为0，独占Triton目录mtime仍推进，符合编译缓存活动，但未读取其CPU栈确认具体阶段；随后全部产出。旧同worker日志已有最长约1476–1569秒的SWDE batch；本次最终以新batch落盘而非Running或cache变化确认恢复。未修改推理算术、分片、batch形状或缓存策略。

证据：precheck.json、before.json、migration-s1.json、migration-s2.json、recovery-audit.json、jobs-verified.json、gpu-history.json、learned-s1-observation.json。
