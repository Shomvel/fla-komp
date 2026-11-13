# KOMP 2026-11-13 资料目录

本目录保存此次整理所读取的原始源码、配置、数学说明、接受与失败回执，以及最终分数。
文件中的 Running、pending、已完成等状态属于原实验记录；当前综合状态见 [RECENT_RESULTS.md](../../RECENT_RESULTS.md)。

## 目录索引

| 来源目录 | 路径数 |
| --- | ---: |
| [komp-exact-triton-20261105](komp-exact-triton-20261105/) | 28 |
| [komp-exact-opt-20261105](komp-exact-opt-20261105/) | 91 |
| [komp-exact-opt2-20261105](komp-exact-opt2-20261105/) | 119 |
| [komp-gdn-8k-20261105](komp-gdn-8k-20261105/) | 20 |
| [komp-c128-triton-20261101](komp-c128-triton-20261101/) | 106 |
| [exact-komp-efficiency-research-20261106](exact-komp-efficiency-research-20261106/) | 175 |
| [komp-slot-attention-prototype](komp-slot-attention-prototype/) | 2273 |
| [komp-value-decay-triton-20261106-r1](komp-value-decay-triton-20261106-r1/) | 10 |
| [komp-value-decay-triton-20261106-r2](komp-value-decay-triton-20261106-r2/) | 11 |
| [komp-value-decay-native-20261106-r1](komp-value-decay-native-20261106-r1/) | 43 |
| [komp-value-decay-native-20261106-r2](komp-value-decay-native-20261106-r2/) | 44 |
| [komp-kda-options-340m-20c-20261106](komp-kda-options-340m-20c-20261106/) | 69 |
| [komp-kda-options-ff2272-340m-20c-20261106](komp-kda-options-ff2272-340m-20c-20261106/) | 88 |
| [komp-gdn-slot-learned-340m-20c-20261107](komp-gdn-slot-learned-340m-20c-20261107/) | 84 |
| [komp-gdn-slot-learned-123-340m-20c-20261107](komp-gdn-slot-learned-123-340m-20c-20261107/) | 88 |
| [komp-value-decay-block-340m-20c-20261106](komp-value-decay-block-340m-20c-20261106/) | 77 |
| [komp-value-decay-learned-123-340m-20c-20261107](komp-value-decay-learned-123-340m-20c-20261107/) | 111 |
| [komp-learned123-c128-opt-20261109](komp-learned123-c128-opt-20261109/) | 917 |
| [komp-learned123-1b-50b-20261108](komp-learned123-1b-50b-20261108/) | 122 |
| [komp-learned123-1b-50b-20261108-r1](komp-learned123-1b-50b-20261108-r1/) | 121 |
| [komp-learned123-1b-50b-20261109-mb16](komp-learned123-1b-50b-20261109-mb16/) | 84 |
| [komp-learned123-1b-50b-20261109-mb16-direct](komp-learned123-1b-50b-20261109-mb16-direct/) | 84 |
| [komp-learned123-1b-50b-20261109-opt1](komp-learned123-1b-50b-20261109-opt1/) | 94 |
| [komp-learned123-1b-50b-20261110-resume1](komp-learned123-1b-50b-20261110-resume1/) | 80 |
| [komp-flame-downstream-eval-20261107](komp-flame-downstream-eval-20261107/) | 153 |
| [komp-learned123-1b-downstream-eval-20261110](komp-learned123-1b-downstream-eval-20261110/) | 21 |
| [komp-recurrent-inference-20261107](komp-recurrent-inference-20261107/) | 40 |
| [komp-recurrent-inference-20261107-r2](komp-recurrent-inference-20261107-r2/) | 31 |
| [komp-recurrent-inference-20261107-r3](komp-recurrent-inference-20261107-r3/) | 26 |
| [komp-recurrent-inference-fixtures-20261107](komp-recurrent-inference-fixtures-20261107/) | 5 |
| [komp-niah-analysis-20261107](komp-niah-analysis-20261107/) | 4 |

## 来源与边界

[learned123-v2.patch](learned123-v2.patch) 由重建脚本生成，记录最终 v2 相对已测 baseline 的四个文件差异。该衍生补丁与本索引不属于原始来源文件，原始来源的路径计数以 manifest 为准。

[manifest.json](manifest.json) 记录 5219 个路径、原文件位置、字节数、SHA256 与保留内容的位置。重复内容使用仓库内相对链接；原始字节保留。

快照仅保留所属 KOMP ops/layers/models、相关 rounded_swiglu 模块、实验入口与验收代码。第三方完整依赖树、数据、checkpoint、虚拟环境、编译缓存、原始 profiler trace 与重复监控快照未收录。部分原始 full-source manifest 因此指向本仓库未携带的依赖；这些 manifest 用于追溯原环境。

本目录是研究源码记录。launchers 保留原 GPFS 路径、资源配置和旧环境导入约定；symlink 的 canonical 路径也可能影响 __file__。它们需要相应冻结环境才能运行，不能将本目录直接当作独立可安装包或直接提交 GPU 作业。包内 API 见 [算子文档](../../../../fla/ops/komp/README.md)。

本次没有覆盖已有 [history](../) 中的版本。GPU 验收均为原实验结果；本次验证的内容是归档哈希、链接和表格与回执的一致性。运行 `python research/komp/summarize_results.py --check` 检查。
