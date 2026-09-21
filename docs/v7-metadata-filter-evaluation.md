# V7 Metadata Filter 评测记录

- 运行 ID：v7-metadata-filter-baseline
- 知识库：labor_law 107 条、labor_contract_law 98 条、labor_arbitration_law 54 条，共 259 条。
- 检索配置：单查询；Chroma Top 8；SiliconFlow Reranker Top 4；同一批 8 道单轮多法律挑战题。
- 对照：全库检索（仅 status=现行有效）与显式法律名/法条号触发的 Metadata Filter。

| 模式 | 候选命中 | 最终来源命中 | 范围覆盖 |
| --- | ---: | ---: | ---: |
| 全库检索 | 8 / 8 | 7 / 8 | 8 / 8 |
| Metadata Filter | 8 / 8 | 7 / 8 | 8 / 8 |

## 结论

当前三法律小型库中，Metadata Filter 没有提高这 8 题的候选或来源命中率。它的价值是对用户明确指定的法律名或法条号收窄候选范围，并避免同号法条混淆；默认不指定范围的问题仍然检索全库。

唯一未达到最终来源命中的题目是“单位少发我的试用期工资，我申请劳动仲裁需要注意什么？”。正确的《劳动合同法》第二十条和《劳动争议调解仲裁法》第二十七条都在 Chroma Top 8 中，但第二十七条排在候选第 5 位，未进入 Reranker Top 4。因此这是重排截断问题，不是法律范围或向量召回问题。

原始逐题 JSON 位于被 Git 忽略的 evals/results/v7-metadata-filter-baseline/，可在本机复核。
