# V7 复合问题：单查询 Top 5 基线

运行日期：2026-09-23。题集：`evals/composite_question_cases.json`，共 8 题。
每题作为独立会话，走当前完整链路：Query Rewrite → Chroma Top 8 →
SiliconFlow Reranker Top 5 → 回答。启用当前 Metadata Filter 配置。
运行命令：`python run_composite_baseline.py --run-id v7-composite-top5-baseline`。
逐题原始结果保存在本地忽略目录 `evals/results/v7-composite-top5-baseline/`。

## 来源覆盖结果

| 题目 ID | 全部预期法条进入 Chroma Top 8 | 全部预期法条进入最终 Top 5 | 主要缺失 |
| --- | --- | --- | --- |
| `one-year-contract-probation-wage` | 是 | 是 | — |
| `illegal-probation-compensation-limitation` | 是 | 是 | — |
| `probation-dismissal-compensation-limitation` | 否 | 否 | 劳动合同法第三十九、四十六条 |
| `contract-expiry-compensation-limitation` | 是 | 是 | — |
| `holiday-overtime-arbitration-application` | 否 | 否 | 劳动法第四十四条、仲裁法第二十八条 |
| `rest-day-overtime-arbitration-application` | 否 | 否 | 劳动法第四十四条、仲裁法第二十七、二十八条 |
| `probation-wage-employment-status-limitation` | 是 | 是 | — |
| `probation-wage-holiday-overtime-arbitration` | 否 | 否 | 劳动法第四十四条、仲裁法第二十七、二十八条 |

汇总：候选完整覆盖 **4/8**，最终来源完整覆盖 **4/8**。
四道失败题都已在 Chroma Top 8 缺少至少一条必要法条；本批题没有出现
“候选完整但被 Reranker Top 5 截断”的案例。

这为复合问题拆分提供了明确的测试目标：先看分开检索各法律意图能否补齐
Chroma 候选，再看合并后的最终来源是否覆盖全部预期法条。
目前只核验了法条来源覆盖，没有给最终回答的法律准确性打分。
