# V7 复合题法条漏选定位

本轮只增加诊断记录，没有更换 Reranker，也没有改变检索、重排、轮流合并或最终回答逻辑。复合题逐题结果新增 `subquestion_reranks`，保留每个子问题对应的检索问题、Reranker Top 5 法条和分数；`metrics.expected_articles_in_reranks` 统计所有子问题重排结果合并后是否覆盖预期法条。后续评测的 summary 也会汇总这一层的命中数。

运行方式：

```text
python run_composite_baseline.py --decompose --run-id <唯一名称>
```

本次实跑 ID：`v7-rerank-stage-diagnostic-20260924`，结果保存在被 Git 忽略的 `evals/results/`。8 道题的完整预期法条覆盖情况：Chroma 候选 8/8，各子问题 Reranker Top 5 合并后 8/8，最终来源 7/8。上一轮无诊断记录的拆分实验最终来源为 6/8；这是两次独立模型运行，不能把 7/8 解释成代码改进了效果。

本次唯一失败题为 `probation-wage-holiday-overtime-arbitration`。《劳动法》第四十四条已进入 Chroma 候选，并在“法定节假日加班费”子问题的重排列表排第 3，但最终五条来源没有它。该子问题排第 1 的《劳动合同法》第八十五条讲欠付加班费的责任，不能替代第四十四条对节假日加班工资标准的规定。最终五条名额被按子问题轮流取文档的规则占满，故这次漏选发生在**合并阶段**。

前次失败的 `probation-dismissal-compensation-limitation` 在本次重排和最终来源中都命中，包括《劳动合同法》第四十六条。由此只能确认当前流水线在不同运行中会出现不同排序与来源组合；尚不能把此前那次失败唯一归因于 Reranker 或合并规则。

下一步若改选择策略，应维持同一 8 题、相同 Top 5 限额，并同时核查每个预期法条在候选、子问题重排和最终来源三个阶段的位置。当前证据不足以支持更换 Reranker 或加入额外大模型筛选调用。
