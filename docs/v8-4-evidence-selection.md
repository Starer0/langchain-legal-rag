# V8.4：复杂问题的证据可答性筛选

复杂问题拆分后，每个子问题仍独立检索和重排，最多支持 6 个子问题。最终证据容量为 `max(配置的 RERANK_TOP_N, 子问题数)`，因此 6 个子问题且配置 Top 5 时会自动扩为 6 条。每个子问题的 Rerank 数和提供给选择器的候选数均为 `max(2, ceil(最终证据容量 / 子问题数))`。例如最终 Top 5 且只拆出两题时，每题重排并提供前 3 条，共 6 条供模型选择。它不生成法律答复，只输出每个子问题是否可答及应保留的资料编号。

当模型输出有效时，最终上下文只使用它选中的资料。标记为不可答的子问题不会占用最终上下文位置。输出无效、矛盾、超时或调用失败时，链路自动退回原有的轮流合并规则。

选择器使用 `EVIDENCE_SELECTOR_MODEL`；未配置时使用 `MODEL_NAME`。性能结果的 `evidence_selection` 阶段记录该调用的次数和耗时，并计入 `model_calls`。

可用以下命令对同一题集比较旧规则与新规则：

```text
python run_composite_baseline.py --run-id <baseline> --decompose --profile --cases-path evals/composite_evidence_selection_cases.json
python run_composite_baseline.py --run-id <selector> --decompose --profile --evidence-selection --cases-path evals/composite_evidence_selection_cases.json
```

结果文件会保留各阶段耗时、最终来源，以及选择器的 `answerable` 判断和选中的资料编号。

## 分组上下文实验

证据选择器目前仅在传入 `--evidence-selection` 时启用；默认的复杂问题链路仍采用原有的轮流合并规则。为避免一个子问题的资料污染另一个子问题的回答，最终回答前会将已入选资料按其检索来源分成子问题组。最终回答提示词要求每一部分只能引用本组资料；某组资料不足时，须明确写出“资料中没有足够依据”。

这种做法不删除某个子问题的资料，也不会增加模型调用次数：无依据资料仍可用于说明该子问题为什么无法从现有资料回答，同时不会成为其他子问题的依据。

默认复杂问题的最终资料预算会随子问题数增加：保持 `RERANK_TOP_N` 的基础额度，同时每个子问题至少分得两条资料，最多 12 条。例如配置 Top 5 时，三题使用 6 条、六题使用 12 条；仍依照轮流、去重、顺延规则取证据。单问题不受影响。
