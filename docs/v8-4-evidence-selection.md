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
