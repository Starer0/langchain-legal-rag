# V8.4：复杂问题的证据可答性筛选

复杂问题拆分后，每个子问题仍独立检索和重排。新增的证据选择器只接收每个子问题重排后的前 2 条资料、原始完整问题和子问题列表；它不生成法律答复，只输出每个子问题是否可答及应保留的资料编号。

当模型输出有效时，最终上下文只使用它选中的资料。标记为不可答的子问题不会占用最终上下文位置。输出无效、矛盾、超时或调用失败时，链路自动退回原有的轮流合并规则。

选择器使用 `EVIDENCE_SELECTOR_MODEL`；未配置时使用 `MODEL_NAME`。性能结果的 `evidence_selection` 阶段记录该调用的次数和耗时，并计入 `model_calls`。

可用以下命令对同一题集比较旧规则与新规则：

```text
python run_composite_baseline.py --run-id <baseline> --decompose --profile --no-evidence-selection --cases-path evals/composite_evidence_selection_cases.json
python run_composite_baseline.py --run-id <selector> --decompose --profile --cases-path evals/composite_evidence_selection_cases.json
```

结果文件会保留各阶段耗时、最终来源，以及选择器的 `answerable` 判断和选中的资料编号。
