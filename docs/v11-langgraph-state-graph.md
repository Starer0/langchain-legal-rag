# V11：用 LangGraph 表达现有 RAG 状态图

## 学习目标

这一版不增加 Agent、不增加工具调用。目标是把原先隐藏在函数调用里的单问题 RAG 流程，改写成明确的 LangGraph 状态图。V11.2 增加了空候选时的条件分支。

```text
START
  ↓
rewrite
  ↓
retrieve
  ├─ 有候选 → rerank → answer → END
  └─ 无候选 → no_evidence → END
```

## State 是什么

图中每个节点都读取同一份 `RagState`，并且只返回自己新增或更新的字段：

```text
初始：question, history
rewrite 后：retrieval_question, include_guide
retrieve 后：candidates
rerank 后：docs
answer 后：answer, candidates（展示格式）, sources
```

例如 `rewrite` 节点内部仍使用现有的 LangChain `RetrievalQuestionRewriter`；`retrieve` 节点仍使用 Chroma；`rerank` 节点仍使用 SiliconFlow Reranker。这说明 LangGraph 负责步骤和状态，LangChain 组件负责每一步实际做什么。

## 如何运行

默认命令仍使用原来的 LangChain 固定链路：

```powershell
python main.py
```

使用状态图学习路径：

```powershell
python main.py --langgraph
```

也可以查看各节点和模型调用耗时：

```powershell
python main.py --langgraph --profile
```

`--langgraph` 当前只支持单问题流程，不能和 `--decompose` 同时使用。复杂问题拆分、循环重试、人工确认和 checkpoint 可以在理解这个图之后再逐项加入。

## V11.1：实时观察节点更新

```powershell
python main.py --langgraph --trace
```

`--trace` 使用 LangGraph 的 `stream_mode="updates"`。每个节点完成后立即显示该节点返回的更新，而不是等整次问答完成后才显示：

```text
[rewrite] 完成
  retrieval_question → 试用期工资有什么规定？
  include_guide → 不需要办事指南
[retrieve] 完成
  candidates → 8 条候选资料
[rerank] 完成
  docs → 5 条重排资料
[answer] 完成
  answer → 已生成；sources → 5 条来源
```

这些更新由服务合并成最终结果。图只执行一次，模型与 Reranker 调用次数相同。显示的是节点完成事件；回答文字仍在 answer 节点完成后显示，不是 Token 流式输出。

同时观察耗时可使用 `python main.py --langgraph --trace --profile`。如果某个节点失败，不保存不完整的问答历史。

## 等价性验证

有候选时，测试使用同一套伪 Retriever、Reranker、Prompt 和模型，对比图路径与原有单问题链路。两者的回答、候选资料和最终来源必须相同。

## V11.2：条件边与空候选出口

`retrieve` 完成后，条件函数 `route_candidates` 检查 `candidates` 是否为空，并选择下一节点。代码用 `add_conditional_edges` 表达这个分支；条件函数是普通 Python 函数，不调用模型。

候选为空时，`no_evidence` 节点返回固定回答“资料中没有足够依据。”以及空的候选、重排资料和来源列表。改写模型仍然调用一次，但不再调用 Reranker 或回答模型。

可以用相同命令观察：

```powershell
python main.py --langgraph --trace --profile
```

输入一个明确限定到未知法律的例子：

```text
《不存在的示例法》第一条是什么？
```

现有 metadata 过滤会将该法律范围设为未知 ID，活动索引内没有匹配资料，因此得到空候选。应看到 `rewrite`、`retrieve`、`no_evidence`，而不会看到 `rerank` 和 `answer`。

这不是语义相关性判断。一般资料外问题仍可能检索到非空候选；这个分支不会删除它们，也不会根据分数判断是否可答。检索异常同样不会被当作空候选，而是继续报错。

测试同时覆盖普通执行与节点更新流，验证空候选的执行路径、固定回答、历史保存和模型调用次数。有候选路径的既有等价性测试继续通过。
