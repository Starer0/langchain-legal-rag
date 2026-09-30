# V11：用 LangGraph 表达现有 RAG 状态图

## 学习目标

这一版不增加 Agent、不增加工具调用，也不改变检索策略。目标是把原先隐藏在函数调用里的单问题 RAG 流程，改写成明确的 LangGraph 状态图。

```text
START
  ↓
rewrite
  ↓
retrieve
  ↓
rerank
  ↓
answer
  ↓
END
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

`--langgraph` 当前只支持单问题流程，不能和 `--decompose` 同时使用。复杂问题拆分、循环重试、人工确认和 checkpoint 会在理解这个固定图之后再逐项加入。

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

测试使用同一套伪 Retriever、Reranker、Prompt 和模型，对比图路径与原有单问题链路。两者的回答、候选资料和最终来源必须相同。这验证了本版本改变的是流程表达方式，而不是问答逻辑。
