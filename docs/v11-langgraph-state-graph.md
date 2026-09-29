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

## 验证

测试使用同一套伪 Retriever、Reranker、Prompt 和模型，对比图路径与原有单问题链路。两者的回答、候选资料和最终来源必须相同。这验证了本版本改变的是流程表达方式，而不是问答逻辑。
