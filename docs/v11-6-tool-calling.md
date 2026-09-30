# V11.6：日期工具与 Tool Calling

## 为什么学这个

前面的 RAG 是固定流程。这一步让模型在一个有限范围内选择下一步：需要简单日期运算时提出工具调用，其他问题继续走 RAG。

Tool 是程序中的可调用函数；Tool Calling 是模型返回工具名称和参数的机制。模型提出请求后，Python 才实际执行。LangChain 的工具提供名称、说明和参数 schema；LangGraph 用节点与条件边组织执行步骤。参考[官方工具说明](https://docs.langchain.com/oss/python/langchain/tools)和[模型工具调用说明](https://docs.langchain.com/oss/python/langchain/models#tool-calling)。

## 运行命令

在项目根目录运行：

```powershell
python main.py --langgraph --tools --trace --profile
```

本次不需要 checkpoint，也不需要暂停参数。普通运行方式没有启用工具选择，成本保持原状。`--tools` 需要 `--langgraph`。

## 第一题：观察真实工具交换

提问：

```text
2026年10月1日到2026年10月15日相隔多少天？
```

应该观察到：

```text
tool_plan → execute_tool → tool_answer → END
```

- tool_plan：模型返回 calculate_date_interval，以及 start_date="2026-10-01"、end_date="2026-10-15" 等参数。
- execute_tool：Python 按日期类型执行相减，返回 days=14。
- tool_answer：模型收到带调用 ID 的 ToolMessage，用中文解释结果。

日期路径不调用检索、改写或重排，最终没有法律资料来源。CLI 单独展示工具输出，便于对照模型答复。该路径通常调用模型两次：一次提出工具请求，一次解释结果；本地函数执行不算模型调用。

## 第二题：普通问题仍走 RAG

再提问：

```text
试用期最长多久？
```

模型应该不提出日期工具调用，流程为：

```text
tool_plan → rewrite → retrieve → rerank → answer → END
```

候选为空时仍进入 no_evidence。工具选择模型返回的普通文字不会作为法律答案；法律回答仍由既有检索流程生成。开启工具学习模式后，正常 RAG 相比默认路径多一次模型调用，因此这项功能默认关闭，尚未据此宣布整体性能改善。

## 工具的规则

本步工具只计算 end_date - start_date 的自然日间隔，不额外加一天。同一天是 0 天；2024-02-28 到 2024-03-01 是 2 天。

输入必须是两个真实、完整的 YYYY-MM-DD 日期，结束日期不能早于起始日期；额外参数、不存在的日期会被代码拒绝。模型仅能请求固定允许列表中的工具，每轮最多一次工具调用，不执行任意代码。错误会显示为工具调用失败，不继续让回答模型解释一个不存在的成功结果。

工具选择提示词要求仅处理纯日期间隔问题。涉及法律期限、工作日、首尾计数规则不明确、日期不完整或同时含其他任务的问题继续走 RAG；它们需要更完整的业务设计。模型是否准确遵守这一选择要求仍需评测，代码层的日期校验不等于法律含义判断。

## 对应代码

- date_tools.py：Pydantic 参数 schema、@tool 日期函数、模型绑定和 ToolMessage 交换。
- langgraph_rag.py：增加工具选择、执行、答复节点和条件边；checkpoint 新轮清空工具字段。
- main.py：--tools 开关与 trace；工具输出与法律来源分别展示。
- performance.py：将 tool_plan 纳入模型调用计数，单独记录本地工具耗时。

## 学到哪一步了

这是一次工具交换和有限分支，尚未实现任意轮数的 Agent 循环、多个工具协作或法律时效计算。之后可以在理解本步骤后继续学习“模型收到工具结果，还要不要再选一个工具”的循环与退出条件。

V11.5 暂停与重启恢复已由用户验证。用户决定将任意关闭窗口、后端崩溃、断线重连和失败恢复的产品设计留到 V12，在真实 React + FastAPI + PostgreSQL 架构中落实。

自动测试使用真实日期工具与 LangGraph，模型返回用本地模拟消息，验证日期与闰年、错误参数、ToolMessage 调用 ID、分支、旧工具字段清理和模型调用次数。真实模型验收以当前配置 API 的运行结果和用户再次运行验证为准。

本版本实际 API 验证：2026-10-01 到 2026-10-15 走 tool_plan → execute_tool → tool_answer，工具返回 14 天，模型调用计数为 2；另一次真实工具选择请求中，“试用期最长多久？”未提出工具调用。这里是小规模链路验收，不是工具选择准确率评测。
