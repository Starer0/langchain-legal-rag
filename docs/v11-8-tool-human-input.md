# V11.8：缺少日期参数时暂停、补充与恢复

本步对应 HANDOFF 第五阶段的 Human-in-the-loop。V11.5 的 `/resume` 只表示“现在继续”；这里的 `/resume 2026-10-15` 表示“把这个日期作为缺失参数，再继续”。代码实现与用户理解、运行验收分开记录，完成本步不自动进入 V12。

## 具体例子

问题：“只计算自然日间隔：起始日期是 2026-10-01，结束日期还没提供。”

1. 模型提出 calculate_date_interval，args 只有 start_date，保留工具调用 ID。
2. Python 检查请求是否合法，在 collect_tool_input 节点请求 end_date，然后 interrupt 暂停。工具尚未执行，没有最终回答，聊天历史不写入半轮。
3. `/resume 2026-02-30`：这个日期不存在，继续暂停并提示错误。
4. `/resume 2026-09-30`：早于起始日期，继续暂停并提示错误。
5. `/resume 2026-10-15`：输入有效，补齐参数，Python 执行工具得到 14 天，模型根据结果回答。

自然日运算采用 end_date - start_date，不额外加一天。本工具不判断法律期限、工作日或节假日。

## 为什么这样设计

规划模型看到的 schema 允许省略日期，提示词要求只填写明确提供的日期。真正执行的 calculate_date_interval 仍使用严格 DateIntervalInput：两个日期必填，拒绝多余参数。允许模型提出不完整请求，不代表允许执行不完整工具。

独立 collect_tool_input 节点只做本地校验与 interrupt；模型规划在 tool_plan / tool_continue，运算在 execute_tool。恢复会从被暂停节点开头执行。循环中的已提交输入按 interrupt 顺序重放，非法输入后的下一次 interrupt 等待新补充值；不能随意改动这段调用顺序。

因此恢复会重跑一些便宜的校验，但不会重新规划、检索或重新执行已完成的工具。不能捕获 interrupt 的控制流异常当作普通输入错误。

输入校验包括：字符串类型、完整 YYYY-MM-DD 格式、日历日期真实存在，以及补齐两个日期后检查结束日期不早于起始日期。已提供的非法日期、多余参数、未知工具和并行请求继续走拒绝路径；本步重点是补充缺失参数，没有扩展为通用参数编辑器。

缺两个日期时依次补 start_date 和 end_date；第二个日期非法不会让第一个补充值丢失。循环模式第二次请求缺参数时，也保留第一次结果。等待和非法补充不消耗成功工具执行次数；一次模式最多 1 次，循环模式最多 2 次。

## 对应代码

- date_tools.py：规划 schema、TOOL_HUMAN_INPUT_PROMPT、collect_tool_input，以及实际运算和 ToolMessage。
- langgraph_rag.py：State 字段、条件边、带 value 的 resume、完成后保存完整一轮聊天。
- main.py：开关约束、`/resume 日期` 解析和暂停信息展示。
- rag_app.py：把新开关传入图，沿用已有 checkpointer。
- tests/test_tool_human_input.py：真实图与本地工具测试，只替换外部模型。

开启后的路径为 tool_plan → collect_tool_input → execute_tool → tool_answer。循环模式还可能从 execute_tool → tool_continue → collect_tool_input → execute_tool。没有工具请求时仍走原 RAG。

## 用户运行验证

在项目目录使用平时可用的 Python 环境（本次验证使用 conda 的 python；项目 .venv 依赖不完整）。

```powershell
cd C:\Users\Starerer\Desktop\langchain-legal-rag
python main.py --langgraph --tools --checkpoint --tool-human-input --thread-id hitl-practice --trace
```

本练习使用内存 checkpoint，不加 --profile；可选 SQLite 使用方式沿用既有配置，不要求重做已验收的 V11.5 重启练习。

依次输入：

```text
只计算自然日间隔：起始日期是2026-10-01，结束日期我还没提供，请等我补充，不要猜测。
/state
/resume 2026-02-30
/resume 2026-09-30
/resume 2026-10-15
/state
/resume 2026-10-16
q
```

观察点：

- 首次只有 tool_plan 完成，然后显示等待补 end_date；不能出现日期计算结果或 RAG 拒答。
- 暂停时 `/state` 的 next 为 collect_tool_input，tool_call_count 为 0。
- 两次非法输入都继续等待，不再出现 tool_plan 完成，也不出现 execute_tool 完成。
- 正确补充后出现 collect_tool_input、execute_tool、tool_answer，days 为 14，计数为 1/1。
- 完成后 interrupts 为空。再次 resume 提示没有暂停任务，不重复执行。

如果模型没有提出工具调用，保留 tool_plan 的 trace 并反馈；模型选路与 Python 校验是两个不同环节，不能把 RAG 拒答当成本练习通过。

循环模式在启动命令上增加 --agent-loop，问：

```text
比较两个自然日区间：第一个2026-10-01到2026-10-15，第二个起始2026-11-01，第二个结束日期我还没提供；不要猜测，缺参数请等我补充。
```

缺日期时补 `/resume 2026-11-20`。应有 14 天与 19 天两个工具结果，计数 2/2。模型可能先请求第二个区间，再请求第一个；观察实际 trace，不强制请求顺序。这个练习补验 V11.7 的循环，不代表已有用户验收记录。

自动测试：

```powershell
python -m unittest discover -s tests -q
```

本次代码验证：180 tests，OK；真实模型单次缺参数路径也已观察到暂停 → 非法日期继续暂停 → 正确日期运算 → 最终回答 14 天。用户运行验收仍待反馈。

## 运行后回顾：Agent State 与消息合并

先对照自己的 trace，再解释这些变化。没有用户确认时只记录“解释已提供”，不标记“用户已理解”。

State 是一次图执行共享的数据：question 是原问题；tool_request 是当前待执行 AIMessage；tool_messages 是供工具规划 / 解释模型读取的消息；tool_results 是成功运算结果；tool_call_count 是成功次数；tool_input_answers 是用户有效补充值。checkpoint 另记录节点 next 与 interrupts。

节点返回的是部分更新。例如 rewrite 返回 retrieval_question 和 include_guide，不会删除没有返回的 question。这份 RagState 没有 reducer：同名字段由新值覆盖，没返回的字段保留。

对于列表，代码主动构造完整的新列表：

```python
return {"tool_messages": [*state["tool_messages"], message]}
```

这表示“程序先把旧消息加新消息组成列表，然后图用这个完整列表覆盖旧字段”。如果只返回 `[message]`，旧消息会被覆盖。CLI updates 流里的 result.update(update) 同样是普通字典覆盖，不是消息 reducer；checkpoint 模式最终以图的 snapshot.values 为准。

补参数时，消息顺序是原 HumanMessage → 有效补充 HumanMessage（含 call ID）→ 参数补齐的 AIMessage 工具请求 → ToolMessage。原始问题仍保留，补充消息让模型知道用户已提供日期；AIMessage 的工具调用 ID 与 ToolMessage.tool_call_id 相同。这里替换最后一条待执行请求，不能把一个仍缺参数的请求和另一个新请求同时发给模型。

如果以后改用 Annotated[list, add_messages]，节点通常只返回消息增量；add_messages 会依据消息自身 ID 合并或替换，并非简单列表拼接。消息自身 ID 与工具调用 ID 不是一回事：前者识别消息，后者配对请求和结果。本步没有引入 reducer，也不能仅改类型注解而继续照搬现在的手动列表更新和新轮清空方式。

最后分清四种保存内容：聊天记录是界面/历史用的问答；短期上下文是当前实际送给模型的有限历史和本轮工具消息；执行状态是 State 与 checkpoint；长期记忆是跨对话偏好/事实，本项目此时尚未实现。保存 checkpoint 不等于自动恢复完整聊天历史。

反馈时请给出暂停、非法补充、有效补充后的 trace，并尝试解释：“为什么补错日期不会重新规划？为什么返回一个字段不会删掉其他 State？为什么 ToolMessage 的调用 ID 要匹配？”之后一起回顾，再决定后续学习；不直接进入 V12。

参考：[官方 interrupt、恢复和输入校验](https://docs.langchain.com/oss/python/langgraph/interrupts)、[官方 State 与 reducer](https://docs.langchain.com/oss/python/langgraph/graph-api#reducers)。实际软件异常中断恢复、重连、重试及副作用控制仍留 V12。
