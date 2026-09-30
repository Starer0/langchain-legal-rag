# V11.3：先观察内存 checkpoint

## 本轮要理解什么

State 是当前执行的数据。Checkpointer 在图的步骤边界保存 State 快照及下一步节点，快照称为 checkpoint。一次直线问答会有多个快照，不是只在最后保存回答。

`thread_id` 把一系列快照归在同一线程下；`checkpoint_id` 标识其中一份快照。`get_state` 读取线程的最新快照，`get_state_history` 读取该线程的历史快照。这些读取不调用模型。

本轮在现有 RAG 图上使用 `InMemorySaver`，尚未使用磁盘存储。退出进程后快照消失，即使重启时给出相同的 thread_id，也不会恢复数据。thread_id 是索引标识，本身不能让内存数据持久化，也不是身份认证凭证。

参考：[LangGraph 持久化文档](https://docs.langchain.com/oss/python/langgraph/persistence)。

## 亲手验证

在项目目录运行：

```powershell
python main.py --langgraph --checkpoint --thread-id practice --trace
```

先输入 `/state`，应显示“当前线程尚无快照”。再输入“试用期最长多久？”。回答完成后输入：

```text
/state
/checkpoints
```

`/state` 显示最新 State 的摘要：原问题、改写问题、候选数量、重排数量、回答、checkpoint_id、step 与 next。

`/checkpoints` 显示历史快照摘要，顺序为最新在前。初始输入和调度也可能产生快照，因此快照数不必等于节点数。查看：

- next 是 retrieve：rewrite 已结束，接下来检索。
- next 是 rerank：候选已写入 State，接下来重排。
- next 是 answer：重排资料已写入 State，接下来生成回答。
- next 是空列表：没有待执行节点。失败后仍需结合错误或任务信息判断，不能单凭快照就认定整轮成功。

输入“那工资呢？”后再次查看 `/state`，应看到第二轮的原问题和独立检索问题。第二轮仍从 rewrite 开始；当前对话历史继续由已有 ConversationService 管理，checkpointer 尚未替代聊天历史管理。每轮输入会清空上轮的回答与检索结果字段，避免旧结果混入当前快照。

输入 q 退出，重新执行同一命令，再输入 `/state`，应再次显示“当前线程尚无快照”。下一轮学习才使用 SQLite 验证重启后读取。

## 实现位置

`build_langgraph_rag(..., checkpointer=...)` 将 saver 传给 `graph.compile`。服务调用图时传入：

```python
config = {"configurable": {"thread_id": "practice"}}
graph.invoke(state, config=config)
```

节点观察路径同样向 `graph.stream` 传入这份 config。服务的快照查看方法调用 get_state 或 get_state_history，并把完整快照转换成适合学习的摘要。

本轮 checkpoint 模式暂不与 --profile 组合。已有性能对象会作为 `_profile` 放入 State，属于运行时计时对象；将它直接序列化到 checkpoint 不合适。后续需要把计时上下文与可保存的业务 State 分开。

## 当前边界与下一步

这一步覆盖快照保存、查看和线程隔离。它尚未实现磁盘持久化、图中断后的恢复、人工确认或长期记忆；也不接入 Web。读取快照不等于恢复执行。

后续顺序：先换 SQLite 并验证跨进程读取，再学习 interrupt/resume。模型和工具的副作用是否会重复，需要在恢复练习时另行验证。
