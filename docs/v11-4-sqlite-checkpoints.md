# V11.4：把图快照保存到 SQLite

## 这次要理解什么

V11.3 的 InMemorySaver 把快照放在进程内存中，退出后丢失。这次通过 SqliteSaver 写入本地数据库，验证相同文件、相同 thread_id 在重启后仍能读取 State。

官方将 SqliteSaver 用于本地开发的文件存储：[LangGraph 持久化说明](https://docs.langchain.com/oss/python/langgraph/persistence)。

## 验证步骤

在项目根目录运行：

```powershell
python main.py --langgraph --checkpoint --checkpoint-db data/langgraph_checkpoints.sqlite3 --thread-id practice --trace
```

1. 输入一个完整问题，例如“试用期最长多久？”。
2. 输入 `/state`：记录 question、answer、checkpoint_id。完成的图通常显示 `next: []`。
3. 输入 `/checkpoints`：查看节点边界快照；中间状态的 next 表示当时下一步节点。
4. 输入 `q`，然后用同一条命令重新启动。
5. 直接输入 `/state`、`/checkpoints`：应读到上次结果，读取操作不调用模型。
6. 退出，把命令中的 thread_id 改成一个未使用的值再启动：应显示“当前线程尚无快照”。

省略 `--checkpoint-db` 仍使用原来的内存模式。`--checkpoint` 需要 `--langgraph`，目前不能与 `--profile` 组合，因为计时对象不属于可保存的业务状态。

## 对应代码

- `rag_app.py`：选择 InMemorySaver 或 SqliteSaver，把 saver 交给图编译，并保持 SQLite 连接存活。
- `langgraph_rag.py`：用 thread_id 查询 State 与历史快照，close() 释放连接。
- `main.py`：传递文件参数；正常退出或发生异常时用 finally 关闭服务资源。

图仍然运行 `rewrite → retrieve → rerank → answer` 或空候选分支，没有增加模型调用。保存快照会增加本地序列化和数据库写入开销；本步骤没有测量真实模型链路的延迟变化。

## 保存状态与恢复聊天的区别

快照保存了图字段，但当前服务的 ChatMessageHistory 仍是内存对象。重启后，`/state` 能显示上次结果；新问题会使用本次启动积累的聊天历史，不会自动把旧 State 转成对话历史。因此重启后不要通过“那工资呢？”来验证聊天记忆已经恢复。

`next: []` 只说明该快照中的图已经结束。本步骤不提供暂停后继续执行、失败恢复或重放命令。后续先学习一次明确的暂停与恢复，再设计对话消息和图线程之间的映射。

数据库保存候选资料与重排资料等中间状态，和 Web 的 `data/web_rag.sqlite3` 消息库用途不同。示例数据库及其 SQLite 附属文件已加入 Git 忽略；自定义路径需要自行保持在本地。CLI thread_id 是本地学习标识，接入多用户 Web 时还需要服务端归属校验。

## 自动验证

使用真实 SqliteSaver 和本地模拟的检索、模型组件，验证关闭后重新打开、节点快照、不同线程隔离、读取不触发检索/模型、重启后历史为空，以及新轮空候选不带入上一轮资料。CLI 测试验证参数约束和异常退出时连接清理；不消耗外部模型额度。
