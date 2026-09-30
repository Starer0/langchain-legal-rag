# V11.5：检索后暂停，再继续执行

## 本次学习目标

V11.4 证明了重启后可以读取快照。这次进一步验证：快照中的图尚未结束时，可以从暂停的步骤继续执行。

启用学习开关后，有候选的路径是：

```text
rewrite → retrieve → review（暂停）
                         ↓ /resume
                     rerank → answer → END
```

没有候选则仍然走 `retrieve → no_evidence → END`。暂停用于观察执行流程，不评判资料是否足够支持答案，也不改善检索质量。

## 怎么运行和验证

在项目根目录运行完整命令，注意本次新增的最后一个参数：

```powershell
python main.py --langgraph --checkpoint --checkpoint-db data/langgraph_checkpoints.sqlite3 --thread-id pause-practice --trace --pause-after-retrieve
```

1. 提问“试用期最长多久？”。应该出现 rewrite、retrieve 的完成信息和“已暂停”，随后展示候选摘要，没有 rerank、answer 的完成信息。
2. 输入 `/state`。此时 `next` 应是 `["review"]`，`interrupts` 包含继续提示；`document_count` 为 0、`answer` 为空。
3. 输入 `/resume`。应该出现 review、rerank、answer 的完成信息，生成回答；rewrite、retrieve 不会重新运行。
4. 再输入 `/state`。此时 `next` 和 `interrupts` 为空，回答已保存。
5. 再输入 `/resume`。程序提示没有等待继续的暂停任务，不会重复生成回答。

还可以做一次重启练习：提出新的完整问题，等它暂停后输入 `q`；再用同一条完整命令启动，输入 `/state` 确认仍停在 review，然后输入 `/resume` 继续。相同数据库文件和 thread_id 用于找到这次待完成的任务。

在另一个未使用的 thread_id 下输入 `/resume`，应提示没有暂停任务。暂停期间直接输入新问题也会被阻止，避免覆盖当前任务；想另开一轮可先完成任务，或退出后换线程。

省略 `--pause-after-retrieve` 时，新问题仍自动走完整流程。暂停需要 `--langgraph --checkpoint`；checkpoint 模式目前仍不能与 `--profile` 组合。内存模式也能暂停继续，但退出后丢失，重启练习需要 SQLite。

## 代码怎么做到的

`langgraph_rag.py` 新增独立的 review 节点，它调用 `interrupt()`。LangGraph 保存 State 和待执行任务，并把中断信息交给调用方；CLI 收到暂停结果后回到输入循环。

`/resume` 调用服务的 resume()，先确认当前线程确实有中断，再把 `Command(resume=True)` 传给同一个图。LangGraph 从 review 节点开头重新执行，interrupt() 此时得到继续值，节点完成后进入 rerank。

**恢复时会重跑包含 interrupt 的节点开头。**因此 review 内只有本地提示，改写、检索等外部操作放在前面的独立节点，避免这次恢复重复调用它们。具体行为遵循[官方 interrupts 文档](https://docs.langchain.com/oss/python/langgraph/interrupts)。这不是所有故障场景下的“外部操作恰好执行一次”保证。

服务只在完整回答生成后记录用户问题和回答。暂停或执行失败时不把空回答写入 ChatMessageHistory；重复 `/resume` 在执行前被拒绝。重启后继续成功，会把这条刚完成的问答加入本次进程的聊天历史，其他旧问答不会自动恢复。

## 本步边界

这是人工选择何时继续的最小练习，目前只提供继续，不支持编辑候选、否决任务、回滚、自动失败重试或网页审批。`next` 非空也可能是执行失败；`/resume` 专门处理 `interrupts` 非空的任务，不作为通用错误重试命令。

CLI 的 thread_id 仍是本地学习标识。接入多用户 Web 后，需要服务端验证任务归属，不能仅靠用户填写 thread_id 授权恢复。

自动测试使用真实 LangGraph、InMemorySaver 和 SqliteSaver，加本地模拟模型与检索组件，覆盖暂停前后调用次数、trace、线程隔离、关闭重开、空候选、重复继续、新问题覆盖保护及失败时的消息记录。
