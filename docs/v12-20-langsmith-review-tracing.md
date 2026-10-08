# V12.20：LangSmith 后台回顾任务关联

沿用现有 LangSmith 项目和开关。V12.19 前台 LangGraph 已记录 understand、answer 等节点及模型子调用，root 的 thread_id 对应服务器生成 task_id；本次不重复添加前台追踪。

后台回顾增加 `memory_review` 父记录，标签 `background_memory`，metadata 包含 job_id、conversation_id、stage 与触发 reason。新版 review 与兼容 extract 都经同一包装执行；原模型调用自然成为子记录。每个回顾尝试仍只有一次模型调用，不新增回顾或自动重试。

父记录 inputs 为空，outputs 仅 model_calls=1；异常只写错误类型。它不复制聊天、记忆正文、原始工具参数、账号身份或异常原文。原模型子记录继续沿用既有 LangSmith 行为，本次没有新增模型正文脱敏机制。父记录范围是模型回顾，不含之后数据库提交，不能把父记录成功当成记忆已保存；最终提交状态以任务/回执为准。

追踪创建或提交异常不跳过、重复模型操作，也不将已得到的回顾结果改成失败；模型本身的异常仍按原流程发布失败状态。关闭 LANGSMITH_TRACING 时不向远端发布新增父记录。

## 验证

本地使用 LangSmith local tracing 和隔离 PostgreSQL，验证父子编号、任务字段、空输入及最小输出；无费用合成记录带 synthetic_validation 标签，已从远端读回确认父子关联。合成子记录名 synthetic_review_probe，类型 chain，不伪装成真实 LLM 调用。第一次读回早于远端索引可见，随后只重读同一记录，没有重复调用。

包含原回顾行为、模型超时不重试、追踪不可用仍执行一次及父记录异常脱敏的回归。最终全量结果与服务更新见本地交接。本轮不调用付费模型，不写真实账号聊天或记忆。

## 查看方式

在现有项目中按 background_memory 标签或 memory_review 名称筛选；展开模型子记录查看耗时与 Token，再用 metadata.job_id 对照本地回顾日志。前台仍查看 LangGraph root，按 thread_id 对照生成任务。

最终包含隔离 PostgreSQL 的 Python 全量516项144.556s通过，回顾专项12项通过；无前端业务改动。
