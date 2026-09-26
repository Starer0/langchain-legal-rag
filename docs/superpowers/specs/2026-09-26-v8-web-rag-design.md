# V8.1：可部署的法律 RAG 网页 MVP

## 目标与边界

在保留命令行入口、V6 History-aware Retrieval 和 V7 三法律知识库的前提下，为项目新增一个可部署的网页聊天界面。网页服务端保管模型密钥，匿名浏览器会话互相隔离；回答在最终 LLM 开始生成后流式显示。第一版面向可信的小范围部署，不包含账号体系或公开互联网滥用防护。

本版包含：FastAPI、原生 HTML/CSS/JavaScript、SQLite 会话持久化、单查询历史感知检索、SSE 格式流式回答、来源与检索问题显示、可选性能信息、自动化测试与依赖清单。

本版不包含：登录注册、权限、复杂问题拆分作为网页默认项、BM25、PostgreSQL、任务队列、答案中断后继续生成、公开部署所需的限流/账单/审计后台。

## 架构

```text
浏览器
  ├─ GET /                 → 静态聊天页，获得匿名 session cookie
  ├─ GET /api/history      → 读取该 session 的历史消息
  └─ POST /api/chat        → text/event-stream
                                 ↓
FastAPI
  ├─ SQLite 会话与消息仓储
  ├─ 共享的 RAG 运行时（启动时加载一次 Chroma、Reranker、模型）
  └─ 现有链路：History-aware Rewrite → Chroma → Reranker → LLM stream
                                 ↓
SQLite（未来由同一仓储接口替换为 PostgreSQL）
```

FastAPI 同时提供 API 与静态页面；不引入 React 构建链。官方 FastAPI 文档支持用 `StaticFiles` 挂载静态资源，且 `StreamingResponse` 可逐块输出 AI 模型生成的字符串。[Static Files](https://fastapi.tiangolo.com/tutorial/static-files/)，[Stream Data](https://fastapi.tiangolo.com/advanced/stream-data/)

服务启动期间只构建一次 RAG 运行时。不能按请求调用当前 `create_conversation_service()`，因为它创建的是单个进程内历史；Web 层必须按 cookie 读取对应 session 的历史，再将其传给共享运行时。

## 会话与数据

浏览器的 `HttpOnly`、`SameSite=Lax` cookie 仅保存一个 32 字节以上的随机 session ID；它不包含聊天内容、账号信息或 API Key。生产环境通过 `Secure` cookie 与 HTTPS 启用；本地 HTTP 开发允许关闭 `Secure`。没有有效 cookie 时，服务端创建新 session。

SQLite 使用两个表：

```text
sessions(id, created_at, last_seen_at)
messages(id, session_id, ordinal, role, content, created_at)
```

`role` 只允许 `user`、`assistant`。完整消息可用于网页回显；历史感知改写只读取最近 `HISTORY_TURNS` 个完整用户/助手消息对，延续 V6 语义。session ID 没有用户身份含义；若 cookie 被清除，会创建新会话。`data/web_rag.sqlite3` 是运行数据，不提交 Git。

一轮问答成功后，用户消息与完整助手回答在同一个短 SQLite 事务中写入。模型、检索、网络或浏览器流失败时，事务不写入本轮任何消息；网页显示错误和“重新发送”入口。这是 MVP 的一致性取舍。

后续增强可为 `messages` 增加 `status`（`streaming`、`complete`、`interrupted`、`failed`），以保留半段内容；但只有 `complete` 的用户/助手对可进入 Rewrite 历史。本版不实现该状态机。

## HTTP 与流式协议

```text
GET  /                  返回聊天页并确保 cookie
GET  /api/history       返回当前 session 的完整可显示消息
POST /api/chat          接收 {"question": "..."}，返回 text/event-stream
```

网页用 `fetch()` 提交 POST，并解析 SSE 文本流；原生 `EventSource` 只适合 GET，不能直接承载本接口的 JSON POST。事件的 `data` 均为 JSON：

```text
event: status   {"stage":"rewrite" | "retrieve" | "rerank" | "answer"}
event: delta    {"text":"..."}
event: done     {"answer":"...", "retrieval_question":"...", "sources":[...], "performance":{...}}
event: error    {"message":"暂时无法完成回答，请稍后重试。"}
```

`status` 在各同步阶段完成前发出，`delta` 仅在最终回答模型开始流式生成后发出；`done` 才携带来源、检索问题和耗时。浏览器在请求进行中禁用发送和清空按钮；同一 session 同时请求时服务端返回 409，避免消息顺序和历史读取发生竞态。

FastAPI 官方 SSE 文档说明 SSE 以 `text/event-stream` 格式传输带 `event`、`data` 等字段的文本块，适用于 AI 聊天流。[Server-Sent Events](https://fastapi.tiangolo.com/tutorial/server-sent-events/)

## RAG 与错误处理

网页默认使用已评测的单查询流程，不自动开启 `--decompose`。每个请求执行：加载完整历史 → Rewrite → Chroma Top 8 → Reranker Top 5（沿用当前本地 `.env`）→ 以原始问题构造回答提示词 → LLM 流式输出。流式实现复用已有 prompt、来源格式和性能计时，不能把聊天历史直接作为法律资料。

空问题在进入模型前拒绝；缺少配置、知识库未导入、模型/Reranker 超时等故障只返回安全错误文本，不返回密钥、内部堆栈或其他用户数据。服务端日志仅记录阶段、错误类别和耗时，不记录完整问题、回答或 API Key。

## 页面 MVP

页面包含：项目标题及“法律资料问答，非法律意见”提示、可滚动消息区、输入框、发送按钮、生成状态、助手回答下方的可展开检索问题/来源/耗时，以及网络错误后的重新发送按钮。用户刷新后先从 `/api/history` 恢复已完成历史。来源显示法律名称、法条号和页码，不直接暴露完整内部 metadata。

不做侧栏、用户资料、会话列表、删改历史、主题切换或移动端精细设计；基础窄屏可读性和键盘发送（Enter 发送、Shift+Enter 换行）属于 MVP。

## 工程与部署

新增一个 Web 应用模块、SQLite 仓储模块、共享 Web 会话/RAG 编排模块，以及 `web/` 下的静态资源。CLI 的 `main.py` 继续保留。新增明确的 Python 依赖清单，包含现有 LangChain/Chroma/PDF 依赖及 FastAPI/Uvicorn；不把虚拟环境、`.env`、Chroma 索引或 SQLite 数据库提交 Git。

本地以 Uvicorn 启动。初期部署使用单个应用进程与 SQLite；多 worker、多实例或公开访问之前，迁移到 PostgreSQL，并加入 HTTPS、反向代理、速率限制、会话过期清理、日志脱敏审查和成本保护。FastAPI 官方数据库指南也将 SQLite 作为单文件起点，并建议生产环境按需要使用 PostgreSQL 等数据库服务。[SQL Databases](https://fastapi.tiangolo.com/tutorial/sql-databases/)

## 测试与验收

- 两个独立 cookie 的历史互不泄漏；重建 Web 服务后 SQLite 历史仍可读取。
- 同一 cookie 的完整历史能正确传入 Rewrite，且只取最近配置数量的完整轮次。
- SSE 事件顺序为 `status* → delta* → done`；`done` 的文本等于所有 `delta` 拼接，且包含来源与性能数据。
- 模型/检索异常、流式生成异常、客户端断开模拟均不写入半轮消息；错误信息不泄露内部细节。
- 普通 CLI 与 V1–V8 测试回归保持通过；网页默认不启用复杂问题拆分。
- 浏览器手动验收：流式文字逐段出现、刷新可见完整历史、来源在结束后可展开、不同无痕窗口不共享消息。

完成标志：在本机打开网页，两个独立浏览器会话各自连续问答；回答开始生成时逐段显示，刷新后完整历史仍在，且无会话串话或 API Key 泄露。
