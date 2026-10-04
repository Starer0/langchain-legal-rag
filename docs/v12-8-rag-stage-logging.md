# V12：统一请求编号与 RAG 阶段日志

承接已有 FastAPI 统一入口、请求日志和资料权限。本步让真实网页问答的内部阶段使用入口产生的 request_id，写到同一个本地 logs/requests.jsonl。没有新部署独立网关，也没有日志查看网页或普通用户读取接口。

request_logging.py 把本次入口 logger 放在 request.state；web_app.py 使用服务器确认的用户 ID、对话 ID、当次允许资料集合和 request_id 创建 RagRequestTrace，显式传入本轮 StreamingRagTurn。每次请求有独立对象，不把身份放在共享 RAG 实例或跨 SSE yield 的线程上下文中。客户端提交的请求编号不能替换服务器编号。

日志事件：

- request_started / request_finished：现有入口开始和真实响应结束，含 HTTP 状态、问答结果与总耗时。
- rag_started：本次原问题、用户、对话和允许资料范围。
- rag_stage_started / rag_stage_finished：rewrite、retrieve、rerank、answer 的阶段开始和结束，含耗时；改写结束含改写问题和指南路由，检索结束含候选片段，重排结束含经过来源核验的片段，回答结束含真正送给回答模型的资料和输出字符数。检索的 performance 还保留向量 / BM25 等已有内部计时。
- rag_stage_skipped：没有可访问候选时跳过重排，没有可用依据时跳过回答模型，明确 reason；不把跳过算作远程调用完成。
- rag_stage_failed：失败阶段、耗时和异常类型；不把提供商异常正文复制到新增阶段日志。既有请求异常处理仍保留原排错信息。
- rag_saved：完整问答已经保存；回答阶段结束不代表数据库保存成功，以此事件区分。

片段包含 Chroma 返回的 ID、所属库、来源文件、版本、法条 / 指南标识、页码、重排分数及当次正文等固定白名单字段。缺标签、非 active 和越权片段不写入本次阶段日志；重排输出先通过权限及候选来源核验，再记录。不会记录 Cookie、Authorization、登录密码、完整历史聊天或模型逐 Token 输出。日志不自动成为模型上下文或长期记忆。

第一版沿用现有约 5 MiB / 5 份备份的文件轮转，单进程使用。这是按容量轮转，不是三天到期或固定天数保留。原问题、改写和每个片段文本最多 32000 字符；每个阶段最多记录 64 片段，超出显式标记 truncated，并保留实际字符数 / 片段总数，不悄悄声称日志完整。普通现有资料在上限内记录完整正文；单条大事件仍可能超过轮转阈值。

本地日志包含问题和可访问资料正文，读取由服务器本地文件权限控制；账号等级并不赋予日志读取权。目前 /logs/requests.jsonl 和 /api/logs 返回 404。以后做日志查看界面时要另行设计管理员权限和文本保存期限；不直接用资料范围更广的 demo_abc 作为管理员。

270 项完整 Python 回归通过（RUN_POSTGRES_TESTS=1）；新增 8 项阶段测试和 1 项真实 PostgreSQL / FastAPI 关联测试覆盖权限正文、阶段计时、异常、跳过、SSE 跨线程推进、不同请求编号隔离、截断标记、日志读取拒绝及保存顺序。独立只读审查无阻塞发现，并独立复跑 8 项阶段测试通过。本步代理未调用远程问答、改写、Embedding 或重排服务。

10 项 Node 行为测试通过。原 8001 服务已重启加载新代码，三个真实示例账号登录 / 身份 / 对话列表 / 退出及退出后拒绝访问检查通过；没有创建真实测试聊天，用户刷新即可使用新日志。

用户验收：网页完成一次新提问后，在项目目录查看日志末尾；找到本次 POST /chat 的 request_id，筛选同一编号，检查改写问题、retrieve.documents、rerank.documents、answer.documents，以及 rag_saved 和 request_finished。此前已经通过的登录与权限隔离练习无需重复；这次核验的是新阶段日志。
