# V12：网页后端接入 PostgreSQL

本步学习目标：区分“把旧数据复制过去”和“让程序以后使用新数据库”。旧数据已导入后，本步增加 PostgreSQL 存储实现，并将本机 `.env` 的 `WEB_STORAGE_BACKEND` 设置为 `postgres`。下次启动现有网页后，聊天记录读写 PostgreSQL。

## 数据经过哪里

例如你在网页提问并得到完整回答，现有 FastAPI 接口调用 `save_complete_conversation_turn` 保存问题和回答。以前调用的是 `SQLiteConversationStore`；现在按配置选择 `PostgresConversationStore`。网页发送的接口和消息格式保持一致。

这叫存储适配：两种实现提供相同操作，由后端选择具体实现。对应 `web_app.py` 的 `create_default_app` / `_create_store` 和 `postgres_storage.py`。模型本次上下文仍由后端从当前对话消息组装；使用 PostgreSQL 不代表模型自动读取数据库，也不代表长期记忆已实现。

## 启动与配置

在项目目录沿用现有环境：

```powershell
docker compose -f compose.postgres.yaml up -d --pull never
python -m uvicorn web_app:app --host 127.0.0.1 --port 8001
```

浏览器打开 `http://127.0.0.1:8001/`，沿用原浏览器和地址才能带上原匿名 Cookie。此步保留匿名隔离，尚未加入账号登录，所以换浏览器不会自动获得旧对话。

- `WEB_STORAGE_BACKEND`：`sqlite` 或 `postgres`；未设置时默认 SQLite，当前本机 `.env` 已显式设为 PostgreSQL。
- `WEB_POSTGRES_HOST` / `WEB_POSTGRES_PORT`：默认 `127.0.0.1` / `5433`。
- `WEB_POSTGRES_DATABASE` / `WEB_POSTGRES_USER`：默认 `legal_rag`。
- `WEB_POSTGRES_PASSWORD_FILE`：默认 `.postgres-password`；只读取文件，不将密码放入版本库。
- `WEB_POSTGRES_SCHEMA`：默认 `public`；测试使用独立随机 schema。
- `WEB_DATABASE_PATH`：仅用于 SQLite 分支，默认 `data/web_rag.sqlite3`。
- 进程环境变量优先于 `.env`；网页配置读取不修改全局环境。模型运行时仍沿用原来的配置加载。

显式给 `create_default_app(database_path=...)` 传路径时使用 SQLite，便于原有测试和本地工具使用独立文件。正常网页启动不传这个参数。

PostgreSQL 必须已经完成显式迁移；启动只检查表与迁移标记，不自动建表或重新导入。连接失败、缺少表或配置拼写错误均报错，不自动回退到 SQLite。

## 保存与隔离

所有对话操作继续检查匿名 session 归属。读取时先检查归属；重命名、删除和保存均在数据库条件中限制归属。删除对话同时删除对应消息。

一次完整问答的两条消息与标题更新在同一个事务提交。第二条插入失败时，第一条也回滚。保存前锁定当前对话行，避免多个连接同时分配相同消息序号；这只保证数据库保存完整，不代表整次问答已有跨进程防重或任务恢复。原有网页生成锁仍是进程内锁。

回答生成失败不保存半轮消息。生成期间对话被删除而无法保存时，后端发送带请求编号的错误，不报保存成功。

## 本次核验与待用户验收

2026-10-03 切换前未发现 8001 监听，逐表完整记录比较一致；保留原 SQLite、数据库迁移备份及配置备份。配置切换后，通过新存储实现只读核对全部原会话的对话列表与全部历史消息，原 SQLite 哈希未变，PostgreSQL 完整比较仍一致：15 条 sessions、4 个 conversations、10 条 conversation_messages、4 条旧 messages、1 条 schema_migrations。

配置切换后完整 Python 回归 206 项、Node 前端行为测试 5 项通过，含真实 PostgreSQL 的迁移与存储测试；测试仅在随机 schema 中写入夹具，不写真实聊天，不调用真实模型。独立审查未发现待修正问题。自动验证不代表用户已在浏览器验收或明确理解。

用户验收只需沿用原浏览器查看旧对话，再做一次普通的新问答并刷新，确认新记录保留。无需重做 V11 日期工具练习。首次验收后再记录“用户已运行验证”，不能把“继续”当作验收或理解。

切换后 SQLite 是旧快照，不会同步 PostgreSQL 的新消息。不要直接切回旧库并认为数据最新；如果要回退，先处理新写入数据。迁移脚本默认只读检查在产生新写入后出现 `target_matches=false` 是预期差异；不要因此再次对不同目标执行导入。

正式 React 前端、账号登录、3 天滑动会话、等级资料权限、长期偏好、RAG 召回片段日志及任务恢复仍按后续计划实施。Chroma 和 LangGraph checkpoint 保持原有存储。本步未提交或推送，HANDOFF、密码、配置和备份均只保留本地。
