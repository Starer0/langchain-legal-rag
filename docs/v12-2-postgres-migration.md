# V12：独立 PostgreSQL 与迁移工具准备

本步学习目标：区分“数据库服务已准备”“数据已导入”和“网页已切换”。2026-10-03 用户授权后已备份并完成真实聊天数据导入，完整记录比较一致。本文件记录导入步骤；后续网页存储适配与本地配置切换已完成，见 [网页后端接入 PostgreSQL](v12-3-postgres-web-storage.md)。

## 本地数据库

- 配置：compose.postgres.yaml，使用已有 postgres:15-alpine 镜像。
- 容器：legal-rag-postgres；数据卷：legal-rag-postgres-data。
- 仅绑定本机 127.0.0.1:5433；数据库 / 初始化用户均为 legal_rag。
- 密码在 .postgres-password，由首次准备生成；Git 忽略此文件。通过 Compose secret 挂载，不放入版本库或命令参数。
- 初始化用户仅用于本机开发与迁移，后续部署时单独配置受限的应用数据库用户。

启动已有资源：

```powershell
docker compose -f compose.postgres.yaml up -d --pull never
```

停止可用 docker compose stop；普通停止不会删除数据卷。不要用 down -v 当作日常停止方式。不要修改 Dify 的配套数据库容器。

## 默认只读检查

在项目目录和现有 conda Python 环境中执行：

```powershell
python migrate_sqlite_to_postgres.py
```

仅输出源表数量、目标是否有相应表、数据是否一致。源使用 mode=ro 和一个一致性读取快照；目标连接设置 readonly。不会建表，不输出聊天正文或密码。

当前源和目标均有：15 条匿名 sessions、4 个 conversations、10 条 conversation_messages、4 条旧版 messages、1 条 schema_migrations，target_matches=true。旧 messages 与对话消息分别保留，不能把它们直接合并，否则可能重复。

## 真实导入（已执行，以下为命令说明）

工具支持显式 --apply。本次导入前未发现 8001 网页监听，已用 SQLite backup API 生成备份 data/backups/web_rag-20261003T092857Z-16319849-pre-postgres.sqlite3 并校验；从备份导入后再与原库比较，原文件哈希未改变。此参数不是切换网页数据库的开关。

```powershell
python migrate_sqlite_to_postgres.py --apply
```

此命令只适用于空目标或已经完全匹配的目标：

- 空目标在一个事务内创建五张等价表并导入，最后逐表比较完整记录，而不是只比较行数。
- 目标完全匹配时返回 already_matches，不重复插入。
- 目标已有不同数据或只有部分目标表时拒绝修改，不覆盖、不自动合并。
- 建表 / 插入 / 验证失败会回滚事务。
- 保留 ID、匿名会话归属、消息顺序、时间文本和旧迁移标记；不把旧聊天自动分配给示例账户。

导入完成的当时，网页仍通过 SQLiteConversationStore 读写原 SQLite；这解释了为什么还需要后续适配与配置切换。Chroma 和 LangGraph checkpoint 都不属于本次导入。

如果再次使用旧网页产生新消息，SQLite 和此次导入快照可能不同。后续网页切换前须复核新增数据并确定补迁方案；工具遇到不同目标会拒绝，不会自动覆盖或持续同步。

## 测试与验收区别

```powershell
$env:RUN_POSTGRES_TESTS = '1'
python -m unittest discover -s tests -p test_postgres_migration.py -q
```

测试使用临时 SQLite 和随机 migration_test_ schema，仅清理测试自己建立的 schema。验证检查模式不写入、内容 / 顺序 / 归属保留、重复导入不重复、不同目标不覆盖、失败整体回滚；不导入真实数据。

普通回归默认跳过需要 PostgreSQL 的集成测试；显式设置 RUN_POSTGRES_TESTS=1 时连接本步独立数据库。自动测试通过不等于用户已完成真实迁移或理解验收。

后续步骤已完成 PostgreSQL 网页存储适配、显式切换配置与只读核对，浏览器实际问答验收仍待用户完成；之后才加入账户、三天滑动登录会话及按等级配置的资料范围。
