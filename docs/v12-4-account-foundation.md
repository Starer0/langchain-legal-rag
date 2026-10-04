# V12：账户基础

本步学习目标：让服务器拥有“具体用户”的记录，为下一步网页登录提供基础。例如 demo_ab 和 demo_c 有不同的稳定用户 ID；登录完成后，新对话才能按用户 ID 归属。当前还没有网页登录入口，既有网页继续使用匿名 Cookie。

## 已实现并实际准备

2026-10-03 在本机独立 PostgreSQL 的 legal_rag/public 中准备了三个普通账户：

- demo_ab → level_ab → 资料库示例标识 A、B。
- demo_c → level_c → 资料库示例标识 C。
- demo_abc → level_abc → 资料库示例标识 A、B、C。

数据库有 users、account_levels、account_level_libraries 三张新增表。用户 ID、等级 ID 和资料库 ID 分别表示不同对象；等级按明确集合分配，不自动向下继承。拥有 A+B+C 范围不意味着管理员操作权。

这些范围现在是账户配置，尚未接到向量检索、BM25、上下文或引用检查，也未把当前 Chroma 资料划到 A/B/C；此时不能声称已经实现分级 RAG 授权。

## 密码保存

数据库保存密码哈希，初始随机密码只在本地 .demo-accounts.json 中。哈希不是可还原密码的加密：输入密码时，用专门的验证方法判断是否匹配。每次创建哈希使用随机盐，同一密码也不会得到相同的存储字符串。

使用 Argon2id，参数为 19 MiB、迭代 2 次、并行度 1，依据 [OWASP 密码保存建议](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html) 和 [argon2-cffi 官方接口](https://argon2-cffi.readthedocs.io/en/stable/api.html)。当前现有 conda 环境已具备该库，requirements.txt 声明依赖，本轮未安装新包。

创建密码要求至少 12 字符、不超过 1024 个 UTF-8 字节；验证保留空白与 Unicode 原文，错误、超长、非法文本或损坏哈希不能通过。Account 数据对象不包含密码或哈希；未知、错误密码、停用账户均验证失败。日志和命令输出不打印凭据。

.demo-accounts.json、.postgres-password、.env、HANDOFF 和 data/backups/ 被 Git 忽略。初始密码文件供本机学习使用，不能上传、提交或分享；其他凭据路径要由操作者单独保证私密和忽略。登录代码将使用数据库哈希，初始文件不是登录运行时的账户数据库。

## 对应代码与命令

- password_security.py：生成、校验密码哈希。
- account_storage.py：查询账户、验证用户名和密码、读取等级资料范围；尚未作为网页身份入口使用。
- prepare_demo_accounts.py：显式准备账户表与示例数据。
- database_settings.py：复用现有 PostgreSQL 配置，环境变量优先于 .env；web_app.py 配置读取已等价移入此模块。

在项目目录默认只读查看状态：

```powershell
python prepare_demo_accounts.py
```

当前返回 check_only、ready=true、account_count=3、level_count=3。ready 表示账户迁移结构就绪，不代表网页登录和检索授权完成。

`--apply` 是显式写入开关，本次已执行。DDL、等级映射、三账户和 v12_accounts_v1 标记在事务中完成；初始凭据文件是独立文件，存在并不证明数据库提交成功。重复执行复用原文件，保留 ID 和哈希；既有不同密码 / 等级 / 状态、缺少原凭据、部分表或未标记的既有完整表均拒绝覆盖，不自动修复或重置。

## 核验与学习状态

操作前生成并校验 PostgreSQL 备份归档：data/backups/legal-rag-20261003T111258Z-e5bc4be8-pre-accounts.dump。新增账户后对全部原聊天记录做完整比较，sessions、messages、conversations、conversation_messages 均保持一致；当时分别为 16、4、5、12 条。仅 schema_migrations 新增账户标记，未变更旧匿名归属。

代理实际验证三个密码能通过、错误密码失败、各等级返回指定集合。完整 Python 回归 222 项通过，含 16 项本步测试（4 项密码测试、12 项真实 PostgreSQL 临时 schema 测试）。独立审查发现未标记表可被错误采用，已先复现失败、再修正并复查通过。

用户已反馈上一小步的 PostgreSQL 网页验收成功；本步账户数据由代理准备和验证，不记为用户已运行登录或已明确理解。现有网页仍能使用，下一步接入登录、退出、身份查询、3 天滑动会话和新对话的账户归属；之后再落实资料范围。旧匿名记录保留，不自动分配给示例账户。

本次无真实模型调用；代码仍是本地未提交改动，HEAD 7849a14。初始 SQLite 现在是旧快照，新增问答和账户标记会使 SQLite → PostgreSQL 比较不同，不要据此覆盖已在使用的 PostgreSQL 数据。
