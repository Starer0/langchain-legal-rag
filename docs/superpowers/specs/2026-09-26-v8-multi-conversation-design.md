# V8.2：多对话 Web RAG 设计

## 目标与边界

在 V8.1 的匿名网页 RAG 上加入 harness 风格的对话侧栏。一个浏览器 session 可创建、切换、重命名及删除多个独立对话；每段对话的消息、History-aware Rewrite 和流式生成状态彼此隔离。

本版包含：会话侧栏、自动标题、手动重命名、确认删除、SQLite 数据迁移、对话级 API 与自动化测试。

本版不包含：账号登录、跨设备同步、对话搜索、归档/恢复、拖拽排序、共享链接、软删除、消息级编辑/删除、跨对话检索或将多个对话合并为一个上下文。

## 数据模型与迁移

数据关系变为：

```text
Session
└─ Conversation (0..n)
   └─ Message (0..n)
```

新增表：

```text
conversations(
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(id),
  title TEXT NOT NULL,
  title_is_custom INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
)

conversation_messages(
  conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  ordinal INTEGER NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
  content TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (conversation_id, ordinal)
)
```

V8.1 的 `messages(session_id, ...)` 保留为旧表，不原地修改。首次打开新版本数据库时，仓储以一个短事务执行迁移：对每个有旧消息的 session 创建一个标题为“已迁移的对话”的 conversation，按原 ordinal 复制消息至 `conversation_messages`，并在 `schema_migrations` 中记录版本号。重复启动不会重复复制；旧表不再被新的读写路径使用。这样已有网页历史不会丢失，且迁移失败时不会发布半套数据。

新对话初始标题为“新对话”、`title_is_custom=0`。首次完整回答落库时，若标题仍非手动设置，则以该轮用户问题压缩空白后截取 24 个字符作为标题，超出部分添加省略号。手动重命名设置 `title_is_custom=1`，后续自动标题永不覆盖它。

`updated_at` 在创建、重命名和成功写入整轮问答时更新。侧栏按 `updated_at DESC` 排序。

## API

所有接口从 `legal_rag_session` cookie 获取 session ID；任何 conversation ID 都必须属于该 session，否则统一返回 404，不能泄露其他匿名 session 的存在。

```text
GET    /api/conversations
POST   /api/conversations                 -> 创建“新对话”
PATCH  /api/conversations/{id}             -> {"title":"..."}
DELETE /api/conversations/{id}             -> 删除该对话及级联消息
GET    /api/conversations/{id}/messages
POST   /api/conversations/{id}/chat        -> text/event-stream
```

`GET /api/conversations` 返回 `{"conversations": [{"id", "title", "updated_at"}, ...]}`。没有任何对话的 session 返回空数组；页面随后调用 POST 创建首段对话，而不是在 GET 中产生写入副作用。

`POST /api/conversations` 返回 201 与新 conversation。重命名时，标题去除首尾空白，空标题或超过 80 个字符返回 422。`DELETE` 成功返回 204；删除是永久操作，由前端确认框保护。最后一段对话被删后服务端不自动生成替代项，前端立即调用创建接口并将新对话设为当前项。

流式接口只接受当前对话的消息作为 Rewrite 历史。对话忙锁从 `session_id` 改为 `conversation_id`：同一对话的并发生成返回 409，不同对话不会争用同一个锁。成功 `done` 前仍以单个 SQLite 事务写入该 conversation 的用户/助手完整对；失败、断开或模型异常不写半轮。

V8.1 的 `/api/history` 和 `/api/chat` 不再由网页调用；本版移除它们，避免“默认会话历史”与显式对话历史并存造成歧义。

## 页面与交互

桌面页面采用左栏 + 主聊天区：

```text
┌ 侧栏 ─────────────────┬ 主聊天区 ─────────────────────┐
│ [+ 新建对话]           │ 当前对话标题                   │
│ 对话列表（当前项高亮） │ 消息历史、来源、状态            │
│   标题  [重命名] [删除]│ 输入框                          │
└────────────────────────┴────────────────────────────────┘
```

- 页面首次加载：读取列表；若为空则创建“新对话”；选中列表首项并加载它的消息。
- 点击项目：切换当前 `conversation_id`、清空旧消息 DOM、读取新对话消息；正在生成时禁用侧栏操作与发送，避免在流尚未完成时切换上下文。
- 新建：创建并立即切换到空对话，输入框聚焦。
- 重命名：项目显示行内输入框；Enter 保存，Escape 取消，失焦保存非空标题。操作控件有文本/aria-label，不能只依赖图标。
- 删除：点击后显示原生 `confirm()`；确认才调用 DELETE。成功后选中删除前列表中的下一项，若没有则创建新对话；取消不改任何数据。
- 窄屏（小于 760px）：侧栏保持可见但变为聊天区上方的可横向滚动对话条；不做抽屉式复杂导航。

延续 V8.1 的安全 Markdown 渲染、来源展示、键盘发送和错误处理。侧栏仅显示标题与操作，不显示消息正文。

## 错误处理与安全

- 前端对所有修改类请求显示明确、可恢复的错误，并保留当前对话状态。
- 空/超长标题、空问题和并发生成由 API 在模型调用前拒绝。
- 删除确认只防误触，不作为权限机制；服务端 conversation→session 归属校验才是隔离边界。
- Cookie、问题、回答及内部异常文本不写入日志；错误响应不泄露数据库或模型细节。
- SQL 全部使用参数绑定；外键开启并用 `ON DELETE CASCADE` 删除消息。

## 测试与验收

- 数据迁移仅运行一次，V8.1 的历史在“已迁移的对话”中仍可读取。
- 两个 session 的列表、消息、重命名和删除完全隔离；猜测他人 conversation ID 返回 404。
- 新建、排序、自动标题、手动标题保护、空/过长标题及删除级联消息均有仓储测试。
- 当前 conversation 的历史（且仅该 conversation 的最近完整轮次）传入 Rewrite；切换对话不会串上下文。
- 流式成功/失败/409 的 V8.1 语义回归保持不变。
- 页面测试覆盖侧栏和聊天资源交付；手动验收覆盖新建、切换、行内重命名、删除确认、刷新恢复与窄屏布局。

完成标志：同一浏览器可维护至少两段独立法律对话，刷新和服务重启后仍可切换；删除一段不会影响另一段；每段追问只引用本段历史。
