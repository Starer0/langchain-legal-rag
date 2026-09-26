# V8.1：流式 Web 法律问答 MVP

这个版本保留原有命令行入口，并新增一个适合继续部署演进的网页聊天入口。网页默认走单查询流程：

```text
历史完整轮次 + 当前问题
→ History-aware Rewrite
→ Chroma 检索
→ Reranker
→ 流式回答
```

复杂问题拆分仍是实验能力；网页默认不会启用它。

## 本地启动

先安装依赖，并确保已经构建过本地法规知识库：

```powershell
pip install -r requirements.txt
python main.py --ingest
uvicorn web_app:app --reload
```

随后在浏览器打开终端显示的本地地址，通常是 `http://127.0.0.1:8000`。

## 会话与数据

- 浏览器会收到名为 `legal_rag_session` 的匿名、高熵 Cookie；Cookie 不含对话内容或密钥。
- 对话内容保存在本地 SQLite 文件 `data/web_rag.sqlite3`，它已被 Git 忽略。
- 刷新页面或重启服务后，同一浏览器仍可读取自己的完整历史对话。
- 每个会话同时只允许一个生成请求；成功完成后才会一起写入用户问题和助手回答。模型失败、流中断或浏览器取消时，不会写入半轮对话。
- 历史改写只会使用最近 `HISTORY_TURNS` 轮完整消息；默认值是 4。

## 配置

- `WEB_DATABASE_PATH`：SQLite 路径，默认 `data/web_rag.sqlite3`。
- `WEB_SECURE_COOKIES=true`：部署在 HTTPS 反向代理之后时启用；本地 HTTP 保持默认 `false`。
- 既有的模型、检索、Reranker 和 `HISTORY_TURNS` 环境变量继续从 `.env` 读取。

## 部署边界

这是本地或受控环境的 MVP，暂不应直接暴露到公共互联网。公开部署前至少需要 HTTPS、反向代理、速率限制、会话与成本保护、日志与监控策略。数据层已经通过 `SQLiteConversationStore` 隔离，后续接入 PostgreSQL 时应替换该存储实现，而不改变 RAG 流程和 HTTP/SSE 接口。
