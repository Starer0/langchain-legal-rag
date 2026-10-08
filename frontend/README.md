# React 聊天工作区

React + TypeScript + Vite，复用现有 FastAPI 接口与认证、安全 Markdown 模块。

## 构建与使用

在 frontend 目录执行 `npm ci`、`npm run build`。项目根目录本地 `.env` 设置 `WEB_FRONTEND=react` 后启动原 FastAPI 服务，访问 http://127.0.0.1:8001 。

`dist/` 与 `node_modules/` 不入库，新检出项目需要重新构建。选定 React 却缺少构建目录时，服务明确报错，避免误以为新界面已经启用。`WEB_FRONTEND=legacy` 可选择保留的原静态网页。

开发使用 `npm run dev`，Vite 将 `/api` 代理到 8001，浏览器请求走同源代理。不要改变代理的 Host 或放宽后端来源校验。改源码后生产页面需要重新构建；只改 React 界面无需迁移数据库。

## 验证

- `npm test`：组件、状态控制器、流解析。
- `npm run typecheck`：静态类型检查。
- `npm run build`：生产构建。
- `npm run test:e2e`：启动本地构建预览，使用已安装 Chrome 与 HTTP/SSE 夹具，不调用真实模型、不写生产聊天。

来源展示本轮 SSE 返回的最终依据快照，并随助手消息保存。更新后的新回答在刷新和切换对话后仍可恢复来源；旧消息没有快照，不追溯补造。部署新版后端前执行 `python prepare_source_history.py --apply`（先备份数据库，命令在项目根目录执行），否则认证存储启动会明确拒绝缺少来源迁移的结构。

登录凭据不存入浏览器本地存储；权限、会话和对话归属仍由服务器执行。历史来源属于该回答的附件，采用与历史正文相同的对话归属访问规则，不重新检索，也不自动注入追问模型上下文。账号资料范围变更后的历史正文/附件统一撤回规则另行设计。
# 后台生成任务

生产 React 页面通过后台任务接口提交并查询生成进度。刷新或关闭页面不取消生成；同账号重新打开可恢复问题、部分回答和状态。新版后端重启会接管具有检查点支持的任务；旧版任务仍标记中断。

首次更新启动前执行 `python prepare_generation_tasks.py --apply`（在仓库根目录；先备份数据库）。仅支持单个 Uvicorn 进程。旧 SSE 接口保留用于旧页面和兼容测试，任务模式关闭该提交入口，避免两套执行同时修改历史。详见 `docs/v12-12-refresh-generation.md`。

新版生产部署还需备份后执行 `python prepare_web_recovery.py --apply`。新版任务在后端重启后从图检查点恢复；未完成的回答节点重新生成并替换部分内容，已完成图直接补保存。最多自动恢复3次，普通执行失败不自动重试；旧任务不补造恢复状态。详见 `docs/v12-14-web-recovery.md`。


## 对话摘要与后台记忆

生产更新前备份并显式执行`prepare_conversation_context.py --apply`与`prepare_memory_reflection.py --apply`，服务启动不会自动迁移。设置中自动积累默认关闭，需同时启用使用记忆。120秒空闲计时由服务器持久任务管理，不依赖前端轮询。详见docs/v12-16-conversation-context.md与docs/v12-17-memory-reflection.md；单后端实例。
