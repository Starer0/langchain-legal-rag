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

来源仅展示本轮 SSE 返回的附件；历史 API 当前只返回角色和正文，刷新不恢复来源附件。登录凭据不存入浏览器本地存储；权限、会话和对话归属仍由服务器执行。
