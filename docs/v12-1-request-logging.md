# V12 第一小步：统一请求编号与请求日志

本步学习目标：把网页中的一次请求与后端处理记录关联起来，分清 HTTP 开始响应和流式问答真正结束。实现不代表用户已经运行验证；登录、等级资料权限、RAG 召回日志和异常恢复继续留在后续步骤。

## 一次请求怎样留下记录

FastAPI 的请求中间件在路由之前生成 `request_id`，记录 `request_started`。响应头 `X-Request-ID` 返回同一个编号；客户端提交的同名头不会覆盖后端编号。

普通响应发送结束后记录 `request_finished`。流式问答则等待响应体结束，同时读取问答事件的结果：

- `completed`：正常完成；问答必须先成功保存完整一轮，再发送 done。
- `failed`：后端错误、问答 error、未收到 done 就结束，或保存回答失败。HTTP 状态码仍可能是 200。
- `rejected`：请求被校验或现有归属检查拒绝，例如 422、404、409。
- `disconnected`：连接断开、发送失败或任务取消导致流未完整传完。这只是观测记录，不代表已经实现后台任务恢复。

`status_code` 与 `outcome` 分开保存；日志含 UTC 时间、请求方法、路径、总耗时以及失败详情。入口不记录请求正文、Cookie、Authorization 或查询参数。此步不增加资料片段日志。

## 保存位置

默认日志为 `logs/requests.jsonl`，同时输出到终端。每行一个 JSON 对象；文件达到约 5 MiB 时轮转，最多保留 5 份备份，当前文件另计。`logs/` 已被 Git 忽略。

可用 `WEB_REQUEST_LOG_PATH` 设置目标文件，修改后重启服务。文件路径相对于启动目录；建议在项目目录启动。当前轮转方案用于单个后端进程，后续多个 worker / 多实例部署时再设计集中日志或进程独立文件。

## 对应代码

- `request_logging.py`：请求中间件、JSON Lines 写入和轮转配置。
- `web_app.py`：挂载中间件，标记流式完成与失败，错误事件携带请求编号。
- `web/static/app.js`：从响应头 / 错误事件取得编号，显示在错误提示中；流结束却没有 done 时显示中断提示。

## 用户运行验证（尚未验收）

沿用现有启动命令：

```powershell
python -m uvicorn web_app:app --host 127.0.0.1 --port 8001
```

打开网页发送一次正常问题，在浏览器开发者工具的 Network 中选中对应 chat 请求，查看响应头 `X-Request-ID`。复制编号后，在项目目录查找：

```powershell
Select-String -Path logs/requests.jsonl* -SimpleMatch '<复制的请求编号>'
```

应能找到开始和结束两条记录；结束包含 `outcome=completed` 和 `duration_ms`。在真实失败时，网页提示的编号应能对应日志中的 failed；具体异常只在日志内。无需重复 V11.7 / V11.8 的日期练习，也不要求故意触发付费模型故障。

## 自动验证

```powershell
python -m unittest discover -s tests -q
node --test tests/test_web_request_errors.cjs
```

Python 测试使用本地替代模型行为，覆盖流式成功、异常 / error / 无 done、拒绝访问、请求关联、轮转以及 ASGI 连接断开。Node 测试执行实际 sendQuestion 函数，控制网络和 DOM 边界，验证错误编号、部分回答清理和正常完成；不等同于真实浏览器用户验收。
