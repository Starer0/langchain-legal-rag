# 网页问答 LangGraph 接入（已批准小步）

目标：后台任务调用接口不变，改用真实 StateGraph 组织 rewrite、retrieve、rerank、answer 和无权限证据的本地回答分支；复用现有处理组件与权限校验。

将 web_rag.StreamingRagTurn 的处理拆成准备状态、改写、召回、重排、回答/无依据事件。顺序适配器仍用于兼容测试，新 LangGraphStreamingRagTurn 用同一组处理方法构图，借 custom 流只输出原有 status/delta/done 格式，不把改写模型 token 或内部 State 发到网页。

State 每次调用独立保存 question/history/allowed scope/profile/trace 和各步结果；图可复用，不存跨用户历史，不配置 checkpoint，不承诺重启恢复。保留服务器权限裁剪、召回后再次校验、重排防篡改验证、日志范围一致、历史仅完整轮次及授权重写仅用户消息。

完成事件仅在图正常结束后交付后台保存；失败和非正常结束不交付完成。逐段事件在模型尚未结束时可被消费。后台任务、React、数据库表不需要迁移。CLI练习图及工具循环/HITL保持原有学习路径，不接入网页。

验证：新旧输出契约、实际图节点顺序、无证据不调用模型、权限与日志原有测试复用、假模型失败/并发账号状态隔离、后台刷新恢复真实浏览器、完整回归。无真实模型自动调用，无生产测试聊天。
