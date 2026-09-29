# langchain-legal-rag：学习交接与当前进度

仓库：[Starer0/langchain-legal-rag](https://github.com/Starer0/langchain-legal-rag)

## 当前状态

- 分支：`main`
- 最近提交：`6af5c34 test: add composite evidence budget evaluation`
- 自动测试：`119 passed`
- 当前本地配置：`RERANK_TOP_N=5`
- 运行策略：每完成一个边界清楚的小版本，先测试，再提交并推送到 `main`。

## 项目已经完成什么

### 基础检索链路

```text
用户问题 + 当前对话短期历史
→ History-aware Rewrite
→ 法律范围识别 / Metadata Filter（仅问题明确限定范围时）
→ Chroma 向量检索
→ SiliconFlow Reranker
→ 使用用户原始问题生成回答
→ 来源展示 / SSE 流式输出
```

- 法律法规按条切分，并保留 `law_name`、`law_id`、`article`、`chapter`、版本和效力等 metadata。
- 支持多部法律导入、Metadata Filter、BM25 / 混合检索、Reranker，以及单轮、多轮、多法律、拒答等评测。
- Rewrite 只用于构造独立的检索问题；最终回答面对的是用户原始问题。历史消息不是知识库资料。
- Multi-Query 已做过比较实验，尚未证明稳定净收益，因此不作为默认策略。

### 异构资料接入

已完成劳动争议仲裁办事指南等非法律条文资料的接入。

- 法律法规按条切分；办事指南按标题层级、流程步骤切分。
- 已增加 `document_type`、`source_file`、`section` 等 metadata。
- 已实现指南路由、混合问题检索、异构资料端到端评测和冻结的异构资料保留题集。
- 结论：Chunk 的边界应由文档结构决定。法律条文、办事指南、FAQ、政策解读、表格和扫描 PDF 不应机械复用同一切分器。

相关说明：[docs/v8-3-heterogeneous-ingestion.md](docs/v8-3-heterogeneous-ingestion.md)。

### Web 本地 MVP

- FastAPI + SSE 流式回答。
- SQLite 本地持久化，匿名 Cookie 只用于标识浏览器 Session。
- 多对话侧栏：新建、切换、自动标题、重命名、删除、短期记忆隔离。
- 回答生成时锁定侧栏操作，支持旧历史迁移到多对话模型。

启动：

```powershell
python -m uvicorn web_app:app --host 127.0.0.1 --port 8001
```

访问：`http://127.0.0.1:8001/`

主要文件：

```text
web_app.py                 FastAPI API 与 SSE
web_storage.py             SQLite Session / Conversation 持久化
web_rag.py                 Web 端 RAG 流式封装
web/static/index.html      页面结构
web/static/app.js          前端对话逻辑
web/static/styles.css      页面样式
```

### 复杂问题拆分

复杂问题可走实验性拆分路径：每个子问题独立检索、独立重排，再由一次最终回答汇总。

当前已落实的规则：

- 最终资料按子问题分组；回答模型不得将 A 子问题的资料当作 B 子问题依据。
- 某一组资料不足时，回答应写“资料中没有足够依据”。这保留了范围外或无依据的负向信息，也防止它污染其他子问题。
- 轮流取资料时会去重；重复资料不占名额，后续资料会顺延补位。
- 单问题仍保留 `RERANK_TOP_N=5` 条。
- 多问题默认预算为：至少每个子问题 2 条，最多 12 条。

配置为 Top 5 时的最终资料数：

```text
1 题：5 条
2 题：5 条
3 题：6 条
4 题：8 条
5 题：10 条
6 题：12 条
```

这项策略不增加回答模型或 Reranker 的调用次数，只会让最终回答读取更多上下文。

最近开发集对比使用预设子问题，隔离了“资料预算”本身：

```text
固定 5 条：3 题中 2 题完整命中预期法条；六子问题题无法覆盖全部 6 个预期法条。
动态预算：3 题全部完整命中；最终资料数为 6、8、12 条。
```

该开发集的平均链路阶段耗时约从 26.8 秒变为 29.1 秒；差异主要来自最终回答读取更多资料。模型服务耗时有波动，不能将单次时延视作精确结论。

运行该对比：

```powershell
python run_composite_budget_evaluation.py --run-id <fixed-five> --min-docs-per-question 0
python run_composite_budget_evaluation.py --run-id <dynamic-two> --min-docs-per-question 2
```

相关文件：

```text
rag_pipeline_articles.py
rag_app.py
evals/composite_evidence_budget_cases.json
run_composite_budget_evaluation.py
docs/v8-4-evidence-selection.md
```

## 当前尚未解决、但已明确的边界

### 子问题的“无依据”判断

曾尝试额外调用模型，判断某个子问题是否可答并删除其资料。开发和保留题集结果没有显示稳定收益，同时增加了数秒时延。

当前结论是：先保留该资料并严格分组回答。不要仅凭重排绝对分数删除资料；高分资料可能是在说明该事项不属于某指南范围，是回答“资料不足”的有用依据。

### 拆分模型稳定性

已建立 2、3、4、6 意图的开发题集，并分别测量“直接拆分”和“Rewrite 后拆分”。2026-09-29 的两组结果均为 4/4 数量匹配，人工检查未发现明显遗漏或重复，因此本轮不修改拆分提示词。

这还不是全局稳定性的证明：先前存在模型把交织的多意图问题合并为一个检索问题的边界样本。下一轮应补充口语化、交织事实、办事指南混合和资料不足题，继续记录实际拆分数、语义遗漏和最终来源覆盖情况。

## 数据与评测纪律

- 知识库资料与测试题分开保存；评测题不是知识库来源。
- 开发题集可以用于设计与调参。
- 冻结的独立保留题集只用于最终验证，不能根据其结果反复修改策略。
- 每次实验应记录配置、题集、提交版本、来源命中、回答质量和性能，避免只凭单次主观体验做决策。

## 接下来的学习计划

### 近期：扩展复杂问题拆分的边界评测

1. 在已有 2、3、4、6 意图题的基础上，加入口语化、交织事实、办事指南混合和资料不足题。
2. 记录实际子问题数量、重复率、遗漏率和每题来源命中，并人工检查语义覆盖。
3. 仅在稳定性问题明确后，修改拆分提示词或输出校验，再用同一开发题集复测。
4. 规则确定后，仅运行一次独立保留题集做确认。

目标是理解：复杂 RAG 的问题不只在最终检索数量，也在于规划出的检索子问题是否真正独立、完整、可检索。

### 中期：异构资料扩展与知识库工程化

在现有办事指南基础上，再选择一类结构不同的真实资料，例如 FAQ、政策解读、表格资料或扫描 PDF：

```text
分析原始结构
→ 选择对应切分单位
→ 设计 metadata
→ 接入知识库
→ 设计开发题与保留题
→ 验证是否干扰原有法律检索
```

随后逐步学习：

```text
配置管理
文档 ID、去重、增量更新
失效版本隔离
日志、错误处理、超时、重试、限流
模型与 Reranker 降级策略
requirements.txt 或 pyproject.toml
Docker
PostgreSQL
账户、权限与线上密钥管理
```

### 后期：LangGraph 与 Agent

暂不优先。普通法律问答仍适合固定链路：

```text
问题 → 检索 → 重排 → 回答
```

只有出现必须按问题主动分支或调用工具的需求时，再学习 LangGraph，例如日期计算、多库冲突处理、条件工具调用、持久化状态、重试、Streaming 和 Human-in-the-loop。

## 当前不优先做的事

- 继续美化网页或扩展前端交互。
- 立即迁移 PostgreSQL、登录、多用户和生产部署。
- 为速度强行减少 Rewrite。
- 固定启用 Multi-Query 或复杂问题拆分。
- 根据少量题目设置 Rerank 分数阈值。
- 让大模型逐个挑选数百个法律库。
- 为了技术复杂度而改成 Agent / LangGraph。
