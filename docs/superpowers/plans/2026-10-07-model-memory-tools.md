# Model-driven Memory Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task in the current conversation. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 用模型语义判断与原生tool calling统一前台记忆更新、混合问答及后台记忆回顾，支持开放文本和有证据的画像建议。

**Architecture:** understand替代网页规则分流并复用法律改写；前台/后台模型都调用同一个update_memory工具。工具先准备受限变更，再在现有完成事务中原子提交开放文本条目、来源和回执，保留人工编辑与确认边界。

**Tech Stack:** Existing Python/LangChain tool binding/LangGraph, PostgreSQL/psycopg2, React/TypeScript, unittest/Vitest/Playwright; no Letta SDK or new service.

**Spec:** docs/superpowers/specs/2026-10-07-model-memory-tools-design.md

**Status:** 2026-10-08 已连续完成Task1–7实现、程序验证和本地部署；统一用户验收待进行。替代旧memory-consolidation计划，未委派、提交或推送。实测与边界见 docs/v12-19-model-memory-tools.md；原15例模型误判记录保留，3例定向复测通过，不把片段指标当语义准确率。

## Global Constraints

- 三种风格仅为样例；无固定含义key/value白名单。语义判断由模型完成，出处校验不能证明语义必然正确。
- 前台一次understand响应：恰好1个plan_answer，0或1个update_memory，每次update至多12项；后台一次响应允许0或1个update_memory，无plan_answer。
- 取消新网页的关键词入口门槛，支持混合意图；不将整个仓库旧CLI强行改写。
- 标准路线无第二次确认模型调用，无自动模型重试；前台20秒/后台90秒可配置上限。供应商能力不满足时停止部署，不退回关键词伪装为语义判断。
- 空闲120秒/上下文压缩批处理；后台新原文4000估算token，前文1000；自动写需使用与自动两开关均开启。
- 核心500估算token/1500字符，扩展12000字符/40段/每段1000字符，最多3段/500估算token注入；核心不向量化。
- 推测学习主题至少两个不同用户轮次，仅建议；删除/清空、冲突替换需用户确认，自动不覆盖手动文本。
- 不修改用户数据作为验收，不默认付费模型评测；保留混合工作区，不提交推送，HANDOFF本地忽略。

## 文件职责

新增 `memory_tool_contract.py`（schema/类型）、`memory_tools.py`（验证/准备/提交适配）、`memory_fact_storage.py`（来源与回执）、`prepare_memory_tools.py`（显式迁移）、`web_understanding.py`（前台一次工具调用决策）。

新增测试 `tests/test_memory_tools.py`、`tests/test_memory_fact_storage.py`、`tests/test_web_understanding.py`；夹具 `tests/fixtures/model_memory_cases.json`；评测 `scripts/evaluate_model_memory.py`；说明 `docs/v12-19-model-memory-tools.md`。

修改 `memory_service.py`（预算/向量复用）、`memory_storage.py`（事务绑定/手动编辑）、`memory_reflection.py`（后台工具调用）、`reflection_storage.py`/`reflection_runner.py`（回顾与提交）、`generation_tasks.py`（完成事务/混合答复/回执）、`task_recovery.py`/`recovery_runner.py`（决策恢复和版本）、`web_langgraph.py`/`graph_runtime.py`（understand节点/运行时）、`conversation_context.py`/`context_storage.py`（语义路线元数据及法律摘要过滤）、`rag_app.py`（factory绑定）、`web_app.py`（服务装配/迁移检查）、`memory_api.py`/`reflection_api.py`（来源与确认）。

前端修改 `frontend/src/useMemory.ts`、`frontend/src/MemorySettings.tsx`、`frontend/src/useChat.ts`及相关memory/stream测试；新增 `frontend/e2e/model-memory.spec.ts`。保留已有RAG权限过滤与日志，web_rag/query_rewrite的旧调用者兼容。

### Task 1：工具契约与开放式回归案例

**Files:** Create contract, tools tests, case fixture.

**Interfaces:** 以下类型由memory_tool_contract.py统一定义，owner/revision/epoch不出现在模型工具schema中：

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class SourceRef:
    kind: Literal['current_input', 'message', 'manual_document', 'legacy_document']
    reference: str  # 服务端提供的来源id，不接受自由选择账号
    quote: str

@dataclass(frozen=True)
class MemoryOperation:
    action: Literal['remember','merge','propose_change','request_delete','request_clear']
    layer: Literal['core','extended']
    category: Literal['preference','background','learning_focus']
    basis: Literal['declared','inferred']
    content: str
    sources: tuple[SourceRef, ...]
    target_ids: tuple[str, ...]
    relation: Literal['new','duplicate','complement','conflict']
    delta: str

@dataclass(frozen=True)
class MemoryToolCall:
    provider_call_id: str
    operations: tuple[MemoryOperation, ...]

@dataclass(frozen=True)
class AnswerPlan:
    route: Literal['rag','preference','general','clarify']
    legal_question: str
    retrieval_question: str
    include_guide: bool | None
    reply_plan: str
    memory_request: bool

@dataclass(frozen=True)
class TurnDecision:
    answer: AnswerPlan
    memory_call: MemoryToolCall | None

@dataclass(frozen=True)
class ToolContext:
    owner: str
    scope: Literal['foreground','background']
    execution_id: str
    revision: int
    policy_epoch: int
    input_snapshot: dict

@dataclass(frozen=True)
class PreparedUpdate:
    context: ToolContext
    call: MemoryToolCall
    facts: tuple[dict, ...]
    suggestions: tuple[dict, ...]
    prepared_document: dict
    argument_digest: str

@dataclass(frozen=True)
class ToolResult:
    status: Literal['saved','noop','suggested','blocked','conflict']
    revision: int
    added: int
    merged: int
    suggestion_ids: tuple[str, ...]
    message: str
```

- [x] 写JSON schema：拒绝额外字段，content/delta每项最多1000字符，target_ids至多12，source引用至多12，operations最多12。允许任意主题文字，不列三种风格value；空动作列表只作为no-op。
- [x] 写失败测试：工具名未知/参数缺失/user_id注入/超长拒绝；开放内容如“解释TCP时联系网络排障例子”能够解析；同一句含记忆+法律映射双行为；空后台响应不报错。

```python
def test_open_content_is_not_a_fixed_style_enum(self):
    call = parse_memory_call(self.call_args('解释TCP时联系网络排障例子'))
    self.assertEqual(call.operations[0].content, '解释TCP时联系网络排障例子')
```

测试辅助call_args返回符合上述字段的remember/new/declared/core字典，quote来自同文本fixture；`parse_memory_call(raw: dict) -> MemoryToolCall`由contract实现，纯解析不授权。

- [x] 运行 `D:\conda\python.exe -m unittest discover -s tests -p test_memory_tools.py -v`，先确认契约未实现而失败，再实现schema解析并通过。

### Task 2：PG条目、证据与幂等回执

**Files:** Create migration/fact storage/test; modify startup/memory storage.

**Interfaces:** `prepare_memory_tools(connection, *, apply=False, schema='public') -> dict`；`MemoryFactStore.snapshot(owner: str, cursor) -> dict`；`apply(update: PreparedUpdate, cursor) -> ToolResult`；`sync_manual(owner: str, core: str, extended: str, revision: int, cursor) -> None`；`save_decision(context: ToolContext, decision: TurnDecision, cursor) -> None`；`read_decision(owner: str, task_id: str, cursor) -> TurnDecision | None`。存储不调用模型且不自行commit。

- [x] 先写临时schema检查/重复apply/跨账号/事务失败/回执重放/重启读取测试；旧文本迁移前后相同。
- [x] marker为 `v12_model_memory_tools_v1`，沿用schema_migrations、advisory lock和显式apply模式。创建四表，DDL字段如下，实际用psycopg2.sql.Identifier指定schema：

```sql
CREATE TABLE account_memory_facts (
 user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 id TEXT NOT NULL, layer TEXT NOT NULL, category TEXT NOT NULL,
 content TEXT NOT NULL, basis TEXT NOT NULL, protected BOOLEAN NOT NULL,
 retired BOOLEAN NOT NULL, revision BIGINT NOT NULL,
 PRIMARY KEY(user_id,id)
);
CREATE TABLE account_memory_fact_evidence (
 user_id TEXT NOT NULL, fact_id TEXT NOT NULL, id TEXT NOT NULL,
 source_kind TEXT NOT NULL, reference TEXT, quote TEXT NOT NULL,
 candidate_content TEXT NOT NULL, recorded_at TIMESTAMPTZ NOT NULL,
 PRIMARY KEY(user_id,fact_id,id),
 FOREIGN KEY(user_id,fact_id) REFERENCES account_memory_facts(user_id,id)
 ON DELETE CASCADE
);
CREATE TABLE account_memory_tool_receipts (
 user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 execution_id TEXT NOT NULL, argument_digest TEXT NOT NULL,
 result JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL,
 PRIMARY KEY(user_id,execution_id)
);
CREATE TABLE generation_turn_decisions (
 task_id TEXT PRIMARY KEY REFERENCES generation_tasks(id) ON DELETE CASCADE,
 user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
 schema_version TEXT NOT NULL, decision JSONB NOT NULL,
 input_digest TEXT NOT NULL, user_ordinal BIGINT,
 created_at TIMESTAMPTZ NOT NULL
);
```

给layer/category/basis加与contract一致CHECK，revision>0，source_kind含manual/legacy；provisional前台reference指向持久task当前输入，完成后绑定真实ordinal。证据id由服务器来源+quote摘要生成。已忽略/退休的防重放信息使用现有policy/progress及最小摘要标识，不保留删除的正文。

- [x] 旧文本按段登记受保护legacy，不模型归纳；手动保存同步段落/退休记录/来源/epoch。对话删除清失效定位但保留记忆摘录，账号删除级联清理；来源界面说明此行为。
- [x] 运行 `... -m unittest discover -s tests -p test_memory_fact_storage.py -v`，现有PG配置启用后只操作随机临时schema；skip不算迁移通过。实库不apply。

### Task 3：统一工具准备与提交，去重不丢新增含义

**Files:** Create memory_tools.py; modify memory service/storage/reflection suggestion APIs; tool/storage tests.

**Interfaces:** `MemoryTools.prepare(call: MemoryToolCall, context: ToolContext) -> PreparedUpdate`；`commit(update: PreparedUpdate, cursor) -> ToolResult`。prepare在事务外复用MemoryService.prepare预算/向量；commit重验持久来源、revision、epoch与参数摘要，再使用Task2.apply。

- [x] 写RED案例：未授权source/助手消息/不同owner/虚构target拒绝；off后台blocked，off明确前台remember可保存，off普通偏好不保存；临时偏好无工具动作；protected只能suggest；inferred仅suggest；delete/clear必须确认。
- [x] 实现open-content规则：duplicate保留原条目并去重追加证据/candidate_content；complement保留原条目新增delta；conflict/propose_change写现有建议；new新增。渲染以条目段落连接并调用预算校验，不用固定STYLE模板。

```python
def test_partial_overlap_preserves_new_details(self):
    initial = self.stored_facts('先给结论', '结合具体例子')
    update = self.prepare_again(initial, duplicate='先给结论', delta='分点解释')
    self.assertEqual(self.active_contents(update),
                     ['先给结论', '结合具体例子', '分点解释'])
```

`stored_facts/prepare_again/active_contents`为该测试类夹具：造账号a/id f1,f2、冻结revision1、同用户当前原文“请记住先给结论，再分点解释”，duplicate目标f1、complement delta，按实际MemoryTools.prepare调用；不硬编码被测函数输出。

- [x] commit用execution_id回执保证重放一次；同id不同digest拒绝；模型不能指定execution_id。超预算、并发修改、自动关掉返回blocked/conflict，原记忆不变；savepoint避免记忆异常毁掉问答完成事务。
- [x] 运行tool/fact storage/memory service三组测试；断言证据新增不触发embedding，扩展文字变化才更新，核心永不embedding。

### Task 4：网页前台语义理解与原生工具调用

**Files:** Create web_understanding.py/test; modify web_langgraph.py/graph_runtime.py/rag_app.py/generation_tasks.py and actual恢复模块、图测试。

**Interfaces:** `TurnUnderstanding.decide(question: str, context_input: dict, memory_snapshot: dict) -> TurnDecision`；`decode_turn_response(message) -> TurnDecision`。runtime注入模型和MemoryTools；State仅存decision、pending update可序列化版本、graph_version。

- [x] RED测试使用AIMessage原生tool_calls，恰好plan_answer、可选update_memory；供应商mock拒绝普通文本冒充工具。覆盖未知表达、混合输入、纯法律追问、记忆off、同指令不同说法、临时要求、引述及恶意要求跨账号。

```python
from langchain_core.messages import AIMessage
message = AIMessage(content='', tool_calls=[
 {'name':'plan_answer','id':'p1','args':{
  'route':'rag','legal_question':'试用期最长多久？',
  'retrieval_question':'劳动合同法试用期最长期限',
  'include_guide':False,'reply_plan':'','memory_request':True}},
 {'name':'update_memory','id':'m1','args':fixture_memory_args}
])
# fixture_memory_args用Task1 schema，来源为当前持久用户输入。
decision = decode_turn_response(message)
self.assertEqual(decision.answer.route, 'rag')
self.assertIsNotNone(decision.memory_call)
```

- [x] 使用现有模型provider的bind_tools暴露这两个schema；一个invoke解析所有调用。额外字段/多个更新调用/无plan_answer不写记忆，反馈无法判断，无自动重试。前台20秒上限可配置；先完成预算检查；保留当前用户原文及必要近期原始片段。
- [x] 网页入口改context→understand；rag直接retrieve→rerank→memory selection→answer，复用检索改写不再调用旧rewrite；preference/general/clarify不检索。plan_answer中的资料范围不可接受为权限，沿用服务端retriever/reranker约束。
- [x] prepare返回pending；generation complete内正式commit并关联消息序号。纯记忆确认由ToolResult生成；混合回答不被memory feedback覆盖，done携带独立结果。终止/超时不提交；提交后断连通过持久任务读取真实回执。
- [x] 图版本 `web-rag-model-memory-v1`。understand返回后在任何副作用前通过save_decision冻结工具参数/原始输入摘要，再checkpoint；恢复优先read_decision，校验task/owner/input_digest并复用。不能因provider_call_id变化重复提交。旧graph_version不可用新契约重解释，按原恢复策略给出明确中断反馈。
- [x] 完成事务把decision关联user_ordinal。法律上下文/摘要按持久route过滤纯偏好与记忆命令，混合输入只取legal_question；理解节点和后台回顾仍可以看到授权原文。旧消息无语义元数据时保留旧摘要过滤兼容，不能为迁移批量重跑模型。后台enqueue不再由command_candidate决定，按持久决策与来源范围登记；明确处理的来源不能重复当新增。
- [x] 理解前原文日志默认private；understand后只把legal_question写入现有RAG日志，不把混合消息里的私人偏好或完整工具参数送日志。新增工具审计只记录账号、执行id、动作计数、状态、版本、耗时，不保存密钥或完整记忆摘录。
- [x] 跑 `test_web_understanding.py`、`test_web_langgraph.py`、`test_generation_tasks.py`和现有恢复/权限测试；断言rag无独立rewrite调用、纯偏好1次understand/0次法律模型、mixed不遗漏法律回答、版本冲突不覆盖用户记忆。

### Task 5：后台回顾调用同一个工具

**Files:** Modify memory_reflection.py/reflection_runner.py/reflection_storage.py and tests.

**Interfaces:** `ReflectionService.review(messages: list, existing: dict, context: list) -> MemoryToolCall | None`；其输出传Task3 prepare/commit，不保留另一套append写入实现。

- [x] RED测试：有声明偏好、有多轮学习话题、无可保存内容、只一次话题、助手推断用户职业、同来源重放、新问题延期、两开关关闭、停服重启、手动编辑epoch失效。
- [x] 同一后台invoke完成回顾/关系判断/native update_memory，可不调用；最多1个更新调用，0调用正常完成推进来源范围。无“总结后再分类”模型链。撤下风格text==quote以及“必须我喜欢/我偏好”的保存条件；quote仍需逐字存在，提示要求原子事实、原始来源、区别declared/inferred，不能把案例人物当用户。每项后台新建议至少引用一个本批未失效新来源；前文只补充证据，不能绕过删除/手动编辑失效边界重新授权。
- [x] declared按开关及保护规则处理；inferred学习主题至少两个不同用户turn的原文支持，只建议。后台不能根据模型自报高置信度绕过确认；类别开放，TCP/n8n/法律学习均可作为样例。

```python
def test_inferred_topic_requires_review(self):
    update = self.review_fixture([
        'n8n的Webhook怎么接收POST？', 'n8n如何把结果返回给调用方？'])
    self.assertEqual(update.facts, ())
    self.assertEqual(update.suggestions[0]['basis'], 'inferred')
```

`review_fixture`构造两个user turn、模型固定inferred/learning_focus提案，通过真实validate与MemoryTools.prepare；expected现有无fact，推测候选只入建议。

- [x] 保留120秒/压缩触发、90秒模型等待、4000新来源+1000前文、单worker/回答优先；model call每job attempt==1。新格式旧job作废，不消费未处理来源进度；合法范围在当前epoch按新格式入队。
- [x] 原apply事务按原conv→user→job顺序核验并commit；job/progress/suggestion/receipt/document原子。被忽略候选与退休来源不被旧任务复活；读当前账号pending候选可追加证据，不能跨全历史扫描。
- [x] 跑reflection服务/runner/storage/API四组测试，临时PG验证重启/幂等/并发，现有自然偏好与法律意图回归保留。

### Task 6：界面来源、工具结果与确认

**Files:** Modify memory/reflection APIs, frontend useMemory/MemorySettings/useChat and tests; new e2e.

**Interfaces:** GET追加facts（id/content/layer/category/basis/protected/sources），done记忆结果ToolResult字段；API不接受客户端owner/原话伪证据。suggestion accept仍要求当前revision/epoch，删除/清空同样走已认证确认。

- [x] 写接口越权与过期建议失败测试，手动编辑/开关保存兼容；前端显示pending不称保存，saved才确认，noop显示已存在，suggested显示待确认，blocked/conflict解释未保存而保留法律回答。

```ts
type MemoryToolStatus = 'saved' | 'noop' | 'suggested' | 'blocked' | 'conflict';
type MemoryToolResult = {
  status: MemoryToolStatus; revision: number; added: number; merged: number;
  suggestion_ids: string[]; message: string;
};
```

- [x] 来源展开显示用户原话/归纳文字/声明或推测/时间；用户手动文本不被后台刷新覆盖。学习画像建议可接受或忽略；删除/清空确认可见并聚焦取消。
- [x] 运行frontend工作目录 `npm.cmd test`、`npm.cmd run typecheck`、`npm.cmd run build`，以及 `npm.cmd run test:e2e -- model-memory.spec.ts` 使用既有隔离夹具。桌面/手机覆盖混合回答回执、确认、刷新读取、草稿保护。

### Task 7：真实语义评测与部署验收记录

**Files:** Create evaluate script/docs; update local HANDOFF.

**Interfaces:** 脚本 `--fixtures PATH --output PATH`默认离线，`--model-results PATH`加载另行采集的供应商结果；网络采集必须明确启用，记录模型/参数/时间，不默认费用。

- [x] 夹具至少包含同义/部分重复/多意图/省略追问/临时/否定/引用/第三人/案件/未知背景/多轮主题/删除/容量/越权。每条写期望route、actions、必须保留的含义、不得推断的内容。
- [x] 汇总route错误、明确保存误判、tool调用遗漏、重复、遗漏、无依据新增、人工覆盖、token/calls/时延；未测为null。程序夹具成绩和模型理解成绩分开。同模型设置同样例对比当前规则方案，不编造百分比。

```python
metrics = {
 'lost_facts': len(set(expected_facts) - set(observed_facts)),
 'unsupported_facts': len(set(observed_facts) - set(expected_facts)),
 'duplicate_facts': len(observed_facts) - len(set(observed_facts)),
}
```

- [x] 所有相关Python/前端检查通过后测试schema备份恢复。部署前停止worker、检查任务、备份PG、显式迁移并核验旧文本未改，再启动新图。供应商原生多工具响应能力在隔离环境验证通过才启用；未实测不能宣布支持。
- [x] 用户实际验收已于2026-10-08反馈通过所测新说法、混合问答、后台候选及新对话使用场景；回退需成对旧代码/备份数据库并停写，不能删新表后让旧代码继续处理新任务。用户自行测试新说法、混合问答、后台候选及新对话使用，助手不代修改记忆。
- [x] HANDOFF分别记录计划/代码/程序验证/用户运行/理解范围。简历素材仅记录有实测证据的指标，当前本轮只有文档。

## 自审映射

开放含义/原生tool→Task1/3；来源与幂等→Task2/3；前台语义与混合意图→Task4；后台回顾与画像建议→Task5；用户控制→Task6；成本/真实评测/部署→Task7。取消旧三个维度约束；不声称“工具化即去重正确”或“原话存在即归纳正确”。
