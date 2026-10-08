# Memory Consolidation Implementation Plan

> 已被 `2026-10-07-model-memory-tools.md` 替代；本稿未实施，三个固定风格维度不再限制新方案。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task in the current conversation. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 统一两种记忆新增入口，合并用户明确表达的重复回答风格，保留新增含义及原话证据。

**Architecture:** 用 PostgreSQL 中的含义条目与来源证据维护记忆，现有核心/扩展文本作为兼容投影。模型提出受限候选，纯函数验证与合并，现有完成事务提交版本一致的变更；手动编辑及含糊冲突不被自动覆盖。

**Tech Stack:** Existing Python/unittest, PostgreSQL/psycopg2, LangGraph, existing model/embedding providers, React/TypeScript/Vitest/Playwright.

**Spec:** docs/superpowers/specs/2026-10-07-memory-consolidation-design.md

**Status:** 审阅草案；所有实施步骤未开始。用户要求计划不等于批准数据库迁移或替用户整理旧记忆。

## Global Constraints

- 当前对话逐步实施，保留混合工作区；不提交、不推送；HANDOFF 本地忽略。
- 不操作用户记忆用于验收；自动测试用伪模型及隔离数据，真实模型评测另行安排。
- 自然积累仍空闲约120秒/上下文整理后触发，每批一次整理调用；明确命令仍一次解析调用。
- 核心500估算token/1500字符；扩展12000字符/40段/每段1000字符；最多3段/500估算token检索。
- 核心不向量化，扩展只在内容变化时更新向量；不得静默截断记忆。
- 用户手动段落受保护，无法确认的归纳和冲突使用建议；相同key/value合并证据，不重复显示。
- 不以来源存在宣称归纳必然正确；本阶段仅三个明确风格维度自动规范化，多轮画像另立设计。

## 文件职责与接口

新增：
- `memory_consolidation.py`：数据类型、候选来源检查、纯函数合并和渲染，无数据库/模型调用。
- `memory_fact_storage.py`：含义/证据读取、版本绑定提交、手动文档同步。
- `prepare_memory_consolidation.py`：独立显式迁移，check/apply，与现有迁移风格一致。
- `tests/test_memory_consolidation.py`、`tests/test_memory_fact_storage.py`：纯函数与临时PG验证。
- `tests/fixtures/memory_consolidation_cases.json`：同一批回归样例和期望含义。
- `scripts/evaluate_memory_consolidation.py`：离线夹具比较与可选真实模型结果汇总，不默认调用网络。
- `docs/v12-19-memory-consolidation.md`：部署、行为、验收与评测说明。

修改：
- `memory_service.py`、`memory_reflection.py`：复用已有调用产出候选；命令操作保留授权边界。
- `memory_storage.py`、`generation_tasks.py`、`reflection_storage.py`：在原事务中核验并提交统一变更。
- `memory_api.py`、`reflection_api.py`：返回来源与合并建议；手动编辑继续文本CAS。
- `web_app.py`：绑定存储，检查迁移版本，不偷偷执行DDL。
- `web_langgraph.py`：更新图版本，旧任务按现有恢复兼容策略处理，不能用新校验器解释旧提案。
- `frontend/src/useMemory.ts`、`frontend/src/MemorySettings.tsx`：来源查看与待确认建议，文本编辑界面保留。
- 对应现有 `tests/test_memory_service.py`、`test_memory_reflection.py`、`test_memory_storage.py`、`test_generation_tasks.py`、`test_reflection_storage.py`、`test_web_memory_api.py`、`test_reflection_api.py` 及前端测试。

类型统一在 `memory_consolidation.py` 定义，后续任务不得另造同名结构：

```python
from dataclasses import dataclass
from typing import Literal

Origin = Literal['explicit_command', 'automatic', 'manual_document', 'legacy_document']

@dataclass(frozen=True)
class Evidence:
    id: str
    origin: Origin
    quote: str
    conversation_id: str | None
    ordinal: int | None
    recorded_at: str

@dataclass(frozen=True)
class Fact:
    id: str
    layer: Literal['core', 'extended']
    key: str
    value: str
    text: str
    protected: bool
    retired: bool
    evidence: tuple[Evidence, ...]

@dataclass(frozen=True)
class Candidate:
    layer: Literal['core', 'extended']
    key: str
    value: str
    text: str
    evidence: tuple[Evidence, ...]

@dataclass(frozen=True)
class Consolidation:
    facts: tuple[Fact, ...]
    suggestions: tuple[Candidate, ...]
    core_text: str
    extended_text: str
    added: int
    merged: int
    rejected: int
```

`validate_candidate(candidate: Candidate, sources: dict[str, Evidence]) -> Candidate` 核对来源并拒绝越界；sources只含当前已冻结、已授权来源。`consolidate(facts: tuple[Fact, ...], candidates: tuple[Candidate, ...], *, allow_readd: bool = False) -> Consolidation` 为纯函数。`render(facts: tuple[Fact, ...]) -> tuple[str, str]` 输出核心/扩展文本。来源摘要一般不能由客户端提交。

### Task 1：建立重复与新增含义的验收样例

**Files:** Create fixtures, `tests/test_memory_consolidation.py`, `memory_consolidation.py`.

**Interfaces:** Produces 上述类型、validate_candidate/consolidate/render。

- [ ] 写固定案例：两句截图原话预期三项含义；重放同一证据预期不变；新消息同义表达预期只增加证据；“这次先给结论”、第三人/引述/案件条件不得新增。
- [ ] 先写缺模块时失败的行为测试并运行：

```python
def test_partial_overlap(self):
    first = self.candidates('我喜欢先看结论，再看具体例子', ordinal=1)
    second = self.candidates('请记住：先给结论，再分点解释。', ordinal=3,
                             origin='explicit_command')
    initial = consolidate((), first)
    result = consolidate(initial.facts, second)
    self.assertEqual({(f.key, f.value) for f in result.facts}, {
        ('answer.order', 'conclusion_first'),
        ('answer.layout', 'bullet_explanation'),
        ('answer.support', 'concrete_examples')})
    self.assertEqual(result.core_text, '回答先给结论，再分点解释，并结合具体例子。')
    self.assertEqual(result.merged, 1)
```

测试辅助 `candidates(text, ordinal, origin='automatic')` 使用固定映射构造 Evidence/Candidate，不调用模型；conversation_id='fixture-c1'、recorded_at固定ISO时间，每种含义单独Candidate，共用消息原文证据。

- [ ] 实现枚举白名单、来源逐字检查、每条消息角色/归属在调用边界核验、临时与引述拒绝；同key/value合并证据按id去重。未知key不自动归纳，转建议；退休匹配除明确重新授权外转建议。

```python
STYLE = {
    ('answer.order', 'conclusion_first'): '先给结论',
    ('answer.layout', 'bullet_explanation'): '分点解释',
    ('answer.support', 'concrete_examples'): '结合具体例子',
}
# 合并只按已验证含义进行；不能把整句相似视为所有新增含义都重复。
identity = (candidate.layer, candidate.key, candidate.value)
```

- [ ] 运行 `D:\conda\python.exe -m unittest discover -s tests -p test_memory_consolidation.py -v`，预期所有固定行为通过。分别断言重复、保留例子、保留分点、拒绝无来源、同key冲突建议、手动保护、退休条目不复活及预算异常不改原值。

### Task 2：加入来源存储和旧文本兼容

**Files:** Create migration/storage/tests; modify `memory_storage.py` and startup check in `web_app.py`.

**Interfaces:** `MemoryFactStore.read(owner: str, cursor) -> tuple[Fact, ...]`；`write(owner: str, result: Consolidation, revision: int, cursor) -> None`；`sync_manual(owner: str, core: str, extended: str, revision: int, cursor) -> None`。均复用调用方事务，不自行提交。

- [ ] 临时PG失败测试：未迁移read报明确状态；迁移检查模式不建表；重复apply幂等；写后重启read含同一原话；跨账号不能读取来源；迁移前后旧文本字节相同。
- [ ] 实现 `prepare_consolidation(connection, *, apply=False, schema='public') -> dict`，marker `v12_memory_consolidation_v1`。通过现有schema_migrations/advisory lock模式建立以下表：

```sql
CREATE TABLE account_memory_facts (
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  id TEXT NOT NULL, layer TEXT NOT NULL CHECK(layer IN ('core','extended')),
  key TEXT NOT NULL, value TEXT NOT NULL, text TEXT NOT NULL,
  protected BOOLEAN NOT NULL, retired BOOLEAN NOT NULL,
  revision BIGINT NOT NULL CHECK(revision > 0),
  PRIMARY KEY(user_id,id)
);
CREATE TABLE account_memory_evidence (
  user_id TEXT NOT NULL, fact_id TEXT NOT NULL, id TEXT NOT NULL,
  origin TEXT NOT NULL CHECK(origin IN
    ('explicit_command','automatic','manual_document','legacy_document')),
  quote TEXT NOT NULL, conversation_id TEXT, ordinal BIGINT,
  recorded_at TIMESTAMPTZ NOT NULL,
  PRIMARY KEY(user_id,fact_id,id),
  FOREIGN KEY(user_id,fact_id) REFERENCES account_memory_facts(user_id,id)
    ON DELETE CASCADE
);
```

实际DDL用 `psycopg2.sql.Identifier` 指定schema，不拼接用户文本。聊天证据写入前校验消息属于owner；对话删除同步清空定位而保留摘录，无悬挂定位。建立仅active facts的部分唯一索引 `(user_id,layer,key,value)`；legacy/manual段落key用稳定段落id，不能按整段文字当SQL列名。

- [ ] 旧文档逐段登记受保护legacy证据，不语义重写。手动保存把未变段落沿用、变化段落登记manual、删除段落退休；每次同事务更新现有策略epoch及失效进度。
- [ ] 运行 `D:\conda\python.exe -m unittest discover -s tests -p test_memory_fact_storage.py -v`；启用现有PG测试配置时只写随机临时schema，预期迁移/事务回滚/归属/删除/重启检查通过。未配置PG时不能将skip算通过。

### Task 3：统一明确命令的新增与原有修改授权

**Files:** Modify `memory_service.py`, `generation_tasks.py`, `memory_storage.py`; service/task tests.

**Interfaces:** 命令proposal新增版本化 `consolidation` JSON（类型序列化），继续 expected_revision/prepared；提交端用相同冻结来源重新运行validate_candidate/consolidate，比较投影后写入Task2存储。

- [ ] 先补测试：重复“记住”不增加段落；部分重叠保留新增；来源不存在拒绝；模型提案改其他段落拒绝；明确删除目标含糊要求澄清；版本冲突不写；失败时文本/证据/消息不出现半提交。
- [ ] 在现有一次命令解析调用中加入三个风格维度输出。保留其他明确背景的原文方式和删除/清空授权。只对通过命令授权的新增调用统一consolidate；不让同义合并取消否定约束。

```python
# 模型调用发生在事务外；提交端不信任prepared文本。
candidate = validate_candidate(candidate, frozen_sources)
result = consolidate(current_facts, (candidate,), allow_readd=True)
core, extended = result.core_text, result.extended_text
# 继续通过现有prepare预算/向量校验，并在revision一致时原子写入。
```

- [ ] 在generation complete原事务内同时写文档、条目和证据；执行现有手动写入invalidate，保留原锁顺序。拒绝不匹配的旧proposal格式；图版本改为 `web-rag-memory-consolidation-v1`，旧版本任务按现有恢复策略终结并明确反馈，不能静默重解释。
- [ ] 运行 `D:\conda\python.exe -m unittest discover -s tests -p test_memory_service.py -v` 和 `... -p test_generation_tasks.py -v`，预期旧命令授权和新增合并行为同时通过。

### Task 4：后台整理复用同一个含义合并器

**Files:** Modify `memory_reflection.py`, `reflection_storage.py`; reflection service/storage/runner tests.

**Interfaces:** 使用Task1候选和Task2写入；后台result保留additions/suggestions/skipped统计并增加merged，JSON格式版本为2。

- [ ] 失败测试覆盖后台先存“结论+例子”、随后命令新增“分点”；反向顺序同结果；手动修改后旧任务作废；关自动/关使用不写；失败重试同范围不重复；不同用户的同key不混用。
- [ ] 在现有批次的一次模型调用中输出规范风格候选及原话来源。取消风格候选text必须逐字等于quote的限制；保留quote逐字存在、身份/范围校验和临时/第三人/案件过滤。未支持的背景继续原话检查，开放式合并转建议。

```json
{"version":2,"candidates":[
  {"layer":"core","key":"answer.order","value":"conclusion_first",
   "text":"先给结论","source_ordinals":[1],
   "source_quotes":["我喜欢先看结论，再看具体例子"]}
]}
```

- [ ] 提交前用数据库当前冻结输入重验；在原conv→user→job锁顺序内原子写fact/证据/文本/任务/进度。automatic不调用manual失效逻辑取消自己。旧版本已冻结任务作废，不消费其来源进度，符合当前epoch且尚未处理的范围可按新版本重新入队。
- [ ] 运行三个 `unittest discover`：`test_memory_reflection.py`、`test_reflection_storage.py`、`test_reflection_runner.py`，预期边界通过；伪模型记录每批invoke==1，同义证据不触发核心embedding。

### Task 5：来源查看和用户主导的确认

**Files:** Modify memory/reflection APIs, `useMemory.ts`, `MemorySettings.tsx` and existing frontend/API tests.

**Interfaces:** 现有记忆GET追加 `facts`（id/text/layer/protected/sources），sources只返回本人证据摘录与可用定位；客户端不得通过GET字段提交证据。建议仍带revision/epoch，accept/ignore走现有授权入口。

- [ ] 测试GET越权拒绝、无证据旧文档标历史文本、后台不能改保护段落、接受建议过期拒绝；前端能展开来源，仍可自由编辑、保存、清空。
- [ ] 每层增加“查看来源”的轻量展开区，文案区分用户原话、整理文字、手动编辑和历史文本；普通用户流程不暴露key/value/epoch。建议展示当前内容和建议内容，确认后再写入。

```ts
type MemorySource = {
  origin: 'explicit_command' | 'automatic' | 'manual_document' | 'legacy_document';
  quote: string;
  recorded_at: string;
  conversation_id: string | null;
  ordinal: number | null;
};
type MemoryFactView = {
  id: string; text: string; layer: 'core' | 'extended';
  protected: boolean; sources: MemorySource[];
};
```

- [ ] 使用前端既有 Vitest、类型检查、构建入口验证；桌面/手机Playwright隔离夹具验证来源展开及确认区可见。预期后台刷新不覆盖未保存草稿、不偷改开关。

### Task 6：回归、评测记录与用户验收交接

**Files:** Create evaluation script and docs; update local HANDOFF.

**Interfaces:** 离线脚本 `--fixtures PATH --output PATH` 默认仅运行固定候选；真实模型结果输入用 `--model-results PATH` 读取另行采集的结果，不默认联网。

- [ ] 在固定案例记录 `expected_facets`/`observed_facets`、duplicate_facets、unsupported_facets、lost_facets、manual_overwrites、model_calls、input_tokens、output_tokens、duration_ms。未测量字段null，不能用估算填成实测。

```python
expected = set(case['expected_facets'])
observed = list(case['observed_facets'])
metrics = {
    'duplicate_facets': len(observed) - len(set(observed)),
    'lost_facets': len(expected - set(observed)),
    'unsupported_facets': len(set(observed) - expected),
}
```

- [ ] 用同一批样例回放原方案和新方案，分别报告“程序边界夹具”与“真实模型结果”；不把硬编码期望输入的纯函数成绩宣传为语义模型准确率。
- [ ] 跑全套Python回归及现有前端检查；PG必须隔离schema，检查现有权限检索、意图分流、后台生成与恢复。只有代码再次变化才重跑相关检查。
- [ ] 部署准备：记录旧图版本/迁移标记，执行测试库备份恢复演练；实际部署前停止worker并检查运行任务，备份数据库、显式迁移、验证投影一致性、再启动。新结构写入后回退旧代码会失配，回退需停写并恢复成对的旧代码和数据库备份，不能只删表。
- [ ] 交给用户自行测试两个入口/新对话/删除，助手不代为整理旧记忆。HANDOFF分别记录代码已实现、程序检查、用户验收和已解释/基本理解；计划阶段只记“待实施”。量化结果有证据后再纳入简历素材。

## 审阅结论

设计要求分别对应Task1含义与过滤、Task2来源与手动编辑、Task3命令、Task4后台、Task5查看确认、Task6评测与部署。类型统一；本计划不包含多轮画像、额外在线判断调用或自动整理现有用户测试数据。上述建议需审阅后再开始Task1。
