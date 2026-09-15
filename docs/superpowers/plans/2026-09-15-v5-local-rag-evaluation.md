# V5 本地 RAG 评测实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 为当前劳动法 RAG 增加一套与终端问答共用链路的 12 题本地评测工具，保存可审计结果并生成检索、拒答和答案审查汇总。

**架构：** 将 `main.py` 的组件初始化移入 `rag_app.create_rag_chain()`，使终端入口和评测入口真正使用同一条链路。把不依赖网络的指标、结果与汇总放进 `evaluation.py`；`evaluate.py` 只负责 CLI、真实调用和 JSON 文件读写。

**技术栈：** Python、unittest、LangChain、Chroma、python-dotenv、标准库 `argparse`、`json`、`pathlib`、`datetime`。

**设计说明：** `docs/superpowers/specs/2026-09-15-v5-local-rag-evaluation-design.md`；中文说明：`docs/superpowers/specs/2026-09-15-v5-local-rag-evaluation-design-zh.md`。

## 全局约束

- 仅评测 `data/labor_law.pdf`，不构成现实法律意见。
- `main.py` 的问答循环和显示行为保持不变。
- `main.py` 与 `evaluate.py` 必须都调用 `rag_app.create_rag_chain()`，不得复制模型、Prompt、Retriever 或 Reranker 配置。
- 版本控制 `evals/cases.json`、代码与测试；忽略 `evals/results/`、`.env` 和 Chroma 数据库。
- 结果不得含 API Key、完整 PDF 文本或全部环境变量；每条来源文本仅保留前 160 字符。
- 自动测试不访问网络、Chroma、聊天模型或 Reranker API。
- 每次真实运行都保存 `RETRIEVAL_K` 与 `RERANK_TOP_N`。

## 文件结构

```text
rag_app.py                 共享的 create_rag_chain() 工厂。
main.py                    终端入口，仅保留问答循环和显示。
evaluation.py              纯函数：指标、结果、审查、汇总。
evaluate.py                CLI：run、review、summary。
evals/cases.json           12 道题及其 PDF 依据。
evals/results/.gitignore   忽略真实运行的 JSON。
tests/test_rag_app.py      共享工厂接口测试。
tests/test_evaluation.py   评测逻辑与数据集测试。
```

## 用例数据

在 `evals/cases.json` 创建数组。每项均有 `id`、`question`、`expected_articles`、`required_facts`、`answerable`：

1. `probation-limit`：试用期最长是多久？；第二十一条；试用期最长不得超过六个月；可答。
2. `resignation-notice`：劳动者辞职需要提前多少天通知单位？；第三十一条；提前三十日、书面形式通知；可答。
3. `working-hours`：劳动者每天和每周工作时间最长是多少？；第三十六条；每日不超过八小时、平均每周不超过四十四小时；可答。
4. `overtime-pay`：加班工资如何计算？；第四十四条；150%、休息日不能补休 200%、法定休假日 300%；可答。
5. `overtime-limit-and-pay`：单位安排加班最多能延长多久，加班费怎样支付？；第四十一条和第四十四条；一般每日一小时、特殊每日三小时、每月三十六小时、加班工资标准；可答。
6. `maternity-leave`：女职工产假至少有多少天？；第六十二条；不少于九十天；可答。
7. `minimum-wage`：用人单位支付的工资最低要满足什么标准？；第四十八条；不得低于当地最低工资标准；可答。
8. `arbitration-limit`：劳动争议申请仲裁的期限是多久？；第八十二条；自劳动争议发生之日起六十日内；可答。
9. `holiday-overtime-premise`：法定节假日加班只需要支付双倍工资吗？；第四十四条；法定休假日、不低于 300%；可答。
10. `probation-salary-eighty-percent`：试用期工资不得低于转正工资的百分之八十吗？；无预期法条与事实；不可答。
11. `annual-leave-five-days`：连续工作满一年后年休假一定是五天吗？；无预期法条与事实；不可答。
12. `piecework-overtime-formula`：计件工作的加班工资具体应该怎样计算？；无预期法条与事实；不可答。

### Task 1: 抽出共享 RAG 工厂

**文件：**

- Create: `rag_app.py`
- Modify: `main.py:1-101`
- Create: `tests/test_rag_app.py`

**接口：**

- Consumes: `SiliconFlowReranker`、`build_rag_chain()` 与当前环境变量。
- Produces: `create_rag_chain() -> Runnable`。

- [ ] **Step 1: 编写失败测试**

在 `tests/test_rag_app.py` 写入以下测试；所有重型依赖均被 mock：

```python
import unittest
from unittest.mock import patch


class RagAppTests(unittest.TestCase):
    @patch("rag_app.build_rag_chain")
    @patch("rag_app.Chroma")
    @patch("rag_app.OpenAIEmbeddings")
    @patch("rag_app.ChatOpenAI")
    @patch("rag_app.load_dotenv")
    def test_create_rag_chain_returns_pipeline(
        self, load_dotenv, chat_model, embeddings, chroma, build_chain
    ):
        import rag_app

        chroma.return_value.as_retriever.return_value = "retriever"
        build_chain.return_value = "shared-chain"

        self.assertEqual(rag_app.create_rag_chain(), "shared-chain")
        build_chain.assert_called_once()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 确认测试失败**

Run: `python -m unittest tests.test_rag_app -v`

Expected: FAIL，`ModuleNotFoundError: No module named 'rag_app'`。

- [ ] **Step 3: 实现工厂并改造入口**

在 `rag_app.py` 中将原 `main.py` 的 `load_dotenv()`、ChatOpenAI、OpenAIEmbeddings、PDF 加载、`split_by_articles()`、Chroma、Retriever、Reranker、Prompt 和 `build_rag_chain()` 逻辑移到：

```python
def create_rag_chain():
    load_dotenv()
    # 保持 main.py 原有的模型名、默认值、Prompt 和向量库路径。
    return build_rag_chain(retriever, reranker, prompt, model)
```

只允许在函数被调用时初始化组件；模块 import 时不得加载 PDF、调用 API 或打印。将 `main.py` 的初始化段替换为：

```python
from rag_app import create_rag_chain

rag_chain = create_rag_chain()
```

保留 `if __name__ == "__main__":` 下的现有终端循环和打印。

- [ ] **Step 4: 验证并提交**

Run: `python -m unittest discover -s tests -v`

Expected: PASS，既有三项测试和工厂测试均通过。

Run: `python main.py`

Expected: 出现 `请输入问题，输入 q 退出：`；输入 `q` 后退出。

```bash
git add rag_app.py main.py tests/test_rag_app.py
git commit -m "refactor: share RAG chain construction"
```

### Task 2: 实现纯评测逻辑

**文件：**

- Create: `evaluation.py`
- Create: `tests/test_evaluation.py`

**接口：**

```python
REFUSAL_PHRASE = "资料中没有足够依据"
PREVIEW_LENGTH = 160

def build_case_result(case, chain_result, config, run_id, evaluated_at) -> dict: ...
def add_assistant_review(result, verdict, evidence_articles, reason) -> dict: ...
def build_summary(results) -> dict: ...
```

- [ ] **Step 1: 编写失败测试**

在 `tests/test_evaluation.py` 创建可复用 fixture 并断言命中、拒答、截断与汇总：

```python
CASE = {
    "id": "overtime-pay",
    "question": "加班工资如何计算？",
    "expected_articles": ["第四十四条"],
    "required_facts": ["不低于150%"],
    "answerable": True,
}
CHAIN_RESULT = {
    "answer": "第四十四条规定延长工作时间支付不低于150%的工资报酬。",
    "candidates": [{"article": "第四十四条", "pages": [6], "content": "x" * 200}],
    "sources": [{"article": "第四十四条", "pages": [6],
                 "content": "y" * 200, "rerank_score": 0.9}],
}

def test_result_records_hits_and_truncates_content(self):
    result = build_case_result(CASE, CHAIN_RESULT, {"retrieval_k": 8,
        "rerank_top_n": 3}, "run-1", "2026-09-15T10:00:00+00:00")
    self.assertTrue(result["metrics"]["expected_articles_in_candidates"])
    self.assertTrue(result["metrics"]["expected_articles_in_sources"])
    self.assertIsNone(result["assistant_review"])
    self.assertEqual(len(result["candidates"][0]["content_preview"]), 160)
    self.assertNotIn("content", result["candidates"][0])

def test_unsupported_case_detects_refusal(self):
    case = {**CASE, "answerable": False, "expected_articles": [],
            "required_facts": []}
    result = build_case_result(case, {**CHAIN_RESULT,
        "answer": "资料中没有足够依据。"}, {}, "run-1", "time")
    self.assertTrue(result["metrics"]["refusal_phrase_present"])
```

另添加测试：`add_assistant_review()` 对 `complete`、`partial`、`incorrect`、`correct_refusal`、`incorrect_refusal` 以外的值抛出 `ValueError`；`build_summary()` 正确统计两份带审查结果的数据。

- [ ] **Step 2: 确认测试失败**

Run: `python -m unittest tests.test_evaluation -v`

Expected: FAIL，`ModuleNotFoundError: No module named 'evaluation'`。

- [ ] **Step 3: 最小实现**

`build_case_result()` 必须复制用例，提取候选和来源的 `article`、`pages`、可选 `rerank_score` 与 160 字符 `content_preview`；按“全部预期法条是否出现”产生两个命中布尔值；不可答题才计算拒答语；默认 `assistant_review: None`。

`add_assistant_review()` 返回新字典，写入 `verdict`、`evidence_articles` 和 `reason`，不修改传入字典。

`build_summary()` 返回总用例数、候选命中数/比例、最终来源命中数/比例、不可答题数/拒答命中数，以及各 verdict 数量。空列表的比例全部为 `0.0`。

- [ ] **Step 4: 验证并提交**

Run: `python -m unittest discover -s tests -v`

Expected: PASS，且没有网络调用。

```bash
git add evaluation.py tests/test_evaluation.py
git commit -m "feat: add local evaluation metrics"
```

### Task 3: 加入数据集与 CLI

**文件：**

- Create: `evals/cases.json`
- Create: `evals/results/.gitignore`
- Create: `evaluate.py`
- Modify: `tests/test_evaluation.py`

**接口：**

```text
python evaluate.py run --case-id <id> --run-id <id>
python evaluate.py review --run-id <id> --case-id <id> --verdict <verdict> --reason <text> [--evidence-article <article> ...]
python evaluate.py summary --run-id <id>
```

- [ ] **Step 1: 为数据集写失败测试**

增加 `load_cases(path: Path) -> list[dict]` 的测试，断言：正好 12 题、所有 `id` 唯一、首题预期法条是第二十一条、索引 9 的题是不可答题。

- [ ] **Step 2: 确认测试失败**

Run: `python -m unittest tests.test_evaluation -v`

Expected: FAIL，`ModuleNotFoundError: No module named 'evaluate'`。

- [ ] **Step 3: 实现数据集与 CLI**

把“用例数据”章节的 12 个 JSON 对象写入 `evals/cases.json`。写入 `evals/results/.gitignore`：

```gitignore
*
!.gitignore
```

实现：`load_cases()` 读取 UTF-8 JSON 且验证必需字段；`find_case()` 找不到 id 时抛出 `ValueError`；`result_path(run_id, case_id)` 返回 `Path("evals/results") / run_id / f"{case_id}.json"`。

`run`：读取指定题、调用一次 `create_rag_chain()` 和一次 `chain.invoke(question)`；从环境记录参数（默认 8/3）；用 UTC ISO 时间写入结果。目标文件已存在时抛出 `FileExistsError`，不得覆盖历史运行。

`review`：读回一题 JSON，调用 `add_assistant_review()`，UTF-8、缩进 2 格写回；`correct_refusal` 可以没有 evidence article，其余 verdict 至少要求一个。

`summary`：读取 run 目录全部单题 JSON（按文件名排序），调用 `build_summary()`，写入 `summary.json`；目录不存在或没有单题文件时抛出 `ValueError`。

- [ ] **Step 4: 验证并提交**

Run: `python -m unittest discover -s tests -v`

Expected: PASS，只读取本地 JSON，不运行真实模型。

Run: `python -m json.tool evals/cases.json`

Expected: JSON 格式正确。

Run: `git check-ignore -v evals/results/demo/example.json`

Expected: 命中 `evals/results/.gitignore` 的 `*` 规则。

```bash
git add evaluate.py evals/cases.json evals/results/.gitignore tests/test_evaluation.py
git commit -m "feat: add local RAG evaluation runner"
```

### Task 4: 运行 V5 基线与逐题审查

**文件：**

- Create at runtime: `evals/results/v5-baseline/<case_id>.json`
- Create at runtime: `evals/results/v5-baseline/summary.json`

**接口：** 使用 Task 3 CLI 和用户当前 `.env`，产出 12 个结果与汇总。

- [ ] **Step 1: 确认基线参数**

只读取并报告 `.env` 的 `RETRIEVAL_K` 与 `RERANK_TOP_N` 数值。若不是 8 和 3，先请用户决定是否恢复；绝不显示 API Key。

- [ ] **Step 2: 运行并审查前五题**

按以下 id 逐题执行 `run`，展示答案、Top K、Top N、预期法条命中和 PDF 依据，暂停等待用户提问；然后以真实观察到的结论调用 `review`：

```text
probation-limit
resignation-notice
working-hours
overtime-pay
overtime-limit-and-pay
```

示例命令：

```bash
python evaluate.py run --case-id probation-limit --run-id v5-baseline
python evaluate.py review --run-id v5-baseline --case-id probation-limit \
  --verdict complete --evidence-article 第二十一条 \
  --reason "答案说明试用期最长不得超过六个月，覆盖该题全部预期事实。"
```

- [ ] **Step 3: 自主完成后七题**

若用户对前五题无额外问题，按顺序执行 `maternity-leave`、`minimum-wage`、`arbitration-limit`、`holiday-overtime-premise`、`probation-salary-eighty-percent`、`annual-leave-five-days`、`piecework-overtime-formula`。每题运行、基于 PDF 进行审查并写回结果；只有 API 错误、与资料明显矛盾的结果或用户插话时暂停。

- [ ] **Step 4: 汇总、回归并提交实现**

Run: `python evaluate.py summary --run-id v5-baseline`

Expected: 生成 12 题总数、Top K/Top N 命中率、拒答结果和 verdict 数量。

Run: `python -m unittest discover -s tests -v`

Expected: PASS；真实结果文件被忽略，不进入 Git 状态。

```bash
git add rag_app.py main.py evaluation.py evaluate.py evals/cases.json evals/results/.gitignore tests/test_rag_app.py tests/test_evaluation.py
git commit -m "feat: add v5 local RAG evaluation"
```

## 计划自检

- 规格覆盖：共享链路（Task 1）、离线指标与审查（Task 2）、12 题和 CLI（Task 3）、五题陪跑及七题自主运行（Task 4）均有任务。
- 范围控制：没有 LangSmith、自动 LLM 评委、阈值、混合检索、对话记忆或 API 服务。
- 接口一致：`build_case_result()` 产出的结果由 `add_assistant_review()` 写回，再被 `build_summary()` 聚合；CLI 只用这三项接口读写。
- 测试隔离：工厂与评测测试都 mock 重型组件或读取本地 JSON，不调用真实服务。

