import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run_composite_baseline import main, run_baseline


class _History:
    def __init__(self):
        self.clear_count = 0

    def clear(self):
        self.clear_count += 1


class _Service:
    def __init__(self):
        self.history = _History()
        self.questions = []

    def ask(self, question):
        self.questions.append(question)
        document = {"law_id": "labor_law", "article": "第四十四条", "content": "加班工资"}
        return {
            "retrieval_question": f"检索：{question}",
            "answer": "加班工资",
            "candidates": [document],
            "sources": [document],
        }


class CompositeBaselineTests(unittest.TestCase):
    def test_cli_accepts_a_frozen_holdout_case_file(self):
        with patch("run_composite_baseline.run_baseline") as run:
            try:
                main(["--run-id", "holdout", "--decompose", "--cases-path", "holdout.json"])
            except SystemExit as error:
                self.fail(f"评测入口拒绝独立题集参数：{error}")

        self.assertEqual(run.call_args.args, ("holdout",))
        self.assertEqual(run.call_args.kwargs, {
            "decompose": True,
            "cases_path": Path("holdout.json"),
        })

    def test_runs_each_case_as_an_independent_turn_and_records_hits(self):
        cases = [
            {
                "id": case_id,
                "question": question,
                "expected_articles": [{"law_id": "labor_law", "article": "第四十四条"}],
                "expected_law_ids": ["labor_law"],
            }
            for case_id, question in (("first", "第一题"), ("second", "第二题"))
        ]
        service = _Service()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cases_path = root / "cases.json"
            cases_path.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
            with patch("run_composite_baseline.load_dotenv"):
                summary_path = run_baseline(
                    "test-run", service=service, cases_path=cases_path, results_dir=root
                )
            first = json.loads((root / "test-run" / "first.json").read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(service.history.clear_count, 2)
        self.assertEqual(service.questions, ["第一题", "第二题"])
        self.assertEqual(first["retrieval_question"], "检索：第一题")
        self.assertEqual(first["config"]["history_turns"], 0)
        self.assertEqual(summary["candidate_hits"], 2)
        self.assertEqual(summary["source_hits"], 2)

    def test_records_decomposition_mode_and_subquestions(self):
        service = _Service()
        original_ask = service.ask

        def ask_with_subquestions(question):
            return {
                **original_ask(question),
                "subquestions": ["工资规定？", "仲裁时效？"],
                "subquestion_reranks": [
                    {
                        "question": "工资规定？",
                        "sources": [{
                            "law_id": "labor_law",
                            "article": "第四十四条",
                            "content": "加班工资",
                            "rerank_score": 0.9,
                        }],
                    },
                    {"question": "仲裁时效？", "sources": []},
                ],
            }

        service.ask = ask_with_subquestions
        case = [{
            "id": "composite",
            "question": "工资和仲裁时效？",
            "expected_articles": [{"law_id": "labor_law", "article": "第四十四条"}],
            "expected_law_ids": ["labor_law"],
        }]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cases_path = root / "cases.json"
            cases_path.write_text(json.dumps(case, ensure_ascii=False), encoding="utf-8")
            with patch("run_composite_baseline.load_dotenv"):
                run_baseline(
                    "decomposed", service=service, cases_path=cases_path,
                    results_dir=root, decompose=True
                )
            result = json.loads(
                (root / "decomposed" / "composite.json").read_text(encoding="utf-8")
            )

        self.assertTrue(result["config"]["decomposition"])
        self.assertEqual(result["subquestions"], ["工资规定？", "仲裁时效？"])
        first_source = result["subquestion_reranks"][0]["sources"][0]
        self.assertEqual(first_source["article"], "第四十四条")
        self.assertEqual(first_source["rerank_score"], 0.9)
        self.assertTrue(result["metrics"]["expected_articles_in_reranks"])


if __name__ == "__main__":
    unittest.main()
