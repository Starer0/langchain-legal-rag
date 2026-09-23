import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run_composite_baseline import run_baseline


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


if __name__ == "__main__":
    unittest.main()
