import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from run_composite_budget_evaluation import run_budget_evaluation


class CompositeBudgetEvaluationTests(unittest.TestCase):
    def test_runs_explicit_subquestions_and_records_budget_configuration(self):
        cases = [{
            "id": "six-part-question",
            "question": "六个问题",
            "retrieval_questions": ["问题一", "问题二"],
            "expected_articles": [{"law_id": "labor_law", "article": "第四十四条"}],
            "expected_law_ids": ["labor_law"],
        }]
        chain = Mock()
        chain.invoke.return_value = {
            "answer": "回答",
            "candidates": [],
            "sources": [{"law_id": "labor_law", "article": "第四十四条"}],
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_path = root / "cases.json"
            cases_path.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
            with patch("run_composite_budget_evaluation.load_dotenv"):
                summary_path = run_budget_evaluation(
                    "dynamic", cases_path=cases_path, results_dir=root / "results",
                    chain_factory=lambda: chain, min_docs_per_question=2,
                )
            result = json.loads((summary_path.parent / "six-part-question.json").read_text(
                encoding="utf-8"
            ))

        self.assertEqual(chain.invoke.call_args.args[0]["retrieval_questions"], ["问题一", "问题二"])
        self.assertEqual(result["subquestions"], ["问题一", "问题二"])
        self.assertEqual(result["config"]["min_docs_per_question"], 2)
        self.assertTrue(result["metrics"]["expected_articles_in_sources"])


if __name__ == "__main__":
    unittest.main()
