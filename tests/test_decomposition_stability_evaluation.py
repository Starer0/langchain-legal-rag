import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run_decomposition_stability_evaluation import run_decomposition_evaluation


class DecompositionStabilityEvaluationTests(unittest.TestCase):
    def test_records_expected_and_actual_subquestion_counts(self):
        cases = [{
            "id": "two-intents",
            "question": "工资和仲裁时效？",
            "expected_subquestion_count": 2,
        }]

        class Decomposer:
            def decompose(self, question):
                self.question = question
                return ["工资标准是什么？", "仲裁时效是多久？"]

        decomposer = Decomposer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_path = root / "cases.json"
            cases_path.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
            with patch("run_decomposition_stability_evaluation.load_dotenv"):
                summary_path = run_decomposition_evaluation(
                    "stability", cases_path=cases_path, results_dir=root / "results",
                    decomposer_factory=lambda: decomposer,
                )
            result = json.loads((summary_path.parent / "two-intents.json").read_text(
                encoding="utf-8"
            ))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(decomposer.question, "工资和仲裁时效？")
        self.assertEqual(result["subquestions"], ["工资标准是什么？", "仲裁时效是多久？"])
        self.assertTrue(result["metrics"]["subquestion_count_matches"])
        self.assertEqual(summary["count_match_rate"], 1.0)

    def test_can_measure_decomposition_after_rewrite(self):
        cases = [{
            "id": "rewritten-question",
            "question": "原始问题",
            "expected_subquestion_count": 2,
        }]

        class Rewriter:
            def rewrite(self, question, history):
                self.values = (question, history)
                return type("Plan", (), {"retrieval_question": "改写后的两个问题"})()

        class Decomposer:
            def decompose(self, question):
                self.question = question
                return ["问题一？", "问题二？"]

        rewriter = Rewriter()
        decomposer = Decomposer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_path = root / "cases.json"
            cases_path.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
            with patch("run_decomposition_stability_evaluation.load_dotenv"):
                summary_path = run_decomposition_evaluation(
                    "rewrite-first", cases_path=cases_path, results_dir=root / "results",
                    decomposer_factory=lambda: decomposer,
                    rewriter_factory=lambda: rewriter,
                )
            result = json.loads((summary_path.parent / "rewritten-question.json").read_text(
                encoding="utf-8"
            ))

        self.assertEqual(rewriter.values, ("原始问题", []))
        self.assertEqual(decomposer.question, "改写后的两个问题")
        self.assertEqual(result["retrieval_question"], "改写后的两个问题")


if __name__ == "__main__":
    unittest.main()
