import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import evaluate


CASE = {"id": "case", "question": "question", "expected_articles": [], "required_facts": [], "answerable": False}
CHAIN_RESULT = {"answer": "answer", "candidates": [], "sources": []}


class EvaluateCliTests(unittest.TestCase):
    def test_run_case_uses_shared_chain_once_and_refuses_existing_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            chain = Mock()
            chain.invoke.return_value = CHAIN_RESULT
            with patch.object(evaluate, "CASES_PATH", root / "cases.json"), patch.object(evaluate, "RESULTS_DIR", root / "results"), patch("rag_app.create_rag_chain", return_value=chain) as factory:
                (root / "cases.json").write_text(json.dumps([CASE]), encoding="utf-8")
                destination = evaluate.run_case("case", "run")
            self.assertEqual(destination, root / "results" / "run" / "case.json")
            factory.assert_called_once_with()
            chain.invoke.assert_called_once_with(CASE["question"])
            destination.write_text("{}", encoding="utf-8")
            with patch.object(evaluate, "CASES_PATH", root / "cases.json"), patch.object(evaluate, "RESULTS_DIR", root / "results"), patch("rag_app.create_rag_chain") as factory:
                with self.assertRaises(FileExistsError):
                    evaluate.run_case("case", "run")
            factory.assert_not_called()

    def test_review_requires_evidence_except_correct_refusal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            destination = root / "results" / "run" / "case.json"
            destination.parent.mkdir(parents=True)
            destination.write_text(json.dumps({"assistant_review": None}), encoding="utf-8")
            with patch.object(evaluate, "RESULTS_DIR", root / "results"):
                with self.assertRaises(ValueError):
                    evaluate.review_case("case", "run", "complete", "reason", [])
                evaluate.review_case("case", "run", "correct_refusal", "reason", [])
            reviewed = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(reviewed["assistant_review"]["verdict"], "correct_refusal")

    def test_summary_rejects_missing_and_empty_runs_then_sorts_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            results_dir = root / "results"
            with patch.object(evaluate, "RESULTS_DIR", results_dir):
                with self.assertRaises(ValueError):
                    evaluate.summarize_run("missing")
                run_dir = results_dir / "run"
                run_dir.mkdir(parents=True)
                with self.assertRaises(ValueError):
                    evaluate.summarize_run("run")
                (run_dir / "b.json").write_text(json.dumps({"case": {"id": "b"}}), encoding="utf-8")
                (run_dir / "a.json").write_text(json.dumps({"case": {"id": "a"}}), encoding="utf-8")
                captured = []
                def aggregate(results):
                    captured.extend(results)
                    return {"total_cases": len(results)}
