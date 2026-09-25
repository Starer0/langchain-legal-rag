import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from run_performance_baseline import run_baseline


class PerformanceBaselineTests(unittest.TestCase):
    def test_compares_same_cases_and_writes_metrics_without_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases_path = root / "cases.json"
            cases_path.write_text(json.dumps([{
                "id": "case-one",
                "question": "试用期工资？",
                "expected_articles": [{
                    "law_id": "labor_contract_law", "article": "第二十条",
                }],
            }], ensure_ascii=False), encoding="utf-8")
            services = []

            def create_service(*, decompose, profile):
                self.assertTrue(profile)
                service = Mock()
                service.ask.return_value = {
                    "answer": "这是完整回答，不应写入性能记录",
                    "sources": [{
                        "law_id": "labor_contract_law", "article": "第二十条",
                    }],
                    "performance": {
                        "total_ms": 12.0, "model_calls": 2, "reranker_calls": 1,
                        "stages": {"answer": {"calls": 1, "duration_ms": 5.0}},
                    },
                }
                services.append((decompose, service))
                return service

            output = run_baseline(
                "comparison", cases_path=cases_path, results_dir=root / "results",
                service_factory=create_service,
            )
            data = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual([mode for mode, _ in services], [False, True])
            for _, service in services:
                service.history.clear.assert_called_once_with()
                service.ask.assert_called_once_with("试用期工资？")
            self.assertEqual(data["cases"][0]["id"], "case-one")
            self.assertTrue(data["cases"][0]["single"]["source_hit"])
            self.assertTrue(data["cases"][0]["decompose"]["source_hit"])
            self.assertEqual(data["cases"][0]["single"]["performance"]["model_calls"], 2)
            self.assertNotIn("这是完整回答", output.read_text(encoding="utf-8"))
            self.assertNotIn("试用期工资？", output.read_text(encoding="utf-8"))

    def test_refuses_to_overwrite_an_existing_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "results" / "same-run").mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                run_baseline(
                    "same-run", cases_path=root / "unused.json",
                    results_dir=root / "results", service_factory=Mock(),
                )


if __name__ == "__main__":
    unittest.main()
