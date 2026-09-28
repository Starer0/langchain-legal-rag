import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from run_heterogeneous_evaluation import run_all_cases


CASE = {
    "id": "mixed-materials",
    "question": "劳动合同法对未签书面合同怎么规定，申请仲裁要带什么材料？",
    "expected_include_guide": True,
    "expected_sources": [
        {"law_id": "labor_contract_law", "article": "第八十二条"},
        {"document_id": "labor_arbitration_guide_experimental_city", "section": "3.1 必备材料"},
    ],
    "required_facts": ["双倍工资", "仲裁申请书"],
    "answerable": True,
}
RESULT = {
    "answer": "未签书面合同可主张双倍工资；提交仲裁申请书。",
    "include_guide": True,
    "sources": [
        {"law_id": "labor_contract_law", "article": "第八十二条"},
        {"document_id": "labor_arbitration_guide_experimental_city", "section": "3.1 必备材料"},
    ],
    "candidates": [],
    "performance": {"total_ms": 12.0},
}


class HeterogeneousEvaluationTests(unittest.TestCase):
    def test_records_end_to_end_metrics_and_clears_history_per_case(self):
        service = Mock()
        service.ask.return_value = RESULT
        service.history.clear = Mock()

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            summary_path = run_all_cases("hetero", results_dir=root, cases=[CASE], service=service)
            result = json.loads((root / "hetero" / "mixed-materials.json").read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertTrue(result["metrics"]["route_matches_expected"])
        self.assertTrue(result["metrics"]["expected_sources_in_sources"])
        self.assertTrue(result["metrics"]["required_facts_present"])
        self.assertEqual(summary["source_hits"], 1)
        service.history.clear.assert_called_once()

    def test_unsupported_case_does_not_require_a_source_hit(self):
        case = {**CASE, "expected_sources": [], "required_facts": [], "answerable": False}
        service = Mock()
        service.ask.return_value = {**RESULT, "answer": "资料中没有足够依据。", "sources": []}
        service.history.clear = Mock()

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_all_cases("unsupported", results_dir=root, cases=[case], service=service)
            result = json.loads((root / "unsupported" / "mixed-materials.json").read_text(encoding="utf-8"))

        self.assertTrue(result["metrics"]["expected_sources_in_sources"])
