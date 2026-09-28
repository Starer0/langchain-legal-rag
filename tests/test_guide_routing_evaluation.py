import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from query_rewrite import RetrievalPlan
from run_guide_routing_evaluation import run_all_cases


LAWS = [{
    "law_id": "labor_contract_law",
    "law_name": "中华人民共和国劳动合同法",
    "aliases": ["劳动合同法"],
}]
CASE = {
    "id": "application-materials",
    "question": "劳动合同法对未签书面合同怎么规定，申请仲裁要带什么材料？",
    "expected_include_guide": True,
}


class GuideRoutingEvaluationTests(unittest.TestCase):
    def test_records_route_and_metadata_scope_for_each_case(self):
        rewriter = Mock()
        rewriter.rewrite.return_value = RetrievalPlan(
            "劳动合同法未签书面合同和仲裁申请材料", include_guide=True
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            results_dir = Path(temporary_directory)
            summary_path = run_all_cases(
                "routing-test", results_dir=results_dir, cases=[CASE],
                laws=LAWS, rewriter=rewriter,
            )
            result = json.loads(
                (results_dir / "routing-test" / "application-materials.json").read_text(
                    encoding="utf-8"
                )
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertTrue(result["metrics"]["route_matches_expected"])
        self.assertTrue(result["metrics"]["filter_includes_guide"])
        self.assertEqual(summary["route_matches_expected"], 1)
        self.assertEqual(summary["guide_scope_hits"], 1)

    def test_refuses_to_overwrite_existing_run_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            results_dir = Path(temporary_directory)
            (results_dir / "existing").mkdir()
            with self.assertRaises(FileExistsError):
                run_all_cases("existing", results_dir=results_dir, cases=[], laws=LAWS, rewriter=Mock())
