import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from run_metadata_filter_evaluation import run_all_cases


LAWS = [
    {"law_id": "labor_law", "law_name": "中华人民共和国劳动法", "aliases": ["劳动法"]},
    {
        "law_id": "labor_contract_law",
        "law_name": "中华人民共和国劳动合同法",
        "aliases": ["劳动合同法"],
    },
    {
        "law_id": "labor_arbitration_law",
        "law_name": "中华人民共和国劳动争议调解仲裁法",
        "aliases": ["劳动争议调解仲裁法"],
    },
]
CASE = {
    "id": "contract-wage",
    "question": "劳动合同法第20条是什么？",
    "expected_articles": [{"law_id": "labor_contract_law", "article": "第二十条"}],
    "expected_law_ids": ["labor_contract_law"],
}
CHAIN_RESULT = {
    "answer": "第二十条。",
    "candidates": [{"law_id": "labor_contract_law", "article": "第二十条"}],
    "sources": [{"law_id": "labor_contract_law", "article": "第二十条"}],
}


class MetadataFilterEvaluationTests(unittest.TestCase):
    def test_compares_all_laws_and_metadata_filtered_retrieval(self):
        unfiltered = Mock()
        filtered = Mock()
        unfiltered.invoke.return_value = CHAIN_RESULT
        filtered.invoke.return_value = CHAIN_RESULT

        with tempfile.TemporaryDirectory() as temporary_directory:
            results_dir = Path(temporary_directory)
            summary_path = run_all_cases(
                "filter-test",
                results_dir=results_dir,
                cases=[CASE],
                laws=LAWS,
                unfiltered_chain=unfiltered,
                filtered_chain=filtered,
            )
            result = json.loads(
                (results_dir / "filter-test" / "contract-wage.json").read_text(
                    encoding="utf-8"
                )
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(result["all_laws"]["filter"], {"status": "现行有效"})
        self.assertEqual(
            result["metadata_filtered"]["filter"]["$and"][-1],
            {"article": "第二十条"},
        )
        self.assertTrue(result["all_laws"]["metrics"]["scope_covers_expected_laws"])
        self.assertTrue(
            result["metadata_filtered"]["metrics"]["scope_covers_expected_laws"]
        )
        self.assertEqual(summary["metadata_filtered"]["source_hits"], 1)
        self.assertEqual(unfiltered.invoke.call_count, 1)
        self.assertEqual(filtered.invoke.call_count, 1)

    def test_refuses_to_overwrite_an_existing_run_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            results_dir = Path(temporary_directory)
            (results_dir / "existing").mkdir()
            with self.assertRaises(FileExistsError):
                run_all_cases("existing", results_dir=results_dir, cases=[], laws=LAWS)


if __name__ == "__main__":
    unittest.main()
