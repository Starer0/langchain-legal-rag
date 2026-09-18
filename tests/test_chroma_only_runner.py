import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


CASE = {
    "id": "minimum-wage",
    "question": "最低工资有什么规定？",
    "expected_articles": ["第四十八条"],
    "required_facts": ["不得低于当地最低工资标准"],
    "answerable": True,
}

CHAIN_RESULT = {
    "answer": "第四十八条规定工资不得低于当地最低工资标准。",
    "candidates": [{"article": "第四十八条", "pages": [6], "content": "候选"}],
    "sources": [{"article": "第四十八条", "pages": [6], "content": "来源"}],
}


class ChromaOnlyRunnerTests(unittest.TestCase):
    @patch("run_chroma_only_evaluation.create_chroma_only_chain")
    @patch("run_chroma_only_evaluation.load_cases")
    def test_runs_all_cases_once_with_one_shared_chain(self, load_cases, create_chain):
        from run_chroma_only_evaluation import run_all_cases

        load_cases.return_value = [CASE]
        chain = create_chain.return_value
        chain.invoke.return_value = CHAIN_RESULT

        with tempfile.TemporaryDirectory() as temporary_directory:
            results_dir = Path(temporary_directory)
            summary_path = run_all_cases("control", results_dir=results_dir)

            with summary_path.open(encoding="utf-8") as handle:
                summary = json.load(handle)
            with (results_dir / "control" / "minimum-wage.json").open(encoding="utf-8") as handle:
                result = json.load(handle)

        create_chain.assert_called_once()
        chain.invoke.assert_called_once_with(CASE["question"])
        self.assertEqual(summary["total_cases"], 1)
        self.assertEqual(result["config"]["use_reranker"], False)
        self.assertEqual(result["config"]["rerank_top_n"], None)


if __name__ == "__main__":
    unittest.main()
