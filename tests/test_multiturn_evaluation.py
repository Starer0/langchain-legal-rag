import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call

from run_multiturn_evaluation import run_all_cases


CASE = {
    "id": "trial-period-wage-follow-up",
    "history": [
        {"role": "human", "content": "试用期最长多久？"},
        {"role": "ai", "content": "最长六个月。"},
    ],
    "question": "那工资呢？",
    "expected_articles": ["第二十条"],
    "expected_retrieval_terms": ["试用期", "工资"],
}
CHAIN_RESULT = {
    "answer": "试用期工资应依法支付。",
    "candidates": [{"article": "第二十条", "pages": [2], "content": "候选"}],
    "sources": [{"article": "第二十条", "pages": [2], "content": "来源"}],
}


class MultiturnEvaluationTests(unittest.TestCase):
    def test_runner_writes_both_modes_for_each_case(self):
        chain = Mock()
        chain.invoke.return_value = CHAIN_RESULT
        rewriter = Mock()
        rewriter.rewrite.side_effect = ["工资规定", "试用期工资规定"]

        with tempfile.TemporaryDirectory() as temporary_directory:
            results_dir = Path(temporary_directory)
            summary_path = run_all_cases(
                "history-test",
                results_dir=results_dir,
                cases=[CASE],
                chain=chain,
                rewriter=rewriter,
            )
            result = json.loads(
                (results_dir / "history-test" / "trial-period-wage-follow-up.json").read_text(
                    encoding="utf-8"
                )
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(result["no_history"]["retrieval_question"], "工资规定")
        self.assertEqual(
            result["history_aware"]["retrieval_question"], "试用期工资规定"
        )
        self.assertTrue(result["no_history"]["metrics"]["expected_articles_in_sources"])
        self.assertTrue(result["history_aware"]["metrics"]["expected_articles_in_sources"])
        self.assertEqual(summary["total_cases"], 1)
        self.assertEqual(
            chain.invoke.call_args_list,
            [
                call({"question": "那工资呢？", "retrieval_question": "工资规定"}),
                call({"question": "那工资呢？", "retrieval_question": "试用期工资规定"}),
            ],
        )
        no_history = rewriter.rewrite.call_args_list[0].args[1]
        history_aware = rewriter.rewrite.call_args_list[1].args[1]
        self.assertEqual(no_history, [])
        self.assertEqual([message.content for message in history_aware], ["试用期最长多久？", "最长六个月。"])

    def test_runner_refuses_to_overwrite_a_run_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            results_dir = Path(temporary_directory)
            (results_dir / "existing").mkdir()
            with self.assertRaises(FileExistsError):
                run_all_cases("existing", results_dir=results_dir, cases=[])


if __name__ == "__main__":
    unittest.main()
