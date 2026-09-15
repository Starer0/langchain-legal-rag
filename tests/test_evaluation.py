import subprocess
import sys
import unittest
from pathlib import Path

from evaluate import find_case, load_cases
from evaluation import (
    PREVIEW_LENGTH,
    add_assistant_review,
    build_case_result,
    build_summary,
)


CASE = {
    "id": "overtime-pay",
    "question": "加班工资如何计算？",
    "expected_articles": ["第四十四条"],
    "required_facts": ["不低于150%"],
    "answerable": True,
}

CHAIN_RESULT = {
    "answer": "第四十四条规定延长工作时间支付不低于150%的工资报酬。",
    "candidates": [{"article": "第四十四条", "pages": [6], "content": "x" * 200}],
    "sources": [
        {
            "article": "第四十四条",
            "pages": [6],
            "content": "y" * 200,
            "rerank_score": 0.9,
        }
    ],
}


class EvaluationTests(unittest.TestCase):
    def test_script_entrypoint_dispatches_to_argparse(self):
        completed = subprocess.run(
            [sys.executable, "evaluate.py"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 2)
        self.assertIn("usage:", completed.stderr)

    def test_load_cases_reads_the_exact_local_dataset(self):
        cases = load_cases(Path("evals/cases.json"))

        self.assertEqual(len(cases), 12)
        self.assertEqual(len({case["id"] for case in cases}), 12)
        self.assertEqual(cases[0]["expected_articles"], ["第二十一条"])
        self.assertFalse(cases[9]["answerable"])

    def test_find_case_rejects_an_unknown_case_id(self):
        with self.assertRaises(ValueError):
            find_case([], "unknown-case")

    def build_result(self):
        return build_case_result(
            CASE,
            CHAIN_RESULT,
            {"retrieval_k": 8, "rerank_top_n": 3},
            "run-1",
            "2026-09-15T10:00:00+00:00",
        )

    def test_result_records_hits_and_truncates_content(self):
        result = self.build_result()

        self.assertEqual(result["case"], CASE)
        self.assertTrue(result["metrics"]["expected_articles_in_candidates"])
        self.assertTrue(result["metrics"]["expected_articles_in_sources"])
        self.assertIsNone(result["assistant_review"])
        self.assertEqual(len(result["candidates"][0]["content_preview"]), PREVIEW_LENGTH)
        self.assertNotIn("content", result["candidates"][0])

    def test_result_serializes_sources_without_full_content(self):
        result = self.build_result()

        self.assertEqual(result["sources"][0]["content_preview"], "y" * PREVIEW_LENGTH)
        self.assertNotIn("content", result["sources"][0])
        self.assertEqual(result["sources"][0]["rerank_score"], 0.9)

    def test_unsupported_case_detects_refusal(self):
        case = {**CASE, "answerable": False, "expected_articles": [], "required_facts": []}
        result = build_case_result(
            case,
            {**CHAIN_RESULT, "answer": "资料中没有足够依据。"},
            {},
            "run-1",
            "time",
        )

        self.assertTrue(result["metrics"]["refusal_phrase_present"])

    def test_answerable_case_does_not_record_refusal_detection(self):
        result = build_case_result(
            CASE,
            {**CHAIN_RESULT, "answer": "资料中没有足够依据。"},
            {},
            "run-1",
            "time",
        )

        self.assertFalse(result["metrics"]["refusal_phrase_present"])

    def test_add_assistant_review_returns_copy_without_mutating_result(self):
        result = self.build_result()

        reviewed = add_assistant_review(result, "complete", ["第四十四条"], "依据完整。")

        self.assertIsNone(result["assistant_review"])
        self.assertEqual(
            reviewed["assistant_review"],
            {"verdict": "complete", "evidence_articles": ["第四十四条"], "reason": "依据完整。"},
        )

    def test_add_assistant_review_rejects_invalid_verdict(self):
        with self.assertRaises(ValueError):
            add_assistant_review(self.build_result(), "unsupported", [], "无效")

    def test_summary_counts_metrics_and_verdicts(self):
        hit = add_assistant_review(self.build_result(), "complete", ["第四十四条"], "完整")
        unsupported = build_case_result(
            {**CASE, "answerable": False, "expected_articles": [], "required_facts": []},
            {**CHAIN_RESULT, "answer": "资料中没有足够依据。"},
            {},
            "run-1",
            "time",
        )
        unsupported = add_assistant_review(unsupported, "correct_refusal", [], "正确拒答")

        summary = build_summary([hit, unsupported])

        self.assertEqual(summary["total_cases"], 2)
        self.assertEqual(summary["candidate_hits"], 2)
        self.assertEqual(summary["candidate_hit_rate"], 1.0)
        self.assertEqual(summary["source_hits"], 2)
        self.assertEqual(summary["source_hit_rate"], 1.0)
        self.assertEqual(summary["unsupported_cases"], 1)
        self.assertEqual(summary["refusal_hits"], 1)
        self.assertEqual(summary["refusal_hit_rate"], 1.0)
        self.assertEqual(summary["verdict_counts"], {"complete": 1, "correct_refusal": 1})

    def test_empty_summary_has_zero_rates(self):
        summary = build_summary([])

        self.assertEqual(summary["total_cases"], 0)
        self.assertEqual(summary["candidate_hit_rate"], 0.0)
        self.assertEqual(summary["source_hit_rate"], 0.0)
        self.assertEqual(summary["refusal_hit_rate"], 0.0)
        self.assertEqual(summary["verdict_counts"], {})


if __name__ == "__main__":
    unittest.main()
