import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(law_id, article, score):
    return {
        "law_id": law_id,
        "version": "2026",
        "article": article,
        "rerank_score": score,
        "content_preview": article,
    }


class MergeReplayTests(unittest.TestCase):
    def test_rejects_directory_without_saved_case_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            completed = subprocess.run(
                [sys.executable, "-m", "experiments.merge_replay", "--run-dir", temp_dir],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("没有可回放的评测记录", completed.stderr)

    def test_diverse_law_extra_slot_recovers_third_ranked_article(self):
        wage = _source("contract", "工资", 0.9)
        liability = _source("contract", "欠付责任", 0.75)
        first_extra = _source("contract", "试用期责任", 0.7)
        holiday = _source("labor", "节假日工资", 0.44)
        limitation = _source("arbitration", "仲裁时效", 0.96)
        application = _source("arbitration", "申请书", 0.99)
        old_limitation = _source("labor", "旧时效", 0.44)
        record = {
            "case": {
                "id": "four-facets",
                "expected_articles": [
                    {"law_id": "contract", "article": "工资"},
                    {"law_id": "labor", "article": "节假日工资"},
                    {"law_id": "arbitration", "article": "仲裁时效"},
                    {"law_id": "arbitration", "article": "申请书"},
                ],
            },
            "subquestion_reranks": [
                {"question": "试用期工资", "sources": [wage, first_extra]},
                {"question": "节假日加班标准", "sources": [
                    liability, _source("contract", "加班义务", 0.44), holiday,
                ]},
                {"question": "仲裁时效", "sources": [limitation, old_limitation]},
                {"question": "申请书", "sources": [application]},
            ],
            "sources": [wage, liability, limitation, application, first_extra],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "four-facets.json").write_text(
                json.dumps(record, ensure_ascii=False), encoding="utf-8"
            )
            completed = subprocess.run(
                [sys.executable, "-m", "experiments.merge_replay", "--run-dir", str(run_dir)],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["summary"], {
            "cases": 1,
            "baseline_matches_logged": 1,
            "baseline_hits": 0,
            "diversity_hits": 1,
        })
        case = report["cases"][0]
        self.assertEqual(case["diversity_sources"][-1], "labor:节假日工资")
        self.assertEqual(case["baseline_sources"][-1], "contract:试用期责任")


if __name__ == "__main__":
    unittest.main()
