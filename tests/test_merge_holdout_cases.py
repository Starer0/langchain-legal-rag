import unittest
from pathlib import Path

from run_metadata_filter_evaluation import load_cases


HOLDOUT = Path("evals/composite_merge_holdout_cases.json")
ORIGINAL = Path("evals/composite_question_cases.json")
EXPECTED = {
    "weekly-hours-rest": {("labor_law", "第三十六条"), ("labor_law", "第三十八条")},
    "overtime-limit-pay": {("labor_law", "第四十一条"), ("labor_law", "第四十四条")},
    "unpaid-wages-quit-compensation": {
        ("labor_contract_law", "第三十八条"), ("labor_contract_law", "第四十六条")
    },
    "arbitration-application-acceptance-award": {
        ("labor_arbitration_law", "第二十八条"),
        ("labor_arbitration_law", "第二十九条"),
        ("labor_arbitration_law", "第四十三条"),
    },
    "unfitness-termination-compensation-arbitration": {
        ("labor_contract_law", "第四十条"),
        ("labor_contract_law", "第四十六条"),
        ("labor_arbitration_law", "第二十七条"),
    },
    "wage-delay-after-departure": {
        ("labor_law", "第五十条"), ("labor_arbitration_law", "第二十七条")
    },
    "holiday-rest-pay-and-late-pay": {
        ("labor_law", "第五十一条"), ("labor_contract_law", "第三十条")
    },
    "medical-period-dismissal-remedy": {
        ("labor_contract_law", "第四十二条"),
        ("labor_contract_law", "第四十八条"),
        ("labor_contract_law", "第八十七条"),
        ("labor_arbitration_law", "第二十七条"),
    },
}


class MergeHoldoutCaseTests(unittest.TestCase):
    def test_holdout_is_distinct_and_has_four_single_and_four_cross_law_cases(self):
        self.assertTrue(HOLDOUT.is_file(), "独立评测题集尚未创建")
        cases = load_cases(HOLDOUT)
        original = load_cases(ORIGINAL)
        actual = {
            case["id"]: {(item["law_id"], item["article"]) for item in case["expected_articles"]}
            for case in cases
        }

        self.assertEqual(actual, EXPECTED)
        self.assertEqual(len({case["question"] for case in cases}), 8)
        self.assertFalse({case["question"] for case in cases} & {case["question"] for case in original})
        self.assertEqual(sum(len(case["expected_law_ids"]) == 1 for case in cases), 4)
        self.assertEqual(sum(len(case["expected_law_ids"]) > 1 for case in cases), 4)
        self.assertTrue(all(
            {item["law_id"] for item in case["expected_articles"]}
            == set(case["expected_law_ids"])
            for case in cases
        ))


if __name__ == "__main__":
    unittest.main()
