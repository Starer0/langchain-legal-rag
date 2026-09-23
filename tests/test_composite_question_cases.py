import unittest
from pathlib import Path

from run_metadata_filter_evaluation import load_cases


COMPOSITE_CASES_PATH = Path("evals/composite_question_cases.json")
EXPECTED_ARTICLES = {
    "one-year-contract-probation-wage": {
        ("labor_contract_law", "第十九条"),
        ("labor_contract_law", "第二十条"),
    },
    "illegal-probation-compensation-limitation": {
        ("labor_contract_law", "第十九条"),
        ("labor_contract_law", "第八十三条"),
        ("labor_arbitration_law", "第二十七条"),
    },
    "probation-dismissal-compensation-limitation": {
        ("labor_contract_law", "第三十九条"),
        ("labor_contract_law", "第四十六条"),
        ("labor_arbitration_law", "第二十七条"),
    },
    "contract-expiry-compensation-limitation": {
        ("labor_contract_law", "第四十六条"),
        ("labor_contract_law", "第四十七条"),
        ("labor_arbitration_law", "第二十七条"),
    },
    "holiday-overtime-arbitration-application": {
        ("labor_law", "第四十四条"),
        ("labor_arbitration_law", "第二十七条"),
        ("labor_arbitration_law", "第二十八条"),
    },
    "rest-day-overtime-arbitration-application": {
        ("labor_law", "第四十四条"),
        ("labor_arbitration_law", "第二十七条"),
        ("labor_arbitration_law", "第二十八条"),
    },
    "probation-wage-employment-status-limitation": {
        ("labor_contract_law", "第二十条"),
        ("labor_arbitration_law", "第二十七条"),
    },
    "probation-wage-holiday-overtime-arbitration": {
        ("labor_contract_law", "第二十条"),
        ("labor_law", "第四十四条"),
        ("labor_arbitration_law", "第二十七条"),
        ("labor_arbitration_law", "第二十八条"),
    },
}


class CompositeQuestionCaseTests(unittest.TestCase):
    def test_composite_challenge_set_covers_each_expected_legal_point(self):
        self.assertTrue(
            COMPOSITE_CASES_PATH.exists(),
            "复合问题挑战集必须独立保存，不能混入 V7 元数据过滤题集",
        )

        cases = load_cases(COMPOSITE_CASES_PATH)
        actual_articles = {
            case["id"]: {
                (article["law_id"], article["article"])
                for article in case["expected_articles"]
            }
            for case in cases
        }

        self.assertEqual(actual_articles, EXPECTED_ARTICLES)
        self.assertTrue(all(len(case["expected_articles"]) >= 2 for case in cases))


if __name__ == "__main__":
    unittest.main()
