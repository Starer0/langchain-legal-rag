from pathlib import Path
from unittest.mock import patch

from run_heterogeneous_evaluation import load_cases, main


DEVELOPMENT_CASES_PATH = Path("evals/heterogeneous_cases.json")
HOLDOUT_CASES_PATH = Path("evals/heterogeneous_holdout_cases.json")


def test_holdout_cases_are_disjoint_from_development_cases():
    development_cases = load_cases(DEVELOPMENT_CASES_PATH)
    holdout_cases = load_cases(HOLDOUT_CASES_PATH)

    assert 6 <= len(holdout_cases) <= 8
    assert {case["id"] for case in development_cases}.isdisjoint(
        case["id"] for case in holdout_cases
    )
    assert {case["question"] for case in development_cases}.isdisjoint(
        case["question"] for case in holdout_cases
    )
    assert {case["category"] for case in holdout_cases} == {
        "guide",
        "law",
        "mixed",
        "unsupported",
    }


@patch("run_heterogeneous_evaluation.run_all_cases")
def test_cli_loads_explicit_holdout_cases_path(mock_run_all_cases):
    main(
        [
            "--run-id",
            "holdout-smoke",
            "--cases-path",
            str(HOLDOUT_CASES_PATH),
        ]
    )

    mock_run_all_cases.assert_called_once_with(
        "holdout-smoke", cases=load_cases(HOLDOUT_CASES_PATH)
    )
