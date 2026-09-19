import unittest
from unittest.mock import Mock

from main import run_cli


class MainCliTests(unittest.TestCase):
    def test_run_cli_prints_retrieval_question_and_uses_service(self):
        service = Mock()
        service.ask.return_value = {
            "retrieval_question": "试用期内劳动者工资有什么规定？",
            "answer": "应按规定支付工资。",
            "candidates": [],
            "sources": [],
        }
        answers = iter(["那工资呢？", "q"])
        output = []

        run_cli(
            service,
            input_fn=lambda _: next(answers),
            output_fn=output.append,
        )

        service.ask.assert_called_once_with("那工资呢？")
        self.assertTrue(
            any(
                "检索问题：试用期内劳动者工资有什么规定？" in text
                for text in output
            )
        )


if __name__ == "__main__":
    unittest.main()
