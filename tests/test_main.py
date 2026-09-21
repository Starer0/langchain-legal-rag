import unittest
from unittest.mock import Mock
from unittest.mock import patch

from main import main, run_cli


class MainCliTests(unittest.TestCase):
    def test_run_cli_prints_retrieval_question_and_uses_service(self):
        service = Mock()
        service.ask.return_value = {
            "retrieval_question": "试用期内劳动者工资有什么规定？",
            "answer": "应按规定支付工资。",
            "candidates": [{
                "law_name": "中华人民共和国劳动合同法",
                "article": "第二十条",
                "pages": [2],
            }],
            "sources": [{
                "law_name": "中华人民共和国劳动合同法",
                "article": "第二十条",
                "pages": [2],
                "content": "试用期工资不得低于规定标准。",
            }],
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
        self.assertTrue(any("中华人民共和国劳动合同法 第二十条" in text for text in output))

    @patch("main.ingest_legal_corpus")
    def test_main_ingests_the_corpus_when_requested(self, ingest):
        main(["--ingest"])
        ingest.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
