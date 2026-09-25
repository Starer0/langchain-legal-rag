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

    @patch("main.run_cli")
    @patch("main.create_conversation_service")
    def test_main_enables_composite_question_mode(self, create_service, run_cli_mock):
        main(["--decompose"])

        create_service.assert_called_once_with(decompose=True)
        run_cli_mock.assert_called_once_with(create_service.return_value)

    @patch("main.run_cli")
    @patch("main.create_conversation_service")
    def test_main_enables_opt_in_profiling(self, create_service, run_cli_mock):
        main(["--profile"])

        create_service.assert_called_once_with(decompose=False, profile=True)
        run_cli_mock.assert_called_once_with(create_service.return_value)

    def test_cli_prints_profiled_stage_times_and_call_counts(self):
        service = Mock()
        service.ask.return_value = {
            "retrieval_question": "试用期工资规定？",
            "answer": "回答",
            "candidates": [],
            "sources": [],
            "performance": {
                "total_ms": 123.45,
                "model_calls": 2,
                "reranker_calls": 1,
                "stages": {
                    "rewrite": {"calls": 1, "duration_ms": 30.0},
                    "chroma": {"calls": 1, "duration_ms": 10.0},
                    "rerank": {"calls": 1, "duration_ms": 20.0},
                    "answer": {"calls": 1, "duration_ms": 60.0},
                },
            },
        }
        answers = iter(["试用期工资？", "q"])
        output = []

        run_cli(service, input_fn=lambda _: next(answers), output_fn=output.append)

        display = "\n".join(output)
        self.assertIn("总耗时：123.45 ms", display)
        self.assertIn("改写：30.00 ms", display)
        self.assertIn("Chroma：10.00 ms", display)
        self.assertIn("Reranker：20.00 ms", display)
        self.assertIn("回答：60.00 ms", display)
        self.assertIn("模型调用：2 次", display)


if __name__ == "__main__":
    unittest.main()
