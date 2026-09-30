import unittest
from unittest.mock import patch


class LangGraphCliTests(unittest.TestCase):
    def test_trace_requires_langgraph_before_starting_service(self):
        import main

        with patch("main.ensure_legal_corpus_ready") as ensure, self.assertRaises(SystemExit):
            main.main(["--trace"])
        ensure.assert_not_called()

    @patch("main.run_cli")
    @patch("main.create_conversation_service")
    @patch("main.ensure_legal_corpus_ready")
    def test_trace_flag_enables_node_observation(self, ensure, create_service, run_cli):
        import main

        main.main(["--langgraph", "--trace"])

        run_cli.assert_called_once_with(create_service.return_value, trace=True)

    @patch("main.run_cli")
    @patch("main.create_conversation_service")
    @patch("main.ensure_legal_corpus_ready")
    def test_langgraph_flag_selects_the_graph_conversation_service(
        self, ensure, create_service, run_cli,
    ):
        import main

        main.main(["--langgraph"])

        create_service.assert_called_once_with(decompose=False, use_langgraph=True)
        run_cli.assert_called_once_with(create_service.return_value)


if __name__ == "__main__":
    unittest.main()
