import unittest
from unittest.mock import Mock, patch


class LangGraphCliTests(unittest.TestCase):
    @patch("main.run_cli", side_effect=RuntimeError("退出"))
    @patch("main.create_conversation_service")
    @patch("main.ensure_legal_corpus_ready")
    def test_sqlite_options_and_connection_cleanup(self, ensure, create_service, run_cli):
        import main

        with self.assertRaisesRegex(RuntimeError, "退出"):
            main.main(["--langgraph", "--checkpoint", "--checkpoint-db", "data/checkpoints.sqlite3", "--thread-id", "A"])
        create_service.assert_called_once_with(
            decompose=False, use_langgraph=True, checkpoint=True, thread_id="A",
            checkpoint_db="data/checkpoints.sqlite3",
        )
        create_service.return_value.close.assert_called_once()

    def test_checkpoint_database_requires_checkpoint(self):
        import main

        with patch("main.ensure_legal_corpus_ready") as ensure, self.assertRaises(SystemExit):
            main.main(["--langgraph", "--checkpoint-db", "data/checkpoints.sqlite3"])
        ensure.assert_not_called()

    def test_checkpoint_commands_only_read_saved_state(self):
        from main import run_cli

        service = Mock()
        service.thread_id = "practice"
        service.inspect_checkpoint.return_value = {"question": "工资？", "next": []}
        service.checkpoint_history.return_value = [{"step": 2, "next": ["rerank"]}]
        inputs = iter(["/state", "/checkpoints", "q"])
        output = []

        run_cli(service, input_fn=lambda _: next(inputs), output_fn=output.append)

        service.ask.assert_not_called()
        service.inspect_checkpoint.assert_called_once()
        service.checkpoint_history.assert_called_once()
        self.assertIn("工资？", "".join(output))

    def test_invalid_checkpoint_options_fail_before_starting_service(self):
        import main

        for args in (["--checkpoint"], ["--langgraph", "--checkpoint", "--profile"], ["--thread-id", "A"]):
            with self.subTest(args=args), patch("main.ensure_legal_corpus_ready") as ensure, self.assertRaises(SystemExit):
                main.main(args)
            ensure.assert_not_called()

    @patch("main.run_cli")
    @patch("main.create_conversation_service")
    @patch("main.ensure_legal_corpus_ready")
    def test_checkpoint_flag_configures_thread(self, ensure, create_service, run_cli):
        import main

        main.main(["--langgraph", "--checkpoint", "--thread-id", "practice"])

        create_service.assert_called_once_with(
            decompose=False, use_langgraph=True, checkpoint=True, thread_id="practice"
        )

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
