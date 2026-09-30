import unittest

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.chat_history import InMemoryChatMessageHistory
from langgraph.checkpoint.memory import InMemorySaver
from unittest.mock import Mock, patch

from langgraph_rag import build_langgraph_rag, LangGraphConversationService
from query_rewrite import RetrievalPlan


class DateToolTests(unittest.TestCase):
    def test_date_interval_has_schema_and_calendar_rules(self):
        from date_tools import calculate_date_interval

        self.assertEqual(calculate_date_interval.name, "calculate_date_interval")
        schema = calculate_date_interval.args_schema.model_json_schema()
        self.assertEqual(set(schema["required"]), {"start_date", "end_date"})
        for start, end, days in [("2026-10-01", "2026-10-15", 14),
                                 ("2024-02-28", "2024-03-01", 2),
                                 ("2026-10-01", "2026-10-01", 0)]:
            result = calculate_date_interval.invoke(dict(start_date=start, end_date=end))
            self.assertEqual(result["days"], days)
            self.assertEqual(result["counting_rule"], "end_date - start_date，按自然日计算，不额外加一天")

    def test_invalid_dates_and_extra_arguments_are_rejected(self):
        from date_tools import calculate_date_interval

        for args in [dict(start_date="2026-02-30", end_date="2026-03-01"),
                     dict(start_date="20261001", end_date="2026-10-15"),
                     dict(start_date="2026-10-15", end_date="2026-10-01"),
                     dict(start_date="2026-10-01", end_date="2026-10-15", expression="1+1")]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                calculate_date_interval.invoke(args)

    def service(self, call, *, checkpoint=False):
        planner = Mock(return_value=AIMessage(content="", tool_calls=call))
        seen = []
        def answer(messages):
            seen.append(messages)
            return AIMessage(content="相隔14个自然日。")
        model = Mock()
        model.bind_tools.return_value = RunnableLambda(planner)
        model.invoke.side_effect = answer
        model_chain = RunnableLambda(answer)
        # Existing RAG composition accepts a Runnable. Bind only the planner API.
        model_chain.bind_tools = model.bind_tools
        rewriter = Mock()
        rewriter.rewrite.return_value = RetrievalPlan("试用期最长多久？", False)
        retrieve = Mock(return_value=[])
        rerank = Mock(side_effect=AssertionError("不应重排"))
        graph = build_langgraph_rag(
            rewriter, RunnableLambda(retrieve), RunnableLambda(rerank),
            RunnableLambda(lambda values: values["question"]), model_chain,
            enable_tools=True, **({"checkpointer": InMemorySaver()} if checkpoint else {}),
        )
        service = LangGraphConversationService(graph, InMemoryChatMessageHistory(),
            **({"thread_id": "tools"} if checkpoint else {"profile": True}))
        return service, planner, rewriter, retrieve, seen

    def call(self, name="calculate_date_interval", args=None):
        return {"id": "date-1", "name": name, "args": args or {
            "start_date": "2026-10-01", "end_date": "2026-10-15"}, "type": "tool_call"}

    def test_model_proposes_tool_python_executes_and_model_receives_result(self):
        for trace in (False, True):
            service, planner, rewriter, retrieve, seen = self.service([self.call()])
            events = []
            result = service.ask("2026年10月1日到10月15日相隔几天？",
                **({"on_node_update": lambda node, update: events.append(node)} if trace else {}))
            self.assertEqual(result["tool_result"]["days"], 14)
            self.assertEqual(result["answer"], "相隔14个自然日。")
            self.assertEqual(result["sources"], [])
            self.assertEqual(result["performance"]["model_calls"], 2)
            rewriter.rewrite.assert_not_called()
            retrieve.assert_not_called()
            self.assertEqual(len(seen), 1)
            tool_message = next(m for m in seen[0] if isinstance(m, ToolMessage))
            self.assertEqual(tool_message.tool_call_id, "date-1")
            self.assertIn('"days": 14', tool_message.content)
            if trace:
                self.assertEqual(events, ["tool_plan", "execute_tool", "tool_answer"])

    def test_no_tool_call_ignores_planner_prose_and_runs_existing_rag(self):
        service, planner, rewriter, retrieve, seen = self.service([])
        planner.return_value = AIMessage(content="模型随意说的未检索结论")
        result = service.ask("试用期最长多久？")
        self.assertEqual(result["answer"], "资料中没有足够依据。")
        self.assertEqual(result["performance"]["model_calls"], 2)
        rewriter.rewrite.assert_called_once()
        retrieve.assert_called_once()
        self.assertEqual(seen, [])

    def test_unknown_tool_multiple_calls_and_bad_dates_do_not_reach_answer_model(self):
        for calls in [[self.call("run_shell")], [self.call(), self.call()],
                      [self.call(args={"start_date": "2026-02-30", "end_date": "2026-03-01"})]]:
            with self.subTest(calls=calls):
                service, planner, rewriter, retrieve, seen = self.service(calls)
                result = service.ask("计算日期")
                self.assertIn("工具调用失败", result["answer"])
                self.assertTrue(result["tool_result"].get("error"))
                self.assertEqual(seen, [])
                retrieve.assert_not_called()

    def test_checkpoint_next_turn_does_not_reuse_previous_tool_call(self):
        service, planner, rewriter, retrieve, seen = self.service([self.call()], checkpoint=True)
        service.ask("计算日期")
        planner.return_value = AIMessage(content="", tool_calls=[])
        result = service.ask("试用期最长多久？")
        self.assertEqual(result["answer"], "资料中没有足够依据。")
        self.assertEqual(result.get("tool_messages"), [])
        self.assertIsNone(result.get("tool_result"))
        self.assertEqual(len(seen), 1)
        retrieve.assert_called_once()

    def test_tools_flag_requires_langgraph(self):
        import main
        with patch("main.ensure_legal_corpus_ready") as ensure, self.assertRaises(SystemExit):
            main.main(["--tools"])
        ensure.assert_not_called()

    @patch("main.run_cli")
    @patch("main.create_conversation_service")
    @patch("main.ensure_legal_corpus_ready")
    def test_tools_flag_selects_optional_branch(self, ensure, create, cli):
        import main
        main.main(["--langgraph", "--tools"])
        create.assert_called_once_with(decompose=False, use_langgraph=True, enable_tools=True)
