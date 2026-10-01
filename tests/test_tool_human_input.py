import json
import unittest
from unittest.mock import Mock, patch

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from langgraph_rag import LangGraphConversationService, build_langgraph_rag


def call(args, call_id="date-1"):
    return AIMessage(content="", tool_calls=[{
        "name": "calculate_date_interval", "args": args,
        "id": call_id, "type": "tool_call",
    }])


class ToolHumanInputTests(unittest.TestCase):
    def service(self, responses, *, trace=False, loop=False):
        remaining = iter(responses)
        self.plans, self.finals = [], []
        def plan(messages):
            self.plans.append(messages)
            return next(remaining)
        def answer(messages):
            self.finals.append(messages)
            return AIMessage(content="相隔14个自然日。")
        model = RunnableLambda(answer)
        model.bind_tools = lambda *args, **kwargs: RunnableLambda(plan)
        graph = build_langgraph_rag(
            Mock(), RunnableLambda(lambda s: self.fail("不应进入 RAG")),
            RunnableLambda(lambda s: []), RunnableLambda(lambda s: ""), model,
            enable_tools=True, tool_human_input=True, agent_loop=loop,
            checkpointer=InMemorySaver(),
        )
        return LangGraphConversationService(graph, InMemoryChatMessageHistory(), thread_id="hitl")

    def test_missing_end_pauses_and_invalid_inputs_never_execute_tool(self):
        for trace in (False, True):
            with self.subTest(trace=trace):
                service = self.service([call({"start_date": "2026-10-01"})])
                options = {"on_node_update": lambda *args: None} if trace else {}
                paused = service.ask("从2026-10-01到结束日期相隔几天？", **options)
                self.assertTrue(paused["paused"])
                self.assertEqual(paused["interrupts"][0]["field"], "end_date")
                self.assertEqual(service.history.messages, [])
                for value in (True, "", "2026-02-30", "2026-9-30", "2026-09-30"):
                    paused = service.resume(value, **options)
                    self.assertTrue(paused["paused"])
                    self.assertTrue(paused["interrupts"][0]["error"])
                    self.assertEqual(service.graph.get_state(service._checkpoint_config()).values["tool_call_count"], 0)
                    self.assertEqual(len(self.plans), 1)
                    self.assertEqual(self.finals, [])
                result = service.resume("2026-10-15", **options)
                self.assertEqual(result["tool_result"]["days"], 14)
                self.assertEqual(result["tool_call_count"], 1)
                messages = self.finals[0]
                tool_message = next(m for m in messages if isinstance(m, ToolMessage))
                request = next(m for m in messages if isinstance(m, AIMessage))
                self.assertEqual(tool_message.tool_call_id, "date-1")
                self.assertEqual(request.tool_calls[0]["args"]["end_date"], "2026-10-15")
                self.assertTrue(any(isinstance(m, HumanMessage) and "end_date=2026-10-15" in m.content for m in messages))
                self.assertIn("2026-10-15", service.history.messages[0].content)
                self.assertEqual(len(service.history.messages), 2)
                with self.assertRaisesRegex(ValueError, "没有"):
                    service.resume("2026-10-16")

    def test_two_missing_dates_are_collected_in_order(self):
        service = self.service([call({})])
        self.assertEqual(service.ask("计算日期间隔")["interrupts"][0]["field"], "start_date")
        paused = service.resume("2026-10-01")
        self.assertEqual(paused["interrupts"][0]["field"], "end_date")
        self.assertEqual(paused["interrupts"][0]["known_args"], {"start_date": "2026-10-01"})
        result = service.resume("2026-10-15")
        self.assertEqual(result["tool_result"]["days"], 14)
        self.assertEqual(len(self.plans), 1)

    def test_second_call_can_pause_without_losing_first_result_or_budget(self):
        service = self.service([
            call({"start_date": "2026-10-01", "end_date": "2026-10-15"}),
            call({"start_date": "2026-11-01"}, "date-2"),
        ], loop=True)
        self.assertTrue(service.ask("比较两个区间")["paused"])
        result = service.resume("2026-11-20")
        self.assertEqual([r["days"] for r in result["tool_results"]], [14, 19])
        self.assertEqual(result["tool_call_count"], 2)
        self.assertEqual(len(self.plans), 2)
        self.assertEqual([m.tool_call_id for m in result["tool_messages"] if isinstance(m, ToolMessage)], ["date-1", "date-2"])

    def test_malformed_requests_are_rejected_without_clarification(self):
        for args in ({"start_date": "2026-10-01", "extra": "x"}, {"start_date": "2026-02-30"}):
            with self.subTest(args=args):
                service = self.service([call(args)])
                result = service.ask("计算日期间隔")
                self.assertFalse(result.get("paused", False))
                self.assertIn("error", result["tool_result"])
                self.assertEqual(result["tool_call_count"], 0)

    def test_cli_accepts_resume_value_and_displays_validation_error(self):
        from main import run_cli
        service = self.service([call({"start_date": "2026-10-01"})])
        inputs = iter(["计算日期间隔", "/resume 2026-02-30", "/state", "/resume 2026-10-15", "q"])
        output = []
        run_cli(service, input_fn=lambda _: next(inputs), output_fn=output.append, trace=True)
        text = "\n".join(output)
        self.assertIn("补充", text)
        self.assertIn("有效", text)
        self.assertIn('"days": 14', text)
        self.assertEqual(len(service.history.messages), 2)

    def test_flag_requires_tools_and_checkpoint_before_loading_corpus(self):
        import main
        for flags in (["--langgraph", "--tool-human-input"], ["--langgraph", "--tools", "--tool-human-input"]):
            with patch("main.ensure_legal_corpus_ready") as ensure, self.assertRaises(SystemExit):
                main.main(flags)
            ensure.assert_not_called()

    @patch("main.run_cli")
    @patch("main.create_conversation_service")
    @patch("main.ensure_legal_corpus_ready")
    def test_flag_reaches_factory(self, ensure, create, cli):
        import main
        main.main(["--langgraph", "--tools", "--checkpoint", "--tool-human-input"])
        self.assertTrue(create.call_args.kwargs["tool_human_input"])
