import json
import unittest
from unittest.mock import Mock, patch

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from langgraph_rag import build_langgraph_rag, LangGraphConversationService
from query_rewrite import RetrievalPlan


def date_call(start="2026-10-01", end="2026-10-15", call_id="date-1"):
    return AIMessage(content="", tool_calls=[{
        "id": call_id, "name": "calculate_date_interval",
        "args": {"start_date": start, "end_date": end}, "type": "tool_call",
    }])


class AgentToolLoopTests(unittest.TestCase):
    def test_loop_requests_serial_tool_calls_from_provider(self):
        from date_tools import build_date_tool_nodes
        model = Mock()
        build_date_tool_nodes(model, agent_loop=True)
        self.assertEqual(model.bind_tools.call_args.kwargs, {"parallel_tool_calls": False})

    def service(self, responses, *, checkpoint=False):
        requests, final_messages = [], []
        remaining = iter(responses)
        def plan(messages):
            requests.append(messages)
            return next(remaining)
        def final(messages):
            final_messages.append(messages)
            return AIMessage(content="两个区间分别14天和19天，第二个长5天。")
        model = RunnableLambda(final)
        model.bind_tools = lambda tools, **options: RunnableLambda(plan)
        rewriter = Mock()
        rewriter.rewrite.return_value = RetrievalPlan("试用期最长多久？", False)
        retrieve = Mock(return_value=[])
        graph = build_langgraph_rag(
            rewriter, RunnableLambda(retrieve), RunnableLambda(lambda state: state["candidates"]),
            RunnableLambda(lambda state: state["question"]), model,
            enable_tools=True, agent_loop=True,
            **({"checkpointer": InMemorySaver()} if checkpoint else {}),
        )
        service = LangGraphConversationService(graph, InMemoryChatMessageHistory(),
            **({"thread_id": "loop"} if checkpoint else {"profile": True}))
        return service, requests, final_messages, retrieve

    def test_second_tool_request_reads_first_result_then_budget_forces_final_answer(self):
        for trace in (False, True):
            with self.subTest(trace=trace):
                service, requests, finals, retrieve = self.service([
                    date_call(), date_call("2026-11-01", "2026-11-20", "date-2"),
                ])
                nodes = []
                result = service.ask("比较两个日期区间", **({"on_node_update": lambda node, update: nodes.append(node)} if trace else {}))
                self.assertEqual([r["days"] for r in result["tool_results"]], [14, 19])
                self.assertEqual(result["tool_call_count"], 2)
                self.assertEqual(result["performance"]["model_calls"], 3)
                self.assertEqual(len(requests), 2)
                first_result = next(m for m in requests[1] if isinstance(m, ToolMessage))
                self.assertEqual(first_result.tool_call_id, "date-1")
                self.assertEqual(json.loads(first_result.content)["days"], 14)
                self.assertEqual([m.tool_call_id for m in finals[0] if isinstance(m, ToolMessage)], ["date-1", "date-2"])
                retrieve.assert_not_called()
                self.assertEqual(len(service.history.messages), 2)
                if trace:
                    self.assertEqual(nodes, ["tool_plan", "execute_tool", "tool_continue", "execute_tool", "tool_answer"])

    def test_model_can_finish_after_one_tool_without_extra_final_call(self):
        service, requests, finals, _ = self.service([date_call(), AIMessage(content="相隔14天。")])
        result = service.ask("计算日期")
        self.assertEqual(result["answer"], "相隔14天。")
        self.assertEqual(result["tool_call_count"], 1)
        self.assertEqual(result["performance"]["model_calls"], 2)
        self.assertEqual(finals, [])
        self.assertEqual(len(requests), 2)

    def test_loop_no_initial_tool_request_runs_rag(self):
        service, requests, finals, retrieve = self.service([AIMessage(content="未检索的法律结论")])
        result = service.ask("试用期最长多久？")
        self.assertEqual(result["answer"], "资料中没有足够依据。")
        self.assertEqual(result["tool_results"], [])
        self.assertEqual(result["tool_call_count"], 0)
        retrieve.assert_called_once()
        self.assertEqual(finals, [])

    def test_error_on_second_call_stops_loop_and_preserves_first_result(self):
        service, requests, finals, _ = self.service([
            date_call(), date_call("2026-02-30", "2026-03-01", "bad-date"),
        ])
        result = service.ask("比较区间")
        self.assertIn("工具调用失败", result["answer"])
        self.assertEqual([r["days"] for r in result["tool_results"]], [14])
        self.assertTrue(result["tool_result"]["error"])
        saved_tool_messages = [m for m in result["tool_messages"] if isinstance(m, ToolMessage)]
        self.assertEqual([m.tool_call_id for m in saved_tool_messages], ["date-1"])
        self.assertEqual(json.loads(saved_tool_messages[0].content)["days"], 14)
        self.assertEqual(finals, [])
        self.assertEqual(len(requests), 2)

    def test_checkpoint_loop_fields_reset_for_next_rag_turn(self):
        service, requests, finals, retrieve = self.service([
            date_call(), AIMessage(content="14天。"), AIMessage(content=""),
        ], checkpoint=True)
        service.ask("日期相隔几天")
        result = service.ask("试用期最长多久？")
        self.assertEqual(result["tool_results"], [])
        self.assertEqual(result["tool_call_count"], 0)
        self.assertEqual(result["tool_messages"], [])
        retrieve.assert_called_once()

    def test_loop_flag_requires_tools(self):
        import main
        with patch("main.ensure_legal_corpus_ready") as ensure, self.assertRaises(SystemExit):
            main.main(["--langgraph", "--agent-loop"])
        ensure.assert_not_called()

    @patch("main.run_cli")
    @patch("main.create_conversation_service")
    @patch("main.ensure_legal_corpus_ready")
    def test_loop_flag_reaches_factory(self, ensure, create, cli):
        import main
        main.main(["--langgraph", "--tools", "--agent-loop"])
        create.assert_called_once_with(decompose=False, use_langgraph=True, enable_tools=True, agent_loop=True)

    def test_cli_trace_shows_two_results_and_loop_count(self):
        from main import run_cli
        service, _, _, _ = self.service([date_call(), date_call("2026-11-01", "2026-11-20", "date-2")])
        inputs = iter(["比较区间", "q"])
        output = []
        run_cli(service, input_fn=lambda _: next(inputs), output_fn=output.append, trace=True)
        text = "\n".join(output)
        self.assertIn('"days": 14', text)
        self.assertIn('"days": 19', text)
        self.assertIn("2/2", text)
        self.assertIn("模型调用：3", text)
