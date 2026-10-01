"""An explicit, single-question RAG state graph for learning LangGraph."""

from time import perf_counter
from typing import Any, TypedDict

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from conversation import recent_complete_turns
from date_tools import MAX_TOOL_CALLS, build_date_tool_nodes, collect_tool_input
from performance import TurnProfile, measure
from rag_pipeline_articles import format_docs, format_sources


class RagState(TypedDict, total=False):
    question: str
    history: list
    retrieval_question: str
    include_guide: bool | None
    candidates: list
    docs: list
    answer: str
    sources: list[dict]
    _profile: Any
    tool_request: Any
    tool_messages: list
    tool_result: dict | None
    tool_results: list[dict]
    tool_call_count: int
    tool_call_limit: int
    tool_input_error: str | None
    tool_input_answers: list[dict]


def build_langgraph_rag(
    rewriter, retriever, reranker, prompt, model, *, checkpointer=None,
    pause_after_retrieve=False, enable_tools=False, agent_loop=False,
    tool_human_input=False,
):
    """Compile single-question RAG with an empty-candidate exit."""
    if pause_after_retrieve and checkpointer is None:
        raise ValueError("检索后暂停需要 checkpoint")
    if agent_loop and not enable_tools:
        raise ValueError("工具循环需要启用 tools")
    if tool_human_input and (not enable_tools or checkpointer is None):
        raise ValueError("工具补参数需要 tools 和 checkpoint")
    answer_chain = (
        {
            "context": RunnableLambda(lambda state: format_docs(state["docs"])),
            "question": RunnableLambda(lambda state: state["question"]),
        }
        | prompt
        | model
        | StrOutputParser()
    )

    def rewrite(state: RagState):
        plan = measure(
            state, "rewrite",
            lambda: rewriter.rewrite(state["question"], state.get("history", [])),
        )
        return {
            "retrieval_question": plan.retrieval_question,
            "include_guide": plan.include_guide,
        }

    def retrieve(state: RagState):
        return {"candidates": retriever.invoke(state)}

    def rerank(state: RagState):
        return {"docs": reranker.invoke(state)}

    def route_candidates(state: RagState):
        if not state["candidates"]:
            return "no_evidence"
        return "review" if pause_after_retrieve else "rerank"

    def review(state: RagState):
        # This node restarts on resume. Keep paid/external operations in other nodes.
        interrupt({
            "message": "检索已完成，输入 /resume 继续重排和回答。",
            "candidate_count": len(state["candidates"]),
        })
        return {}

    def no_evidence(state: RagState):
        return {
            "answer": "资料中没有足够依据。",
            "candidates": [],
            "docs": [],
            "sources": [],
        }

    def answer(state: RagState):
        documents = state["docs"]
        return {
            "answer": measure(state, "answer", lambda: answer_chain.invoke(state)),
            "candidates": format_sources(state["candidates"]),
            "sources": format_sources(documents),
        }

    graph = StateGraph(RagState)
    graph.add_node("rewrite", rewrite)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("answer", answer)
    graph.add_node("no_evidence", no_evidence)
    graph.add_node("review", review)
    if enable_tools:
        tool_plan, execute_tool, tool_answer, tool_continue = build_date_tool_nodes(
            model, agent_loop=agent_loop, tool_human_input=tool_human_input,
        )
        next_tool_node = "collect_tool_input" if tool_human_input else "execute_tool"
        if tool_human_input:
            graph.add_node("collect_tool_input", collect_tool_input)
            graph.add_edge("collect_tool_input", "execute_tool")
        graph.add_node("tool_plan", tool_plan)
        graph.add_node("execute_tool", execute_tool)
        graph.add_node("tool_answer", tool_answer)
        graph.add_edge(START, "tool_plan")
        graph.add_conditional_edges(
            "tool_plan", lambda state: next_tool_node if state.get("tool_request") is not None else "rewrite",
            {next_tool_node: next_tool_node, "rewrite": "rewrite"},
        )
        if agent_loop:
            graph.add_node("tool_continue", tool_continue)
            graph.add_conditional_edges(
                "execute_tool",
                lambda state: "tool_answer" if state["tool_result"].get("error") or state["tool_call_count"] >= MAX_TOOL_CALLS else "tool_continue",
                {"tool_answer": "tool_answer", "tool_continue": "tool_continue"},
            )
            graph.add_conditional_edges(
                "tool_continue", lambda state: next_tool_node if state.get("tool_request") is not None else END,
                {next_tool_node: next_tool_node, END: END},
            )
        else:
            graph.add_edge("execute_tool", "tool_answer")
        graph.add_edge("tool_answer", END)
    else:
        graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_conditional_edges(
        "retrieve", route_candidates,
        {"rerank": "rerank", "review": "review", "no_evidence": "no_evidence"},
    )
    graph.add_edge("review", "rerank")
    graph.add_edge("rerank", "answer")
    graph.add_edge("answer", END)
    graph.add_edge("no_evidence", END)
    return graph.compile(checkpointer=checkpointer)


class LangGraphConversationService:
    """Keep the existing chat-history contract while using the compiled graph."""

    def __init__(
        self, graph, history: BaseChatMessageHistory, max_turns: int = 4,
        profile: bool = False, thread_id: str | None = None,
        checkpoint_resources=None, checkpoint_db: str | None = None,
    ):
        if thread_id is not None and (not thread_id.strip() or profile):
            raise ValueError("checkpoint 需要非空 thread_id，本轮暂不支持 --profile")
        self.graph = graph
        self.history = history
        self.max_turns = max_turns
        self.profile = profile
        self.thread_id = thread_id
        self.checkpoint_db = checkpoint_db
        self._checkpoint_resources = checkpoint_resources

    def close(self):
        """Release the SQLite connection owned by the service factory."""
        if self._checkpoint_resources is not None:
            self._checkpoint_resources.close()

    def _checkpoint_config(self):
        if self.thread_id is None:
            raise ValueError("请使用 --langgraph --checkpoint 启用快照查看")
        return {"configurable": {"thread_id": self.thread_id}}

    def inspect_checkpoint(self):
        snapshot = self.graph.get_state(self._checkpoint_config())
        return _checkpoint_summary(snapshot) if snapshot.values else None

    def checkpoint_history(self):
        return [
            _checkpoint_summary(snapshot)
            for snapshot in self.graph.get_state_history(self._checkpoint_config())
        ]

    def _execute(self, graph_input, initial_state, on_node_update=None):
        config_options = {"config": self._checkpoint_config()} if self.thread_id is not None else {}
        if on_node_update is None:
            result = dict(self.graph.invoke(graph_input, **config_options))
        else:
            result = dict(initial_state)
            for event in self.graph.stream(graph_input, stream_mode="updates", **config_options):
                for node, update in event.items():
                    if node == "__interrupt__":
                        continue
                    # A node returning {} is surfaced as None by updates streaming.
                    update = update or {}
                    result.update(update)
                    on_node_update(node, update)
        if self.thread_id is not None:
            snapshot = self.graph.get_state(self._checkpoint_config())
            result = dict(snapshot.values)
            pending = _interrupt_values(snapshot)
            if pending:
                return {
                    "paused": True, "interrupts": pending,
                    "question": result["question"],
                    "retrieval_question": result.get("retrieval_question", ""),
                    "candidates": format_sources(result.get("candidates", [])),
                }
        return result

    def _save_completed_turn(self, question, answer):
        self.history.add_user_message(question)
        self.history.add_ai_message(answer)
        self.history.messages[:] = recent_complete_turns(self.history.messages, self.max_turns)

    def _completed_question(self, result, original_question=None):
        question = result.get("question", original_question)
        for item in result.get("tool_input_answers", []):
            values = ", ".join(f"{k}={v}" for k, v in item["args"].items())
            question += f"\n用户补充（{item['tool_call_id']}）：{values}"
        return question

    def resume(self, value=True, *, on_node_update=None):
        snapshot = self.graph.get_state(self._checkpoint_config())
        if not _interrupt_values(snapshot):
            raise ValueError("当前线程没有等待继续的暂停任务。")
        if _interrupt_values(snapshot)[0].get("kind") != "tool_input" and value is not True:
            raise ValueError("检索审核暂停请使用不带参数的 /resume。")
        result = self._execute(Command(resume=value), snapshot.values, on_node_update)
        if not result.get("paused"):
            self._save_completed_turn(self._completed_question(result, snapshot.values["question"]), result["answer"])
        return result

    def ask(self, question: str, *, on_node_update=None) -> dict:
        original_question = question.strip()
        if not original_question:
            raise ValueError("问题不能为空")
        if self.thread_id is not None:
            snapshot = self.graph.get_state(self._checkpoint_config())
            if _interrupt_values(snapshot):
                raise ValueError("当前线程有暂停任务，请先输入 /resume，或退出后换一个 thread_id。")

        started = perf_counter()
        profile = TurnProfile() if self.profile else None
        state = {
            "question": original_question,
            "history": recent_complete_turns(self.history.messages, self.max_turns),
        }
        if profile:
            state["_profile"] = profile
        if self.thread_id is not None:
            # Each question starts a new turn; do not retain previous answer fields.
            state.update(
                retrieval_question="", include_guide=None,
                candidates=[], docs=[], answer="", sources=[],
                tool_request=None, tool_messages=[], tool_result=None,
                tool_results=[], tool_call_count=0,
                tool_call_limit=0,
                tool_input_error=None, tool_input_answers=[],
            )
        def invoke():
            return self._execute(state, state, on_node_update)

        result = dict(profile.measure("pipeline", invoke) if profile else invoke())
        if not result.get("paused"):
            self._save_completed_turn(self._completed_question(result, original_question), result["answer"])
        if profile:
            result["performance"] = profile.snapshot(
                total_ms=(perf_counter() - started) * 1000
            )
        return result


def _interrupt_values(snapshot):
    return [item.value for task in snapshot.tasks for item in task.interrupts]


def _checkpoint_summary(snapshot):
    values = snapshot.values
    return {
        "checkpoint_id": snapshot.config["configurable"].get("checkpoint_id"),
        "step": (snapshot.metadata or {}).get("step"),
        "next": list(snapshot.next),
        "interrupts": _interrupt_values(snapshot),
        "question": values.get("question"),
        "retrieval_question": values.get("retrieval_question"),
        "include_guide": values.get("include_guide"),
        "candidate_count": len(values.get("candidates", [])),
        "document_count": len(values.get("docs", [])),
        "answer": values.get("answer", ""),
        "tool_call_count": values.get("tool_call_count", 0),
        "tool_call_limit": values.get("tool_call_limit", 0),
        "tool_input_answers": values.get("tool_input_answers", []),
    }
