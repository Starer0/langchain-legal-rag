"""An explicit, single-question RAG state graph for learning LangGraph."""

from time import perf_counter
from typing import Any, TypedDict

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph

from conversation import recent_complete_turns
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


def build_langgraph_rag(rewriter, retriever, reranker, prompt, model, *, checkpointer=None):
    """Compile single-question RAG with an empty-candidate exit."""
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
        return "rerank" if state["candidates"] else "no_evidence"

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
    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_conditional_edges(
        "retrieve", route_candidates,
        {"rerank": "rerank", "no_evidence": "no_evidence"},
    )
    graph.add_edge("rerank", "answer")
    graph.add_edge("answer", END)
    graph.add_edge("no_evidence", END)
    return graph.compile(checkpointer=checkpointer)


class LangGraphConversationService:
    """Keep the existing chat-history contract while using the compiled graph."""

    def __init__(
        self, graph, history: BaseChatMessageHistory, max_turns: int = 4,
        profile: bool = False, thread_id: str | None = None,
    ):
        if thread_id is not None and (not thread_id.strip() or profile):
            raise ValueError("checkpoint 需要非空 thread_id，本轮暂不支持 --profile")
        self.graph = graph
        self.history = history
        self.max_turns = max_turns
        self.profile = profile
        self.thread_id = thread_id

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

    def ask(self, question: str, *, on_node_update=None) -> dict:
        original_question = question.strip()
        if not original_question:
            raise ValueError("问题不能为空")

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
            )
        config_options = {"config": self._checkpoint_config()} if self.thread_id is not None else {}
        def invoke():
            if on_node_update is None:
                return self.graph.invoke(state, **config_options)
            result = dict(state)
            for event in self.graph.stream(state, stream_mode="updates", **config_options):
                for node, update in event.items():
                    result.update(update)
                    on_node_update(node, update)
            return result

        result = dict(profile.measure("pipeline", invoke) if profile else invoke())
        self.history.add_user_message(original_question)
        self.history.add_ai_message(result["answer"])
        self.history.messages[:] = recent_complete_turns(
            self.history.messages, self.max_turns
        )
        if profile:
            result["performance"] = profile.snapshot(
                total_ms=(perf_counter() - started) * 1000
            )
        return result


def _checkpoint_summary(snapshot):
    values = snapshot.values
    return {
        "checkpoint_id": snapshot.config["configurable"].get("checkpoint_id"),
        "step": (snapshot.metadata or {}).get("step"),
        "next": list(snapshot.next),
        "question": values.get("question"),
        "retrieval_question": values.get("retrieval_question"),
        "include_guide": values.get("include_guide"),
        "candidate_count": len(values.get("candidates", [])),
        "document_count": len(values.get("docs", [])),
        "answer": values.get("answer", ""),
    }
