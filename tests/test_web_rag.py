import unittest

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from web_rag import StreamingRagTurn
from query_rewrite import RetrievalPlan


class FakeRewriter:
    def __init__(self, rewritten_question):
        self.rewritten_question = rewritten_question
        self.calls = []

    def rewrite(self, question, history, *, allowed_knowledge_bases=None):
        self.calls.append((question, list(history)))
        return RetrievalPlan(self.rewritten_question, include_guide=False)


class FakeRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.states = []

    def invoke(self, state):
        self.states.append(state)
        return self.documents


class FakeReranker:
    def __init__(self, documents):
        self.documents = documents
        self.states = []

    def invoke(self, state):
        self.states.append(state)
        return self.documents


class FakePrompt:
    def __init__(self):
        self.values = []

    def invoke(self, values):
        self.values.append(values)
        return values


class FakeStreamingModel:
    def __init__(self, chunks):
        self.chunks = chunks
        self.prompts = []

    def stream(self, prompt):
        self.prompts.append(prompt)
        return iter(self.chunks)


class StreamingRagTurnTests(unittest.TestCase):
    def setUp(self):
        self.article_20 = Document(
            page_content="第二十条 试用期工资应当符合规定。",
            metadata={"article": "第二十条", "pages": [2]},
        )
        self.rewriter = FakeRewriter("试用期工资有什么规定？")
        self.retriever = FakeRetriever([self.article_20])
        self.reranker = FakeReranker([self.article_20])
        self.prompt = FakePrompt()
        self.model = FakeStreamingModel(["应", "符合第二十条。"])
        self.turn = StreamingRagTurn(
            rewriter=self.rewriter,
            retriever=self.retriever,
            reranker=self.reranker,
            prompt=self.prompt,
            model=self.model,
            history_turns=4,
        )

    def test_stream_rewrites_with_complete_history_and_emits_ordered_events(self):
        history = [
            HumanMessage(content="试用期最长多久？"),
            AIMessage(content="最长六个月。"),
        ]

        events = list(self.turn.stream("那工资呢？", history))

        self.assertEqual(
            [event["event"] for event in events],
            ["status", "status", "status", "status", "delta", "delta", "done"],
        )
        self.assertEqual(
            "".join(
                event["data"]["text"]
                for event in events
                if event["event"] == "delta"
            ),
            "应符合第二十条。",
        )
        self.assertEqual(events[-1]["data"]["sources"][0]["article"], "第二十条")
        self.assertEqual(self.rewriter.calls, [("那工资呢？", history)])

    def test_stream_retrieves_rewrite_but_prompts_with_original_question(self):
        list(self.turn.stream("那工资呢？", []))

        self.assertEqual(
            self.retriever.states[0]["retrieval_question"],
            "试用期工资有什么规定？",
        )
        self.assertFalse(self.retriever.states[0]["include_guide"])
        self.assertEqual(self.prompt.values[0]["question"], "那工资呢？")

    def test_length_terminated_answer_never_emits_completed_legal_answer(self):
        from langchain_core.messages import AIMessageChunk
        from conversation_context import ContextTooLong
        self.model.chunks=[AIMessageChunk(content='尚未完整的结论'),AIMessageChunk(content='',response_metadata={'finish_reason':'length'})]
        events=[]
        with self.assertRaises(ContextTooLong):
            for event in self.turn.stream('工资呢',[]):events.append(event)
        self.assertNotIn('done',[e['event'] for e in events])


if __name__ == "__main__":
    unittest.main()
