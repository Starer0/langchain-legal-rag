import json
import re
from operator import itemgetter

import requests
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


ARTICLE_PATTERN = re.compile(
    r"(?m)^\s*第[一二三四五六七八九十百千万零〇0-9]+条"
)


class SiliconFlowReranker:
    def __init__(
        self,
        api_key: str | None,
        base_url: str = "https://api.siliconflow.cn/v1",
        model: str = "BAAI/bge-reranker-v2-m3",
        top_n: int = 4,
        timeout: int = 30,
        post=requests.post,
    ):
        if top_n < 1:
            raise ValueError("top_n 必须大于等于 1")

        self.api_key = api_key
        self.endpoint = f"{base_url.rstrip('/')}/rerank"
        self.model = model
        self.top_n = top_n
        self.timeout = timeout
        self.post = post

    def rerank(
        self,
        question: str,
        documents: list[Document],
    ) -> list[Document]:
        if not documents:
            return []
        if not self.api_key:
            raise ValueError("缺少 SILICONFLOW_API_KEY，无法调用 Reranker")

        response = self.post(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "query": question,
                "documents": [doc.page_content for doc in documents],
                "return_documents": False,
                "top_n": min(self.top_n, len(documents)),
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results")

        if not isinstance(results, list):
            raise RuntimeError("Reranker 响应中缺少 results 列表")

        reranked_documents = []
        for result in results:
            index = result.get("index")
            score = result.get("relevance_score")

            if not isinstance(index, int) or not 0 <= index < len(documents):
                raise RuntimeError(f"Reranker 返回了无效文档索引：{index}")

            document = documents[index]
            metadata = dict(document.metadata)
            metadata["rerank_score"] = score
            reranked_documents.append(Document(
                page_content=document.page_content,
                metadata=metadata,
            ))

        return reranked_documents


def split_by_articles(documents: list[Document]) -> list[Document]:
    articles = []
    current = None
    chapter = ""
    for doc in documents:
        page = doc.metadata.get("page", 0) + 1
        for line in doc.page_content.splitlines():
            line = line.strip()
            if not line:
                continue
            if re.match(r"^第[一二三四五六七八九十百0-9]+章", line):
                chapter = line
                current = None
                continue
            if re.match(r"^第[一二三四五六七八九十百0-9]+节", line):
                current = None
                continue
            match = ARTICLE_PATTERN.match(line)
            if match:
                current = Document(
                    page_content=line,
                    metadata={
                        "article": match.group(0).strip(),
                        "chapter": chapter,
                        "pages": [page],
                        "source": doc.metadata.get("source", "未知来源"),
                        "chunk_type": "article",
                    },
                )
                articles.append(current)
            elif current is not None:
                current.page_content += "\n" + line
                if page not in current.metadata["pages"]:
                    current.metadata["pages"].append(page)
    return articles


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"来源信息：{doc.metadata}\n{doc.page_content}" for doc in docs
    )


def format_sources(docs: list[Document]) -> list[dict[str, object]]:
    sources = []

    for doc in docs:
        source = {
            "article": doc.metadata.get("article"),
            "pages": (
                json.loads(doc.metadata["pages"])
                if isinstance(doc.metadata.get("pages"), str)
                else doc.metadata.get("pages", [])
            ),
            "source": doc.metadata.get("source", "未知来源"),
            "content": doc.page_content,
        }
        for field in (
            "law_id",
            "law_name",
            "chapter",
            "version",
            "effective_date",
            "status",
            "source_file",
            "source_url",
        ):
            if field in doc.metadata:
                source[field] = doc.metadata[field]

        if "rerank_score" in doc.metadata:
            source["rerank_score"] = doc.metadata["rerank_score"]

        sources.append(source)

    return sources


def _document_key(doc: Document):
    metadata = doc.metadata
    if metadata.get("law_id") and metadata.get("article"):
        return (metadata["law_id"], metadata.get("version"), metadata["article"])
    return (metadata.get("source"), metadata.get("article"), doc.page_content)


class CompositeRagChain:
    """Retrieve and rerank each distinct intent before one final answer."""

    def __init__(self, single_chain, retriever, reranker, answer_chain, top_n):
        if top_n < 1:
            raise ValueError("top_n 必须大于等于 1")
        self.single_chain = single_chain
        self.retriever = retriever
        self.reranker = reranker
        self.answer_chain = answer_chain
        self.top_n = top_n

    def invoke(self, state):
        questions = state.get("retrieval_questions", [])
        if len(questions) < 2:
            return self.single_chain.invoke(state)

        candidates = []
        candidate_keys = set()
        ranked_batches = []
        for question in questions:
            substate = {"question": question, "retrieval_question": question}
            batch = self.retriever.invoke(substate)
            for doc in batch:
                key = _document_key(doc)
                if key not in candidate_keys:
                    candidates.append(doc)
                    candidate_keys.add(key)
            ranked_batches.append(self.reranker.invoke({**substate, "candidates": batch}))

        selected = []
        selected_keys = set()
        for rank in range(max((len(batch) for batch in ranked_batches), default=0)):
            for batch in ranked_batches:
                if rank >= len(batch):
                    continue
                doc = batch[rank]
                key = _document_key(doc)
                if key not in selected_keys:
                    selected.append(doc)
                    selected_keys.add(key)
                if len(selected) >= self.top_n:
                    break
            if len(selected) >= self.top_n:
                break

        answer = self.answer_chain.invoke({"question": state["question"], "docs": selected})
        return {
            "answer": answer,
            "candidates": format_sources(candidates),
            "sources": format_sources(selected),
        }


def _normalize_rag_input(value):
    if isinstance(value, str):
        return {
            "question": value,
            "retrieval_question": value,
        }
    return value


def build_rag_chain(retriever, reranker, prompt, model, stateful_retriever=False):
    original_question_from_state = RunnableLambda(itemgetter("question"))
    retrieval_question_from_state = RunnableLambda(itemgetter("retrieval_question"))
    candidates_from_state = RunnableLambda(itemgetter("candidates"))
    docs_from_state = RunnableLambda(itemgetter("docs"))

    candidate_retriever = (
        retriever
        if stateful_retriever
        else retrieval_question_from_state | retriever
    )
    state = (
        RunnableLambda(_normalize_rag_input)
        | RunnablePassthrough.assign(
            candidates=candidate_retriever
        )
        | RunnablePassthrough.assign(docs=reranker)
    )

    answer_chain = (
        {
            "context": docs_from_state | format_docs,
            "question": original_question_from_state,
        }
        | prompt
        | model
        | StrOutputParser()
    )

    return state | {
        "answer": answer_chain,
        "candidates": candidates_from_state | format_sources,
        "sources": docs_from_state | format_sources,
    }
