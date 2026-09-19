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
        top_n: int = 3,
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
    full_text = ""
    page_ranges = []

    for doc in documents:
        start = len(full_text)
        full_text += doc.page_content.rstrip() + "\n"
        page_ranges.append({
            "start": start,
            "end": len(full_text),
            "page": doc.metadata.get("page", 0),
            "source": doc.metadata.get("source", "未知来源"),
        })

    matches = list(ARTICLE_PATTERN.finditer(full_text))
    articles = []

    for index, match in enumerate(matches):
        start = match.start()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(full_text)
        )
        content = full_text[start:end].strip()

        if not content:
            continue

        pages = []
        source = "未知来源"
        for page_info in page_ranges:
            if start < page_info["end"] and end > page_info["start"]:
                pages.append(page_info["page"] + 1)
                source = page_info["source"]

        articles.append(Document(
            page_content=content,
            metadata={
                "article": match.group(0).strip(),
                "pages": sorted(set(pages)),
                "source": source,
                "chunk_type": "article",
            },
        ))

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
            "pages": doc.metadata.get("pages", []),
            "source": doc.metadata.get("source", "未知来源"),
            "content": doc.page_content,
        }

        if "rerank_score" in doc.metadata:
            source["rerank_score"] = doc.metadata["rerank_score"]

        sources.append(source)

    return sources


def _normalize_rag_input(value):
    if isinstance(value, str):
        return {
            "question": value,
            "retrieval_question": value,
        }
    return value


def build_rag_chain(retriever, reranker, prompt, model):
    original_question_from_state = RunnableLambda(itemgetter("question"))
    retrieval_question_from_state = RunnableLambda(itemgetter("retrieval_question"))
    candidates_from_state = RunnableLambda(itemgetter("candidates"))
    docs_from_state = RunnableLambda(itemgetter("docs"))

    state = (
        RunnableLambda(_normalize_rag_input)
        | RunnablePassthrough.assign(
            candidates=retrieval_question_from_state | retriever
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
