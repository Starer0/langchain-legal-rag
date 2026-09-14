import re
from operator import itemgetter

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


ARTICLE_PATTERN = re.compile(
    r"(?m)^\s*第[一二三四五六七八九十百千万零〇0-9]+条"
)


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
    return [
        {
            "article": doc.metadata.get("article"),
            "pages": doc.metadata.get("pages", []),
            "source": doc.metadata.get("source", "未知来源"),
            "content": doc.page_content,
        }
        for doc in docs
    ]


def build_rag_chain(retriever, prompt, model):
    question_from_state = RunnableLambda(itemgetter("question"))
    docs_from_state = RunnableLambda(itemgetter("docs"))

    state = (
        {"question": RunnablePassthrough()}
        | RunnablePassthrough.assign(docs=question_from_state | retriever)
    )

    answer_chain = (
        {
            "context": docs_from_state | format_docs,
            "question": question_from_state,
        }
        | prompt
        | model
        | StrOutputParser()
    )

    return state | {
        "answer": answer_chain,
        "sources": docs_from_state | format_sources,
    }
