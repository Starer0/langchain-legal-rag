from operator import itemgetter

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"来源信息：{doc.metadata}\n{doc.page_content}" for doc in docs
    )


def format_sources(docs: list[Document]) -> list[dict[str, object]]:
    return [
        {
            "page": doc.metadata.get("page", 0) + 1,
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
