"""Build the Chroma-only control chain used by the local evaluation."""

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from legal_corpus import open_corpus, resolve_filter
from rag_app import _create_embeddings, _embedding_config
from rag_pipeline_articles import build_rag_chain


def create_chroma_only_chain():
    """Return the shared RAG chain without a reranking stage."""
    load_dotenv()

    model = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0,
    )
    vectorstore, manifest, laws = open_corpus(
        _create_embeddings(),
        _embedding_config(),
    )
    print(f"加载法律知识库：{manifest['article_count']} 条，{len(laws)} 部法律")
    retriever = RunnableLambda(
        lambda state: vectorstore.similarity_search(
            state["retrieval_question"],
            k=int(os.getenv("RETRIEVAL_K", "8")),
            filter=resolve_filter(
                state,
                laws,
                enabled=os.getenv("METADATA_FILTER", "true").lower() == "true",
            ),
        )
    )
    direct_documents = RunnableLambda(lambda state: state["candidates"])
    prompt = ChatPromptTemplate.from_template("""
你是一名劳动法律法规知识问答助手。

请严格根据参考资料回答问题。
如果资料中没有足够依据，请明确说“资料中没有足够依据”，不要自行编造。

参考资料：
{context}

用户问题：
{question}

请给出清晰、谨慎的回答，并尽可能引用相关条文或页码。
""")

    return build_rag_chain(
        retriever,
        direct_documents,
        prompt,
        model,
        stateful_retriever=True,
    )
