import os

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from conversation import ConversationRagService
from legal_corpus import ingest_corpus, open_corpus, resolve_filter
from query_rewrite import RetrievalQuestionRewriter

from rag_pipeline_articles import (
    SiliconFlowReranker,
    build_rag_chain,
)


def _create_embeddings():
    return OpenAIEmbeddings(
        model=os.getenv("SILICONFLOW_EMBEDDING_MODEL", "BAAI/bge-m3"),
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        check_embedding_ctx_length=False,
    )


def _embedding_config():
    return {
        "model": os.getenv("SILICONFLOW_EMBEDDING_MODEL", "BAAI/bge-m3"),
        "base_url": os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
    }


def ingest_legal_corpus():
    """Build or verify the versioned multi-law corpus without starting the CLI."""
    load_dotenv()
    manifest = ingest_corpus(_create_embeddings(), _embedding_config())
    print(f"法律知识库已就绪：{manifest['article_count']} 条")
    print(f"分法律数量：{manifest['law_counts']}")
    return manifest


def create_rag_chain():
    load_dotenv()
    model = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0,
    )
    embeddings = _create_embeddings()
    vectorstore, manifest, laws = open_corpus(embeddings, _embedding_config())
    print(f"加载法律知识库：{manifest['article_count']} 条，{len(laws)} 部法律")

    candidate_k = int(os.getenv("RETRIEVAL_K", "8"))
    rerank_top_n = int(os.getenv("RERANK_TOP_N", "4"))
    use_reranker = os.getenv("USE_RERANKER", "true").lower() == "true"
    filter_enabled = os.getenv("METADATA_FILTER", "true").lower() == "true"
    retriever = RunnableLambda(
        lambda state: vectorstore.similarity_search(
            state["retrieval_question"],
            k=candidate_k,
            filter=resolve_filter(state, laws, enabled=filter_enabled),
        )
    )

    if use_reranker:
        reranker_client = SiliconFlowReranker(
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
            model=os.getenv("SILICONFLOW_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
            top_n=rerank_top_n,
        )
        reranker = RunnableLambda(
            lambda state: reranker_client.rerank(
                state["retrieval_question"], state["candidates"]
            )
        )
    else:
        reranker = RunnableLambda(lambda state: state["candidates"])
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
        reranker,
        prompt,
        model,
        stateful_retriever=True,
    )


def create_conversation_service():
    """Create the CLI service with bounded in-memory conversation history."""
    load_dotenv()
    history_turns = int(os.getenv("HISTORY_TURNS", "4"))
    rewrite_model = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0,
    )
    return ConversationRagService(
        rag_chain=create_rag_chain(),
        rewriter=RetrievalQuestionRewriter(rewrite_model),
        history=InMemoryChatMessageHistory(),
        max_turns=history_turns,
    )
