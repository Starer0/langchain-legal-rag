import os

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from conversation import ConversationRagService
from evidence_selector import EvidenceSelector
from legal_corpus import ingest_corpus, load_guides, open_corpus, resolve_filter
from performance import measure
from query_decomposition import CompositeQuestionDecomposer
from query_rewrite import RetrievalQuestionRewriter

from rag_pipeline_articles import (
    SiliconFlowReranker,
    CompositeRagChain,
    build_rag_chain,
    format_docs,
)
from web_rag import StreamingRagTurn


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


def create_rag_chain(
    metadata_filter=None, decompose=False, profile=False, evidence_selection=False,
):
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
    filter_enabled = (
        os.getenv("METADATA_FILTER", "true").lower() == "true"
        if metadata_filter is None
        else metadata_filter
    )
    retriever = RunnableLambda(
        lambda state: measure(
            state, "chroma",
            lambda: vectorstore.similarity_search(
                state["retrieval_question"],
                k=candidate_k,
                filter=resolve_filter(state, laws, enabled=filter_enabled),
            ),
        )
    )

    if use_reranker:
        reranker_client = SiliconFlowReranker(
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
            model=os.getenv("SILICONFLOW_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
            top_n=rerank_top_n,
        )
        def rerank(state):
            options = (
                {"top_n": state["rerank_top_n"]}
                if "rerank_top_n" in state else {}
            )
            operation = lambda: reranker_client.rerank(
                state["retrieval_question"], state["candidates"], **options
            )
            # No candidates means no remote Reranker request.
            return measure(state, "rerank", operation) if state["candidates"] else operation()

        reranker = RunnableLambda(rerank)
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
    composite_prompt = ChatPromptTemplate.from_template("""
你是一名劳动法律法规知识问答助手。

请按参考资料中的子问题顺序分别回答。每个部分只能使用该子问题组内的资料；
不得把其他子问题组的资料作为本部分依据。某一组资料不足以支持回答时，
请对该部分明确说“资料中没有足够依据”，不要自行编造。

按子问题分组的参考资料：
{context}

用户问题：
{question}

请给出清晰、谨慎的分项回答，并尽可能引用相关条文或页码。
""")

    single_chain = build_rag_chain(
        retriever,
        reranker,
        prompt,
        model,
        stateful_retriever=True,
        **({"profile": True} if profile else {}),
    )
    if not decompose:
        return single_chain

    answer_chain = (
        {
            "context": RunnableLambda(lambda state: state["context"]),
            "question": RunnableLambda(lambda state: state["question"]),
        }
        | composite_prompt
        | model
        | StrOutputParser()
    )
    evidence_selector = None
    if evidence_selection:
        evidence_model = ChatOpenAI(
            model=os.getenv(
                "EVIDENCE_SELECTOR_MODEL", os.getenv("MODEL_NAME", "deepseek-chat")
            ),
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL"),
            temperature=0,
        )
        evidence_selector = EvidenceSelector(evidence_model)
    return CompositeRagChain(
        single_chain,
        retriever,
        reranker,
        answer_chain,
        top_n=rerank_top_n,
        evidence_selector=evidence_selector,
    )


def create_conversation_service(decompose=False, profile=False, evidence_selection=False):
    """Create the CLI service with bounded in-memory conversation history."""
    load_dotenv()
    history_turns = int(os.getenv("HISTORY_TURNS", "4"))
    rewrite_model = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0,
    )
    chain_options = {"decompose": True} if decompose else {}
    if profile:
        chain_options["profile"] = True
    if evidence_selection:
        chain_options["evidence_selection"] = True
    return ConversationRagService(
        rag_chain=create_rag_chain(**chain_options),
        rewriter=RetrievalQuestionRewriter(rewrite_model, load_guides()),
        history=InMemoryChatMessageHistory(),
        max_turns=history_turns,
        decomposer=CompositeQuestionDecomposer(rewrite_model) if decompose else None,
        profile=profile,
    )


def create_web_rag_turn():
    """Create the shared, single-query RAG turn used by the web application."""
    load_dotenv()
    history_turns = int(os.getenv("HISTORY_TURNS", "4"))
    model_options = {
        "model": os.getenv("MODEL_NAME", "deepseek-chat"),
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": os.getenv("DEEPSEEK_BASE_URL"),
        "temperature": 0,
    }
    answer_model = ChatOpenAI(**model_options)
    rewrite_model = ChatOpenAI(**model_options)
    embeddings = _create_embeddings()
    vectorstore, manifest, laws = open_corpus(embeddings, _embedding_config())
    print(f"加载法律知识库：{manifest['article_count']} 条，{len(laws)} 部法律")

    candidate_k = int(os.getenv("RETRIEVAL_K", "8"))
    rerank_top_n = int(os.getenv("RERANK_TOP_N", "4"))
    filter_enabled = os.getenv("METADATA_FILTER", "true").lower() == "true"
    retriever = RunnableLambda(
        lambda state: measure(
            state,
            "chroma",
            lambda: vectorstore.similarity_search(
                state["retrieval_question"],
                k=candidate_k,
                filter=resolve_filter(state, laws, enabled=filter_enabled),
            ),
        )
    )

    use_reranker = os.getenv("USE_RERANKER", "true").lower() == "true"
    if use_reranker:
        reranker_client = SiliconFlowReranker(
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
            model=os.getenv("SILICONFLOW_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
            top_n=rerank_top_n,
        )

        def rerank(state):
            operation = lambda: reranker_client.rerank(
                state["retrieval_question"], state["candidates"]
            )
            return measure(state, "rerank", operation) if state["candidates"] else operation()

        reranker = RunnableLambda(rerank)
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
    return StreamingRagTurn(
        rewriter=RetrievalQuestionRewriter(rewrite_model, load_guides()),
        retriever=retriever,
        reranker=reranker,
        prompt=prompt,
        model=answer_model,
        history_turns=history_turns,
    )
