import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag_pipeline_articles import (
    SiliconFlowReranker,
    build_rag_chain,
    split_by_articles,
)


def create_rag_chain():
    from langchain_community.document_loaders import PyPDFLoader

    load_dotenv()

    model = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=0,
    )
    embeddings = OpenAIEmbeddings(
        model=os.getenv("SILICONFLOW_EMBEDDING_MODEL", "BAAI/bge-m3"),
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        check_embedding_ctx_length=False,
    )

    loader = PyPDFLoader("data/labor_law.pdf")
    documents = loader.load()
    chunks = split_by_articles(documents)
    print(f"按法律条文切分后的 chunks 数量：{len(chunks)}")

    db_dir = "./chroma_articles_db"
    if Path(db_dir).exists():
        print("加载已有法律条文向量数据库")
        vectorstore = Chroma(
            collection_name="labor_law_articles",
            persist_directory=db_dir,
            embedding_function=embeddings,
        )
    else:
        print("首次建立法律条文向量数据库")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="labor_law_articles",
            persist_directory=db_dir,
        )

    candidate_k = int(os.getenv("RETRIEVAL_K", "8"))
    rerank_top_n = int(os.getenv("RERANK_TOP_N", "3"))
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": candidate_k},
    )

    reranker_client = SiliconFlowReranker(
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        model=os.getenv("SILICONFLOW_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
        top_n=rerank_top_n,
    )
    reranker = RunnableLambda(
        lambda state: reranker_client.rerank(
            state["question"], state["candidates"]
        )
    )

    prompt = ChatPromptTemplate.from_template("""
你是一名劳动法知识问答助手。

请严格根据参考资料回答问题。
如果资料中没有足够依据，请明确说“资料中没有足够依据”，不要自行编造。

参考资料：
{context}

用户问题：
{question}

请给出清晰、谨慎的回答，并尽可能引用相关条文或页码。
""")

    return build_rag_chain(retriever, reranker, prompt, model)
