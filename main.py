import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path


load_dotenv()

model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")

model = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "deepseek-chat"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    temperature=0,
)

embeddings = OpenAIEmbeddings(
    model=os.getenv(
        "SILICONFLOW_EMBEDDING_MODEL",
        "BAAI/bge-m3",
    ),
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=os.getenv(
        "SILICONFLOW_BASE_URL",
        "https://api.siliconflow.cn/v1",
    ),
    check_embedding_ctx_length=False,
)

# 1. 加载 PDF
loader = PyPDFLoader("data/labor_law.pdf")
documents = loader.load()

print(f"原始文档数量：{len(documents)}")

# 2. 切分文本
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print(f"切分后的文本块数量：{len(chunks)}")

# 3. 建立向量数据库

DB_DIR = "./chroma_db"

if Path(DB_DIR).exists():
    print("加载已有向量数据库")
    vectorstore = Chroma(
        collection_name="labor_law",
        persist_directory=DB_DIR,
        embedding_function=embeddings,
    )
else:
    print("首次建立向量数据库")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="labor_law",
        persist_directory=DB_DIR,
    )

# 4. 创建检索器
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 8}
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


def format_docs(docs):
    return "\n\n".join(
        f"来源信息：{doc.metadata}\n{doc.page_content}"
        for doc in docs
    )


rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)


def ask(question: str):
    # 仅用于学习和调试：先单独查看检索结果
    docs = retriever.invoke(question)

    print("\n=== 本次检索到的文本块 ===")
    for i, doc in enumerate(docs, 1):
        print(f"\n--- 结果 {i} ---")
        print("元数据：", doc.metadata)
        print("内容：", doc.page_content[:500])

    # 仍然用 Runnable 生成最终答案
    return rag_chain.invoke(question)


if __name__ == "__main__":
    while True:
        question = input("\n请输入问题，输入 q 退出：")

        if question.lower() == "q":
            break

        answer = ask(question)
        print("\n回答：")
        print(answer)