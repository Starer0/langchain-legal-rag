import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_pipeline import build_rag_chain


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

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(documents)

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

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 8},
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

rag_chain = build_rag_chain(retriever, prompt, model)


if __name__ == "__main__":
    while True:
        question = input("\n请输入问题，输入 q 退出：")
        if question.lower() == "q":
            break

        result = rag_chain.invoke(question)

        print("\n回答：")
        print(result["answer"])

        print("\n参考来源：")
        for source in result["sources"]:
            summary = source["content"].replace("\n", " ")[:120]
            print(f"- PDF 第 {source['page']} 页：{summary}")
