from rag_app import create_rag_chain


rag_chain = create_rag_chain()


if __name__ == "__main__":
    while True:
        question = input("\n请输入问题，输入 q 退出：")
        if question.lower() == "q":
            break

        result = rag_chain.invoke(question)

        print("\nChroma 召回的候选法条（重排前）：")
        for index, candidate in enumerate(result["candidates"], start=1):
            pages = ", ".join(str(page) for page in candidate["pages"])
            print(f"{index}. {candidate['article']}，PDF 第 {pages} 页")

        print("\n回答：")
        print(result["answer"])

        print("\n参考来源（Reranker 重排后）：")
        for source in result["sources"]:
            pages = ", ".join(str(page) for page in source["pages"])
            content = source["content"].replace("\n", " ")
            summary = content[:160]

            if len(content) > 160:
                summary += "……"
            score = source.get("rerank_score")
            score_text = (
                f"，相关度：{score:.4f}"
                if isinstance(score, (int, float))
                else ""
            )
            print(
                f"- {source['article']}，PDF 第 {pages} 页"
                f"{score_text}：{summary}"
            )
