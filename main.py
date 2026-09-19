from rag_app import create_conversation_service


def run_cli(service, input_fn=input, output_fn=print):
    while True:
        question = input_fn("\n请输入问题，输入 q 退出：")
        if question.lower() == "q":
            break
        if not question.strip():
            continue

        result = service.ask(question)

        output_fn(f"\n检索问题：{result['retrieval_question']}")
        output_fn("\nChroma 召回的候选法条（重排前）：")
        for index, candidate in enumerate(result["candidates"], start=1):
            pages = ", ".join(str(page) for page in candidate["pages"])
            output_fn(f"{index}. {candidate['article']}，PDF 第 {pages} 页")

        output_fn("\n回答：")
        output_fn(result["answer"])

        output_fn("\n参考来源（Reranker 重排后）：")
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
            output_fn(
                f"- {source['article']}，PDF 第 {pages} 页"
                f"{score_text}：{summary}"
            )


if __name__ == "__main__":
    run_cli(create_conversation_service())
