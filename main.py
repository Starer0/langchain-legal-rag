import argparse

from rag_app import create_conversation_service, ingest_legal_corpus


def run_cli(service, input_fn=input, output_fn=print):
    while True:
        question = input_fn("\n请输入问题，输入 q 退出：")
        if question.lower() == "q":
            break
        if not question.strip():
            continue

        result = service.ask(question)

        output_fn(f"\n检索问题：{result['retrieval_question']}")
        if len(result.get("subquestions", [])) > 1:
            output_fn("拆分后的检索问题：")
            for index, subquestion in enumerate(result["subquestions"], start=1):
                output_fn(f"{index}. {subquestion}")
        output_fn("\nChroma 召回的候选法条（重排前）：")
        for index, candidate in enumerate(result["candidates"], start=1):
            pages = ", ".join(str(page) for page in candidate["pages"])
            law_name = candidate.get("law_name", "")
            law_prefix = f"{law_name} " if law_name else ""
            output_fn(f"{index}. {law_prefix}{candidate['article']}，PDF 第 {pages} 页")

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
                f"- {source.get('law_name', '') + ' ' if source.get('law_name') else ''}"
                f"{source['article']}，PDF 第 {pages} 页"
                f"{score_text}：{summary}"
            )
        performance = result.get("performance")
        if performance:
            output_fn(f"\n总耗时：{performance['total_ms']:.2f} ms")
            labels = {
                "rewrite": "改写",
                "decompose": "拆分",
                "chroma": "Chroma",
                "rerank": "Reranker",
                "answer": "回答",
            }
            for stage, label in labels.items():
                if stage in performance["stages"]:
                    item = performance["stages"][stage]
                    output_fn(
                        f"{label}：{item['duration_ms']:.2f} ms"
                        f"（{item['calls']} 次）"
                    )
            output_fn(
                f"模型调用：{performance['model_calls']} 次；"
                f"Reranker 调用：{performance['reranker_calls']} 次"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description="法律 RAG 命令行工具")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="导入或校验 data/laws.json 中登记的法律 PDF",
    )
    parser.add_argument(
        "--decompose",
        action="store_true",
        help="对复合问题分别检索与重排，再统一回答",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="显示本轮各阶段耗时和调用次数",
    )
    args = parser.parse_args(argv)
    if args.ingest:
        return ingest_legal_corpus()
    service_options = {"decompose": args.decompose}
    if args.profile:
        service_options["profile"] = True
    return run_cli(create_conversation_service(**service_options))


if __name__ == "__main__":
    main()
