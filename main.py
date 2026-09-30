import argparse
import json

from rag_app import (
    create_conversation_service, ensure_legal_corpus_ready, ingest_legal_corpus,
    inspect_legal_corpus, prune_stale_legal_indexes,
)


def display_node_update(node, update, output_fn=print):
    output_fn(f"\n[{node}] 完成")
    if node == "rewrite":
        output_fn(f"  retrieval_question → {update['retrieval_question']}")
        labels = {True: "需要", False: "不需要", None: "不确定"}
        output_fn(f"  include_guide → {labels[update['include_guide']]}办事指南")
    elif node == "retrieve":
        output_fn(f"  candidates → {len(update['candidates'])} 条候选资料")
    elif node == "rerank":
        output_fn(f"  docs → {len(update['docs'])} 条重排资料")
    elif node == "answer":
        output_fn(f"  answer → 已生成；sources → {len(update['sources'])} 条来源")
    elif node == "no_evidence":
        output_fn("  candidates 为空 → 直接结束；跳过重排和回答模型")
        output_fn(f"  answer → {update['answer']}")


def run_cli(service, input_fn=input, output_fn=print, *, trace=False):
    if getattr(service, "thread_id", None) is not None:
        output_fn(f"内存 checkpoint 线程：{service.thread_id}；/state 查看最新状态，/checkpoints 查看快照历史；退出后清空。")
    while True:
        question = input_fn("\n请输入问题，输入 q 退出：")
        if question.lower() == "q":
            break
        if not question.strip():
            continue

        if question.strip() in ("/state", "/checkpoints"):
            if getattr(service, "thread_id", None) is None:
                output_fn("请使用 --langgraph --checkpoint 启用快照查看。")
            else:
                value = (
                    service.inspect_checkpoint() if question.strip() == "/state"
                    else service.checkpoint_history()
                )
                output_fn(json.dumps(value, ensure_ascii=False, indent=2) if value else "当前线程尚无快照。")
            continue

        if trace:
            result = service.ask(
                question,
                on_node_update=lambda node, update: display_node_update(node, update, output_fn),
            )
        else:
            result = service.ask(question)

        output_fn(f"\n检索问题：{result['retrieval_question']}")
        if len(result.get("subquestions", [])) > 1:
            output_fn("拆分后的检索问题：")
            for index, subquestion in enumerate(result["subquestions"], start=1):
                output_fn(f"{index}. {subquestion}")
        output_fn("\n召回候选资料（重排前）：")
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
                "bm25": "BM25",
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
        "--corpus-status",
        action="store_true",
        help="检查资料是否变化，不修改知识库",
    )
    parser.add_argument(
        "--prune-stale-indexes",
        action="store_true",
        help="删除未被当前知识库使用的旧 legal_ 索引",
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
    parser.add_argument(
        "--langgraph",
        action="store_true",
        help="使用 LangGraph 状态图运行单问题 RAG 学习路径",
    )
    parser.add_argument(
        "--trace", action="store_true",
        help="配合 --langgraph 实时显示每个节点的状态更新",
    )
    parser.add_argument("--checkpoint", action="store_true", help="启用内存 checkpoint 学习模式")
    parser.add_argument("--thread-id", help="checkpoint 线程标识，默认 learning")
    args = parser.parse_args(argv)
    if args.ingest:
        return ingest_legal_corpus()
    if args.corpus_status:
        return inspect_legal_corpus()
    if args.prune_stale_indexes:
        return prune_stale_legal_indexes()
    if args.langgraph and args.decompose:
        parser.error("--langgraph 暂不支持与 --decompose 同时使用")
    if args.trace and not args.langgraph:
        parser.error("--trace 需要配合 --langgraph 使用")
    if args.checkpoint and not args.langgraph:
        parser.error("--checkpoint 需要配合 --langgraph 使用")
    if args.checkpoint and args.profile:
        parser.error("本轮 checkpoint 模式暂不支持 --profile")
    if args.thread_id is not None and (not args.checkpoint or not args.thread_id.strip()):
        parser.error("--thread-id 需要配合 --checkpoint 使用，并且不能为空")
    ensure_legal_corpus_ready()
    service_options = {"decompose": args.decompose}
    if args.profile:
        service_options["profile"] = True
    if args.langgraph:
        service_options["use_langgraph"] = True
    if args.checkpoint:
        service_options.update(checkpoint=True, thread_id=args.thread_id or "learning")
    service = create_conversation_service(**service_options)
    return run_cli(service, trace=True) if args.trace else run_cli(service)


if __name__ == "__main__":
    main()
