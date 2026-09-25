"""Replay two final-evidence merge rules using saved per-question rerank lists."""

import argparse
import json
from pathlib import Path

from evaluation import _contains_all_articles


def _key(source):
    return source.get("law_id"), source.get("version"), source.get("article")


def _select(selected, source, top_n):
    if len(selected) < top_n and _key(source) not in {_key(item) for item in selected}:
        selected.append(source)


def round_robin(batches, top_n):
    selected = []
    for rank in range(max((len(batch) for batch in batches), default=0)):
        for batch in batches:
            if rank < len(batch):
                _select(selected, batch[rank], top_n)
    return selected


def diverse_law_extra(batches, top_n):
    selected = []
    for batch in batches:
        if batch:
            _select(selected, batch[0], top_n)

    represented_laws = {source.get("law_id") for source in selected}
    alternatives = [
        (source["rerank_score"] / max(batch[0]["rerank_score"], 1e-9), source)
        for batch in batches if batch
        for source in batch[1:]
        if source.get("law_id") not in represented_laws
    ]
    if alternatives and len(selected) < top_n:
        _, source = max(alternatives, key=lambda item: item[0])
        _select(selected, source, top_n)

    for source in round_robin(batches, sum(map(len, batches))):
        _select(selected, source, top_n)
    return selected


def _references(sources):
    return [f"{source.get('law_id')}:{source.get('article')}" for source in sources]


def replay(run_dir, top_n):
    cases = []
    for path in sorted(Path(run_dir).glob("*.json")):
        if path.name == "summary.json":
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        batches = [entry["sources"] for entry in result["subquestion_reranks"]]
        baseline = round_robin(batches, top_n)
        diverse = diverse_law_extra(batches, top_n)
        expected = result["case"]["expected_articles"]
        cases.append({
            "id": result["case"]["id"],
            "baseline_matches_logged": _references(baseline) == _references(result["sources"]),
            "baseline_hit": _contains_all_articles(expected, baseline),
            "diversity_hit": _contains_all_articles(expected, diverse),
            "baseline_sources": _references(baseline),
            "diversity_sources": _references(diverse),
        })
    if not cases:
        raise ValueError("没有可回放的评测记录")
    return {
        "summary": {
            "cases": len(cases),
            "baseline_matches_logged": sum(case["baseline_matches_logged"] for case in cases),
            "baseline_hits": sum(case["baseline_hit"] for case in cases),
            "diversity_hits": sum(case["diversity_hit"] for case in cases),
        },
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(replay(args.run_dir, args.top_n), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
