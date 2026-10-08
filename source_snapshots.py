"""JSON-safe, display-only snapshots of the final answer's reference sources."""

import json


FIELDS = frozenset(('article', 'pages', 'source', 'content', 'law_id', 'law_name',
                    'chapter', 'version', 'effective_date', 'status', 'source_file',
                    'source_url', 'document_id', 'document_type', 'title', 'section',
                    'chunk_kind', 'rerank_score', 'knowledge_base_id'))


def snapshot_sources(sources):
    if sources is None:
        return []
    if not isinstance(sources, list) or any(not isinstance(item, dict) for item in sources):
        raise ValueError('Sources must be a list of source objects')
    # Drop unrelated fields and sever references to mutable RAG objects.
    return json.loads(json.dumps([{key: value for key, value in item.items() if key in FIELDS}
                                 for item in sources], ensure_ascii=False, allow_nan=False))


def history_message(role, content, sources=None):
    message = {'role': role, 'content': content}
    if role == 'assistant' and sources:
        message['sources'] = sources
    return message
