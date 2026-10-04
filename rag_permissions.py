"""Server-owned knowledge-base scope and evidence checks for web RAG."""

import json

from corpus_metadata import knowledge_base_id


def validated_scope(values):
    if not isinstance(values, (set, frozenset, list, tuple)):
        raise ValueError('缺少有效的资料权限范围')
    return frozenset(knowledge_base_id({'knowledge_base_id': value}) for value in values)


def scoped_filter(metadata_filter, scope):
    return {'$and': [metadata_filter, {'knowledge_base_id': {'$in': sorted(scope)}}]}


def accessible_documents(documents, scope):
    return [doc for doc in documents if doc.metadata.get('knowledge_base_id') in scope
            and doc.metadata.get('index_status') == 'active']


def evidence_identity(doc):
    metadata = {key: value for key, value in doc.metadata.items() if key != 'rerank_score'}
    return doc.page_content, json.dumps(metadata, sort_keys=True, ensure_ascii=False)


def verified_reranked_documents(documents, candidates, scope):
    identities = {evidence_identity(doc) for doc in candidates}
    return [doc for doc in accessible_documents(documents, scope) if evidence_identity(doc) in identities]
