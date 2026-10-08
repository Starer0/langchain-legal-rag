"""Per-request RAG diagnostics, using the gateway's logger and identity."""

from contextlib import contextmanager
from time import perf_counter

from rag_permissions import accessible_documents, validated_scope
from request_logging import write_event

MAX_TEXT_CHARS = 32000
MAX_DOCUMENTS = 64
SOURCE_FIELDS = ('knowledge_base_id', 'index_status', 'law_id', 'guide_id', 'document_id',
                 'version', 'source_file', 'document_type', 'title', 'article', 'section',
                 'pages', 'rerank_score')


def logged_text(text):
    return {'text': text[:MAX_TEXT_CHARS], 'chars': len(text), 'truncated': len(text) > MAX_TEXT_CHARS}


class RagRequestTrace:
    """An explicit object per turn; never stores identity on the shared RAG runtime."""

    def __init__(self, *, request_id, user_id, conversation_id, allowed_knowledge_bases, logger):
        self.scope = validated_scope(allowed_knowledge_bases)
        self.logger = logger
        self.identity = {'request_id': request_id, 'user_id': user_id,
                         'conversation_id': conversation_id, 'allowed_knowledge_bases': sorted(self.scope)}

    def emit(self, event, **fields):
        write_event(self.logger, event, **self.identity, **fields)

    def begin(self, question, *, private=False):
        value = logged_text(question)
        if private:
            self.emit('memory_command_started', question_chars=value['chars'])
            return
        self.emit('rag_started', question=value['text'], question_chars=value['chars'],
                  question_truncated=value['truncated'])

    def documents(self, documents):
        # Only permission-checked evidence and a fixed metadata whitelist are logged.
        allowed = accessible_documents(documents, self.scope)
        result = []
        for doc in allowed[:MAX_DOCUMENTS]:
            content = logged_text(doc.page_content)
            result.append({'id': doc.id, 'content': content['text'],
                           'content_chars': content['chars'], 'content_truncated': content['truncated'],
                           **{key: doc.metadata[key] for key in SOURCE_FIELDS if key in doc.metadata}})
        return {'documents': result, 'document_count': len(allowed),
                'documents_truncated': len(allowed) > MAX_DOCUMENTS}

    @contextmanager
    def stage(self, stage):
        started = perf_counter()
        self.emit('rag_stage_started', stage=stage)
        details = {}
        try:
            yield details
        except BaseException as error:
            self.emit('rag_stage_failed', stage=stage, duration_ms=(perf_counter() - started) * 1000,
                      error_type=type(error).__name__,
                      outcome='interrupted' if isinstance(error, (GeneratorExit, KeyboardInterrupt)) else 'failed')
            raise
        else:
            self.emit('rag_stage_finished', stage=stage, duration_ms=(perf_counter() - started) * 1000, **details)

    def skipped(self, stage, reason):
        self.emit('rag_stage_skipped', stage=stage, reason=reason)

    def saved(self):
        self.emit('rag_saved')
