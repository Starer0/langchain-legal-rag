"""Authenticated settings endpoints; vectors stay in the memory subsystem."""
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, StrictBool, StrictInt, StrictStr
from memory_service import MemoryConflict, MemoryValidation
from memory_storage import public_document


def log_saved(request, saved):
    import logging
    from request_logging import write_event
    from memory_service import estimate_tokens
    try:
        write_event(logging.getLogger('legal_rag.requests'),'memory_saved',
            request_id=request.state.request_id,user_id=request.state.login_session.user.id,
            revision=saved['revision'],source='settings',entry_count=len(saved['entries']),
            core_estimated_tokens=estimate_tokens(saved['core_text']),extended_chars=len(saved['extended_text']))
    except Exception:
        # A diagnostics failure after commit must not turn a saved edit into a retry.
        pass


class MemoryRequest(BaseModel):
    core_text: StrictStr
    extended_text: StrictStr
    enabled: StrictBool
    expected_revision: StrictInt


def install_memory_routes(app):
    def providers():
        store, service = app.state.memory_store, app.state.memory_service
        if store is None or service is None: raise HTTPException(503, '记忆服务暂时不可用')
        return store, service

    def document(owner,saved,service):
        doc={**public_document(saved),'limits':vars(service.limits)}
        facts=getattr(app.state.memory_store,'facts',None)
        if facts is not None:doc.update(facts.snapshot(owner))
        reflection=app.state.reflection_store
        if reflection is not None:
            policy=reflection.policy(owner)
            jobs=reflection.jobs(owner)
            visible=jobs[:10]+[job for job in jobs[10:] if job['status']=='failed']
            doc.update(auto_accumulate=policy['auto_accumulate'],policy_epoch=policy['epoch'],
                       reflection_status=visible,suggestions=reflection.suggestions(owner))
        return doc

    @app.get('/api/memory')
    def read_memory(request: Request):
        store, service = providers()
        owner = request.state.login_session.user.id
        try:
            doc = document(owner,store.read(owner),service)
            return JSONResponse(doc, headers={'Cache-Control': 'no-store'})
        except Exception:
            raise HTTPException(503, '暂时无法读取记忆，请稍后重试') from None

    @app.put('/api/memory')
    def save_memory(payload: MemoryRequest, request: Request):
        store, service = providers()
        owner = request.state.login_session.user.id
        try:
            old = store.read(owner)
            if payload.expected_revision < 0 or old['revision'] != payload.expected_revision:
                raise MemoryConflict('记忆已在其他页面更新，请重新加载后编辑')
            prepared = service.prepare(old, payload.core_text, payload.extended_text, payload.enabled)
            saved = store.save(owner, prepared, payload.expected_revision)
            log_saved(request,saved)
            try:doc=document(owner,saved,service)
            except Exception:doc={**public_document(saved),'limits':vars(service.limits)}
            return JSONResponse(doc, headers={'Cache-Control': 'no-store'})
        except MemoryConflict as error: raise HTTPException(409, str(error)) from None
        except MemoryValidation as error: raise HTTPException(422, str(error)) from None
        except PermissionError: raise HTTPException(401, '请重新登录') from None
        except Exception: raise HTTPException(503, '记忆保存失败，原记录保持不变，请稍后重试') from None


def create_memory_service():
    from database_settings import read_settings
    settings = read_settings()
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from memory_service import MemoryService, MemoryLimits
    model = settings.get('SILICONFLOW_EMBEDDING_MODEL', 'BAAI/bge-m3')
    base = settings.get('SILICONFLOW_BASE_URL', 'https://api.siliconflow.cn/v1')
    embeddings = OpenAIEmbeddings(model=model, base_url=base,
        api_key=settings.get('SILICONFLOW_API_KEY'), check_embedding_ctx_length=False,
        request_timeout=8, max_retries=0)
    command_model = ChatOpenAI(model=settings.get('MODEL_NAME', 'deepseek-chat'),
        api_key=settings.get('DEEPSEEK_API_KEY'), base_url=settings.get('DEEPSEEK_BASE_URL'),
        temperature=0, timeout=20, max_retries=0)
    limits = MemoryLimits(similarity_floor=float(settings.get('MEMORY_SIMILARITY_FLOOR', '.5')))
    if not 0 <= limits.similarity_floor <= 1: raise ValueError('Invalid memory similarity floor')
    return MemoryService(embeddings, command_model, fingerprint=base.rstrip('/') + '|' + model, limits=limits)
