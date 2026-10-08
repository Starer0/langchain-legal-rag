"""Authenticated control and review of private background memory jobs."""
import time
from fastapi import Request,HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel,StrictBool,StrictInt,ConfigDict
from memory_service import MemoryConflict,MemoryValidation
from memory_storage import public_document


class AutomationRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    enabled:StrictBool
    expected_epoch:StrictInt


class SuggestionRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    expected_revision:StrictInt
    expected_epoch:StrictInt


def install_reflection_routes(app):
    def call(request,operation):
        store=app.state.reflection_store
        if store is None:raise HTTPException(503,'自动记忆服务暂时不可用。')
        owner=request.state.login_session.user.id
        try:return JSONResponse(operation(store,owner),headers={'Cache-Control':'no-store'})
        except PermissionError:raise HTTPException(404,'该记录不存在。') from None
        except MemoryConflict as error:raise HTTPException(409,str(error)) from None
        except MemoryValidation as error:raise HTTPException(422,str(error)) from None
        except Exception:raise HTTPException(503,'暂时无法处理自动记忆，请稍后重试。') from None

    @app.put('/api/memory/automation')
    def automation(payload:AutomationRequest,request:Request):
        return call(request,lambda store,owner:store.set_policy(owner,payload.enabled,payload.expected_epoch))

    @app.get('/api/memory/suggestions')
    def suggestions(request:Request):
        return call(request,lambda store,owner:{'suggestions':store.suggestions(owner)})

    @app.post('/api/memory/suggestions/{sid}/accept')
    def accept(sid:str,payload:SuggestionRequest,request:Request):
        return call(request,lambda store,owner:public_document(store.accept(owner,sid,payload.expected_revision,payload.expected_epoch,app.state.memory_service)))

    @app.post('/api/memory/suggestions/{sid}/ignore')
    def ignore(sid:str,request:Request):return call(request,lambda store,owner:store.ignore(owner,sid))

    @app.post('/api/memory/reflection/{job_id}/retry')
    def retry(job_id:str,request:Request):return call(request,lambda store,owner:store.retry(owner,job_id,time.time()))
