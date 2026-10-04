"""FastAPI transport for the legal RAG web chat."""

import json
import mimetypes
import os
import traceback
from _thread import LockType
from contextlib import asynccontextmanager
from threading import Lock
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field, SecretStr
from database_settings import postgres_settings, read_settings

from rag_app import create_web_rag_turn
from web_storage import SQLiteConversationStore
from request_logging import RequestLoggingMiddleware
from web_auth import AUTH_COOKIE, AuthenticationMiddleware, cookie_headers, session_data
from rag_logging import RagRequestTrace


COOKIE_NAME = "legal_rag_session"
SAFE_ERROR_MESSAGE = "暂时无法完成回答，请稍后重试。"

mimetypes.add_type("application/javascript", ".mjs")


class ChatRequest(BaseModel):
    question: str

class RenameConversationRequest(BaseModel):
    title: str


class LoginRequest(BaseModel):
    username: str = Field(max_length=32)
    password: SecretStr


def create_app(store, rag_turn, secure_cookies: bool = False, lifespan=None, authentication=None,
               frontend_directory: Path | None = None) -> FastAPI:
    """Create a web app whose state is isolated by an opaque browser cookie."""
    app = FastAPI(lifespan=lifespan)
    if authentication is not None:
        app.add_middleware(AuthenticationMiddleware, authentication=authentication, secure_cookies=secure_cookies)
    app.add_middleware(RequestLoggingMiddleware)
    static_directory = Path(__file__).parent / "web" / "static"
    app.mount("/static", StaticFiles(directory=static_directory), name="static")
    page_directory = static_directory
    if frontend_directory is not None:
        page_directory = Path(frontend_directory)
        if not (page_directory / "index.html").is_file() or not (page_directory / "assets").is_dir():
            raise RuntimeError(
                f"Frontend build is missing or incomplete at {page_directory}. "
                "Run npm ci and npm run build in frontend/ before starting the web app."
            )
        app.mount("/assets", StaticFiles(directory=page_directory / "assets"), name="frontend-assets")
    locks: dict[str, LockType] = {}
    locks_guard = Lock()

    def session_for(request: Request) -> tuple[str, bool]:
        if authentication is not None:
            return request.state.login_session.user.id, False
        supplied_id = request.cookies.get(COOKIE_NAME)
        return store.ensure_session(supplied_id), supplied_id is None

    def set_session_cookie(response, session_id: str) -> None:
        response.set_cookie(
            COOKIE_NAME,
            session_id,
            httponly=True,
            samesite="lax",
            secure=secure_cookies,
        )

    def acquire_conversation_lock(conversation_id: str) -> LockType | None:
        with locks_guard:
            lock = locks.setdefault(conversation_id, Lock())
        return lock if lock.acquire(blocking=False) else None

    @app.get("/")
    def page(request: Request):
        if authentication is not None:
            return FileResponse(page_directory / "index.html", headers={'Cache-Control': 'no-store'})
        session_id, is_new = session_for(request)
        response = FileResponse(page_directory / "index.html", headers={'Cache-Control': 'no-store'})
        if is_new:
            set_session_cookie(response, session_id)
        return response

    if authentication is not None:
        @app.exception_handler(RequestValidationError)
        async def validation_error(request, error):
            if request.url.path == '/api/auth/login':
                return JSONResponse({'detail': '请输入有效的用户名和密码'}, status_code=400, headers={'Cache-Control': 'no-store'})
            from fastapi.exception_handlers import request_validation_exception_handler
            return await request_validation_exception_handler(request, error)

        def auth_operation(request, operation):
            try:
                return operation()
            except Exception as error:
                request.state.request_error = f'Authentication operation error: {type(error).__name__}'
                raise HTTPException(503, '登录服务暂时不可用，请稍后重试') from None

        @app.post('/api/auth/login')
        def login(payload: LoginRequest, request: Request):
            session = auth_operation(request, lambda: authentication.login(payload.username, payload.password.get_secret_value()))
            if session is None:
                raise HTTPException(401, '用户名或密码不正确')
            # Replace any prior session in this browser, including a changed user.
            old_token = request.cookies.get(AUTH_COOKIE)
            if old_token:
                auth_operation(request, lambda: authentication.logout(old_token))
            response = JSONResponse(session_data(session))
            response.raw_headers.extend(cookie_headers(session, secure=secure_cookies))
            return response

        @app.get('/api/auth/me')
        def me(request: Request):
            session = auth_operation(request, lambda: authentication.lookup(request.cookies.get(AUTH_COOKIE), renew=True))
            if session is None:
                raise HTTPException(401, '请先登录')
            response = JSONResponse(session_data(session))
            response.raw_headers.extend(cookie_headers(session, secure=secure_cookies))
            return response

        @app.post('/api/auth/logout', status_code=204)
        def logout(request: Request):
            auth_operation(request, lambda: authentication.logout(request.cookies.get(AUTH_COOKIE)))
            from fastapi.responses import Response
            response = Response(status_code=204)
            response.raw_headers.extend(cookie_headers(secure=secure_cookies))
            return response

    def conversation_data(conversation):
        return {"id": conversation.id, "title": conversation.title, "updated_at": conversation.updated_at}

    @app.get("/api/conversations")
    def conversations(request: Request):
        session_id, is_new = session_for(request)
        response = JSONResponse({"conversations": [conversation_data(item) for item in store.list_conversations(session_id)]})
        if is_new:
            set_session_cookie(response, session_id)
        return response

    @app.post("/api/conversations", status_code=201)
    def create_conversation(request: Request):
        session_id, is_new = session_for(request)
        response = JSONResponse(conversation_data(store.create_conversation(session_id)), status_code=201)
        if is_new: set_session_cookie(response, session_id)
        return response

    @app.patch("/api/conversations/{conversation_id}")
    def rename_conversation(conversation_id: str, payload: RenameConversationRequest, request: Request):
        session_id, is_new = session_for(request)
        try: conversation = store.rename_conversation(session_id, conversation_id, payload.title)
        except ValueError as error: raise HTTPException(status_code=422, detail=str(error))
        if conversation is None: raise HTTPException(status_code=404, detail="对话不存在")
        response = JSONResponse(conversation_data(conversation))
        if is_new: set_session_cookie(response, session_id)
        return response

    @app.delete("/api/conversations/{conversation_id}", status_code=204)
    def delete_conversation(conversation_id: str, request: Request):
        session_id, _ = session_for(request)
        if not store.delete_conversation(session_id, conversation_id): raise HTTPException(status_code=404, detail="对话不存在")

    @app.get("/api/conversations/{conversation_id}/messages")
    def history(conversation_id: str, request: Request):
        session_id, is_new = session_for(request)
        messages = store.load_conversation_messages(session_id, conversation_id)
        if messages is None: raise HTTPException(status_code=404, detail="对话不存在")
        response = JSONResponse({"messages": [{"role": role, "content": content} for role, content in messages]})
        if is_new: set_session_cookie(response, session_id)
        return response

    @app.post("/api/conversations/{conversation_id}/chat")
    def chat(conversation_id: str, payload: ChatRequest, request: Request):
        question = payload.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="问题不能为空")

        session_id, is_new = session_for(request)
        if not store.conversation_belongs_to(session_id, conversation_id): raise HTTPException(status_code=404, detail="对话不存在")
        stream_options = {}
        trace = None
        if authentication is not None:
            from rag_permissions import validated_scope
            try:
                scope = validated_scope(authentication.accounts.allowed_knowledge_bases(session_id))
            except Exception:
                request.state.request_error = traceback.format_exc()
                raise HTTPException(status_code=503, detail='暂时无法确认资料权限，请稍后重试')
            if not scope:
                raise HTTPException(status_code=403, detail='当前账号没有可访问的资料库')
            stream_options['allowed_knowledge_bases'] = scope
            trace = RagRequestTrace(request_id=request.state.request_id, user_id=session_id,
                                    conversation_id=conversation_id, allowed_knowledge_bases=scope,
                                    logger=request.state.request_logger)
            stream_options['trace'] = trace
        session_lock = acquire_conversation_lock(conversation_id)
        if session_lock is None:
            raise HTTPException(status_code=409, detail="当前会话正在生成回答")

        def events():
            try:
                messages = _to_langchain_messages(store.load_conversation_messages(session_id, conversation_id))
                answer_parts = []
                for item in rag_turn.stream(question, messages, **stream_options):
                    event = item["event"]
                    data = item["data"]
                    if event == "delta":
                        answer_parts.append(data["text"])
                    if event == "done":
                        saved = store.save_complete_conversation_turn(session_id, conversation_id, question, "".join(answer_parts))
                        if saved is None:
                            raise RuntimeError('对话在回答期间已被删除，无法保存本次问答')
                        request.state.request_outcome = "completed"
                        if trace is not None:
                            trace.saved()
                        yield _sse(event, data)
                        return
                    if event == "error":
                        request.state.request_outcome = "failed"
                        request.state.request_error = str(data.get("message", "RAG stream reported an error"))
                        yield _sse(event, {"message": SAFE_ERROR_MESSAGE, "request_id": request.state.request_id})
                        return
                    yield _sse(event, data)
                request.state.request_outcome = "failed"
                request.state.request_error = "RAG stream ended without a done event"
                yield _sse("error", {"message": SAFE_ERROR_MESSAGE, "request_id": request.state.request_id})
            except Exception:
                request.state.request_outcome = "failed"
                request.state.request_error = traceback.format_exc()
                yield _sse("error", {"message": SAFE_ERROR_MESSAGE, "request_id": request.state.request_id})
            finally:
                session_lock.release()

        response = StreamingResponse(events(), media_type="text/event-stream")
        if is_new:
            set_session_cookie(response, session_id)
        return response

    return app


class _StartupRagTurn:
    """Delegate to the one expensive RAG runtime created during app startup."""

    def __init__(self):
        self.turn = None

    def stream(self, question, messages, **options):
        if self.turn is None:
            raise RuntimeError("RAG 服务尚未启动")
        return self.turn.stream(question, messages, **options)


class _StartupDelegate:
    def __init__(self):
        self.delegate = None

    def __getattr__(self, name):
        if self.delegate is None:
            raise RuntimeError('网页服务尚未启动')
        return getattr(self.delegate, name)


def create_default_app(database_path: Path | None = None, secure_cookies: bool | None = None):
    """Build the Uvicorn app without loading model clients until startup."""
    settings = read_settings()
    frontend = settings.get('WEB_FRONTEND', 'legacy').strip().lower()
    if frontend not in ('legacy', 'react'):
        raise ValueError('WEB_FRONTEND must be legacy or react')
    frontend_directory = Path(__file__).parent / 'frontend' / 'dist' if frontend == 'react' else None
    secure = (
        _env_flag("WEB_SECURE_COOKIES", settings)
        if secure_cookies is None
        else secure_cookies
    )
    original_store = _create_store(database_path, settings)
    # Explicit file paths preserve the existing local SQLite test/debug factory.
    authentication = _StartupDelegate() if database_path is None else None
    store = _StartupDelegate() if authentication is not None else original_store
    startup_turn = _StartupRagTurn()

    @asynccontextmanager
    async def lifespan(_app):
        if authentication is not None:
            from postgres_storage import PostgresConversationStore, PostgresUserConversationStore
            if not isinstance(original_store, PostgresConversationStore):
                raise RuntimeError('网页登录需要已迁移的 PostgreSQL 数据库')
            from account_storage import PostgresAccountStore
            from login_sessions import PostgresLoginSessions
            config = postgres_settings(settings)
            store.delegate = PostgresUserConversationStore(**config)
            authentication.delegate = PostgresLoginSessions(PostgresAccountStore(**config), **config)
        startup_turn.turn = create_web_rag_turn()
        try:
            yield
        finally:
            startup_turn.turn = None
            if authentication is not None:
                authentication.delegate = None
                store.delegate = None

    return create_app(store, startup_turn, secure, lifespan=lifespan, authentication=authentication,
                      frontend_directory=frontend_directory)


def _create_store(database_path, settings):
    # Explicit paths retain SQLite for local tools/tests regardless of deployment.
    backend = 'sqlite' if database_path is not None else settings.get('WEB_STORAGE_BACKEND', 'sqlite').strip().lower()
    if backend == 'sqlite':
        return SQLiteConversationStore(database_path or Path(settings.get('WEB_DATABASE_PATH', 'data/web_rag.sqlite3')))
    if backend != 'postgres':
        raise ValueError('WEB_STORAGE_BACKEND must be sqlite or postgres')
    from postgres_storage import PostgresConversationStore
    return PostgresConversationStore(**postgres_settings(settings))


def _to_langchain_messages(messages):
    return [
        HumanMessage(content=content) if role == "user" else AIMessage(content=content)
        for role, content in messages
    ]


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _env_flag(name: str, settings) -> bool:
    return settings.get(name, "false").lower() == "true"


app = create_default_app()
