"""FastAPI transport for the legal RAG web chat."""

import json
import os
from _thread import LockType
from contextlib import asynccontextmanager
from threading import Lock
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from rag_app import create_web_rag_turn
from web_storage import SQLiteConversationStore


COOKIE_NAME = "legal_rag_session"
SAFE_ERROR_MESSAGE = "暂时无法完成回答，请稍后重试。"


class ChatRequest(BaseModel):
    question: str


def create_app(store, rag_turn, secure_cookies: bool = False, lifespan=None) -> FastAPI:
    """Create a web app whose state is isolated by an opaque browser cookie."""
    app = FastAPI(lifespan=lifespan)
    static_directory = Path(__file__).parent / "web" / "static"
    app.mount("/static", StaticFiles(directory=static_directory), name="static")
    locks: dict[str, LockType] = {}
    locks_guard = Lock()

    def session_for(request: Request) -> tuple[str, bool]:
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

    def acquire_session_lock(session_id: str) -> LockType | None:
        with locks_guard:
            lock = locks.setdefault(session_id, Lock())
        return lock if lock.acquire(blocking=False) else None

    @app.get("/")
    def page(request: Request):
        session_id, is_new = session_for(request)
        response = FileResponse(static_directory / "index.html")
        if is_new:
            set_session_cookie(response, session_id)
        return response

    @app.get("/api/history")
    def history(request: Request):
        session_id, is_new = session_for(request)
        response = JSONResponse({
            "messages": [
                {"role": role, "content": content}
                for role, content in store.load_messages(session_id)
            ]
        })
        if is_new:
            set_session_cookie(response, session_id)
        return response

    @app.post("/api/chat")
    def chat(payload: ChatRequest, request: Request):
        question = payload.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="问题不能为空")

        session_id, is_new = session_for(request)
        session_lock = acquire_session_lock(session_id)
        if session_lock is None:
            raise HTTPException(status_code=409, detail="当前会话正在生成回答")

        def events():
            try:
                messages = _to_langchain_messages(store.load_messages(session_id))
                answer_parts = []
                for item in rag_turn.stream(question, messages):
                    event = item["event"]
                    data = item["data"]
                    if event == "delta":
                        answer_parts.append(data["text"])
                    if event == "done":
                        store.save_complete_turn(session_id, question, "".join(answer_parts))
                        yield _sse(event, data)
                        return
                    if event == "error":
                        yield _sse(event, data)
                        return
                    yield _sse(event, data)
                yield _sse("error", {"message": SAFE_ERROR_MESSAGE})
            except Exception:
                yield _sse("error", {"message": SAFE_ERROR_MESSAGE})
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

    def stream(self, question, messages):
        if self.turn is None:
            raise RuntimeError("RAG 服务尚未启动")
        return self.turn.stream(question, messages)


def create_default_app(database_path: Path | None = None, secure_cookies: bool | None = None):
    """Build the Uvicorn app without loading model clients until startup."""
    path = database_path or Path(os.getenv("WEB_DATABASE_PATH", "data/web_rag.sqlite3"))
    secure = (
        _env_flag("WEB_SECURE_COOKIES")
        if secure_cookies is None
        else secure_cookies
    )
    store = SQLiteConversationStore(path)
    startup_turn = _StartupRagTurn()

    @asynccontextmanager
    async def lifespan(_app):
        startup_turn.turn = create_web_rag_turn()
        try:
            yield
        finally:
            startup_turn.turn = None

    return create_app(store, startup_turn, secure, lifespan=lifespan)


def _to_langchain_messages(messages):
    return [
        HumanMessage(content=content) if role == "user" else AIMessage(content=content)
        for role, content in messages
    ]


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _env_flag(name: str) -> bool:
    return os.getenv(name, "false").lower() == "true"


app = create_default_app()
