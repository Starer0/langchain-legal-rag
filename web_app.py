"""FastAPI transport for the legal RAG web chat."""

import json
from _thread import LockType
from threading import Lock
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel


COOKIE_NAME = "legal_rag_session"
SAFE_ERROR_MESSAGE = "暂时无法完成回答，请稍后重试。"


class ChatRequest(BaseModel):
    question: str


def create_app(store, rag_turn, secure_cookies: bool = False) -> FastAPI:
    """Create a web app whose state is isolated by an opaque browser cookie."""
    app = FastAPI()
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


def _to_langchain_messages(messages):
    return [
        HumanMessage(content=content) if role == "user" else AIMessage(content=content)
        for role, content in messages
    ]


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
