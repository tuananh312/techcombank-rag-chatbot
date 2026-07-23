"""
FastAPI entrypoint.

Endpoints:
  GET  /health          -> liveness check
  POST /chat            -> ask a question, optionally within an existing session
  POST /session/reset   -> clear a session's history

Session state is kept in-memory (fine for a take-home demo; swap for
DynamoDB/Redis if this needed to survive restarts or scale horizontally).
"""

import uuid
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag import answer_question

app = FastAPI(title="Techcombank FY Report Chatbot")

# Allow the static frontend (served from anywhere during local dev / demo) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# session_id -> list of {"role": "user"|"assistant", "content": str}
_sessions: dict[str, list[dict]] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class Source(BaseModel):
    page: int
    snippet: str
    kind: str = "prose"


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    history = _sessions.get(session_id, [])

    result = answer_question(req.message, history)

    # persist turn for follow-up questions
    history = history + [
        {"role": "user", "content": req.message},
        {"role": "assistant", "content": result["answer"]},
    ]
    _sessions[session_id] = history

    return ChatResponse(session_id=session_id, answer=result["answer"], sources=result["sources"])


@app.post("/session/reset")
def reset_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"status": "reset"}


# --- Lambda adapter -----------------------------------------------------
# Only used when deployed behind AWS Lambda (via a container image + Function
# URL). Locally / on ECS you just run uvicorn against `app` directly.
try:
    from mangum import Mangum

    handler = Mangum(app)
except ImportError:
    handler = None
