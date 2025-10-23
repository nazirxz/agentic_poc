from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .graph import RAGState, build_graph
from .settings import AgentSettings

settings = AgentSettings()
graph = build_graph(settings)

app = FastAPI()


class ActRequest(BaseModel):
    question: str


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "domain": settings.DOMAIN}


@app.post("/act")
async def act(payload: ActRequest) -> dict:
    initial_state: RAGState = {"question": payload.question}
    try:
        result = await graph.ainvoke(initial_state)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    answer = result.get("answer") or settings.REFUSAL_TEXT
    citations = result.get("citations") or []
    diagnostic = result.get("diag") or {}

    return {
        "domain": settings.DOMAIN,
        "answer": answer,
        "citations": citations,
        "diagnostic": diagnostic,
    }
