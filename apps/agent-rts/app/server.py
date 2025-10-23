from __future__ import annotations

import orjson
from fastapi import FastAPI, HTTPException
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from .graph import RAGState, build_graph
from .settings import AgentSettings

settings = AgentSettings()
graph = build_graph(settings)

llm = ChatOllama(
    base_url=settings.OLLAMA_BASE_URL,
    model=settings.OLLAMA_MODEL,
    temperature=0.2,
)


async def run_rag(question: str) -> dict:
    initial_state: RAGState = {"question": question}
    result = await graph.ainvoke(initial_state)

    answer = result.get("answer") or settings.REFUSAL_TEXT
    citations = result.get("citations") or []
    diagnostic = dict(result.get("diag") or {})

    return {
        "domain": settings.DOMAIN,
        "answer": answer,
        "citations": citations,
        "diagnostic": diagnostic,
    }


@tool
async def answer_rts_general(question: str) -> str:
    """Gunakan untuk pertanyaan standar teknis RTS secara umum."""
    return orjson.dumps(await run_rag(question)).decode()


system_prompt = (
    "Anda Agent RTS. Gunakan tool yang tersedia untuk memperoleh jawaban berbasis dokumen RTS. "
    "Analisis pertanyaan langkah demi langkah, panggil tool untuk mendapatkan jawaban, dan jangan membuat jawaban sendiri. "
    "Jawaban akhir harus berupa JSON persis seperti keluaran tool." 
)

agent_executor = create_react_agent(
    llm,
    [answer_rts_general],
    prompt=system_prompt,
)

app = FastAPI()


class ActRequest(BaseModel):
    question: str


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "domain": settings.DOMAIN}


@app.post("/act")
async def act(payload: ActRequest) -> dict:
    try:
        result = await agent_executor.ainvoke({"messages": [("user", payload.question)]})
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    messages = result.get("messages", [])
    if not messages:
        raise HTTPException(status_code=502, detail="Agent tidak menghasilkan respons")

    final_msg = messages[-1]
    content = getattr(final_msg, "content", final_msg)
    if isinstance(content, list):
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)

    if not isinstance(content, str):
        content = str(content)

    try:
        payload_json = orjson.loads(content)
    except orjson.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="Output agent tidak dapat dibaca sebagai JSON") from exc

    return payload_json
