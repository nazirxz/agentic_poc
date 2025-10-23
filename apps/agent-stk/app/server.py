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


async def run_rag(question: str, collection: str | None = None) -> dict:
    initial_state: RAGState = {"question": question}
    if collection:
        initial_state["collection"] = collection

    result = await graph.ainvoke(initial_state)

    answer = result.get("answer") or settings.REFUSAL_TEXT
    citations = result.get("citations") or []
    diagnostic = dict(result.get("diag") or {})
    if result.get("collection"):
        diagnostic.setdefault("collection", result["collection"])

    return {
        "domain": settings.DOMAIN,
        "answer": answer,
        "citations": citations,
        "diagnostic": diagnostic,
    }


@tool
async def answer_stk_auto(question: str) -> str:
    """Gunakan pemilihan otomatis (pedoman, TKO, TKI, TKPA) untuk menjawab."""
    return orjson.dumps(await run_rag(question)).decode()


@tool
async def answer_stk_pedoman(question: str) -> str:
    """Gunakan koleksi pedoman/manual untuk pertanyaan kebijakan umum."""
    return orjson.dumps(await run_rag(question, collection="pedoman")).decode()


@tool
async def answer_stk_tko(question: str) -> str:
    """Gunakan koleksi TKO (tata kerja organisasi, prosedur kerja)."""
    return orjson.dumps(await run_rag(question, collection="TKO")).decode()


@tool
async def answer_stk_tki(question: str) -> str:
    """Gunakan koleksi TKI (tata kerja individu)."""
    return orjson.dumps(await run_rag(question, collection="TKI")).decode()


@tool
async def answer_stk_tkpa(question: str) -> str:
    """Gunakan koleksi TKPA (instruksi penggunaan alat)."""
    return orjson.dumps(await run_rag(question, collection="TKPA")).decode()


system_prompt = (
    "Anda Agent STK. Analisis pertanyaan dan pilih tool yang paling relevan. "
    "Gunakan tool pedoman untuk panduan umum, TKO untuk prosedur kerja organisasi, "
    "TKI untuk tata kerja individu, TKPA untuk instruksi alat. Tool auto akan memilih otomatis bila ragu. "
    "Gunakan tool minimal satu kali dan jangan membuat jawaban sendiri. "
    "Jawaban akhir wajib berupa JSON persis seperti yang dikembalikan tool, tanpa tambahan teks."
)

agent_executor = create_react_agent(
    llm,
    [answer_stk_auto, answer_stk_pedoman, answer_stk_tko, answer_stk_tki, answer_stk_tkpa],
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
