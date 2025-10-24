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
    """Cari jawaban dari database dokumen RTS untuk pertanyaan tentang standar teknis.
    Return JSON: {"domain":"RTS", "answer":"...", "citations":[...], "diagnostic":{...}}
    """
    return orjson.dumps(await run_rag(question)).decode()


# ReAct agent untuk RTS dengan tool description yang jelas
# Tambahkan system prompt yang lebih eksplisit untuk memastikan penggunaan tool
system_prompt = """Anda adalah Agent RTS yang bertugas menjawab pertanyaan tentang standar teknis RTS.

ATURAN PENTING:
1. SELALU gunakan tool answer_rts_general untuk menjawab pertanyaan
2. JANGAN memberikan jawaban langsung tanpa menggunakan tool
3. Tool akan mengembalikan JSON dengan format: {"domain":"RTS", "answer":"...", "citations":[...], "diagnostic":{...}}
4. Jika tool mengembalikan JSON, kembalikan JSON tersebut sebagai jawaban final
5. Jika terjadi error, laporkan error tersebut

Contoh penggunaan:
User: "berapa nilai bil di rts?"
Assistant: Saya akan mencari informasi tentang nilai bil di RTS menggunakan database dokumen.
Action: answer_rts_general
Action Input: {"question": "berapa nilai bil di rts?"}
Observation: {"domain":"RTS", "answer":"Nilai bil di RTS adalah...", "citations":["doc.pdf p.5"], "diagnostic":{...}}
Final Answer: {"domain":"RTS", "answer":"Nilai bil di RTS adalah...", "citations":["doc.pdf p.5"], "diagnostic":{...}}
"""

agent_executor = create_react_agent(
    llm,
    [answer_rts_general],
    state_modifier=system_prompt,
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
        result = await agent_executor.ainvoke(
            {"messages": [("user", payload.question)]},
            config={
                "recursion_limit": 8,
                "max_iterations": 3,
            }
        )
    except Exception as exc:  # pragma: no cover - defensive
        import traceback
        error_detail = f"Agent RTS failure: {exc}\n{traceback.format_exc()}"
        print(error_detail)
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

    print(f"Agent RTS response: {content}")

    try:
        payload_json = orjson.loads(content)
    except orjson.JSONDecodeError as exc:
        error_msg = f"Output agent tidak dapat dibaca sebagai JSON. Content: {content[:500]}"
        print(error_msg)
        raise HTTPException(status_code=502, detail=error_msg) from exc

    return payload_json
