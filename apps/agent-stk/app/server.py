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

# Create LLM with system prompt
system_prompt = """Anda adalah Agent STK yang bertugas menjawab pertanyaan tentang dokumen STK (Sistem Tata Kerja).

ATURAN PENTING:
1. SELALU gunakan salah satu tool yang tersedia untuk menjawab pertanyaan
2. JANGAN memberikan jawaban langsung tanpa menggunakan tool
3. Tool akan mengembalikan JSON dengan format: {"domain":"STK", "answer":"...", "citations":[...], "diagnostic":{...}}
4. Jika tool mengembalikan JSON, kembalikan JSON tersebut sebagai jawaban final
5. Jika terjadi error, laporkan error tersebut

PILIHAN TOOL:
- answer_stk_auto: Untuk pertanyaan umum atau jika tidak yakin kategori dokumen
- answer_stk_pedoman: Untuk kebijakan umum atau panduan
- answer_stk_tko: Untuk prosedur kerja dan tata kerja organisasi
- answer_stk_tki: Untuk tata kerja individu
- answer_stk_tkpa: Untuk instruksi penggunaan alat/peralatan

Contoh penggunaan:
User: "bagaimana prosedur kerja organisasi?"
Assistant: Saya akan mencari informasi tentang prosedur kerja organisasi di dokumen STK.
Action: answer_stk_tko
Action Input: {"question": "bagaimana prosedur kerja organisasi?"}
Observation: {"domain":"STK", "answer":"Prosedur kerja organisasi adalah...", "citations":["doc.pdf p.5"], "diagnostic":{...}}
Final Answer: {"domain":"STK", "answer":"Prosedur kerja organisasi adalah...", "citations":["doc.pdf p.5"], "diagnostic":{...}}
"""

llm = ChatOllama(
    base_url=settings.OLLAMA_BASE_URL,
    model=settings.OLLAMA_MODEL,
    temperature=0.2,
    system=system_prompt,
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
    """Pilihan otomatis - sistem akan memilih koleksi (pedoman/TKO/TKI/TKPA) yang paling sesuai.
    Gunakan ini jika tidak yakin kategori dokumen yang tepat.
    Return JSON: {"domain":"STK", "answer":"...", "citations":[...], "diagnostic":{...}}
    """
    return orjson.dumps(await run_rag(question)).decode()


@tool
async def answer_stk_pedoman(question: str) -> str:
    """Cari di koleksi PEDOMAN/MANUAL untuk pertanyaan tentang kebijakan umum atau panduan.
    Return JSON: {"domain":"STK", "answer":"...", "citations":[...], "diagnostic":{...}}
    """
    return orjson.dumps(await run_rag(question, collection="pedoman")).decode()


@tool
async def answer_stk_tko(question: str) -> str:
    """Cari di koleksi TKO (Tata Kerja Organisasi) untuk prosedur kerja dan tata kerja organisasi.
    Return JSON: {"domain":"STK", "answer":"...", "citations":[...], "diagnostic":{...}}
    """
    return orjson.dumps(await run_rag(question, collection="TKO")).decode()


@tool
async def answer_stk_tki(question: str) -> str:
    """Cari di koleksi TKI (Tata Kerja Individu) untuk tata kerja individu.
    Return JSON: {"domain":"STK", "answer":"...", "citations":[...], "diagnostic":{...}}
    """
    return orjson.dumps(await run_rag(question, collection="TKI")).decode()


@tool
async def answer_stk_tkpa(question: str) -> str:
    """Cari di koleksi TKPA (Tata Kerja Penggunaan Alat) untuk instruksi penggunaan alat/peralatan.
    Return JSON: {"domain":"STK", "answer":"...", "citations":[...], "diagnostic":{...}}
    """
    return orjson.dumps(await run_rag(question, collection="TKPA")).decode()


# ReAct agent untuk STK dengan tool descriptions yang jelas
agent_executor = create_react_agent(
    llm,
    [answer_stk_auto, answer_stk_pedoman, answer_stk_tko, answer_stk_tki, answer_stk_tkpa],
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
        error_detail = f"Agent STK failure: {exc}\n{traceback.format_exc()}"
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

    print(f"Agent STK response: {content}")

    try:
        payload_json = orjson.loads(content)
    except orjson.JSONDecodeError as exc:
        error_msg = f"Output agent tidak dapat dibaca sebagai JSON. Content: {content[:500]}"
        print(error_msg)
        raise HTTPException(status_code=502, detail=error_msg) from exc

    return payload_json
