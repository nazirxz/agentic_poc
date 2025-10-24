from pathlib import Path
from typing import Any, Dict, List

import httpx
import orjson
from fastapi import FastAPI, HTTPException
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class OrchestratorSettings(BaseSettings):
    AGENT_STK_URL: str = "http://localhost:7001/act"
    AGENT_RTS_URL: str = "http://localhost:7002/act"
    AGENT_GENERAL_URL: str = "http://localhost:7003/chat"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5"

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = OrchestratorSettings()

app = FastAPI()


class OrchestrationRequest(BaseModel):
    question: str


async def _call_agent(url: str, domain: str, question: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(url, json={"question": question})
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=f"Agent {domain} error: {exc}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"Agent {domain} returned invalid JSON") from exc

    payload.setdefault("domain", domain)
    return payload


@tool
async def call_stk_agent(question: str) -> str:
    """Gunakan ketika pertanyaan menyangkut domain STK (TKO/TKI/TKPA/pedoman)."""
    result = await _call_agent(settings.AGENT_STK_URL, "STK", question)
    return orjson.dumps(result).decode()


@tool
async def call_rts_agent(question: str) -> str:
    """Gunakan ketika pertanyaan menyangkut domain RTS atau standar teknis."""
    result = await _call_agent(settings.AGENT_RTS_URL, "RTS", question)
    return orjson.dumps(result).decode()


@tool
async def call_general_agent(question: str) -> str:
    """Gunakan untuk pertanyaan umum, sapaan, atau obrolan yang tidak memerlukan pencarian dokumen."""
    result = await _call_agent(settings.AGENT_GENERAL_URL, "GENERAL", question)
    return orjson.dumps(result).decode()


llm = ChatOllama(
    base_url=settings.OLLAMA_BASE_URL,
    model=settings.OLLAMA_MODEL,
    temperature=0.2,
)

system_prompt = (
    "Anda adalah Orchestrator URBUDDY. Tugas Anda WAJIB memilih dan memanggil tool agent yang tepat. "
    "ATURAN PENTING:\n"
    "1. Untuk sapaan, obrolan umum, pertanyaan tentang kemampuan sistem → call_general_agent\n"
    "2. Untuk pertanyaan kategori STK (TKO, TKI, TKPA, pedoman, prosedur kerja) → call_stk_agent\n"
    "3. Untuk pertanyaan standar teknis RTS → call_rts_agent\n"
    "4. WAJIB panggil tool - JANGAN jawab sendiri\n"
    "5. Jawaban akhir HANYA JSON murni dari tool tanpa teks tambahan."
)

agent_executor = create_react_agent(
    llm,
    [call_general_agent, call_stk_agent, call_rts_agent],
    prompt=system_prompt,
)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/orchestrate")
async def orchestrate(request: OrchestrationRequest) -> dict:
    try:
        result = await agent_executor.ainvoke({"messages": [("user", request.question)]})
    except Exception as exc:  # pragma: no cover - defensive
        import traceback
        error_detail = f"Orchestrator failure: {exc}\n{traceback.format_exc()}"
        print(error_detail)  # Log to console
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    messages: List[Any] = result.get("messages", [])
    if not messages:
        raise HTTPException(status_code=502, detail="Orchestrator produced no response")

    final_msg = messages[-1]
    content = getattr(final_msg, "content", final_msg)
    if isinstance(content, list):
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)

    if not isinstance(content, str):
        content = str(content)

    # Debug logging
    print(f"Agent response content: {content}")

    try:
        payload = orjson.loads(content)
    except orjson.JSONDecodeError as exc:
        error_msg = f"Failed to parse orchestrator output as JSON. Content: {content[:500]}"
        print(error_msg)  # Log to console
        raise HTTPException(status_code=502, detail=error_msg) from exc

    return payload
