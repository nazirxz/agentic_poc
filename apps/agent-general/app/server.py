from __future__ import annotations

import orjson
from fastapi import FastAPI, HTTPException
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from .settings import AgentSettings

settings = AgentSettings()

llm = ChatOllama(
    base_url=settings.OLLAMA_BASE_URL,
    model=settings.OLLAMA_MODEL,
    temperature=0.7,
)

app = FastAPI()


class ChatRequest(BaseModel):
    question: str


async def chat_general(question: str) -> dict:
    """Handle general chat tanpa RAG - langsung ke LLM."""
    system_prompt = (
        "Anda adalah asisten URBUDDY yang ramah dan membantu. "
        "Jawab pertanyaan dengan sopan dan informatif dalam bahasa Indonesia. "
        "Jika ditanya tentang kemampuan, jelaskan bahwa Anda dapat membantu dengan pertanyaan umum, "
        "serta memberikan informasi dari dokumen STK dan RTS jika diperlukan."
    )
    
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        
        response = await llm.ainvoke(messages)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        return {
            "domain": settings.DOMAIN,
            "answer": answer,
            "citations": [],
            "diagnostic": {
                "mode": "general_chat",
                "model": settings.OLLAMA_MODEL,
            }
        }
    except Exception as exc:
        import traceback
        print(f"General chat error: {exc}\n{traceback.format_exc()}")
        return {
            "domain": settings.DOMAIN,
            "answer": "Maaf, saya mengalami kesulitan memproses pertanyaan Anda saat ini.",
            "citations": [],
            "diagnostic": {"error": str(exc)}
        }


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "domain": settings.DOMAIN}


@app.post("/chat")
async def chat(payload: ChatRequest) -> dict:
    """Endpoint untuk general chat."""
    try:
        result = await chat_general(payload.question)
        print(f"Agent GENERAL response: {orjson.dumps(result).decode()}")
        return result
    except Exception as exc:
        import traceback
        error_detail = f"Agent GENERAL failure: {exc}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

