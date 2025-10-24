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
    OLLAMA_MODEL: str = "qwen3:0.6b"

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
async def call_general_agent(question: str) -> str:
    """ONLY use for general greetings, casual chat, or questions about system capabilities.
    DO NOT use for technical questions about documents.
    Use for: 'halo', 'hello', 'apa kabar', 'siapa kamu', 'apa yang bisa kamu lakukan'.
    This agent does NOT search documents, only conversational AI.
    Return format JSON: {"domain": "GENERAL", "answer": "...", "citations": [], "diagnostic": {...}}
    """
    print(f"DEBUG: Routing to GENERAL agent for: {question}")
    result = await _call_agent(settings.AGENT_GENERAL_URL, "GENERAL", question)
    return orjson.dumps(result).decode()


@tool
async def call_stk_agent(question: str) -> str:
    """Use for questions about STK documents (organizational procedures, work instructions).
    Use for: TKO, TKI, TKPA, pedoman, prosedur kerja, tata kerja organisasi/individu, instruksi alat.
    This agent searches STK document database with RAG.
    Return format JSON: {"domain": "STK", "answer": "...", "citations": ["doc.pdf p.5"], "diagnostic": {...}}
    """
    print(f"DEBUG: Routing to STK agent for: {question}")
    result = await _call_agent(settings.AGENT_STK_URL, "STK", question)
    return orjson.dumps(result).decode()


@tool
async def call_rts_agent(question: str) -> str:
    """Use for questions about RTS technical standards and specifications.
    Use for: RTS, rokan technical standard, standar teknis, nilai bil, technical standards, specifications.
    This agent searches RTS document database with RAG.
    Return format JSON: {"domain": "RTS", "answer": "...", "citations": ["doc.pdf p.5"], "diagnostic": {...}}
    """
    print(f"DEBUG: Routing to RTS agent for: {question}")
    result = await _call_agent(settings.AGENT_RTS_URL, "RTS", question)
    return orjson.dumps(result).decode()


# System prompt untuk routing yang tepat
system_prompt = """You are a routing agent that MUST choose the correct tool based on the question content.

ROUTING RULES:
1. If question contains "RTS", "rokan technical standard", "standar teknis", "nilai bil", "technical standard" -> use call_rts_agent
2. If question contains "STK", "TKO", "TKI", "TKPA", "pedoman", "prosedur kerja", "tata kerja" -> use call_stk_agent  
3. If question is general greeting, casual chat, or asking about capabilities -> use call_general_agent

CRITICAL: 
- "nilai bil di rokan technical standard" = RTS domain
- "berapa nilai bil" in RTS context = RTS domain
- Always choose the most specific tool for the question domain
- Do NOT default to general agent unless it's truly a general question

EXAMPLES:
- "berapa nilai bil di rokan technical standard" -> call_rts_agent
- "halo, apa kabar" -> call_general_agent
- "prosedur kerja di STK" -> call_stk_agent
"""

llm = ChatOllama(
    base_url=settings.OLLAMA_BASE_URL,
    model=settings.OLLAMA_MODEL,
    temperature=0.1,  # Lower temperature for more consistent routing
    system=system_prompt,
)

# System prompt untuk ReAct agent
# Note: create_react_agent sudah punya prompt default yang baik
# Kita hanya perlu pastikan tool descriptions jelas
agent_executor = create_react_agent(
    llm,
    [call_general_agent, call_stk_agent, call_rts_agent],
)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/orchestrate")
async def orchestrate(request: OrchestrationRequest) -> dict:
    print(f"DEBUG: Orchestrator received question: '{request.question}'")
    
    try:
        # Tambahkan recursion_limit dan konfigurasi lainnya
        result = await agent_executor.ainvoke(
            {"messages": [("user", request.question)]},
            config={
                "recursion_limit": 10,  # Batasi maksimal iterasi
                "max_iterations": 5,     # Maksimal 5 iterasi tool calls
            }
        )
        
        # Debug: Log all messages to see tool selection
        messages = result.get("messages", [])
        print(f"DEBUG: Agent executed {len(messages)} messages")
        for i, msg in enumerate(messages):
            if hasattr(msg, 'content'):
                print(f"DEBUG: Message {i}: {msg.content[:200]}...")
            elif isinstance(msg, tuple):
                print(f"DEBUG: Message {i}: {msg[0]} - {msg[1][:200]}...")
            else:
                print(f"DEBUG: Message {i}: {str(msg)[:200]}...")
                
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

    # Try to parse as JSON first
    try:
        payload = orjson.loads(content)
        print(f"DEBUG: Parsed JSON response: {payload.get('domain', 'UNKNOWN')}")
        return payload
    except orjson.JSONDecodeError:
        # If not JSON, check if it looks like it should be RTS based on question content
        question_lower = request.question.lower()
        if any(keyword in question_lower for keyword in ["rts", "rokan technical standard", "nilai bil", "standar teknis"]):
            print(f"DEBUG: Non-JSON response for RTS question, trying direct RTS call")
            try:
                # Try direct RTS call as fallback
                rts_result = await _call_agent(settings.AGENT_RTS_URL, "RTS", request.question)
                return rts_result
            except Exception as e:
                print(f"DEBUG: Direct RTS call failed: {e}")
        
        # If not JSON, treat as plain text response and wrap it in expected format
        print(f"DEBUG: Non-JSON response detected, wrapping as general response")
        return {
            "domain": "GENERAL",
            "answer": content,
            "citations": [],
            "diagnostic": {
                "mode": "orchestrator_fallback",
                "original_content": content[:200] + "..." if len(content) > 200 else content
            }
        }
