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

# Create LLM with stronger system prompt
system_prompt = """You are an STK Agent that MUST use tools to answer questions about STK documents.

CRITICAL RULES:
1. You MUST ALWAYS use one of the available tools for ANY question
2. NEVER provide direct answers without using a tool
3. Tools return JSON format: {"domain":"STK", "answer":"...", "citations":[...], "diagnostic":{...}}
4. When a tool returns JSON, return that JSON as your final answer
5. If you don't use a tool, you will fail

AVAILABLE TOOLS:
- answer_stk_auto: For general questions or when unsure about document category
- answer_stk_pedoman: For general policies or guidelines
- answer_stk_tko: For organizational work procedures
- answer_stk_tki: For individual work procedures
- answer_stk_tkpa: For equipment usage instructions

EXAMPLE:
User: "bagaimana prosedur kerja organisasi?"
You: I need to search STK documents for organizational work procedures.
Action: answer_stk_tko
Action Input: {"question": "bagaimana prosedur kerja organisasi?"}
Observation: {"domain":"STK", "answer":"Prosedur kerja organisasi adalah...", "citations":["doc.pdf p.5"], "diagnostic":{...}}
Final Answer: {"domain":"STK", "answer":"Prosedur kerja organisasi adalah...", "citations":["doc.pdf p.5"], "diagnostic":{...}}

REMEMBER: ALWAYS use a tool. NEVER answer directly."""

llm = ChatOllama(
    base_url=settings.OLLAMA_BASE_URL,
    model=settings.OLLAMA_MODEL,
    temperature=0.1,  # Lower temperature for more consistent behavior
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
    """MANDATORY tool for automatic collection selection - system will choose the most suitable collection (pedoman/TKO/TKI/TKPA).
    Use this if unsure about document category.
    Returns JSON: {"domain":"STK", "answer":"...", "citations":[...], "diagnostic":{...}}
    """
    return orjson.dumps(await run_rag(question)).decode()


@tool
async def answer_stk_pedoman(question: str) -> str:
    """MANDATORY tool to search PEDOMAN/MANUAL collection for general policy or guideline questions.
    Returns JSON: {"domain":"STK", "answer":"...", "citations":[...], "diagnostic":{...}}
    """
    return orjson.dumps(await run_rag(question, collection="pedoman")).decode()


@tool
async def answer_stk_tko(question: str) -> str:
    """MANDATORY tool to search TKO (Tata Kerja Organisasi) collection for organizational work procedures.
    Returns JSON: {"domain":"STK", "answer":"...", "citations":[...], "diagnostic":{...}}
    """
    return orjson.dumps(await run_rag(question, collection="TKO")).decode()


@tool
async def answer_stk_tki(question: str) -> str:
    """MANDATORY tool to search TKI (Tata Kerja Individu) collection for individual work procedures.
    Returns JSON: {"domain":"STK", "answer":"...", "citations":[...], "diagnostic":{...}}
    """
    return orjson.dumps(await run_rag(question, collection="TKI")).decode()


@tool
async def answer_stk_tkpa(question: str) -> str:
    """MANDATORY tool to search TKPA (Tata Kerja Penggunaan Alat) collection for equipment usage instructions.
    Returns JSON: {"domain":"STK", "answer":"...", "citations":[...], "diagnostic":{...}}
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
