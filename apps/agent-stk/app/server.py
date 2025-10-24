from __future__ import annotations

import orjson
from fastapi import FastAPI, HTTPException
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel
from typing import TypedDict, List, Any

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
    # Direct vector search to Milvus
    try:
        import httpx
        
        # Prepare search payload
        payload = {
            "q": question,
            "top_k": settings.TOP_K,
            "filters": {
                "category": settings.CATEGORY_FILTER,
                "access_rights": "internal",
            }
        }
        
        # Search Milvus
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.MILVUS_RAG_URL}/search",
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
        
        passages = data.get("passages") or data.get("results") or []
        
        # Process results
        if not passages:
            return orjson.dumps({
                "domain": "STK",
                "answer": settings.REFUSAL_TEXT,
                "citations": [],
                "diagnostic": {
                    "mode": "vector_search",
                    "hits": 0,
                    "filters": payload["filters"]
                }
            }).decode()
        
        # Generate answer using LLM
        context_lines = []
        citations = []
        seen_citations = set()
        
        for passage in passages[:settings.MAX_CONTEXT]:
            # Extract citation
            doc_name = passage.get("document_name") or passage.get("document_id") or "Unknown"
            page = passage.get("number_page") or passage.get("page") or passage.get("page_number")
            page_str = str(page) if page is not None else "?"
            citation = f"{doc_name} p.{page_str}"
            
            if citation not in seen_citations:
                seen_citations.add(citation)
                citations.append(citation)
            
            # Add to context
            text = passage.get("text") or ""
            context_lines.append(f"{doc_name} p.{page_str}: {text}")
        
        context = "\n\n".join(context_lines)
        
        # Generate answer using LLM
        system_prompt = (
            "Anda adalah asisten teknis yang ahli dalam dokumen STK (Sistem Tata Kerja). "
            "Jawab pertanyaan berdasarkan konteks yang diberikan. "
            "Gunakan bahasa Indonesia yang ringkas dan berbutir. "
            "Sertakan sitasi pada setiap pernyataan faktual. "
            f"Jika bukti lemah, balas: {settings.REFUSAL_TEXT}"
        )
        
        user_prompt = (
            f"Pertanyaan: {question}\n"
            f"Konteks (maks {settings.MAX_CONTEXT} potongan):\n{context}\n"
            f"Gaya: {settings.STYLE}\n"
            f"Instruksi: Jawab ringkas, bahasa Indonesia. Sertakan sitasi pada setiap pernyataan faktual. "
            f"Jika bukti lemah, balas {settings.REFUSAL_TEXT}."
        )
        
        llm_payload = {
            "model": settings.OLLAMA_MODEL,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        
        async with httpx.AsyncClient() as client:
            llm_response = await client.post(
                f"{settings.OLLAMA_BASE_URL}/v1/chat/completions",
                json=llm_payload,
                timeout=60,
            )
            llm_response.raise_for_status()
            llm_data = llm_response.json()
        
        answer = llm_data["choices"][0]["message"]["content"]
        
        return orjson.dumps({
            "domain": "STK",
            "answer": answer,
            "citations": citations,
            "diagnostic": {
                "mode": "vector_search",
                "hits": len(passages),
                "used_passages": len(context_lines),
                "filters": payload["filters"]
            }
        }).decode()
        
    except Exception as e:
        import traceback
        error_detail = f"Vector search error: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        return orjson.dumps({
            "domain": "STK",
            "answer": f"Error: {str(e)}",
            "citations": [],
            "diagnostic": {"error": str(e)}
        }).decode()


@tool
async def answer_stk_pedoman(question: str) -> str:
    """MANDATORY tool to search PEDOMAN/MANUAL collection for general policy or guideline questions.
    Returns JSON: {"domain":"STK", "answer":"...", "citations":[...], "diagnostic":{...}}
    """
    return await _search_stk_collection(question, "pedoman")


@tool
async def answer_stk_tko(question: str) -> str:
    """MANDATORY tool to search TKO (Tata Kerja Organisasi) collection for organizational work procedures.
    Returns JSON: {"domain":"STK", "answer":"...", "citations":[...], "diagnostic":{...}}
    """
    return await _search_stk_collection(question, "TKO")


@tool
async def answer_stk_tki(question: str) -> str:
    """MANDATORY tool to search TKI (Tata Kerja Individu) collection for individual work procedures.
    Returns JSON: {"domain":"STK", "answer":"...", "citations":[...], "diagnostic":{...}}
    """
    return await _search_stk_collection(question, "TKI")


@tool
async def answer_stk_tkpa(question: str) -> str:
    """MANDATORY tool to search TKPA (Tata Kerja Penggunaan Alat) collection for equipment usage instructions.
    Returns JSON: {"domain":"STK", "answer":"...", "citations":[...], "diagnostic":{...}}
    """
    return await _search_stk_collection(question, "TKPA")


# Helper function for STK collection search
async def _search_stk_collection(question: str, collection: str) -> str:
    """Search specific STK collection"""
    try:
        import httpx
        
        # Prepare search payload with collection filter
        payload = {
            "q": question,
            "top_k": settings.TOP_K,
            "filters": {
                "category": settings.CATEGORY_FILTER,
                "access_rights": "internal",
                "doc_level_2": collection,  # Filter by collection
            }
        }
        
        # Search Milvus
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.MILVUS_RAG_URL}/search",
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
        
        passages = data.get("passages") or data.get("results") or []
        
        # Process results
        if not passages:
            return orjson.dumps({
                "domain": "STK",
                "answer": settings.REFUSAL_TEXT,
                "citations": [],
                "diagnostic": {
                    "mode": "vector_search",
                    "collection": collection,
                    "hits": 0,
                    "filters": payload["filters"]
                }
            }).decode()
        
        # Generate answer using LLM
        context_lines = []
        citations = []
        seen_citations = set()
        
        for passage in passages[:settings.MAX_CONTEXT]:
            # Extract citation
            doc_name = passage.get("document_name") or passage.get("document_id") or "Unknown"
            page = passage.get("number_page") or passage.get("page") or passage.get("page_number")
            page_str = str(page) if page is not None else "?"
            citation = f"{doc_name} p.{page_str}"
            
            if citation not in seen_citations:
                seen_citations.add(citation)
                citations.append(citation)
            
            # Add to context
            text = passage.get("text") or ""
            context_lines.append(f"{doc_name} p.{page_str}: {text}")
        
        context = "\n\n".join(context_lines)
        
        # Generate answer using LLM
        system_prompt = (
            f"Anda adalah asisten teknis yang ahli dalam dokumen STK {collection}. "
            "Jawab pertanyaan berdasarkan konteks yang diberikan. "
            "Gunakan bahasa Indonesia yang ringkas dan berbutir. "
            "Sertakan sitasi pada setiap pernyataan faktual. "
            f"Jika bukti lemah, balas: {settings.REFUSAL_TEXT}"
        )
        
        user_prompt = (
            f"Pertanyaan: {question}\n"
            f"Konteks (maks {settings.MAX_CONTEXT} potongan):\n{context}\n"
            f"Gaya: {settings.STYLE}\n"
            f"Instruksi: Jawab ringkas, bahasa Indonesia. Sertakan sitasi pada setiap pernyataan faktual. "
            f"Jika bukti lemah, balas {settings.REFUSAL_TEXT}."
        )
        
        llm_payload = {
            "model": settings.OLLAMA_MODEL,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        
        async with httpx.AsyncClient() as client:
            llm_response = await client.post(
                f"{settings.OLLAMA_BASE_URL}/v1/chat/completions",
                json=llm_payload,
                timeout=60,
            )
            llm_response.raise_for_status()
            llm_data = llm_response.json()
        
        answer = llm_data["choices"][0]["message"]["content"]
        
        return orjson.dumps({
            "domain": "STK",
            "answer": answer,
            "citations": citations,
            "diagnostic": {
                "mode": "vector_search",
                "collection": collection,
                "hits": len(passages),
                "used_passages": len(context_lines),
                "filters": payload["filters"]
            }
        }).decode()
        
    except Exception as e:
        import traceback
        error_detail = f"Vector search error for {collection}: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        return orjson.dumps({
            "domain": "STK",
            "answer": f"Error: {str(e)}",
            "citations": [],
            "diagnostic": {"error": str(e), "collection": collection}
        }).decode()


# Custom agent state
class AgentState(TypedDict):
    messages: List[Any]
    question: str

# Custom agent executor that forces tool usage
async def custom_agent_executor(state: AgentState) -> dict:
    """Custom agent that ALWAYS uses the auto tool"""
    question = state.get("question", "")
    
    # Always call the auto tool directly
    try:
        tool_result = await answer_stk_auto.ainvoke({"question": question})
        return {"messages": [AIMessage(content=tool_result)]}
    except Exception as e:
        error_response = orjson.dumps({
            "domain": "STK",
            "answer": f"Error: {str(e)}",
            "citations": [],
            "diagnostic": {"error": str(e)}
        }).decode()
        return {"messages": [AIMessage(content=error_response)]}

# Create custom graph
def create_custom_agent():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", custom_agent_executor)
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", END)
    return workflow.compile()

agent_executor = create_custom_agent()

app = FastAPI()


class ActRequest(BaseModel):
    question: str


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "domain": settings.DOMAIN}


@app.post("/act")
async def act(payload: ActRequest) -> dict:
    try:
        # Use custom agent that forces tool usage
        result = await agent_executor.ainvoke({
            "question": payload.question,
            "messages": []
        })
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
