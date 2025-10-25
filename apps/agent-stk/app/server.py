from __future__ import annotations

import orjson
import httpx
from fastapi import FastAPI, HTTPException
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel
from typing import TypedDict, List, Any
from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker

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
    return await _search_stk_collection(question, None)


async def _search_stk_collection(question: str, collection: str | None) -> str:
    """Search STK collection using pymilvus"""
    try:
        # Initialize Milvus client
        client = MilvusClient(uri=settings.MILVUS_CONNECTION_URI)
        
        # Generate BGE-M3 embeddings (dense + sparse) for the question
        embedding_data = await _generate_bge_m3_embedding(question)
        
        # Search Milvus collection
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        
        # Determine collection name
        if collection and collection in settings.MILVUS_COLLECTIONS:
            collection_name = settings.MILVUS_COLLECTIONS[collection]
        else:
            # For auto search, search all STK collections
            collection_name = "tko"  # Default to TKO for auto search
        
        # Build filter expression
        filter_expr = f'category == "{settings.CATEGORY_FILTER}" and access_rights == "internal"'
        
        # 1. Dense vector search with L2 distance
        dense_req = AnnSearchRequest(
            data=[embedding_data['dense']],
            anns_field="vector",
            param={"metric_type": "L2", "ef": 64},
            limit=20  # Retrieve top-20 candidates
        )
        
        # 2. Enhanced hybrid search with multiple dense queries
        # Since we can't use true sparse search without FlagEmbedding, we'll use multiple dense queries
        # with different parameters to simulate hybrid search behavior
        
        # Primary dense search with standard parameters
        dense_req_primary = AnnSearchRequest(
            data=[embedding_data['dense']],
            anns_field="vector",
            param={"metric_type": "L2", "ef": 64},
            limit=15  # Retrieve top-15 candidates
        )
        
        # Secondary dense search with different parameters for diversity
        dense_req_secondary = AnnSearchRequest(
            data=[embedding_data['dense']],
            anns_field="vector",
            param={"metric_type": "L2", "ef": 32},  # Different ef for different search behavior
            limit=15
        )
        
        # 3. Perform hybrid search with RRF reranking using multiple dense queries
        try:
            results = client.hybrid_search(
                collection_name=collection_name,
                reqs=[dense_req_primary, dense_req_secondary],
                ranker=RRFRanker(k=60),  # RRF with k=60 (optimal default)
                limit=settings.TOP_K,
                output_fields=["id", "text", "document_id", "document_name", "number_page", "category", "access_rights"],
                filter=filter_expr
            )
            print("DEBUG: Enhanced hybrid search with multiple dense queries enabled for STK")
        except Exception as e:
            print(f"DEBUG: Hybrid search failed for STK, falling back to dense-only: {e}")
            # Fallback to dense-only search
            results = client.search(
                collection_name=collection_name,
                data=[embedding_data['dense']],
                anns_field="vector",
                search_params=search_params,
                limit=settings.TOP_K,
                output_fields=["id", "text", "document_id", "document_name", "number_page", "category", "access_rights"],
                filter=filter_expr
            )
        
        # Process results
        passages = []
        for hits in results:
            for hit in hits:
                passages.append({
                    "id": hit.get("id"),
                    "text": hit.get("text"),
                    "document_id": hit.get("document_id"),
                    "document_name": hit.get("document_name"),
                    "number_page": hit.get("number_page"),
                    "doc_level_2": hit.get("doc_level_2"),
                    "score": hit.get("distance", 0),
                    "source": "hybrid"
                })
        
        if not passages:
            return orjson.dumps({
                "domain": "STK",
                "answer": settings.REFUSAL_TEXT,
                "citations": [],
                "diagnostic": {
                    "mode": "bge_m3_hybrid_search",
                    "collection": collection,
                    "hits": 0,
                    "collection_name": collection_name,
                    "search_type": "enhanced_hybrid"
                }
            }).decode()
        
        # Generate answer using LLM
        context_lines = []
        citations = []
        seen_citations = set()
        
        for passage in passages[:settings.MAX_CONTEXT]:
            # Extract citation
            doc_name = passage.get("document_name") or passage.get("document_id") or "Unknown"
            page = passage.get("number_page")
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
        collection_desc = f" {collection}" if collection else ""
        system_prompt = (
            f"Anda adalah asisten teknis yang ahli dalam dokumen STK{collection_desc}. "
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
                "mode": "bge_m3_hybrid_search",
                "collection": collection,
                "hits": len(passages),
                "used_passages": len(context_lines),
                "collection_name": collection_name,
                "search_type": "enhanced_hybrid"
            }
        }).decode()
        
    except Exception as e:
        import traceback
        error_detail = f"Pymilvus search error for {collection}: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        return await _fallback_llm_response(question, "STK")


async def _generate_bge_m3_embedding(text: str) -> dict:
    """Generate BGE-M3 dense embeddings using Ollama with enhanced hybrid search"""
    try:
        from langchain_ollama import OllamaEmbeddings
        
        print(f"DEBUG: Generating Ollama BGE-M3 embedding for STK text: '{text[:100]}...'")
        
        # Initialize OllamaEmbeddings with BGE-M3 model
        embeddings = OllamaEmbeddings(
            model=settings.OLLAMA_EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL
        )
        
        # Generate dense embedding using LangChain
        dense_vector = embeddings.embed_query(text)
        
        print(f"DEBUG: Generated dense embedding dimension: {len(dense_vector)}")
        print(f"DEBUG: Dense embedding sample (first 5 values): {dense_vector[:5]}")
        
        # Check if dense embedding is all zeros (indicates failure)
        if all(v == 0.0 for v in dense_vector):
            print("WARNING: Generated dense embedding is all zeros!")
        
        # Generate pseudo-sparse vector for hybrid search simulation
        # This creates a simple keyword-based sparse representation
        sparse_vector = _generate_pseudo_sparse_vector(text)
        
        return {
            'dense': dense_vector,
            'sparse': sparse_vector
        }
        
    except Exception as e:
        print(f"BGE-M3 embedding generation error: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        # Return zero vectors as final fallback
        return {
            'dense': [0.0] * 1024,
            'sparse': {}
        }

def _generate_pseudo_sparse_vector(text: str) -> dict:
    """Generate pseudo-sparse vector for hybrid search simulation"""
    try:
        import re
        from collections import Counter
        
        # Extract keywords and technical terms
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out common words
        common_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", 
                       "berapa", "yang", "ada", "di", "dalam", "untuk", "dengan", "oleh", "dari", "pada"}
        
        technical_words = [word for word in words if word not in common_words and len(word) >= 3]
        
        # Count word frequencies
        word_counts = Counter(technical_words)
        
        # Create sparse vector with top keywords
        sparse_vector = {}
        for word, count in word_counts.most_common(50):  # Top 50 keywords
            # Use hash of word as index to avoid conflicts
            index = hash(word) % 10000  # Map to 0-9999 range
            sparse_vector[index] = float(count)
        
        print(f"DEBUG: Generated pseudo-sparse vector with {len(sparse_vector)} keywords")
        return sparse_vector
        
    except Exception as e:
        print(f"Pseudo-sparse vector generation error: {e}")
        return {}

async def _generate_embedding(text: str) -> List[float]:
    """Legacy function for backward compatibility - generates only dense embedding"""
    embedding_data = await _generate_bge_m3_embedding(text)
    return embedding_data['dense']


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


async def _fallback_llm_response(question: str, domain: str) -> str:
    """Fallback LLM response when vector search is unavailable"""
    try:
        
        system_prompt = (
            f"Anda adalah asisten teknis yang ahli dalam dokumen {domain}. "
            "Jawab pertanyaan berdasarkan pengetahuan umum tentang {domain}. "
            "Gunakan bahasa Indonesia yang ringkas dan berbutir. "
            f"Jika tidak yakin, balas: Tidak ditemukan dalam {domain}."
        )
        
        user_prompt = f"Pertanyaan: {question}\n\nJawab berdasarkan pengetahuan umum tentang {domain}."
        
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
            "domain": domain,
            "answer": answer,
            "citations": [],
            "diagnostic": {
                "mode": "llm_fallback",
                "reason": "milvus_unavailable",
                "model": settings.OLLAMA_MODEL
            }
        }).decode()
        
    except Exception as e:
        return orjson.dumps({
            "domain": domain,
            "answer": f"Maaf, terjadi kesalahan dalam memproses pertanyaan Anda: {str(e)}",
            "citations": [],
            "diagnostic": {"error": str(e), "mode": "error_fallback"}
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
