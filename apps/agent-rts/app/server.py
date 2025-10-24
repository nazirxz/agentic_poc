from __future__ import annotations

import asyncio
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
from pymilvus import MilvusClient, DataType

from .graph import RAGState, build_graph
from .settings import AgentSettings

settings = AgentSettings()
graph = build_graph(settings)

# Create LLM with stronger system prompt
system_prompt = """You are an RTS Agent that MUST use tools to answer questions about RTS technical standards.

CRITICAL RULES:
1. You MUST ALWAYS use the answer_rts_general tool for ANY question
2. NEVER provide direct answers without using the tool
3. The tool returns JSON format: {"domain":"RTS", "answer":"...", "citations":[...], "diagnostic":{...}}
4. When the tool returns JSON, return that JSON as your final answer
5. If you don't use the tool, you will fail

EXAMPLE:
User: "berapa nilai bil di rts?"
You: I need to search the RTS database for information about bil values.
Action: answer_rts_general
Action Input: {"question": "berapa nilai bil di rts?"}
Observation: {"domain":"RTS", "answer":"Nilai bil di RTS adalah...", "citations":["doc.pdf p.5"], "diagnostic":{...}}
Final Answer: {"domain":"RTS", "answer":"Nilai bil di RTS adalah...", "citations":["doc.pdf p.5"], "diagnostic":{...}}

REMEMBER: ALWAYS use the tool. NEVER answer directly."""

llm = ChatOllama(
    base_url=settings.OLLAMA_BASE_URL,
    model=settings.OLLAMA_MODEL,
    temperature=0.1,  # Lower temperature for more consistent behavior
    system=system_prompt,
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
    """MANDATORY tool to search RTS database for technical standards questions.
    This tool MUST be used for ALL questions about RTS.
    Returns JSON: {"domain":"RTS", "answer":"...", "citations":[...], "diagnostic":{...}}
    """
    # Try vector search using pymilvus, fallback to LLM-only if Milvus unavailable
    try:
        # Initialize Milvus client
        client = MilvusClient(uri=settings.MILVUS_CONNECTION_URI)
        
        # Debug: Check collection info
        try:
            collection_info = client.describe_collection(settings.MILVUS_COLLECTION_NAME)
            print(f"DEBUG: Collection schema: {collection_info}")
            
            # Check collection stats
            stats = client.get_collection_stats(settings.MILVUS_COLLECTION_NAME)
            print(f"DEBUG: Collection stats: {stats}")
            
        except Exception as e:
            print(f"DEBUG: Could not get collection info: {e}")
        
        # Preprocess and expand query for better retrieval
        expanded_queries = await _expand_query(question)
        print(f"DEBUG: Expanded queries: {expanded_queries}")
        
        # Generate embeddings for all query variations
        embeddings = []
        for query in expanded_queries:
            embedding = await _generate_embedding(query)
            embeddings.append(embedding)
        
        # Use the first (original) embedding for primary search
        embedding = embeddings[0]
        
        # Search Milvus collection
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        
        # Available categories for reference
        available_categories = ['ASSET', 'CIVIL', 'CONSTRUCTION', 'ELECTRICAL', 'FIRE', 'GOVERNANCE', 
                               'INSTRUMENT', 'MATERIAL', 'PIPING', 'PROCESS', 'QA', 'ROTATING', 
                               'STATIC', 'rokan_technical_standard']
        
        # Debug: Print search parameters
        print(f"DEBUG: Searching collection: {settings.MILVUS_COLLECTION_NAME}")
        print(f"DEBUG: Embedding dimension: {len(embedding)}")
        print(f"DEBUG: Search limit: {settings.TOP_K}")
        print(f"DEBUG: Available categories: {available_categories}")
        
        # Hybrid search: Combine vector search with keyword search
        all_passages = []
        
        # 1. Vector search
        vector_results = client.search(
            collection_name=settings.MILVUS_COLLECTION_NAME,
            data=[embedding],
            anns_field="vector",
            search_params=search_params,
            limit=settings.TOP_K,
            output_fields=["id", "text", "document_id", "document_name", "number_page", "category", "access_rights", "keyword", "summary"]
        )
        
        print(f"DEBUG: Vector search returned {len(vector_results[0]) if vector_results else 0} hits")
        
        # 2. Dynamic keyword search based on extracted keywords
        keyword_results = []
        extracted_keywords = _extract_keywords_from_question(question)
        print(f"DEBUG: Extracted keywords from question: {extracted_keywords}")
        
        if extracted_keywords:
            try:
                # Build dynamic filter for keyword search
                keyword_filters = []
                for keyword in extracted_keywords:
                    keyword_filters.append(f'text like "%{keyword}%"')
                
                if keyword_filters:
                    # Use OR logic for multiple keywords
                    filter_expression = " or ".join(keyword_filters)
                    
                    keyword_results = client.query(
                        collection_name=settings.MILVUS_COLLECTION_NAME,
                        filter=filter_expression,
                        output_fields=["id", "text", "document_id", "document_name", "number_page", "category", "access_rights", "keyword", "summary"],
                        limit=10
                    )
                    print(f"DEBUG: Dynamic keyword search returned {len(keyword_results)} hits")
                    print(f"DEBUG: Keyword results type: {type(keyword_results)}")
                
                # Additional search for technical specifications if relevant keywords found
                technical_keywords = [kw for kw in extracted_keywords if kw in ["specification", "standard", "requirement", "technical"]]
                if technical_keywords:
                    try:
                        tech_results = client.query(
                            collection_name=settings.MILVUS_COLLECTION_NAME,
                            filter='text like "%specification%" or text like "%standard%"',
                            output_fields=["id", "text", "document_id", "document_name", "number_page", "category", "access_rights", "keyword", "summary"],
                            limit=5
                        )
                        print(f"DEBUG: Technical specification search returned {len(tech_results)} hits")
                        
                        # Safely extend keyword_results
                        if isinstance(keyword_results, list):
                            keyword_results.extend(tech_results)
                        else:
                            # Convert to list first
                            keyword_results = list(keyword_results) + list(tech_results)
                    except Exception as e:
                        print(f"DEBUG: Technical specification search failed: {e}")
                
            except Exception as e:
                print(f"DEBUG: Dynamic keyword search failed: {e}")
        
        # Combine results
        results = vector_results
        
        # If we have results, check if we need category filtering
        if results and results[0]:
            # Analyze categories in results
            result_categories = [hit.get('category') for hit in results[0] if hit.get('category')]
            unique_categories = list(set(result_categories))
            print(f"DEBUG: Categories found in results: {unique_categories}")
            
            # Check if we should apply category filtering based on settings
            # Category filtering is optional and only used to reduce scope if needed
            if settings.CATEGORY_FILTER and settings.CATEGORY_FILTER != "rokan_technical_standard":
                if settings.CATEGORY_FILTER in available_categories:
                    print(f"DEBUG: Applying category filter: {settings.CATEGORY_FILTER}")
                    # Re-search with category filter
                    results = client.search(
                        collection_name=settings.MILVUS_COLLECTION_NAME,
                        data=[embedding],
                        anns_field="vector",
                        search_params=search_params,
                        limit=settings.TOP_K,
                        output_fields=["id", "text", "document_id", "document_name", "number_page", "category", "access_rights"],
                        filter=f'category == "{settings.CATEGORY_FILTER}"'
                    )
                    print(f"DEBUG: Search with category filter returned {len(results[0]) if results else 0} hits")
                else:
                    print(f"DEBUG: Category filter '{settings.CATEGORY_FILTER}' not in available categories, using all results")
            else:
                print("DEBUG: Using all results without category filtering")
        
        # Process and combine results
        passages = []
        seen_ids = set()
        
        # Process vector search results
        if results and results[0]:
            for hit in results[0]:
                hit_id = hit.get("id")
                if hit_id not in seen_ids:
                    passages.append({
                        "id": hit_id,
                        "text": hit.get("text"),
                        "document_id": hit.get("document_id"),
                        "document_name": hit.get("document_name"),
                        "number_page": hit.get("number_page"),
                        "score": hit.get("distance", 0),
                        "source": "vector",
                        "keyword": hit.get("keyword", ""),
                        "summary": hit.get("summary", "")
                    })
                    seen_ids.add(hit_id)
        
        # Process keyword search results with error handling
        try:
            # Convert to list if it's a Milvus result object
            if hasattr(keyword_results, '__iter__') and not isinstance(keyword_results, list):
                keyword_results = list(keyword_results)
            
            for hit in keyword_results:
                hit_id = hit.get("id")
                if hit_id not in seen_ids:
                    passages.append({
                        "id": hit_id,
                        "text": hit.get("text"),
                        "document_id": hit.get("document_id"),
                        "document_name": hit.get("document_name"),
                        "number_page": hit.get("number_page"),
                        "score": 0.1,  # Lower score for keyword matches
                        "source": "keyword",
                        "keyword": hit.get("keyword", ""),
                        "summary": hit.get("summary", "")
                    })
                    seen_ids.add(hit_id)
        except Exception as e:
            print(f"DEBUG: Error processing keyword results: {e}")
            # Continue without keyword results
        
        print(f"DEBUG: Combined results: {len(passages)} passages ({len([p for p in passages if p['source'] == 'vector'])} vector, {len([p for p in passages if p['source'] == 'keyword'])} keyword)")
        
        # Rerank passages based on relevance
        passages = await _rerank_passages(passages, question)
        
        if not passages:
            return orjson.dumps({
                "domain": "RTS",
                "answer": settings.REFUSAL_TEXT,
                "citations": [],
                "diagnostic": {
                    "mode": "hybrid_search_with_reranking",
                    "vector_hits": 0,
                    "keyword_hits": 0,
                    "total_hits": 0,
                    "collection": settings.MILVUS_COLLECTION_NAME,
                    "search_strategy": "hybrid_no_filter",
                    "available_categories": available_categories,
                    "embedding_dimension": len(embedding),
                    "search_limit": settings.TOP_K,
                    "query_expansion": len(expanded_queries) > 1,
                    "reranking_applied": True
                }
            }).decode()
        
        # Generate answer using LLM with context
        context_lines = []
        citations = []
        seen_citations = set()
        
        # Use top passages after reranking, but limit context length to prevent timeout
        relevant_passages = passages[:min(settings.MAX_CONTEXT, settings.MAX_PASSAGES)]
        print(f"DEBUG: Using top {len(relevant_passages)} passages after reranking (limited to prevent timeout)")
        
        # Additional quality check - prefer technical content but don't filter everything
        quality_passages = []
        filtering_rules = _get_filtering_rules()
        
        for passage in relevant_passages:
            text = passage.get("text", "").lower()
            # Skip only if it's clearly administrative content with no technical value
            if (any(term in text for term in filtering_rules["administrative_terms"]) and 
                not any(term in text for term in filtering_rules["technical_value_terms"])):
                print(f"DEBUG: Skipping pure administrative passage: {passage['text'][:50]}...")
                continue
            # Skip only very short content
            if len(text) < settings.MIN_TEXT_LENGTH:
                print(f"DEBUG: Skipping very short passage: {passage['text'][:50]}...")
                continue
            quality_passages.append(passage)
        
        # Use up to MAX_PASSAGES, but prefer quality ones
        relevant_passages = quality_passages[:settings.MAX_PASSAGES] if quality_passages else relevant_passages[:settings.MAX_PASSAGES]
        print(f"DEBUG: Using {len(relevant_passages)} passages after quality filtering")
        
        for passage in relevant_passages:
            # Extract citation
            doc_name = passage.get("document_name") or passage.get("document_id") or "Unknown"
            page = passage.get("number_page")
            page_str = str(page) if page is not None else "?"
            citation = f"{doc_name} p.{page_str}"
            
            if citation not in seen_citations:
                seen_citations.add(citation)
                citations.append(citation)
            
            # Add to context with text truncation
            text = passage.get("text") or ""
            # Truncate text to prevent very long contexts
            if len(text) > settings.MAX_TEXT_LENGTH:
                text = text[:settings.MAX_TEXT_LENGTH] + "..."
            context_lines.append(f"{doc_name} p.{page_str}: {text}")
        
        context = "\n\n".join(context_lines)
        
        # Additional context length check
        print(f"DEBUG: Context length: {len(context)} characters")
        if len(context) > settings.MAX_CONTEXT_LENGTH:
            print(f"DEBUG: Context too long ({len(context)} chars), truncating to {settings.MAX_CONTEXT_LENGTH}")
            context = context[:settings.MAX_CONTEXT_LENGTH] + "..."
        
        # Generate answer using LLM
        system_prompt = (
            "Anda adalah asisten teknis yang ahli dalam standar RTS. "
            "Jawab pertanyaan secara SPESIFIK dan LANGSUNG berdasarkan konteks yang diberikan. "
            "JANGAN berikan ringkasan umum atau daftar dokumen. "
            "Fokus pada jawaban spesifik yang diminta. "
            "Gunakan bahasa Indonesia yang formal dan teknis. "
            "Sertakan sitasi pada setiap pernyataan faktual. "
            f"Jika tidak menemukan informasi spesifik yang diminta, balas: {settings.REFUSAL_TEXT}"
        )
        
        user_prompt = (
            f"PERTANYAAN SPESIFIK: {question}\n\n"
            f"KONTEKS YANG TERSEDIA:\n{context}\n\n"
            f"INSTRUKSI PENTING:\n"
            f"- Jawab PERTANYAAN SPESIFIK yang diminta, bukan ringkasan umum\n"
            f"- Jika ditanya tentang nilai/nilai spesifik, berikan nilai yang tepat\n"
            f"- Jika ditanya tentang prosedur, berikan langkah-langkah spesifik\n"
            f"- JANGAN buat daftar dokumen atau ringkasan\n"
            f"- Fokus pada jawaban yang diminta\n"
            f"- Gunakan bahasa Indonesia formal dan teknis\n"
            f"- Sertakan sitasi pada setiap pernyataan faktual\n"
            f"- Jika tidak ada informasi spesifik, balas: {settings.REFUSAL_TEXT}"
        )
        
        llm_payload = {
            "model": settings.OLLAMA_MODEL,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        
        # Try LLM call with timeout and retry
        for attempt in range(settings.MAX_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT) as client:
                    llm_response = await client.post(
                        f"{settings.OLLAMA_BASE_URL}/v1/chat/completions",
                        json=llm_payload,
                    )
                    llm_response.raise_for_status()
                    llm_data = llm_response.json()
                    break  # Success, exit retry loop
            except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                print(f"DEBUG: LLM timeout attempt {attempt + 1}/{settings.MAX_RETRIES}: {e}")
                if attempt == settings.MAX_RETRIES - 1:
                    # Last attempt failed, use fallback
                    print("DEBUG: All LLM attempts failed, using fallback")
                    return await _fallback_llm_response(question, "RTS")
                # Wait before retry
                await asyncio.sleep(1)
            except Exception as e:
                print(f"DEBUG: LLM error attempt {attempt + 1}/{settings.MAX_RETRIES}: {e}")
                if attempt == settings.MAX_RETRIES - 1:
                    return await _fallback_llm_response(question, "RTS")
                await asyncio.sleep(1)
        
        answer = llm_data["choices"][0]["message"]["content"]
        
        return orjson.dumps({
            "domain": "RTS",
            "answer": answer,
            "citations": citations,
            "diagnostic": {
                "mode": "hybrid_search_with_reranking",
                "vector_hits": len([p for p in passages if p.get("source") == "vector"]),
                "keyword_hits": len([p for p in passages if p.get("source") == "keyword"]),
                "total_hits": len(passages),
                "used_passages": len(context_lines),
                "collection": settings.MILVUS_COLLECTION_NAME,
                "query_expansion": len(expanded_queries) > 1,
                "reranking_applied": True
            }
        }).decode()
        
    except Exception as e:
        import traceback
        error_detail = f"Pymilvus search error: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        return await _fallback_llm_response(question, "RTS")


def _get_expansion_rules() -> dict:
    """Get configurable query expansion rules"""
    return {
        "nilai": [
            "value", "parameter", "specification", "standard",
            "requirement", "criteria", "measurement", "threshold"
        ],
        "rokan technical standard": [
            "RTS", "rokan standard", "technical standard", "engineering standard",
            "rokan specification", "rokan requirement"
        ],
        "berapa": [
            "what is", "what are", "how much", "how many", "what value", "what parameter"
        ],
        "technical": [
            "specification", "standard", "requirement", "criteria", "parameter"
        ]
    }

def _get_filtering_rules() -> dict:
    """Get configurable filtering rules"""
    return {
        "non_technical_terms": ["persetujuan", "pengkinian", "approval", "update", "<!-- image -->", "date", "subject"],
        "technical_terms": ["specification", "requirement", "standard", "voltage", "insulation", "electrical", "technical"],
        "rts_terms": ["rts", "rokan", "technical", "standard", "specification"],
        "administrative_terms": ["persetujuan", "pengkinian", "approval"],
        "technical_value_terms": ["specification", "standard", "technical", "voltage", "insulation", "requirement", "criteria"]
    }

def _extract_keywords_from_question(question: str) -> List[str]:
    """Extract relevant keywords from question for keyword search"""
    question_lower = question.lower()
    keywords = []
    
    # Extract technical terms (3+ characters, alphanumeric)
    import re
    words = re.findall(r'\b[a-zA-Z]{3,}\b', question_lower)
    
    # Filter out common words and keep technical terms
    common_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "berapa", "yang", "ada", "di", "dalam", "untuk", "dengan", "oleh"}
    technical_words = [word for word in words if word not in common_words and len(word) >= 3]
    
    # Add specific technical patterns
    technical_patterns = [
        r'\b[a-z]{2,}\d+\b',  # Alphanumeric codes like "bil", "rts"
        r'\b[a-z]+[-_][a-z]+\b',  # Hyphenated terms
        r'\b[a-z]{4,}\b'  # Longer technical terms
    ]
    
    for pattern in technical_patterns:
        matches = re.findall(pattern, question_lower)
        keywords.extend(matches)
    
    # Remove duplicates and limit
    keywords = list(set(keywords))[:5]  # Max 5 keywords
    return keywords

async def _expand_query(question: str) -> List[str]:
    """Expand query with synonyms and related terms for better retrieval"""
    expanded = [question]  # Always include original question
    expansion_rules = _get_expansion_rules()
    
    # Apply expansion rules
    question_lower = question.lower()
    for key, expansions in expansion_rules.items():
        if key in question_lower:
            for expansion in expansions:
                expanded_query = question_lower.replace(key, expansion)
                if expanded_query not in expanded:
                    expanded.append(expanded_query)
    
    # Limit to reasonable number of queries
    return expanded[:5]


async def _rerank_passages(passages: List[dict], question: str) -> List[dict]:
    """Rerank passages based on relevance to the question"""
    if not passages:
        return passages
    
    question_lower = question.lower()
    question_words = set(question_lower.split())
    
    def calculate_relevance_score(passage):
        score = passage.get("score", 0)
        text = passage.get("text", "").lower()
        keyword = passage.get("keyword", "").lower()
        summary = passage.get("summary", "").lower()
        
        # Get filtering rules
        filtering_rules = _get_filtering_rules()
        
        # Apply penalties for non-technical documents
        for term in filtering_rules["non_technical_terms"]:
            if term in text:
                score -= settings.NON_TECHNICAL_PENALTY
        
        # Boost score for keyword matches
        if passage.get("source") == "keyword":
            score += settings.KEYWORD_BOOST
        
        # Boost score for exact keyword matches
        extracted_keywords = _extract_keywords_from_question(question)
        for keyword in extracted_keywords:
            if keyword in text:
                score += settings.KEYWORD_BOOST
        
        # Boost score for technical terms
        for term in filtering_rules["technical_terms"]:
            if term in text:
                score += settings.TECHNICAL_TERM_BOOST
        
        # Boost score for RTS-related terms
        for term in filtering_rules["rts_terms"]:
            if term in text:
                score += settings.RTS_TERM_BOOST
        
        # Boost score for keyword field matches
        if keyword and any(word in keyword for word in question_words):
            score += settings.KEYWORD_BOOST * 0.6  # Slightly less than direct keyword match
        
        # Boost score for summary matches
        if summary and any(word in summary for word in question_words):
            score += settings.TECHNICAL_TERM_BOOST
        
        # Penalize very short passages
        if len(text) < settings.MIN_TEXT_LENGTH:
            score -= settings.SHORT_TEXT_PENALTY
        
        # Penalize passages that are mostly HTML/formatting
        if text.count("<") > 5 or text.count("&") > 3:
            score -= settings.HTML_PENALTY
        
        return score
    
    # Calculate relevance scores
    for passage in passages:
        passage["relevance_score"] = calculate_relevance_score(passage)
    
    # Sort by relevance score (higher is better)
    passages.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    # Filter out passages with very low relevance scores
    filtered_passages = [p for p in passages if p["relevance_score"] > settings.MIN_RELEVANCE_SCORE]
    
    print(f"DEBUG: Passages after filtering: {len(filtered_passages)}/{len(passages)} (min score: {settings.MIN_RELEVANCE_SCORE})")
    print(f"DEBUG: Top 3 passages after reranking:")
    for i, passage in enumerate(filtered_passages[:3]):
        print(f"  {i+1}. Score: {passage['relevance_score']:.3f}, Source: {passage['source']}, Text: {passage['text'][:100]}...")
    
    return filtered_passages


async def _generate_embedding(text: str) -> List[float]:
    """Generate embedding for text using Ollama BGE-M3 model"""
    try:
        from langchain_ollama import OllamaEmbeddings
        
        print(f"DEBUG: Generating embedding for text: '{text[:100]}...'")
        print(f"DEBUG: Using embedding model: {settings.OLLAMA_EMBEDDING_MODEL}")
        print(f"DEBUG: Ollama base URL: {settings.OLLAMA_BASE_URL}")
        
        # Initialize OllamaEmbeddings with BGE-M3 model
        embeddings = OllamaEmbeddings(
            model=settings.OLLAMA_EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL
        )
        
        # Generate embedding using LangChain
        embedding_vector = embeddings.embed_query(text)
        
        print(f"DEBUG: Generated embedding dimension: {len(embedding_vector)}")
        print(f"DEBUG: Embedding sample (first 5 values): {embedding_vector[:5]}")
        
        # Check if embedding is all zeros (indicates failure)
        if all(v == 0.0 for v in embedding_vector):
            print("WARNING: Generated embedding is all zeros!")
        
        return embedding_vector
        
    except Exception as e:
        print(f"Embedding generation error: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        # Return zero vector as fallback (BGE-M3 has 3584 dimensions)
        return [0.0] * 3584


async def _fallback_llm_response(question: str, domain: str) -> str:
    """Fallback LLM response when vector search is unavailable"""
    try:
        import httpx
        
        system_prompt = (
            f"Anda adalah asisten teknis yang ahli dalam dokumen {domain}. "
            "Jawab pertanyaan berdasarkan pengetahuan umum tentang {domain}. "
            "Gunakan bahasa Indonesia yang formal dan teknis. "
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
    """Custom agent that ALWAYS uses the tool"""
    question = state.get("question", "")
    
    # Always call the tool directly
    try:
        tool_result = await answer_rts_general.ainvoke({"question": question})
        return {"messages": [AIMessage(content=tool_result)]}
    except Exception as e:
        error_response = orjson.dumps({
            "domain": "RTS",
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
