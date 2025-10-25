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
from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker

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
        
        # Generate BGE-M3 embeddings (dense + sparse) for all query variations
        embeddings_data = []
        for query in expanded_queries:
            embedding_data = await _generate_bge_m3_embedding(query)
            embeddings_data.append(embedding_data)
        
        # Use the first (original) embedding for primary search
        primary_embedding_data = embeddings_data[0]
        
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
        print(f"DEBUG: Dense embedding dimension: {len(primary_embedding_data['dense'])}")
        print(f"DEBUG: Sparse embedding keys: {len(primary_embedding_data['sparse'])}")
        print(f"DEBUG: Search limit: {settings.TOP_K}")
        print(f"DEBUG: Available categories: {available_categories}")
        
        # Hybrid search: BGE-M3 dense + sparse with RRF reranking
        all_passages = []
        
        # 1. Dense vector search with L2 distance
        dense_req = AnnSearchRequest(
            data=[primary_embedding_data['dense']],
            anns_field="vector",
            param={"metric_type": "L2", "ef": 64},
            limit=20  # Retrieve top-20 candidates
        )
        
        # 2. Hybrid search with BGE-M3 dense + sparse embeddings
        # Following Milvus best practices: https://milvus.io/docs/contextual_retrieval_with_milvus.md
        
        # Check if we have sparse embeddings available
        has_sparse = len(primary_embedding_data.get('sparse', {})) > 0
        
        if has_sparse and settings.USE_BGE_M3_HYBRID:
            # True hybrid search with dense and sparse vectors
            print("DEBUG: Using BGE-M3 hybrid search (dense + sparse)")
            
            # Dense search with HNSW index
            dense_req = AnnSearchRequest(
                data=[primary_embedding_data['dense']],
                anns_field="vector",
                param={"metric_type": "L2", "ef": 64},
                limit=settings.RERANKER_TOP_K if settings.USE_RERANKER else settings.TOP_K
            )
            
            # Sparse search with inverted index
            # Check if collection has sparse_vector field
            try:
                collection_info = client.describe_collection(settings.MILVUS_COLLECTION_NAME)
                has_sparse_field = any(field.get('name') == 'sparse_vector' for field in collection_info.get('fields', []))
                
                if has_sparse_field:
                    sparse_req = AnnSearchRequest(
                        data=[primary_embedding_data['sparse']],
                        anns_field="sparse_vector",
                        param={"metric_type": "IP"},  # Inner Product for sparse
                        limit=settings.RERANKER_TOP_K if settings.USE_RERANKER else settings.TOP_K
                    )
                    
                    # Perform hybrid search with RRF ranker
                    results = client.hybrid_search(
            collection_name=settings.MILVUS_COLLECTION_NAME,
                        reqs=[dense_req, sparse_req],
                        ranker=RRFRanker(k=60),  # RRF with k=60 (optimal)
                        limit=settings.RERANKER_TOP_K if settings.USE_RERANKER else settings.TOP_K,
                        output_fields=["id", "text", "document_id", "document_name", "number_page", "category", "access_rights", "keyword", "summary"]
                    )
                    print(f"DEBUG: Hybrid search (dense + sparse) completed, retrieved {len(results[0]) if results else 0} candidates")
                else:
                    print("DEBUG: Sparse field not found, using dense-only search")
                    results = client.search(
                        collection_name=settings.MILVUS_COLLECTION_NAME,
                        data=[primary_embedding_data['dense']],
            anns_field="vector",
            search_params=search_params,
                        limit=settings.RERANKER_TOP_K if settings.USE_RERANKER else settings.TOP_K,
                        output_fields=["id", "text", "document_id", "document_name", "number_page", "category", "access_rights", "keyword", "summary"]
                    )
            except Exception as e:
                print(f"DEBUG: Error in sparse search: {e}, falling back to dense-only")
                results = client.search(
                    collection_name=settings.MILVUS_COLLECTION_NAME,
                    data=[primary_embedding_data['dense']],
                    anns_field="vector",
                    search_params=search_params,
                    limit=settings.RERANKER_TOP_K if settings.USE_RERANKER else settings.TOP_K,
                    output_fields=["id", "text", "document_id", "document_name", "number_page", "category", "access_rights", "keyword", "summary"]
                )
        else:
            # Fallback to dense-only search
            print("DEBUG: Using dense-only search")
            results = client.search(
                collection_name=settings.MILVUS_COLLECTION_NAME,
                data=[primary_embedding_data['dense']],
                anns_field="vector",
                search_params=search_params,
                limit=settings.RERANKER_TOP_K if settings.USE_RERANKER else settings.TOP_K,
                output_fields=["id", "text", "document_id", "document_name", "number_page", "category", "access_rights", "keyword", "summary"]
            )
        
        print(f"DEBUG: Hybrid search returned {len(results[0]) if results else 0} hits")
        
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
                    # Check if we have sparse embeddings for hybrid search
                    has_sparse_for_filter = has_sparse and settings.USE_BGE_M3_HYBRID
                    
                    if has_sparse_for_filter:
                        try:
                            # Hybrid search with category filter
                            dense_req_filter = AnnSearchRequest(
                                data=[primary_embedding_data['dense']],
                                anns_field="vector",
                                param={"metric_type": "L2", "ef": 64},
                                limit=settings.RERANKER_TOP_K if settings.USE_RERANKER else settings.TOP_K
                            )
                            sparse_req_filter = AnnSearchRequest(
                                data=[primary_embedding_data['sparse']],
                                anns_field="sparse_vector",
                                param={"metric_type": "IP"},
                                limit=settings.RERANKER_TOP_K if settings.USE_RERANKER else settings.TOP_K
                            )
                            results = client.hybrid_search(
                                collection_name=settings.MILVUS_COLLECTION_NAME,
                                reqs=[dense_req_filter, sparse_req_filter],
                                ranker=RRFRanker(k=60),
                                limit=settings.TOP_K,
                                output_fields=["id", "text", "document_id", "document_name", "number_page", "category", "access_rights"],
                                filter=f'category == "{settings.CATEGORY_FILTER}"'
                            )
                        except Exception as e:
                            print(f"DEBUG: Hybrid search with category filter failed, using dense-only: {e}")
                            # Dense-only search with category filter
                            results = client.search(
                                collection_name=settings.MILVUS_COLLECTION_NAME,
                                data=[primary_embedding_data['dense']],
                                anns_field="vector",
                                search_params=search_params,
                                limit=settings.TOP_K,
                                output_fields=["id", "text", "document_id", "document_name", "number_page", "category", "access_rights"],
                                filter=f'category == "{settings.CATEGORY_FILTER}"'
                            )
                    else:
                        # Dense-only search with category filter
                        results = client.search(
                            collection_name=settings.MILVUS_COLLECTION_NAME,
                            data=[primary_embedding_data['dense']],
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
        
        # Process hybrid search results
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
                        "source": "hybrid",
                        "keyword": hit.get("keyword", ""),
                        "summary": hit.get("summary", ""),
                        "context": hit.get("context", "")
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
                        "summary": hit.get("summary", ""),
                        "context": hit.get("context", "")
                    })
                    seen_ids.add(hit_id)
        except Exception as e:
            print(f"DEBUG: Error processing keyword results: {e}")
            # Continue without keyword results
        
        print(f"DEBUG: Combined results: {len(passages)} passages ({len([p for p in passages if p['source'] == 'hybrid'])} hybrid, {len([p for p in passages if p['source'] == 'keyword'])} keyword)")
        
        # Apply BM25-based reranking
        passages = await _rerank_passages(passages, question)
        
        # Apply cross-encoder reranking if enabled
        if settings.USE_RERANKER and len(passages) > 0:
            passages = await _apply_cross_encoder_reranking(passages, question)
        
        if not passages:
            return orjson.dumps({
                "domain": "RTS",
                "answer": settings.REFUSAL_TEXT,
                "citations": [],
                "diagnostic": {
                    "mode": "bge_m3_hybrid_search_with_reranking",
                    "hybrid_hits": 0,
                    "keyword_hits": 0,
                    "total_hits": 0,
                    "collection": settings.MILVUS_COLLECTION_NAME,
                    "search_strategy": "hybrid_no_filter",
                    "available_categories": available_categories,
                    "embedding_dimension": len(primary_embedding_data['dense']),
                    "search_limit": settings.TOP_K,
                    "query_expansion": len(expanded_queries) > 1,
                    "reranking_applied": True,
                    "reason": "no_passages_after_filtering",
                    "min_relevance_score": settings.MIN_RELEVANCE_SCORE,
                    "extracted_keywords": _extract_keywords_from_question(question)
                }
            }).decode()
        
        # Generate answer using LLM with context
        context_lines = []
        citations = []
        seen_citations = set()
        
        # Use top passages after reranking, but limit context length to prevent timeout
        relevant_passages = passages[:min(settings.MAX_CONTEXT, settings.MAX_PASSAGES)]
        print(f"DEBUG: Using top {len(relevant_passages)} passages after reranking (limited to prevent timeout)")
        
        # Quality filtering based on general criteria
        quality_passages = []
        
        for passage in relevant_passages:
            text = passage.get("text", "").lower()
            
            # Skip very short content (likely not informative)
            if len(text) < settings.MIN_TEXT_LENGTH:
                print(f"DEBUG: Skipping very short passage: {text[:50]}...")
                continue
            
            # Skip passages with very low relevance score
            if passage.get("relevance_score", 0) < settings.MIN_RELEVANCE_SCORE:
                continue
            
            # Skip passages that are mostly administrative/approval headers
            # Check if passage has high density of non-content markers
            admin_markers = text.count("subject") + text.count("date") + text.count("persetujuan") + text.count("approval")
            text_length = len(text.split())
            
            # If more than 30% of short text is admin markers, likely not technical content
            if text_length < 50 and admin_markers > 3:
                print(f"DEBUG: Skipping administrative header: {text[:50]}...")
                continue
            
            # Skip passages that are mostly HTML/images
            html_markers = text.count("<!--") + text.count("image") + text.count("&amp;")
            if html_markers > 5 and text_length < 100:
                print(f"DEBUG: Skipping HTML/image-heavy passage: {text[:50]}...")
                continue
            
            quality_passages.append(passage)
        
        # Use up to MAX_PASSAGES
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
        
        # Generate answer using LLM (production-ready prompt engineering)
        system_prompt = (
            f"Anda adalah asisten teknis ahli untuk domain {settings.DOMAIN}. "
            "Tugas Anda adalah menjawab pertanyaan berdasarkan konteks dokumen yang diberikan.\n\n"
            "Prinsip penting:\n"
            "- PRIORITAS: Cari dan jawab informasi spesifik yang diminta\n"
            "- Jika ada informasi relevan di konteks, JAWAB dengan detail\n"
            "- Gunakan bahasa teknis yang tepat dan profesional\n"
            "- Sertakan nilai, angka, atau spesifikasi jika ada\n"
            f"- HANYA jika sama sekali tidak ada informasi relevan, jawab: '{settings.REFUSAL_TEXT}'"
        )
        
        user_prompt = (
            f"Konteks dari dokumen {settings.DOMAIN}:\n{context}\n\n"
            f"Pertanyaan: {question}\n\n"
            "Instruksi:\n"
            "1. Analisis konteks untuk menemukan informasi yang menjawab pertanyaan\n"
            "2. Jika ada informasi teknis, nilai, atau spesifikasi yang relevan - JAWAB dengan lengkap\n"
            "3. Jika konteks membahas topik terkait - berikan penjelasan berdasarkan konteks\n"
            "4. Sertakan referensi dokumen untuk kredibilitas\n"
            f"5. Hanya jika konteks TIDAK relevan sama sekali dengan pertanyaan, jawab: '{settings.REFUSAL_TEXT}'\n\n"
            "Jawaban (berikan detail teknis jika ada):"
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
                "mode": "bge_m3_hybrid_search_with_reranking",
                "hybrid_hits": len([p for p in passages if p.get("source") == "hybrid"]),
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


# Removed hardcoded expansion and filtering rules
# These can be loaded from configuration files or databases in production

def _extract_keywords_from_question(question: str) -> List[str]:
    """Extract relevant keywords from question using TF-IDF-like approach (production-ready)"""
    import re
    from collections import Counter
    
    question_lower = question.lower()
    
    # Extract all meaningful tokens (alphanumeric, including numbers and mixed)
    tokens = re.findall(r'\b[\w-]+\b', question_lower)
    
    # Load or use dynamic stopwords (can be expanded from config/database)
    stopwords = _get_stopwords()
    
    # Filter tokens by various criteria
    filtered_tokens = []
    for token in tokens:
        # Skip if it's a stopword
        if token in stopwords:
            continue
        
        # Skip if too short (single char) or all digits
        if len(token) < 2 or token.isdigit():
            continue
        
        # Keep the token
        filtered_tokens.append(token)
    
    # Calculate importance score for each token based on:
    # 1. Token length (shorter technical terms can be important)
    # 2. Frequency in question (repeated terms are important)
    # 3. Position (earlier terms might be more important)
    
    token_scores = {}
    token_counts = Counter(filtered_tokens)
    
    for idx, token in enumerate(filtered_tokens):
        if token not in token_scores:
            # Base score from frequency
            freq_score = token_counts[token] * settings.KEYWORD_FREQ_WEIGHT
            
            # Position score (earlier = more important, but not too heavily weighted)
            position_score = 1.0 / (idx + 1) * settings.KEYWORD_POSITION_WEIGHT
            
            # Length score (balanced - not too short, not too long)
            if 2 <= len(token) <= 4:
                length_score = 1.5  # Short technical terms
            elif 5 <= len(token) <= 8:
                length_score = 1.2  # Medium terms
            else:
                length_score = 1.0  # Longer terms
            
            # Check if it contains numbers (technical codes often do)
            has_numbers = bool(re.search(r'\d', token))
            number_score = 1.3 if has_numbers else 1.0
            
            # Combined score
            token_scores[token] = freq_score + position_score + length_score + number_score
    
    # Sort by score and return top keywords
    sorted_tokens = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return top N keywords (configurable from settings)
    top_n = min(settings.MAX_KEYWORDS, len(sorted_tokens))
    return [token for token, score in sorted_tokens[:top_n]]

def _get_stopwords() -> set:
    """Get stopwords for filtering (can be loaded from config/database in production)"""
    # Common stopwords in Indonesian and English
    # In production, this could be loaded from a file or database
    return {
        # English
        "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "a", "an", "as", "are", "be", "been", "is", "was", "were", "will", "would",
        "can", "could", "do", "does", "did", "have", "has", "had",
        "this", "that", "these", "those", "what", "where", "when", "which", "who", "whom",
        "how", "why", "there", "here", "about", "into", "through", "during",
        # Indonesian
        "yang", "dan", "atau", "di", "ke", "dari", "pada", "untuk", "dengan", "oleh",
        "dalam", "adalah", "akan", "telah", "sudah", "belum", "dapat", "bisa",
        "ada", "tidak", "ini", "itu", "tersebut", "juga", "saja", "hanya",
        "berapa", "apa", "siapa", "dimana", "kapan", "mengapa", "bagaimana",
        # Question words that are usually not content-bearing
        "what", "what's", "whats",
    }

async def _expand_query(question: str) -> List[str]:
    """
    Query expansion using an on-premise LLM to generate variations.
    """
    # Start with original query
    expanded = {question}  # Use a set to handle duplicates

    try:
        import httpx

        prompt = (
            "You are an expert query expander. Your task is to reformulate the user's question to improve search retrieval. "
            "Generate 2 alternative phrasings for the following question. "
            'Return a JSON list of strings, like ["variation 1", "variation 2"].\n\n'
            f'Question: "{question}"\n\n'
            "JSON list of variations:"
        )

        llm_payload = {
            "model": settings.OLLAMA_MODEL,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that provides JSON output."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"}
        }

        async with httpx.AsyncClient(timeout=15.0) as client:  # Shorter timeout for this
            llm_response = await client.post(
                f"{settings.OLLAMA_BASE_URL}/v1/chat/completions",
                json=llm_payload,
            )
            llm_response.raise_for_status()
            llm_data = llm_response.json()

        # Extract variations from the response
        variations_str = llm_data["choices"][0]["message"]["content"]
        # The response is a JSON string, so we need to parse it.
        # It might be inside a larger JSON object from the LLM.
        try:
            # The model might return a markdown JSON block, so we clean it.
            import re
            json_str_match = re.search(r'\[.*\]', variations_str, re.DOTALL)
            if json_str_match:
                variations = orjson.loads(json_str_match.group(0))
                if isinstance(variations, list):
                    expanded.update(variations)
        except (orjson.JSONDecodeError, TypeError) as e:
            print(f"DEBUG: Failed to parse query expansion variations: {e}. Content: {variations_str}")

    except Exception as e:
        print(f"DEBUG: Query expansion LLM call failed: {e}")

    final_queries = list(expanded)
    print(f"DEBUG: Expanded queries: {final_queries}")
    return final_queries[:settings.MAX_EXPANDED_QUERIES]  # Limit number of queries


async def _apply_cross_encoder_reranking(passages: List[dict], question: str) -> List[dict]:
    """
    Apply cross-encoder reranking using BGE-reranker-v2-m3
    Following Milvus best practices: https://milvus.io/docs/contextual_retrieval_with_milvus.md
    
    This provides more accurate relevance scoring than vector similarity alone.
    """
    if not passages:
        return passages
    
    try:
        from FlagEmbedding import FlagReranker
        
        print(f"DEBUG: Applying BGE cross-encoder reranking to {len(passages)} passages")
        
        # Initialize BGE reranker from local path
        reranker = FlagReranker(
            settings.BGE_RERANKER_MODEL_PATH,
            use_fp16=True,  # Use FP16 for faster inference
            device='cuda'  # Use GPU if available
        )
        
        # Prepare query-passage pairs for reranking
        pairs = [
            [question, f'{p.get("text", "")}\n\n{p.get("context", "")}'] 
            for p in passages if p.get("text")
        ]
        
        # Get reranking scores
        scores = reranker.compute_score(pairs, normalize=True)
        
        # If scores is a single value, wrap it in a list
        if not isinstance(scores, list):
            scores = [scores]
        
        # Update passages with reranker scores
        for i, passage in enumerate(passages):
            passage["reranker_score"] = scores[i]
        
        # Sort by reranker score (higher is better)
        passages.sort(key=lambda x: x.get("reranker_score", 0), reverse=True)
        
        # Filter by minimum reranker score to remove irrelevant results
        filtered_passages = [
            p for p in passages 
            if p.get("reranker_score", 0) >= settings.MIN_RERANKER_SCORE
        ]
        
        print(f"DEBUG: Reranker filtered: {len(filtered_passages)}/{len(passages)} passages above threshold {settings.MIN_RERANKER_SCORE}")
        
        # If too few passages pass threshold, take top N anyway but warn
        if len(filtered_passages) < 3:
            print(f"WARNING: Only {len(filtered_passages)} passages above reranker threshold, taking top {settings.FINAL_TOP_K} anyway")
            final_passages = passages[:settings.FINAL_TOP_K]
        else:
            # Return top N passages after reranking
            final_passages = filtered_passages[:settings.FINAL_TOP_K]
        
        print(f"DEBUG: Cross-encoder reranking completed, selected top {len(final_passages)} passages")
        print(f"DEBUG: Top 3 reranked passages:")
        for i, passage in enumerate(final_passages[:3]):
            print(f"  {i+1}. Reranker score: {passage.get('reranker_score', 0):.4f}, Text: {passage['text'][:100]}...")
        
        return final_passages
        
    except Exception as e:
        print(f"DEBUG: Cross-encoder reranking error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        # Return original passages if reranking fails
        return passages[:settings.FINAL_TOP_K]

async def _rerank_passages(passages: List[dict], question: str) -> List[dict]:
    """
    Rerank passages using BM25-like scoring + metadata signals (production-ready)
    
    This approach is general and doesn't rely on hardcoded patterns.
    Can be enhanced with:
    - Cross-encoder reranking models
    - Learning-to-rank models
    - User feedback signals
    """
    if not passages:
        return passages
    
    import re
    from collections import Counter
    
    question_lower = question.lower()
    question_tokens = re.findall(r'\b[\w-]+\b', question_lower)
    question_keywords = _extract_keywords_from_question(question)
    
    # Get stopwords for filtering
    stopwords = _get_stopwords()
    question_content_tokens = [t for t in question_tokens if t not in stopwords]
    
    def calculate_bm25_like_score(text: str, keywords: List[str], content_tokens: List[str]) -> float:
        """Calculate BM25-like relevance score"""
        text_lower = text.lower()
        text_tokens = re.findall(r'\b[\w-]+\b', text_lower)
        text_token_counts = Counter(text_tokens)
        
        score = 0.0
        
        # 1. Keyword matching (highest priority)
        for kw in keywords:
            # Exact word boundary match
            pattern = r'\b' + re.escape(kw) + r'\b'
            matches = len(re.findall(pattern, text_lower))
            if matches > 0:
                # BM25-like term frequency saturation
                # tf_score = (k1 + 1) * tf / (k1 * (1 - b + b * dl/avgdl) + tf)
                # Simplified version (using configurable k1 from settings)
                k1 = settings.BM25_K1_KEYWORD
                tf = matches
                tf_score = ((k1 + 1) * tf) / (k1 + tf)
                score += tf_score * 3.0  # Weight for exact keyword matches
        
        # 2. Content token matching (secondary)
        for token in content_tokens:
            if token in text_token_counts:
                tf = text_token_counts[token]
                k1 = settings.BM25_K1_CONTENT
                tf_score = ((k1 + 1) * tf) / (k1 + tf)
                score += tf_score * 0.5
        
        return score
    
    def calculate_relevance_score(passage):
        # Start with vector search score (already normalized by Milvus)
        vector_score = -passage.get("score", 0)  # Negative because L2 distance (lower is better)
        
        text = passage.get("text", "")
        keyword_field = passage.get("keyword", "")
        summary_field = passage.get("summary", "")
        context_field = passage.get("context", "")
        
        # Calculate text relevance
        text_score = calculate_bm25_like_score(text, question_keywords, question_content_tokens)
        
        # Boost from metadata fields
        metadata_score = 0.0
        if keyword_field:
            metadata_score += calculate_bm25_like_score(keyword_field, question_keywords, question_content_tokens) * 0.5
        if summary_field:
            metadata_score += calculate_bm25_like_score(summary_field, question_keywords, question_content_tokens) * 0.5
        
        # Source type bonus (hybrid search is generally better)
        source_score = 0.3 if passage.get("source") == "hybrid" else 0.1
        
        # Quality signals
        quality_score = 0.0
        text_len = len(text)
        
        # Penalize very short texts (likely not informative)
        if text_len < 50:
            quality_score -= 1.0
        elif text_len < 100:
            quality_score -= 0.3
        
        # Penalize texts with too much HTML/formatting
        html_ratio = (text.count("<") + text.count(">")) / max(text_len, 1)
        if html_ratio > 0.1:
            quality_score -= html_ratio * 5.0
        
        # Combined score
        # Weights are configurable from settings and can be tuned per domain
        final_score = (
            vector_score * settings.VECTOR_SCORE_WEIGHT +      # Vector similarity
            text_score * settings.TEXT_SCORE_WEIGHT +          # Text relevance (highest weight)
            metadata_score * settings.METADATA_SCORE_WEIGHT +  # Metadata signals (base)
            quality_score                                       # Quality signals
        )
        
        return final_score
    
    # Calculate relevance scores
    for passage in passages:
        passage["relevance_score"] = calculate_relevance_score(passage)
    
    # Sort by relevance score (higher is better)
    passages.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    # Filter out passages with very low relevance scores
    filtered_passages = [p for p in passages if p["relevance_score"] > settings.MIN_RELEVANCE_SCORE]
    
    # Additional filter: Dynamic keyword presence check
    # This ensures passages have at least some overlap with query content
    extracted_keywords = _extract_keywords_from_question(question)
    if extracted_keywords and len(filtered_passages) > settings.MAX_PASSAGES * 2:
        # Only apply keyword filter if we have too many results
        # This prevents over-filtering when results are already limited
        keyword_filtered = []
        for passage in filtered_passages:
            text_lower = passage.get("text", "").lower()
            # Check if at least one keyword exists (dynamic, based on extracted keywords)
            # Prioritize top keywords but check all
            keyword_match_score = sum(1 for kw in extracted_keywords if kw in text_lower)
            if keyword_match_score > 0:
                keyword_filtered.append(passage)
        
        # Only apply keyword filter if it doesn't remove too many results
        if len(keyword_filtered) >= min(5, len(filtered_passages) // 2):
            filtered_passages = keyword_filtered
            print(f"DEBUG: Applied dynamic keyword filter, {len(filtered_passages)} passages remain")
    
    print(f"DEBUG: Passages after filtering: {len(filtered_passages)}/{len(passages)} (min score: {settings.MIN_RELEVANCE_SCORE})")
    print(f"DEBUG: Top 5 passages after reranking:")
    for i, passage in enumerate(filtered_passages[:5]):
        print(f"  {i+1}. Score: {passage['relevance_score']:.3f}, Source: {passage['source']}, Text: {passage['text'][:100]}...")
    
    return filtered_passages


async def _generate_bge_m3_embedding(text: str) -> dict:
    """
    Generate BGE-M3 dense and sparse embeddings using local model
    Following Milvus best practices for hybrid retrieval
    Reference: https://milvus.io/docs/contextual_retrieval_with_milvus.md
    """
    try:
        if settings.USE_BGE_M3_HYBRID:
            # Use FlagEmbedding BGE-M3 for true hybrid search
            from FlagEmbedding import BGEM3FlagModel
            
            print(f"DEBUG: Generating BGE-M3 hybrid embedding for text: '{text[:100]}...'")
            
            # Initialize BGE-M3 model from local path
            # This model generates both dense (1024-dim) and sparse embeddings
            bge_m3 = BGEM3FlagModel(
                settings.BGE_M3_MODEL_PATH,
                use_fp16=True,  # Use FP16 for faster inference
                device='cuda'  # Use GPU if available, fallback to CPU
            )
            
            # Generate both dense and sparse embeddings in single pass
            embeddings = bge_m3.encode(
                [text],
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False  # Disable ColBERT for speed
            )
            
            # Debug: Print embeddings structure
            print(f"DEBUG: Embeddings keys: {embeddings.keys() if isinstance(embeddings, dict) else type(embeddings)}")
            
            # Handle different output formats from BGE-M3
            if isinstance(embeddings, dict):
                # Standard dictionary format
                dense_vector = embeddings.get('dense_vecs', embeddings.get('dense', []))[0]
                sparse_vector = embeddings.get('lexical_weights', embeddings.get('sparse', {}))[0]
                
                # Convert numpy array to list if needed
                if hasattr(dense_vector, 'tolist'):
                    dense_vector = dense_vector.tolist()
            else:
                # If embeddings is not a dict, try to extract from object
                dense_vector = getattr(embeddings, 'dense_vecs', getattr(embeddings, 'dense', None))
                if dense_vector is not None:
                    dense_vector = dense_vector[0]
                    if hasattr(dense_vector, 'tolist'):
                        dense_vector = dense_vector.tolist()
                else:
                    raise ValueError(f"Cannot extract dense embeddings from: {type(embeddings)}")
                
                sparse_vector = getattr(embeddings, 'lexical_weights', getattr(embeddings, 'sparse', {}))
                if sparse_vector and len(sparse_vector) > 0:
                    sparse_vector = sparse_vector[0]
                else:
                    sparse_vector = {}
            
            print(f"DEBUG: Generated dense embedding dimension: {len(dense_vector)}")
            print(f"DEBUG: Generated sparse embedding with {len(sparse_vector) if isinstance(sparse_vector, dict) else 'N/A'} non-zero elements")
            print(f"DEBUG: Dense embedding sample (first 5 values): {dense_vector[:5]}")
            
            return {
                'dense': dense_vector,
                'sparse': sparse_vector
            }
        else:
            # Fallback: Use Ollama for dense-only embeddings
            from langchain_ollama import OllamaEmbeddings
            
            print(f"DEBUG: Generating Ollama dense embedding for text: '{text[:100]}...'")
            
            embeddings = OllamaEmbeddings(
                model="bge-m3",
                base_url=settings.OLLAMA_BASE_URL
            )
            
            dense_vector = embeddings.embed_query(text)
            
            print(f"DEBUG: Generated dense embedding dimension: {len(dense_vector)}")
            
            return {
                'dense': dense_vector,
                'sparse': {}  # No sparse embeddings
            }
        
    except Exception as e:
        print(f"BGE-M3 embedding generation error: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        
        # Fallback to zero vectors
        return {
            'dense': [0.0] * 1024,
            'sparse': {}
        }

# Removed pseudo-sparse vector generation
# Now using real sparse embeddings from BGE-M3

async def _generate_embedding(text: str) -> List[float]:
    """Legacy function for backward compatibility - generates only dense embedding"""
    embedding_data = await _generate_bge_m3_embedding(text)
    return embedding_data['dense']


async def _generate_chunk_context(doc_content: str, chunk_content: str) -> str:
    """Generates a succinct context for a chunk based on the full document."""
    try:
        import httpx

        # Prompt inspired by Anthropic's contextual retrieval cookbook
        DOCUMENT_CONTEXT_PROMPT = f"""
        <document>
        {doc_content}
        </document>
        """

        CHUNK_CONTEXT_PROMPT = f"""
        Here is the chunk we want to situate within the whole document
        <chunk>
        {chunk_content}
        </chunk>

        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
        Answer only with the succinct context and nothing else.
        """

        llm_payload = {
            "model": settings.OLLAMA_MODEL,
            "temperature": 0.0,
            "messages": [
                {"role": "user", "content": DOCUMENT_CONTEXT_PROMPT + CHUNK_CONTEXT_PROMPT},
            ],
        }

        async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT) as client:
            llm_response = await client.post(
                f"{settings.OLLAMA_BASE_URL}/v1/chat/completions",
                json=llm_payload,
            )
            llm_response.raise_for_status()
            llm_data = llm_response.json()
        
        context = llm_data["choices"][0]["message"]["content"]
        return context.strip()

    except Exception as e:
        print(f"DEBUG: Context generation failed: {e}")
        return "" # Return empty string on failure


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
    # By returning the tool output directly, we prevent the LLM from summarizing it.
    # The 'content' of the AIMessage will be the raw JSON string from the tool.
    return workflow.compile().with_config(output_keys=["messages"])

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
