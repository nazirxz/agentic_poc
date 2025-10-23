from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import httpx
from langgraph.graph import END, StateGraph

from .settings import AgentSettings

DEFAULT_MODEL = "qwen2.5:latest"
SYSTEM_PROMPT = (
    "Anda Agent RTS. Jawab hanya dari dokumen kategori RTS. Alur: detect_intent -> (doc_lookup) -> "
    "vector_search -> (rerank) -> guardrails -> compose -> cite. Selalu sertakan sitasi {document_name} "
    "p.{number_page}. Jika bukti lemah: 'Tidak ditemukan dalam RTS.'"
)


class Candidate(TypedDict, total=False):
    document_id: str
    document_name: str


class Passage(TypedDict, total=False):
    document_id: str
    document_name: str
    number_page: int
    text: str
    score: float


class RAGState(TypedDict, total=False):
    question: str
    intent: str
    candidates: List[Candidate]
    passages: List[Passage]
    answer: str
    citations: List[str]
    diag: Dict[str, Any]


def finalize_answer_and_citations(
    answer: Optional[str],
    passages: List[Passage],
    settings: AgentSettings,
    max_context: Optional[int] = None,
) -> Tuple[str, List[str]]:
    max_items = max_context or settings.MAX_CONTEXT
    if answer == settings.REFUSAL_TEXT:
        return settings.REFUSAL_TEXT, []

    citations: List[str] = []
    seen = set()
    for passage in (passages or [])[:max_items]:
        name = passage.get("document_name") or passage.get("document_id") or "Unknown"
        page = passage.get("number_page") or passage.get("page") or passage.get("page_number")
        page_str = str(page) if page is not None else "?"
        citation = f"{name} p.{page_str}"
        if citation in seen:
            continue
        seen.add(citation)
        citations.append(citation)

    if not citations:
        return settings.REFUSAL_TEXT, []

    return (answer or "").strip() or settings.REFUSAL_TEXT, citations


def build_graph(settings: AgentSettings | None = None):
    settings = settings or AgentSettings()

    def merge_diag(state: RAGState, key: str, value: Any) -> Dict[str, Any]:
        diag = dict(state.get("diag", {}))
        diag[key] = value
        return diag

    def detect_intent(state: RAGState) -> Dict[str, Any]:
        question = state.get("question", "") or ""
        normalized = question.lower()
        markers = ("dokumen", ".pdf", "halaman ")
        intent = "doc" if any(marker in normalized for marker in markers) else "general"
        diag = merge_diag(state, "detect_intent", {"intent": intent})
        return {"intent": intent, "diag": diag}

    def doc_lookup(state: RAGState) -> Dict[str, Any]:
        question = state.get("question", "") or ""
        matches = re.findall(r"([\w\-]+\.pdf)", question, flags=re.IGNORECASE)
        seen = set()
        candidates: List[Candidate] = []
        for match in matches:
            key = match.lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append({"document_id": match, "document_name": match})
        diag = merge_diag(state, "doc_lookup", {"candidates": [c["document_name"] for c in candidates]})
        return {"candidates": candidates, "diag": diag}

    async def retrieve(state: RAGState) -> Dict[str, Any]:
        filters: Dict[str, Any] = {
            "category": settings.CATEGORY_FILTER,
            "access_rights": "internal",
        }

        candidates = state.get("candidates") or []
        if state.get("intent") == "doc" and candidates:
            filters["document_id"] = [c["document_id"] for c in candidates if c.get("document_id")]

        payload = {
            "q": state.get("question", ""),
            "top_k": settings.TOP_K,
            "filters": filters,
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{settings.MILVUS_RAG_URL}/search",
                    json=payload,
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:  # pragma: no cover - network failure path
            diag = merge_diag(state, "retrieve", {"error": str(exc), "payload": payload})
            return {"passages": [], "diag": diag}

        passages = data.get("passages") or data.get("results") or []
        diag = merge_diag(
            state,
            "retrieve",
            {"count": len(passages), "filters": filters, "top_k": settings.TOP_K},
        )
        return {"passages": passages, "diag": diag}

    async def rerank(state: RAGState) -> Dict[str, Any]:
        passages = state.get("passages") or []
        if not settings.RERANK_ENABLED or not passages:
            reason = "disabled" if not settings.RERANK_ENABLED else "no_passages"
            diag = merge_diag(state, "rerank", {"skipped": reason})
            return {"diag": diag}

        payload = {
            "q": state.get("question", ""),
            "passages": passages,
            "top_n": settings.RERANK_TOP_N,
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{settings.RERANK_URL}/rerank",
                    json=payload,
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:  # pragma: no cover - network failure path
            diag = merge_diag(state, "rerank", {"error": str(exc)})
            return {"diag": diag}

        ranked = data.get("passages") or data.get("results") or passages
        limited = ranked[: settings.RERANK_TOP_N]
        diag = merge_diag(state, "rerank", {"count": len(limited)})
        return {"passages": limited, "diag": diag}

    def guardrails(state: RAGState) -> Dict[str, Any]:
        passages = state.get("passages") or []
        scores: List[float] = []
        for passage in passages:
            try:
                scores.append(float(passage.get("score", 0)))
            except (TypeError, ValueError):
                continue

        hits = len(passages)
        avg_score = sum(scores) / len(scores) if scores else 0.0
        diag = merge_diag(
            state,
            "guardrails",
            {"hits": hits, "avg_score": avg_score, "threshold": settings.THRESHOLD},
        )

        if hits < 2 or avg_score < settings.THRESHOLD:
            return {"answer": settings.REFUSAL_TEXT, "diag": diag}

        return {"diag": diag}

    async def compose(state: RAGState) -> Dict[str, Any]:
        if state.get("answer") == settings.REFUSAL_TEXT:
            diag = merge_diag(state, "compose", {"skipped": "guardrails_refusal"})
            return {"diag": diag}

        passages = state.get("passages") or []
        if not passages:
            diag = merge_diag(state, "compose", {"skipped": "no_passages"})
            return {"answer": settings.REFUSAL_TEXT, "diag": diag}

        limited = passages[: settings.MAX_CONTEXT]
        context_lines = []
        for passage in limited:
            name = passage.get("document_name") or passage.get("document_id") or "Unknown"
            page = passage.get("number_page") or passage.get("page") or passage.get("page_number")
            page_str = str(page) if page is not None else "?"
            text = passage.get("text") or ""
            context_lines.append(f"{name} p.{page_str}: {text}")

        context = "\n\n".join(context_lines)
        user_prompt = (
            f"Pertanyaan: {state.get('question', '')}\n"
            f"Konteks (maks {settings.MAX_CONTEXT} potongan):\n{context}\n"
            f"Gaya: {settings.STYLE}\n"
            f"Instruksi: Jawab ringkas, bahasa Indonesia. Sertakan sitasi pada setiap pernyataan faktual. "
            f"Jika bukti lemah, balas {settings.REFUSAL_TEXT}."
        )

        payload = {
            "model": DEFAULT_MODEL,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{settings.OLLAMA_BASE_URL}/v1/chat/completions",
                    json=payload,
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as exc:  # pragma: no cover - network failure path
            diag = merge_diag(state, "compose", {"error": str(exc)})
            return {"answer": settings.REFUSAL_TEXT, "diag": diag}

        try:
            answer = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            diag = merge_diag(state, "compose", {"error": f"invalid_response: {exc}"})
            return {"answer": settings.REFUSAL_TEXT, "diag": diag}

        diag = merge_diag(state, "compose", {"used_passages": len(limited)})
        return {"answer": answer, "diag": diag}

    def cite(state: RAGState) -> Dict[str, Any]:
        answer, citations = finalize_answer_and_citations(
            state.get("answer"),
            state.get("passages") or [],
            settings,
            settings.MAX_CONTEXT,
        )

        diag = merge_diag(state, "cite", {"count": len(citations)})
        update: Dict[str, Any] = {"citations": citations, "diag": diag}

        if answer != state.get("answer"):
            update["answer"] = answer

        return update

    workflow = StateGraph(RAGState)
    workflow.add_node("detect_intent", detect_intent)
    workflow.add_node("doc_lookup", doc_lookup)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rerank", rerank)
    workflow.add_node("guardrails", guardrails)
    workflow.add_node("compose", compose)
    workflow.add_node("cite", cite)

    workflow.set_entry_point("detect_intent")
    workflow.add_edge("detect_intent", "doc_lookup")
    workflow.add_edge("doc_lookup", "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "guardrails")
    workflow.add_edge("guardrails", "compose")
    workflow.add_edge("compose", "cite")
    workflow.add_edge("cite", END)

    return workflow.compile()
