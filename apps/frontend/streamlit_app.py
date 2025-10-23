from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import httpx
import streamlit as st


def load_local_env() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


load_local_env()

DEFAULT_BASE_URL = "http://localhost:7000"
DEFAULT_TIMEOUT = 30.0

ORCHESTRATOR_BASE_URL = os.getenv("ORCHESTRATOR_BASE_URL", DEFAULT_BASE_URL)
ORCHESTRATOR_TIMEOUT = float(os.getenv("ORCHESTRATOR_TIMEOUT", DEFAULT_TIMEOUT))
DEBUG_DEFAULT = os.getenv("FRONTEND_DEBUG", "0").lower() in {"1", "true", "yes"}


def build_url(path: str) -> str:
    base = ORCHESTRATOR_BASE_URL.rstrip("/")
    return f"{base}/{path.lstrip('/')}"


def call_orchestrator(question: str) -> Dict[str, Any]:
    payload = {"question": question}
    with httpx.Client(timeout=ORCHESTRATOR_TIMEOUT) as client:
        response = client.post(build_url("/orchestrate"), json=payload)
        response.raise_for_status()
        return response.json()


def ping_health() -> Dict[str, Any]:
    with httpx.Client(timeout=ORCHESTRATOR_TIMEOUT) as client:
        response = client.get(build_url("/healthz"))
        response.raise_for_status()
        return response.json()


def render_history(messages: List[Dict[str, Any]], show_diag: bool) -> None:
    for message in messages:
        role = message.get("role", "assistant")
        content = message.get("content", "")
        metadata = message.get("metadata", {})
        with st.chat_message(role):
            st.markdown(content)
            if role == "assistant":
                citations = metadata.get("citations") or []
                domain = metadata.get("domain")
                if citations:
                    st.markdown("**Sitasi:** " + ", ".join(citations))
                if show_diag and metadata.get("diagnostic"):
                    with st.expander("Detail Diagnostic"):
                        st.json(metadata["diagnostic"])
                elif show_diag and metadata.get("error"):
                    st.warning(metadata["error"])


def main() -> None:
    st.set_page_config(page_title="URBUDDY Agentic Assistant", page_icon="🤖", layout="wide")
    st.title("URBUDDY Agentic Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []  # type: ignore[attr-defined]

    with st.sidebar:
        st.subheader("Konfigurasi")
        st.write(f"Orchestrator: `{ORCHESTRATOR_BASE_URL}`")
        show_diag = st.checkbox("Tampilkan diagnostic", value=DEBUG_DEFAULT)

        if st.button("Tes Koneksi Orchestrator"):
            try:
                health = ping_health()
            except httpx.HTTPError as exc:
                st.error(f"Gagal menjangkau orchestrator: {exc}")
            else:
                st.success(f"Healthz: {health}")
    render_history(st.session_state.messages, show_diag)  # type: ignore[arg-type]

    if prompt := st.chat_input("Ajukan pertanyaan Anda di sini..."):
        st.session_state.messages.append({"role": "user", "content": prompt})  # type: ignore[attr-defined]

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Menghubungi orchestrator..."):
                try:
                    result = call_orchestrator(prompt)
                    domain = result.get("domain")
                    answer = result.get("answer") or ""
                    citations = result.get("citations") or []
                    diagnostic = result.get("diagnostic") or {}
                    content = f"**Domain:** {domain or 'Tidak diketahui'}\n\n{answer}"
                    st.markdown(content)
                    if citations:
                        st.markdown("**Sitasi:** " + ", ".join(citations))
                    if show_diag and diagnostic:
                        with st.expander("Detail Diagnostic"):
                            st.json(diagnostic)
                    st.session_state.messages.append(  # type: ignore[attr-defined]
                        {
                            "role": "assistant",
                            "content": content,
                            "metadata": {
                                "citations": citations,
                                "domain": domain,
                                "diagnostic": diagnostic,
                            },
                        }
                    )
                except httpx.HTTPError as exc:
                    error_text = f"Gagal mendapatkan jawaban dari orchestrator: {exc}"
                    st.error(error_text)
                    st.session_state.messages.append(  # type: ignore[attr-defined]
                        {
                            "role": "assistant",
                            "content": "Maaf, terjadi kesalahan saat menghubungi orchestrator.",
                            "metadata": {"error": error_text},
                        }
                    )

        st.experimental_rerun()


if __name__ == "__main__":
    main()
