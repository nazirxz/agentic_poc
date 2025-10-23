from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class OrchestratorSettings(BaseSettings):
    AGENT_STK_URL: str = "http://localhost:7001/act"
    AGENT_RTS_URL: str = "http://localhost:7002/act"

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = OrchestratorSettings()

app = FastAPI()


class OrchestrationRequest(BaseModel):
    question: str


AGENT_ENDPOINTS = {
    "STK": settings.AGENT_STK_URL,
    "RTS": settings.AGENT_RTS_URL,
}


def pick_domain(question: str) -> str:
    normalized = question.lower()
    if any(keyword in normalized for keyword in ("rts", "rokan technical standard", "standard teknis")):
        return "RTS"
    return "STK"


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/orchestrate")
async def orchestrate(request: OrchestrationRequest) -> dict:
    domain = pick_domain(request.question)
    target_url = AGENT_ENDPOINTS[domain]
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(target_url, json=request.model_dump())
        except httpx.HTTPError as exc:  # pragma: no cover - network failure path
            raise HTTPException(status_code=502, detail=f"Failed to reach agent {domain}: {exc}") from exc

    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    try:
        payload = response.json()
    except ValueError as exc:  # pragma: no cover - unexpected agent response
        raise HTTPException(status_code=502, detail=f"Invalid JSON from agent {domain}") from exc

    payload["domain"] = domain
    return payload
