# Streamlit Frontend

Simple chat UI that talks to the orchestration service.

## Environment

Configure backend endpoint via environment variables:

- `ORCHESTRATOR_BASE_URL` (default `http://localhost:7000`)
- `ORCHESTRATOR_TIMEOUT` (seconds, default `30`)
- `FRONTEND_DEBUG` (`1` to show diagnostic details)

## Run Locally

```bash
cd apps/frontend
streamlit run streamlit_app.py --server.port 8501
```

Ensure the orchestrator service is reachable at the configured base URL before starting the frontend.
