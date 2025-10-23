# URBUDDY Agentic POC

Eksperimen agentic AI yang terdiri dari dua agent domain (STK & RTS), sebuah orchestrator, dan antarmuka Streamlit. Setiap komponen berada di folder terpisah dalam `apps/`.

## Direktori

- `apps/orchestrator` &mdash; FastAPI untuk routing pertanyaan ke agent domain.
- `apps/agent-stk` &mdash; Agent STK dengan LangGraph RAG pipeline.
- `apps/agent-rts` &mdash; Agent RTS dengan LangGraph RAG pipeline.
- `apps/frontend` &mdash; Aplikasi Streamlit untuk interaksi pengguna.

## Konfigurasi Lingkungan (.env)

Setiap komponen membaca konfigurasi dari berkas `.env` di direktori masing-masing:

- `apps/orchestrator/.env` — URL endpoint agent (`AGENT_STK_URL`, `AGENT_RTS_URL`).
- `apps/agent-stk/.env` — Parameter domain STK (rerank dimatikan default).
- `apps/agent-stk/.env` — Parameter domain STK (rerank dimatikan default). Pertanyaan akan otomatis diarahkan ke koleksi TKI/TKO/TKPA/Pedoman berdasarkan kata kunci.
- `apps/agent-rts/.env` — Parameter domain RTS (rerank dimatikan default).
- `apps/frontend/.env` — Konfigurasi Streamlit (`ORCHESTRATOR_BASE_URL`, dll).

Saat berjalan di container, override nilai tersebut memakai variabel lingkungan pada Deployment/OpenShift.

## Prasyarat

- Python 3.11 atau lebih baru.
- Dependensi pihak ketiga (FastAPI, LangGraph, Streamlit, dst) di-install melalui `pip`.
- Layanan eksternal (Milvus RAG API, Reranker, LLM Ollama/OpenAI-compatible) tersedia bila ingin menjalankan alur RAG penuh.

## Instalasi Lokal

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip

# Instal tiap komponen (editable install memudahkan pengembangan)
pip install -e ./apps/orchestrator \
             -e ./apps/agent-stk \
             -e ./apps/agent-rts \
             -e ./apps/frontend
```

Setiap agent dapat diubah konfigurasinya melalui variabel lingkungan (lihat `app/settings.py` masing-masing).

> Catatan: jika tidak melakukan instalasi editable, jalankan service dengan `PYTHONPATH=$PWD python -m uvicorn ...` agar paket `apps` dapat ditemukan.

## Menjalankan Layanan Secara Lokal

Di terminal terpisah (setelah virtualenv aktif):

```bash
# Agent STK
uvicorn apps.agent-stk.app.server:app --port 7001

# Agent RTS
uvicorn apps.agent-rts.app.server:app --port 7002

# Orchestrator
uvicorn apps.orchestrator.app.main:app --port 7000

# Streamlit Frontend
streamlit run apps/frontend/streamlit_app.py --server.port 8501
```

Endpoint kesehatan:

- `GET http://localhost:7001/healthz`
- `GET http://localhost:7002/healthz`
- `GET http://localhost:7000/healthz`

Konfigurasi default pada `.env` mematikan reranker (karena belum tersedia layanan `RERANK_URL`). Aktifkan kembali dengan mengubah `RERANK_ENABLED=true` ketika service siap.

Contoh orkestrasi:

```bash
curl -s localhost:7000/healthz
curl -s -X POST localhost:7000/orchestrate \
     -H "Content-Type: application/json" \
     -d '{"question":"Apa kebijakan workover menurut STK?"}'
```

Streamlit akan otomatis mengirim pertanyaan ke `/orchestrate`; atur basis URL via env var:

```bash
export ORCHESTRATOR_BASE_URL=http://localhost:7000
streamlit run apps/frontend/streamlit_app.py --server.port 8501
```

## Catatan Pengujian

Saat ini tidak ada test suite aktif (folder `tests/` telah dihapus). Disarankan menambahkan test unit/integrasi seputar guardrails, routing, dan formatting sebelum masuk tahap produksi.

## Tahap Deployment ke OpenShift

### 1. Siapkan Image Container

Setiap komponen dapat memakai pola Dockerfile serupa berikut (sesuaikan path):

```dockerfile
FROM python:3.11-slim
WORKDIR /app

COPY apps/orchestrator /app
RUN pip install --no-cache-dir .

EXPOSE 7000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7000"]
```

Gunakan path `apps/agent-stk`, `apps/agent-rts`, dan `apps/frontend` untuk image lainnya. Pastikan dependency sistem (mis. build-essential) ditambahkan bila diperlukan oleh paket Python tertentu.

Build dan push image ke registry yang diakses OpenShift:

```bash
docker build -t registry.example.com/urbuddy/agent-orchestrator:v1 -f Dockerfile.orchestrator .
docker push registry.example.com/urbuddy/agent-orchestrator:v1
```

### 2. Buat Project dan Secret (opsional)

```bash
oc new-project urbbuddy-agentic
# simpan credential eksternal (RAG/LLM) bila diperlukan
oc create secret generic rag-secrets --from-literal=OLLAMA_BASE_URL=... --from-literal=MILVUS_RAG_URL=...
```

### 3. Deploy Agent STK dan RTS

```bash
oc new-app registry.example.com/urbuddy/agent-stk:v1 \
  --name=agent-stk \
  -e OLLAMA_BASE_URL=http://ollama:11434 \
  -e MILVUS_RAG_URL=http://milvus:19530 \
  -e RERANK_URL=http://reranker:8082

oc expose deployment/agent-stk --port=7001 --target-port=7001

oc new-app registry.example.com/urbuddy/agent-rts:v1 \
  --name=agent-rts \
  -e OLLAMA_BASE_URL=http://ollama:11434 \
  -e MILVUS_RAG_URL=http://milvus:19530 \
  -e RERANK_URL=http://reranker:8082

oc expose deployment/agent-rts --port=7002 --target-port=7002
```

Jika hanya perlu akses internal cluster, cukup buat Service (`oc expose`) tanpa Route publik.

### 4. Deploy Orchestrator

```bash
oc new-app registry.example.com/urbuddy/agent-orchestrator:v1 \
  --name=agent-orchestrator \
  -e AGENT_STK_URL=http://agent-stk:7001/act \
  -e AGENT_RTS_URL=http://agent-rts:7002/act

oc expose deployment/agent-orchestrator --port=7000 --target-port=7000
oc expose service/agent-orchestrator  # Membuat Route publik (opsional)
```

Jika menggunakan env var custom, tambahkan di Dockerfile atau deployment config lalu referensikan melalui `oc set env`.

### 5. Deploy Streamlit Frontend

```bash
oc new-app registry.example.com/urbuddy/agent-frontend:v1 \
  --name=agent-frontend \
  -e ORCHESTRATOR_BASE_URL=http://agent-orchestrator:7000

oc expose deployment/agent-frontend --port=8501 --target-port=8501
oc expose service/agent-frontend  # Route agar pengguna dapat mengakses UI
```

### 6. Probes & Scaling

- Tambahkan readiness/liveness probe pada setiap deployment (contoh: `HTTP GET /healthz`).
- Atur resource requests/limits sesuai baseline penggunaan.
- Gunakan Horizontal Pod Autoscaler jika beban fluktuatif.

### 7. Observability & Logging

- Arahkan log ke stdout/stderr (sudah default pada uvicorn/streamlit).
- Pertimbangkan menambahkan prom metrics (mis. via middleware) bila ingin integrasi dengan OpenShift Monitoring.

### 8. Konfigurasi Lanjutan

- Gunakan `ConfigMap` untuk konfigurasi non-rahasia (threshold, top_k).
- Simpan kredensial (API key, endpoint sensitif) dalam `Secret` dan mount sebagai env var.
- Jika memerlukan autentikasi, tambahkan reverse proxy (mis. OAuth2 Proxy) di depan Streamlit.

Dengan langkah di atas, keempat komponen dapat berjalan di OpenShift sebagai deployment terpisah dan saling berkomunikasi melalui Service internal cluster, sementara UI Streamlit menjadi satu-satunya titik akses publik.
