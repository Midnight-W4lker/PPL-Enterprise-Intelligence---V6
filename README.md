# PPL Enterprise Intelligence — V8

Unified local-first, multi-document corporate RAG for PPL reports. One codebase with a pluggable LLM backend (`ollama` or `vertex`).

## Features

- Annual + quarterly + other PDF corpus under `data/raw`
- Document/year/period metadata with v7 ingest (layout confidence, manifest, chunking, table handling)
- BGE-M3 embeddings + BGE reranker-v2-m3 (full-size models)
- BM25 + FAISS hybrid retrieval + RRF fusion
- Pluggable generation: local Ollama/Qwen or Vertex AI Gemini
- Strict evidence citations `[E1]`, exact table/field/row metadata
- Integrated PDF page renderer: citation click → exact page
- `/api/health`, `/api/system`, `/api/documents`, `/api/chat`
- Hugging Face offline mode; no silent model downloads

## Local setup (Ollama)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For GPU embeddings locally, install a CUDA-enabled PyTorch build instead of the CPU wheel in `requirements.txt`:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Copy model directories into `models\bge-m3` and `models\bge-reranker-v2-m3`, and reports into `data\raw`.

Optional: verify CUDA (not required — CPU fallback works):

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Set environment (defaults to `LLM_BACKEND=ollama` when `VERTEX_PROJECT_ID` is unset):

```powershell
$env:LLM_BACKEND = "ollama"
$env:OLLAMA_MODEL = "qwen2.5:7b"
# $env:DEVICE = "cuda"   # optional override; auto-detects if omitted
```

Build the index:

```powershell
python -m app.ingest
```

Run:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

Check Ollama with `ollama list` and `Invoke-RestMethod http://127.0.0.1:11434/api/tags`. Do not run `ollama serve` if the server is already running.

## GCP deploy

See [README-GCP.md](README-GCP.md) for Cloud Run + Vertex AI deployment. Set `LLM_BACKEND=vertex` (auto-selected when `VERTEX_PROJECT_ID` is set).

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BACKEND` | `ollama` (local) / `vertex` (when `VERTEX_PROJECT_ID` set) | Generation backend |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama API base URL |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Ollama model name |
| `VERTEX_PROJECT_ID` | _(empty)_ | GCP project for Vertex AI |
| `VERTEX_LOCATION` | `us-central1` | Vertex region |
| `VERTEX_MODEL` | `gemini-2.5-flash` | Gemini model |
| `DEVICE` | auto (`cuda` if available, else `cpu`) | Embedding device |
| `RERANKER_DEVICE` | `cpu` | Reranker device |
| `DEPLOYMENT_PROFILE` | `standard` | `standard`, `gcp-trial`, or `corporate` |

## Git

PDFs, indexes, model weights and `.venv` are intentionally excluded. Keep them in controlled local/object storage rather than GitHub.

## Important limitation

RAG does not guarantee an answer to every possible question. It answers from retrieved indexed evidence and should say when evidence is insufficient. Production use requires retrieval/answer evaluation and access control.
