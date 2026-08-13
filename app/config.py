import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
RAW_DIR = DATA / 'raw'
INDEX_DIR = DATA / 'index'
MODEL_DIR = ROOT / 'models'

# --- Device: auto-detect CUDA when available unless DEVICE is set explicitly.
# Local dev may use GPU; GCP Cloud Run defaults to CPU (full-size models still fit).
def _detect_device():
 env = os.getenv('DEVICE')
 if env:
  return env
 try:
  import torch
  return 'cuda' if torch.cuda.is_available() else 'cpu'
 except ImportError:
  return 'cpu'

DEVICE = _detect_device()
EMBED_MODEL = os.getenv('EMBED_MODEL', str(MODEL_DIR / 'bge-m3'))
EMBED_MODEL_HF = os.getenv('EMBED_MODEL_HF', 'BAAI/bge-m3')  # used only by download_models.py

RERANKER_MODEL = os.getenv('RERANKER_MODEL', str(MODEL_DIR / 'bge-reranker-v2-m3'))
RERANKER_MODEL_HF = os.getenv('RERANKER_MODEL_HF', 'BAAI/bge-reranker-v2-m3')
RERANKER_DEVICE = os.getenv('RERANKER_DEVICE', 'cpu')

# --- Pluggable LLM backend: ollama (local) or vertex (GCP).
# Default: vertex when VERTEX_PROJECT_ID is set, otherwise ollama.
LLM_BACKEND = os.getenv('LLM_BACKEND', 'vertex' if os.getenv('VERTEX_PROJECT_ID') else 'ollama').lower()
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434').rstrip('/')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen2.5:7b')

# --- Vertex AI (Gemini) — used when LLM_BACKEND=vertex.
VERTEX_PROJECT_ID = os.getenv('VERTEX_PROJECT_ID', '')
VERTEX_LOCATION = os.getenv('VERTEX_LOCATION', 'us-central1')
VERTEX_MODEL = os.getenv('VERTEX_MODEL', 'gemini-2.5-flash')

# --- Deployment profile: standard | gcp-trial | corporate
DEPLOYMENT_PROFILE = os.getenv('DEPLOYMENT_PROFILE', 'standard').lower()

# Backends that send document content to external APIs (not allowed under corporate profile).
_EXTERNAL_ONLY_BACKENDS = frozenset({'groq'})
_IN_BOUNDARY_BACKENDS = frozenset({'ollama', 'vertex'})

if LLM_BACKEND not in _IN_BOUNDARY_BACKENDS:
 raise RuntimeError(
  f'LLM_BACKEND={LLM_BACKEND!r} is not supported. Use ollama or vertex.'
 )
if DEPLOYMENT_PROFILE == 'corporate' and LLM_BACKEND in _EXTERNAL_ONLY_BACKENDS:
 raise RuntimeError(
  f'DEPLOYMENT_PROFILE=corporate rejects external-only LLM backend {LLM_BACKEND!r}. '
  'Use ollama (fully self-hosted) or vertex (in-boundary GCP contract).'
 )

RERANK_CANDIDATES = int(os.getenv('RERANK_CANDIDATES', '36'))
TOP_K_SEM = int(os.getenv('TOP_K_SEM', '40'))
TOP_K_BM25 = int(os.getenv('TOP_K_BM25', '40'))
TOP_K_FINAL = int(os.getenv('TOP_K_FINAL', '10'))
RRF_K = int(os.getenv('RRF_K', '60'))
MAX_CONTEXT_CHARS = int(os.getenv('MAX_CONTEXT_CHARS', '36000'))

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
