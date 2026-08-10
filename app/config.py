import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = Path(os.getenv("PDF_PATH", ROOT / "data" / "raw" / "PPL AR 2025.pdf"))
INDEX_DIR = ROOT / "data" / "index"
MODEL_DIR = ROOT / "models"

EMBED_MODEL = os.getenv("EMBED_MODEL", str(MODEL_DIR / "bge-m3"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", str(MODEL_DIR / "bge-reranker-v2-m3"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

TOP_K_SEM = int(os.getenv("TOP_K_SEM", "48"))
TOP_K_BM25 = int(os.getenv("TOP_K_BM25", "48"))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "10"))
MAX_RERANK_CANDIDATES = int(os.getenv("MAX_RERANK_CANDIDATES", "48"))
RRF_K = int(os.getenv("RRF_K", "60"))

# RTX 5070: using CUDA for both embedding and reranking substantially reduces
# the CPU saturation seen in v4. Set RERANKER_DEVICE=cpu if Ollama needs more VRAM.
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "cuda")

# Strict local-only model policy. These are deliberately set during config import,
# before sentence-transformers is imported by the application modules.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
