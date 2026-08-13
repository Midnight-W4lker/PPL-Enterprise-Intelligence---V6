"""
Run at Cloud Build time (see Dockerfile) to fetch the full-size embedding + reranker models into
./models/, so the running container loads them with local_files_only=True — no network dependency,
no HF rate limits, fast cold starts.

Unlike the Render build, these are the SAME full-size models used locally (bge-m3,
bge-reranker-v2-m3) — Cloud Run's memory ceiling is generous enough not to need smaller ones.
"""
import os
os.environ.pop('HF_HUB_OFFLINE', None)
os.environ.pop('TRANSFORMERS_OFFLINE', None)
from sentence_transformers import SentenceTransformer, CrossEncoder
from .config import MODEL_DIR, EMBED_MODEL_HF, RERANKER_MODEL_HF

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    embed_path = MODEL_DIR / 'bge-m3'
    rerank_path = MODEL_DIR / 'bge-reranker-v2-m3'

    if not embed_path.exists():
        print(f'Downloading embedding model {EMBED_MODEL_HF} -> {embed_path}')
        SentenceTransformer(EMBED_MODEL_HF, device='cpu').save(str(embed_path))
    else:
        print(f'Embedding model already present at {embed_path}')

    if not rerank_path.exists():
        print(f'Downloading reranker model {RERANKER_MODEL_HF} -> {rerank_path}')
        CrossEncoder(RERANKER_MODEL_HF, device='cpu').save(str(rerank_path))
    else:
        print(f'Reranker model already present at {rerank_path}')

    print('Model download complete.')

if __name__ == '__main__':
    main()
